"""DLP (Data Loss Prevention) hook engine.

Scans *.env / .env* files in the project CWD and prevents secret values
from reaching the model via two hook points:

  PreToolUse  → maybe_redact_input()   redirects .env reads to a sanitised temp copy;
                                        blocks bash commands that cat env files.
  PostToolUse → maybe_redact_output()  scrubs known secrets from tool output strings.

The secret map is re-parsed on every call so it stays fresh if files change mid-session.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum secret value length to enter the scrub set.
MIN_SECRET_LEN = 5

# Values that are obviously non-secrets even if they pass the length check.
_SAFE_VALUES: frozenset[str] = frozenset(
    {
        "true",
        "false",
        "True",
        "False",
        "TRUE",
        "FALSE",
        "yes",
        "no",
        "Yes",
        "No",
        "YES",
        "NO",
        "null",
        "none",
        "None",
        "NULL",
        "NONE",
        "localhost",
        "0.0.0.0",
        "127.0.0.1",
        "::1",
    }
)

# Bash commands that print file contents verbatim.
_CAT_LIKE = re.compile(r"\b(cat|less|more|head|tail|bat|batcat)\b")


# ---------------------------------------------------------------------------
# Env file detection
# ---------------------------------------------------------------------------


def _is_env_file(name: str) -> bool:
    """Return True if the filename looks like an env file."""
    return (
        name == ".env"
        or name.startswith(".env.")  # .env.local, .env.production
        or name.endswith(".env")  # dev.env, prod.env, llm.env
        or name == ".envrc"
    )


def _find_env_files(cwd: str) -> list[Path]:
    """Return all env-like files directly in cwd (non-recursive)."""
    try:
        base = Path(cwd)
        if not base.is_dir():
            return []
        return [p for p in base.iterdir() if p.is_file() and _is_env_file(p.name)]
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Env file parsing
# ---------------------------------------------------------------------------


def _strip_inline_comment(val: str) -> str:
    """Remove trailing inline comment (# ...) that is outside quotes."""
    in_quote: str | None = None
    for i, ch in enumerate(val):
        if ch in ('"', "'"):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
        elif ch == "#" and in_quote is None:
            return val[:i]
    return val


def _unquote(val: str) -> str:
    """Strip matching surrounding single or double quotes."""
    if len(val) >= 2 and val[0] in ('"', "'") and val[-1] == val[0]:
        return val[1:-1]
    return val


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse an env file into {var_name: value}.

    Handles:
    - KEY=VALUE
    - export KEY=VALUE
    - KEY="quoted value"  and  KEY='single quoted'
    - KEY VALUE  (space-separated)
    - # comments (full-line and inline)
    """
    secrets: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return secrets

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Strip leading 'export '
        if line.startswith("export "):
            line = line[7:].strip()

        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            if not key:
                continue
            val = _unquote(_strip_inline_comment(val).strip())
        else:
            # Space-separated: KEY VALUE
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, val = parts[0].strip(), parts[1].strip()

        if key and val:
            secrets[key] = val

    return secrets


def _load_secrets(cwd: str) -> dict[str, str]:
    """Load all secrets from env files in cwd. Always re-parsed (never cached)."""
    if not cwd:
        return {}
    env_files = _find_env_files(cwd)
    if not env_files:
        logger.debug("rclm DLP: no env files found in %s", cwd)
        return {}
    secrets: dict[str, str] = {}
    for env_file in env_files:
        secrets.update(_parse_env_file(env_file))
    return secrets


# ---------------------------------------------------------------------------
# Scrub set construction and application
# ---------------------------------------------------------------------------


def _build_scrub_set(secrets: dict[str, str]) -> list[tuple[str, str]]:
    """Return (value, placeholder) pairs for secrets worth scrubbing.

    Filters out:
    - Values shorter than MIN_SECRET_LEN chars
    - Known-safe values (true, false, localhost, etc.)
    - Pure integers

    Sorted longest-first so longer secrets are replaced before shorter substrings.
    """
    result: list[tuple[str, str]] = []
    for key, val in secrets.items():
        if len(val) < MIN_SECRET_LEN:
            continue
        if val in _SAFE_VALUES:
            continue
        if val.isdigit():
            continue
        result.append((val, f"[REDACTED:{key}]"))
    result.sort(key=lambda t: len(t[0]), reverse=True)
    return result


def _scrub(text: str, scrub_set: list[tuple[str, str]]) -> str:
    """Apply all (value → placeholder) substitutions to text."""
    for val, placeholder in scrub_set:
        text = text.replace(val, placeholder)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def maybe_redact_input(
    tool_name: str,
    tool_input: dict,
    cwd: str,
    *,
    track_temp: Callable[[str], None] | None = None,
) -> dict | None:
    """PreToolUse: return a delta updatedInput dict if DLP applies, else None.

    For Read:  redirects .env file reads to a sanitised temp copy.
    For Bash:  replaces commands that cat env files with a block message.

    track_temp: optional callback invoked with the temp file path so the
                caller can clean it up at session Stop.
    """
    if tool_name == "Read":
        return _redact_read_input(tool_input, cwd, track_temp=track_temp)
    if tool_name == "Bash":
        return _redact_bash_input(tool_input)
    return None


def maybe_redact_output(
    tool_name: str,
    tool_response: object,
    cwd: str,
) -> str | None:
    """PostToolUse: return scrubbed response string if secrets were found, else None."""
    _ = tool_name  # reserved for future per-tool filtering
    secrets = _load_secrets(cwd)
    if not secrets:
        return None
    scrub_set = _build_scrub_set(secrets)
    if not scrub_set:
        return None
    response_str = tool_response if isinstance(tool_response, str) else str(tool_response or "")
    scrubbed = _scrub(response_str, scrub_set)
    return scrubbed if scrubbed != response_str else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _redact_read_input(
    tool_input: dict,
    cwd: str,
    *,
    track_temp: Callable[[str], None] | None = None,
) -> dict | None:
    file_path = tool_input.get("file_path", "")
    if not file_path or not _is_env_file(os.path.basename(file_path)):
        return None

    # Read the target file first — it IS the secret source.
    # We also merge in any other env files from cwd (e.g. dev.env + .env.local),
    # but the target file alone is sufficient even when cwd is unavailable.
    try:
        original = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    # Build secrets from the file itself, then augment from cwd siblings.
    secrets = _parse_env_file(Path(file_path))
    if cwd:
        for sibling in _find_env_files(cwd):
            if sibling != Path(file_path):
                secrets.update(_parse_env_file(sibling))

    if not secrets:
        return None

    scrub_set = _build_scrub_set(secrets)
    if not scrub_set:
        return None

    sanitized = _scrub(original, scrub_set)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".env",
            prefix="rclm_dlp_",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(sanitized)
            tmp_path = tmp.name
    except OSError:
        return None

    if track_temp is not None:
        track_temp(tmp_path)

    # Return a delta dict — only the fields we want to override.
    return {"file_path": tmp_path}


def _redact_bash_input(tool_input: dict) -> dict | None:
    """Block bash commands that directly print env files."""
    command = tool_input.get("command", "")
    if not command or not _CAT_LIKE.search(command):
        return None

    for token in command.split():
        token = token.strip("'\"")
        if _is_env_file(os.path.basename(token)):
            return {
                "command": (f"echo '[rclm DLP] Blocked: reading {token} is disabled (DLP policy).'")
            }

    return None
