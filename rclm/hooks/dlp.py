"""DLP (Data Loss Prevention) hook engine.

Scans *.env / .env* files recursively below the project CWD and prevents secret values
from reaching the model via two hook points:

  PreToolUse  → maybe_redact_input()   redirects .env reads to a sanitised temp copy;
                                        blocks bash commands that cat env files.
  PostToolUse → maybe_redact_output()  scrubs known secrets from tool output strings.

The secret map is re-parsed on every call so it stays fresh if files change mid-session.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum secret value length to enter the scrub set.
MIN_SECRET_LEN = 5
MAX_ENV_FILE_BYTES = 1024 * 1024
MAX_TOTAL_ENV_BYTES = 8 * 1024 * 1024

_NON_SECRET_ENV_FILE_MARKERS = frozenset(
    {"example", "sample", "template", "test", "testing", "pytest", "fixture", "mock"}
)
_SECRET_KEY_MARKERS = frozenset(
    {
        "apikey",
        "auth",
        "credential",
        "encryption",
        "key",
        "master",
        "pass",
        "passwd",
        "password",
        "private",
        "secret",
        "signing",
        "token",
    }
)
_PUBLIC_KEY_MARKERS = frozenset({"anon", "public", "publishable"})
_KNOWN_SECRET_PREFIXES = (
    "AKIA",
    "AIza",
    "SG.",
    "gho_",
    "ghp_",
    "github_pat_",
    "glpat-",
    "sk-",
    "sk_",
    "xapp-",
    "xoxb-",
    "xoxp-",
)


class DLPRedactionError(RuntimeError):
    """Raised when DLP cannot safely inspect a recognized secret source."""


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
_ENV_DUMP = re.compile(r"(^|[;&|]\s*)(env|printenv|set)(\s|$)")
_ENV_PATH_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_])(?:\.env(?:\.[A-Za-z0-9_-]+)?|[A-Za-z0-9_.-]+\.env)(?![A-Za-z0-9_])"
)
_CREDENTIAL_URL = re.compile(r"^[a-z][a-z0-9+.-]*://[^/\s:@]+:[^@\s/]+@", re.IGNORECASE)
_OPAQUE_TOKEN = re.compile(r"^[A-Za-z0-9_+./=-]+$")
_HEX_TOKEN = re.compile(r"^[A-Fa-f0-9]{32,}$")
_JWT_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
)


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
    """Return env-like regular files recursively below cwd.

    Traversal is deterministic and never follows symlinks. An unreadable subtree
    is an error rather than a silent coverage gap.
    """
    try:
        base = Path(cwd).expanduser().resolve()
        if not base.is_dir():
            raise DLPRedactionError(f"DLP workspace is not an accessible directory: {base}")
        found: list[Path] = []

        def _walk_error(error: OSError) -> None:
            raise DLPRedactionError(f"cannot scan env files below {base}: {error}") from error

        for root, dirnames, filenames in os.walk(base, followlinks=False, onerror=_walk_error):
            root_path = Path(root)
            dirnames[:] = sorted(name for name in dirnames if not (root_path / name).is_symlink())
            for name in sorted(filenames):
                path = root_path / name
                if _is_env_file(name) and not path.is_symlink() and path.is_file():
                    found.append(path)
        return found
    except DLPRedactionError:
        raise
    except (OSError, RuntimeError) as exc:
        raise DLPRedactionError(f"cannot scan env files below {cwd}: {exc}") from exc


# ---------------------------------------------------------------------------
# Env file parsing
# ---------------------------------------------------------------------------


def _strip_inline_comment(val: str) -> str:
    """Remove trailing inline comment (# ...) that is outside quotes."""
    in_quote: str | None = None
    escaped = False
    for i, ch in enumerate(val):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
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


def _read_env_text(path: Path) -> str:
    try:
        size = path.stat().st_size
        if size > MAX_ENV_FILE_BYTES:
            raise DLPRedactionError(
                f"env file exceeds {MAX_ENV_FILE_BYTES} byte safety limit: {path}"
            )
        return path.read_text(encoding="utf-8", errors="replace")
    except DLPRedactionError:
        raise
    except OSError as exc:
        raise DLPRedactionError(f"cannot read env file {path}: {exc}") from exc


def _quote_is_open(value: str) -> bool:
    quote: str | None = None
    escaped = False
    for char in value:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in ('"', "'"):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
    return quote is not None


def _logical_env_lines(text: str) -> list[str]:
    """Join newline-containing quoted values without interpreting escapes."""
    logical: list[str] = []
    pending = ""
    for raw_line in text.splitlines():
        pending = f"{pending}\n{raw_line}" if pending else raw_line
        value = pending.partition("=")[2] if "=" in pending else pending
        if not _quote_is_open(value):
            logical.append(pending)
            pending = ""
    if pending:
        logical.append(pending)
    return logical


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
    text = _read_env_text(path)

    for raw_line in _logical_env_lines(text):
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


def _is_global_secret_source(path: Path) -> bool:
    """Exclude fixture/template env files from ambient output scanning."""
    markers = {part for part in re.split(r"[._-]+", path.name.lower()) if part}
    return not bool(markers & _NON_SECRET_ENV_FILE_MARKERS)


def _load_secrets(cwd: str, *, include_non_secret_files: bool = False) -> dict[str, str]:
    """Load all secrets from env files in cwd. Always re-parsed (never cached)."""
    if not cwd:
        return {}
    env_files = _find_env_files(cwd)
    if not env_files:
        logger.debug("rclm DLP: no env files found in %s", cwd)
        return {}
    secrets: dict[str, str] = {}
    total_bytes = 0
    for env_file in env_files:
        if not include_non_secret_files and not _is_global_secret_source(env_file):
            continue
        try:
            total_bytes += env_file.stat().st_size
        except OSError as exc:
            raise DLPRedactionError(f"cannot inspect env file {env_file}: {exc}") from exc
        if total_bytes > MAX_TOTAL_ENV_BYTES:
            raise DLPRedactionError(
                f"env files exceed {MAX_TOTAL_ENV_BYTES} byte workspace safety limit"
            )
        secrets.update(_parse_env_file(env_file))
    return secrets


# ---------------------------------------------------------------------------
# Scrub set construction and application
# ---------------------------------------------------------------------------


def _key_parts(key: str) -> set[str]:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_").lower()
    return {part.rstrip("0123456789") for part in normalized.split("_") if part}


def _key_is_public(key: str) -> bool:
    return bool(_key_parts(key) & _PUBLIC_KEY_MARKERS)


def _key_looks_secret(key: str) -> bool:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_").lower()
    parts = _key_parts(key)
    if parts & _PUBLIC_KEY_MARKERS:
        return False
    if parts & _SECRET_KEY_MARKERS:
        return True
    return any(
        marker in normalized
        for marker in ("api_key", "access_key", "client_secret", "service_role")
    )


def _shannon_entropy(value: str) -> float:
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _value_looks_secret(value: str) -> bool:
    if value.startswith("-----BEGIN ") and "PRIVATE KEY-----" in value:
        return True
    if _CREDENTIAL_URL.match(value):
        return True
    if value.startswith(_KNOWN_SECRET_PREFIXES):
        return True
    if _HEX_TOKEN.fullmatch(value):
        return True
    if len(value) < 24 or not _OPAQUE_TOKEN.fullmatch(value):
        return False
    classes = sum(
        (
            any(char.islower() for char in value),
            any(char.isupper() for char in value),
            any(char.isdigit() for char in value),
            any(not char.isalnum() for char in value),
        )
    )
    return classes >= 3 and _shannon_entropy(value) >= 3.5


def _build_scrub_set(
    secrets: dict[str, str], *, include_all_values: bool = False
) -> list[tuple[str, str]]:
    """Return (value, placeholder) pairs for secrets worth scrubbing.

    Ambient output scanning filters out:
    - Values shorter than MIN_SECRET_LEN chars
    - Known-safe values (true, false, localhost, etc.)
    - Pure integers
    - Values without a sensitive key name or high-confidence secret shape

    Direct env-file access sets include_all_values so every non-empty assignment
    is hidden, including ordinary configuration values.

    Sorted longest-first so longer secrets are replaced before shorter substrings.
    """
    result: list[tuple[str, str]] = []
    for key, val in secrets.items():
        if include_all_values:
            result.append((val, f"[REDACTED:{key}]"))
            continue
        if len(val) < MIN_SECRET_LEN:
            continue
        if val in _SAFE_VALUES:
            continue
        if val.isdigit():
            continue
        if _key_is_public(key):
            continue
        if not (_key_looks_secret(key) or _value_looks_secret(val)):
            continue
        result.append((val, f"[REDACTED:{key}]"))
    result.sort(key=lambda t: len(t[0]), reverse=True)
    return result


def _scrub(text: str, scrub_set: list[tuple[str, str]]) -> str:
    """Apply all (value → placeholder) substitutions to text."""
    for val, placeholder in scrub_set:
        text = text.replace(val, placeholder)
    return _JWT_TOKEN.sub("[REDACTED:JWT]", text)


def _scrub_value(value: object, scrub_set: list[tuple[str, str]]) -> tuple[object, bool]:
    """Redact strings recursively while preserving JSON-compatible envelopes."""
    if isinstance(value, str):
        scrubbed = _scrub(value, scrub_set)
        return scrubbed, scrubbed != value
    if isinstance(value, list):
        changed = False
        items = []
        for item in value:
            scrubbed, item_changed = _scrub_value(item, scrub_set)
            items.append(scrubbed)
            changed = changed or item_changed
        return items, changed
    if isinstance(value, dict):
        changed = False
        result = {}
        for key, item in value.items():
            scrubbed, item_changed = _scrub_value(item, scrub_set)
            result[key] = scrubbed
            changed = changed or item_changed
        return result, changed
    return value, False


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
        return _redact_bash_input(tool_input, cwd)
    return None


def input_may_read_env(tool_name: str, tool_input: dict) -> bool:
    """Return whether a failed DLP scan belongs to a recognized env access."""
    normalized = tool_name.lower()
    if normalized in {"read", "read_file", "readfile"}:
        path = tool_input.get("file_path") or tool_input.get("path")
        return isinstance(path, str) and _is_env_file(os.path.basename(path))
    if normalized not in {"bash", "shell", "exec", "exec_command", "run_shell_command"}:
        return False
    command = tool_input.get("command") or tool_input.get("cmd") or tool_input.get("input")
    if not isinstance(command, str):
        return False
    if _ENV_DUMP.search(command):
        return True
    if _ENV_PATH_REFERENCE.search(command):
        return True
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    return any(_is_env_file(os.path.basename(token.strip("'\""))) for token in tokens)


def maybe_redact_output(
    tool_name: str,
    tool_response: object,
    cwd: str,
    *,
    redact_all: bool = False,
) -> str | None:
    """PostToolUse: return scrubbed response string if secrets were found, else None."""
    _ = tool_name  # reserved for future per-tool filtering
    secrets = _load_secrets(cwd, include_non_secret_files=redact_all)
    scrub_set = _build_scrub_set(secrets, include_all_values=redact_all)
    response_str = tool_response if isinstance(tool_response, str) else str(tool_response or "")
    scrubbed = _scrub(response_str, scrub_set)
    return scrubbed if scrubbed != response_str else None


def maybe_redact_value(value: object, cwd: str, *, redact_all: bool = False) -> object | None:
    """Return a shape-preserving redacted copy, or None when no value changed."""
    secrets = _load_secrets(cwd, include_non_secret_files=redact_all)
    scrub_set = _build_scrub_set(secrets, include_all_values=redact_all)
    scrubbed, changed = _scrub_value(value, scrub_set)
    return scrubbed if changed else None


def redact_high_confidence_value(value: object) -> object | None:
    """Redact self-identifying secret shapes without reading workspace files."""
    scrubbed, changed = _scrub_value(value, [])
    return scrubbed if changed else None


def redact_json_payload(payload: str, cwd: str) -> str:
    """Redact every known env value from a serialized record payload."""
    secrets = _load_secrets(cwd)
    scrub_set = _build_scrub_set(secrets)
    for value, placeholder in scrub_set:
        payload = payload.replace(value, placeholder)
        encoded_value = json.dumps(value)[1:-1]
        encoded_placeholder = json.dumps(placeholder)[1:-1]
        payload = payload.replace(encoded_value, encoded_placeholder)
    return _scrub(payload, [])


def reconcile_captured_tool_results(tool_calls: list[object], events: list[dict]) -> None:
    """Replace transcript results only where a hook captured a DLP-redacted value."""
    redacted_by_id = {
        event.get("tool_use_id"): event.get("tool_response")
        for event in events
        if event.get("dlp_redacted") and event.get("tool_use_id")
    }
    for call in tool_calls:
        tool_use_id = getattr(call, "tool_use_id", None)
        if tool_use_id in redacted_by_id:
            call.tool_result = redacted_by_id[tool_use_id]


def reconcile_captured_tool_inputs(tool_calls: list[object], events: list[dict]) -> None:
    """Replace transcript inputs with the already-sanitized hook capture."""
    redacted_by_id = {
        event.get("tool_use_id"): event.get("tool_input")
        for event in events
        if event.get("event_type") == "PreToolUse" and event.get("tool_use_id")
    }
    for call in tool_calls:
        tool_use_id = getattr(call, "tool_use_id", None)
        if tool_use_id in redacted_by_id:
            call.tool_input = redacted_by_id[tool_use_id]


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
    target = Path(file_path).expanduser()
    if not target.is_absolute() and cwd:
        target = Path(cwd) / target
    try:
        target = target.resolve(strict=True)
        original = _read_env_text(target)
    except (OSError, RuntimeError) as exc:
        raise DLPRedactionError(f"cannot resolve env file {file_path}: {exc}") from exc

    # A direct env-file read hides every assignment, not only values that look secret.
    secrets = _parse_env_file(target)

    if not secrets:
        return None

    scrub_set = _build_scrub_set(secrets, include_all_values=True)
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
    except OSError as exc:
        raise DLPRedactionError(f"cannot create sanitized env file: {exc}") from exc

    if track_temp is not None:
        track_temp(tmp_path)

    # Return a delta dict — only the fields we want to override.
    return {"file_path": tmp_path}


def _redact_bash_input(tool_input: dict, cwd: str) -> dict | None:
    """Block bash commands that directly print env files."""
    command = tool_input.get("command", "")
    if not command:
        return None

    try:
        from rclm.hooks import read_cache

        request = read_cache.parse_shell_read(
            command,
            cwd=cwd,
            shell=tool_input.get("shell") or ("posix" if os.name == "posix" else os.name),
        )
    except Exception:
        request = None
    if request is not None and _is_env_file(Path(request.path).name):
        return {
            "command": (
                f"echo '[rclm DLP] Blocked: reading {request.display_path} is disabled "
                "(DLP policy).'"
            )
        }

    if not _CAT_LIKE.search(command):
        return None

    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for token in tokens:
        token = token.strip("'\"")
        if _is_env_file(os.path.basename(token)):
            return {
                "command": (f"echo '[rclm DLP] Blocked: reading {token} is disabled (DLP policy).'")
            }

    return None
