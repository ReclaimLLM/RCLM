"""Cross-session outline: serve an outline when a session re-reads a known file.

Agents start each session cold and re-read the same large files. When a native
Read starts at line 1 of a big file whose content hash matches what an earlier
session of the same project already read, the result is replaced with a symbol
outline (with real file line numbers) and a pointer to read exact ranges.

The saving is reported net: the outline's reduction is recorded when served,
and every later read of that file in the same session is recorded as a negative
saving, since those reads may be follow-ups the outline caused.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from rclm.hooks import session_store
from rclm.hooks._analytics import estimate_tokens, mechanism_saving_event
from rclm.hooks._project_key import _git_root
from rclm.hooks.read_cache import ReadRequest

MECHANISM = "cross_session_outline"

MIN_FILE_LINES = 300
MIN_RETURNED_LINES = 200
MAX_OUTLINE_ENTRIES = 150
MIN_OUTLINE_ENTRIES = 3
_MAX_STORE_ENTRIES = 5000
_MAX_SESSIONS_PER_FILE = 5
_STORE_ROOT = Path.home() / ".reclaimllm" / "project_reads"

_SYMBOL_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^\s*(?:async\s+def|def|class)\s+\w+"),
    "script": re.compile(
        r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?"
        r"(?:function\*?\s+\w+|class\s+\w+|interface\s+\w+|type\s+\w+\s*=|enum\s+\w+"
        r"|(?:const|let)\s+\w+\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>)"
    ),
    "go": re.compile(r"^(?:func|type)\s+"),
    "rust": re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:fn|struct|enum|trait|impl|mod)\b"),
    "sql": re.compile(
        r"^\s*(?:CREATE|ALTER)\s+(?:OR\s+REPLACE\s+)?"
        r"(?:TABLE|VIEW|MATERIALIZED\s+VIEW|FUNCTION|TRIGGER|POLICY|INDEX|TYPE)\b",
        re.IGNORECASE,
    ),
    "markdown": re.compile(r"^#{1,4}\s+\S"),
}
_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "script",
    ".jsx": "script",
    ".ts": "script",
    ".tsx": "script",
    ".mjs": "script",
    ".cjs": "script",
    ".go": "go",
    ".rs": "rust",
    ".sql": "sql",
    ".md": "markdown",
    ".mdx": "markdown",
}


@dataclass
class OutlineApplication:
    replacement: str | None = None
    events: list[dict] = field(default_factory=list)
    # True when this Read must not be recorded as seen by the range cache.
    claimed: bool = False


def build_outline(text: str, path: str) -> list[str] | None:
    """Return "L<n>: <signature>" entries, or None for unsupported/sparse files."""
    language = _LANGUAGE_BY_SUFFIX.get(Path(path).suffix.lower())
    pattern = _SYMBOL_PATTERNS.get(language or "")
    if pattern is None:
        return None
    entries = [
        f"L{number}: {line.strip()[:120]}"
        for number, line in enumerate(text.splitlines(), start=1)
        if pattern.match(line)
    ]
    if len(entries) < MIN_OUTLINE_ENTRIES:
        return None
    if len(entries) > MAX_OUTLINE_ENTRIES:
        hidden = len(entries) - MAX_OUTLINE_ENTRIES
        entries = [*entries[:MAX_OUTLINE_ENTRIES], f"… {hidden} more definitions not listed"]
    return entries


def apply_outline(
    request: ReadRequest,
    content: str,
    *,
    session_id: str,
    state_id: str,
    cwd: str,
    shadow: bool,
) -> OutlineApplication:
    """Record this read and, when eligible, outline it (or measure in shadow)."""
    path = request.path
    result = OutlineApplication()
    session_state = session_store.read_outline_state(state_id)
    outlined = session_state.setdefault("outlined", {})
    seen = set(session_state.get("seen", []))

    if path in outlined:
        follow_up = estimate_tokens(content)
        result.events.append(
            mechanism_saving_event(
                MECHANISM,
                applied=True,
                tokens_saved_estimate=-follow_up,
                measurement_kind="measured",
                file_path=request.display_path,
                raw_token_estimate=follow_up,
                compressed_token_estimate=follow_up,
            )
        )
    elif path not in seen and _eligible(request, _load_store(cwd), session_id):
        replacement = _replacement(request)
        if replacement is not None:
            raw = estimate_tokens(content)
            compressed = estimate_tokens(replacement)
            if raw > compressed:
                result.events.append(
                    mechanism_saving_event(
                        MECHANISM,
                        applied=not shadow,
                        tokens_saved_estimate=raw - compressed,
                        measurement_kind="measured",
                        file_path=request.display_path,
                        raw_token_estimate=raw,
                        compressed_token_estimate=compressed,
                    )
                )
                if not shadow:
                    outlined[path] = {"served_at": time.time()}
                    result.replacement = replacement
                    result.claimed = True

    seen.add(path)
    session_state["seen"] = sorted(seen)
    session_store.write_outline_state(state_id, session_state)
    _record_read(cwd, request, session_id)
    return result


def _eligible(request: ReadRequest, store: dict, session_id: str) -> bool:
    returned_lines = request.end_line - request.start_line + 1
    if (
        request.start_line != 1
        or returned_lines < MIN_RETURNED_LINES
        or request.metadata.line_count < MIN_FILE_LINES
    ):
        return False
    entry = store.get(request.path)
    if not isinstance(entry, dict) or entry.get("content_hash") != request.metadata.content_hash:
        return False
    return any(other != session_id for other in entry.get("sessions", []))


def _replacement(request: ReadRequest) -> str | None:
    try:
        raw = Path(request.path).read_bytes()
    except OSError:
        return None
    if hashlib.sha256(raw).hexdigest() != request.metadata.content_hash:
        return None  # changed since the hook hashed it: never outline stale content
    text = raw.decode("utf-8", errors="replace")
    entries = build_outline(text, request.path)
    if entries is None:
        return None
    header = (
        f"[RCLM] {request.display_path} ({request.metadata.line_count} lines) is unchanged "
        "since an earlier session of this project read it, so its outline is shown instead "
        "of the content. The L numbers below are real file line numbers (the left margin "
        "numbers this outline, not the file). Read with offset/limit for the exact lines "
        "you need; read without them again for the full content."
    )
    return "\n".join([header, "", *entries]) + "\n"


def _store_path(cwd: str) -> Path:
    root = str(_git_root(cwd))
    return _STORE_ROOT / f"{hashlib.sha256(root.encode('utf-8')).hexdigest()[:16]}.json"


def _load_store(cwd: str) -> dict:
    try:
        data = json.loads(_store_path(cwd).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _record_read(cwd: str, request: ReadRequest, session_id: str) -> None:
    """Remember which sessions read this exact file content (bounded, best effort)."""
    store = _load_store(cwd)
    entry = store.get(request.path)
    if not isinstance(entry, dict) or entry.get("content_hash") != request.metadata.content_hash:
        entry = {"content_hash": request.metadata.content_hash, "sessions": []}
    sessions = [other for other in entry.get("sessions", []) if other != session_id]
    entry["sessions"] = [*sessions, session_id][-_MAX_SESSIONS_PER_FILE:]
    entry["last_read"] = time.time()
    store[request.path] = entry
    if len(store) > _MAX_STORE_ENTRIES:
        ranked = sorted(store.items(), key=lambda item: item[1].get("last_read", 0), reverse=True)
        store = dict(ranked[:_MAX_STORE_ENTRIES])
    session_store.atomic_private_write(_store_path(cwd), json.dumps(store, separators=(",", ":")))
