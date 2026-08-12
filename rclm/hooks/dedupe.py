"""Conservative, session-scoped duplicate tool-result detection."""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict

MIN_DEDUPE_CHARS = 500
MAX_DEDUPE_ENTRIES = 500

# Varied plain-language phrasings for the same fact (a duplicate result wasn't
# repeated) so the message doesn't read as a canned/robotic string every time.
_DUPLICATE_PHRASES = (
    "{count:,} chars skipped",
    "{count:,} chars left out — same as before",
    "{count:,} chars not repeated",
)

_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_TIMESTAMP = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\b")
_DURATION = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:ms|s|m)(?!\w)")
_PID = re.compile(r"\b(?:pid|process)\s*[=:]?\s*\d+\b", re.IGNORECASE)
_PORT = re.compile(r"\bport\s*[=:]?\s*\d{2,5}\b", re.IGNORECASE)
_EPHEMERAL_ID = re.compile(
    r"\b(?:request|trace|run|job)[ _-]?(?:id)?[=:]\s*[a-f0-9-]{8,}\b", re.IGNORECASE
)


def normalize_result(text: str, *, cwd: str = "") -> str:
    """Return a deliberately narrow canonical form suitable for hashing.

    Do not normalize arbitrary numbers: test counts, assertion values, and exit
    details are semantic and must remain able to produce distinct hashes.
    """
    value = _ANSI.sub("", text)
    if cwd:
        normalized_cwd = cwd.rstrip("/")
        if normalized_cwd:
            value = value.replace(normalized_cwd + "/", "./")
            value = value.replace(normalized_cwd, ".")
    value = _TIMESTAMP.sub("<timestamp>", value)
    value = _DURATION.sub("<duration>", value)
    value = _PID.sub("pid=<pid>", value)
    value = _PORT.sub("port=<port>", value)
    value = _EPHEMERAL_ID.sub("<ephemeral-id>", value)
    return "\n".join(line.rstrip() for line in value.splitlines()).rstrip()


def is_error_result(result: object) -> bool:
    """Only skip explicitly marked errors; do not guess from ordinary text."""
    return isinstance(result, dict) and bool(result.get("is_error") or result.get("error"))


def maybe_dedupe(
    result: object,
    state: dict[str, dict],
    *,
    tool_name: str,
    turn: int,
    cwd: str = "",
    min_chars: int = MIN_DEDUPE_CHARS,
    max_entries: int = MAX_DEDUPE_ENTRIES,
) -> tuple[str | None, dict[str, dict], dict | None]:
    """Return (replacement, updated_state, metadata), never raising for callers."""
    if not isinstance(result, str) or len(result) < min_chars or is_error_result(result):
        return None, state, None
    normalized = normalize_result(result, cwd=cwd)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    lru: OrderedDict[str, dict] = OrderedDict(state)
    seen = lru.get(digest)
    if seen:
        lru.move_to_end(digest)
        phrase_index = int(digest[:8], 16) % len(_DUPLICATE_PHRASES)
        phrase = _DUPLICATE_PHRASES[phrase_index].format(count=seen["char_count"])
        pointer = (
            f"[RCLM] Identical to the result of `{seen['tool_name']}` at turn "
            f"{seen['turn']} ({phrase})."
        )
        return pointer, dict(lru), {"result_hash": digest, **seen}
    lru[digest] = {"tool_name": tool_name or "tool", "turn": turn, "char_count": len(result)}
    while len(lru) > max_entries:
        lru.popitem(last=False)
    return None, dict(lru), None
