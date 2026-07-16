"""Session-scoped read cache: diff-on-change for repeated reads of a file.

Every tool result gets re-sent on every subsequent API call in a session, so
a full-file re-read late in a long session is far more expensive than the
same read early on. This tracks the last-seen content for each (file_path,
offset, limit) a session has read — via the native Read tool or a shell dump
(cat/sed/head/tail/type/Get-Content) — and replaces a byte-identical re-read
with a short notice, or a changed re-read with a unified diff, instead of
sending the full content again.

Different ranges of the same file (different offset/limit) are tracked
independently and never compared against each other, since they aren't
redundant with one another.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import shutil

_DIFF_MAX_LINES = 60

# Commands that just print a file's current content, eligible for the same
# read-cache treatment as the native Read tool.
_DUMP_COMMANDS = {"cat", "sed", "head", "tail", "type", "get-content", "gc"}

# Presence of any of these means the command isn't "just dump one file" —
# leave it alone rather than risk misattributing piped/redirected output.
_UNSAFE_TOKENS = ("|", ">", "<", "&&", "||", ";")


def maybe_wrap_dump_command(tool_input: dict) -> dict | None:
    """PreToolUse: rewrite a simple file-dump command to route through rclm-read-cache.

    Only rewrites single, simple commands (no pipes/redirects/chaining), since
    the output must be exactly "this file's current content" for the cache
    comparison to be meaningful.
    """
    command = tool_input.get("command", "")
    if not command or not command.strip():
        return None
    if "rclm-read-cache" in command or "rclm-compress" in command:
        return None
    if any(token in command for token in _UNSAFE_TOKENS):
        return None

    parts = command.strip().split()
    if not parts:
        return None

    base = os.path.basename(parts[0]).lower()
    if base.endswith(".exe"):
        base = base[:-4]
    if base not in _DUMP_COMMANDS:
        return None

    if not shutil.which("rclm-read-cache"):
        return None

    return {"command": f"rclm-read-cache {command}"}


def build_delta(
    file_path: str,
    offset: int | None,
    limit: int | None,
    content: str,
    events: list[dict],
) -> dict | None:
    """Return a hookSpecificOutput-style delta, or None on the first read of this range."""
    key = _key(file_path, offset, limit)
    previous = _last_snapshot(key, events)
    if previous is None:
        return None

    if _hash(content) == previous.get("content_hash"):
        return {"updatedToolOutput": _unchanged_notice(file_path, previous.get("timestamp", ""))}

    return {"updatedToolOutput": _bounded_diff(previous.get("content", ""), content, file_path)}


def snapshot_event(
    file_path: str,
    offset: int | None,
    limit: int | None,
    content: str,
    timestamp: str,
) -> dict:
    """Build the ReadSnapshot event to record after this read, for future comparisons."""
    return {
        "event_type": "ReadSnapshot",
        "key": _key(file_path, offset, limit),
        "file_path": file_path,
        "content_hash": _hash(content),
        "content": content,
        "timestamp": timestamp,
    }


def _key(file_path: str, offset: int | None, limit: int | None) -> str:
    return f"{file_path}::{offset}::{limit}"


def _hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()


def _last_snapshot(key: str, events: list[dict]) -> dict | None:
    for ev in reversed(events):
        if ev.get("event_type") == "ReadSnapshot" and ev.get("key") == key:
            return ev
    return None


def _unchanged_notice(file_path: str, previous_timestamp: str) -> str:
    return (
        f"[rclm read-cache] Unchanged since the last read of {file_path}"
        f"{f' (at {previous_timestamp})' if previous_timestamp else ''}. "
        "Re-read with a different offset/limit if you need to see it again in full."
    )


def _bounded_diff(before: str, after: str, file_path: str) -> str:
    unified = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path} (current)",
        )
    )
    if not unified:
        # Hash differed but the diff is empty (e.g. trailing-whitespace-only change).
        return after

    if len(unified) > _DIFF_MAX_LINES:
        omitted = len(unified) - _DIFF_MAX_LINES
        unified = [
            *unified[:_DIFF_MAX_LINES],
            f"... ({omitted} more diff lines omitted; re-read without offset/limit for "
            "the full current content)\n",
        ]

    return f"[rclm read-cache] {file_path} changed since the last read — diff:\n" + "".join(unified)
