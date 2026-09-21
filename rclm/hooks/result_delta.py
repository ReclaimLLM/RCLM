"""Bounded differential output for repeated stateful tool calls.

The mechanism is deliberately limited to repeated commands, searches, and
directory listings with the same semantic target.  It never rewrites failed
commands and retains a digest that resolves to the exact captured result.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from rclm.hooks.tool_semantics import semantic_target, tool_family

MIN_DELTA_CHARS = 600
MIN_OVERLAP = 0.60
MAX_STATE_ENTRIES = 128
MAX_CHANGED_LINES = 80
MAX_RESULT_CHARS = 256_000
MAX_STATE_CHARS = 8_000_000

_TIMING_LINE = re.compile(r"^(?:Created|Completed) At: .*$", re.MULTILINE)
_FAILED_COMMAND = re.compile(r"The command exited with code (?!0\.)\d+\.")


@dataclass(frozen=True)
class DeltaDecision:
    replacement: str | None
    state: dict


def _canonical_lines(text: str) -> list[str]:
    canonical = _TIMING_LINE.sub("", text)
    return [line.rstrip() for line in canonical.splitlines() if line.strip()]


def _bounded_state(state: object) -> dict:
    if not isinstance(state, dict):
        return {"entries": {}, "clock": 0}
    entries = state.get("entries")
    clock = state.get("clock")
    return {
        "entries": dict(entries) if isinstance(entries, dict) else {},
        "clock": int(clock) if isinstance(clock, int) else 0,
    }


def process_delta(
    tool_name: str,
    tool_input: object,
    text: str,
    state: object,
) -> DeltaDecision:
    """Compare one result with the preceding result for the same target."""
    normalized_state = _bounded_state(state)
    family = tool_family(tool_name)
    target = semantic_target(tool_name, tool_input)
    if family not in {"shell", "search", "listing"} or target is None:
        return DeltaDecision(None, normalized_state)

    key = f"{family}:{target}"
    entries = normalized_state["entries"]
    normalized_state["clock"] += 1
    previous = entries.get(key)
    if len(text) > MAX_RESULT_CHARS:
        entries.pop(key, None)
        return DeltaDecision(None, normalized_state)
    entries[key] = {"text": text, "clock": normalized_state["clock"]}
    while (
        len(entries) > MAX_STATE_ENTRIES
        or sum(len(entry.get("text", "")) for entry in entries.values() if isinstance(entry, dict))
        > MAX_STATE_CHARS
    ):
        oldest = min(entries, key=lambda candidate: int(entries[candidate].get("clock", 0)))
        entries.pop(oldest, None)

    if (
        len(text) < MIN_DELTA_CHARS
        or _FAILED_COMMAND.search(text)
        or not isinstance(previous, dict)
        or not isinstance(previous.get("text"), str)
    ):
        return DeltaDecision(None, normalized_state)

    old_lines = _canonical_lines(previous["text"])
    new_lines = _canonical_lines(text)
    if not old_lines or not new_lines:
        return DeltaDecision(None, normalized_state)
    old_set = set(old_lines)
    new_set = set(new_lines)
    overlap = len(old_set & new_set) / max(1, len(new_set))
    if overlap < MIN_OVERLAP:
        return DeltaDecision(None, normalized_state)

    added = [line for line in new_lines if line not in old_set]
    removed = [line for line in old_lines if line not in new_set]
    digest = hashlib.sha256(text.encode("utf-8", errors="surrogateescape")).hexdigest()
    header = (
        f"[rclm state delta] family={family} unchanged_lines={len(new_set & old_set)} "
        f"added={len(added)} removed={len(removed)}\n"
    )
    sections = [header]
    if added:
        sections.append("Current additions/changes:\n")
        sections.extend(f"+ {line}\n" for line in added[:MAX_CHANGED_LINES])
    if removed:
        sections.append("Resolved/removed since prior result:\n")
        sections.extend(f"- {line}\n" for line in removed[:MAX_CHANGED_LINES])
    if len(added) > MAX_CHANGED_LINES or len(removed) > MAX_CHANGED_LINES:
        sections.append("[additional changed lines omitted; inspect captured result]\n")
    sections.append(f"source_sha256={digest} (exact result retained in session capture)\n")
    replacement = "".join(sections)
    return DeltaDecision(
        replacement if len(replacement) < len(text) else None,
        normalized_state,
    )


__all__ = ["DeltaDecision", "process_delta"]
