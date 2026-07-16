"""PreToolUse loop-breaker.

Detects two waste patterns from the token-waste analysis: N consecutive
identical tool calls (retry without changing anything), and N consecutive
failures targeting the same file/command (stale-match edit thrash, the
Gemini `replace`-retry pattern). Injects a corrective note first; escalates
to a permission prompt only once the pattern is well established, so a
false positive costs a click rather than silently blocking real work.
"""

from __future__ import annotations

import hashlib
import json

WARN_AFTER = 2  # inject a corrective note once this many repeats/failures seen
ASK_AFTER = 4  # escalate to a permission prompt once this many seen

# Tools whose target is a file path, for grouping failures.
_FILE_TARGET_TOOLS = {"Edit", "MultiEdit", "Write", "Read", "NotebookEdit"}


def _input_hash(tool_name: str, tool_input: dict) -> str:
    return hashlib.md5(
        (tool_name + json.dumps(tool_input, sort_keys=True, default=str)).encode()
    ).hexdigest()


def _target(tool_name: str, tool_input: dict) -> str | None:
    """Return the file/command a call operates on, for failure grouping."""
    if tool_name in _FILE_TARGET_TOOLS:
        return tool_input.get("file_path")
    if tool_name == "Bash":
        return tool_input.get("command")
    return None


def analyze(tool_name: str, tool_input: dict, events: list[dict]) -> dict | None:
    """Return a PreToolUse hookSpecificOutput delta, or None if nothing to flag.

    `events` is the session's event history *before* this call is recorded.
    """
    repeat_count = _consecutive_repeat_calls(tool_name, tool_input, events)
    failure_count = _consecutive_target_failures(tool_name, tool_input, events)
    count = max(repeat_count, failure_count)

    if count < WARN_AFTER:
        return None

    reason = _reason(tool_name, tool_input, repeat_count, failure_count)

    if count >= ASK_AFTER:
        return {"permissionDecision": "ask", "permissionDecisionReason": reason}
    return {"additionalContext": f"[rclm loop-breaker] {reason}"}


def _consecutive_repeat_calls(tool_name: str, tool_input: dict, events: list[dict]) -> int:
    """Count trailing PreToolUse events identical to this call, most recent first."""
    target_hash = _input_hash(tool_name, tool_input)
    count = 0
    for ev in reversed(events):
        if ev.get("event_type") != "PreToolUse":
            continue
        if _input_hash(ev.get("tool_name", ""), ev.get("tool_input", {})) != target_hash:
            break
        count += 1
    return count


def _consecutive_target_failures(tool_name: str, tool_input: dict, events: list[dict]) -> int:
    """Count trailing failures on the same (tool, target); a success on that
    target resets the streak. Unrelated tools/targets in between don't."""
    target = _target(tool_name, tool_input)
    if target is None:
        return 0

    count = 0
    for ev in reversed(events):
        event_type = ev.get("event_type")
        if event_type not in ("ToolFailure", "PostToolUse"):
            continue
        if ev.get("tool_name") != tool_name:
            continue
        if _target(tool_name, ev.get("tool_input", {})) != target:
            continue
        if event_type == "PostToolUse":
            break  # a successful call on this target ends the streak
        count += 1
    return count


def _reason(tool_name: str, tool_input: dict, repeat_count: int, failure_count: int) -> str:
    if failure_count >= repeat_count:
        target = _target(tool_name, tool_input) or "this call"
        return (
            f"{failure_count} consecutive failures on {tool_name} for {target!r}. "
            "The current approach isn't working — re-read the current state before "
            "trying again, or try a different strategy."
        )
    return (
        f"{repeat_count} consecutive identical {tool_name} calls with the same input. "
        "This looks like a retry loop — check whether the input actually needs to change."
    )
