"""Token estimation and session analytics for hook records."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from rclm._models import FileDiff, ToolCall


def estimate_tokens(content: str | dict | list | None) -> int:
    """Estimate token count using ~4 chars/token heuristic."""
    if content is None:
        return 0
    if isinstance(content, str):
        if not content:
            return 0
        return max(1, len(content) // 4)
    # For dicts/lists, serialize and estimate
    try:
        return max(1, len(json.dumps(content)) // 4)
    except (TypeError, ValueError):
        return 0


def compute_session_analytics(
    tool_calls: list[ToolCall],
    file_diffs: list[FileDiff],
) -> dict:
    """Compute per-tool stats, counts, and dominant tool from a session."""
    tool_stats: dict[str, dict] = {}

    for tc in tool_calls:
        name = tc.tool_name
        if name not in tool_stats:
            tool_stats[name] = {"count": 0, "input_tokens": 0, "output_tokens": 0}

        tool_stats[name]["count"] += 1

        input_est = tc.input_token_estimate or estimate_tokens(tc.tool_input)
        output_est = tc.output_token_estimate or estimate_tokens(tc.tool_result)
        tool_stats[name]["input_tokens"] += input_est
        tool_stats[name]["output_tokens"] += output_est

    # Unique files from file diffs
    unique_files = {d.path for d in file_diffs}

    # Dominant tool by call count
    dominant = None
    if tool_stats:
        dominant = max(tool_stats, key=lambda k: tool_stats[k]["count"])

    return {
        "tool_token_stats": tool_stats if tool_stats else None,
        "tool_call_count": len(tool_calls) if tool_calls else None,
        "unique_files_modified": len(unique_files) if unique_files else None,
        "dominant_tool": dominant,
    }


def mechanism_saving_event(
    mechanism: str,
    *,
    applied: bool,
    tokens_saved_estimate: int,
) -> dict:
    """Build a MechanismSaving event dict, ready for session_store.append_event.

    `applied=False` marks a shadow-mode measurement: the mechanism detected an
    opportunity and estimated the savings, but did not rewrite anything.
    """
    return {
        "event_type": "MechanismSaving",
        "mechanism": mechanism,
        "applied": applied,
        "tokens_saved_estimate": tokens_saved_estimate,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def aggregate_mechanism_savings(events: list[dict]) -> dict | None:
    """Aggregate MechanismSaving events from session JSONL into a per-mechanism summary.

    Returns {mechanism: {applied_count, shadow_count, tokens_saved_estimate}} or None
    if the session has no MechanismSaving events at all.
    """
    savings_events = [e for e in events if e.get("event_type") == "MechanismSaving"]
    if not savings_events:
        return None

    summary: dict[str, dict[str, int]] = {}
    for ev in savings_events:
        mechanism = ev.get("mechanism") or "unknown"
        bucket = summary.setdefault(
            mechanism, {"applied_count": 0, "shadow_count": 0, "tokens_saved_estimate": 0}
        )
        if ev.get("applied"):
            bucket["applied_count"] += 1
        else:
            bucket["shadow_count"] += 1
        bucket["tokens_saved_estimate"] += int(ev.get("tokens_saved_estimate") or 0)

    return summary
