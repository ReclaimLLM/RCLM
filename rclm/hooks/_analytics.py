"""Token estimation and session analytics for hook records."""

from __future__ import annotations

import json

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


def aggregate_compression_savings(events: list[dict]) -> dict | None:
    """Aggregate CompressionSaving events from session JSONL."""
    savings_events = [e for e in events if e.get("event_type") == "CompressionSaving"]
    if not savings_events:
        return None

    total_original = sum(e.get("original_chars", 0) for e in savings_events)
    total_compressed = sum(e.get("compressed_chars", 0) for e in savings_events)

    return {
        "total_original_chars": total_original,
        "total_compressed_chars": total_compressed,
        "savings_pct": round((1 - total_compressed / total_original) * 100, 1)
        if total_original > 0
        else 0.0,
        "command_count": len(savings_events),
    }
