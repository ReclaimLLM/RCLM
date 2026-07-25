"""Token estimation and session analytics for hook records."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

from rclm._models import FileDiff, ToolCall

_MAX_FILES_PER_MECHANISM = 100


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
    measurement_kind: str = "estimated",
    file_path: str | None = None,
    raw_token_estimate: int | None = None,
    compressed_token_estimate: int | None = None,
) -> dict:
    """Build a MechanismSaving event dict, ready for session_store.append_event.

    `applied=False` marks a shadow-mode measurement: the mechanism detected an
    opportunity and estimated the savings, but did not rewrite anything.
    """
    event = {
        "event_type": "MechanismSaving",
        "mechanism": mechanism,
        "applied": applied,
        "tokens_saved_estimate": tokens_saved_estimate,
        "measurement_kind": measurement_kind,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if file_path:
        event["file_path"] = file_path
    if raw_token_estimate is not None:
        event["raw_token_estimate"] = raw_token_estimate
    if compressed_token_estimate is not None:
        event["compressed_token_estimate"] = compressed_token_estimate
    return event


def aggregate_mechanism_savings(events: list[dict]) -> dict | None:
    """Aggregate MechanismSaving events from session JSONL into a per-mechanism summary.

    Returns {mechanism: {applied_count, shadow_count, tokens_saved_estimate}} or None
    if the session has no MechanismSaving events at all.
    """
    savings_events = [e for e in events if e.get("event_type") == "MechanismSaving"]
    if not savings_events:
        return None

    summary: dict[str, dict] = {}
    for ev in savings_events:
        mechanism = ev.get("mechanism") or "unknown"
        bucket = summary.setdefault(
            mechanism,
            {
                "measurement_kind": ev.get("measurement_kind") or "estimated",
                "applied_count": 0,
                "shadow_count": 0,
                "tokens_saved_estimate": 0,
            },
        )
        if ev.get("applied"):
            bucket["applied_count"] += 1
        else:
            bucket["shadow_count"] += 1
        bucket["tokens_saved_estimate"] += int(ev.get("tokens_saved_estimate") or 0)
        file_path = ev.get("file_path")
        if isinstance(file_path, str) and file_path:
            files = bucket.setdefault("files", {})
            file_bucket = files.setdefault(
                file_path,
                {"applied_count": 0, "shadow_count": 0, "tokens_saved_estimate": 0},
            )
            file_bucket["applied_count" if ev.get("applied") else "shadow_count"] += 1
            file_bucket["tokens_saved_estimate"] += int(ev.get("tokens_saved_estimate") or 0)

    return summary


def merge_mechanism_savings(
    *rollups: dict | None,
    max_files_per_mechanism: int = _MAX_FILES_PER_MECHANISM,
) -> dict | None:
    """Merge per-turn rollups without losing prior turns' measured savings.

    Overall counters remain exact. Per-file details are bounded because they are
    diagnostic metadata used only for the top-files view.
    """
    merged: dict[str, dict] = {}
    for rollup in rollups:
        if not isinstance(rollup, dict):
            continue
        for mechanism, value in rollup.items():
            if not isinstance(value, dict):
                continue
            if "tokens_saved_estimate" not in value:
                merged.setdefault(mechanism, {}).update(copy.deepcopy(value))
                continue

            kind = value.get("measurement_kind") or "estimated"
            bucket = merged.get(mechanism)
            if not isinstance(bucket, dict) or "tokens_saved_estimate" not in bucket:
                bucket = {
                    "measurement_kind": kind,
                    "applied_count": 0,
                    "shadow_count": 0,
                    "tokens_saved_estimate": 0,
                }
                merged[mechanism] = bucket
            bucket["applied_count"] += int(value.get("applied_count") or 0)
            bucket["shadow_count"] += int(value.get("shadow_count") or 0)
            bucket["tokens_saved_estimate"] += int(value.get("tokens_saved_estimate") or 0)
            if bucket.get("measurement_kind") != kind:
                bucket["measurement_kind"] = "estimated"

            files = value.get("files")
            if not isinstance(files, dict):
                continue
            file_buckets = bucket.setdefault("files", {})
            for path, file_value in files.items():
                if not isinstance(path, str) or not isinstance(file_value, dict):
                    continue
                file_bucket = file_buckets.setdefault(
                    path,
                    {"applied_count": 0, "shadow_count": 0, "tokens_saved_estimate": 0},
                )
                for key in ("applied_count", "shadow_count", "tokens_saved_estimate"):
                    file_bucket[key] += int(file_value.get(key) or 0)

    for bucket in merged.values():
        files = bucket.get("files")
        if not isinstance(files, dict):
            continue
        ranked = sorted(
            files.items(),
            key=lambda item: (-item[1]["tokens_saved_estimate"], item[0]),
        )
        bucket["files"] = dict(ranked[:max_files_per_mechanism])
    return merged or None
