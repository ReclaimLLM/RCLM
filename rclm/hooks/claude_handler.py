"""Entry point for Claude Code hooks: rclm-claude-hooks <EventName>.

Claude Code calls this binary for every lifecycle event, passing JSON on stdin.
All handlers are wrapped in try/except — hook failures must never disrupt Claude Code.
This process always exits 0.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import sys
from datetime import datetime, timezone

from rclm import _config
from rclm._models import FileDiff, HookSessionRecord, ToolCall
from rclm._uploader import upload_single
from rclm.hooks import (
    session_store,
    transcript,  # noqa: E402
)
from rclm.hooks._analytics import (
    aggregate_compression_savings,
    compute_session_analytics,
)
from rclm.hooks.compress import maybe_compress

logger = logging.getLogger(__name__)

THRESHOLD_ZERO_DURATION = (
    5.0  # seconds; if session duration is below this, treat as zero and omit timestamps
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _handle_session_start(session_id: str, payload: dict) -> None:
    session_store.append_event(
        session_id,
        {
            "event_type": "SessionStart",
            "cwd": payload.get("cwd", ""),
            "timestamp": payload.get("timestamp", _now()),
            "model": payload.get("model", "claude-unknown"),
        },
    )


def _handle_pre_tool_use(session_id: str, payload: dict) -> None:
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})

    session_store.append_event(
        session_id,
        {
            "event_type": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_use_id": payload.get("tool_use_id"),
            "timestamp": payload.get("timestamp", _now()),
        },
    )

    # Attempt compression if enabled — output updatedInput to stdout if applicable.
    if _config.load().get("compress", False):
        try:
            updated = maybe_compress(tool_name, tool_input)
            if updated:
                output = json.dumps({"hookSpecificOutput": {"updatedInput": updated}})
                print(output)
        except Exception:
            pass  # Never let compression disrupt Claude Code


def _handle_post_tool_use(session_id: str, payload: dict) -> None:
    session_store.append_event(
        session_id,
        {
            "event_type": "PostToolUse",
            "tool_name": payload.get("tool_name", ""),
            "tool_input": payload.get("tool_input", {}),
            "tool_response": payload.get("tool_response"),
            "tool_use_id": payload.get("tool_use_id"),
            "timestamp": payload.get("timestamp", _now()),
        },
    )


def _handle_user_prompt_submit(session_id: str, payload: dict) -> None:
    session_store.append_event(
        session_id,
        {
            "event_type": "UserPromptSubmit",
            "prompt": payload.get("prompt", ""),
            "timestamp": payload.get("timestamp", _now()),
        },
    )


def _extract_file_diffs_from_tool_calls(
    tool_calls: list[ToolCall],
) -> list[FileDiff]:
    """Extract FileDiff objects from Write/Edit/MultiEdit tool inputs."""
    diffs: list[FileDiff] = []

    for tc in tool_calls:
        name = tc.tool_name
        inp = tc.tool_input

        if name == "Write":
            file_path = inp.get("file_path", "")
            content = inp.get("content", "")
            unified = "".join(
                difflib.unified_diff(
                    [],
                    content.splitlines(keepends=True),
                    fromfile=f"a/{file_path}",
                    tofile=f"b/{file_path}",
                )
            )
            diffs.append(
                FileDiff(
                    path=file_path,
                    before=None,
                    after=content,
                    unified_diff=unified,
                )
            )

        elif name == "Edit":
            file_path = inp.get("file_path", "")
            old_string = inp.get("old_string", "")
            new_string = inp.get("new_string", "")
            unified = "".join(
                difflib.unified_diff(
                    old_string.splitlines(keepends=True),
                    new_string.splitlines(keepends=True),
                    fromfile=f"a/{file_path}",
                    tofile=f"b/{file_path}",
                )
            )
            diffs.append(
                FileDiff(
                    path=file_path,
                    before=old_string,
                    after=new_string,
                    unified_diff=unified,
                )
            )

        elif name == "MultiEdit":
            file_path = inp.get("file_path", "")
            for edit in inp.get("edits", []):
                old_string = edit.get("old_string", "")
                new_string = edit.get("new_string", "")
                unified = "".join(
                    difflib.unified_diff(
                        old_string.splitlines(keepends=True),
                        new_string.splitlines(keepends=True),
                        fromfile=f"a/{file_path}",
                        tofile=f"b/{file_path}",
                    )
                )
                diffs.append(
                    FileDiff(
                        path=file_path,
                        before=old_string,
                        after=new_string,
                        unified_diff=unified,
                    )
                )

    return diffs


def _handle_stop(session_id: str, payload: dict) -> None:
    now = _now()
    events = session_store.read_events(session_id)

    # Find cwd, started_at, and model from SessionStart event; use fallbacks if missing.
    cwd = payload.get("cwd", "")
    started_at = now
    session_start_model: str | None = None
    for ev in events:
        if ev.get("event_type") == "SessionStart":
            cwd = cwd or ev.get("cwd", "")
            started_at = ev.get("timestamp", now)
            session_start_model = ev.get("model")
            break

    ended_at = payload.get("timestamp", now)
    try:
        duration_s = (
            datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)
        ).total_seconds()
    except (ValueError, TypeError):
        duration_s = 0.0
    if duration_s < THRESHOLD_ZERO_DURATION:
        duration_s = 0.0
        started_at = None
        ended_at = None
    transcript_path = payload.get("transcript_path")
    transcript_data = transcript.parse_transcript(transcript_path)
    file_diffs = _extract_file_diffs_from_tool_calls(transcript_data.tool_calls)

    # Compute analytics from tool calls and file diffs.
    analytics = compute_session_analytics(transcript_data.tool_calls, file_diffs)
    compression = aggregate_compression_savings(events)

    record = HookSessionRecord(
        session_id=session_id,
        cwd=cwd,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=duration_s,
        transcript_path=transcript_path,
        model=transcript_data.model or session_start_model or "claude-unknown",
        messages=transcript_data.messages,
        tool_calls=transcript_data.tool_calls,
        file_diffs=file_diffs,
        total_input_tokens=transcript_data.total_input_tokens,
        total_output_tokens=transcript_data.total_output_tokens,
        tool_token_stats=analytics.get("tool_token_stats"),
        tool_call_count=analytics.get("tool_call_count"),
        unique_files_modified=analytics.get("unique_files_modified"),
        dominant_tool=analytics.get("dominant_tool"),
        compression_savings=compression,
    )

    asyncio.run(upload_single(record))
    session_store.cleanup(session_id)


_HANDLERS = {
    "SessionStart": _handle_session_start,
    "PreToolUse": _handle_pre_tool_use,
    "PostToolUse": _handle_post_tool_use,
    "UserPromptSubmit": _handle_user_prompt_submit,
    "Stop": _handle_stop,
    "SubagentStop": _handle_stop,
}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: rclm-claude-hooks <EventName>", file=sys.stderr)
        sys.exit(0)

    event_name = sys.argv[1]
    handler = _HANDLERS.get(event_name)
    if handler is None:
        # Unknown event; do nothing.
        sys.exit(0)

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        logger.warning(
            "rclm-claude-hooks: could not parse stdin JSON for event %s",
            event_name,
        )
        sys.exit(0)

    session_id = payload.get("session_id", "unknown")

    try:
        handler(session_id, payload)
    except Exception:
        logger.exception(
            "rclm-claude-hooks: unhandled error in handler for event %s",
            event_name,
        )

    sys.exit(0)
