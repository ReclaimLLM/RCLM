"""Entry point for Antigravity hooks: rclm-antigravity-hooks <EventName>.

Antigravity's hook contract has no channel to mutate tool input or redact tool
output: PreToolUse output is only decision (allow/deny/ask/...), PostToolUse
output must be exactly {}. None of rclm's active mechanisms (DLP redaction,
compression, dedupe, range-cache) have a supported path here, so this
integration is capture-only. It registers -- and only needs -- a single Stop
hook: every message and tool call for the session is available directly from
the transcript.jsonl file Antigravity already writes (see
antigravity_transcript.py), so there is nothing to gain from also registering
PreToolUse/PostToolUse/PreInvocation/PostInvocation, and real risk in doing so
-- a bug returning a malformed PreToolUse decision could block the user's
actual tool calls for zero capture benefit.

Hook failures must never disrupt Antigravity, so this process always exits 0.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone

from rclm._models import HookSessionRecord
from rclm._uploader import close_session, upload_single
from rclm.hooks._analytics import compute_session_analytics
from rclm.hooks.antigravity_transcript import parse_transcript

logger = logging.getLogger(__name__)


def _iso(ts: str) -> str:
    return ts.replace("Z", "+00:00") if ts.endswith("Z") else ts


def _duration_s(started_at: str, ended_at: str) -> float:
    try:
        return max(
            0.0,
            (
                datetime.fromisoformat(_iso(ended_at)) - datetime.fromisoformat(_iso(started_at))
            ).total_seconds(),
        )
    except (TypeError, ValueError):
        return 0.0


async def _upload_and_close(record: HookSessionRecord) -> None:
    """upload_single, then close the module-level aiohttp session before this
    asyncio.run() call's event loop is torn down -- see claude_handler's
    identical helper for why (aiohttp session/loop binding)."""
    try:
        await upload_single(record)
    finally:
        await close_session()


def _handle_stop(payload: dict) -> None:
    session_id = str(payload.get("conversationId") or "")
    if not session_id:
        logger.warning("rclm-antigravity-hooks: Stop payload missing conversationId")
        return

    transcript_path = payload.get("transcriptPath")
    transcript_data = parse_transcript(transcript_path)
    messages = transcript_data.messages

    now = datetime.now(timezone.utc).isoformat()
    timestamps = [m["timestamp"] for m in messages if m.get("timestamp")]
    started_at = min(timestamps) if timestamps else now
    ended_at = max(timestamps) if timestamps else started_at

    workspace_paths = payload.get("workspacePaths")
    cwd = workspace_paths[0] if isinstance(workspace_paths, list) and workspace_paths else ""

    analytics = compute_session_analytics(transcript_data.tool_calls, [])

    record = HookSessionRecord(
        session_id=session_id,
        cwd=cwd,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=_duration_s(started_at, ended_at),
        transcript_path=transcript_path,
        model=payload.get("modelName") or "antigravity-unknown",
        messages=messages,
        tool_calls=transcript_data.tool_calls,
        file_diffs=[],
        tool_token_stats=analytics["tool_token_stats"],
        tool_call_count=analytics["tool_call_count"],
        unique_files_modified=analytics["unique_files_modified"],
        dominant_tool=analytics["dominant_tool"],
    )

    asyncio.run(_upload_and_close(record))


def _handle_pre_tool_use(payload: dict) -> None:
    print(json.dumps({"decision": "allow"}))


def _handle_post_tool_use(payload: dict) -> None:
    print("{}")


def main() -> None:
    hook_name = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        logger.warning("rclm-antigravity-hooks: could not parse stdin JSON for hook %s", hook_name)
        sys.exit(0)

    try:
        if hook_name == "PreToolUse":
            _handle_pre_tool_use(payload)
        elif hook_name == "PostToolUse":
            _handle_post_tool_use(payload)
        elif hook_name == "Stop":
            _handle_stop(payload)
    except Exception:
        logger.exception("rclm-antigravity-hooks: unhandled error in hook %s", hook_name)

    sys.exit(0)


if __name__ == "__main__":
    main()
