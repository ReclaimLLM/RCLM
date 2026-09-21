"""Entry point for Antigravity hooks: rclm-antigravity-hooks <EventName>.

Antigravity PreToolUse supports a shallow ``overwrite`` of tool arguments.
That lets us bound ``view_file`` reads and route supported commands through
the existing compressor before execution. PostToolUse must still return
exactly ``{}``, so result-side transforms remain capture/replay-only for this
provider. Stop uploads the complete transcript.

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
from rclm.hooks import session_store
from rclm.hooks._analytics import compute_session_analytics
from rclm.hooks.antigravity_transcript import parse_transcript
from rclm.hooks.capture_metadata import native_metadata
from rclm.hooks.compress import READ_INJECT_LIMIT, maybe_compress

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


async def _upload_and_close(record: HookSessionRecord):
    """upload_single, then close the module-level aiohttp session before this
    asyncio.run() call's event loop is torn down -- see claude_handler's
    identical helper for why (aiohttp session/loop binding)."""
    try:
        return await upload_single(record)
    finally:
        await close_session()


def _handle_stop(payload: dict) -> None:
    session_id = str(payload.get("conversationId") or "")
    if not session_id:
        logger.warning("rclm-antigravity-hooks: Stop payload missing conversationId")
        return
    if session_store.has_marker(session_id, "finalized"):
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

    model = payload.get("modelName") or "antigravity-unknown"
    record = HookSessionRecord(
        session_id=session_id,
        cwd=cwd,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=_duration_s(started_at, ended_at),
        transcript_path=transcript_path,
        model=model,
        messages=messages,
        tool_calls=transcript_data.tool_calls,
        file_diffs=[],
        tool_token_stats=analytics["tool_token_stats"],
        tool_call_count=analytics["tool_call_count"],
        unique_files_modified=analytics["unique_files_modified"],
        dominant_tool=analytics["dominant_tool"],
        **native_metadata(
            "antigravity",
            adapter_name="antigravity_hooks",
            model=model,
            agent_client_version=payload.get("clientVersion") or payload.get("version"),
            capabilities={"transcript": True, "tool_calls": True, "file_diffs": False},
            warnings=transcript_data.warnings,
        ),
    )

    outcome = asyncio.run(_upload_and_close(record))
    if getattr(outcome, "cleanup_safe", outcome is None):
        session_store.write_marker(session_id, "finalized")


def _handle_pre_tool_use(payload: dict) -> None:
    response: dict = {"decision": "allow"}
    tool_call = payload.get("toolCall")
    if isinstance(tool_call, dict):
        tool_name = tool_call.get("name")
        tool_input = tool_call.get("args")
        if isinstance(tool_name, str) and isinstance(tool_input, dict):
            normalized = tool_name.lower()
            if normalized == "view_file":
                start = tool_input.get("StartLine", 1)
                end = tool_input.get("EndLine")
                if isinstance(start, int) and not isinstance(start, bool) and start >= 1:
                    capped_end = start + READ_INJECT_LIMIT - 1
                    if end is None or (
                        isinstance(end, int) and not isinstance(end, bool) and end > capped_end
                    ):
                        response["overwrite"] = {"EndLine": capped_end}
            elif normalized == "run_command":
                command = tool_input.get("CommandLine")
                if isinstance(command, str):
                    updated = maybe_compress(
                        "Bash",
                        {"command": command},
                        session_id=str(payload.get("conversationId") or "") or None,
                    )
                    if updated is not None and isinstance(updated.get("command"), str):
                        response["overwrite"] = {"CommandLine": updated["command"]}
    print(json.dumps(response))


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
