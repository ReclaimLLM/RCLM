"""Entry point for Antigravity hooks: rclm-antigravity-hooks <EventName>.

Antigravity Lifecycle Hooks supported:
  PreInvocation  → injects project context packs, brevity instructions, handoff advice
  PostInvocation → loop tracking and step checkpoints
  PreToolUse     → DLP credential masking/denial, loop breaker, view_file bounding,
                   and command compression
  PostToolUse    → real-time tool completion/failure recording, read-cache invalidation
  Stop           → builds and uploads full session record with Google provider linking

Hook failures must never disrupt Antigravity, so this process always exits 0.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
from datetime import datetime, timezone

from rclm import _config
from rclm._models import HookSessionRecord
from rclm._uploader import close_session, upload_single
from rclm.hooks import (
    bootstrap,
    brevity,
    dlp,
    loop_breaker,
    read_cache,
    session_store,
)
from rclm.hooks._analytics import compute_session_analytics
from rclm.hooks.antigravity_transcript import parse_transcript
from rclm.hooks.capture_metadata import native_metadata
from rclm.hooks.compress import READ_INJECT_LIMIT, maybe_compress
from rclm.hooks.handoff_advisor import thresholds_from_config

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _handle_pre_invocation(payload: dict) -> None:
    session_id = str(payload.get("conversationId") or "")
    workspace_paths = payload.get("workspacePaths") or []
    cwd = workspace_paths[0] if isinstance(workspace_paths, list) and workspace_paths else ""
    model = payload.get("modelName") or "antigravity-unknown"
    now = _now()

    cfg = _config.load()
    inject_steps: list[dict] = []

    if session_id:
        events = session_store.read_events(session_id)
        has_start = any(ev.get("event_type") == "SessionStart" for ev in events)
        if not has_start:
            session_store.append_event(
                session_id,
                {
                    "event_type": "SessionStart",
                    "cwd": cwd,
                    "model": model,
                    "artifact_directory_path": payload.get("artifactDirectoryPath"),
                    "timestamp": now,
                },
            )
            session_store.append_event(
                session_id,
                {
                    "event_type": "HookPolicySnapshot",
                    "policy": bootstrap.policy_snapshot("antigravity"),
                },
            )

        invocation_num = payload.get("invocationNum", 1)
        if isinstance(invocation_num, int) and invocation_num <= 1:
            bootstrap_data: dict = {}
            if bool(cfg.get("context_pack", False) and cwd):
                with contextlib.suppress(Exception):
                    bootstrap_data = asyncio.run(
                        asyncio.wait_for(
                            bootstrap.fetch(cwd, include_context=True),
                            3.0,
                        )
                    )
                ctx = bootstrap.context_text(bootstrap_data)
                if ctx:
                    inject_steps.append({"ephemeralMessage": ctx})
                nudge = bootstrap.signal_nudge_text(bootstrap_data, cfg)
                if nudge:
                    inject_steps.append({"ephemeralMessage": nudge})

            if cfg.get("brevity", False):
                with contextlib.suppress(Exception):
                    brevity_result = brevity.build_session_start_context(cwd, cfg)
                    if brevity_result:
                        inject_steps.append({"ephemeralMessage": brevity_result["text"]})
                        session_store.append_event(
                            session_id,
                            {
                                "event_type": "BrevityInjected",
                                "instruction_hash": brevity_result["instruction_hash"],
                                "instruction_tokens": brevity_result["instruction_tokens"],
                            },
                        )

        if cfg.get("handoff_advisor", True):
            _, tool_call_thresh = thresholds_from_config(cfg)
            initial_steps = payload.get("initialNumSteps") or 0
            if (
                isinstance(initial_steps, int)
                and initial_steps >= tool_call_thresh
                and not session_store.has_marker(session_id, "handoff_advised")
            ):
                inject_steps.append(
                    {
                        "ephemeralMessage": (
                            "[rclm handoff] Session length has reached threshold. Consider running "
                            "the `handoff` MCP tool to preserve context and begin a fresh session."
                        )
                    }
                )
                session_store.write_marker(session_id, "handoff_advised")

    print(json.dumps({"injectSteps": inject_steps}))


def _handle_post_invocation(payload: dict) -> None:
    session_id = str(payload.get("conversationId") or "")
    if session_id:
        session_store.append_event(
            session_id,
            {
                "event_type": "PostInvocation",
                "invocation_num": payload.get("invocationNum"),
                "timestamp": _now(),
            },
        )
    print(json.dumps({"injectSteps": []}))


def _handle_pre_tool_use(payload: dict) -> None:
    response: dict = {"decision": "allow"}
    tool_call = payload.get("toolCall")
    if not isinstance(tool_call, dict):
        print(json.dumps(response))
        return

    tool_name = tool_call.get("name")
    tool_input = tool_call.get("args")
    if not isinstance(tool_name, str) or not isinstance(tool_input, dict):
        print(json.dumps(response))
        return

    session_id = str(payload.get("conversationId") or "")
    cfg = _config.load()
    normalized = tool_name.lower()

    # 1. DLP Protection
    if _config.dlp_enabled(cfg):
        if normalized == "view_file":
            target = tool_input.get("AbsolutePath") or ""
            if isinstance(target, str) and target and dlp._is_env_file(os.path.basename(target)):
                print(
                    json.dumps(
                        {
                            "decision": "deny",
                            "reason": f"[rclm DLP] Blocked: reading {target} is disabled (DLP policy).",
                        }
                    )
                )
                return
        elif normalized in {"write_to_file", "replace_file_content"}:
            target = tool_input.get("TargetFile") or ""
            if isinstance(target, str) and target and dlp._is_env_file(os.path.basename(target)):
                print(
                    json.dumps(
                        {
                            "decision": "deny",
                            "reason": f"[rclm DLP] Blocked: modifying {target} is disabled (DLP policy).",
                        }
                    )
                )
                return
        elif normalized == "run_command":
            cmd = tool_input.get("CommandLine") or ""
            if isinstance(cmd, str) and cmd:
                workspace_paths = payload.get("workspacePaths") or []
                cwd = (
                    workspace_paths[0]
                    if isinstance(workspace_paths, list) and workspace_paths
                    else ""
                )
                redacted_bash = dlp._redact_bash_input({"command": cmd}, cwd=cwd)
                if redacted_bash:
                    print(
                        json.dumps(
                            {
                                "decision": "deny",
                                "reason": "[rclm DLP] Blocked: command may access sensitive env file (DLP policy).",
                            }
                        )
                    )
                    return

    # 2. Loop Breaker
    if cfg.get("loop_breaker", True) and session_id:
        prior_events = session_store.read_events(session_id)
        loop_result = loop_breaker.analyze(tool_name, tool_input, prior_events)
        if loop_result and loop_result.get("permissionDecision") == "ask":
            print(
                json.dumps(
                    {
                        "decision": "ask",
                        "reason": f"[rclm loop-breaker] {loop_result.get('permissionDecisionReason', '')}",
                    }
                )
            )
            return

    # Record PreToolUse event
    if session_id:
        session_store.append_event(
            session_id,
            {
                "event_type": "PreToolUse",
                "tool_name": tool_name,
                "tool_input": tool_input,
                "step_idx": payload.get("stepIdx"),
                "timestamp": _now(),
            },
        )

    # 3. Argument Overwriting (view_file bounding & run_command compression)
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
                session_id=session_id or None,
            )
            if updated is not None and isinstance(updated.get("command"), str):
                response["overwrite"] = {"CommandLine": updated["command"]}

    print(json.dumps(response))


def _handle_post_tool_use(payload: dict) -> None:
    session_id = str(payload.get("conversationId") or "")
    if session_id:
        error = payload.get("error")
        step_idx = payload.get("stepIdx")
        tool_call = payload.get("toolCall")
        tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else ""
        tool_input = (
            tool_call.get("args", {})
            if isinstance(tool_call, dict) and isinstance(tool_call.get("args"), dict)
            else {}
        )
        timestamp = _now()

        if error:
            session_store.append_event(
                session_id,
                {
                    "event_type": "ToolFailure",
                    "step_idx": step_idx,
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "error": str(error),
                    "timestamp": timestamp,
                },
            )
        else:
            session_store.append_event(
                session_id,
                {
                    "event_type": "PostToolUse",
                    "step_idx": step_idx,
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "timestamp": timestamp,
                },
            )
            # Invalidate read cache if an edit/write tool succeeded
            if tool_name and tool_name.lower() in {"write_to_file", "replace_file_content"}:
                with contextlib.suppress(Exception):
                    cfg = _config.load()
                    policy = _config.effective_hook_policy(cfg, provider="antigravity")
                    if policy.enabled("range_cache"):
                        workspace_paths = payload.get("workspacePaths") or []
                        cwd = (
                            workspace_paths[0]
                            if isinstance(workspace_paths, list) and workspace_paths
                            else ""
                        )
                        state = session_store.read_read_cache_state(session_id)
                        target = tool_input.get("TargetFile")
                        if target:
                            state = read_cache.invalidate_tool_path(
                                state, "Write", {"file_path": target}, cwd=cwd
                            )
                            session_store.write_read_cache_state(session_id, state)

    print("{}")


def _handle_stop(payload: dict) -> None:
    session_id = str(payload.get("conversationId") or "")
    if not session_id:
        logger.warning("rclm-antigravity-hooks: Stop payload missing conversationId")
        return

    transcript_path = payload.get("transcriptPath")
    transcript_data = parse_transcript(transcript_path)
    messages = transcript_data.messages

    now = _now()
    timestamps = [m["timestamp"] for m in messages if m.get("timestamp")]
    started_at = min(timestamps) if timestamps else now
    ended_at = max(timestamps) if timestamps else started_at

    workspace_paths = payload.get("workspacePaths")
    cwd = (
        workspace_paths[0]
        if isinstance(workspace_paths, list) and workspace_paths
        else (transcript_data.cwd or "")
    )

    file_diffs = transcript_data.file_diffs
    analytics = compute_session_analytics(transcript_data.tool_calls, file_diffs)

    extra_fields: dict = {}
    if payload.get("terminationReason"):
        extra_fields["termination_reason"] = payload.get("terminationReason")
    if payload.get("error"):
        extra_fields["termination_error"] = payload.get("error")
    if payload.get("fullyIdle") is not None:
        extra_fields["fully_idle"] = payload.get("fullyIdle")
    if payload.get("artifactDirectoryPath"):
        extra_fields["artifact_directory_path"] = payload.get("artifactDirectoryPath")

    model = payload.get("modelName") or transcript_data.model or "antigravity-unknown"
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
        file_diffs=file_diffs,
        tool_token_stats=analytics["tool_token_stats"],
        tool_call_count=analytics["tool_call_count"],
        unique_files_modified=analytics["unique_files_modified"],
        dominant_tool=analytics["dominant_tool"],
        **native_metadata(
            "antigravity",
            adapter_name="antigravity_hooks",
            model=model,
            provider="google",
            agent_client_version=payload.get("clientVersion") or payload.get("version"),
            capabilities={"transcript": True, "tool_calls": True, "file_diffs": bool(file_diffs)},
            warnings=transcript_data.warnings,
            extra_fields=extra_fields,
        ),
    )

    outcome = asyncio.run(_upload_and_close(record))
    if getattr(outcome, "cleanup_safe", outcome is None):
        session_store.cleanup_events(session_id)


def main() -> None:
    hook_name = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        logger.warning("rclm-antigravity-hooks: could not parse stdin JSON for hook %s", hook_name)
        if hook_name in {"PreInvocation", "PostInvocation"}:
            print(json.dumps({"injectSteps": []}))
        elif hook_name == "PreToolUse":
            print(json.dumps({"decision": "allow"}))
        else:
            print("{}")
        sys.exit(0)

    try:
        if hook_name == "PreInvocation":
            _handle_pre_invocation(payload)
        elif hook_name == "PostInvocation":
            _handle_post_invocation(payload)
        elif hook_name == "PreToolUse":
            _handle_pre_tool_use(payload)
        elif hook_name == "PostToolUse":
            _handle_post_tool_use(payload)
        elif hook_name == "Stop":
            _handle_stop(payload)
        else:
            print("{}")
    except Exception:
        logger.exception("rclm-antigravity-hooks: unhandled error in hook %s", hook_name)
        if hook_name in {"PreInvocation", "PostInvocation"}:
            print(json.dumps({"injectSteps": []}))
        elif hook_name == "PreToolUse":
            print(json.dumps({"decision": "allow"}))
        else:
            print("{}")

    sys.exit(0)


if __name__ == "__main__":
    main()
