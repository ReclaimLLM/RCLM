"""Entry point for VS Code / GitHub Copilot hooks: rclm-copilot-hooks <EventName>.

VS Code Local agent / GitHub Copilot calls this binary for every lifecycle event,
passing JSON on stdin. All handlers are wrapped in try/except — hook failures
must never disrupt the agent session. This process always exits 0.

VS Code Agent Hook event mapping:
  SessionStart     → record cwd + started_at + model, fetch bootstrap policy/context
  UserPromptSubmit → record user prompt
  PreToolUse       → record tool invocation, check DLP, apply compression if enabled
  PostToolUse      → record tool result, check DLP on result
  Stop             → assemble HookSessionRecord from accumulated events + upload
  SubagentStart    → record subagent start
  SubagentStop     → record subagent stop
  PreCompact       → record pre-compact checkpoint

VS Code stdin schema (common fields):
  session_id, transcript_path, cwd, hook_event_name, timestamp

Event-specific fields:
  SessionStart:     source ("new")
  UserPromptSubmit: prompt
  PreToolUse:       tool_name, tool_input, tool_use_id
  PostToolUse:      tool_name, tool_input, tool_use_id, tool_response
  Stop:             stop_hook_active
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import logging
import os
import sys
from datetime import datetime, timezone

from rclm import _config
from rclm._models import FileDiff, HookSessionRecord, ToolCall
from rclm._uploader import close_session, upload_single
from rclm.hooks import (
    bootstrap,
    dlp,
    session_store,
)
from rclm.hooks._analytics import aggregate_mechanism_savings
from rclm.hooks.capture_metadata import native_metadata
from rclm.hooks.compress import maybe_compress
from rclm.hooks.updater import schedule_session_end_update

logger = logging.getLogger(__name__)

COPILOT_CAPTURE_CAPABILITIES = {
    "native_lifecycle_hooks": True,
    "transcript_primary": False,
    "hook_event_fallback": True,
    "messages": True,
    "tool_calls": True,
    "file_changes": True,
    "provider_usage": False,
    "model_facing_transformations": True,
    "historical_sync": True,
}


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


def _resolve_cwd(session_id: str, payload: dict) -> str:
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        return cwd
    for event in session_store.read_events(session_id):
        if event.get("event_type") == "SessionStart":
            stored_cwd = event.get("cwd")
            if isinstance(stored_cwd, str) and stored_cwd.strip():
                return stored_cwd
    return ""


async def _upload_and_close(record: HookSessionRecord):
    try:
        return await upload_single(record)
    finally:
        await close_session()


def _handle_session_start(session_id: str, payload: dict) -> None:
    cwd = _resolve_cwd(session_id, payload)
    now = payload.get("timestamp") or _now()
    model = payload.get("model") or "copilot-unknown"
    session_store.append_event(
        session_id,
        {
            "event_type": "SessionStart",
            "cwd": cwd,
            "model": model,
            "source": payload.get("source", "new"),
            "timestamp": now,
        },
    )
    bootstrap_data: dict = {}
    with contextlib.suppress(Exception):
        bootstrap_data = asyncio.run(
            asyncio.wait_for(
                bootstrap.fetch(cwd, include_context=True),
                3.0,
            )
        )
    session_store.append_event(
        session_id,
        {"event_type": "HookPolicySnapshot", "policy": bootstrap.policy_snapshot("copilot")},
    )

    cfg = _config.load()
    if cfg.get("context_pack", False) and cwd and bootstrap_data:
        ctx = bootstrap.context_text(bootstrap_data)
        if ctx:
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "SessionStart",
                            "additionalContext": ctx,
                        }
                    }
                )
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


def _handle_pre_tool_use(session_id: str, payload: dict) -> None:
    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input")
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    tool_use_id = payload.get("tool_use_id") or ""
    now = payload.get("timestamp", _now())
    cfg = _config.load()
    cwd = _resolve_cwd(session_id, payload)

    effective_input = dict(tool_input)
    changed = False

    dlp_tool = (
        "Bash"
        if tool_name.lower() in {"shell", "bash", "terminal", "runcommand", "run_command"}
        else tool_name
    )

    if _config.dlp_enabled(cfg):
        try:

            def _track(path: str) -> None:
                session_store.append_event(session_id, {"event_type": "DLPTempFile", "path": path})

            delta = dlp.maybe_redact_input(
                dlp_tool,
                effective_input,
                cwd,
                track_temp=_track,
            )
            if delta:
                effective_input.update(delta)
                changed = True
        except dlp.DLPRedactionError as exc:
            logger.warning("Copilot cannot safely rewrite this env-file read: %s", exc)
            session_store.append_event(
                session_id,
                {
                    "event_type": "PreToolUse",
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_use_id": tool_use_id,
                    "timestamp": now,
                    "blocked": True,
                },
            )
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": f"ReclaimLLM DLP blocked tool: {exc}",
                        }
                    }
                )
            )
            return

    session_store.append_event(
        session_id,
        {
            "event_type": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": effective_input,
            "tool_use_id": tool_use_id,
            "timestamp": now,
        },
    )

    policy = _config.effective_hook_policy(cfg, provider="copilot")
    if policy.enabled("exec_compaction"):
        try:
            delta = maybe_compress(
                dlp_tool,
                effective_input,
                shadow=policy.shadow_for("exec_compaction"),
                session_id=None if session_id == "unknown" else session_id,
            )
            if delta:
                effective_input.update(delta)
                changed = True
        except Exception:
            logger.exception("Copilot input compaction failed; passing through tool input")

    if changed:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "updatedInput": effective_input,
                    }
                }
            )
        )


def _handle_post_tool_use(session_id: str, payload: dict) -> None:
    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input")
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    tool_response = payload.get("tool_response")
    tool_use_id = payload.get("tool_use_id") or ""
    now = payload.get("timestamp", _now())
    cfg = _config.load()
    cwd = _resolve_cwd(session_id, payload)

    captured_output = tool_response
    redacted_output = None
    if _config.dlp_enabled(cfg):
        try:
            redacted_output = dlp.maybe_redact_value(
                tool_response,
                cwd,
                redact_all=dlp.input_may_read_env(tool_name, tool_input),
            )
            if redacted_output is not None:
                captured_output = redacted_output
        except dlp.DLPRedactionError as exc:
            if dlp.input_may_read_env(tool_name, tool_input):
                captured_output = f"[rclm DLP] Output withheld: {exc}"
            else:
                logger.warning("DLP could not inspect %s output: %s", tool_name, exc)

    session_store.append_event(
        session_id,
        {
            "event_type": "PostToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": captured_output,
            "dlp_redacted": redacted_output is not None or captured_output is not tool_response,
            "tool_use_id": tool_use_id,
            "timestamp": now,
        },
    )


def _handle_subagent_start(session_id: str, payload: dict) -> None:
    session_store.append_event(
        session_id,
        {
            "event_type": "SubagentStart",
            "agent_id": payload.get("agent_id"),
            "agent_type": payload.get("agent_type"),
            "timestamp": payload.get("timestamp", _now()),
        },
    )


def _handle_subagent_stop(session_id: str, payload: dict) -> None:
    session_store.append_event(
        session_id,
        {
            "event_type": "SubagentStop",
            "agent_id": payload.get("agent_id"),
            "agent_type": payload.get("agent_type"),
            "timestamp": payload.get("timestamp", _now()),
        },
    )


def _handle_pre_compact(session_id: str, payload: dict) -> None:
    session_store.append_event(
        session_id,
        {
            "event_type": "PreCompact",
            "timestamp": payload.get("timestamp", _now()),
        },
    )


def _build_unified_diff(path: str, before: str | None, after: str | None) -> str:
    before_lines = (before or "").splitlines(keepends=True)
    after_lines = (after or "").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def _extract_file_diffs_from_payload(payload: dict) -> list[FileDiff]:
    path = payload.get("file_path") or payload.get("filepath") or payload.get("path") or "unknown"
    timestamp = payload.get("timestamp", "")
    edits = payload.get("edits")
    if isinstance(edits, list):
        diffs = []
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            before = edit.get("old_string")
            after = edit.get("new_string")
            diffs.append(
                FileDiff(
                    path=path,
                    before=before,
                    after=after,
                    unified_diff=_build_unified_diff(path, before, after),
                    timestamp=timestamp,
                )
            )
        return diffs
    return []


def _extract_file_diffs_from_events(events: list[dict]) -> list[FileDiff]:
    diffs = []
    for ev in events:
        if ev.get("event_type") == "FileEdit":
            diffs.extend(_extract_file_diffs_from_payload(ev))
    return diffs


def _build_messages_from_events(events: list[dict]) -> list[dict]:
    messages = []
    for ev in events:
        if ev.get("event_type") == "UserPromptSubmit":
            messages.append(
                {
                    "role": "user",
                    "content": ev.get("prompt", ""),
                    "timestamp": ev.get("timestamp", ""),
                }
            )
    return messages


def _build_tool_calls_from_events(events: list[dict]) -> list[ToolCall]:
    pre_events: list[dict] = []
    tool_calls: list[ToolCall] = []
    counter = 0
    emitted_ids: set[str] = set()

    def _synthetic_id() -> str:
        nonlocal counter
        candidate = f"copilot-tool-{counter}"
        while candidate in emitted_ids:
            counter += 1
            candidate = f"copilot-tool-{counter}"
        emitted_ids.add(candidate)
        return candidate

    for ev in events:
        if ev.get("event_type") == "PreToolUse":
            pre_events.append(ev)
        elif ev.get("event_type") == "PostToolUse":
            tool_use_id = ev.get("tool_use_id")
            match_index = None
            if tool_use_id:
                match_index = next(
                    (
                        index
                        for index, pre_event in enumerate(pre_events)
                        if pre_event.get("tool_use_id") == tool_use_id
                    ),
                    None,
                )
            if match_index is None and pre_events:
                match_index = next(
                    (
                        index
                        for index, pre_event in enumerate(pre_events)
                        if pre_event.get("tool_name") == ev.get("tool_name")
                    ),
                    0,
                )
            pre = (
                pre_events.pop(match_index)
                if match_index is not None and match_index < len(pre_events)
                else None
            )
            tool_input = pre.get("tool_input", {}) if pre else ev.get("tool_input", {})
            timestamp = (
                pre.get("timestamp", ev.get("timestamp", "")) if pre else ev.get("timestamp", "")
            )
            call_id = tool_use_id or (pre or {}).get("tool_use_id")
            if call_id:
                call_id = str(call_id)
                emitted_ids.add(call_id)
            else:
                call_id = _synthetic_id()
            tool_calls.append(
                ToolCall(
                    tool_use_id=call_id,
                    tool_name=ev.get("tool_name") or (pre or {}).get("tool_name") or "unknown",
                    tool_input=tool_input if isinstance(tool_input, dict) else {},
                    tool_result=ev.get("tool_response"),
                    timestamp=timestamp,
                )
            )

    for pre in pre_events:
        call_id = pre.get("tool_use_id")
        if call_id:
            call_id = str(call_id)
            emitted_ids.add(call_id)
        else:
            call_id = _synthetic_id()
        tool_calls.append(
            ToolCall(
                tool_use_id=call_id,
                tool_name=pre.get("tool_name") or "unknown",
                tool_input=pre.get("tool_input", {})
                if isinstance(pre.get("tool_input"), dict)
                else {},
                tool_result=None,
                timestamp=pre.get("timestamp", ""),
            )
        )

    return tool_calls


def _handle_stop(session_id: str, payload: dict) -> None:
    if session_store.has_marker(session_id, "finalized"):
        return
    now = _now()
    events = session_store.read_events(session_id)
    cwd = _resolve_cwd(session_id, payload)

    transcript_path = payload.get("transcript_path")
    started_at = events[0].get("timestamp", now) if events else now
    ended_at = payload.get("timestamp", now)
    duration_s = _duration_s(started_at, ended_at)

    messages = _build_messages_from_events(events)
    tool_calls = _build_tool_calls_from_events(events)
    file_diffs = _extract_file_diffs_from_events(events)
    mechanism_savings = aggregate_mechanism_savings(events)
    hook_policy_snapshot = bootstrap.policy_snapshot_from_events(events, "copilot")

    model = payload.get("model")
    if not model:
        for ev in events:
            if ev.get("event_type") == "SessionStart" and ev.get("model"):
                model = ev["model"]
                break
    model = model or "copilot-unknown"

    capture_warnings = []
    if not messages and not tool_calls:
        capture_warnings.append("empty_session_content")

    record = HookSessionRecord(
        session_id=session_id,
        cwd=cwd,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=duration_s,
        transcript_path=transcript_path,
        model=model,
        messages=messages,
        tool_calls=tool_calls,
        file_diffs=file_diffs,
        mechanism_savings=mechanism_savings,
        hook_policy_snapshot=hook_policy_snapshot,
        **native_metadata(
            "copilot",
            adapter_name="copilot_hooks",
            model=model,
            provider="github",
            agent_client_version=payload.get("copilot_version") or payload.get("version"),
            capabilities=dict(COPILOT_CAPTURE_CAPABILITIES),
            warnings=capture_warnings,
            extra_fields={"client_session_id": session_id},
        ),
    )

    outcome = asyncio.run(_upload_and_close(record))
    schedule_session_end_update()
    if not getattr(outcome, "cleanup_safe", outcome is None):
        return
    for event in events:
        if event.get("event_type") == "DLPTempFile":
            with contextlib.suppress(OSError):
                os.unlink(event["path"])
    session_store.cleanup(session_id)
    session_store.write_marker(session_id, "finalized")


_HANDLERS = {
    "SessionStart": _handle_session_start,
    "UserPromptSubmit": _handle_user_prompt_submit,
    "PreToolUse": _handle_pre_tool_use,
    "PostToolUse": _handle_post_tool_use,
    "Stop": _handle_stop,
    "SubagentStart": _handle_subagent_start,
    "SubagentStop": _handle_subagent_stop,
    "PreCompact": _handle_pre_compact,
}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: rclm-copilot-hooks <EventName>", file=sys.stderr)
        sys.exit(0)

    event_name = sys.argv[1]
    handler_fn = _HANDLERS.get(event_name)

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        logger.warning("rclm-copilot-hooks: could not parse stdin JSON for event %s", event_name)
        sys.exit(0)

    session_id = payload.get("session_id") or payload.get("sessionId") or "unknown"

    if handler_fn is not None:
        try:
            handler_fn(session_id, payload)
        except Exception:
            logger.exception(
                "rclm-copilot-hooks: unhandled error in handler for event %s", event_name
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
