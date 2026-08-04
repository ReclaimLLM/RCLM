"""Entry point for Cursor IDE hooks: rclm-cursor-hooks <EventName>.

Cursor calls this binary for every lifecycle event, passing JSON on stdin.
All handlers are wrapped in try/except — hook failures must never disrupt Cursor.
This process always exits 0.

Cursor hook events are registered in _HANDLERS below.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from rclm import _config
from rclm._models import FileDiff, HookSessionRecord, ToolCall
from rclm._uploader import upload_single
from rclm.hooks import bootstrap, cursor_transcript, dedupe, session_store, tool_result_transform
from rclm.hooks._analytics import aggregate_mechanism_savings
from rclm.hooks.compress import maybe_compress
from rclm.hooks.cursor_transcript import _clean_user_text

logger = logging.getLogger(__name__)
CURSOR_MODEL_DEFAULT = "cursor-unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_cwd(session_id: str, payload: dict) -> str:
    """Return the CWD for this session: payload first, then stored events, then ''."""
    cwd = payload.get("cwd", "")
    if cwd:
        return cwd
    for ev in session_store.read_events(session_id):
        if "cwd" in ev:
            return ev["cwd"]
    return ""


def _derive_session_id_from_transcript_path(transcript_path: str | None) -> str | None:
    if not transcript_path:
        return None
    path = Path(transcript_path)
    return path.stem or path.parent.name or None


def _resolve_transcript_path(cwd: str, session_id: str) -> str | None:
    """
    Resolve the Cursor transcript path based on CWD and session_id.
    Pattern: ~/.cursor/projects/[project-slug]/agent-transcripts/[session-id]/[session-id].jsonl
    """
    if not cwd or not session_id or session_id == "unknown":
        return None

    # Cursor project slugs are typically the absolute path with slashes replaced by dashes.
    # e.g., /Users/maziz/Desktop/Project -> Users-maziz-Desktop-Project
    project_slug = cwd.strip("/").replace("/", "-")

    path = (
        Path.home()
        / ".cursor"
        / "projects"
        / project_slug
        / "agent-transcripts"
        / session_id
        / f"{session_id}.jsonl"
    )

    if path.exists():
        return str(path)

    # Try a recursive search in projects if direct slug fails (slug might be different)
    base_projects = Path.home() / ".cursor" / "projects"
    if base_projects.exists():
        # Look for the session-id directory
        for session_dir in base_projects.glob(f"*/agent-transcripts/{session_id}"):
            f = session_dir / f"{session_id}.jsonl"
            if f.exists():
                return str(f)

    return None


def _transcript_path_from_payload(payload: dict) -> str | None:
    transcript_path = payload.get("transcript_path") or payload.get("transcriptPath")
    if isinstance(transcript_path, str) and transcript_path:
        return transcript_path
    return None


def _handle_before_submit_prompt(session_id: str, payload: dict) -> None:
    session_store.append_event(
        session_id,
        {
            "event_type": "UserPromptSubmit",
            "prompt": _clean_user_text(payload.get("prompt", "")),
            "timestamp": payload.get("timestamp", _now()),
        },
    )


def _handle_session_start(session_id: str, payload: dict) -> None:
    cwd = payload.get("cwd", "")
    session_store.append_event(
        session_id,
        {
            "event_type": "SessionStart",
            "cwd": cwd,
            "timestamp": payload.get("timestamp", _now()),
        },
    )
    with contextlib.suppress(Exception):
        asyncio.run(
            asyncio.wait_for(
                bootstrap.fetch(cwd, include_context=False),
                3.0,
            )
        )
    session_store.append_event(
        session_id,
        {"event_type": "HookPolicySnapshot", "policy": bootstrap.policy_snapshot("cursor")},
    )


def _handle_before_shell_execution(session_id: str, payload: dict) -> None:
    session_store.append_event(
        session_id,
        {
            "event_type": "PreToolUse",
            "tool_name": "shell",
            "tool_input": {"command": payload.get("command", "")},
            "timestamp": payload.get("timestamp", _now()),
        },
    )


def _handle_pre_tool_use(session_id: str, payload: dict) -> None:
    """Record a generic tool call and rewrite supported input when Cursor allows it."""
    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input")
    tool_input = tool_input if isinstance(tool_input, dict) else {}
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

    cfg = _config.load()
    policy = _config.effective_hook_policy(cfg, provider="cursor")
    if not policy.enabled("exec_compaction"):
        return

    try:
        compression_tool = "Bash" if tool_name.lower() in {"shell", "bash"} else tool_name
        delta = maybe_compress(
            compression_tool,
            tool_input,
            shadow=policy.shadow_for("exec_compaction"),
            session_id=None if session_id == "unknown" else session_id,
        )
        if delta:
            print(json.dumps({"updated_input": {**tool_input, **delta}}))
    except Exception:
        logger.exception("Cursor input compaction failed; passing through tool input")


def _handle_post_tool_use(session_id: str, payload: dict) -> None:
    """Record a result and deduplicate MCP text through Cursor's replacement field."""
    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input")
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    tool_output = payload.get("tool_output")
    prior_events = session_store.read_events(session_id)
    session_store.append_event(
        session_id,
        {
            "event_type": "PostToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": tool_output,
            "tool_use_id": payload.get("tool_use_id"),
            "timestamp": payload.get("timestamp", _now()),
        },
    )

    # Cursor documents updated_mcp_tool_output for MCP tools only. Shell and
    # native-tool compaction therefore happens in PreToolUse, not by assuming
    # the contradictory Shell example supports post-result replacement.
    is_mcp = tool_name.startswith("MCP:") or tool_name.startswith("mcp__")
    if not is_mcp:
        return

    cfg = _config.load()
    policy = _config.effective_hook_policy(cfg, provider="cursor")
    compression = _config.compression_config(cfg)
    if not policy.enabled("hash_dedupe") or not compression["dedupe"]:
        return

    envelope = tool_result_transform.extract_text_envelope(tool_output)
    if envelope is None:
        return
    try:
        state = session_store.read_dedupe_state(session_id)
        turn = sum(1 for event in prior_events if event.get("event_type") == "PostToolUse") + 1
        replacement, state, match = dedupe.maybe_dedupe(
            envelope.text,
            state,
            tool_name=tool_name,
            turn=turn,
            cwd=_resolve_cwd(session_id, payload),
            min_chars=int(compression["min_dedupe_chars"]),
        )
        session_store.write_dedupe_state(session_id, state)
        if not replacement or match is None:
            return
        decision = tool_result_transform.decision_from_replacement(
            envelope,
            replacement,
            mechanism="hash_dedupe",
        )
        if decision is None or not isinstance(decision.structured_replacement, dict):
            return
        shadow = policy.shadow_for("hash_dedupe")
        for event in tool_result_transform.analytics_events(
            decision,
            tool_use_id=payload.get("tool_use_id"),
            applied=not shadow,
        ):
            session_store.append_event(session_id, event)
        if not shadow:
            print(json.dumps({"updated_mcp_tool_output": decision.structured_replacement}))
    except Exception:
        logger.exception("Cursor MCP dedupe failed; passing through tool output")


def _unified_diff(path: str, before: str | None, after: str | None) -> str:
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
                    unified_diff=_unified_diff(path, before, after),
                    timestamp=timestamp,
                )
            )
        return diffs

    diff = payload.get("diff")
    if isinstance(diff, str) and diff:
        return [
            FileDiff(
                path=path,
                before=None,
                after=None,
                unified_diff=diff,
                timestamp=timestamp,
            )
        ]

    return []


def _handle_after_file_edit(
    session_id: str, payload: dict, hook_event: str = "afterFileEdit"
) -> None:
    session_store.append_event(
        session_id,
        {
            "event_type": "FileEdit",
            "hook_event": hook_event,
            "filepath": payload.get("file_path")
            or payload.get("filepath")
            or payload.get("path")
            or "",
            "edits": payload.get("edits", []),
            "diff": payload.get("diff", ""),
            "timestamp": payload.get("timestamp", _now()),
        },
    )


def _handle_after_tab_file_edit(session_id: str, payload: dict) -> None:
    _handle_after_file_edit(session_id, payload, hook_event="afterTabFileEdit")


def _handle_generic_event(event_name: str):
    def _handler(session_id: str, payload: dict) -> None:
        session_store.append_event(
            session_id,
            {
                "event_type": event_name,
                "payload": payload,
                "timestamp": payload.get("timestamp", _now()),
            },
        )

    return _handler


def _build_tool_calls_from_events(events: list[dict]) -> list[ToolCall]:
    tool_calls = []
    for i, ev in enumerate(events):
        if ev.get("event_type") == "PreToolUse":
            tool_calls.append(
                ToolCall(
                    tool_use_id=f"cursor-shell-{i}",
                    tool_name=ev.get("tool_name", "shell"),
                    tool_input=ev.get("tool_input", {}),
                    tool_result=None,
                    timestamp=ev.get("timestamp", ""),
                )
            )
    return tool_calls


def _extract_file_diffs_from_events(events: list[dict]) -> list[FileDiff]:
    diffs = []
    for ev in events:
        if ev.get("event_type") == "FileEdit":
            diffs.extend(_extract_file_diffs_from_payload(ev))
    return diffs


def _merge_file_diffs(primary: list[FileDiff], secondary: list[FileDiff]) -> list[FileDiff]:
    merged = list(primary)
    seen = {(diff.path, diff.before, diff.after, diff.unified_diff) for diff in merged}
    for diff in secondary:
        key = (diff.path, diff.before, diff.after, diff.unified_diff)
        if key not in seen:
            merged.append(diff)
            seen.add(key)
    return merged


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


def _handle_stop(session_id: str, payload: dict) -> None:
    now = _now()
    events = session_store.read_events(session_id)
    cwd = _resolve_cwd(session_id, payload)

    # Try to resolve and parse the Cursor transcript file.
    transcript_path = _transcript_path_from_payload(payload) or _resolve_transcript_path(
        cwd, session_id
    )
    transcript_data = None
    if transcript_path:
        transcript_data = cursor_transcript.parse_transcript(transcript_path)
    if transcript_data and transcript_data.cwd:
        cwd = cwd or transcript_data.cwd
    event_file_diffs = _extract_file_diffs_from_events(events)
    mechanism_savings = aggregate_mechanism_savings(events)
    hook_policy_snapshot = bootstrap.policy_snapshot_from_events(events, "cursor")

    # Reconstruct the session record.
    started_at = events[0].get("timestamp", now) if events else now
    ended_at = payload.get("timestamp", now)

    try:
        duration_s = (
            datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)
        ).total_seconds()
    except (ValueError, TypeError):
        duration_s = 0.0

    if transcript_data and (transcript_data.messages or transcript_data.tool_calls):
        # Use data parsed from transcript
        record = HookSessionRecord(
            session_id=session_id,
            cwd=cwd,
            started_at=started_at,
            ended_at=ended_at,
            duration_s=duration_s,
            transcript_path=transcript_path,
            model=transcript_data.model or CURSOR_MODEL_DEFAULT,
            messages=transcript_data.messages,
            tool_calls=transcript_data.tool_calls,
            file_diffs=_merge_file_diffs(event_file_diffs, transcript_data.file_diffs),
            total_input_tokens=transcript_data.total_input_tokens,
            total_output_tokens=transcript_data.total_output_tokens,
            mechanism_savings=mechanism_savings,
            hook_policy_snapshot=hook_policy_snapshot,
        )
    else:
        # Fallback to accumulated events
        record = HookSessionRecord(
            session_id=session_id,
            cwd=cwd,
            started_at=started_at,
            ended_at=ended_at,
            duration_s=duration_s,
            transcript_path=transcript_path,
            model=CURSOR_MODEL_DEFAULT,
            messages=_build_messages_from_events(events),
            tool_calls=_build_tool_calls_from_events(events),
            file_diffs=event_file_diffs,
            mechanism_savings=mechanism_savings,
            hook_policy_snapshot=hook_policy_snapshot,
        )

    asyncio.run(upload_single(record))
    session_store.cleanup(session_id)


_HANDLERS = {
    "preToolUse": _handle_pre_tool_use,
    "postToolUse": _handle_post_tool_use,
    "postToolUseFailure": _handle_generic_event("postToolUseFailure"),
    "subagentStart": _handle_generic_event("subagentStart"),
    "subagentStop": _handle_generic_event("subagentStop"),
    "beforeSubmitPrompt": _handle_before_submit_prompt,
    "beforeShellExecution": _handle_before_shell_execution,
    "beforeMCPExecution": _handle_generic_event("beforeMCPExecution"),
    "afterShellExecution": _handle_generic_event("afterShellExecution"),
    "afterMCPExecution": _handle_generic_event("afterMCPExecution"),
    "afterFileEdit": _handle_after_file_edit,
    "beforeReadFile": _handle_generic_event("beforeReadFile"),
    "beforeTabFileRead": _handle_generic_event("beforeTabFileRead"),
    "afterTabFileEdit": _handle_after_tab_file_edit,
    "afterAgentResponse": _handle_generic_event("afterAgentResponse"),
    "afterAgentThought": _handle_generic_event("afterAgentThought"),
    "stop": _handle_stop,
    "sessionStart": _handle_session_start,
    "sessionEnd": _handle_stop,
    "preCompact": _handle_generic_event("preCompact"),
}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: rclm-cursor-hooks <EventName>", file=sys.stderr)
        sys.exit(0)

    event_name = sys.argv[1]
    handler = _HANDLERS.get(event_name)
    if handler is None:
        sys.exit(0)

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        logger.warning(
            "rclm-cursor-hooks: could not parse stdin JSON for event %s",
            event_name,
        )
        sys.exit(0)

    # Cursor uses conversation_id or generation_id. We'll prefer conversation_id as session_id.
    session_id = (
        payload.get("conversation_id")
        or payload.get("session_id")
        or payload.get("generation_id")
        or _derive_session_id_from_transcript_path(_transcript_path_from_payload(payload))
        or "unknown"
    )

    try:
        handler(session_id, payload)
    except Exception:
        logger.exception(
            "rclm-cursor-hooks: unhandled error in handler for event %s",
            event_name,
        )

    sys.exit(0)
