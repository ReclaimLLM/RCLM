"""Entry point for Claude Code hooks: rclm-claude-hooks <EventName>.

Claude Code calls this binary for every lifecycle event, passing JSON on stdin.
All handlers are wrapped in try/except — hook failures must never disrupt Claude Code.
This process always exits 0.
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
from rclm._uploader import upload_single
from rclm.hooks import (
    dlp,
    read_cache,
    session_store,
    transcript,  # noqa: E402
)
from rclm.hooks._analytics import (
    aggregate_mechanism_savings,
    compute_session_analytics,
    estimate_tokens,
    mechanism_saving_event,
)
from rclm.hooks.compress import maybe_compress
from rclm.hooks.loop_breaker import analyze as analyze_loop
from rclm.mcp_server import ReclaimLLMClient, ReclaimLLMError

logger = logging.getLogger(__name__)

THRESHOLD_ZERO_DURATION = (
    5.0  # seconds; if session duration is below this, treat as zero and omit timestamps
)

CONTEXT_PACK_TIMEOUT_S = 3.0  # SessionStart blocks Claude Code's startup; keep this tight.
CONTEXT_PACK_LIMIT = 3

# Handoff-advisor thresholds: transcript-estimate tokens (see PRD G1 — this
# undercounts real billed tokens, but is a fine relative growth signal within
# one session) or tool-call count, whichever trips first.
HANDOFF_TOKEN_THRESHOLD = 80_000
HANDOFF_TOOL_CALL_THRESHOLD = 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_cwd(session_id: str, payload: dict) -> str:
    """Return the CWD for this session: payload first, then SessionStart event, then ''."""
    cwd = payload.get("cwd", "")
    if cwd:
        return cwd
    for ev in session_store.read_events(session_id):
        if ev.get("event_type") == "SessionStart":
            return ev.get("cwd", "")
    return ""


async def _build_context_pack(cwd: str) -> str | None:
    """Recent ReclaimLLM sessions that touched this project, as SessionStart additionalContext.

    Reuses the same search-by-filename backend call as the search_by_filename MCP tool, passing
    cwd as the path — RCLM already indexes sessions by the file paths they touched, and cwd is a
    reasonable proxy for "this project" without needing the server-side project_name resolution.
    """
    try:
        client = ReclaimLLMClient()
    except ReclaimLLMError:
        return None  # no MCP credentials configured; skip silently, don't block startup

    try:
        result = await client.search_sessions(
            None,
            project_name=None,
            file_path=cwd,
            record_type="session",
            limit=CONTEXT_PACK_LIMIT,
        )
    except ReclaimLLMError:
        return None

    sessions = result.get("sessions") or []
    if not sessions:
        return None

    lines = ["[rclm] Recent sessions in this project (via ReclaimLLM):"]
    for s in sessions:
        title = s.get("title") or "Untitled session"
        highlight = s.get("highlight") or ""
        lines.append(f"- {title}" + (f" — {highlight}" if highlight else ""))
    return "\n".join(lines)


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

    if not _config.load().get("context_pack", False):
        return

    cwd = payload.get("cwd", "")
    if not cwd:
        return

    try:
        context = asyncio.run(asyncio.wait_for(_build_context_pack(cwd), CONTEXT_PACK_TIMEOUT_S))
    except Exception:
        context = None  # Never let a slow/failed backend call disrupt Claude Code startup.

    if context:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": context,
                    }
                }
            )
        )


def _handle_pre_tool_use(session_id: str, payload: dict) -> None:
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})

    # Captured before this call is recorded, so loop-breaker sees only prior history.
    prior_events = session_store.read_events(session_id)

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
    shadow = cfg.get("shadow_mode", False)
    # Accumulate delta updates from DLP and compress.  DLP runs first (security
    # takes priority over optimisation, and is never shadow-suppressed); compress
    # sees the DLP-adjusted input.
    effective_input = dict(tool_input)
    changed = False
    hook_output: dict = {"hookEventName": "PreToolUse"}

    if cfg.get("dlp", False):
        try:
            cwd = _resolve_cwd(session_id, payload)

            def _track(path: str) -> None:
                session_store.append_event(session_id, {"event_type": "DLPTempFile", "path": path})

            dlp_delta = dlp.maybe_redact_input(tool_name, effective_input, cwd, track_temp=_track)
            if dlp_delta:
                effective_input.update(dlp_delta)
                changed = True
        except Exception:
            pass  # Never let DLP disrupt Claude Code

    if cfg.get("compress", False):
        try:
            # Bash rewrites always apply — they route through rclm-compress, which
            # measures and makes its own shadow/enforce output decision. Native
            # Read/Grep shaping is suppressed in shadow mode (see maybe_compress).
            compress_delta = maybe_compress(tool_name, effective_input, shadow=shadow)
            if compress_delta:
                effective_input.update(compress_delta)
                changed = True
        except Exception:
            pass  # Never let compression disrupt Claude Code

    if cfg.get("read_cache", False) and tool_name == "Bash":
        try:
            # Always rewrites — routes through rclm-read-cache, which measures and
            # makes its own shadow/enforce output decision (same as compress above).
            read_cache_delta = read_cache.maybe_wrap_dump_command(effective_input)
            if read_cache_delta:
                effective_input.update(read_cache_delta)
                changed = True
        except Exception:
            pass  # Never let read-cache disrupt Claude Code

    if cfg.get("loop_breaker", False):
        try:
            loop_delta = analyze_loop(tool_name, tool_input, prior_events)
            if loop_delta:
                session_store.append_event(
                    session_id,
                    mechanism_saving_event(
                        "H4_loop_breaker",
                        applied=not shadow,
                        tokens_saved_estimate=estimate_tokens(tool_input),
                    ),
                )
                if not shadow:
                    hook_output.update(loop_delta)
        except Exception:
            pass  # Never let loop detection disrupt Claude Code

    if changed:
        hook_output["updatedInput"] = effective_input

    if len(hook_output) > 1:
        print(json.dumps({"hookSpecificOutput": hook_output}))


def _response_text(tool_response: object) -> str:
    return tool_response if isinstance(tool_response, str) else str(tool_response or "")


def _handle_post_tool_use(session_id: str, payload: dict) -> None:
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})
    tool_response = payload.get("tool_response")

    # Captured before this call is recorded, so read-cache compares against
    # prior reads only.
    prior_events = session_store.read_events(session_id)

    session_store.append_event(
        session_id,
        {
            "event_type": "PostToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": tool_response,
            "tool_use_id": payload.get("tool_use_id"),
            "timestamp": payload.get("timestamp", _now()),
        },
    )

    cfg = _config.load()
    shadow = cfg.get("shadow_mode", False)
    hook_output: dict = {"hookEventName": "PostToolUse"}
    context_notes: list[str] = []

    # DLP runs first so every downstream mechanism (read-cache included) only
    # ever sees secret-free content — never the raw response.
    effective_text = _response_text(tool_response)

    if cfg.get("dlp", False):
        try:
            cwd = _resolve_cwd(session_id, payload)
            scrubbed = dlp.maybe_redact_output(tool_name, tool_response, cwd)
            if scrubbed is not None:
                effective_text = scrubbed
                hook_output["updatedToolOutput"] = scrubbed
                context_notes.append("[rclm DLP] Secrets were redacted from the tool response.")
        except Exception:
            pass  # Never let DLP disrupt Claude Code

    if cfg.get("read_cache", False) and tool_name == "Read":
        try:
            file_path = tool_input.get("file_path", "")
            if file_path:
                offset = tool_input.get("offset")
                limit = tool_input.get("limit")
                timestamp = payload.get("timestamp", _now())

                delta = read_cache.build_delta(
                    file_path, offset, limit, effective_text, prior_events
                )
                session_store.append_event(
                    session_id,
                    read_cache.snapshot_event(file_path, offset, limit, effective_text, timestamp),
                )
                if delta:
                    tokens_saved = max(
                        0,
                        (len(effective_text) - len(delta["updatedToolOutput"])) // 4,
                    )
                    session_store.append_event(
                        session_id,
                        mechanism_saving_event(
                            "H1_read_cache",
                            applied=not shadow,
                            tokens_saved_estimate=tokens_saved,
                        ),
                    )
                    if not shadow:
                        hook_output["updatedToolOutput"] = delta["updatedToolOutput"]
        except Exception:
            pass  # Never let read-cache disrupt Claude Code

    if context_notes:
        hook_output["additionalContext"] = " ".join(context_notes)

    if len(hook_output) > 1:
        print(json.dumps({"hookSpecificOutput": hook_output}))


def _handle_post_tool_use_failure(session_id: str, payload: dict) -> None:
    """Record a failed tool call so the loop-breaker can see consecutive failures."""
    session_store.append_event(
        session_id,
        {
            "event_type": "ToolFailure",
            "tool_name": payload.get("tool_name", ""),
            "tool_input": payload.get("tool_input", {}),
            "tool_output": payload.get("tool_output"),
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
                    timestamp=tc.timestamp,
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
                    timestamp=tc.timestamp,
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
                        timestamp=tc.timestamp,
                    )
                )

    return diffs


def _handoff_advisory(transcript_data) -> str | None:
    """Suggest the handoff MCP tool once a session has grown large.

    Every tool result gets re-sent on every subsequent turn, so cost grows faster than
    turn/token counts alone suggest — flag well before a session reaches an extreme size.
    """
    total_tokens = (transcript_data.total_input_tokens or 0) + (
        transcript_data.total_output_tokens or 0
    )
    tool_call_count = len(transcript_data.tool_calls or [])

    if total_tokens < HANDOFF_TOKEN_THRESHOLD and tool_call_count < HANDOFF_TOOL_CALL_THRESHOLD:
        return None

    return (
        "[rclm] This session has grown large "
        f"(~{total_tokens:,} tokens, {tool_call_count} tool calls so far). Every tool result gets "
        "re-sent on every future turn, so cost grows faster than it looks. Consider calling the "
        "ReclaimLLM `handoff` MCP tool to package current context and continue in a fresh session."
    )


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
    mechanism_savings = aggregate_mechanism_savings(events)

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
        cache_read_tokens=transcript_data.cache_read_tokens,
        cache_creation_tokens=transcript_data.cache_creation_tokens,
        usage_source=transcript_data.usage_source,
        tool_token_stats=analytics.get("tool_token_stats"),
        tool_call_count=analytics.get("tool_call_count"),
        unique_files_modified=analytics.get("unique_files_modified"),
        dominant_tool=analytics.get("dominant_tool"),
        mechanism_savings=mechanism_savings,
    )

    asyncio.run(upload_single(record))

    if _config.load().get("handoff_advisor", False):
        advisory = _handoff_advisory(transcript_data)
        if advisory:
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "Stop",
                            "additionalContext": advisory,
                        }
                    }
                )
            )

    # Clean up any DLP temp files created during this session.
    for ev in events:
        if ev.get("event_type") == "DLPTempFile":
            with contextlib.suppress(OSError):
                os.unlink(ev["path"])

    session_store.cleanup(session_id)


_HANDLERS = {
    "SessionStart": _handle_session_start,
    "PreToolUse": _handle_pre_tool_use,
    "PostToolUse": _handle_post_tool_use,
    "PostToolUseFailure": _handle_post_tool_use_failure,
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
