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
import shlex
import sys
from datetime import datetime, timezone

from rclm import _config
from rclm._models import FileDiff, HookSessionRecord, ToolCall
from rclm._uploader import upload_single
from rclm.hooks import (
    brevity,
    dedupe,
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
    merge_mechanism_savings,
)
from rclm.hooks.compress import maybe_compress
from rclm.hooks.handoff_advisor import thresholds_from_config
from rclm.hooks.loop_breaker import analyze as analyze_loop
from rclm.hooks.updater import schedule_session_end_update
from rclm.mcp_server import ReclaimLLMClient, ReclaimLLMError

logger = logging.getLogger(__name__)

THRESHOLD_ZERO_DURATION = (
    5.0  # seconds; if session duration is below this, treat as zero and omit timestamps
)

CONTEXT_PACK_TIMEOUT_S = 3.0  # SessionStart blocks Claude Code's startup; keep this tight.
CONTEXT_PACK_LIMIT = 3

# Handoff-advisor thresholds use transcript-estimated tokens. This undercounts
# real billed tokens, but is a fine relative growth signal within one session.
HANDOFF_ADVISOR_MARKER = "handoff_advisor_shown"


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
    # Optimization cleanup must never block Claude Code startup.
    with contextlib.suppress(OSError):
        session_store.prune_stale_sidecars()

    session_store.append_event(
        session_id,
        {
            "event_type": "SessionStart",
            "cwd": payload.get("cwd", ""),
            "timestamp": payload.get("timestamp", _now()),
            "model": payload.get("model", "claude-unknown"),
        },
    )

    cwd = payload.get("cwd", "")
    cfg = _config.load()
    context_blocks: list[str] = []

    if cfg.get("context_pack", False) and cwd:
        try:
            context = asyncio.run(
                asyncio.wait_for(_build_context_pack(cwd), CONTEXT_PACK_TIMEOUT_S)
            )
        except Exception:
            context = None  # Never let a slow/failed backend call disrupt Claude Code startup.
        if context:
            context_blocks.append(context)

    try:
        brevity_result = brevity.build_session_start_context(cwd, cfg)
    except Exception:
        brevity_result = None  # Never let brevity detection disrupt Claude Code startup.
    if brevity_result:
        context_blocks.append(brevity_result["text"])
        session_store.append_event(
            session_id,
            {
                "event_type": "BrevityInjected",
                "instruction_hash": brevity_result["instruction_hash"],
                "instruction_tokens": brevity_result["instruction_tokens"],
            },
        )

    if context_blocks:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": "\n\n".join(context_blocks),
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
    cwd = _resolve_cwd(session_id, payload)

    if cfg.get("dlp", False):
        try:

            def _track(path: str) -> None:
                session_store.append_event(session_id, {"event_type": "DLPTempFile", "path": path})

            dlp_delta = dlp.maybe_redact_input(tool_name, effective_input, cwd, track_temp=_track)
            if dlp_delta:
                effective_input.update(dlp_delta)
                changed = True
        except Exception:
            pass  # Never let DLP disrupt Claude Code

    read_offset: int | None = None
    if cfg.get("read_cache", False):
        try:
            read_state = session_store.read_read_cache_state(session_id)
            if tool_name in {"Write", "Edit", "NotebookEdit"}:
                read_state = read_cache.invalidate_tool_path(
                    read_state, tool_name, effective_input, cwd=cwd
                )
                session_store.write_read_cache_state(session_id, read_state)
            elif tool_name == "Read" and not shadow:
                read_offset, read_state = read_cache.next_unseen_offset(
                    effective_input, read_state, cwd=cwd
                )
                session_store.write_read_cache_state(session_id, read_state)
            elif tool_name == "Bash":
                command = effective_input.get("command", "")
                shell = effective_input.get("shell") or ("posix" if os.name == "posix" else os.name)
                request = (
                    read_cache.parse_shell_read(command, cwd=cwd, shell=shell)
                    if isinstance(command, str)
                    else None
                )
                # rclm-read-cache executes argv directly. Keep the one supported
                # read pipeline on PostToolUse measurement-only until the wrapper
                # grows an equally explicit pipeline executor.
                if (
                    request is not None
                    and "|" not in command
                    and not command.lstrip().startswith("rclm-read-cache ")
                ):
                    effective_input["command"] = (
                        f"CLAUDE_SESSION_ID={shlex.quote(session_id)} rclm-read-cache {command}"
                    )
                    changed = True
                    hook_output.update(
                        {
                            "permissionDecision": "allow",
                            "permissionDecisionReason": (
                                "ReclaimLLM validated this as an exact local file read and "
                                "routed it through the session cache."
                            ),
                        }
                    )
                    session_store.append_event(
                        session_id,
                        {
                            "event_type": "ReadCacheWrapped",
                            "tool_use_id": payload.get("tool_use_id"),
                        },
                    )
        except Exception:
            pass  # Never let read-cache state handling disrupt Claude Code

    if _config.compression_config(cfg)["enabled"]:
        try:
            # Bash rewrites always apply — they route through rclm-compress, which
            # measures and makes its own shadow/enforce output decision. Native
            # Read/Grep shaping is suppressed in shadow mode (see maybe_compress).
            compress_delta = maybe_compress(
                tool_name,
                effective_input,
                shadow=shadow,
                read_offset=read_offset,
            )
            if compress_delta:
                effective_input.update(compress_delta)
                changed = True
        except Exception:
            pass  # Never let compression disrupt Claude Code

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
    """Extract the textual payload Claude exposed for a completed tool call."""
    if isinstance(tool_response, str):
        return tool_response
    if isinstance(tool_response, dict):
        stdout = tool_response.get("stdout")
        stderr = tool_response.get("stderr")
        if isinstance(stdout, str) or isinstance(stderr, str):
            return (stdout if isinstance(stdout, str) else "") + (
                stderr if isinstance(stderr, str) else ""
            )

        content = tool_response.get("content")
        if isinstance(content, str):
            return content

        file_result = tool_response.get("file")
        if isinstance(file_result, dict) and isinstance(file_result.get("content"), str):
            return file_result["content"]

    return str(tool_response or "")


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
    cwd = _resolve_cwd(session_id, payload)

    if cfg.get("dlp", False):
        try:
            cwd = _resolve_cwd(session_id, payload)
            scrubbed = dlp.maybe_redact_output(tool_name, effective_text, cwd)
            if scrubbed is not None:
                effective_text = scrubbed
                hook_output["updatedToolOutput"] = scrubbed
                context_notes.append("[rclm DLP] Secrets were redacted from the tool response.")
        except Exception:
            pass  # Never let DLP disrupt Claude Code

    tool_use_id = payload.get("tool_use_id")
    range_claimed = any(
        event.get("event_type") == "ReadCacheWrapped" and event.get("tool_use_id") == tool_use_id
        for event in prior_events
    )
    if cfg.get("read_cache", False) and tool_name in {"Read", "Bash"}:
        try:
            request = None
            if tool_name == "Read":
                request = read_cache.native_read_request(tool_input, cwd=cwd)
            elif not range_claimed:
                command = tool_input.get("command", "")
                shell = tool_input.get("shell") or ("posix" if os.name == "posix" else os.name)
                if isinstance(command, str):
                    request = read_cache.parse_shell_read(command, cwd=cwd, shell=shell)
            if request is not None:
                range_claimed = True
                state = session_store.read_read_cache_state(session_id)
                turn = (
                    sum(1 for event in prior_events if event.get("event_type") == "PostToolUse") + 1
                )
                application = read_cache.apply_range_cache(
                    request,
                    effective_text,
                    state,
                    turn=turn,
                    tool_use_id=tool_use_id,
                    # Claude 2.1.205 ignores PostToolUse.updatedToolOutput.
                    # Track conservative potential here; Bash enforcement occurs
                    # in rclm-read-cache before Claude receives the tool result.
                    shadow=True,
                )
                session_store.write_read_cache_state(session_id, application.state)
                for event in application.events:
                    session_store.append_event(session_id, event)
        except Exception:
            logger.exception("range cache failed; passing through tool result")

    compression = _config.compression_config(cfg)
    if compression["dedupe"] and not range_claimed:
        try:
            state = session_store.read_dedupe_state(session_id)
            turn = sum(1 for event in prior_events if event.get("event_type") == "PostToolUse") + 1
            replacement, state, match = dedupe.maybe_dedupe(
                effective_text,
                state,
                tool_name=tool_name,
                turn=turn,
                cwd=cwd,
                min_chars=int(compression["min_dedupe_chars"]),
            )
            session_store.write_dedupe_state(session_id, state)
            if replacement and match:
                raw_tokens = estimate_tokens(effective_text)
                compressed_tokens = estimate_tokens(replacement)
                saved = max(0, raw_tokens - compressed_tokens)
                session_store.append_event(
                    session_id,
                    mechanism_saving_event(
                        "hash_dedupe",
                        applied=not shadow,
                        tokens_saved_estimate=saved,
                        measurement_kind="measured",
                    ),
                )
                session_store.append_event(
                    session_id,
                    {
                        "event_type": "ToolTransformation",
                        "tool_use_id": payload.get("tool_use_id"),
                        "was_compressed": True,
                        "compression_strategy": "hash_dedupe",
                        "raw_token_estimate": raw_tokens,
                        "compressed_token_estimate": compressed_tokens,
                        "tokens_saved_estimate": saved,
                        "compression_ratio": len(replacement) / max(1, len(effective_text)),
                        "measurement_kind": "measured",
                        "applied": not shadow,
                    },
                )
                if not shadow:
                    effective_text = replacement
                    hook_output["updatedToolOutput"] = replacement
        except Exception:
            logger.exception("hash dedupe failed; passing through tool result")

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


def _handoff_advisory(transcript_data, cfg: dict) -> str | None:
    """Suggest the handoff MCP tool once a session has grown large.

    Every tool result gets re-sent on every subsequent turn, so cost grows faster than
    turn/token counts alone suggest — flag well before a session reaches an extreme size.
    """
    total_tokens = (transcript_data.total_input_tokens or 0) + (
        transcript_data.total_output_tokens or 0
    )
    tool_call_count = len(transcript_data.tool_calls or [])
    token_threshold, tool_call_threshold = thresholds_from_config(cfg)

    if total_tokens < token_threshold and tool_call_count < tool_call_threshold:
        return None

    return (
        "[rclm] This session has grown large "
        f"(~{total_tokens:,} tokens, {tool_call_count} tool calls so far). Every tool result gets "
        "re-sent on every future turn, so cost grows faster than it looks. Ask the user to compact "
        "with a focused message, for example `/compact focus on the login bug`, then call the "
        "ReclaimLLM `handoff` MCP tool if they want to continue in a fresh session."
    )


def _handle_stop(
    session_id: str,
    payload: dict,
    *,
    schedule_update: bool = False,
) -> None:
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
    cfg = _config.load()

    transcript_tool_calls = len(transcript_data.tool_calls)
    previous_hook_health = session_store.read_hook_health(session_id)
    pre_tool_events = previous_hook_health.get("pre_tool_use_count", 0) + sum(
        1 for event in events if event.get("event_type") == "PreToolUse"
    )
    post_tool_events = previous_hook_health.get("post_tool_use_count", 0) + sum(
        1 for event in events if event.get("event_type") == "PostToolUse"
    )
    tool_failure_events = previous_hook_health.get("tool_failure_count", 0) + sum(
        1 for event in events if event.get("event_type") == "ToolFailure"
    )
    completed_tool_events = post_tool_events + tool_failure_events
    if transcript_tool_calls == 0:
        hook_health_status = "no_tool_calls"
    elif pre_tool_events == 0 and completed_tool_events == 0:
        hook_health_status = "missing_tool_hooks"
    elif pre_tool_events == 0 or completed_tool_events == 0:
        hook_health_status = "incomplete_tool_hooks"
    else:
        hook_health_status = "healthy"

    hook_health = {
        "schema_version": 1,
        "provider": "claude",
        "session_id": session_id,
        "status": hook_health_status,
        "recorded_at": now,
        "transcript_tool_call_count": transcript_tool_calls,
        "pre_tool_use_count": pre_tool_events,
        "post_tool_use_count": post_tool_events,
        "tool_failure_count": tool_failure_events,
        "read_cache_enabled": bool(cfg.get("read_cache", False)),
    }
    try:
        session_store.write_hook_health(session_id, hook_health)
    except OSError:
        logger.exception("failed to persist Claude hook-health metadata")
    transformations = {
        event.get("tool_use_id"): event
        for event in events
        if event.get("event_type") == "ToolTransformation" and event.get("tool_use_id")
    }
    for call in transcript_data.tool_calls:
        transformation = transformations.get(call.tool_use_id)
        if transformation:
            for key in (
                "was_compressed",
                "compression_strategy",
                "raw_token_estimate",
                "compressed_token_estimate",
                "tokens_saved_estimate",
                "compression_ratio",
            ):
                setattr(call, key, transformation.get(key))
            call.extra_fields["compression_applied"] = transformation.get("applied", True)
            if transformation.get("measurement_kind"):
                call.extra_fields["measurement_kind"] = transformation["measurement_kind"]
            if transformation.get("file_path"):
                call.extra_fields["compression_file_path"] = transformation["file_path"]
    file_diffs = _extract_file_diffs_from_tool_calls(transcript_data.tool_calls)

    # Compute analytics from tool calls and file diffs.
    analytics = compute_session_analytics(transcript_data.tool_calls, file_diffs)
    turn_mechanism_savings = aggregate_mechanism_savings(events)
    brevity_event = next((ev for ev in events if ev.get("event_type") == "BrevityInjected"), None)
    if brevity_event:
        turn_mechanism_savings = turn_mechanism_savings or {}
        turn_mechanism_savings["brevity"] = {
            "enabled": True,
            "instruction_hash": brevity_event.get("instruction_hash"),
            "instruction_tokens": brevity_event.get("instruction_tokens"),
        }
    mechanism_savings = merge_mechanism_savings(
        session_store.read_mechanism_savings_state(session_id),
        turn_mechanism_savings,
    )

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
    if mechanism_savings:
        try:
            session_store.write_mechanism_savings_state(session_id, mechanism_savings)
        except OSError:
            logger.exception("failed to persist cumulative mechanism savings")
    if schedule_update:
        schedule_session_end_update()

    stop_context_blocks: list[str] = []
    if (
        hook_health_status == "missing_tool_hooks"
        and previous_hook_health.get("status") != "missing_tool_hooks"
    ):
        stop_context_blocks.append(
            "[rclm] Hook health warning: Claude's transcript contains "
            f"{transcript_tool_calls} tool call(s), but ReclaimLLM received no PreToolUse or "
            "PostToolUse events. Read cache and other tool-level mechanisms were inactive. "
            "Run Claude with `--debug hooks` and verify the tool hooks in "
            "`~/.claude/settings.json`."
        )

    if cfg.get("handoff_advisor", False) and not session_store.has_marker(
        session_id,
        HANDOFF_ADVISOR_MARKER,
    ):
        advisory = _handoff_advisory(transcript_data, cfg)
        if advisory:
            session_store.write_marker(session_id, HANDOFF_ADVISOR_MARKER)
            stop_context_blocks.append(advisory)

    if stop_context_blocks:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "Stop",
                        "additionalContext": "\n\n".join(stop_context_blocks),
                    }
                }
            )
        )

    # Clean up any DLP temp files created during this session.
    for ev in events:
        if ev.get("event_type") == "DLPTempFile":
            with contextlib.suppress(OSError):
                os.unlink(ev["path"])

    session_store.cleanup_events(session_id)


def _handle_session_end(session_id: str, payload: dict) -> None:
    """Clean all session-scoped state only when Claude declares the session finished."""
    for event in session_store.read_events(session_id):
        if event.get("event_type") == "DLPTempFile":
            with contextlib.suppress(OSError):
                os.unlink(event["path"])
    session_store.cleanup(session_id)


_HANDLERS = {
    "SessionStart": _handle_session_start,
    "SessionEnd": _handle_session_end,
    "PreToolUse": _handle_pre_tool_use,
    "PostToolUse": _handle_post_tool_use,
    "PostToolUseFailure": _handle_post_tool_use_failure,
    "UserPromptSubmit": _handle_user_prompt_submit,
    "Stop": lambda session_id, payload: _handle_stop(
        session_id,
        payload,
        schedule_update=True,
    ),
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
