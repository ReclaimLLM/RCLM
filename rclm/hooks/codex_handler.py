"""Entry point for OpenAI Codex CLI hooks: rclm-codex-hooks <EventName>.

Codex CLI calls this binary for every lifecycle event, passing JSON on stdin.
All handlers are wrapped in try/except — hook failures must never disrupt Codex CLI.
This process always exits 0.

Event mapping from Codex CLI → ReclaimLLM:
  SessionStart     → record cwd + started_at + model
  UserPromptSubmit → record user prompt
  PreToolUse       → record tool invocation and conservatively rewrite supported
                     Bash input through the shared compression CLI.
  PostToolUse      → record tool result and replace recognized text results via
                     Codex's documented feedback + continue:false contract.
  Stop             → assemble HookSessionRecord from accumulated events + upload

Codex stdin schema (all events):
  session_id, transcript_path, cwd, hook_event_name, model
  turn_id (PreToolUse, PostToolUse, UserPromptSubmit, Stop)

Event-specific fields:
  PreToolUse:       tool_input.command
  PostToolUse:      tool_response
  UserPromptSubmit: prompt
  Stop:             last_assistant_message, stop_hook_active

File diffs are extracted from the transcript JSONL file at Stop time.
Codex records file edits as ``apply_patch`` tool calls in the transcript
(type=custom_tool_call, name=apply_patch). The patch format is::

    *** Begin Patch
    *** Update File: /path/to/file
    @@
     context line
    -removed line
    +added line
    *** Add File: /path/to/new/file
    +content line
    *** Delete File: /path/to/file
    *** End Patch
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
from rclm._models import HookSessionRecord, ToolCall
from rclm._uploader import close_session, upload_single
from rclm.hooks import (
    bootstrap,
    codex_transcript,
    dedupe,
    dlp,
    image_eviction,
    image_lifecycle,
    read_cache,
    session_store,
    tool_result_transform,
)
from rclm.hooks._analytics import (
    aggregate_mechanism_savings,
    estimate_tokens,
    mechanism_saving_event,
)
from rclm.hooks.compress import maybe_compress
from rclm.hooks.updater import schedule_session_end_update

logger = logging.getLogger(__name__)

THRESHOLD_ZERO_DURATION = 5.0  # seconds

_DLP_STOP_REASON = (
    "ReclaimLLM DLP withheld the original tool result because it contained an env-file "
    "secret; a redacted result was returned instead."
)
_DLP_SCAN_STOP_REASON = (
    "ReclaimLLM DLP withheld the tool result because the env-file secret scan could not "
    "complete safely."
)
_RANGE_CACHE_STOP_REASON = "ReclaimLLM replaced this repeated file read with a range-cache notice."
_COMPACTION_STOP_REASON = (
    "ReclaimLLM compacted this tool result before it entered the model context."
)
_DEDUPE_STOP_REASON = "ReclaimLLM replaced this repeated tool result with a deduplication notice."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_cwd(session_id: str, payload: dict) -> str:
    if payload.get("cwd"):
        return payload["cwd"]
    for event in session_store.read_events(session_id):
        if event.get("event_type") == "SessionStart":
            return event.get("cwd", "")
    return ""


def _handle_session_start(session_id: str, payload: dict) -> None:
    session_store.append_event(
        session_id,
        {
            "event_type": "SessionStart",
            "cwd": payload.get("cwd", ""),
            "model": payload.get("model"),
            "timestamp": _now(),
        },
    )
    with contextlib.suppress(Exception):
        asyncio.run(
            asyncio.wait_for(
                bootstrap.fetch(payload.get("cwd", ""), include_context=False),
                3.0,
            )
        )
    session_store.append_event(
        session_id,
        {"event_type": "HookPolicySnapshot", "policy": bootstrap.policy_snapshot("codex")},
    )


def _handle_user_prompt_submit(session_id: str, payload: dict) -> None:
    session_store.append_event(
        session_id,
        {
            "event_type": "UserPromptSubmit",
            "prompt": payload.get("prompt", ""),
            "turn_id": payload.get("turn_id"),
            "timestamp": _now(),
        },
    )


def _handle_pre_tool_use(session_id: str, payload: dict) -> None:
    # Codex nests the bash command inside tool_input: {"command": "..."}. For
    # non-Bash (MCP) tool calls, tool_input carries whatever args that tool
    # takes (e.g. url/viewport for a screenshot tool) — captured here so
    # _handle_post_tool_use can look it up by tool_use_id (with a legacy
    # turn_id fallback) for compaction and eviction keying.
    tool_name = payload.get("tool_name") or "Bash"
    tool_input = payload.get("tool_input", {})
    session_store.append_event(
        session_id,
        {
            "event_type": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_use_id": payload.get("tool_use_id"),
            "turn_id": payload.get("turn_id"),
            "timestamp": _now(),
        },
    )

    cfg = _config.load()
    policy = _config.effective_hook_policy(cfg, provider="codex")
    if not isinstance(tool_input, dict):
        return

    effective_input = dict(tool_input)
    changed = False
    dlp_tool = "Bash" if tool_name in {"Bash", "exec", "exec_command", "shell"} else tool_name

    if _config.dlp_enabled(cfg):
        try:

            def _track(path: str) -> None:
                session_store.append_event(session_id, {"event_type": "DLPTempFile", "path": path})

            delta = dlp.maybe_redact_input(
                dlp_tool,
                effective_input,
                _resolve_cwd(session_id, payload),
                track_temp=_track,
            )
            if delta:
                effective_input.update(delta)
                changed = True
        except dlp.DLPRedactionError as exc:
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                f"ReclaimLLM DLP blocked this env-file read: {exc}"
                            ),
                        }
                    }
                )
            )
            return

    if policy.enabled("exec_compaction"):
        try:
            delta = maybe_compress(
                dlp_tool,
                effective_input,
                shadow=policy.shadow_for("exec_compaction"),
                session_id=session_id,
            )
            if delta:
                effective_input.update(delta)
                changed = True
        except Exception:
            logger.exception("Codex input compaction failed; passing through tool input")

    if changed:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "updatedInput": effective_input,
                    }
                }
            )
        )


def _handle_post_tool_use(session_id: str, payload: dict) -> None:
    tool_name = payload.get("tool_name") or "Bash"
    tool_response = payload.get("tool_response")
    prior_events = session_store.read_events(session_id)

    cfg = _config.load()
    policy = _config.effective_hook_policy(cfg, provider="codex")
    cwd = _resolve_cwd(session_id, payload)
    dlp_replacement: str | None = None
    replacement_stop_reason: str | None = None
    captured_response = tool_response
    is_image_result = (
        tool_name.startswith("mcp__")
        and policy.enabled("image_downscale")
        and image_lifecycle.find_image(tool_response) is not None
    )
    if _config.dlp_enabled(cfg) and not is_image_result:
        try:
            tool_input = payload.get("tool_input", {})
            redact_all = dlp.input_may_read_env(tool_name, tool_input)
            if isinstance(tool_response, str):
                dlp_replacement = dlp.maybe_redact_output(
                    tool_name, tool_response, cwd, redact_all=redact_all
                )
                if dlp_replacement is not None:
                    captured_response = dlp_replacement
                    replacement_stop_reason = _DLP_STOP_REASON
            else:
                redacted_value = dlp.maybe_redact_value(tool_response, cwd, redact_all=redact_all)
                if redacted_value is not None:
                    captured_response = redacted_value
                    dlp_replacement = str(redacted_value)
                    replacement_stop_reason = _DLP_STOP_REASON
        except dlp.DLPRedactionError as exc:
            if dlp.input_may_read_env(tool_name, payload.get("tool_input", {})):
                dlp_replacement = f"[rclm DLP] Output withheld: {exc}"
                captured_response = dlp_replacement
                replacement_stop_reason = _DLP_SCAN_STOP_REASON
            else:
                logger.warning("DLP could not inspect %s output: %s", tool_name, exc)

    session_store.append_event(
        session_id,
        {
            "event_type": "PostToolUse",
            "tool_name": tool_name,
            "tool_response": captured_response,
            "dlp_redacted": dlp_replacement is not None,
            "turn_id": payload.get("turn_id"),
            "timestamp": _now(),
        },
    )

    shadow = policy.legacy_shadow
    turn_id = payload.get("turn_id")
    tool_use_id = payload.get("tool_use_id")
    pre_event = None
    if tool_use_id:
        pre_event = next(
            (
                event
                for event in reversed(prior_events)
                if event.get("event_type") == "PreToolUse"
                and event.get("tool_use_id") == tool_use_id
            ),
            None,
        )
    if pre_event is None:
        pre_event = next(
            (
                event
                for event in reversed(prior_events)
                if event.get("event_type") == "PreToolUse" and event.get("turn_id") == turn_id
            ),
            None,
        )
    tool_input = pre_event.get("tool_input", {}) if pre_event else {}

    # MCP tool calls carry the unambiguous "mcp__<server>__<tool>" naming
    # convention (confirmed empirically). Everything else — "Bash",
    # "exec_command", or any other shell-tool spelling Codex may use across
    # versions/platforms — stays on the unchanged pipeline below, which
    # assumes tool_input.command/.cmd and a text result. Matching only the
    # unambiguous MCP prefix (rather than an exhaustive "!= Bash" check) means
    # this branch can never accidentally divert a real shell call.
    if tool_name.startswith("mcp__"):
        # MCP tool results: measurement-only, never a rewrite. Codex's
        # updatedMCPToolOutput field is parsed but not applied by Codex CLI
        # (confirmed by direct reproduction, not just this comment — the hook
        # run is marked failed and the original output passes through
        # unchanged).
        if policy.enabled("image_downscale"):
            try:
                if image_lifecycle.find_image(tool_response) is not None:
                    image_result = image_lifecycle.maybe_downscale_image_result(
                        tool_response,
                        max_dim=int(cfg.get("image_max_dim", image_lifecycle.DEFAULT_MAX_DIM)),
                    )
                    if image_result is not None:
                        session_store.append_event(
                            session_id,
                            mechanism_saving_event(
                                image_lifecycle.MECHANISM,
                                # Codex can never apply this — a platform
                                # limitation (broken updatedMCPToolOutput), not
                                # a shadow_mode choice.
                                applied=False,
                                tokens_saved_estimate=image_result.tokens_saved_estimate,
                                measurement_kind="measured",
                                raw_token_estimate=image_result.raw_token_estimate,
                                compressed_token_estimate=image_result.compressed_token_estimate,
                            ),
                        )

                    turn = (
                        sum(1 for ev in prior_events if ev.get("event_type") == "PostToolUse") + 1
                    )
                    eviction_state = session_store.read_image_eviction_state(session_id)
                    eviction_measurement, eviction_state = image_eviction.maybe_track_eviction(
                        tool_response,
                        eviction_state,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        turn=turn,
                    )
                    session_store.write_image_eviction_state(session_id, eviction_state)
                    if eviction_measurement is not None:
                        session_store.append_event(
                            session_id,
                            mechanism_saving_event(
                                image_eviction.MECHANISM,
                                applied=False,
                                tokens_saved_estimate=eviction_measurement["net_tokens"],
                                measurement_kind="estimated",
                                raw_token_estimate=eviction_measurement["would_save_tokens"],
                                modeled_cost_estimate=eviction_measurement["modeled_cost_tokens"],
                            ),
                        )
            except Exception:
                logger.exception("codex image lifecycle measurement failed; no-op")
        return

    # DLP runs first so dedupe only ever hashes secret-free content.
    effective_text = (
        captured_response if isinstance(captured_response, str) else str(captured_response or "")
    )
    replaced = False

    if dlp_replacement is not None:
        effective_text = dlp_replacement
        replaced = True

    range_claimed = False
    if policy.enabled("range_cache"):
        shadow = policy.shadow_for("range_cache")
        try:
            command = tool_input.get("command") or tool_input.get("cmd")
            shell = tool_input.get("shell") or ("posix" if os.name == "posix" else os.name)
            request = (
                read_cache.parse_shell_read(command, cwd=cwd, shell=shell)
                if isinstance(command, str)
                else None
            )
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
                    tool_use_id=f"codex-turn-{turn_id}",
                    shadow=shadow,
                )
                session_store.write_read_cache_state(session_id, application.state)
                for event in application.events:
                    session_store.append_event(session_id, {**event, "turn_id": turn_id})
                if application.replacement is not None:
                    effective_text = application.replacement
                    replaced = True
                    replacement_stop_reason = replacement_stop_reason or _RANGE_CACHE_STOP_REASON
        except Exception:
            logger.exception("range cache failed; passing through tool result")

    if policy.enabled("exec_compaction") and not range_claimed:
        shadow = policy.shadow_for("exec_compaction")
        try:
            transform_input = effective_text if replaced else tool_response
            decision = tool_result_transform.compact_tool_result(
                tool_name,
                tool_input,
                transform_input,
            )
            if decision is not None:
                for event in tool_result_transform.analytics_events(
                    decision,
                    tool_use_id=payload.get("tool_use_id"),
                    applied=not shadow,
                    turn_id=turn_id,
                ):
                    session_store.append_event(session_id, event)
                if not shadow:
                    effective_text = decision.model_text
                    replaced = True
                    replacement_stop_reason = replacement_stop_reason or _COMPACTION_STOP_REASON
        except Exception:
            logger.exception("tool-result compaction failed; passing through tool result")

    compression = _config.compression_config(cfg)
    if policy.enabled("hash_dedupe") and compression["dedupe"] and not range_claimed:
        shadow = policy.shadow_for("hash_dedupe")
        try:
            state = session_store.read_dedupe_state(session_id)
            turn = sum(1 for ev in prior_events if ev.get("event_type") == "PostToolUse") + 1
            replacement, state, match = dedupe.maybe_dedupe(
                effective_text,
                state,
                tool_name="Bash",
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
                        "applied": not shadow,
                        "measurement_kind": "measured",
                        "turn_id": turn_id,
                    },
                )
                if not shadow:
                    effective_text = replacement
                    replaced = True
                    replacement_stop_reason = replacement_stop_reason or _DEDUPE_STOP_REASON
        except Exception:
            logger.exception("hash dedupe failed; passing through tool result")

    if replaced:
        # Codex documents this pair as model-visible feedback replacement.
        # continue:false prevents normal processing of the original result and,
        # unlike block-only output, does not reject a nested code-mode promise.
        print(
            json.dumps(
                {
                    "continue": False,
                    "decision": "block",
                    "reason": effective_text,
                    "stopReason": replacement_stop_reason,
                }
            )
        )


# ---------------------------------------------------------------------------
# Stop assembly helpers
# ---------------------------------------------------------------------------


def _build_messages(events: list[dict], last_assistant_message: str) -> list[dict]:
    """Reconstruct conversation turns from UserPromptSubmit events + final assistant message."""
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
    if last_assistant_message:
        messages.append(
            {
                "role": "assistant",
                "content": last_assistant_message,
                "timestamp": _now(),
            }
        )
    return messages


def _build_tool_calls(events: list[dict]) -> list[ToolCall]:
    """Pair PreToolUse + PostToolUse events by turn_id to build ToolCall list.

    Codex fires PreToolUse then PostToolUse for each tool invocation (Bash, or
    an MCP tool call when --image-lifecycle widens the hook matcher). They
    share the same turn_id. Unmatched PreToolUse events (no PostToolUse) are
    still recorded with tool_result=None.

    tool_name is read from the PostToolUse event, not the paired PreToolUse
    event: PostToolUse is guaranteed to carry it going forward, while
    PreToolUse pairing can legitimately fail (session killed mid-tool).
    Sessions captured before this change have no tool_name field on either
    event, hence the "Bash" fallback — every event recorded back when this
    pipeline was hardcoded Bash-only really was a Bash call.
    """
    pre_events: dict[str | None, dict] = {}  # turn_id → event
    tool_calls: list[ToolCall] = []
    counter = 0

    for ev in events:
        if ev.get("event_type") == "PreToolUse":
            turn_id = ev.get("turn_id")
            pre_events[turn_id] = ev
        elif ev.get("event_type") == "PostToolUse":
            turn_id = ev.get("turn_id")
            pre = pre_events.pop(turn_id, None)
            tool_input = pre.get("tool_input", {}) if pre else {}
            timestamp = (
                pre.get("timestamp", ev.get("timestamp", "")) if pre else ev.get("timestamp", "")
            )
            tool_calls.append(
                ToolCall(
                    tool_use_id=f"codex-turn-{turn_id}",
                    tool_name=ev.get("tool_name") or "Bash",
                    tool_input=tool_input,
                    tool_result=ev.get("tool_response"),
                    timestamp=timestamp,
                )
            )
            counter += 1

    # Any PreToolUse events with no matching PostToolUse (e.g. session killed mid-tool)
    for pre in pre_events.values():
        tool_calls.append(
            ToolCall(
                tool_use_id=f"codex-tool-{counter}",
                tool_name=pre.get("tool_name") or "Bash",
                tool_input=pre.get("tool_input", {}),
                tool_result=None,
                timestamp=pre.get("timestamp", ""),
            )
        )
        counter += 1

    return tool_calls


_TRANSFORMATION_FIELDS = (
    "was_compressed",
    "compression_strategy",
    "raw_token_estimate",
    "compressed_token_estimate",
    "tokens_saved_estimate",
    "compression_ratio",
)


def _apply_transformation(call: ToolCall, transformation: dict) -> None:
    for key in _TRANSFORMATION_FIELDS:
        setattr(call, key, transformation.get(key))
    call.extra_fields["compression_applied"] = transformation.get("applied", True)
    if transformation.get("measurement_kind"):
        call.extra_fields["measurement_kind"] = transformation["measurement_kind"]
    if transformation.get("file_path"):
        call.extra_fields["compression_file_path"] = transformation["file_path"]


def _tool_command(call: ToolCall) -> str | None:
    command = call.tool_input.get("command") or call.tool_input.get("cmd")
    if isinstance(command, list) and all(isinstance(part, str) for part in command):
        return " ".join(command)
    return command if isinstance(command, str) else None


def _attach_transformations(
    tool_calls: list[ToolCall], fallback_calls: list[ToolCall], events: list[dict]
) -> None:
    """Map hook transformations onto transcript calls without relying on provider IDs."""
    transformations = {
        event.get("tool_use_id"): event
        for event in events
        if event.get("event_type") == "ToolTransformation" and event.get("tool_use_id")
    }
    for call in fallback_calls:
        transformation = transformations.get(call.tool_use_id)
        if transformation:
            _apply_transformation(call, transformation)
    if tool_calls is fallback_calls:
        return

    cursor = 0
    for fallback in fallback_calls:
        command = _tool_command(fallback)
        if command is None:
            continue
        match_index = next(
            (
                index
                for index in range(cursor, len(tool_calls))
                if _tool_command(tool_calls[index]) == command
            ),
            None,
        )
        if match_index is None:
            continue
        transformation = transformations.get(fallback.tool_use_id)
        if transformation:
            _apply_transformation(tool_calls[match_index], transformation)
        cursor = match_index + 1


async def _upload_and_close(record: HookSessionRecord) -> None:
    """upload_single, then close the module-level aiohttp session before this
    asyncio.run() call's event loop is torn down -- see claude_handler's
    identical helper for why (aiohttp session/loop binding)."""
    try:
        await upload_single(record)
    finally:
        await close_session()


def _handle_stop(session_id: str, payload: dict) -> None:
    now = _now()
    events = session_store.read_events(session_id)

    cwd = payload.get("cwd", "")
    started_at = now
    model = payload.get("model")
    for ev in events:
        if ev.get("event_type") == "SessionStart":
            cwd = cwd or ev.get("cwd", "")
            started_at = ev.get("timestamp", now)
            model = model or ev.get("model")
            break

    ended_at = now
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
    transcript_data = codex_transcript.parse_transcript(transcript_path)

    last_assistant_message = payload.get("last_assistant_message", "")
    fallback_messages = _build_messages(events, last_assistant_message)
    fallback_tool_calls = _build_tool_calls(events)

    # The transcript is richer than the hook payloads, but the hook-event
    # reconstruction remains as a safety net for missing or unreadable transcripts.
    messages = transcript_data.messages or fallback_messages
    tool_calls = transcript_data.tool_calls or fallback_tool_calls
    _attach_transformations(tool_calls, fallback_tool_calls, events)
    if _config.dlp_enabled():
        dlp.reconcile_captured_tool_results(tool_calls, events)
    file_diffs = transcript_data.file_diffs
    model = transcript_data.model or model
    usage = transcript_data.usage

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
        total_input_tokens=usage.input_tokens if usage else None,
        total_output_tokens=usage.output_tokens if usage else None,
        cache_read_tokens=usage.cached_input_tokens if usage else None,
        usage_source="provider" if usage else None,
        mechanism_savings=aggregate_mechanism_savings(events),
        hook_policy_snapshot=bootstrap.policy_snapshot_from_events(events, "codex"),
    )

    asyncio.run(_upload_and_close(record))
    schedule_session_end_update()
    for event in events:
        if event.get("event_type") == "DLPTempFile":
            with contextlib.suppress(OSError):
                os.unlink(event["path"])
    session_store.cleanup(session_id)


# ---------------------------------------------------------------------------
# Dispatch table + main
# ---------------------------------------------------------------------------

_HANDLERS = {
    "SessionStart": _handle_session_start,
    "UserPromptSubmit": _handle_user_prompt_submit,
    "PreToolUse": _handle_pre_tool_use,
    "PostToolUse": _handle_post_tool_use,
    "Stop": _handle_stop,
}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: rclm-codex-hooks <EventName>", file=sys.stderr)
        sys.exit(0)

    event_name = sys.argv[1]
    handler_fn = _HANDLERS.get(event_name)

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        logger.warning("rclm-codex-hooks: could not parse stdin JSON for event %s", event_name)
        sys.exit(0)

    session_id = payload.get("session_id", "unknown")

    if handler_fn is not None:
        try:
            handler_fn(session_id, payload)
        except Exception:
            logger.exception(
                "rclm-codex-hooks: unhandled error in handler for event %s", event_name
            )

    sys.exit(0)
