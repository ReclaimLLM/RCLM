"""Entry point for OpenAI Codex CLI hooks: rclm-codex-hooks <EventName>.

Codex CLI calls this binary for every lifecycle event, passing JSON on stdin.
All handlers are wrapped in try/except — hook failures must never disrupt Codex CLI.
This process always exits 0.

Event mapping from Codex CLI → ReclaimLLM:
  SessionStart     → record cwd + started_at + model
  UserPromptSubmit → record user prompt
  PreToolUse       → record tool invocation (Bash only)
  PostToolUse      → record tool result (Bash only)
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
import json
import logging
import sys
from datetime import datetime, timezone

from rclm import _config
from rclm._models import HookSessionRecord, ToolCall
from rclm._uploader import upload_single
from rclm.hooks import codex_transcript, dedupe, dlp, session_store
from rclm.hooks._analytics import estimate_tokens, mechanism_saving_event
from rclm.hooks.updater import schedule_session_end_update

logger = logging.getLogger(__name__)

THRESHOLD_ZERO_DURATION = 5.0  # seconds


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    # Codex nests the bash command inside tool_input: {"command": "..."}
    tool_input = payload.get("tool_input", {})
    session_store.append_event(
        session_id,
        {
            "event_type": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": tool_input,
            "turn_id": payload.get("turn_id"),
            "timestamp": _now(),
        },
    )


def _handle_post_tool_use(session_id: str, payload: dict) -> None:
    tool_response = payload.get("tool_response")
    prior_events = session_store.read_events(session_id)

    session_store.append_event(
        session_id,
        {
            "event_type": "PostToolUse",
            "tool_name": "Bash",
            "tool_response": tool_response,
            "turn_id": payload.get("turn_id"),
            "timestamp": _now(),
        },
    )

    cfg = _config.load()
    shadow = cfg.get("shadow_mode", False)
    # DLP runs first so dedupe only ever hashes secret-free content.
    effective_text = tool_response if isinstance(tool_response, str) else str(tool_response or "")
    replaced = False

    if cfg.get("dlp", False):
        try:
            cwd = payload.get("cwd", "")
            scrubbed = dlp.maybe_redact_output("Bash", tool_response, cwd)
            if scrubbed is not None:
                effective_text = scrubbed
                replaced = True
        except Exception:
            pass  # Never let DLP disrupt Codex CLI

    compression = _config.compression_config(cfg)
    if compression["dedupe"]:
        try:
            state = session_store.read_dedupe_state(session_id)
            turn = sum(1 for ev in prior_events if ev.get("event_type") == "PostToolUse") + 1
            replacement, state, match = dedupe.maybe_dedupe(
                effective_text,
                state,
                tool_name="Bash",
                turn=turn,
                cwd=payload.get("cwd", ""),
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
                        "hash_dedupe", applied=not shadow, tokens_saved_estimate=saved
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
                    },
                )
                if not shadow:
                    effective_text = replacement
                    replaced = True
        except Exception:
            logger.exception("hash dedupe failed; passing through tool result")

    if replaced:
        # updatedMCPToolOutput is parsed but not applied by Codex CLI (hook run
        # is marked failed and the original output passes through unchanged).
        # decision:"block" + reason is the documented, supported way to
        # replace PostToolUse output.
        print(json.dumps({"decision": "block", "reason": effective_text}))


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

    Codex fires PreToolUse then PostToolUse for each Bash invocation. They
    share the same turn_id. Unmatched PreToolUse events (no PostToolUse) are
    still recorded with tool_result=None.
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
                    tool_use_id=f"codex-tool-{counter}",
                    tool_name="Bash",
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
                tool_name="Bash",
                tool_input=pre.get("tool_input", {}),
                tool_result=None,
                timestamp=pre.get("timestamp", ""),
            )
        )
        counter += 1

    return tool_calls


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
    file_diffs = transcript_data.file_diffs
    model = transcript_data.model or model

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
        total_input_tokens=None,
        total_output_tokens=None,
    )

    asyncio.run(upload_single(record))
    schedule_session_end_update()
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
