"""Entry point for Gemini CLI hooks: rclm-gemini-hooks <EventName>.

Gemini CLI calls this binary for every lifecycle event, passing JSON on stdin.
All handlers are wrapped in try/except — hook failures must never disrupt Gemini CLI.
This process always exits 0 and always prints a JSON object to stdout (Gemini requirement).

Event mapping from Gemini CLI → ReclaimLLM:
  SessionStart  → record cwd + started_at
  BeforeAgent   → record user prompt
  AfterAgent    → record assistant response
  AfterTool     → record tool call + result; extract file diffs
  SessionEnd    → assemble HookSessionRecord from accumulated events + upload

Gemini's tool names for file operations:
  write_file  (fields: file_path, content)       — equivalent to Claude's Write
  replace     (fields: file_path, old_string, new_string) — equivalent to Claude's Edit
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
from rclm.hooks import bootstrap, dedupe, dlp, read_cache, session_store
from rclm.hooks._analytics import (
    aggregate_mechanism_savings,
    estimate_tokens,
    mechanism_saving_event,
)
from rclm.hooks.updater import schedule_session_end_update

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _handle_session_start(session_id: str, payload: dict) -> None:
    session_store.append_event(
        session_id,
        {
            "event_type": "SessionStart",
            "cwd": payload.get("cwd", ""),
            "timestamp": payload.get("timestamp", _now()),
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
        {"event_type": "HookPolicySnapshot", "policy": bootstrap.policy_snapshot("gemini")},
    )


def _handle_before_agent(session_id: str, payload: dict) -> None:
    """Fires before each agentic loop turn; captures the user's prompt."""
    session_store.append_event(
        session_id,
        {
            "event_type": "BeforeAgent",
            "prompt": payload.get("prompt", ""),
            "timestamp": payload.get("timestamp", _now()),
        },
    )


def _handle_after_agent(session_id: str, payload: dict) -> None:
    """Fires after each agentic loop turn; captures the assistant's final response."""
    session_store.append_event(
        session_id,
        {
            "event_type": "AfterAgent",
            "prompt_response": payload.get("prompt_response", ""),
            "timestamp": payload.get("timestamp", _now()),
        },
    )


def _normalise_tool_response(raw: object) -> str:
    """Flatten Gemini's {llmContent, returnDisplay, error} response dict to a string."""
    if isinstance(raw, dict):
        if raw.get("error"):
            return f"Error: {raw['error']}"
        return raw.get("returnDisplay") or raw.get("llmContent") or ""
    return str(raw) if raw is not None else ""


def _resolve_cwd(session_id: str, payload: dict) -> str:
    """Return the CWD for this session: payload first, then SessionStart event, then ''."""
    cwd = payload.get("cwd", "")
    if cwd:
        return cwd
    for ev in session_store.read_events(session_id):
        if ev.get("event_type") == "SessionStart":
            return ev.get("cwd", "")
    return ""


# Gemini tool names whose output may contain secrets.
_DLP_SCRUB_TOOLS = {"run_shell_command", "read_file"}


def _handle_after_tool(session_id: str, payload: dict) -> dict | None:
    """Fires after a tool executes; captures tool name, input, and normalised response.

    Returns a decision:deny dict (per Gemini's AfterTool contract) if DLP or
    dedupe replaced the response, else None.
    """
    tool_name = payload.get("tool_name", "")
    tool_response = _normalise_tool_response(payload.get("tool_response"))
    prior_events = session_store.read_events(session_id)

    session_store.append_event(
        session_id,
        {
            "event_type": "AfterTool",
            "tool_name": tool_name,
            "tool_input": payload.get("tool_input", {}),
            "tool_response": tool_response,
            "tool_use_id": payload.get("tool_use_id"),
            "timestamp": payload.get("timestamp", _now()),
        },
    )

    cfg = _config.load()
    policy = _config.effective_hook_policy(cfg, provider="gemini")
    shadow = policy.legacy_shadow
    cwd = _resolve_cwd(session_id, payload)
    # DLP runs first so dedupe only ever hashes secret-free content.
    effective_text = tool_response
    replaced = False
    tool_input = payload.get("tool_input", {})
    tool_use_id = payload.get("tool_use_id") or f"gemini-tool-{len(prior_events)}"

    if cfg.get("dlp", False) and tool_name in _DLP_SCRUB_TOOLS:
        try:
            scrubbed = dlp.maybe_redact_output(tool_name, tool_response, cwd)
            if scrubbed is not None:
                effective_text = scrubbed
                replaced = True
        except Exception:
            pass  # Never let DLP disrupt Gemini CLI

    if policy.enabled("range_cache") and tool_name in {"write_file", "replace"}:
        try:
            state = session_store.read_read_cache_state(session_id)
            mapped_name = "Write" if tool_name == "write_file" else "Edit"
            state = read_cache.invalidate_tool_path(state, mapped_name, tool_input, cwd=cwd)
            session_store.write_read_cache_state(session_id, state)
        except Exception:
            pass

    range_claimed = False
    if policy.enabled("range_cache") and tool_name in {"read_file", "run_shell_command"}:
        shadow = policy.shadow_for("range_cache")
        try:
            if tool_name == "read_file":
                request = read_cache.native_read_request(tool_input, cwd=cwd)
            else:
                command = tool_input.get("command", "")
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
                    sum(1 for event in prior_events if event.get("event_type") == "AfterTool") + 1
                )
                application = read_cache.apply_range_cache(
                    request,
                    effective_text,
                    state,
                    turn=turn,
                    tool_use_id=tool_use_id,
                    shadow=shadow,
                )
                session_store.write_read_cache_state(session_id, application.state)
                for event in application.events:
                    session_store.append_event(session_id, event)
                if application.replacement is not None:
                    effective_text = application.replacement
                    replaced = True
        except Exception:
            logger.exception("range cache failed; passing through tool result")

    compression = _config.compression_config(cfg)
    if policy.enabled("hash_dedupe") and compression["dedupe"] and not range_claimed:
        shadow = policy.shadow_for("hash_dedupe")
        try:
            state = session_store.read_dedupe_state(session_id)
            turn = sum(1 for ev in prior_events if ev.get("event_type") == "AfterTool") + 1
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
                        "tool_use_id": tool_use_id,
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
                    replaced = True
        except Exception:
            logger.exception("hash dedupe failed; passing through tool result")

    if replaced:
        return {"decision": "deny", "reason": effective_text}

    return None


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------


def _parse_gemini_transcript(transcript_path: str | None) -> dict:
    """Read Gemini's session JSON and extract model + cumulative token counts.

    Gemini writes a live session file at transcript_path. Each assistant turn
    has type=="gemini" and carries:
      - "model": e.g. "gemini-3-flash-preview"
      - "tokens": {"input": int, "output": int, "cached": int, ...}

    Returns a dict with keys model, total_input_tokens, total_output_tokens
    (all None if the file is missing or unreadable).
    """
    result: dict = {
        "model": None,
        "total_input_tokens": None,
        "total_output_tokens": None,
    }
    if not transcript_path:
        return result
    try:
        with open(transcript_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return result

    total_input = 0
    total_output = 0
    has_tokens = False

    for msg in data.get("messages") or []:
        if msg.get("type") != "gemini":
            continue
        if result["model"] is None and msg.get("model"):
            result["model"] = msg["model"]
        tokens = msg.get("tokens") or {}
        if tokens:
            has_tokens = True
            total_input += tokens.get("input", 0)
            total_output += tokens.get("output", 0)

    if has_tokens:
        result["total_input_tokens"] = total_input
        result["total_output_tokens"] = total_output

    return result


# ---------------------------------------------------------------------------
# SessionEnd assembly helpers
# ---------------------------------------------------------------------------


def _build_messages(events: list[dict]) -> list[dict]:
    """Reconstruct conversation turns from BeforeAgent / AfterAgent events."""
    messages = []
    for ev in events:
        if ev.get("event_type") == "BeforeAgent":
            messages.append(
                {
                    "role": "user",
                    "content": ev.get("prompt", ""),
                    "timestamp": ev.get("timestamp", ""),
                }
            )
        elif ev.get("event_type") == "AfterAgent":
            messages.append(
                {
                    "role": "assistant",
                    "content": ev.get("prompt_response", ""),
                    "timestamp": ev.get("timestamp", ""),
                }
            )
    return messages


def _build_tool_calls(events: list[dict]) -> list[ToolCall]:
    """Build ToolCall list from AfterTool events (each has both input and response)."""
    tool_calls = []
    transformations = {
        event.get("tool_use_id"): event
        for event in events
        if event.get("event_type") == "ToolTransformation" and event.get("tool_use_id")
    }
    for i, ev in enumerate(events):
        if ev.get("event_type") != "AfterTool":
            continue
        tool_use_id = ev.get("tool_use_id") or f"gemini-tool-{i}"
        call = ToolCall(
            tool_use_id=tool_use_id,
            tool_name=ev.get("tool_name", ""),
            tool_input=ev.get("tool_input", {}),
            tool_result=ev.get("tool_response"),
            timestamp=ev.get("timestamp", ""),
        )
        transformation = transformations.get(tool_use_id)
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
        tool_calls.append(call)
    return tool_calls


def _extract_file_diffs(events: list[dict]) -> list[FileDiff]:
    """Extract FileDiff objects from write_file and replace tool events."""
    diffs: list[FileDiff] = []
    for ev in events:
        if ev.get("event_type") != "AfterTool":
            continue
        name = ev.get("tool_name", "")
        inp = ev.get("tool_input", {})
        timestamp = ev.get("timestamp", "")

        if name == "write_file":
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
                    timestamp=timestamp,
                )
            )

        elif name == "replace":
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
                    timestamp=timestamp,
                )
            )
    return diffs


def _handle_session_end(session_id: str, payload: dict) -> None:
    now = _now()
    events = session_store.read_events(session_id)

    cwd = payload.get("cwd", "")
    started_at = now
    for ev in events:
        if ev.get("event_type") == "SessionStart":
            cwd = cwd or ev.get("cwd", "")
            started_at = ev.get("timestamp", now)
            break

    ended_at = payload.get("timestamp", now)
    try:
        duration_s = (
            datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)
        ).total_seconds()
    except (ValueError, TypeError):
        duration_s = 0.0

    transcript_path = payload.get("transcript_path")
    transcript_data = _parse_gemini_transcript(transcript_path)

    record = HookSessionRecord(
        session_id=session_id,
        cwd=cwd,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=duration_s,
        transcript_path=transcript_path,
        model=transcript_data["model"] or "gemini-unknown",
        messages=_build_messages(events),
        tool_calls=_build_tool_calls(events),
        file_diffs=_extract_file_diffs(events),
        total_input_tokens=transcript_data["total_input_tokens"],
        total_output_tokens=transcript_data["total_output_tokens"],
        mechanism_savings=aggregate_mechanism_savings(events),
        hook_policy_snapshot=bootstrap.policy_snapshot_from_events(events, "gemini"),
    )

    asyncio.run(upload_single(record))
    schedule_session_end_update()
    session_store.cleanup(session_id)


# ---------------------------------------------------------------------------
# Dispatch table + main
# ---------------------------------------------------------------------------

_HANDLERS = {
    "SessionStart": _handle_session_start,
    "BeforeAgent": _handle_before_agent,
    "AfterAgent": _handle_after_agent,
    "AfterTool": _handle_after_tool,
    "SessionEnd": _handle_session_end,
}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: rclm-gemini-hooks <EventName>", file=sys.stderr)
        print("{}")
        sys.exit(0)

    event_name = sys.argv[1]
    handler_fn = _HANDLERS.get(event_name)

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        logger.warning(
            "rclm-gemini-hooks: could not parse stdin JSON for event %s",
            event_name,
        )
        print("{}")
        sys.exit(0)

    session_id = payload.get("session_id", "unknown")

    hook_output: dict = {}
    if handler_fn is not None:
        try:
            result = handler_fn(session_id, payload)
            if isinstance(result, dict):
                hook_output = result
        except Exception:
            logger.exception(
                "rclm-gemini-hooks: unhandled error in handler for event %s",
                event_name,
            )

    # Gemini CLI requires a JSON object on stdout for every hook invocation.
    print(json.dumps(hook_output))
    sys.exit(0)
