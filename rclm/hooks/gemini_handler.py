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
import difflib
import json
import logging
import sys
from datetime import datetime, timezone

from rclm import _config
from rclm._models import FileDiff, HookSessionRecord, ToolCall
from rclm._uploader import upload_single
from rclm.hooks import dlp, session_store

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

    Returns a hookSpecificOutput dict if DLP scrubbed the response, else None.
    """
    tool_name = payload.get("tool_name", "")
    tool_response = _normalise_tool_response(payload.get("tool_response"))

    session_store.append_event(
        session_id,
        {
            "event_type": "AfterTool",
            "tool_name": tool_name,
            "tool_input": payload.get("tool_input", {}),
            "tool_response": tool_response,
            "timestamp": payload.get("timestamp", _now()),
        },
    )

    if _config.load().get("dlp", False) and tool_name in _DLP_SCRUB_TOOLS:
        try:
            cwd = _resolve_cwd(session_id, payload)
            scrubbed = dlp.maybe_redact_output(tool_name, tool_response, cwd)
            if scrubbed is not None:
                return {"decision": "deny", "reason": scrubbed}
        except Exception:
            pass  # Never let DLP disrupt Gemini CLI

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
    for i, ev in enumerate(events):
        if ev.get("event_type") != "AfterTool":
            continue
        tool_calls.append(
            ToolCall(
                tool_use_id=f"gemini-tool-{i}",
                tool_name=ev.get("tool_name", ""),
                tool_input=ev.get("tool_input", {}),
                tool_result=ev.get("tool_response"),
                timestamp=ev.get("timestamp", ""),
            )
        )
    return tool_calls


def _extract_file_diffs(events: list[dict]) -> list[FileDiff]:
    """Extract FileDiff objects from write_file and replace tool events."""
    diffs: list[FileDiff] = []
    for ev in events:
        if ev.get("event_type") != "AfterTool":
            continue
        name = ev.get("tool_name", "")
        inp = ev.get("tool_input", {})

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
    )

    asyncio.run(upload_single(record))
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
