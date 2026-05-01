"""Parse OpenClaw hook events and historical JSONL session files."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rclm._models import ToolCall

logger = logging.getLogger(__name__)

MAX_RAW_STRING_CHARS = 8000
NOISY_MESSAGE_MARKERS = ("HEARTBEAT_OK",)


@dataclass
class OpenClawTranscriptData:
    session_id: str | None = None
    cwd: str = ""
    started_at: str | None = None
    ended_at: str | None = None
    duration_s: float = 0.0
    model: str | None = None
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def first_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def session_id_from_payload(payload: dict) -> str:
    event = as_dict(payload.get("event"))
    context = as_dict(event.get("context") or payload.get("context"))
    return (
        first_string(
            payload.get("session_id"),
            payload.get("sessionId"),
            payload.get("sessionKey"),
            event.get("session_id"),
            event.get("sessionId"),
            event.get("sessionKey"),
            context.get("session_id"),
            context.get("sessionId"),
            context.get("sessionKey"),
        )
        or "unknown"
    )


def cwd_from_payload(payload: dict) -> str:
    event = as_dict(payload.get("event"))
    context = as_dict(event.get("context") or payload.get("context"))
    return first_string(
        payload.get("cwd"),
        payload.get("workspaceDir"),
        event.get("cwd"),
        event.get("workspaceDir"),
        context.get("cwd"),
        context.get("workspaceDir"),
    )


def timestamp_from_payload(payload: dict) -> str:
    event = as_dict(payload.get("event"))
    return (
        first_string(payload.get("timestamp"), event.get("timestamp"), payload.get("received_at"))
        or now_iso()
    )


def model_from_payload(payload: dict) -> str | None:
    event = as_dict(payload.get("event"))
    data = as_dict(payload.get("data"))
    model = first_string(
        payload.get("model"),
        payload.get("modelId"),
        payload.get("modelName"),
        payload.get("model_id"),
        event.get("model"),
        event.get("modelId"),
        event.get("modelName"),
        event.get("model_id"),
        data.get("model"),
        data.get("modelId"),
        data.get("modelName"),
        data.get("model_id"),
        as_dict(payload.get("model")).get("id"),
        as_dict(payload.get("model")).get("name"),
        as_dict(event.get("model")).get("id"),
        as_dict(event.get("model")).get("name"),
    )
    return model or None


def bounded(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= MAX_RAW_STRING_CHARS:
            return value
        return value[:MAX_RAW_STRING_CHARS] + "...[truncated]"
    if isinstance(value, list):
        return [bounded(item) for item in value[:100]]
    if isinstance(value, dict):
        return {str(k): bounded(v) for k, v in list(value.items())[:100]}
    return value


def content_from(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [content_from(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("content", "text", "output", "message", "prompt", "response"):
            content = content_from(value.get(key))
            if content:
                return content
    return ""


def is_noisy_message(content: str) -> bool:
    return any(marker in content for marker in NOISY_MESSAGE_MARKERS)


def messages_from_llm_input(event: dict, timestamp: str) -> list[dict]:
    messages = []
    raw_messages = event.get("messages")
    if isinstance(raw_messages, list):
        for msg in raw_messages:
            msg_dict = as_dict(msg)
            role = first_string(msg_dict.get("role"), msg_dict.get("speaker")) or "user"
            content = content_from(msg_dict)
            if content and not is_noisy_message(content):
                messages.append({"role": role, "content": content, "timestamp": timestamp})
    else:
        prompt = content_from(event.get("prompt") or event.get("input"))
        if prompt and not is_noisy_message(prompt):
            messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
    return messages


def message_from_llm_output(event: dict, timestamp: str) -> dict | None:
    content = content_from(
        event.get("message")
        or event.get("output")
        or event.get("response")
        or event.get("content")
        or event.get("text")
    )
    if not content or is_noisy_message(content):
        return None
    return {"role": "assistant", "content": content, "timestamp": timestamp}


def tool_name_from_event(event: dict) -> str:
    return (
        first_string(event.get("toolName"), event.get("tool_name"), event.get("name")) or "unknown"
    )


def tool_input_from_event(event: dict) -> dict:
    params = event.get("params", event.get("tool_input", event.get("input", {})))
    return params if isinstance(params, dict) else {"value": params}


def tool_result_from_event(event: dict) -> Any:
    if "result" in event:
        return event.get("result")
    if "output" in event:
        return event.get("output")
    if "response" in event:
        return event.get("response")
    if "error" in event:
        return {"error": event.get("error")}
    return None


def tool_id_from_event(event: dict, fallback_index: int) -> str:
    return (
        first_string(
            event.get("toolCallId"),
            event.get("tool_call_id"),
            event.get("toolUseId"),
            event.get("id"),
        )
        or f"openclaw-tool-{fallback_index}"
    )


def started_at_from_events(events: list[dict], fallback: str) -> str:
    for ev in events:
        if ev.get("event_type") == "SessionStart":
            return ev.get("timestamp") or fallback
    return fallback


def first_cwd_from_events(events: list[dict], fallback: str) -> str:
    for ev in events:
        cwd = ev.get("cwd")
        if isinstance(cwd, str) and cwd:
            return cwd
    return fallback


def first_model_from_events(events: list[dict], fallback: str | None) -> str | None:
    for ev in events:
        model = ev.get("model")
        if isinstance(model, str) and model:
            return model
    return fallback or "openclaw-unknown"


def duration_s(started_at: str | None, ended_at: str | None) -> float:
    if not started_at or not ended_at:
        return 0.0
    try:
        start = started_at.replace("Z", "+00:00") if started_at.endswith("Z") else started_at
        end = ended_at.replace("Z", "+00:00") if ended_at.endswith("Z") else ended_at
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except (TypeError, ValueError):
        return 0.0


def build_messages(events: list[dict]) -> list[dict]:
    messages = []
    for ev in events:
        raw_messages = ev.get("messages")
        if isinstance(raw_messages, list):
            for msg in raw_messages:
                msg_dict = as_dict(msg)
                content = msg_dict.get("content")
                if content and not is_noisy_message(str(content)):
                    messages.append(msg_dict)
    return messages


def build_tool_calls(events: list[dict]) -> list[ToolCall]:
    before_by_id: dict[str, dict] = {}
    tool_calls: list[ToolCall] = []
    counter = 0

    for ev in events:
        if ev.get("event_type") == "before_tool_call":
            tool_id = str(ev.get("tool_use_id") or f"openclaw-tool-{counter}")
            before_by_id[tool_id] = ev
        elif ev.get("event_type") == "after_tool_call":
            tool_id = str(ev.get("tool_use_id") or f"openclaw-tool-{counter}")
            before = before_by_id.pop(tool_id, None)
            source = before or ev
            tool_calls.append(
                ToolCall(
                    tool_use_id=tool_id,
                    tool_name=str(source.get("tool_name") or ev.get("tool_name") or "unknown"),
                    tool_input=as_dict(source.get("tool_input") or ev.get("tool_input")),
                    tool_result=ev.get("tool_result"),
                    timestamp=str(source.get("timestamp") or ev.get("timestamp") or ""),
                )
            )
            counter += 1

    for tool_id, ev in before_by_id.items():
        tool_calls.append(
            ToolCall(
                tool_use_id=tool_id,
                tool_name=str(ev.get("tool_name") or "unknown"),
                tool_input=as_dict(ev.get("tool_input")),
                tool_result=None,
                timestamp=str(ev.get("timestamp") or ""),
            )
        )

    return tool_calls


def parse_transcript(transcript_path: str | None) -> OpenClawTranscriptData:
    """Parse an OpenClaw historical JSONL session file.

    OpenClaw historical files may contain forwarded hook payloads or direct event
    objects. The parser accepts both shapes and only retains normalized messages
    and tool calls, so large raw event payloads are not kept in memory.
    """
    data = OpenClawTranscriptData()
    if not transcript_path:
        return data

    path = Path(transcript_path)
    if not path.exists():
        logger.warning("openclaw transcript: file not found: %s", transcript_path)
        return data

    before_by_id: dict[str, dict] = {}
    tool_counter = 0
    seen_messages: set[tuple[str, str, str]] = set()

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "openclaw transcript: malformed JSON line in %s, skipping",
                    transcript_path,
                )
                continue
            if not isinstance(payload, dict):
                continue
            _ingest_payload(data, payload, before_by_id, seen_messages, tool_counter)
            tool_counter = len(data.tool_calls) + len(before_by_id)

    for tool_id, ev in before_by_id.items():
        data.tool_calls.append(
            ToolCall(
                tool_use_id=tool_id,
                tool_name=str(ev.get("tool_name") or "unknown"),
                tool_input=as_dict(ev.get("tool_input")),
                tool_result=None,
                timestamp=str(ev.get("timestamp") or ""),
            )
        )

    data.duration_s = duration_s(data.started_at, data.ended_at)
    return data


def _ingest_payload(
    data: OpenClawTranscriptData,
    payload: dict,
    before_by_id: dict[str, dict],
    seen_messages: set[tuple[str, str, str]],
    tool_counter: int,
) -> None:
    event = as_dict(payload.get("event")) or payload
    hook_name = _normalized_hook_name(payload, event)
    timestamp = timestamp_from_payload(payload)

    sid = _native_session_id(payload) or session_id_from_payload(payload)
    if data.session_id is None and sid != "unknown":
        data.session_id = sid

    cwd = cwd_from_payload(payload)
    if not data.cwd and cwd:
        data.cwd = cwd

    model = model_from_payload(payload)
    if data.model is None and model:
        data.model = model

    if data.started_at is None:
        data.started_at = timestamp
    data.ended_at = timestamp

    if payload.get("type") == "message" and isinstance(payload.get("message"), dict):
        _ingest_native_message(data, payload, before_by_id, seen_messages, tool_counter)
        return
    if payload.get("type") == "session":
        data.started_at = timestamp
        return
    if payload.get("type") in {"model_change", "custom"}:
        return

    if hook_name == "session_start":
        data.started_at = timestamp
        return
    if hook_name == "session_end":
        data.ended_at = timestamp
        return
    if hook_name == "llm_input":
        for message in messages_from_llm_input(event, timestamp):
            _append_message(data.messages, seen_messages, message)
        return
    if hook_name == "llm_output":
        message = message_from_llm_output(event, timestamp)
        if message:
            _append_message(data.messages, seen_messages, message)
        return
    if hook_name == "before_tool_call":
        tool_id = tool_id_from_event(event, tool_counter)
        before_by_id[tool_id] = {
            "tool_name": tool_name_from_event(event),
            "tool_input": tool_input_from_event(event),
            "timestamp": timestamp,
        }
        return
    if hook_name == "after_tool_call":
        tool_id = _historical_after_tool_id(event, before_by_id, tool_counter)
        before = before_by_id.pop(tool_id, None)
        source = before or {}
        data.tool_calls.append(
            ToolCall(
                tool_use_id=tool_id,
                tool_name=str(source.get("tool_name") or tool_name_from_event(event)),
                tool_input=as_dict(source.get("tool_input") or tool_input_from_event(event)),
                tool_result=tool_result_from_event(event),
                timestamp=str(source.get("timestamp") or timestamp),
            )
        )
        return

    _append_direct_message(data.messages, seen_messages, event, timestamp)


def _native_session_id(payload: dict) -> str:
    if payload.get("type") == "session":
        return first_string(payload.get("id"))
    return ""


def _ingest_native_message(
    data: OpenClawTranscriptData,
    payload: dict,
    before_by_id: dict[str, dict],
    seen_messages: set[tuple[str, str, str]],
    tool_counter: int,
) -> None:
    message = as_dict(payload.get("message"))
    timestamp = first_string(payload.get("timestamp"), message.get("timestamp")) or now_iso()
    role = first_string(message.get("role"), message.get("speaker"))

    if role in {"user", "assistant", "system"}:
        content = _native_text_content(message.get("content"))
        if content and not is_noisy_message(content):
            _append_message(
                data.messages,
                seen_messages,
                {"role": role, "content": content, "timestamp": timestamp},
            )
        _capture_native_tool_calls(message.get("content"), before_by_id, timestamp, tool_counter)
        return

    if role == "toolResult":
        _capture_native_tool_result(message, before_by_id, data.tool_calls, timestamp, tool_counter)


def _native_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content_from(content)

    parts = []
    for item in content:
        item_dict = as_dict(item)
        item_type = item_dict.get("type")
        if item_type in {"text", "input_text", "output_text"}:
            text = item_dict.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts)


def _capture_native_tool_calls(
    content: Any,
    before_by_id: dict[str, dict],
    timestamp: str,
    tool_counter: int,
) -> None:
    if not isinstance(content, list):
        return
    for offset, item in enumerate(content):
        item_dict = as_dict(item)
        if item_dict.get("type") != "toolCall":
            continue
        tool_id = first_string(item_dict.get("id")) or f"openclaw-tool-{tool_counter + offset}"
        arguments = item_dict.get("arguments")
        before_by_id[tool_id] = {
            "tool_name": first_string(item_dict.get("name")) or "unknown",
            "tool_input": arguments if isinstance(arguments, dict) else {"value": arguments},
            "timestamp": timestamp,
        }


def _capture_native_tool_result(
    message: dict,
    before_by_id: dict[str, dict],
    tool_calls: list[ToolCall],
    timestamp: str,
    tool_counter: int,
) -> None:
    tool_id = first_string(message.get("toolCallId"), message.get("tool_call_id"))
    if not tool_id:
        tool_id = _historical_after_tool_id(message, before_by_id, tool_counter)
    before = before_by_id.pop(tool_id, None)
    result = _native_text_content(message.get("content")) or tool_result_from_event(message)
    tool_calls.append(
        ToolCall(
            tool_use_id=tool_id,
            tool_name=str((before or {}).get("tool_name") or message.get("toolName") or "unknown"),
            tool_input=as_dict((before or {}).get("tool_input")),
            tool_result=result,
            timestamp=str((before or {}).get("timestamp") or timestamp),
        )
    )


def _append_direct_message(
    messages: list[dict],
    seen_messages: set[tuple[str, str, str]],
    event: dict,
    timestamp: str,
) -> None:
    role = first_string(event.get("role"), event.get("speaker"))
    if role not in {"user", "assistant", "system"}:
        return
    content = content_from(event)
    if not content or is_noisy_message(content):
        return
    _append_message(
        messages, seen_messages, {"role": role, "content": content, "timestamp": timestamp}
    )


def _append_message(
    messages: list[dict],
    seen_messages: set[tuple[str, str, str]],
    message: dict,
) -> None:
    role = str(message.get("role") or "")
    content = str(message.get("content") or "")
    timestamp = str(message.get("timestamp") or "")
    if not role or not content:
        return
    fingerprint = (role, content, timestamp)
    if fingerprint in seen_messages:
        return
    seen_messages.add(fingerprint)
    messages.append({"role": role, "content": content, "timestamp": timestamp})


def _normalized_hook_name(payload: dict, event: dict) -> str:
    raw = first_string(
        payload.get("hook_name"),
        payload.get("hookName"),
        payload.get("event_type"),
        event.get("hook_name"),
        event.get("hookName"),
        event.get("event_type"),
        payload.get("type"),
        event.get("type"),
    )
    mapping = {
        "SessionStart": "session_start",
        "session_start": "session_start",
        "SessionEnd": "session_end",
        "session_end": "session_end",
        "llm_input": "llm_input",
        "llm_output": "llm_output",
        "before_tool_call": "before_tool_call",
        "after_tool_call": "after_tool_call",
    }
    return mapping.get(raw, raw)


def _historical_after_tool_id(
    event: dict, before_by_id: dict[str, dict], fallback_index: int
) -> str:
    explicit = first_string(
        event.get("toolCallId"),
        event.get("tool_call_id"),
        event.get("toolUseId"),
        event.get("id"),
    )
    if explicit:
        return explicit
    if len(before_by_id) == 1:
        return next(iter(before_by_id))
    return tool_id_from_event(event, fallback_index)
