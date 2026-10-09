"""Parse GitHub Copilot CLI event logs into normalized session data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rclm._models import ToolCall
from rclm.hooks.transcript_io import read_jsonl

logger = logging.getLogger(__name__)


@dataclass
class CopilotTranscriptData:
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    session_id: str | None = None
    cwd: str = ""
    model: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    warnings: list[str] = field(default_factory=list)


def parse_transcript(transcript_path: str | None) -> CopilotTranscriptData:
    """Parse a Copilot CLI events.jsonl file, retaining only conversation data."""
    entries, warnings = read_jsonl(transcript_path, logger=logger)
    data = _extract(entries)
    data.warnings = warnings
    return data


def _extract(entries: list[dict]) -> CopilotTranscriptData:
    data = CopilotTranscriptData()
    completed_tools: dict[str, dict] = {}
    started_tools: list[dict] = []
    timestamps: list[str] = []

    for entry in entries:
        event_type = entry.get("type")
        event_data = entry.get("data")
        if not isinstance(event_data, dict):
            continue

        timestamp = entry.get("timestamp")
        if isinstance(timestamp, str) and timestamp:
            timestamps.append(timestamp)

        if event_type == "session.start":
            session_id = event_data.get("sessionId")
            if isinstance(session_id, str) and session_id:
                data.session_id = data.session_id or session_id
        elif event_type in {"user.message", "assistant.message"}:
            content = event_data.get("content")
            if isinstance(content, str) and content:
                role = "user" if event_type == "user.message" else "assistant"
                message = {"role": role, "content": content}
                if isinstance(timestamp, str) and timestamp:
                    message["timestamp"] = timestamp
                data.messages.append(message)
            model = event_data.get("model")
            if isinstance(model, str) and model:
                data.model = data.model or model
        elif event_type == "tool.execution_start":
            tool_call_id = event_data.get("toolCallId")
            if isinstance(tool_call_id, str) and tool_call_id:
                started_tools.append(
                    {
                        "tool_use_id": tool_call_id,
                        "tool_name": event_data.get("toolName"),
                        "tool_input": event_data.get("arguments"),
                        "timestamp": timestamp if isinstance(timestamp, str) else "",
                    }
                )
        elif event_type == "tool.execution_complete":
            tool_call_id = event_data.get("toolCallId")
            if isinstance(tool_call_id, str) and tool_call_id:
                completed_tools[tool_call_id] = event_data
        elif event_type == "hook.start":
            hook_input = event_data.get("input")
            if isinstance(hook_input, dict):
                cwd = hook_input.get("cwd")
                if isinstance(cwd, str) and cwd:
                    data.cwd = data.cwd or cwd
                session_id = hook_input.get("sessionId")
                if isinstance(session_id, str) and session_id:
                    data.session_id = data.session_id or session_id

    for started in started_tools:
        tool_call_id = started["tool_use_id"]
        completion = completed_tools.get(tool_call_id)
        result = completion.get("result") if completion else None
        tool_name = started.get("tool_name")
        tool_input = started.get("tool_input")
        data.tool_calls.append(
            ToolCall(
                tool_use_id=tool_call_id,
                tool_name=tool_name if isinstance(tool_name, str) and tool_name else "unknown",
                tool_input=tool_input if isinstance(tool_input, dict) else {},
                tool_result=result,
                timestamp=started["timestamp"],
            )
        )

    if timestamps:
        data.started_at = min(timestamps)
        data.ended_at = max(timestamps)
    return data
