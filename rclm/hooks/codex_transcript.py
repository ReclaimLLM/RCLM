"""Parse Codex CLI JSONL transcripts into normalized session data."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from rclm._models import FileDiff, ToolCall

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodexUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0


@dataclass
class CodexUsageAccumulator:
    """Accumulate Codex cumulative token snapshots, including counter resets."""

    completed: CodexUsage = field(default_factory=CodexUsage)
    current: CodexUsage | None = None

    def add_snapshot(self, raw: object) -> None:
        snapshot = _parse_usage_snapshot(raw)
        if snapshot is None:
            return
        if self.current is not None and _usage_decreased(self.current, snapshot):
            self.completed = _add_usage(self.completed, self.current)
        self.current = snapshot

    def total(self) -> CodexUsage | None:
        if self.current is None:
            return None
        return _add_usage(self.completed, self.current)


@dataclass
class CodexTranscriptData:
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    file_diffs: list[FileDiff] = field(default_factory=list)
    model: str | None = None
    usage: CodexUsage | None = None


def parse_transcript(transcript_path: str | None) -> CodexTranscriptData:
    """Parse a Codex CLI JSONL transcript file."""
    if not transcript_path:
        return CodexTranscriptData()

    path = Path(transcript_path)
    if not path.exists():
        logger.warning("codex transcript: file not found: %s", transcript_path)
        return CodexTranscriptData()

    with open(path, encoding="utf-8") as fh:
        return _extract(_jsonl_entries(fh, transcript_path))


def _jsonl_entries(lines: Iterable[str], transcript_path: str) -> Iterable[dict]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(
                "codex transcript: malformed JSON line in %s, skipping",
                transcript_path,
            )
            continue
        if isinstance(entry, dict):
            yield entry


def _extract(entries: Iterable[dict]) -> CodexTranscriptData:
    data = CodexTranscriptData()
    seen_messages: set[tuple[str, str, str]] = set()
    usage = CodexUsageAccumulator()
    # Function call results arrive in separate transcript items keyed by call_id.
    pending_calls: dict[str, ToolCall] = {}

    for entry in entries:
        timestamp = entry.get("timestamp", "")
        entry_type = entry.get("type")
        payload = entry.get("payload") or {}
        if not isinstance(payload, dict):
            continue

        if entry_type == "session_meta":
            model = (
                payload.get("model") or payload.get("rollout_model") or payload.get("model_slug")
            )
            if model and data.model is None:
                data.model = model
            continue

        if entry_type == "turn_context":
            model = payload.get("model")
            if model:
                data.model = str(model)
            continue

        if entry_type == "event_msg":
            if payload.get("type") == "token_count":
                info = payload.get("info")
                if isinstance(info, dict):
                    usage.add_snapshot(info.get("total_token_usage"))
            _extract_event_message(payload, timestamp, data, seen_messages)
            continue

        if entry_type != "response_item":
            continue

        response_type = payload.get("type")
        if response_type == "message":
            _extract_response_message(payload, timestamp, data, seen_messages)
        elif response_type in {"function_call", "custom_tool_call"}:
            call = _build_tool_call(payload, timestamp)
            if call is not None:
                pending_calls[payload.get("call_id", "")] = call
                data.tool_calls.append(call)
                if call.tool_name == "apply_patch":
                    # Preserve provider-neutral FileDiffs by extracting patch hunks
                    # at parse time instead of leaking raw Codex patch text upward.
                    patch_text = call.tool_input.get("input", "")
                    if isinstance(patch_text, str) and patch_text:
                        data.file_diffs.extend(_parse_apply_patch(patch_text, timestamp))
        elif response_type == "function_call_output":
            call_id = payload.get("call_id", "")
            call = pending_calls.get(call_id)
            if call is not None:
                call.tool_result = payload.get("output")

    data.usage = usage.total()
    return data


def _parse_usage_snapshot(raw: object) -> CodexUsage | None:
    if not isinstance(raw, dict):
        return None

    values: list[int] = []
    for key in (
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    ):
        value = raw.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        values.append(value)
    return CodexUsage(*values)


def _usage_decreased(previous: CodexUsage, current: CodexUsage) -> bool:
    return any(
        next_value < previous_value
        for previous_value, next_value in zip(
            (
                previous.input_tokens,
                previous.cached_input_tokens,
                previous.output_tokens,
                previous.reasoning_output_tokens,
            ),
            (
                current.input_tokens,
                current.cached_input_tokens,
                current.output_tokens,
                current.reasoning_output_tokens,
            ),
            strict=True,
        )
    )


def _add_usage(left: CodexUsage, right: CodexUsage) -> CodexUsage:
    return CodexUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        cached_input_tokens=left.cached_input_tokens + right.cached_input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        reasoning_output_tokens=(left.reasoning_output_tokens + right.reasoning_output_tokens),
    )


def _extract_event_message(
    payload: dict,
    timestamp: str,
    data: CodexTranscriptData,
    seen_messages: set[tuple[str, str, str]],
) -> None:
    message_type = payload.get("type")
    if message_type == "user_message":
        _append_message(
            data.messages,
            seen_messages,
            "user",
            payload.get("message", ""),
            timestamp,
        )
    elif message_type == "agent_message":
        _append_message(
            data.messages,
            seen_messages,
            "assistant",
            payload.get("message", ""),
            timestamp,
        )


def _extract_response_message(
    payload: dict,
    timestamp: str,
    data: CodexTranscriptData,
    seen_messages: set[tuple[str, str, str]],
) -> None:
    role = payload.get("role")
    if role not in {"user", "assistant"}:
        return

    parts = payload.get("content") or []
    if not isinstance(parts, list):
        return

    text_parts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "output_text"}:
            text = part.get("text")
            if text:
                text_parts.append(text)

    if text_parts:
        _append_message(
            data.messages,
            seen_messages,
            role,
            "\n".join(text_parts),
            timestamp,
        )


def _append_message(
    messages: list[dict],
    seen_messages: set[tuple[str, str, str]],
    role: str,
    content: str,
    timestamp: str,
) -> None:
    if not content:
        return
    # Codex may emit the same human-visible text through both `event_msg` and
    # `response_item`; keep one copy so downstream session blobs stay coherent.
    fingerprint = (role, content, timestamp)
    if fingerprint in seen_messages:
        return
    seen_messages.add(fingerprint)
    messages.append(
        {
            "role": role,
            "content": content,
            "timestamp": timestamp,
        }
    )


def _build_tool_call(payload: dict, timestamp: str) -> ToolCall | None:
    call_id = payload.get("call_id")
    name = payload.get("name")
    if not call_id or not name:
        return None

    raw_input = payload.get("arguments", payload.get("input", ""))
    tool_input = _parse_tool_input(raw_input)
    return ToolCall(
        tool_use_id=call_id,
        tool_name=name,
        tool_input=tool_input,
        tool_result=None,
        timestamp=timestamp,
    )


def _parse_tool_input(arguments: object) -> dict:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    text = arguments.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"input": arguments}
    return parsed if isinstance(parsed, dict) else {"input": parsed}


def _parse_apply_patch(patch_text: str, timestamp: str = "") -> list[FileDiff]:
    """Parse one Codex ``apply_patch`` input string into FileDiff objects."""
    diffs: list[FileDiff] = []
    files: list[tuple[str, str, list[str]]] = []
    cur_path: str | None = None
    cur_op: str | None = None
    lines: list[str] = []

    directives = (
        ("*** Update File: ", "update"),
        ("*** Add File: ", "add"),
        ("*** Delete File: ", "delete"),
    )

    for raw in patch_text.split("\n"):
        if raw in ("*** Begin Patch", "*** End Patch", ""):
            continue

        matched = False
        for prefix, op in directives:
            if raw.startswith(prefix):
                if cur_path is not None:
                    files.append((cur_path, cur_op or "update", lines))
                cur_path = raw[len(prefix) :]
                cur_op = op
                lines = []
                matched = True
                break
        if matched or raw == "@@":
            continue
        lines.append(raw)

    if cur_path is not None:
        files.append((cur_path, cur_op or "update", lines))

    for path, op, content in files:
        unified_diff = f"--- a/{path}\n+++ b/{path}\n@@ -1,1 +1,1 @@\n" + "\n".join(content)
        if op == "add":
            after = "\n".join(line[1:] for line in content if line.startswith("+"))
            diffs.append(
                FileDiff(
                    path=path,
                    before=None,
                    after=after,
                    unified_diff=unified_diff,
                    timestamp=timestamp,
                )
            )
        elif op == "delete":
            diffs.append(
                FileDiff(
                    path=path,
                    before=None,
                    after=None,
                    unified_diff=unified_diff,
                    timestamp=timestamp,
                )
            )
        else:
            before_parts: list[str] = []
            after_parts: list[str] = []
            for line in content:
                if line.startswith("+"):
                    after_parts.append(line[1:])
                elif line.startswith("-"):
                    before_parts.append(line[1:])
                elif line.startswith(" "):
                    before_parts.append(line[1:])
                    after_parts.append(line[1:])
            diffs.append(
                FileDiff(
                    path=path,
                    before="\n".join(before_parts),
                    after="\n".join(after_parts),
                    unified_diff=unified_diff,
                    timestamp=timestamp,
                )
            )

    return diffs
