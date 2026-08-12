"""Parse Cursor agent JSONL transcripts into normalized session data."""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from rclm._models import FileDiff, ToolCall

logger = logging.getLogger(__name__)


@dataclass
class CursorTranscriptData:
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    file_diffs: list[FileDiff] = field(default_factory=list)
    model: str | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    session_id: str | None = None
    cwd: str = ""


def parse_transcript(transcript_path: str | None) -> CursorTranscriptData:
    """Parse a Cursor JSONL transcript file.

    Returns an empty CursorTranscriptData if transcript_path is None or missing.
    Skips malformed JSON lines.
    """
    if not transcript_path:
        return CursorTranscriptData()

    path = Path(transcript_path)
    if not path.exists():
        logger.warning("cursor transcript: file not found: %s", transcript_path)
        return CursorTranscriptData()

    raw_lines: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw_lines.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    "cursor transcript: malformed JSON line in %s, skipping", transcript_path
                )

    return _extract(raw_lines)


def _extract(entries: list[dict]) -> CursorTranscriptData:
    """Extract normalized session data from Cursor's transcript entries."""
    data = CursorTranscriptData()

    for entry in entries:
        entry_type = entry.get("type", "")
        timestamp = entry.get("timestamp", entry.get("time", ""))

        # Meta information
        if entry_type == "session_meta" or "conversation_id" in entry:
            if not data.session_id:
                data.session_id = entry.get("conversation_id") or entry.get("sessionId")
            if not data.cwd:
                data.cwd = entry.get("cwd", "")
            if not data.model:
                data.model = entry.get("model")

        message = entry.get("message")
        message_dict = message if isinstance(message, dict) else {}
        role = entry.get("role") or message_dict.get("role")
        content_field = (
            entry.get("content") or entry.get("text") or message_dict.get("content") or message
        )

        # Process message roles (user/assistant)
        if role in ("user", "assistant", "human", "ai", "agent") or entry_type in (
            "user",
            "assistant",
            "human",
            "ai",
            "agent",
            "message",
        ):
            actual_role = role or entry_type
            if actual_role in ("human", "user"):
                actual_role = "user"
            elif actual_role in ("ai", "agent", "assistant"):
                actual_role = "assistant"
                # Extract model if available in assistant entry
                if not data.model:
                    data.model = entry.get("model") or entry.get("model_name")

            flattened_text = ""

            # Handle nested content structure: {"content": {"content": [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]}}
            if isinstance(content_field, dict) and "content" in content_field:
                blocks = content_field["content"]
                if isinstance(blocks, list):
                    for block in blocks:
                        b_type = block.get("type")
                        if b_type == "text":
                            flattened_text += block.get("text", "")
                        elif b_type == "tool_use":
                            _process_tool_use(block, data, timestamp)
                else:
                    # Fallback for dict content that isn't the standard list
                    flattened_text = str(content_field)
            elif isinstance(content_field, list):
                for block in content_field:
                    if not isinstance(block, dict):
                        continue
                    b_type = block.get("type")
                    if b_type == "text":
                        flattened_text += block.get("text", "")
                    elif b_type == "tool_use":
                        _process_tool_use(block, data, timestamp)
            elif isinstance(content_field, str):
                flattened_text = content_field
            elif content_field:
                flattened_text = str(content_field)

            if actual_role == "user":
                flattened_text = _clean_user_text(flattened_text)

            if flattened_text and not _is_redacted_only_assistant(actual_role, flattened_text):
                data.messages.append(
                    {"role": actual_role, "content": flattened_text, "timestamp": timestamp or ""}
                )

        # Top-level tool use (if any)
        elif entry_type in ("tool", "action", "call", "tool_use"):
            _process_tool_use(entry, data, timestamp)

        # Accumulate tokens
        usage = entry.get("usage") or entry.get("tokens")
        if isinstance(usage, dict):
            in_tokens = usage.get("input_tokens") or usage.get("input") or 0
            out_tokens = usage.get("output_tokens") or usage.get("output") or 0
            if in_tokens or out_tokens:
                data.total_input_tokens = (data.total_input_tokens or 0) + in_tokens
                data.total_output_tokens = (data.total_output_tokens or 0) + out_tokens

    data.file_diffs = coalesce_file_diffs(data.file_diffs)
    return data


def _clean_user_text(text: str) -> str:
    text = re.sub(r"<timestamp>.*?</timestamp>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?user_query>", "", text)
    return text.strip()


def _is_redacted_only_assistant(role: str, text: str) -> bool:
    return role == "assistant" and text.strip() == "[REDACTED]"


def build_unified_diff(path: str, before: str | None, after: str | None) -> str:
    """Build a displayable unified diff from available file or snippet content."""
    return "".join(
        difflib.unified_diff(
            (before or "").splitlines(keepends=True),
            (after or "").splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def coalesce_file_diffs(diffs: list[FileDiff]) -> list[FileDiff]:
    """Keep one ordered diff per path while retaining every replacement hunk."""
    by_path: dict[str, FileDiff] = {}
    for diff in diffs:
        previous = by_path.get(diff.path)
        if previous is None or diff.before is None:
            # A full Write supersedes earlier partial edits for the same path.
            by_path[diff.path] = diff
            continue

        if (
            previous.before is None
            and previous.after is not None
            and diff.before
            and diff.after is not None
            and diff.before in previous.after
        ):
            # Apply a later replacement to content captured by an earlier Write.
            final_content = previous.after.replace(diff.before, diff.after, 1)
            by_path[diff.path] = FileDiff(
                path=diff.path,
                before=None,
                after=final_content,
                unified_diff=build_unified_diff(diff.path, None, final_content),
                timestamp=diff.timestamp or previous.timestamp,
            )
            continue

        by_path[diff.path] = FileDiff(
            path=diff.path,
            before=_join_fragments(previous.before, diff.before),
            after=_join_fragments(previous.after, diff.after),
            unified_diff=_join_diff_chunks(previous.unified_diff, diff.unified_diff),
            timestamp=diff.timestamp or previous.timestamp,
        )

    return list(by_path.values())


def _join_fragments(left: str | None, right: str | None) -> str | None:
    if left is None and right is None:
        return None
    return "\n".join(part for part in (left, right) if part)


def _join_diff_chunks(left: str, right: str) -> str:
    return "\n".join(chunk.rstrip("\n") for chunk in (left, right) if chunk)


def _string_input(tool_input: dict, *keys: str) -> str | None:
    for key in keys:
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    return None


def _process_tool_use(block: dict, data: CursorTranscriptData, timestamp: str | None) -> None:
    """Normalize tool_use block and add to tool_calls and file_diffs if applicable."""
    tool_name = block.get("name") or block.get("tool_name") or ""
    tool_input = block.get("input") or block.get("args") or block.get("arguments", {})
    tool_result = block.get("output") or block.get("result") or block.get("response")

    if not tool_name:
        return

    call_id = block.get("id") or block.get("call_id") or f"cursor-tool-{len(data.tool_calls)}"

    # Normalize tool input
    if not isinstance(tool_input, dict):
        tool_input = {"input": tool_input}

    data.tool_calls.append(
        ToolCall(
            tool_use_id=call_id,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_result=tool_result,
            timestamp=timestamp or "",
        )
    )

    path = _string_input(tool_input, "path", "file_path", "filepath")

    # Cursor uses `contents` in current transcripts; retain older aliases too.
    if tool_name == "Write":
        content = _string_input(tool_input, "contents", "content", "text")
        if path and content is not None:
            data.file_diffs.append(
                FileDiff(
                    path=path,
                    before=None,  # We don't have the original content in the transcript
                    after=content,
                    unified_diff=build_unified_diff(path, None, content),
                    timestamp=timestamp or "",
                )
            )
    elif tool_name in {"StrReplace", "Edit"}:
        before = _string_input(tool_input, "old_string", "old_text")
        after = _string_input(tool_input, "new_string", "new_text")
        if path and before is not None and after is not None:
            data.file_diffs.append(
                FileDiff(
                    path=path,
                    before=before,
                    after=after,
                    unified_diff=build_unified_diff(path, before, after),
                    timestamp=timestamp or "",
                )
            )
