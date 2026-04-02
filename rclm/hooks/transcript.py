"""Parse Claude Code's JSONL transcript into structured data.

Each line of the transcript is a JSON object with the shape:
  {"type": "user"|"assistant"|"tool", "message": {...}, "timestamp": "...",
   "model": "...", "usage": {"input_tokens": N, "output_tokens": N}}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from rclm._models import ToolCall
from rclm.hooks._analytics import estimate_tokens

logger = logging.getLogger(__name__)


@dataclass
class TranscriptData:
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None


def parse_transcript(transcript_path: str | None) -> TranscriptData:
    """Parse a Claude Code JSONL transcript file.

    Returns an empty TranscriptData if transcript_path is None or missing.
    Skips malformed JSON lines.
    """
    if not transcript_path:
        return TranscriptData()

    path = Path(transcript_path)
    if not path.exists():
        logger.warning("transcript: file not found: %s", transcript_path)
        return TranscriptData()

    raw_lines: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw_lines.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("transcript: malformed JSON line in %s, skipping", transcript_path)

    return _extract(raw_lines)


def _extract(entries: list[dict]) -> TranscriptData:
    data = TranscriptData()

    # Index tool_result blocks by tool_use_id so we can pair them with tool_use.
    # Format: tool_result blocks appear as user-role content items.
    tool_results: dict[str, str | dict | list | None] = {}
    for entry in entries:
        msg = entry.get("message", {})
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                tool_results[tool_use_id] = block.get("content")

    total_in: int = 0
    total_out: int = 0
    has_tokens = False

    for entry in entries:
        entry_type = entry.get("type", "")
        msg = entry.get("message", {})
        timestamp = entry.get("timestamp", "")

        if entry_type in ("user", "assistant"):
            role = msg.get("role", entry_type)
            content = msg.get("content")
            data.messages.append({"role": role, "content": content, "timestamp": timestamp})

        if entry_type == "assistant":
            # Extract model from first assistant entry.
            if data.model is None and entry.get("model"):
                data.model = entry["model"]

            # Accumulate token usage.
            usage = entry.get("usage", {})
            if usage:
                has_tokens = True
                total_in += usage.get("input_tokens", 0)
                total_out += usage.get("output_tokens", 0)

            # Extract tool_use blocks and pair with results.
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not (isinstance(block, dict) and block.get("type") == "tool_use"):
                    continue
                tool_use_id = block.get("id", "")
                tool_input = block.get("input", {})
                tool_result = tool_results.get(tool_use_id)
                data.tool_calls.append(
                    ToolCall(
                        tool_use_id=tool_use_id,
                        tool_name=block.get("name", ""),
                        tool_input=tool_input,
                        tool_result=tool_result,
                        timestamp=timestamp,
                        input_token_estimate=estimate_tokens(tool_input),
                        output_token_estimate=estimate_tokens(tool_result),
                    )
                )

    if has_tokens:
        data.total_input_tokens = total_in
        data.total_output_tokens = total_out

    return data
