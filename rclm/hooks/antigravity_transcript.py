"""Parse Antigravity's transcript.jsonl into structured data.

Each line is a JSON object with the shape:
  {"step_index": N, "source": "USER_EXPLICIT"|"SYSTEM"|"MODEL",
   "type": "USER_INPUT"|"PLANNER_RESPONSE"|"VIEW_FILE"|"LIST_DIRECTORY"|...,
   "status": "DONE", "created_at": "...", "content": "...", "thinking": "...",
   "tool_calls": [{"name": "...", "args": {...}}]}

Tool calls have no id linking them to their result. In every observed sample a
tool-call entry is immediately followed by exactly one result entry, so results
are paired by adjacency: the line right after a tool_calls entry is treated as
that call's result. If a single entry ever contains more than one tool call
(not observed so far -- Antigravity appears to emit one call per step), only
the first is paired with the following line; the rest are captured with
tool_result=None rather than guessing which result belongs to which call.
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
class AntigravityTranscriptData:
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)


def parse_transcript(transcript_path: str | None) -> AntigravityTranscriptData:
    """Parse an Antigravity transcript.jsonl file.

    Returns empty data if transcript_path is None or missing. Skips malformed
    JSON lines.
    """
    if not transcript_path:
        return AntigravityTranscriptData()

    path = Path(transcript_path)
    if not path.exists():
        logger.warning("antigravity_transcript: file not found: %s", transcript_path)
        return AntigravityTranscriptData()

    entries: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "antigravity_transcript: malformed JSON line in %s, skipping",
                    transcript_path,
                )
                continue
            if isinstance(parsed, dict):
                entries.append(parsed)

    return _extract(entries)


def _role_for(entry: dict) -> str:
    source = entry.get("source")
    if source == "USER_EXPLICIT":
        return "user"
    if source == "SYSTEM":
        return "system"
    return "assistant"


def _extract(entries: list[dict]) -> AntigravityTranscriptData:
    data = AntigravityTranscriptData()
    consumed_as_result: set[int] = set()

    for i, entry in enumerate(entries):
        if i in consumed_as_result:
            continue

        timestamp = entry.get("created_at", "")
        content = entry.get("content")
        thinking = entry.get("thinking")
        raw_calls = entry.get("tool_calls")

        if isinstance(raw_calls, list) and raw_calls:
            message: dict = {"role": "assistant", "timestamp": timestamp}
            if content is not None:
                message["content"] = content
            if thinking:
                message["thinking"] = thinking
            data.messages.append(message)

            next_entry = entries[i + 1] if i + 1 < len(entries) else None
            next_is_own_step = isinstance(next_entry, dict) and next_entry.get("tool_calls")
            result_content = None
            result_timestamp = timestamp
            if isinstance(next_entry, dict) and not next_is_own_step:
                result_content = next_entry.get("content")
                result_timestamp = next_entry.get("created_at", timestamp)
                consumed_as_result.add(i + 1)

            for idx, call in enumerate(raw_calls):
                if not isinstance(call, dict):
                    continue
                args = call.get("args")
                tool_input = args if isinstance(args, dict) else {}
                is_first = idx == 0
                data.tool_calls.append(
                    ToolCall(
                        tool_use_id=f"{entry.get('step_index', i)}:{idx}",
                        tool_name=call.get("name", ""),
                        tool_input=tool_input,
                        tool_result=result_content if is_first else None,
                        timestamp=result_timestamp if is_first else timestamp,
                        input_token_estimate=estimate_tokens(tool_input),
                        output_token_estimate=estimate_tokens(result_content) if is_first else None,
                    )
                )
            continue

        # Plain conversational entry with no tool call of its own (user input,
        # a text-only planner response, a checkpoint summary, ...). Entries
        # with neither content nor thinking (e.g. CONVERSATION_HISTORY marker
        # lines) carry no capturable data and are skipped.
        if content is None and not thinking:
            continue
        message = {"role": _role_for(entry), "timestamp": timestamp}
        if content is not None:
            message["content"] = content
        if thinking:
            message["thinking"] = thinking
        data.messages.append(message)

    return data
