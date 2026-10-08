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

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from rclm._models import FileDiff, ToolCall
from rclm.hooks._analytics import estimate_tokens
from rclm.hooks.transcript_io import read_jsonl

logger = logging.getLogger(__name__)

_MODEL_SETTING_RE = re.compile(r"Model Selection` from None to (.+?)\.\s*(?:No need|$)")
_DIR_METADATA_RE = re.compile(r"@\[[^\]]+\] is a \[Directory\]:\s*\n([^\n]+)")


@dataclass
class AntigravityTranscriptData:
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    file_diffs: list[FileDiff] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cwd: str = ""
    model: str = ""


def _coerce_str(val: object) -> str:
    if val is None:
        return ""
    if not isinstance(val, str):
        return str(val)
    s = val.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        try:
            decoded = json.loads(s)
            if isinstance(decoded, str):
                return decoded
        except Exception:
            pass
        return s[1:-1]
    return val


def extract_antigravity_file_diffs(tool_calls: list[ToolCall]) -> list[FileDiff]:
    """Extract FileDiff objects from write_to_file and replace_file_content tool calls."""
    diffs: list[FileDiff] = []
    for tc in tool_calls:
        name = tc.tool_name
        inp = tc.tool_input if isinstance(tc.tool_input, dict) else {}
        if name == "write_to_file":
            file_path = _coerce_str(inp.get("TargetFile", ""))
            content = _coerce_str(inp.get("CodeContent", ""))
            if file_path:
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
                        timestamp=tc.timestamp,
                    )
                )
        elif name == "replace_file_content":
            file_path = _coerce_str(inp.get("TargetFile", ""))
            old = _coerce_str(inp.get("TargetContent", ""))
            new = _coerce_str(inp.get("ReplacementContent", ""))
            if file_path:
                unified = "".join(
                    difflib.unified_diff(
                        old.splitlines(keepends=True),
                        new.splitlines(keepends=True),
                        fromfile=f"a/{file_path}",
                        tofile=f"b/{file_path}",
                    )
                )
                diffs.append(
                    FileDiff(
                        path=file_path,
                        before=old,
                        after=new,
                        unified_diff=unified,
                        timestamp=tc.timestamp,
                    )
                )
    return diffs


def parse_transcript(transcript_path: str | None) -> AntigravityTranscriptData:
    """Parse an Antigravity transcript.jsonl file.

    Returns empty data if transcript_path is None or missing. Skips malformed
    JSON lines. Reads transcript_full.jsonl when available to avoid truncated tool args.
    """
    if not transcript_path:
        return AntigravityTranscriptData(warnings=["transcript_path_missing"])

    target_path = transcript_path
    path = Path(transcript_path)
    if path.name == "transcript.jsonl":
        full_candidate = path.with_name("transcript_full.jsonl")
        if full_candidate.is_file() and not full_candidate.is_symlink():
            target_path = str(full_candidate)

    entries, warnings = read_jsonl(target_path, logger=logger)
    if not entries and target_path != transcript_path:
        entries, warnings = read_jsonl(transcript_path, logger=logger)

    data = _extract(entries)
    data.warnings = warnings
    data.file_diffs = extract_antigravity_file_diffs(data.tool_calls)
    return data


def _role_for(entry: dict) -> str:
    source = entry.get("source")
    if source == "USER_EXPLICIT":
        return "user"
    if source == "SYSTEM":
        return "system"
    return "assistant"


def _is_tool_result_entry(entry: dict | None) -> bool:
    """Return True if an entry represents a genuine tool result.

    In Antigravity transcripts, tool results have source == 'MODEL' and are
    neither user inputs nor planner responses. USER_INPUT, SYSTEM, and
    PLANNER_RESPONSE steps following a tool call must never be swallowed.
    """
    if not isinstance(entry, dict) or entry.get("tool_calls"):
        return False
    if entry.get("source") in {"USER_EXPLICIT", "SYSTEM"}:
        return False
    if entry.get("type") in {
        "USER_INPUT",
        "PLANNER_RESPONSE",
        "CHECKPOINT",
        "SYSTEM_MESSAGE",
        "ERROR_MESSAGE",
    }:
        return False
    return entry.get("source") == "MODEL"


def _extract(entries: list[dict]) -> AntigravityTranscriptData:
    data = AntigravityTranscriptData()
    consumed_as_result: set[int] = set()
    cwd = ""
    model = ""

    for i, entry in enumerate(entries):
        if i in consumed_as_result:
            continue

        timestamp = entry.get("created_at", "")
        content = entry.get("content")
        thinking = entry.get("thinking")
        raw_calls = entry.get("tool_calls")

        # Discover model and directory metadata if not yet found
        if isinstance(content, str):
            if not model:
                m_model = _MODEL_SETTING_RE.search(content)
                if m_model:
                    model = m_model.group(1).strip()
            if not cwd:
                m_dir = _DIR_METADATA_RE.search(content)
                if m_dir:
                    cwd = m_dir.group(1).strip()

        if isinstance(raw_calls, list) and raw_calls:
            message: dict = {"role": "assistant", "timestamp": timestamp}
            if content is not None:
                message["content"] = content
            if thinking:
                message["thinking"] = thinking
            data.messages.append(message)

            next_entry = entries[i + 1] if i + 1 < len(entries) else None
            result_content = None
            result_timestamp = timestamp
            if _is_tool_result_entry(next_entry):
                result_content = next_entry.get("content")
                result_timestamp = next_entry.get("created_at", timestamp)
                consumed_as_result.add(i + 1)

            for idx, call in enumerate(raw_calls):
                if not isinstance(call, dict):
                    continue
                args = call.get("args")
                tool_input = args if isinstance(args, dict) else {}
                if not cwd and isinstance(tool_input, dict):
                    cwd_arg = tool_input.get("Cwd")
                    if isinstance(cwd_arg, str) and cwd_arg.strip():
                        cwd = cwd_arg.strip("\"'")
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

    data.cwd = cwd
    data.model = model
    return data
