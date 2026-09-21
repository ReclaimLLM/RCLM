"""Provider-neutral, fail-open tool-result text compaction.

Provider handlers own their wire contracts. This module owns only the shared
decision: identify one safe textual payload, apply the existing compression
engine, and return both structured and text renderings of the replacement.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import TypeAlias

from rclm.compress.filters.lossless import MECHANISM as LOSSLESS_SEARCH_PATH_MECHANISM
from rclm.compress.filters.lossless import compact_search_and_paths
from rclm.compress.filters.shell import filter_generic
from rclm.compress.runner import apply_filter
from rclm.hooks import image_lifecycle
from rclm.hooks._analytics import estimate_tokens, mechanism_saving_event
from rclm.hooks.compress import is_compressible_command
from rclm.hooks.tool_semantics import command_text, normalized_tool_name, tool_family

_PathPart: TypeAlias = str | int
_MAX_JSON_PARSE_CHARS = 2_000_000
_MAX_CONTENT_BLOCKS = 64
_SQL_READ_PREFIX = re.compile(r"^\s*(?:select|explain|show)\b", re.IGNORECASE)
_SQL_UNTRUSTED_BLOCK = re.compile(
    r"<(?P<tag>untrusted-data-[A-Za-z0-9-]+)>\n(?P<data>.*?)\n</(?P=tag)>",
    re.DOTALL,
)
_SQL_HEAD_ROWS = 12
_SQL_TAIL_ROWS = 6
_SQL_MIN_ROWS = 30
_ANTIGRAVITY_COMMAND_EXIT = re.compile(r"The command exited with code (?P<code>\d+)\.")
_ANTIGRAVITY_OUTPUT = re.compile(r"\nOutput:\n(?P<output>.*)\Z", re.DOTALL)
_DIFF_BLOCK = re.compile(
    r"\[diff_block_start\]\n(?P<diff>.*?)\n\[diff_block_end\]",
    re.DOTALL,
)
_DIFF_HUNK = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@",
    re.MULTILINE,
)


@dataclass(frozen=True)
class SavingsStep:
    """One independently attributable stage of a model-visible transform."""

    mechanism: str
    original_text: str
    compressed_text: str


@dataclass(frozen=True)
class TransformDecision:
    """A model-visible replacement produced by the shared compression core."""

    original_text: str
    compressed_text: str
    mechanism: str
    wire_replacement: object
    structured_replacement: object
    model_text: str
    savings_steps: tuple[SavingsStep, ...]

    @property
    def raw_chars(self) -> int:
        return len(self.original_text)

    @property
    def compressed_chars(self) -> int:
        return len(self.compressed_text)


@dataclass(frozen=True)
class TextEnvelope:
    value: object
    path: tuple[_PathPart, ...]
    text: str
    outer_json_string: bool = False

    def replace(self, replacement: str) -> tuple[object, object, str]:
        structured = _replace_path(self.value, self.path, replacement)
        if self.outer_json_string:
            wire: object = json.dumps(structured, separators=(",", ":"), ensure_ascii=False)
        else:
            wire = structured
        model_text = (
            structured
            if isinstance(structured, str)
            else json.dumps(structured, separators=(",", ":"), ensure_ascii=False)
        )
        return wire, structured, model_text


def decision_from_replacement(
    envelope: TextEnvelope,
    replacement: str,
    *,
    mechanism: str,
    savings_steps: tuple[SavingsStep, ...] | None = None,
) -> TransformDecision | None:
    """Build a standard decision for stateful mechanisms such as dedupe."""
    if not replacement or replacement == envelope.text or len(replacement) >= len(envelope.text):
        return None
    wire, structured, model_text = envelope.replace(replacement)
    return TransformDecision(
        original_text=envelope.text,
        compressed_text=replacement,
        mechanism=mechanism,
        wire_replacement=wire,
        structured_replacement=structured,
        model_text=model_text,
        savings_steps=savings_steps or (SavingsStep(mechanism, envelope.text, replacement),),
    )


def attach_recall_stub(
    source: object,
    decision: TransformDecision,
    stub: str,
    *,
    tool_name: str = "",
) -> TransformDecision:
    """Append a persisted exact-recall handle without changing wire shape."""
    envelope = extract_tool_text_envelope(tool_name, source)
    if envelope is None:
        return decision
    replacement = decision.compressed_text.rstrip("\n") + "\n" + stub + "\n"
    recalled = decision_from_replacement(
        envelope,
        replacement,
        mechanism=decision.mechanism,
    )
    return recalled or decision


def compact_tool_result(
    tool_name: str,
    tool_input: object,
    tool_response: object,
) -> TransformDecision | None:
    """Compact one recognized successful textual tool result.

    Unknown commands, failures, images, malformed structured output, and
    already-small results pass through unchanged.
    """
    if (
        not isinstance(tool_input, dict)
        or _is_error_result(tool_response)
        or _has_oversized_content_list(tool_response)
    ):
        return None
    if image_lifecycle.find_image(tool_response) is not None:
        return None
    if isinstance(tool_response, str) and _looks_like_encoded_image(tool_response):
        return None

    envelope = extract_tool_text_envelope(tool_name, tool_response)
    if envelope is None:
        return None
    if _looks_like_encoded_image(envelope.text):
        return None

    exit_code = _exit_code(tool_response)
    if exit_code not in (None, 0):
        return None
    wrapped_exit_code = _wrapped_command_exit_code(envelope.text)
    if wrapped_exit_code not in (None, 0):
        return None

    native_decision = _compact_known_native_tool(tool_name, tool_input, envelope.text)
    if native_decision is not None:
        native_replacement, native_mechanism = native_decision
        return decision_from_replacement(
            envelope,
            native_replacement,
            mechanism=native_mechanism,
        )

    if tool_family(tool_name) != "shell":
        return None
    command = command_text(tool_input)
    if isinstance(command, str):
        shell = tool_input.get("shell")
        if not isinstance(shell, str) or not shell.strip():
            shell = "posix" if os.name == "posix" else os.name
        if is_compressible_command(command, shell=shell):
            result = apply_filter(command, envelope.text, "", exit_code=exit_code or 0)
            if (
                result.mechanism is not None
                and result.text != envelope.text
                and len(result.text) < len(envelope.text)
            ):
                # Preserve the command-aware mechanism as the primary label,
                # then remove repeated search/path prefixes from its result.
                # The second fold is independently round-trip verified.
                folded_text = compact_search_and_paths(result.text)
                filtered_text = folded_text or result.text
                savings_steps = None
                if folded_text is not None:
                    savings_steps = (
                        SavingsStep(result.mechanism, envelope.text, result.text),
                        SavingsStep(
                            LOSSLESS_SEARCH_PATH_MECHANISM,
                            result.text,
                            folded_text,
                        ),
                    )
                return decision_from_replacement(
                    envelope,
                    filtered_text,
                    mechanism=result.mechanism,
                    savings_steps=savings_steps,
                )

    # Chained commands and absolute binaries often obscure an otherwise clear
    # grep/find result from the command parser. Use only reversible shape folds
    # as the final shell fallback; ambiguous output remains byte-for-byte intact.
    folded = compact_search_and_paths(envelope.text)
    if folded is None:
        return None
    return decision_from_replacement(
        envelope,
        folded,
        mechanism=LOSSLESS_SEARCH_PATH_MECHANISM,
    )


def analytics_events(
    decision: TransformDecision,
    *,
    tool_use_id: str | None,
    applied: bool,
    **extra: object,
) -> tuple[dict, ...]:
    """Build the standard mechanism and per-tool telemetry for a decision."""
    saving_events = []
    for step in decision.savings_steps:
        step_raw_tokens = estimate_tokens(step.original_text)
        step_compressed_tokens = estimate_tokens(step.compressed_text)
        saving_events.append(
            mechanism_saving_event(
                step.mechanism,
                applied=applied,
                tokens_saved_estimate=max(0, step_raw_tokens - step_compressed_tokens),
                measurement_kind="measured",
                raw_token_estimate=step_raw_tokens,
                compressed_token_estimate=step_compressed_tokens,
            )
        )

    raw_tokens = estimate_tokens(decision.original_text)
    compressed_tokens = estimate_tokens(decision.compressed_text)
    transformation_event = {
        "event_type": "ToolTransformation",
        "tool_use_id": tool_use_id,
        "was_compressed": True,
        "compression_strategy": decision.mechanism,
        "compression_strategies": [step.mechanism for step in decision.savings_steps],
        "raw_token_estimate": raw_tokens,
        "compressed_token_estimate": compressed_tokens,
        "tokens_saved_estimate": max(0, raw_tokens - compressed_tokens),
        "raw_chars": decision.raw_chars,
        "compressed_chars": decision.compressed_chars,
        "token_estimator": "chars_div_4_v1",
        "compression_ratio": decision.compressed_chars / max(1, decision.raw_chars),
        "measurement_kind": "measured",
        "applied": applied,
        **extra,
    }
    return (*saving_events, transformation_event)


def extract_text_envelope(value: object) -> TextEnvelope | None:
    """Return one unambiguous non-error, non-image textual result payload."""
    if (
        _is_error_result(value)
        or _has_oversized_content_list(value)
        or image_lifecycle.find_image(value) is not None
    ):
        return None
    if isinstance(value, str):
        if "\x00" in value or _looks_like_encoded_image(value):
            return None
        stripped = value.lstrip()
        if len(value) <= _MAX_JSON_PARSE_CHARS and stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None and not _is_error_result(parsed):
                nested = _find_text_path(parsed)
                if nested is not None:
                    path, text = nested
                    return TextEnvelope(parsed, path, text, outer_json_string=True)
        return TextEnvelope(value, (), value)

    if not isinstance(value, (dict, list)):
        return None
    nested = _find_text_path(value)
    if nested is None:
        return None
    path, text = nested
    return TextEnvelope(value, path, text)


def extract_tool_text_envelope(tool_name: str, value: object) -> TextEnvelope | None:
    """Extract text with narrow support for Codex's ordered exec block arrays."""
    envelope = extract_text_envelope(value)
    if envelope is not None:
        return envelope
    if tool_family(tool_name) != "shell" or not isinstance(value, list) or not value:
        return None
    if not all(
        isinstance(block, dict)
        and block.get("type") in {"text", "output_text", "input_text"}
        and isinstance(block.get("text"), str)
        for block in value
    ):
        return None
    combined = "".join(block["text"] for block in value)
    return TextEnvelope(value, (), combined) if combined else None


def _find_text_path(value: object) -> tuple[tuple[_PathPart, ...], str] | None:
    if isinstance(value, list):
        if len(value) > _MAX_CONTENT_BLOCKS:
            return None
        text_blocks = [
            (index, block.get("text"))
            for index, block in enumerate(value)
            if isinstance(block, dict)
            and block.get("type") in {"text", "output_text", "input_text"}
            and isinstance(block.get("text"), str)
        ]
        if len(text_blocks) == 1:
            index, text = text_blocks[0]
            return (index, "text"), text
        return None
    if not isinstance(value, dict):
        return None

    for key in ("stdout", "output", "content", "text"):
        candidate = value.get(key)
        if isinstance(candidate, str):
            return (key,), candidate

    file_result = value.get("file")
    if isinstance(file_result, dict) and isinstance(file_result.get("content"), str):
        return ("file", "content"), file_result["content"]

    content = value.get("content")
    if isinstance(content, list):
        if len(content) > _MAX_CONTENT_BLOCKS:
            return None
        text_blocks = [
            (index, block.get("text"))
            for index, block in enumerate(content)
            if isinstance(block, dict)
            and block.get("type") in {"text", "output_text", "input_text"}
            and isinstance(block.get("text"), str)
        ]
        # Replacing one unambiguous text block preserves the rest of the MCP
        # envelope. Multiple text blocks may have independent semantics.
        if len(text_blocks) == 1:
            index, text = text_blocks[0]
            return ("content", index, "text"), text
    return None


def _compact_known_native_tool(
    tool_name: str, tool_input: dict, text: str
) -> tuple[str, str] | None:
    """Compact only native tools with a stable, explicitly recognized shape."""
    normalized = normalized_tool_name(tool_name)
    if normalized in {"browser_snapshot", "browser_accessibility_snapshot"}:
        replacement = filter_generic(text)
        return (replacement, "H3_exec_compaction") if replacement is not None else None
    if "supabase" in tool_name.lower() and normalized == "execute_sql":
        query = tool_input.get("query")
        if isinstance(query, str) and _SQL_READ_PREFIX.match(query):
            replacement = _compact_supabase_rows(text)
            return (replacement, "H3_exec_compaction") if replacement is not None else None
    if normalized == "view_file":
        replacement = _compact_antigravity_view(text)
        return (replacement, "provider_canonicalization") if replacement is not None else None
    if normalized == "run_command":
        replacement = _compact_antigravity_command(tool_input, text)
        return (replacement, "H3_exec_compaction") if replacement is not None else None
    if normalized == "exec" and text.startswith(("Script completed\n", "Chunk ID:")):
        replacement = filter_generic(text)
        return (replacement, "provider_output_shaping") if replacement is not None else None
    if normalized in {"grep_search", "find_by_name", "list_dir"}:
        canonical = _strip_antigravity_timing(text)
        replacement = filter_generic(canonical) or (canonical if canonical != text else None)
        return (replacement, "provider_output_shaping") if replacement is not None else None
    if normalized in {"replace_file_content", "multi_replace_file_content", "write_to_file"}:
        replacement = _compact_edit_receipt(normalized, tool_input, text)
        return (replacement, "edit_receipt") if replacement is not None else None
    return None


def _strip_antigravity_timing(text: str) -> str:
    """Drop volatile capture timestamps while leaving result semantics intact."""
    lines = text.splitlines(keepends=True)
    while lines and lines[0].startswith(("Created At: ", "Completed At: ")):
        lines.pop(0)
    return "".join(lines)


def _compact_antigravity_view(text: str) -> str | None:
    """Remove verbose boilerplate from the observed ``view_file`` envelope."""
    canonical = _strip_antigravity_timing(text)
    lines = canonical.splitlines(keepends=True)
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(
            "The following code has been modified to include a line number before every line"
        ):
            continue
        if stripped.startswith("The above content does NOT show the entire file contents"):
            continue
        kept.append(line)
    replacement = "".join(kept).strip("\n") + "\n"
    return replacement if len(replacement) < len(text) else None


def _wrapped_command_exit_code(text: str) -> int | None:
    match = _ANTIGRAVITY_COMMAND_EXIT.search(text)
    return int(match.group("code")) if match is not None else None


def _compact_antigravity_command(tool_input: dict, text: str) -> str | None:
    """Compact the command output while retaining the provider's exit status."""
    exit_code = _wrapped_command_exit_code(text)
    command = command_text(tool_input)
    output_match = _ANTIGRAVITY_OUTPUT.search(text)
    if exit_code != 0 or command is None or output_match is None:
        return None
    output = output_match.group("output")
    shell = tool_input.get("shell")
    if not isinstance(shell, str) or not shell.strip():
        shell = "posix" if os.name == "posix" else os.name
    filtered = None
    if is_compressible_command(command, shell=shell):
        result = apply_filter(command, output, "", exit_code=0)
        if result.text != output and len(result.text) < len(output):
            filtered = result.text
    if filtered is None:
        filtered = filter_generic(output)
    canonical_output = filtered if filtered is not None else output
    replacement = f"The command exited with code 0.\nOutput:\n{canonical_output}"
    return replacement if len(replacement) < len(text) else None


def _compact_edit_receipt(normalized: str, tool_input: dict, text: str) -> str | None:
    """Replace a successful echoed patch with a deterministic, auditable receipt."""
    match = _DIFF_BLOCK.search(text)
    if match is None:
        return None
    diff = match.group("diff")
    target = tool_input.get("TargetFile") or tool_input.get("file_path")
    if not isinstance(target, str) or not target:
        return None
    # The observed provider response names the target before the diff.  Refuse
    # a receipt when it disagrees with the input rather than guessing.
    if normalized != "write_to_file" and target not in text[: match.start()]:
        return None
    added = sum(
        1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")
    )
    hunks = list(_DIFF_HUNK.finditer(diff))
    if not hunks and not (added or removed):
        return None
    ranges = []
    for hunk in hunks[:12]:
        start = int(hunk.group("new_start"))
        count = int(hunk.group("new_count") or 1)
        ranges.append(f"{start}-{max(start, start + count - 1)}")
    digest = sha256(text.encode("utf-8", errors="surrogateescape")).hexdigest()
    receipt = (
        f"[rclm edit receipt] tool={normalized} target={target}\n"
        f"hunks={len(hunks)} added={added} removed={removed} "
        f"new_line_ranges={','.join(ranges) or 'unknown'}\n"
        f"source_sha256={digest} (exact result retained in session capture)\n"
    )
    return receipt if len(receipt) < len(text) else None


def _compact_supabase_rows(text: str) -> str | None:
    """Sample a large Supabase read result while preserving its safety envelope.

    The adapter only accepts the observed execute_sql JSON envelope containing
    one untrusted-data block whose body is a JSON array. Mutations, malformed
    values, small results, and future wire shapes pass through unchanged.
    """
    try:
        outer = json.loads(text)
        if not isinstance(outer, dict) or not isinstance(outer.get("result"), str):
            return None
        result_text = outer["result"]
        match = _SQL_UNTRUSTED_BLOCK.search(result_text)
        if match is None:
            return None
        rows = json.loads(match.group("data"))
        if not isinstance(rows, list) or len(rows) <= _SQL_MIN_ROWS:
            return None
        omitted = len(rows) - _SQL_HEAD_ROWS - _SQL_TAIL_ROWS
        sampled = [
            *rows[:_SQL_HEAD_ROWS],
            {
                "_rclm_omitted_rows": omitted,
                "_rclm_note": "Re-run with a narrower WHERE or LIMIT/OFFSET to inspect omitted rows.",
            },
            *rows[-_SQL_TAIL_ROWS:],
        ]
        sampled_json = json.dumps(sampled, separators=(",", ":"), ensure_ascii=False)
        outer["result"] = (
            result_text[: match.start("data")] + sampled_json + result_text[match.end("data") :]
        )
        replacement = json.dumps(outer, separators=(",", ":"), ensure_ascii=False)
        return replacement if len(replacement) < len(text) else None
    except (TypeError, ValueError):
        return None


def _replace_path(value: object, path: tuple[_PathPart, ...], replacement: str) -> object:
    if not path:
        return replacement
    head, *tail = path
    if isinstance(head, str) and isinstance(value, dict):
        updated = dict(value)
        updated[head] = _replace_path(value.get(head), tuple(tail), replacement)
        return updated
    if isinstance(head, int) and isinstance(value, list) and 0 <= head < len(value):
        updated = list(value)
        updated[head] = _replace_path(value[head], tuple(tail), replacement)
        return updated
    return value


def _exit_code(value: object) -> int | None:
    if not isinstance(value, dict):
        return None
    for key in ("exit_code", "exitCode", "status_code"):
        candidate = value.get(key)
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    metadata = value.get("metadata")
    if isinstance(metadata, dict):
        return _exit_code(metadata)
    return None


def _is_error_result(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("isError") is True or value.get("is_error") is True:
        return True
    status = value.get("status")
    if isinstance(status, str) and status.lower() in {"error", "failed", "failure"}:
        return True
    return (_exit_code(value) or 0) != 0


def _looks_like_encoded_image(text: str) -> bool:
    sample = text[:4096].lower()
    return "data:image/" in sample or (
        len(text) > _MAX_JSON_PARSE_CHARS
        and '"type"' in sample
        and '"image"' in sample
        and ('"data"' in sample or "base64" in sample)
    )


def _has_oversized_content_list(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("content"), list)
        and len(value["content"]) > _MAX_CONTENT_BLOCKS
    )
