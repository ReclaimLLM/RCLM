"""Provider-neutral, fail-open tool-result text compaction.

Provider handlers own their wire contracts. This module owns only the shared
decision: identify one safe textual payload, apply the existing compression
engine, and return both structured and text renderings of the replacement.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TypeAlias

from rclm.compress.runner import apply_filter
from rclm.hooks import image_lifecycle
from rclm.hooks._analytics import estimate_tokens, mechanism_saving_event
from rclm.hooks.compress import is_compressible_command

_PathPart: TypeAlias = str | int
_MAX_JSON_PARSE_CHARS = 2_000_000
_MAX_CONTENT_BLOCKS = 64
_SHELL_TOOL_NAMES = frozenset({"bash", "exec", "exec_command", "shell"})


@dataclass(frozen=True)
class TransformDecision:
    """A model-visible replacement produced by the shared compression core."""

    original_text: str
    compressed_text: str
    mechanism: str
    wire_replacement: object
    structured_replacement: object
    model_text: str

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
    )


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
        tool_name.lower() not in _SHELL_TOOL_NAMES
        or not isinstance(tool_input, dict)
        or _is_error_result(tool_response)
        or _has_oversized_content_list(tool_response)
    ):
        return None
    if image_lifecycle.find_image(tool_response) is not None:
        return None
    if isinstance(tool_response, str) and _looks_like_encoded_image(tool_response):
        return None

    command = tool_input.get("command") or tool_input.get("cmd")
    if not isinstance(command, str):
        return None
    shell = tool_input.get("shell")
    if not isinstance(shell, str) or not shell.strip():
        shell = "posix" if os.name == "posix" else os.name
    if not is_compressible_command(command, shell=shell):
        return None

    envelope = extract_text_envelope(tool_response)
    if envelope is None:
        return None
    if _looks_like_encoded_image(envelope.text):
        return None

    exit_code = _exit_code(tool_response)
    if exit_code not in (None, 0):
        return None
    result = apply_filter(command, envelope.text, "", exit_code=exit_code or 0)
    if (
        result.mechanism is None
        or result.text == envelope.text
        or len(result.text) >= len(envelope.text)
    ):
        return None

    return decision_from_replacement(
        envelope,
        result.text,
        mechanism=result.mechanism,
    )


def analytics_events(
    decision: TransformDecision,
    *,
    tool_use_id: str | None,
    applied: bool,
    **extra: object,
) -> tuple[dict, dict]:
    """Build the standard mechanism and per-tool telemetry for a decision."""
    raw_tokens = estimate_tokens(decision.original_text)
    compressed_tokens = estimate_tokens(decision.compressed_text)
    saved = max(0, raw_tokens - compressed_tokens)
    return (
        mechanism_saving_event(
            decision.mechanism,
            applied=applied,
            tokens_saved_estimate=saved,
            measurement_kind="measured",
            raw_token_estimate=raw_tokens,
            compressed_token_estimate=compressed_tokens,
        ),
        {
            "event_type": "ToolTransformation",
            "tool_use_id": tool_use_id,
            "was_compressed": True,
            "compression_strategy": decision.mechanism,
            "raw_token_estimate": raw_tokens,
            "compressed_token_estimate": compressed_tokens,
            "tokens_saved_estimate": saved,
            "raw_chars": decision.raw_chars,
            "compressed_chars": decision.compressed_chars,
            "token_estimator": "chars_div_4_v1",
            "compression_ratio": decision.compressed_chars / max(1, decision.raw_chars),
            "measurement_kind": "measured",
            "applied": applied,
            **extra,
        },
    )


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


def _find_text_path(value: object) -> tuple[tuple[_PathPart, ...], str] | None:
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
