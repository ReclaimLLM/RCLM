"""Core replay loop: reproduce the shipped mechanisms over captured tool calls.

Mirrors the real hook precedence exactly (see e.g. claude_handler.py's
PostToolUse: range_cache claims Read/shell-read calls first; exec_compaction
then only runs for shell calls the range cache didn't claim; hash_dedupe then
only runs for calls neither of the above claimed):

    range_cache (H1)  ->  exec_compaction (shell compaction)  ->  hash_dedupe

Every mechanism function used here is imported from the shipped modules —
never reimplemented — per docs/work_context/PRD_Replay_MCP.md §8.

Denominator ("text tool-result tokens", PRD §8 step 5) is broader than what
the mechanisms can reach: it is every non-image tool result's text, whether
or not any mechanism could touch it (Report 1's total spans all captured
tool calls, not just shell-tool ones). Where the shipped
`extract_text_envelope` finds one unambiguous text field, that field is what
gets tokenized and is also what the mechanisms operate on. Where it can't
(structured results with no single text field, or explicit error results),
replay falls back to a full serialization of the captured result — still
counted, but never mechanism-eligible.

`is_reachable` marks whether some selected mechanism actually looked at a
call's tool type at all (regardless of whether it found anything to reduce)
— it is set per call by whichever `_try_*` function produced the record, not
inferred after the fact from the tool name, so a Read call H1 examined but
didn't cache-hit on is correctly "reachable", not "uncoverable".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

from rclm.hooks import dedupe, image_lifecycle
from rclm.hooks.read_cache import process_read
from rclm.hooks.tool_result_transform import compact_tool_result, extract_text_envelope
from rclm.replay.read_request import build_read_request
from rclm.replay.tokenizer import count_tokens

Mechanism = Literal["range_cache", "shell_compaction", "hash_dedupe"]
ALL_MECHANISMS: tuple[Mechanism, ...] = ("range_cache", "shell_compaction", "hash_dedupe")

_SHELL_TOOL_NAMES = frozenset({"bash", "exec", "exec_command", "shell"})
_READ_ATTEMPT_TOOL_NAMES = frozenset({"read", "bash", "exec", "exec_command", "shell"})
_IMAGE_SNIFF_CHARS = 4096

Classification = Literal["shaped", "uncovered", "unresolvable", "image"]


@dataclass(frozen=True)
class ToolCallReplay:
    index: int
    tool_name: str
    classification: Classification
    mechanism: str | None
    original_tokens: int
    compressed_tokens: int
    is_reachable: bool


@dataclass
class ReplayResult:
    calls: list[ToolCallReplay] = field(default_factory=list)

    @property
    def text_result_tokens(self) -> int:
        return sum(c.original_tokens for c in self.calls if c.classification != "image")

    @property
    def tokens_removed(self) -> int:
        return sum(
            max(0, c.original_tokens - c.compressed_tokens)
            for c in self.calls
            if c.classification == "shaped"
        )

    @property
    def unresolvable_calls(self) -> list[ToolCallReplay]:
        return [c for c in self.calls if c.classification == "unresolvable"]

    @property
    def image_calls(self) -> list[ToolCallReplay]:
        return [c for c in self.calls if c.classification == "image"]

    @property
    def shaped_calls(self) -> list[ToolCallReplay]:
        return [c for c in self.calls if c.classification == "shaped"]

    def coverage_by_class(self) -> dict[str, int]:
        """Tokens grouped by reach class, matching PRD §9's coverage.by_class.

        `unresolvable` and `image` calls are excluded here: we don't know
        their reduction potential either way, so they're reported separately
        rather than folded into a reach bucket.
        """
        buckets = {"shaped": 0, "uncovered_shell": 0, "uncoverable": 0}
        for call in self.calls:
            if call.classification in ("image", "unresolvable"):
                continue
            if call.classification == "shaped":
                buckets["shaped"] += call.original_tokens
            elif call.is_reachable:
                buckets["uncovered_shell"] += call.original_tokens
            else:
                buckets["uncoverable"] += call.original_tokens
        return buckets

    def pct_of_result_chars_reachable(self) -> float:
        """Share of (non-image, resolvable) text tokens the transform core can
        reach at all — shaped or uncovered_shell — regardless of whether a
        reduction actually happened."""
        by_class = self.coverage_by_class()
        reachable = by_class["shaped"] + by_class["uncovered_shell"]
        total = reachable + by_class["uncoverable"]
        return round(100 * reachable / total, 2) if total else 0.0


def _full_text(tool_response: object) -> str:
    """Broad text representation of a captured result, for the denominator
    only. Mechanisms never operate on this — they use the narrower
    `extract_text_envelope` field, matching what the shipped transforms see."""
    if tool_response is None:
        return ""
    if isinstance(tool_response, str):
        return tool_response
    try:
        return json.dumps(tool_response, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(tool_response)


def _looks_like_encoded_image(sample: str) -> bool:
    return "data:image/" in sample[:_IMAGE_SNIFF_CHARS].lower()


def _is_image(tool_response: object) -> bool:
    """Broader than `image_lifecycle.find_image`: also catches shapes it
    doesn't recognize (e.g. `{"type": "input_image", "image_url": "data:..."}`
    lists), matching Report 1's exclusion of "recognized image payloads" from
    the text denominator regardless of the exact wire shape."""
    if image_lifecycle.find_image(tool_response) is not None:
        return True
    if isinstance(tool_response, str):
        return _looks_like_encoded_image(tool_response)
    if isinstance(tool_response, (dict, list)):
        try:
            sample = json.dumps(tool_response, ensure_ascii=False)
        except (TypeError, ValueError):
            return False
        return _looks_like_encoded_image(sample)
    return False


def _is_shell_family(tool_name: str) -> bool:
    return tool_name.lower() in _SHELL_TOOL_NAMES


def _try_range_cache(
    index: int,
    tool_name: str,
    tool_input: object,
    original_text: str,
    read_state: dict,
    turn: int,
) -> tuple[ToolCallReplay | None, dict]:
    """Return (record_or_None, updated_read_state). None means H1 doesn't
    claim this call — caller falls through to the next mechanism.

    Only called when `extract_text_envelope` already succeeded, so
    `original_text` is always real captured text here.
    """
    if tool_name.lower() not in _READ_ATTEMPT_TOOL_NAMES or not isinstance(tool_input, dict):
        return None, read_state
    built = build_read_request(tool_name, tool_input, original_text)
    if built is None:
        # A Read-shaped call H1 can't confidently parse a range for is
        # unresolvable — we cannot tell whether H1 would have reduced it.
        # A Bash call that isn't a recognizable file read is simply not
        # claimed by H1 and falls through to shell compaction / dedupe.
        if tool_name.lower() == "read":
            return (
                ToolCallReplay(
                    index=index,
                    tool_name=tool_name,
                    classification="unresolvable",
                    mechanism=None,
                    original_tokens=0,
                    compressed_tokens=0,
                    is_reachable=True,
                ),
                read_state,
            )
        return None, read_state

    request, trailing = built
    block = original_text[: len(original_text) - len(trailing)] if trailing else original_text
    decision = process_read(request, block, read_state, turn=turn)
    read_state = decision.state
    original_tokens = count_tokens(original_text)
    if decision.cache_hit and decision.replacement is not None:
        compressed_text = decision.replacement + trailing
        return (
            ToolCallReplay(
                index=index,
                tool_name=tool_name,
                classification="shaped",
                mechanism="range_cache",
                original_tokens=original_tokens,
                compressed_tokens=count_tokens(compressed_text),
                is_reachable=True,
            ),
            read_state,
        )
    return (
        ToolCallReplay(
            index=index,
            tool_name=tool_name,
            classification="uncovered",
            mechanism=None,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            is_reachable=True,
        ),
        read_state,
    )


def _try_shell_compaction(
    index: int,
    tool_name: str,
    tool_input: object,
    tool_response: object,
    original_text: str,
) -> ToolCallReplay | None:
    if not _is_shell_family(tool_name):
        return None
    decision = compact_tool_result(tool_name, tool_input, tool_response)
    original_tokens = count_tokens(original_text)
    if decision is not None:
        return ToolCallReplay(
            index=index,
            tool_name=tool_name,
            classification="shaped",
            mechanism=decision.mechanism,
            original_tokens=original_tokens,
            compressed_tokens=count_tokens(decision.compressed_text),
            is_reachable=True,
        )
    return ToolCallReplay(
        index=index,
        tool_name=tool_name,
        classification="uncovered",
        mechanism=None,
        original_tokens=original_tokens,
        compressed_tokens=original_tokens,
        is_reachable=True,
    )


def _try_dedupe(
    index: int,
    tool_name: str,
    original_text: str,
    dedupe_state: dict,
    turn: int,
) -> tuple[ToolCallReplay, dict]:
    replacement, dedupe_state, match = dedupe.maybe_dedupe(
        original_text,
        dedupe_state,
        tool_name=tool_name,
        turn=turn,
    )
    original_tokens = count_tokens(original_text)
    if replacement and match:
        return (
            ToolCallReplay(
                index=index,
                tool_name=tool_name,
                classification="shaped",
                mechanism="hash_dedupe",
                original_tokens=original_tokens,
                compressed_tokens=count_tokens(replacement),
                is_reachable=True,
            ),
            dedupe_state,
        )
    return (
        ToolCallReplay(
            index=index,
            tool_name=tool_name,
            classification="uncovered",
            mechanism=None,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            is_reachable=True,
        ),
        dedupe_state,
    )


def _uncovered_from_full_text(index: int, tool_name: str, tool_response: object) -> ToolCallReplay:
    tokens = count_tokens(_full_text(tool_response))
    return ToolCallReplay(
        index=index,
        tool_name=tool_name,
        classification="uncovered",
        mechanism=None,
        original_tokens=tokens,
        compressed_tokens=tokens,
        is_reachable=False,
    )


def replay_blob(blob: dict, mechanisms: tuple[Mechanism, ...] = ALL_MECHANISMS) -> ReplayResult:
    """Replay one session blob's tool_calls, in captured order.

    Deterministic and offline: no disk, no network, no wall-clock in the
    accounting. `mechanisms` controls which of range_cache / shell_compaction
    / hash_dedupe are attempted — pass ("shell_compaction",) alone to
    reproduce Report 1, whose replay only covered that mechanism.
    """
    result = ReplayResult()
    read_state: dict = {}
    dedupe_state: dict = {}

    for index, tool_call in enumerate(blob.get("tool_calls") or []):
        tool_name = tool_call.get("tool_name") or ""
        tool_input = tool_call.get("tool_input")
        tool_response = tool_call.get("tool_result")
        turn = index + 1  # monotonic proxy for turn order; see module docstring

        if _is_image(tool_response):
            result.calls.append(
                ToolCallReplay(
                    index=index,
                    tool_name=tool_name,
                    classification="image",
                    mechanism=None,
                    original_tokens=0,
                    compressed_tokens=0,
                    is_reachable=False,
                )
            )
            continue

        envelope = extract_text_envelope(tool_response)
        if envelope is None:
            result.calls.append(_uncovered_from_full_text(index, tool_name, tool_response))
            continue
        original_text = envelope.text

        record: ToolCallReplay | None = None
        range_claimed = False
        if "range_cache" in mechanisms:
            record, read_state = _try_range_cache(
                index, tool_name, tool_input, original_text, read_state, turn
            )
            range_claimed = record is not None

        if record is None and "shell_compaction" in mechanisms:
            record = _try_shell_compaction(
                index, tool_name, tool_input, tool_response, original_text
            )

        if record is None and "hash_dedupe" in mechanisms and not range_claimed:
            record, dedupe_state = _try_dedupe(index, tool_name, original_text, dedupe_state, turn)

        if record is None:
            tokens = count_tokens(original_text)
            record = ToolCallReplay(
                index=index,
                tool_name=tool_name,
                classification="uncovered",
                mechanism=None,
                original_tokens=tokens,
                compressed_tokens=tokens,
                is_reachable=False,
            )
        result.calls.append(record)

    return result


__all__ = [
    "ALL_MECHANISMS",
    "Mechanism",
    "ReplayResult",
    "ToolCallReplay",
    "replay_blob",
]
