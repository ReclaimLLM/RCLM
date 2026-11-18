"""Elide read-back of text the model itself just wrote.

After an Edit/Write the model frequently re-reads the file to verify it. Every
line of its own ``new_string``/``content`` is already in context, so a read
result that repeats a contiguous run of those lines can carry a one-line
reference instead. Matching is content-addressed (exact line equality, in
order), so it needs neither the file hash nor line-number bookkeeping: a line
that is identical to text already delivered is redundant wherever it lands.

Fail-open: anything ambiguous returns the original text unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STATE_VERSION = 1
MAX_TRACKED_FILES = 64
MAX_BLOCKS_PER_FILE = 8
MAX_AUTHORED_LINES = 2_000
# Elide only runs long enough to pay for the marker and to be unambiguous.
MIN_RUN_LINES = 6
MIN_RUN_NON_BLANK = 4

_NUMBERED_PREFIX = re.compile(r"^\s*\d+(?:→|\t|: ?)(.*)$", re.DOTALL)
_WRITE_TEXT_KEYS = (
    "content",
    "contents",
    "CodeContent",
    "new_string",
    "ReplacementContent",
    "new_str",
)
_PATH_KEYS = ("file_path", "path", "Path", "AbsolutePath", "TargetFile")


@dataclass(frozen=True)
class AuthoredDecision:
    state: dict
    replacement: str | None = None
    elided_lines: int = 0


def authored_text(tool_input: object) -> tuple[str, str] | None:
    """Return (path, text) a successful edit/write call put into context."""
    if not isinstance(tool_input, dict):
        return None
    path = next((tool_input[k] for k in _PATH_KEYS if isinstance(tool_input.get(k), str)), None)
    text = next(
        (tool_input[k] for k in _WRITE_TEXT_KEYS if isinstance(tool_input.get(k), str)), None
    )
    if not path or not text:
        return None
    return path, text


def _normalise(state: dict | None) -> dict:
    if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
        return {"version": STATE_VERSION, "files": {}}
    if not isinstance(state.get("files"), dict):
        state["files"] = {}
    return state


def record_authored(state: dict | None, path: str, text: str, *, turn: int) -> dict:
    """Remember the lines the model wrote to ``path`` at ``turn``."""
    state = _normalise(state)
    lines = text.splitlines()
    if not lines or len(lines) > MAX_AUTHORED_LINES:
        return state
    blocks = state["files"].setdefault(path, [])
    blocks.append({"turn": turn, "lines": lines})
    del blocks[:-MAX_BLOCKS_PER_FILE]
    while len(state["files"]) > MAX_TRACKED_FILES:
        oldest = min(state["files"], key=lambda key: state["files"][key][-1]["turn"])
        state["files"].pop(oldest)
    return state


def elide_authored_runs(
    state: dict | None,
    path: str,
    block: str,
    *,
    numbered: bool,
    turn: int,
) -> AuthoredDecision:
    """Replace runs of ``block`` that equal text the model wrote to ``path``.

    ``numbered`` says each line carries a ``N→``/``N<TAB>``/``N: `` prefix that
    must be ignored when comparing and kept when re-emitting.
    """
    state = _normalise(state)
    blocks = state["files"].get(path, [])
    if not blocks:
        return AuthoredDecision(state)
    raw = block.splitlines(keepends=True)
    if numbered:
        texts: list[str | None] = []
        for line in raw:
            match = _NUMBERED_PREFIX.match(line.rstrip("\r\n"))
            texts.append(match.group(1) if match else None)
    else:
        texts = [line.rstrip("\r\n") for line in raw]

    index: list[dict[str, list[int]]] = []
    for authored in blocks:
        positions: dict[str, list[int]] = {}
        for pos, text in enumerate(authored["lines"]):
            positions.setdefault(text, []).append(pos)
        index.append(positions)

    out: list[str] = []
    i = elided = 0
    while i < len(raw):
        best_len, best_turn = 0, 0
        head = texts[i]
        if head is not None:
            for authored, positions in zip(blocks, index, strict=True):
                for pos in positions.get(head, ()):
                    n = 0
                    while (
                        i + n < len(raw)
                        and pos + n < len(authored["lines"])
                        and texts[i + n] == authored["lines"][pos + n]
                    ):
                        n += 1
                    if n > best_len:
                        best_len, best_turn = n, authored["turn"]
        run = [t for t in texts[i : i + best_len] if t is not None]
        if best_len >= MIN_RUN_LINES and sum(1 for t in run if t.strip()) >= MIN_RUN_NON_BLANK:
            first = _line_number(raw[i]) if numbered else None
            last = _line_number(raw[i + best_len - 1]) if numbered else None
            where = f"Lines {first}-{last} of " if first and last else f"{best_len} lines of "
            out.append(
                f"[RCLM] {where}{path} are identical to the text you wrote at turn "
                f"{best_turn} ({best_len} lines, not repeated).\n"
            )
            elided += best_len
            i += best_len
        else:
            out.append(raw[i])
            i += 1
    if not elided:
        return AuthoredDecision(state)
    replacement = "".join(out)
    if len(replacement) >= len(block):
        return AuthoredDecision(state)
    return AuthoredDecision(state, replacement, elided)


def _line_number(line: str) -> int | None:
    match = re.match(r"^\s*(\d+)", line)
    return int(match.group(1)) if match else None


__all__ = [
    "AuthoredDecision",
    "authored_text",
    "elide_authored_runs",
    "record_authored",
]
