"""Filters for shell commands (ls, find, generic truncation)."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterator
from io import StringIO

# CSI sequences (colors, cursor movement) and OSC sequences (terminal titles).
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

# Collapse runs of 3+ identical lines (progress bars, repeated log lines).
_MIN_REPEAT_RUN = 3

# Generic fallback shape: head+tail cap for commands with no dedicated filter.
_GENERIC_MAX_LINES = 60
_GENERIC_HEAD = 40
_GENERIC_TAIL = 20


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (colors, cursor control, terminal titles)."""
    return _ANSI_RE.sub("", text)


def collapse_repeats(lines: list[str]) -> list[str]:
    """Collapse runs of 3+ consecutive identical lines into one + a count marker."""
    result: list[str] = []
    i = 0
    while i < len(lines):
        j = i
        while j < len(lines) and lines[j] == lines[i]:
            j += 1
        run_len = j - i
        result.append(lines[i])
        if run_len >= _MIN_REPEAT_RUN:
            result.append(f"... (repeated {run_len - 1} more times)")
        else:
            result.extend(lines[i + 1 : j])
        i = j
    return result


def filter_generic(output: str) -> str | None:
    """Fallback compaction for commands with no dedicated filter.

    Collapses repeated lines, then caps to head+tail if still large. Returns
    None if the output is already small enough that no shaping is needed.
    """
    raw_line_count = 0
    collapsed_count = 0
    head: list[str] = []
    tail: deque[str] = deque(maxlen=_GENERIC_TAIL)

    for line, run_length in _iter_runs(output):
        raw_line_count += run_length
        collapsed_lines = [line]
        if run_length >= _MIN_REPEAT_RUN:
            collapsed_lines.append(f"... (repeated {run_length - 1} more times)")
        else:
            collapsed_lines.extend([line] * (run_length - 1))

        for collapsed_line in collapsed_lines:
            collapsed_count += 1
            if len(head) < _GENERIC_HEAD:
                head.append(collapsed_line)
            else:
                tail.append(collapsed_line)

    if raw_line_count <= _GENERIC_MAX_LINES:
        return None
    if collapsed_count <= _GENERIC_MAX_LINES:
        return "\n".join([*head, *tail])

    omitted = collapsed_count - _GENERIC_HEAD - _GENERIC_TAIL
    return "\n".join([*head, f"... ({omitted} lines omitted) ...", *tail])


def _iter_runs(output: str) -> Iterator[tuple[str, int]]:
    """Yield consecutive line runs without materializing the full output."""
    previous: str | None = None
    run_length = 0
    for raw_line in StringIO(output, newline=None):
        line = raw_line.rstrip("\r\n")
        if previous is None:
            previous = line
            run_length = 1
        elif line == previous:
            run_length += 1
        else:
            yield previous, run_length
            previous = line
            run_length = 1
    if previous is not None:
        yield previous, run_length


def filter_shell(command: str, output: str) -> str | None:
    """Filter shell command output. Returns compressed string or None if no filter."""
    parts = command.strip().split()
    if not parts:
        return None

    base_cmd = parts[0]
    if base_cmd in ("ls", "find"):
        return _filter_listing(output)

    return None


def _filter_listing(output: str) -> str:
    """Compress directory listings: group by directory, show counts."""
    if not output.strip():
        return "(empty)"

    lines = [line.rstrip() for line in output.splitlines() if line.strip()]

    if len(lines) <= 30:
        return output  # Small enough, keep as-is

    # Group files by parent directory
    dirs: dict[str, list[str]] = {}
    plain_files: list[str] = []

    for line in lines:
        # Skip header/summary lines from ls -l
        if line.startswith("total ") or line.startswith("d") or line.startswith("-"):
            plain_files.append(line)
            continue

        parts = line.rsplit("/", 1)
        if len(parts) == 2:
            parent, name = parts
            dirs.setdefault(parent, []).append(name)
        else:
            plain_files.append(line)

    if not dirs:
        # No directory structure found, just truncate
        return _truncate_lines(lines, 30)

    result: list[str] = []
    for parent, files in sorted(dirs.items()):
        if len(files) <= 5:
            for f in files:
                result.append(f"{parent}/{f}")
        else:
            result.append(f"{parent}/ ({len(files)} files)")
            for f in files[:3]:
                result.append(f"  {f}")
            result.append(f"  ... +{len(files) - 3} more")

    if plain_files:
        result.extend(plain_files[:10])
        if len(plain_files) > 10:
            result.append(f"... +{len(plain_files) - 10} more files")

    return "\n".join(result)


def _truncate_lines(lines: list[str], max_lines: int) -> str:
    """Truncate a list of lines to max_lines with a count of omitted lines."""
    if len(lines) <= max_lines:
        return "\n".join(lines)
    kept = lines[:max_lines]
    kept.append(f"... ({len(lines) - max_lines} more lines)")
    return "\n".join(kept)
