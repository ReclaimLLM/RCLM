"""Filters for search command output (ripgrep, grep).

Shapes raw match dumps into: total counts, per-file match counts, and the
first matching line per file — capped. Full per-line detail is still
available to the agent via a follow-up call with a narrower pattern/path or
an explicit -A/-B/-C context flag, which this filter leaves untouched.
"""

from __future__ import annotations

import re

_MAX_ITEMS = 20
_SMALL_OUTPUT_LINES = 30

# Already-compact output modes — nothing to shape.
_COMPACT_FLAGS = re.compile(r"(?:^|\s)(-l|-c|--files-with-matches|--count)(?:\s|$)")

# `path:line:content` — grep/rg with -r/-n and no heading (the common
# non-tty default for both tools when invoked from an agent shell).
_FLAT_MATCH_RE = re.compile(r"^(?P<file>[^\n:]+?):(?P<line>\d+):(?P<content>.*)$")

# `line:content` — single-file search, or a match line under an rg heading.
_LINE_ONLY_RE = re.compile(r"^(?P<line>\d+):(?P<content>.*)$")


def filter_search(command: str, output: str) -> str | None:
    """Filter rg/grep output. Returns compressed string or None if no filter applies."""
    parts = command.strip().split()
    if not parts:
        return None

    base_cmd = parts[0]
    if base_cmd not in ("rg", "grep"):
        return None

    if not output.strip():
        return None

    if _COMPACT_FLAGS.search(command):
        return None  # already asked for files-with-matches / count mode

    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) <= _SMALL_OUTPUT_LINES:
        return None  # small enough, keep as-is

    by_file = _parse_flat(lines) or _parse_heading(lines)
    if by_file:
        return _shape_by_file(by_file)

    single_file = _parse_line_only(lines)
    if single_file:
        return _shape_flat(single_file)

    return None


def _parse_flat(lines: list[str]) -> dict[str, list[tuple[str, str]]] | None:
    """Parse `path:line:content` lines. None if not every line matches."""
    by_file: dict[str, list[tuple[str, str]]] = {}
    for line in lines:
        m = _FLAT_MATCH_RE.match(line)
        if not m:
            return None
        by_file.setdefault(m["file"], []).append((m["line"], m["content"]))
    return by_file


def _parse_heading(lines: list[str]) -> dict[str, list[tuple[str, str]]] | None:
    """Parse ripgrep's headed format: bare path line, then `line:content` lines."""
    by_file: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    for line in lines:
        m = _LINE_ONLY_RE.match(line)
        if m:
            if current is None:
                return None
            by_file.setdefault(current, []).append((m["line"], m["content"]))
        else:
            current = line.strip()
            by_file.setdefault(current, [])
    if not by_file or any(not v for v in by_file.values()):
        return None
    return by_file


def _parse_line_only(lines: list[str]) -> list[tuple[str, str]] | None:
    """Parse plain `line:content` matches with no file grouping at all."""
    result: list[tuple[str, str]] = []
    for line in lines:
        m = _LINE_ONLY_RE.match(line)
        if not m:
            return None
        result.append((m["line"], m["content"]))
    return result


def _shape_by_file(by_file: dict[str, list[tuple[str, str]]]) -> str:
    total_matches = sum(len(v) for v in by_file.values())
    files = list(by_file.items())

    result = [f"{total_matches} matches across {len(files)} files"]
    for path, matches in files[:_MAX_ITEMS]:
        label = "match" if len(matches) == 1 else "matches"
        result.append(f"{path} ({len(matches)} {label})")
        line_no, content = matches[0]
        result.append(f"  {line_no}: {content.strip()}")

    if len(files) > _MAX_ITEMS:
        result.append(f"... +{len(files) - _MAX_ITEMS} more files with matches")

    return "\n".join(result)


def _shape_flat(matches: list[tuple[str, str]]) -> str:
    result = [f"{len(matches)} matching lines"]
    for line_no, content in matches[:_MAX_ITEMS]:
        result.append(f"  {line_no}: {content.strip()}")

    if len(matches) > _MAX_ITEMS:
        result.append(f"... +{len(matches) - _MAX_ITEMS} more matches")

    return "\n".join(result)
