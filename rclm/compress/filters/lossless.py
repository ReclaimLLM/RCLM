"""Fast, reversible compaction for search results and path listings.

These folds change presentation without dropping result data. Every candidate
is expanded again before use; unfamiliar or ambiguous shapes pass through.
"""

from __future__ import annotations

import re

_MAX_INPUT_CHARS = 2_000_000
MECHANISM = "lossless_search_path_folding"

_SEARCH_ROW_RE = re.compile(r"^(?P<path>[^\n:]+):(?P<line>\d+):(?P<content>.*)$")
_SEARCH_DATA_RE = re.compile(r"^(?P<line>\d+):(?P<content>.*)$")
_DIRECTORY_DATA_RE = re.compile(r"^(?P<base>[^/\n:]+):(?P<line>\d+):(?P<content>.*)$")
_PATH_ROW_RE = re.compile(r"^(?P<directory>(?:\.{0,2}/)?(?:[^/\s:]+/)+)(?P<base>[^/\s:]+)$")


def compact_search_and_paths(output: str) -> str | None:
    """Return the smallest verified search/path fold, or ``None``.

    Work is linear in the output size and bounded so this can run in the hook
    critical path. A fold is accepted only when its inverse recreates the
    original output exactly and the candidate is shorter.
    """
    if not output or len(output) > _MAX_INPUT_CHARS:
        return None

    candidates = (
        (_fold_search_files(output), _unfold_search_files),
        (_fold_search_directories(output), _unfold_search_directories),
        (_fold_path_listing(output), _unfold_path_listing),
    )
    best = output
    for candidate, inverse in candidates:
        if len(candidate) < len(best) and inverse(candidate) == output:
            best = candidate
    return best if best != output else None


def _split_keep_trailing(text: str) -> tuple[list[str], bool]:
    if not text:
        return [], False
    trailing = text.endswith("\n")
    body = text[:-1] if trailing else text
    return body.split("\n"), trailing


def _join(lines: list[str], trailing: bool) -> str:
    result = "\n".join(lines)
    return result + "\n" if trailing else result


def _fold_search_files(text: str) -> str:
    lines, trailing = _split_keep_trailing(text)
    result: list[str] = []
    current_path: str | None = None
    for line in lines:
        match = _SEARCH_ROW_RE.match(line)
        if match is None:
            result.append(line)
            current_path = None
            continue
        path = match.group("path")
        if path != current_path:
            result.append(path)
            current_path = path
        result.append(f"{match.group('line')}:{match.group('content')}")
    return _join(result, trailing)


def _unfold_search_files(text: str) -> str:
    lines, trailing = _split_keep_trailing(text)
    result: list[str] = []
    current_path: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        data = _SEARCH_DATA_RE.match(line)
        if current_path is not None and data is not None:
            result.append(f"{current_path}:{line}")
            index += 1
            continue
        if data is None and index + 1 < len(lines) and _SEARCH_DATA_RE.match(lines[index + 1]):
            current_path = line
            index += 1
            continue
        current_path = None
        result.append(line)
        index += 1
    return _join(result, trailing)


def _fold_search_directories(text: str) -> str:
    lines, trailing = _split_keep_trailing(text)
    result: list[str] = []
    current_directory: str | None = None
    for line in lines:
        match = _SEARCH_ROW_RE.match(line)
        path = match.group("path") if match is not None else ""
        if match is None or "/" not in path:
            result.append(line)
            current_directory = None
            continue
        split = path.rindex("/") + 1
        directory, base = path[:split], path[split:]
        if directory != current_directory:
            result.append(directory)
            current_directory = directory
        result.append(f"{base}:{match.group('line')}:{match.group('content')}")
    return _join(result, trailing)


def _unfold_search_directories(text: str) -> str:
    lines, trailing = _split_keep_trailing(text)
    result: list[str] = []
    current_directory: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        data = _DIRECTORY_DATA_RE.match(line)
        if current_directory is not None and data is not None:
            result.append(f"{current_directory}{line}")
            index += 1
            continue
        if (
            line.endswith("/")
            and index + 1 < len(lines)
            and _DIRECTORY_DATA_RE.match(lines[index + 1]) is not None
        ):
            current_directory = line
            index += 1
            continue
        current_directory = None
        result.append(line)
        index += 1
    return _join(result, trailing)


def _fold_path_listing(text: str) -> str:
    lines, trailing = _split_keep_trailing(text)
    if sum(_PATH_ROW_RE.match(line) is not None for line in lines) < 2:
        return text

    result: list[str] = []
    current_directory: str | None = None
    for line in lines:
        match = _PATH_ROW_RE.match(line)
        if match is None:
            result.append(line)
            current_directory = None
            continue
        directory = match.group("directory")
        if directory != current_directory:
            result.append(directory)
            current_directory = directory
        result.append(match.group("base"))
    return _join(result, trailing)


def _unfold_path_listing(text: str) -> str:
    lines, trailing = _split_keep_trailing(text)
    result: list[str] = []
    current_directory: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        is_base = bool(line) and "/" not in line
        if current_directory is not None and is_base:
            result.append(current_directory + line)
            index += 1
            continue
        if line.endswith("/") and index + 1 < len(lines):
            next_line = lines[index + 1]
            if next_line and "/" not in next_line:
                current_directory = line
                index += 1
                continue
        current_directory = None
        result.append(line)
        index += 1
    return _join(result, trailing)
