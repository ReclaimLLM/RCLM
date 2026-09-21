"""Capture-derived ReadRequest construction for offline H1 range-cache replay.

The shipped H1 request builders (`read_cache.native_read_request`,
`read_cache.parse_shell_read`) call `read_cache.inspect_file()`, which opens
the file on disk to hash it and count its lines. Replaying a past session
that way is not viable: the file may no longer exist, or may have changed
since capture (see docs/work_context/PRD_Replay_MCP.md, Issue 2b).

This module builds the same `ReadRequest`/`FileMetadata` value objects the
real `read_cache.process_read` interval machine consumes, but derives them
entirely from the captured tool call (`tool_input` + the captured result
text) instead of the live filesystem. `process_read` itself is reused
unmodified — only the input construction differs.

Where the captured data does not let us derive a reliable line range (an
ambiguous shell read, a tail-style read whose file length we can't know
without the disk, a native Read whose result isn't in the expected numbered
format), we return None. Callers must treat that as `unresolvable`, never as
an estimate.
"""

from __future__ import annotations

import hashlib
import re

from rclm.hooks.read_cache import FileMetadata, ReadRequest, deserialize_read_request
from rclm.hooks.tool_semantics import command_text

_NUMBERED_LINE = re.compile(r"^\s*(\d+)→(.*)$")
_ANTIGRAVITY_NUMBERED_LINE = re.compile(r"^(\d+):(?: (.*))?$")
_ANTIGRAVITY_FILE_PATH = re.compile(r"^File Path: `(?:file://)?(?P<path>.+)`$")
_ANTIGRAVITY_TOTAL_LINES = re.compile(r"^Total Lines: (?P<count>\d+)$")
_ANTIGRAVITY_SHOWING = re.compile(r"^Showing lines (?P<start>\d+) to (?P<end>\d+)$")
_SED_RANGE = re.compile(r"^(?P<start>[1-9]\d*)(?:,(?P<end>[1-9]\d*))?p$")
_AWK_RANGE = re.compile(r"^\s*NR\s*>=\s*(?P<start>[1-9]\d*)\s*&&\s*NR\s*<=\s*(?P<end>[1-9]\d*)\s*$")
_HEAD_N = re.compile(r"(?:^|\s)-(?:n\s*)?(\d+)(?:\s|$)")

_SHELL_READ_TOOL_NAMES = frozenset(
    {"bash", "exec", "exec_command", "shell", "run_command", "run_shell_command"}
)
_NATIVE_READ_TOOL_NAMES = frozenset({"read", "read_file", "read_text_file", "view_file"})


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="surrogateescape")).hexdigest()


def _line_count(content: str) -> int:
    if not content:
        return 0
    return content.count("\n") + (0 if content.endswith("\n") else 1)


def split_native_read_block(content: str) -> tuple[str, str] | None:
    """Split captured Read output into (numbered file block, trailing remainder).

    Real captures append non-file text after the numbered lines (e.g. Claude
    Code's `<system-reminder>` suffix) — only the leading contiguous run of
    `NNNN→text` lines is the file's content. Returns None if the content
    doesn't start with a numbered line at all.
    """
    lines = content.splitlines(keepends=True)
    if not lines or _NUMBERED_LINE.match(lines[0].rstrip("\n")) is None:
        return None
    cut = len(lines)
    for index, line in enumerate(lines):
        if _NUMBERED_LINE.match(line.rstrip("\n")) is None:
            cut = index
            break
    return "".join(lines[:cut]), "".join(lines[cut:])


def split_antigravity_view_block(content: str) -> tuple[str, str, str] | None:
    """Split a captured Antigravity view into prefix, numbered file, suffix."""
    lines = content.splitlines(keepends=True)
    first = next(
        (
            index
            for index, line in enumerate(lines)
            if _ANTIGRAVITY_NUMBERED_LINE.fullmatch(line.rstrip("\r\n"))
        ),
        None,
    )
    if first is None:
        return None
    end = first
    while end < len(lines) and _ANTIGRAVITY_NUMBERED_LINE.fullmatch(lines[end].rstrip("\r\n")):
        end += 1
    return "".join(lines[:first]), "".join(lines[first:end]), "".join(lines[end:])


def build_antigravity_view_request(
    tool_input: dict, content: str
) -> tuple[ReadRequest, str, str, str] | None:
    """Build a replay request from Antigravity's documented ``view_file`` shape."""
    split = split_antigravity_view_block(content)
    if split is None:
        return None
    prefix, block, suffix = split
    path = tool_input.get("AbsolutePath")
    if not isinstance(path, str) or not path:
        return None

    prefix_lines = prefix.splitlines()
    captured_path = next(
        (
            match.group("path")
            for line in prefix_lines
            if (match := _ANTIGRAVITY_FILE_PATH.fullmatch(line)) is not None
        ),
        None,
    )
    total_lines = next(
        (
            int(match.group("count"))
            for line in prefix_lines
            if (match := _ANTIGRAVITY_TOTAL_LINES.fullmatch(line)) is not None
        ),
        None,
    )
    showing = next(
        (
            (int(match.group("start")), int(match.group("end")))
            for line in prefix_lines
            if (match := _ANTIGRAVITY_SHOWING.fullmatch(line)) is not None
        ),
        None,
    )
    block_lines = block.splitlines()
    matches = [_ANTIGRAVITY_NUMBERED_LINE.fullmatch(line) for line in block_lines]
    if (
        captured_path != path
        or total_lines is None
        or showing is None
        or not block_lines
        or not all(matches)
    ):
        return None
    numbers = [int(match.group(1)) for match in matches if match is not None]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        return None
    start, end = showing
    if start != numbers[0] or end != numbers[-1] or end > total_lines:
        return None
    metadata = FileMetadata(
        absolute_path=path,
        display_path=path,
        content_hash=_content_hash(block),
        line_count=total_lines,
        size=len(block),
    )
    return ReadRequest(metadata, start, end, "native"), block, prefix, suffix


def _native_metadata(path: str, block: str) -> FileMetadata | None:
    """Parse Claude/Codex's numbered `NNNN→text` Read output.

    Returns metadata scoped to the *returned range*, not the full file:
    replay only ever needs the interval this call covers plus a stable
    content hash for change detection within the session.
    """
    lines = block.split("\n")
    if block.endswith("\n"):
        lines = lines[:-1]
    if not lines:
        return None
    matches = [_NUMBERED_LINE.match(line) for line in lines]
    if not all(matches):
        return None
    numbers = [int(m.group(1)) for m in matches]  # type: ignore[union-attr]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        return None
    return FileMetadata(
        absolute_path=path,
        display_path=path,
        content_hash=_content_hash(block),
        line_count=numbers[-1],
        size=len(block),
    )


def build_native_read_request(tool_input: dict, content: str) -> tuple[ReadRequest, str] | None:
    """Build (ReadRequest, trailing_remainder) for a native Read call.

    `trailing_remainder` is any non-file text captured after the numbered
    block (see `split_native_read_block`) and must be reattached unchanged
    by the caller after applying any H1 replacement.
    """
    path = tool_input.get("file_path")
    if not isinstance(path, str) or not path:
        return None
    split = split_native_read_block(content)
    if split is None:
        return None
    block, trailing = split
    metadata = _native_metadata(path, block)
    if metadata is None:
        return None
    lines = block.split("\n")
    if block.endswith("\n"):
        lines = lines[:-1]
    first_match = _NUMBERED_LINE.match(lines[0])
    if first_match is None:
        return None
    start_line = int(first_match.group(1))
    end_line = start_line + len(lines) - 1
    return ReadRequest(metadata, start_line, end_line, "native"), trailing


def build_plain_native_read_request(
    tool_input: dict, content: str
) -> tuple[ReadRequest, str] | None:
    """Build a request for Gemini-style plain-text ``read_file`` output."""
    path = tool_input.get("file_path") or tool_input.get("path")
    if not isinstance(path, str) or not path or not content:
        return None
    raw_offset = tool_input.get("offset", 0)
    if isinstance(raw_offset, bool) or not isinstance(raw_offset, int) or raw_offset < 0:
        return None
    start = raw_offset + 1
    count = _line_count(content)
    end = start + count - 1
    metadata = FileMetadata(
        absolute_path=path,
        display_path=path,
        content_hash=_content_hash(content),
        line_count=end,
        size=len(content),
    )
    return ReadRequest(metadata, start, end, "native"), ""


def _extract_shell_range(command: str) -> tuple[str, int, int | None, str] | None:
    """Return (path, start, end_or_none, style) for a narrow, unambiguous set
    of read-shaped shell commands: `sed -n 'X,Yp' FILE`, `sed -n 'Np' FILE`,
    `awk 'NR>=X && NR<=Y' FILE`, `head [-n] N FILE`. Anything else -> None.

    Deliberately conservative, matching the fail-open posture used everywhere
    else in this codebase (read_cache.py, tool_result_transform.py): a command
    this can't confidently parse is not H1-eligible, not estimated.
    """
    parts = command.strip().split()
    if len(parts) < 2:
        return None
    base = parts[0].rsplit("/", 1)[-1]

    if base == "sed" and "-n" in parts:
        script = next((p for p in parts[1:] if p not in {"-n"} and not p.startswith("-")), None)
        if script is None:
            return None
        match = _SED_RANGE.match(script.strip("'\""))
        if match is None:
            return None
        path = parts[-1]
        start = int(match.group("start"))
        end = int(match.group("end")) if match.group("end") else start
        return path, start, end, "plain"

    if base == "awk":
        script = next((p for p in parts[1:] if not p.startswith("-")), None)
        if script is None:
            return None
        match = _AWK_RANGE.match(script.strip("'\""))
        if match is None:
            return None
        path = parts[-1]
        return path, int(match.group("start")), int(match.group("end")), "plain"

    if base == "head":
        rest = " ".join(parts[1:-1])
        match = _HEAD_N.search(rest)
        n = int(match.group(1)) if match else 10
        return parts[-1], 1, n, "plain"

    return None


def build_shell_read_request(command: str, content: str) -> tuple[ReadRequest, str] | None:
    """Build (ReadRequest, "") for a recognized shell read from its captured output.

    The trailing element is always empty: shell read output carries no
    appended non-file suffix the way native Read captures do. Kept for a
    uniform return contract with `build_native_read_request`.

    Tail-style reads (`tail`) are not supported: resolving "last N lines"
    requires the file's total line count, which is only known on disk.
    """
    parsed = _extract_shell_range(command)
    if parsed is None:
        return None
    path, start, end, style = parsed
    output_lines = _line_count(content)
    if end is None or output_lines != (end - start + 1):
        return None
    metadata = FileMetadata(
        absolute_path=path,
        display_path=path,
        content_hash=_content_hash(content),
        line_count=end,
        size=len(content),
    )
    return ReadRequest(metadata, start, end, style), ""


def build_read_request(
    tool_name: str,
    tool_input: object,
    content: str,
    captured_metadata: object = None,
) -> tuple[ReadRequest, str] | None:
    """Dispatch to the native or shell read-request builder, or None.

    Returns (ReadRequest, trailing_remainder); see `build_native_read_request`.
    """
    name = tool_name.lower()
    if not isinstance(tool_input, dict):
        return None
    captured = deserialize_read_request(captured_metadata)
    if captured is not None and name in _NATIVE_READ_TOOL_NAMES | _SHELL_READ_TOOL_NAMES:
        trailing = ""
        if name == "read":
            split = split_native_read_block(content)
            if split is not None:
                _block, trailing = split
        return captured, trailing
    if name == "view_file":
        built = build_antigravity_view_request(tool_input, content)
        if built is None:
            return None
        request, _block, _prefix, suffix = built
        return request, suffix
    if name in {"read_file", "read_text_file"}:
        return build_plain_native_read_request(tool_input, content)
    if name in _NATIVE_READ_TOOL_NAMES:
        return build_native_read_request(tool_input, content)
    if name in _SHELL_READ_TOOL_NAMES:
        command = command_text(tool_input)
        if not isinstance(command, str):
            return None
        return build_shell_read_request(command, content)
    return None


__all__ = [
    "build_antigravity_view_request",
    "build_native_read_request",
    "build_plain_native_read_request",
    "build_read_request",
    "build_shell_read_request",
    "split_antigravity_view_block",
    "split_native_read_block",
]
