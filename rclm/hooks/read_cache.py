"""Session-scoped, range-aware caching for native and shell file reads.

The parser is deliberately narrow. An unrecognised command, ambiguous output,
filesystem error, or malformed state always passes the original result through.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shlex
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_TRACKED_FILES = 200
MAX_TRACKED_SPANS = 2_000
STATE_VERSION = 1

_SED_RANGE = re.compile(r"^(?P<start>[1-9]\d*)(?:,(?P<end>[1-9]\d*))?p$")
_AWK_RANGE = re.compile(r"^\s*NR\s*>=\s*(?P<start>[1-9]\d*)\s*&&\s*NR\s*<=\s*(?P<end>[1-9]\d*)\s*$")
_POSIX_OPERATORS = {"|", "||", "&", "&&", ";", ">", ">>", "<", "<<"}
_UNSAFE_PATH_CHARS = set("*?[]{}$`\n\r")


@dataclass(frozen=True)
class FileMetadata:
    absolute_path: str
    display_path: str
    content_hash: str
    line_count: int
    size: int


@dataclass(frozen=True)
class ReadRequest:
    metadata: FileMetadata
    start_line: int
    end_line: int
    output_style: str = "plain"

    @property
    def path(self) -> str:
        return self.metadata.absolute_path

    @property
    def display_path(self) -> str:
        return self.metadata.display_path


@dataclass(frozen=True)
class CacheDecision:
    state: dict
    replacement: str | None = None
    cache_hit: bool = False
    reliable: bool = False
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True)
class CacheApplication:
    state: dict
    replacement: str | None
    events: tuple[dict, ...] = ()


def parse_read_range(segment: str) -> tuple[str, int, int] | None:
    """Return ``(path, start_line, end_line)`` for a confident local read."""
    request = parse_shell_read(segment, cwd=os.getcwd(), shell="posix")
    if request is None:
        return None
    return request.display_path, request.start_line, request.end_line


def parse_shell_read(command: str, *, cwd: str, shell: str = "posix") -> ReadRequest | None:
    """Parse one exact file-read command using the selected shell grammar."""
    try:
        shell_name = os.path.basename((shell or "posix").strip().lower())
        if shell_name in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
            parsed = _parse_powershell(command)
        elif shell_name in {"posix", "sh", "bash", "zsh", "dash", "ksh"}:
            parsed = _parse_posix(command)
        else:
            return None
        if parsed is None:
            return None

        path, start, end, style = parsed
        metadata = inspect_file(path, cwd=cwd)
        if metadata is None or metadata.line_count == 0:
            return None
        resolved_end = metadata.line_count if end is None else min(end, metadata.line_count)
        resolved_start = max(1, metadata.line_count - start + 1) if style == "tail" else start
        if resolved_start > resolved_end:
            return None
        return ReadRequest(metadata, resolved_start, resolved_end, style)
    except Exception:
        return None


def native_read_request(tool_input: dict, *, cwd: str) -> ReadRequest | None:
    """Normalise a native Read invocation to one-based inclusive lines."""
    try:
        path = tool_input.get("file_path")
        if not isinstance(path, str) or not path:
            return None
        metadata = inspect_file(path, cwd=cwd)
        if metadata is None or metadata.line_count == 0:
            return None

        raw_offset = tool_input.get("offset")
        raw_limit = tool_input.get("limit")
        offset = int(raw_offset) if raw_offset is not None else 0
        if offset < 0:
            return None
        start = offset + 1
        if raw_limit is None:
            end = metadata.line_count
        else:
            limit = int(raw_limit)
            if limit <= 0:
                return None
            end = min(metadata.line_count, start + limit - 1)
        if start > end:
            return None
        return ReadRequest(metadata, start, end, "native")
    except Exception:
        return None


def inspect_file(path: str, *, cwd: str) -> FileMetadata | None:
    """Hash and count a regular, non-binary file in one streaming pass."""
    try:
        base = Path(cwd).expanduser().resolve()
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved = candidate.resolve(strict=True)
        mode = resolved.stat().st_mode
        if not stat.S_ISREG(mode):
            return None

        digest = hashlib.sha256()
        line_count = 0
        size = 0
        last_byte = b""
        with resolved.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                if b"\x00" in chunk:
                    return None
                digest.update(chunk)
                line_count += chunk.count(b"\n")
                size += len(chunk)
                last_byte = chunk[-1:]
        if size and last_byte != b"\n":
            line_count += 1

        try:
            display = resolved.relative_to(base).as_posix()
        except ValueError:
            display = resolved.name
        return FileMetadata(str(resolved), display, digest.hexdigest(), line_count, size)
    except (OSError, RuntimeError, ValueError):
        return None


def process_read(request: ReadRequest, content: str, state: dict, *, turn: int) -> CacheDecision:
    """Update interval state and optionally build a smaller tool result."""
    try:
        state = _normalise_state(state)
        entry = _entry_for_metadata(state, request.metadata)
        expected_lines = request.end_line - request.start_line + 1
        if _output_line_count(content) != expected_lines:
            return CacheDecision(state=state)

        pieces = _partition_range(
            request.start_line,
            request.end_line,
            entry.get("spans", []),
        )
        covered = [piece for piece in pieces if piece[2] is not None]
        uncovered = [piece for piece in pieces if piece[2] is None]

        for start, end, first_turn in uncovered:
            _add_span(entry, start, end, turn if first_turn is None else first_turn)
        _touch_and_trim(state, request.path)

        if not covered:
            return CacheDecision(
                state=state,
                reliable=True,
                file_path=request.display_path,
                start_line=request.start_line,
                end_line=request.end_line,
            )

        replacement = _build_replacement(content, request, pieces)
        if not replacement or len(replacement) >= len(content):
            replacement = None
        return CacheDecision(
            state=state,
            replacement=replacement,
            cache_hit=replacement is not None,
            reliable=True,
            file_path=request.display_path,
            start_line=request.start_line,
            end_line=request.end_line,
        )
    except Exception:
        return CacheDecision(state=_normalise_state(state))


def apply_range_cache(
    request: ReadRequest,
    content: str,
    state: dict,
    *,
    turn: int,
    tool_use_id: str | None,
    shadow: bool,
) -> CacheApplication:
    """Apply one read and build consistent session/per-call telemetry events."""
    decision = process_read(request, content, state, turn=turn)
    if not decision.cache_hit or decision.replacement is None or decision.file_path is None:
        return CacheApplication(state=decision.state, replacement=None)

    from rclm.hooks._analytics import estimate_tokens, mechanism_saving_event

    raw_tokens = estimate_tokens(content)
    compressed_tokens = estimate_tokens(decision.replacement)
    saved = max(0, raw_tokens - compressed_tokens)
    if saved <= 0:
        return CacheApplication(state=decision.state, replacement=None)
    events = (
        mechanism_saving_event(
            "range_cache",
            applied=not shadow,
            tokens_saved_estimate=saved,
            measurement_kind="measured",
            file_path=decision.file_path,
            raw_token_estimate=raw_tokens,
            compressed_token_estimate=compressed_tokens,
        ),
        transformation_event(
            tool_use_id=tool_use_id,
            raw_tokens=raw_tokens,
            compressed_tokens=compressed_tokens,
            applied=not shadow,
            file_path=decision.file_path,
        ),
    )
    return CacheApplication(
        state=decision.state,
        replacement=None if shadow else decision.replacement,
        events=events,
    )


def next_unseen_offset(tool_input: dict, state: dict, *, cwd: str) -> tuple[int | None, dict]:
    """Return a zero-based offset for the first unseen line of a native Read file."""
    try:
        path = tool_input.get("file_path")
        if not isinstance(path, str) or not path:
            return None, _normalise_state(state)
        metadata = inspect_file(path, cwd=cwd)
        if metadata is None:
            return None, _normalise_state(state)
        state = _normalise_state(state)
        entry = _entry_for_metadata(state, metadata)
        line = 1
        for span in entry.get("spans", []):
            if int(span["start"]) > line:
                break
            if int(span["end"]) >= line:
                line = int(span["end"]) + 1
        _touch_and_trim(state, metadata.absolute_path)
        return (line - 1 if line <= metadata.line_count else None), state
    except Exception:
        return None, _normalise_state(state)


def invalidate_tool_path(state: dict, tool_name: str, tool_input: dict, *, cwd: str) -> dict:
    """Remove cached intervals targeted by Write/Edit/NotebookEdit."""
    state = _normalise_state(state)
    if tool_name not in {"Write", "Edit", "NotebookEdit"}:
        return state
    raw_path = (
        tool_input.get("notebook_path")
        if tool_name == "NotebookEdit"
        else tool_input.get("file_path")
    )
    if not isinstance(raw_path, str) or not raw_path:
        return state
    try:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(cwd).expanduser().resolve() / candidate
        state["files"].pop(str(candidate.resolve(strict=False)), None)
    except (OSError, RuntimeError, ValueError):
        pass
    return state


def _parse_posix(command: str) -> tuple[str, int, int | None, str] | None:
    tokens = _posix_tokens(command)
    if not tokens:
        return None
    if tokens.count("|") == 1:
        pipe = tokens.index("|")
        left, right = tokens[:pipe], tokens[pipe + 1 :]
        nl_path = None
        if len(left) == 2 and _base(left[0]) == "nl":
            nl_path = left[1]
        elif len(left) == 3 and _base(left[0]) == "nl" and left[1] == "-ba":
            nl_path = left[2]
        if (
            nl_path is not None
            and len(right) == 3
            and _base(right[0]) == "sed"
            and right[1] == "-n"
            and _safe_path(nl_path)
        ):
            bounds = _sed_bounds(right[2])
            if bounds is not None:
                return nl_path, bounds[0], bounds[1], "nl"
        return None
    if any(token in _POSIX_OPERATORS for token in tokens):
        return None

    base = _base(tokens[0])
    if base == "sed" and len(tokens) == 4 and tokens[1] == "-n" and _safe_path(tokens[3]):
        bounds = _sed_bounds(tokens[2])
        return (tokens[3], bounds[0], bounds[1], "plain") if bounds else None
    if base == "head" and len(tokens) == 3 and tokens[1] == "-n" and tokens[2].isdigit():
        return None
    if base == "head" and len(tokens) == 3 and re.fullmatch(r"-[1-9]\d*", tokens[1]):
        return (tokens[2], 1, int(tokens[1][1:]), "plain") if _safe_path(tokens[2]) else None
    if base == "head" and len(tokens) == 4 and tokens[1] == "-n" and tokens[2].isdigit():
        return (tokens[3], 1, int(tokens[2]), "plain") if _safe_path(tokens[3]) else None
    if base == "tail" and len(tokens) == 4 and tokens[1] == "-n" and tokens[2].isdigit():
        return (
            (tokens[3], int(tokens[2]), None, "tail")
            if int(tokens[2]) > 0 and _safe_path(tokens[3])
            else None
        )
    if base in {"cat", "nl"} and len(tokens) == 2 and _safe_path(tokens[1]):
        return tokens[1], 1, None, base
    if base == "nl" and len(tokens) == 3 and tokens[1] == "-ba" and _safe_path(tokens[2]):
        return tokens[2], 1, None, "nl"
    if base == "awk" and len(tokens) == 3 and _safe_path(tokens[2]):
        match = _AWK_RANGE.fullmatch(tokens[1])
        if match:
            start, end = int(match["start"]), int(match["end"])
            if start <= end:
                return tokens[2], start, end, "plain"
    return None


def _parse_powershell(command: str) -> tuple[str, int, int | None, str] | None:
    tokens = _powershell_tokens(command)
    if not tokens or _base(tokens[0]).lower() != "get-content":
        return None
    args = tokens[1:]
    if len(args) == 1 and _safe_path(args[0]):
        return args[0], 1, None, "plain"
    if len(args) == 2 and args[0].lower() in {"-path", "-literalpath"} and _safe_path(args[1]):
        return args[1], 1, None, "plain"
    if len(args) == 3 and args[0].lower() in {"-totalcount", "-tail"} and args[1].isdigit():
        count = int(args[1])
        if count <= 0 or not _safe_path(args[2]):
            return None
        return (
            args[2],
            (count if args[0].lower() == "-tail" else 1),
            (None if args[0].lower() == "-tail" else count),
            ("tail" if args[0].lower() == "-tail" else "plain"),
        )
    return None


def _posix_tokens(command: str) -> list[str]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|;&<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except (ValueError, TypeError):
        return []


def _powershell_tokens(command: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for char in command.strip():
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "`":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            else:
                current.append(char)
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
            continue
        if char in "|;&><\n\r":
            return []
        current.append(char)
    if escaped or quote:
        return []
    if current:
        tokens.append("".join(current))
    return tokens


def _sed_bounds(expression: str) -> tuple[int, int] | None:
    match = _SED_RANGE.fullmatch(expression)
    if not match:
        return None
    start = int(match["start"])
    end = int(match["end"] or start)
    return (start, end) if start <= end else None


def _base(token: str) -> str:
    return os.path.basename(token).lower()


def _safe_path(path: str) -> bool:
    return (
        bool(path)
        and not path.startswith("-")
        and not any(char in path for char in _UNSAFE_PATH_CHARS)
    )


def _normalise_state(state: dict | None) -> dict:
    if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
        return {"version": STATE_VERSION, "clock": 0, "files": {}}
    if not isinstance(state.get("files"), dict):
        state["files"] = {}
    state["clock"] = int(state.get("clock") or 0)
    return state


def _entry_for_metadata(state: dict, metadata: FileMetadata) -> dict:
    files = state["files"]
    entry = files.get(metadata.absolute_path)
    if not isinstance(entry, dict) or entry.get("hash") != metadata.content_hash:
        entry = {
            "hash": metadata.content_hash,
            "display_path": metadata.display_path,
            "line_count": metadata.line_count,
            "last_used": 0,
            "spans": [],
        }
        files[metadata.absolute_path] = entry
    else:
        entry["display_path"] = metadata.display_path
        entry["line_count"] = metadata.line_count
        if not isinstance(entry.get("spans"), list):
            entry["spans"] = []
    return entry


def _touch_and_trim(state: dict, path: str) -> None:
    state["clock"] += 1
    if path in state["files"]:
        state["files"][path]["last_used"] = state["clock"]
    while len(state["files"]) > MAX_TRACKED_FILES or _span_count(state) > MAX_TRACKED_SPANS:
        oldest = min(
            state["files"],
            key=lambda key: int(state["files"][key].get("last_used") or 0),
        )
        state["files"].pop(oldest, None)


def _span_count(state: dict) -> int:
    return sum(len(entry.get("spans", [])) for entry in state["files"].values())


def _add_span(entry: dict, start: int, end: int, turn: int) -> None:
    spans = [
        {"start": int(span["start"]), "end": int(span["end"]), "turn": int(span["turn"])}
        for span in entry.get("spans", [])
        if isinstance(span, dict) and {"start", "end", "turn"} <= span.keys()
    ]
    spans.append({"start": start, "end": end, "turn": turn})
    spans.sort(key=lambda span: (span["start"], span["end"]))
    merged: list[dict[str, int]] = []
    for span in spans:
        if merged and span["turn"] == merged[-1]["turn"] and span["start"] <= merged[-1]["end"] + 1:
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
        else:
            merged.append(span)
    entry["spans"] = merged


def _partition_range(start: int, end: int, spans: list[dict]) -> list[tuple[int, int, int | None]]:
    pieces: list[tuple[int, int, int | None]] = []
    cursor = start
    for span in sorted(spans, key=lambda value: int(value.get("start") or 0)):
        span_start = max(start, int(span.get("start") or 0))
        span_end = min(end, int(span.get("end") or 0))
        if span_end < cursor or span_start > end:
            continue
        if span_start > cursor:
            pieces.append((cursor, span_start - 1, None))
        covered_start = max(cursor, span_start)
        if covered_start <= span_end:
            pieces.append((covered_start, span_end, int(span.get("turn") or 0)))
            cursor = span_end + 1
    if cursor <= end:
        pieces.append((cursor, end, None))
    return pieces


def _output_line_count(content: str) -> int:
    if not content:
        return 0
    return sum(1 for _ in io.StringIO(content))


def _build_replacement(
    content: str,
    request: ReadRequest,
    pieces: list[tuple[int, int, int | None]],
) -> str:
    source = iter(io.StringIO(content))
    output: list[str] = []
    for start, end, first_turn in pieces:
        if first_turn is None:
            for _ in range(end - start + 1):
                output.append(next(source))
        else:
            for _ in range(end - start + 1):
                next(source)
            output.append(
                f"[RCLM] Lines {start}-{end} of {request.display_path} unchanged "
                f"since turn {first_turn}.\n"
            )
    return "".join(output).rstrip("\n") + "\n"


def transformation_event(
    *,
    tool_use_id: str | None,
    raw_tokens: int,
    compressed_tokens: int,
    applied: bool,
    file_path: str,
) -> dict[str, Any]:
    saved = max(0, raw_tokens - compressed_tokens)
    return {
        "event_type": "ToolTransformation",
        "tool_use_id": tool_use_id,
        "was_compressed": True,
        "compression_strategy": "range_cache",
        "raw_token_estimate": raw_tokens,
        "compressed_token_estimate": compressed_tokens,
        "tokens_saved_estimate": saved,
        "compression_ratio": compressed_tokens / max(1, raw_tokens),
        "measurement_kind": "measured",
        "file_path": file_path,
        "applied": applied,
    }
