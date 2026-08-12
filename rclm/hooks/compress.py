"""PreToolUse compression engine.

Decides whether to modify tool input to reduce context window bloat.
Returns an updatedInput dict for Claude Code's hookSpecificOutput, or None.
"""

from __future__ import annotations

import base64
import os
import re
import shlex
import shutil

# File size threshold (lines) above which Read tool gets a limit injected.
READ_LINE_THRESHOLD = 500
READ_INJECT_LIMIT = 200

# Default head_limit for Grep when none is set.
GREP_DEFAULT_HEAD_LIMIT = 50

# Default output_mode for Grep when none is set: per-file match counts
# instead of full matching lines. Cheapest native mode short of
# files_with_matches, and it keeps the "how much is in this file" signal
# an agent needs to decide whether to read further.
GREP_DEFAULT_OUTPUT_MODE = "count"

# Commands eligible for rewriting to rclm-compress.
# Patterns: base command → True (rewrite the full command).
_BASH_REWRITE_COMMANDS = {
    "Get-Content",
    "cat",
    "git",
    "pytest",
    "python",
    "python3",
    "npm",
    "npx",
    "cargo",
    "ls",
    "find",
    "rg",
    "grep",
    "go",
    "nl",
    "sed",
}


def is_safe_session_id(value: object) -> bool:
    """Return whether a session ID is safe as one local filename component."""
    return (
        isinstance(value, str)
        and 0 < len(value) <= 200
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and all(ord(char) >= 32 for char in value)
    )


def maybe_compress(
    tool_name: str,
    tool_input: dict,
    *,
    shadow: bool = False,
    read_offset: int | None = None,
    session_id: str | None = None,
) -> dict | None:
    """Return updatedInput dict if compression applies, None otherwise.

    `shadow=True` suppresses the Read/Grep native-tool shaping — those have no
    before/after size available at PreToolUse time to measure savings against,
    so shadow mode just skips them rather than rewriting unmeasured. The Bash
    rewrite is unaffected: it always routes through rclm-compress, which
    measures and handles its own shadow/enforce output decision.
    """
    if tool_name == "Read":
        return None if shadow else _compress_read(tool_input, read_offset=read_offset)
    if tool_name == "Grep":
        return None if shadow else _compress_grep(tool_input)
    if tool_name == "Bash":
        return _compress_bash(tool_input, session_id=session_id)
    return None


def _compress_read(tool_input: dict, *, read_offset: int | None = None) -> dict | None:
    """If a file is large and unbounded, inject a progressing read window."""
    if tool_input.get("limit"):
        return None  # User/agent already set a limit

    file_path = tool_input.get("file_path", "")
    if not file_path:
        return None

    try:
        line_count = _count_lines(file_path)
    except OSError:
        return None

    if line_count <= READ_LINE_THRESHOLD:
        return None

    delta = {"limit": READ_INJECT_LIMIT}
    if read_offset is not None and tool_input.get("offset") is None:
        delta["offset"] = read_offset
    return delta


def _compress_grep(tool_input: dict) -> dict | None:
    """Default output_mode to counts and inject head_limit, where unset.

    Full per-match content is still one call away: the agent gets file +
    match-count shape first, then can re-call with output_mode="content"
    (or a narrower pattern/path) for detail on demand.
    """
    delta: dict = {}
    if not tool_input.get("output_mode"):
        delta["output_mode"] = GREP_DEFAULT_OUTPUT_MODE
    if not tool_input.get("head_limit"):
        delta["head_limit"] = GREP_DEFAULT_HEAD_LIMIT

    return delta or None


def _compress_bash(tool_input: dict, *, session_id: str | None = None) -> dict | None:
    """Rewrite command to rclm-compress if it matches a known filter."""
    command = tool_input.get("command", "")
    if not command or not command.strip():
        return None

    # Don't rewrite if already wrapped
    if "rclm-compress" in command:
        return None

    # Don't rewrite if already wrapped by RTK or similar
    if command.strip().startswith("rtk "):
        return None

    # Check if rclm-compress is available
    if not _compress_available():
        return None

    if is_compressible_command(command, shell=_detect_shell(tool_input)):
        session_arg = (
            f" --session-id {shlex.quote(session_id)}" if is_safe_session_id(session_id) else ""
        )
        encoded_command = base64.urlsafe_b64encode(command.encode("utf-8")).decode("ascii")
        return {"command": f"rclm-compress{session_arg} --encoded-command {encoded_command}"}

    return None


def is_compressible_command(command: str, *, shell: str = "posix") -> bool:
    """Return whether a command has a conservative, recognized output shape.

    The same predicate gates PreToolUse command wrapping and PostToolUse
    fallback compaction. Keeping it here prevents provider handlers from
    growing subtly different command allowlists.
    """
    if not isinstance(command, str) or not command.strip():
        return False

    posix_shell = _is_posix_shell(shell)
    if posix_shell:
        segments = split_command_segments(command, shell=shell)
    else:
        # PowerShell is not parsed as POSIX, but simple Get-Content commands
        # still have a stable text-result shape. Compound PowerShell input is
        # deliberately left unchanged rather than guessed at.
        if any(separator in command for separator in (";", "|", "&&", "||")):
            return False
        segments = [command]

    for segment in segments:
        base_cmd = extract_base_command(segment)
        if base_cmd not in _BASH_REWRITE_COMMANDS:
            continue
        if not posix_shell and base_cmd != "Get-Content":
            continue

        if base_cmd in {"python", "python3"} and "-m pytest" not in segment:
            continue
        if base_cmd in ("npm", "npx") and not any(
            keyword in segment for keyword in ("test", "jest", "vitest")
        ):
            continue
        if base_cmd == "go" and not segment.lstrip().startswith("go test"):
            continue
        return True

    return False


def split_command_segments(command: str, shell: str = "posix") -> list[str]:
    """Split a shell command on top-level command separators."""
    try:
        if not _is_posix_shell(shell):
            return []

        stripped = command.strip()
        if not stripped:
            return []

        segments: list[str] = []
        current: list[str] = []
        quote: str | None = None
        escaped = False
        subshell_depth = 0
        i = 0

        while i < len(command):
            char = command[i]
            next_char = command[i + 1] if i + 1 < len(command) else ""

            if escaped:
                current.append(char)
                escaped = False
                i += 1
                continue

            if char == "\\":
                current.append(char)
                escaped = True
                i += 1
                continue

            if quote:
                current.append(char)
                if char == quote:
                    quote = None
                i += 1
                continue

            if char in ("'", '"'):
                current.append(char)
                quote = char
                i += 1
                continue

            if char == "$" and next_char == "(":
                current.append(char)
                current.append(next_char)
                subshell_depth += 1
                i += 2
                continue

            if subshell_depth and char == ")":
                current.append(char)
                subshell_depth -= 1
                i += 1
                continue

            if subshell_depth:
                current.append(char)
                i += 1
                continue

            separator_length = 0
            if char in ("&", "|") and next_char == char:
                separator_length = 2
            elif char in ("|", ";"):
                separator_length = 1

            if separator_length:
                segment = "".join(current).strip()
                if segment:
                    segments.append(segment)
                current = []
                i += separator_length
                continue

            current.append(char)
            i += 1

        if quote or escaped or subshell_depth:
            return []

        segment = "".join(current).strip()
        if segment:
            segments.append(segment)
        return segments
    except Exception:
        return []


def _detect_shell(tool_input: dict) -> str:
    """Detect shell syntax from hook input, falling back to the current OS."""
    shell = tool_input.get("shell")
    if isinstance(shell, str) and shell.strip():
        return shell
    return "posix" if os.name == "posix" else os.name


def _is_posix_shell(shell: str) -> bool:
    shell_name = os.path.basename(shell.strip().lower())
    return shell_name in {"posix", "sh", "bash", "zsh", "dash", "ksh"}


def extract_base_command(segment: str) -> str:
    """Extract the base command from one shell command segment."""
    try:
        parts = shlex.split(segment)
    except Exception:
        return ""

    assignment_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
    for part in parts:
        if assignment_pattern.match(part):
            continue
        return os.path.basename(part)
    return ""


def _count_lines(file_path: str) -> int:
    """Count lines in a file without reading it all into memory."""
    count = 0
    with open(file_path, "rb") as f:
        for _ in f:
            count += 1
    return count


_compress_bin_cached: bool | None = None


def _compress_available() -> bool:
    """Check if rclm-compress is on PATH (cached)."""
    global _compress_bin_cached
    if _compress_bin_cached is None:
        _compress_bin_cached = shutil.which("rclm-compress") is not None
    return _compress_bin_cached
