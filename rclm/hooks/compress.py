"""PreToolUse compression engine.

Decides whether to modify tool input to reduce context window bloat.
Returns an updatedInput dict for Claude Code's hookSpecificOutput, or None.
"""
from __future__ import annotations

import os
import shutil

# File size threshold (lines) above which Read tool gets a limit injected.
READ_LINE_THRESHOLD = 500
READ_INJECT_LIMIT = 200

# Default head_limit for Grep when none is set.
GREP_DEFAULT_HEAD_LIMIT = 50

# Commands eligible for rewriting to rclm-compress.
# Patterns: base command → True (rewrite the full command).
_BASH_REWRITE_COMMANDS = {
    "git",
    "pytest", "python",
    "npm", "npx",
    "cargo",
    "ls", "find",
}


def maybe_compress(tool_name: str, tool_input: dict) -> dict | None:
    """Return updatedInput dict if compression applies, None otherwise."""
    if tool_name == "Read":
        return _compress_read(tool_input)
    if tool_name == "Grep":
        return _compress_grep(tool_input)
    if tool_name == "Bash":
        return _compress_bash(tool_input)
    return None


def _compress_read(tool_input: dict) -> dict | None:
    """If file is large and no limit set, inject a limit."""
    if tool_input.get("limit"):
        return None  # User/agent already set a limit

    file_path = tool_input.get("file_path", "")
    if not file_path:
        return None

    try:
        line_count = _count_lines(file_path)
    except (OSError, IOError):
        return None

    if line_count <= READ_LINE_THRESHOLD:
        return None

    return {"limit": READ_INJECT_LIMIT}


def _compress_grep(tool_input: dict) -> dict | None:
    """Inject head_limit if not already set."""
    if tool_input.get("head_limit"):
        return None  # Already has a limit

    return {"head_limit": GREP_DEFAULT_HEAD_LIMIT}


def _compress_bash(tool_input: dict) -> dict | None:
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

    base_cmd = _extract_base_command(command)
    if base_cmd not in _BASH_REWRITE_COMMANDS:
        return None

    # For python, only rewrite if it's a test command
    if base_cmd == "python" and "-m pytest" not in command:
        return None

    # For npm/npx, only rewrite test-related commands
    if base_cmd in ("npm", "npx"):
        if not any(kw in command for kw in ("test", "jest", "vitest")):
            return None

    return {"command": f"rclm-compress {command}"}


def _extract_base_command(command: str) -> str:
    """Extract the base command from a potentially complex shell string."""
    stripped = command.strip()
    # Handle env var prefixes like "FOO=bar git status"
    parts = stripped.split()
    for part in parts:
        if "=" in part and not part.startswith("-"):
            continue
        # Found the actual command
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
