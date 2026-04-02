"""Execute a command, apply output filter, and track compression savings."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from rclm.compress.filters.git import filter_git
from rclm.compress.filters.shell import filter_shell
from rclm.compress.filters.test import filter_test

_SESSIONS_DIR = Path.home() / ".reclaimllm" / "sessions"


def execute(command: str) -> tuple[str, str, int]:
    """Run command via shell, return (stdout, stderr, exit_code)."""
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.stdout, result.stderr, result.returncode


def apply_filter(command: str, stdout: str, stderr: str) -> str:
    """Route command to appropriate filter. Returns filtered output."""
    parts = _parse_command(command)
    if not parts:
        return stdout + stderr

    base_cmd = parts[0]

    if base_cmd == "git" and len(parts) >= 2:
        subcommand = parts[1]
        filtered = filter_git(subcommand, stdout + stderr)
        if filtered is not None:
            return filtered

    filtered = filter_test(command, stdout + stderr)
    if filtered is not None:
        return filtered

    filtered = filter_shell(command, stdout + stderr)
    if filtered is not None:
        return filtered

    # No filter matched — return original output
    return stdout + stderr


def _parse_command(command: str) -> list[str]:
    """Parse command string into parts, handling pipes and chains."""
    # Take the first command in a pipe chain
    first_cmd = command.split("|")[0].strip()
    # Take the first command in a && chain
    first_cmd = first_cmd.split("&&")[0].strip()
    # Take the first command in a ; chain
    first_cmd = first_cmd.split(";")[0].strip()
    try:
        return shlex.split(first_cmd)
    except ValueError:
        return first_cmd.split()


def track_savings(
    command: str,
    original: str,
    compressed: str,
    session_id: str | None = None,
) -> None:
    """Append compression stats to the active session JSONL."""
    if session_id is None:
        session_id = os.environ.get("CLAUDE_SESSION_ID")
    if not session_id:
        return

    original_chars = len(original)
    compressed_chars = len(compressed)

    event = {
        "event_type": "CompressionSaving",
        "command": command[:200],  # truncate long commands
        "original_chars": original_chars,
        "compressed_chars": compressed_chars,
        "savings_pct": round((1 - compressed_chars / original_chars) * 100, 1)
        if original_chars > 0
        else 0.0,
    }

    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _SESSIONS_DIR / f"{session_id}.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")
