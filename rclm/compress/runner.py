"""Execute a command, apply output filter, and track mechanism savings."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass

from rclm import _config
from rclm.compress.filters.git import filter_git
from rclm.compress.filters.search import filter_search
from rclm.compress.filters.shell import filter_generic, filter_shell, strip_ansi
from rclm.compress.filters.test import filter_test
from rclm.hooks._analytics import mechanism_saving_event
from rclm.hooks.session_store import append_event


@dataclass
class FilterResult:
    text: str
    # Which mechanism produced `text`, or None if nothing changed the output.
    # See PRD_project_cost_analytics.md §5 for the H1-H4/legacy_compress vocabulary.
    mechanism: str | None


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


def apply_filter(command: str, stdout: str, stderr: str, *, exit_code: int = 0) -> FilterResult:
    """Route command to appropriate filter. Returns the (possibly) filtered output
    tagged with which mechanism produced it, for savings telemetry."""
    combined = strip_ansi(stdout + stderr)

    parts = _parse_command(command)
    if not parts:
        return FilterResult(combined, None)

    base_cmd = parts[0]

    if base_cmd == "git" and len(parts) >= 2:
        subcommand = parts[1]
        filtered = filter_git(subcommand, combined)
        if filtered is not None:
            return FilterResult(filtered, "legacy_compress")

    filtered = filter_search(command, combined)
    if filtered is not None:
        return FilterResult(filtered, "H2_search_shaping")

    compression = _config.compression_config()
    max_chars = int(compression["test_filter_max_chars"])
    filtered = (
        filter_test(command, combined, exit_code=exit_code, max_chars=max_chars)
        if compression["test_filter"]
        else None
    )
    if filtered is not None:
        return FilterResult(filtered, "test_filter")

    filtered = filter_shell(command, combined)
    if filtered is not None:
        return FilterResult(filtered, "legacy_compress")

    # No dedicated filter matched — generic repeat-collapse + head/tail cap.
    filtered = filter_generic(combined)
    if filtered is not None:
        return FilterResult(filtered, "H3_exec_compaction")

    return FilterResult(combined, None)


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
    original: str,
    compressed: str,
    mechanism: str | None,
    *,
    applied: bool,
    session_id: str | None = None,
) -> None:
    """Append a MechanismSaving event to the active session JSONL.

    No-ops if nothing changed (mechanism is None) or no session is known —
    those aren't measurements worth recording.
    """
    if mechanism is None:
        return
    if session_id is None:
        session_id = os.environ.get("CLAUDE_SESSION_ID")
    if not session_id:
        return

    tokens_saved_estimate = max(0, (len(original) - len(compressed)) // 4)
    append_event(
        session_id,
        mechanism_saving_event(
            mechanism,
            applied=applied,
            tokens_saved_estimate=tokens_saved_estimate,
            measurement_kind="measured",
            raw_token_estimate=max(0, len(original) // 4),
            compressed_token_estimate=max(0, len(compressed) // 4),
        ),
    )
