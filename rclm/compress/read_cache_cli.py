"""Entry point for rclm-read-cache CLI.

Wraps cat/sed/head/tail/type/Get-Content with the session-scoped read cache
(diff-on-change) — the shell counterpart of the native Read tool's
PostToolUse handling in claude_handler.py. Executes the command, then either
prints the raw output (first read of this file) or a cache-derived
replacement (unchanged notice / diff), always preserving the original exit
code.

In shadow mode (config `shadow_mode: true`), the cache lookup and savings
measurement still happen, but the original output is always printed — the
agent sees exactly what it would have without this mechanism.

Usage: rclm-read-cache <command...>
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone

from rclm import _config
from rclm.hooks import read_cache, session_store
from rclm.hooks._analytics import mechanism_saving_event


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _target_file(args: list[str]) -> str | None:
    """Best-effort: last non-flag argument is the target file."""
    for token in reversed(args):
        if not token.startswith("-"):
            return token
    return None


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: rclm-read-cache <command...>", file=sys.stderr)
        sys.exit(1)

    args = sys.argv[1:]
    command = " ".join(args)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:
        print(f"rclm-read-cache: execution error: {exc}", file=sys.stderr)
        sys.exit(1)

    output = result.stdout + result.stderr
    file_path = _target_file(args)
    session_id = os.environ.get("CLAUDE_SESSION_ID")

    if session_id and file_path:
        try:
            events = session_store.read_events(session_id)
            delta = read_cache.build_delta(file_path, None, None, output, events)
            session_store.append_event(
                session_id,
                read_cache.snapshot_event(file_path, None, None, output, _now()),
            )
            if delta is not None:
                shadow = _config.load().get("shadow_mode", False)
                tokens_saved = max(0, (len(output) - len(delta["updatedToolOutput"])) // 4)
                session_store.append_event(
                    session_id,
                    mechanism_saving_event(
                        "H1_read_cache",
                        applied=not shadow,
                        tokens_saved_estimate=tokens_saved,
                    ),
                )
                if not shadow:
                    print(delta["updatedToolOutput"], end="")
                    sys.exit(result.returncode)
        except Exception:
            pass  # Any cache error falls through to raw output below.

    print(output, end="")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
