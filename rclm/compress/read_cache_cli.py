"""Entry point for rclm-read-cache CLI.

Wraps cat/sed/head/tail/Get-Content with the session-scoped read cache
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
import shlex
import subprocess
import sys

from rclm import _config
from rclm.hooks import read_cache, session_store


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: rclm-read-cache <command...>", file=sys.stderr)
        sys.exit(1)

    args = sys.argv[1:]
    command = shlex.join(args)

    try:
        result = subprocess.run(
            args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:
        print(f"rclm-read-cache: execution error: {exc}", file=sys.stderr)
        sys.exit(1)

    output = result.stdout + result.stderr
    session_id = os.environ.get("CLAUDE_SESSION_ID")

    if session_id:
        try:
            request = read_cache.parse_shell_read(command, cwd=os.getcwd(), shell="posix")
            if request is not None:
                state = session_store.read_read_cache_state(session_id)
                events = session_store.read_events(session_id)
                turn = sum(1 for event in events if event.get("event_type") == "PostToolUse") + 1
                shadow = _config.load().get("shadow_mode", False)
                application = read_cache.apply_range_cache(
                    request,
                    output,
                    state,
                    turn=turn,
                    tool_use_id=None,
                    shadow=shadow,
                )
                session_store.write_read_cache_state(session_id, application.state)
                for event in application.events:
                    session_store.append_event(session_id, event)
                if application.replacement is not None:
                    print(application.replacement, end="")
                    sys.exit(result.returncode)
        except Exception:
            pass  # Any cache error falls through to raw output below.

    print(output, end="")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
