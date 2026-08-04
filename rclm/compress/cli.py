"""Entry point for rclm-compress CLI.

Usage: rclm-compress [--session-id ID] [--encoded-command VALUE | <command...>]

Executes the command, applies output compression filters, tracks savings,
and prints the (possibly compressed) output. Preserves the original exit code.

In shadow mode (config `shadow_mode: true`), savings are still measured and
recorded, but the original, unfiltered output is printed — the agent sees
exactly what it would have without this mechanism.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import sys

from rclm import _config
from rclm.compress.runner import apply_filter, execute, track_savings
from rclm.hooks.compress import is_safe_session_id


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: rclm-compress [--session-id ID] [--encoded-command VALUE | <command...>]",
            file=sys.stderr,
        )
        sys.exit(1)

    args = sys.argv[1:]
    session_id = None
    if args[0] == "--session-id":
        if len(args) < 3 or not args[1]:
            print("rclm-compress: --session-id requires an ID and a command", file=sys.stderr)
            sys.exit(1)
        session_id = args[1]
        if not is_safe_session_id(session_id):
            print("rclm-compress: invalid --session-id", file=sys.stderr)
            sys.exit(1)
        args = args[2:]
    if args and args[0] == "--encoded-command":
        if len(args) != 2:
            print("rclm-compress: --encoded-command requires one value", file=sys.stderr)
            sys.exit(1)
        try:
            command = base64.b64decode(
                args[1],
                altchars=b"-_",
                validate=True,
            ).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            print("rclm-compress: invalid --encoded-command", file=sys.stderr)
            sys.exit(1)
        if not command:
            print("rclm-compress: encoded command is empty", file=sys.stderr)
            sys.exit(1)
    else:
        command = " ".join(args)

    try:
        stdout, stderr, exit_code = execute(command)
    except Exception as exc:
        print(f"rclm-compress: execution error: {exc}", file=sys.stderr)
        sys.exit(1)

    original = stdout + stderr
    result = apply_filter(command, stdout, stderr, exit_code=exit_code)
    shadow = _config.effective_hook_policy().shadow_for("exec_compaction")

    with contextlib.suppress(Exception):
        track_savings(
            original,
            result.text,
            result.mechanism,
            applied=not shadow,
            session_id=session_id,
        )

    print(original if shadow else result.text, end="")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
