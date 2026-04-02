"""Entry point for rclm-compress CLI.

Usage: rclm-compress <command...>

Executes the command, applies output compression filters, tracks savings,
and prints compressed output. Preserves the original exit code.
"""
from __future__ import annotations

import sys

from rclm.compress.runner import apply_filter, execute, track_savings


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: rclm-compress <command...>", file=sys.stderr)
        sys.exit(1)

    command = " ".join(sys.argv[1:])

    try:
        stdout, stderr, exit_code = execute(command)
    except Exception as exc:
        print(f"rclm-compress: execution error: {exc}", file=sys.stderr)
        sys.exit(1)

    original = stdout + stderr
    compressed = apply_filter(command, stdout, stderr)

    try:
        track_savings(command, original, compressed)
    except Exception:
        pass  # Never let tracking disrupt the command output

    print(compressed, end="")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
