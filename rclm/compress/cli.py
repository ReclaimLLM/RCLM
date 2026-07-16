"""Entry point for rclm-compress CLI.

Usage: rclm-compress <command...>

Executes the command, applies output compression filters, tracks savings,
and prints the (possibly compressed) output. Preserves the original exit code.

In shadow mode (config `shadow_mode: true`), savings are still measured and
recorded, but the original, unfiltered output is printed — the agent sees
exactly what it would have without this mechanism.
"""

from __future__ import annotations

import contextlib
import sys

from rclm import _config
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
    result = apply_filter(command, stdout, stderr)
    shadow = _config.load().get("shadow_mode", False)

    with contextlib.suppress(Exception):
        track_savings(original, result.text, result.mechanism, applied=not shadow)

    print(original if shadow else result.text, end="")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
