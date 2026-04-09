"""Top-level rclm CLI entry point."""

from __future__ import annotations

import argparse

from rclm.hooks.updater import installed_version


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rclm",
        description="ReclaimLLM — capture, store, and search your LLM sessions",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"rclm {installed_version()}",
    )
    parser.parse_args()
    parser.print_help()
