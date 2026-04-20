"""Top-level rclm CLI entry point."""

from __future__ import annotations

import argparse
import sys

from rclm.hooks.updater import installed_version

_VALID_TOOLS = ("claude", "gemini", "codex", "generic")


def _cmd_convert_session(args: argparse.Namespace) -> None:
    from rclm.convert import convert_session

    convert_session(
        args.session_id,
        args.target_tool,
        output_path=args.output,
        include_diffs=args.include_diffs,
        max_diff_lines=args.max_diff_lines,
        force_regenerate=args.force_regenerate,
    )


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

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # ── convert-session ───────────────────────────────────────────────────────
    cs = subparsers.add_parser(
        "convert-session",
        help="Export a session as a context document for another AI tool",
        description=(
            "Fetch a captured session and produce a markdown context document "
            "suitable for continuing work in a different AI coding tool."
        ),
    )
    cs.add_argument("session_id", help="Session UUID (from reclaimllm.com or rclm list)")
    cs.add_argument(
        "target_tool",
        choices=_VALID_TOOLS,
        help="Target tool format: claude (CLAUDE.md), gemini (.gemini), codex (AGENTS.md), generic",
    )
    cs.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        default=None,
        help=(
            "Write output to FILE instead of stdout. "
            "Defaults: CLAUDE.md, .gemini, AGENTS.md, context.md depending on target_tool."
        ),
    )
    cs.add_argument(
        "--no-diffs",
        dest="include_diffs",
        action="store_false",
        default=True,
        help="Omit unified diffs from the output",
    )
    cs.add_argument(
        "--max-diff-lines",
        type=int,
        default=50,
        metavar="N",
        help="Maximum lines of diff to include per file (default: 50, range: 10-200)",
    )
    cs.add_argument(
        "--force-regenerate",
        action="store_true",
        default=False,
        help="Force LLM regeneration even when existing annotations are available",
    )
    cs.set_defaults(func=_cmd_convert_session)

    # ─────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(0)
