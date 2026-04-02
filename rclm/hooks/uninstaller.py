"""Remove rclm hooks from Claude Code, Gemini CLI, and/or Codex CLI settings.

Removes any hook entries whose command starts with ``rclm-``.
All other hooks are left untouched.

When no provider flag is given, all three providers are targeted.
Targets the global home-directory config by default; pass --local for the
current project directory.

Usage:
    rclm-hooks-uninstall                 # all providers, global
    rclm-hooks-uninstall --local         # all providers, current dir
    rclm-hooks-uninstall --claude        # Claude Code only
    rclm-hooks-uninstall --gemini        # Gemini CLI only
    rclm-hooks-uninstall --codex         # Codex CLI only
    rclm-hooks-uninstall --purge-config  # also delete ~/.reclaimllm/config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rclm import _config

# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------


def _parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove rclm hooks (all providers by default, global by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s                    # all providers, global
  %(prog)s --local            # all providers, current project directory
  %(prog)s --claude           # Claude Code only
  %(prog)s --gemini           # Gemini CLI only
  %(prog)s --codex            # Codex CLI only
  %(prog)s --purge-config     # also delete ~/.reclaimllm/config.json""",
    )

    parser.add_argument(
        "--claude",
        action="store_true",
        help="Target Claude Code settings",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Target Gemini CLI settings",
    )
    parser.add_argument(
        "--codex",
        action="store_true",
        help="Target Codex CLI hooks.json",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Target the current project directory instead of the home directory",
    )
    parser.add_argument(
        "--purge-config",
        action="store_true",
        help="Also delete ~/.reclaimllm/config.json (removes saved API key and server URL)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Removal helpers
# ---------------------------------------------------------------------------


def _is_rclm_hook(entry: dict) -> bool:
    """Return True if every command in this entry belongs to rclm.

    Matches both the current prefix (rclm-) and the legacy prefix
    (rclm-) so stale entries from older installs are cleaned up.
    """
    hooks = entry.get("hooks", [])
    if not hooks:
        return False
    return all(hook.get("command", "").strip().startswith(("rclm-", "rclm-")) for hook in hooks)


def _remove_from_settings(settings: dict) -> tuple[dict, int]:
    """Strip rclm entries from a settings.json dict (Claude Code / Gemini format)."""
    hooks_section: dict = settings.get("hooks", {})
    total_removed = 0

    for event_name, entries in list(hooks_section.items()):
        before = len(entries)
        hooks_section[event_name] = [e for e in entries if not _is_rclm_hook(e)]
        total_removed += before - len(hooks_section[event_name])
        if not hooks_section[event_name]:
            del hooks_section[event_name]

    if not hooks_section:
        settings.pop("hooks", None)

    return settings, total_removed


# ---------------------------------------------------------------------------
# Per-provider uninstall helpers
# ---------------------------------------------------------------------------


def _uninstall_settings_provider(path: Path) -> None:
    """Uninstall rclm hooks from a settings.json-format file (Claude Code or Gemini)."""
    if not path.exists():
        print(f"Nothing to do — {path} does not exist.")
        return

    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError:
        print(
            f"Warning: {path} contains invalid JSON — cannot safely modify it.",
            file=sys.stderr,
        )
        return

    updated, count = _remove_from_settings(data)
    if count == 0:
        print(f"No rclm hooks found in {path}.")
    else:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(updated, fh, indent=2)
            fh.write("\n")
        print(f"Removed {count} rclm hook entr{'y' if count == 1 else 'ies'} from {path}.")


def _uninstall_codex(path: Path) -> None:
    """Uninstall rclm hooks from a Codex hooks.json file (same nested format as Claude/Gemini)."""
    _uninstall_settings_provider(path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_flags()

    providers = [p for p in ("claude", "gemini", "codex") if getattr(args, p)]
    if not providers:
        providers = ["claude", "gemini", "codex"]

    use_global = not args.local

    for provider in providers:
        if provider == "claude":
            path = (
                Path.home() / ".claude" / "settings.json"
                if use_global
                else Path(".claude") / "settings.json"
            )
            _uninstall_settings_provider(path)
        elif provider == "gemini":
            path = (
                Path.home() / ".gemini" / "settings.json"
                if use_global
                else Path(".gemini") / "settings.json"
            )
            _uninstall_settings_provider(path)
        elif provider == "codex":
            path = (
                Path.home() / ".codex" / "hooks.json"
                if use_global
                else Path(".codex") / "hooks.json"
            )
            _uninstall_codex(path)

    if args.purge_config:
        _purge_config()


def _purge_config() -> None:
    config_path = _config.CONFIG_PATH
    if config_path.exists():
        config_path.unlink()
        print(f"Deleted {config_path}.")
    else:
        print(f"Config file {config_path} does not exist — nothing to delete.")
