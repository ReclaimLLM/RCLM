"""Remove rclm hooks from supported coding-agent settings.

Removes any hook entries whose command starts with ``rclm-``.
All other hooks are left untouched.

When no provider flag is given, all providers are targeted.
Targets the global home-directory config by default; pass --local for the
current project directory.

Usage:
    rclm-hooks-uninstall                 # all providers, global
    rclm-hooks-uninstall --local         # all providers, current dir
    rclm-hooks-uninstall --claude        # Claude Code only
    rclm-hooks-uninstall --gemini        # Gemini CLI only
    rclm-hooks-uninstall --codex         # Codex CLI only
    rclm-hooks-uninstall --cursor        # Cursor only
    rclm-hooks-uninstall --antigravity   # Antigravity only
    rclm-hooks-uninstall --openclaw      # OpenClaw only
    rclm-hooks-uninstall --purge-config  # also delete ~/.reclaimllm/config.json
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
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
  %(prog)s --cursor           # Cursor only
  %(prog)s --antigravity      # Antigravity only
  %(prog)s --openclaw         # OpenClaw only
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
        "--openclaw",
        action="store_true",
        help="Target OpenClaw plugin hooks",
    )
    parser.add_argument("--cursor", action="store_true", help="Target Cursor hooks.json")
    parser.add_argument("--antigravity", action="store_true", help="Target Antigravity hooks.json")
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


def _command_belongs_to_rclm(command: str) -> bool:
    """Return True if a hook/statusLine command invokes an rclm-* binary.

    Installers resolve binaries to an absolute path via shutil.which when found on
    PATH (e.g. /home/user/.venv/bin/rclm-claude-hooks), so this checks the basename
    of the first token rather than a literal string prefix on the full command.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = []
    first_token = tokens[0] if tokens else ""
    return Path(first_token).name.startswith("rclm-")


def _is_rclm_hook(entry: dict) -> bool:
    """Return True if every command in this entry belongs to rclm."""
    hooks = entry.get("hooks", [])
    if not hooks:
        return False
    return all(_command_belongs_to_rclm(hook.get("command", "")) for hook in hooks)


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

    total_removed += _remove_statusline(settings)

    return settings, total_removed


def _remove_statusline(settings: dict) -> int:
    """Strip an rclm-installed statusLine, restoring any backed-up prior one.

    No-op for providers (Gemini, Codex) that never had a statusLine key written.
    """
    status_line = settings.get("statusLine")
    if not isinstance(status_line, dict) or not _command_belongs_to_rclm(
        str(status_line.get("command", ""))
    ):
        return 0

    saved = _config.load()
    backup = saved.get("statusline_backup")
    if backup:
        settings["statusLine"] = backup
        _config.patch(statusline_backup=None)
    else:
        settings.pop("statusLine", None)
    return 1


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
        _write_json(path, updated)
        print(f"Removed {count} rclm hook entr{'y' if count == 1 else 'ies'} from {path}.")


def _uninstall_codex(path: Path) -> None:
    """Uninstall rclm hooks from a Codex hooks.json file (same nested format as Claude/Gemini)."""
    _uninstall_settings_provider(path)


def _uninstall_cursor(path: Path) -> None:
    """Remove direct command entries from Cursor's event-keyed hooks object."""
    if not path.exists():
        print(f"Nothing to do — {path} does not exist.")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"Warning: {path} contains invalid JSON — cannot safely modify it.", file=sys.stderr)
        return
    hooks = data.get("hooks")
    removed = 0
    if isinstance(hooks, dict):
        for event_name, entries in list(hooks.items()):
            if not isinstance(entries, list):
                continue
            kept = [
                entry
                for entry in entries
                if not (
                    isinstance(entry, dict)
                    and _command_belongs_to_rclm(str(entry.get("command", "")))
                )
            ]
            removed += len(entries) - len(kept)
            if kept:
                hooks[event_name] = kept
            else:
                del hooks[event_name]
        if not hooks:
            data.pop("hooks", None)
    if removed:
        _write_json(path, data)
        print(f"Removed {removed} rclm hook entr{'y' if removed == 1 else 'ies'} from {path}.")
    else:
        print(f"No rclm hooks found in {path}.")


def _uninstall_antigravity(path: Path) -> None:
    """Remove the single Antigravity hook namespace owned by ReclaimLLM."""
    if not path.exists():
        print(f"Nothing to do — {path} does not exist.")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"Warning: {path} contains invalid JSON — cannot safely modify it.", file=sys.stderr)
        return
    removed = data.pop("rclm-antigravity-hooks", None) is not None
    if removed:
        _write_json(path, data)
        print(f"Removed rclm Antigravity hooks from {path}.")
    else:
        print(f"No rclm hooks found in {path}.")


def _uninstall_openclaw(use_global: bool) -> None:
    if not use_global:
        print(
            "Warning: OpenClaw plugin hooks are installed globally; ignoring --local for OpenClaw.",
            file=sys.stderr,
        )
    from rclm.hooks.openclaw_plugin import uninstall_plugin

    removed_files, removed_config = uninstall_plugin(use_global=True)
    if removed_files or removed_config:
        print("Removed rclm OpenClaw plugin hooks.")
    else:
        print("No rclm OpenClaw plugin hooks found.")


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(temporary, path)
    os.chmod(path, 0o600)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_flags()

    providers = [
        p
        for p in ("claude", "gemini", "codex", "cursor", "antigravity", "openclaw")
        if getattr(args, p)
    ]
    if not providers:
        providers = (
            ["claude", "gemini", "codex", "cursor", "antigravity"]
            if args.local
            else [
                "claude",
                "gemini",
                "codex",
                "cursor",
                "antigravity",
                "openclaw",
            ]
        )

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
        elif provider == "cursor":
            path = (
                Path.home() / ".cursor" / "hooks.json"
                if use_global
                else Path(".cursor") / "hooks.json"
            )
            _uninstall_cursor(path)
        elif provider == "antigravity":
            path = (
                Path.home() / ".gemini" / "config" / "hooks.json"
                if use_global
                else Path(".agents") / "hooks.json"
            )
            _uninstall_antigravity(path)
        elif provider == "openclaw":
            _uninstall_openclaw(use_global)

    if args.purge_config:
        _purge_config()


def _purge_config() -> None:
    config_path = _config.CONFIG_PATH
    if config_path.exists():
        config_path.unlink()
        print(f"Deleted {config_path}.")
    else:
        print(f"Config file {config_path} does not exist — nothing to delete.")
