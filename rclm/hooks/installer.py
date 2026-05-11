"""Merge rclm hooks into Claude Code, Gemini CLI, Codex CLI, and/or OpenClaw settings.

Installs globally (home directory) by default. Pass --local to install into
the current project directory instead.

When no provider flag is given, all providers are installed.

Usage:
    rclm-hooks-install                            # all providers, global
    rclm-hooks-install --local                    # file-based providers, current dir
    rclm-hooks-install --claude                   # Claude Code only, global
    rclm-hooks-install --gemini                   # Gemini CLI only, global
    rclm-hooks-install --codex                    # Codex CLI only, global
    rclm-hooks-install --openclaw                 # OpenClaw only, global
    rclm-hooks-install --claude --codex           # Claude + Codex, global
    rclm-hooks-install --api-key=<key>            # explicit key (skips browser)
    rclm-hooks-install --compress                 # enable compression (Claude only)
    rclm-hooks-install --with-mcp                 # also install ReclaimLLM MCP server

Credentials are stored in ~/.reclaimllm/config.json and reused on subsequent runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import secrets
import shutil
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from rclm import _config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_FRONTEND_URL = "https://reclaimllm.com"
DEFAULT_BACKEND_SERVER_URL = "https://api.reclaimllm.com"
SETUP_URL = DEFAULT_FRONTEND_URL + "/settings"
_CALLBACK_TIMEOUT_S = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Hook command definitions
# ---------------------------------------------------------------------------

_CLAUDE_HOOKS_TO_INJECT: dict[str, list[dict]] = {
    "SessionStart": [
        {
            "matcher": "startup",
            "hooks": [{"type": "command", "command": "rclm-claude-hooks SessionStart"}],
        },
        {
            "matcher": "resume",
            "hooks": [{"type": "command", "command": "rclm-claude-hooks SessionStart"}],
        },
    ],
    "PreToolUse": [
        {
            "matcher": "",
            "hooks": [{"type": "command", "command": "rclm-claude-hooks PreToolUse"}],
        }
    ],
    "PostToolUse": [
        {
            "matcher": "",
            "hooks": [{"type": "command", "command": "rclm-claude-hooks PostToolUse"}],
        }
    ],
    "UserPromptSubmit": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": "rclm-claude-hooks UserPromptSubmit",
                }
            ]
        }
    ],
    "Stop": [{"hooks": [{"type": "command", "command": "rclm-claude-hooks Stop"}]}],
    "SubagentStop": [{"hooks": [{"type": "command", "command": "rclm-claude-hooks SubagentStop"}]}],
}

_GEMINI_HOOKS_TO_INJECT: dict[str, list[dict]] = {
    "SessionStart": [{"hooks": [{"type": "command", "command": "rclm-gemini-hooks SessionStart"}]}],
    "BeforeAgent": [{"hooks": [{"type": "command", "command": "rclm-gemini-hooks BeforeAgent"}]}],
    "AfterAgent": [{"hooks": [{"type": "command", "command": "rclm-gemini-hooks AfterAgent"}]}],
    "AfterTool": [
        {
            "matcher": "",
            "hooks": [{"type": "command", "command": "rclm-gemini-hooks AfterTool"}],
        }
    ],
    "SessionEnd": [{"hooks": [{"type": "command", "command": "rclm-gemini-hooks SessionEnd"}]}],
}

# Codex hooks.json uses the same nested format as Claude Code settings.json.
_CODEX_HOOKS_TO_INJECT: dict[str, list[dict]] = {
    "SessionStart": [{"hooks": [{"type": "command", "command": "rclm-codex-hooks SessionStart"}]}],
    "UserPromptSubmit": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": "rclm-codex-hooks UserPromptSubmit",
                }
            ]
        }
    ],
    "PreToolUse": [
        {
            "matcher": "Bash",
            "hooks": [{"type": "command", "command": "rclm-codex-hooks PreToolUse"}],
        }
    ],
    "PostToolUse": [
        {
            "matcher": "Bash",
            "hooks": [{"type": "command", "command": "rclm-codex-hooks PostToolUse"}],
        }
    ],
    "Stop": [{"hooks": [{"type": "command", "command": "rclm-codex-hooks Stop"}]}],
}

# Cursor hooks.json format: {"version": 1, "hooks": {"event": [{"command": "..."}]}}
_CURSOR_HOOKS_TO_INJECT: dict[str, list[dict]] = {
    "preToolUse": [{"command": "rclm-cursor-hooks preToolUse"}],
    "postToolUse": [{"command": "rclm-cursor-hooks postToolUse"}],
    "postToolUseFailure": [{"command": "rclm-cursor-hooks postToolUseFailure"}],
    "subagentStart": [{"command": "rclm-cursor-hooks subagentStart"}],
    "subagentStop": [{"command": "rclm-cursor-hooks subagentStop"}],
    "beforeSubmitPrompt": [{"command": "rclm-cursor-hooks beforeSubmitPrompt"}],
    "beforeShellExecution": [{"command": "rclm-cursor-hooks beforeShellExecution"}],
    "beforeMCPExecution": [{"command": "rclm-cursor-hooks beforeMCPExecution"}],
    "afterShellExecution": [{"command": "rclm-cursor-hooks afterShellExecution"}],
    "afterMCPExecution": [{"command": "rclm-cursor-hooks afterMCPExecution"}],
    "afterFileEdit": [{"command": "rclm-cursor-hooks afterFileEdit"}],
    "beforeReadFile": [{"command": "rclm-cursor-hooks beforeReadFile"}],
    "beforeTabFileRead": [{"command": "rclm-cursor-hooks beforeTabFileRead"}],
    "afterTabFileEdit": [{"command": "rclm-cursor-hooks afterTabFileEdit"}],
    "afterAgentResponse": [{"command": "rclm-cursor-hooks afterAgentResponse"}],
    "afterAgentThought": [{"command": "rclm-cursor-hooks afterAgentThought"}],
    "stop": [{"command": "rclm-cursor-hooks stop"}],
    "sessionStart": [{"command": "rclm-cursor-hooks sessionStart"}],
    "sessionEnd": [{"command": "rclm-cursor-hooks sessionEnd"}],
    "preCompact": [{"command": "rclm-cursor-hooks preCompact"}],
}

# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------


def _parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install rclm hooks (all providers by default, global by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s                          # all providers, global (~/.claude, ~/.gemini, ~/.codex, ~/.openclaw, ~/.cursor)
  %(prog)s --local                  # file-based providers, current project directory
  %(prog)s --claude                 # Claude Code only
  %(prog)s --gemini                   # Gemini CLI only
  %(prog)s --codex                  # Codex CLI only
  %(prog)s --cursor                 # Cursor only
  %(prog)s --openclaw               # OpenClaw only
  %(prog)s --claude --cursor         # Claude Code + Cursor
  %(prog)s --api-key=<key>          # explicit key (skips browser prompt)
  %(prog)s --compress               # enable compression for Claude Code
  %(prog)s --with-mcp               # also register the ReclaimLLM MCP server

Subsequent installs without --api-key reuse the saved config.""",
    )

    parser.add_argument(
        "--claude",
        action="store_true",
        help="Install hooks for Claude Code (one of possibly several providers)",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Install hooks for Gemini CLI",
    )
    parser.add_argument(
        "--codex",
        action="store_true",
        help="Install hooks for OpenAI Codex CLI",
    )
    parser.add_argument(
        "--cursor",
        action="store_true",
        help="Install hooks for Cursor IDE",
    )
    parser.add_argument(
        "--openclaw",
        action="store_true",
        help="Install plugin hooks for OpenClaw",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Install into the current project directory instead of the home directory",
    )
    parser.add_argument(
        "--api-key",
        help="API key for ReclaimLLM server authentication",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help=f"ReclaimLLM server URL (default: {DEFAULT_BACKEND_SERVER_URL})",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Enable context compression for Claude Code (rewrites Bash/Read/Grep inputs to reduce tokens)",
    )
    parser.add_argument(
        "--dlp",
        action="store_true",
        help="Enable Data Loss Prevention: redact secrets from .env files before they reach the model",
    )
    parser.add_argument(
        "--with-mcp",
        action="store_true",
        help="Also install the ReclaimLLM MCP server into supported local MCP clients",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _resolve_binary(name: str) -> str:
    """Return absolute path of a hook binary, falling back to bare name if not found."""
    resolved = shutil.which(name)
    if resolved:
        return resolved
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    return name


def _with_absolute_binary(
    hooks_to_inject: dict, binary_name: str, resolved: str, is_cursor: bool = False
) -> dict:
    """Return a deep copy of hooks_to_inject with bare binary name replaced by absolute path."""
    if resolved == binary_name:
        return hooks_to_inject
    result = copy.deepcopy(hooks_to_inject)
    if is_cursor:
        for entries in result.values():
            for hook in entries:
                cmd = hook.get("command", "")
                if cmd == binary_name or cmd.startswith(binary_name + " "):
                    hook["command"] = resolved + cmd[len(binary_name) :]
    else:
        for entries in result.values():
            for entry in entries:
                for hook in entry.get("hooks", []):
                    cmd = hook.get("command", "")
                    if cmd == binary_name or cmd.startswith(binary_name + " "):
                        hook["command"] = resolved + cmd[len(binary_name) :]
    return result


def _is_rclm_command(command: str) -> bool:
    """Check if a command is an rclm hook command, regardless of path or spaces."""
    if not command:
        return False
    # More robust check that handles spaces in paths by looking for the known binary names.
    for name in ["rclm-claude-hooks", "rclm-gemini-hooks", "rclm-codex-hooks", "rclm-cursor-hooks"]:
        if name in command:
            return True
    return False


def _command_already_present(existing_entries: list[dict], command: str, matcher: str = "") -> bool:
    """Check if a command is already registered for the given matcher."""
    for entry in existing_entries:
        if entry.get("matcher", "") != matcher:
            continue
        for hook in entry.get("hooks", []):
            if hook.get("command") == command:
                return True
    return False


def _is_rtk_entry(entry: dict) -> bool:
    for hook in entry.get("hooks", []):
        cmd = hook.get("command", "")
        if "rtk " in cmd or cmd.strip().startswith("rtk"):
            return True
    return False


def _remove_rtk_hooks(settings: dict) -> dict:
    """Remove RTK hook entries from PreToolUse since rclm-compress replaces it."""
    hooks_section = settings.get("hooks", {})
    pre_tool_entries = hooks_section.get("PreToolUse", [])
    if pre_tool_entries:
        hooks_section["PreToolUse"] = [
            entry for entry in pre_tool_entries if not _is_rtk_entry(entry)
        ]
    return settings


def _merge_settings_hooks(settings: dict, hooks_to_inject: dict) -> dict:
    """Merge hooks into a settings.json dict (Claude Code / Gemini format), replacing ALL existing rclm hooks for a matcher."""
    hooks_section: dict = settings.setdefault("hooks", {})
    for event_name, new_entries in hooks_to_inject.items():
        existing_entries: list[dict] = hooks_section.setdefault(event_name, [])
        for new_entry in new_entries:
            matcher = new_entry.get("matcher", "")
            # Assume each rclm entry has exactly one hook in the list based on definitions above.
            new_hook = new_entry.get("hooks", [{}])[0]
            new_command = new_hook.get("command", "")

            # 1. Identify all entries that contain rclm commands for this matcher
            rclm_indices = []
            for i, entry in enumerate(existing_entries):
                if entry.get("matcher", "") != matcher:
                    continue
                if any(_is_rclm_command(h.get("command", "")) for h in entry.get("hooks", [])):
                    rclm_indices.append(i)

            if rclm_indices:
                # Update the FIRST one found with the new command/path
                first_idx = rclm_indices[0]
                for hook in existing_entries[first_idx].get("hooks", []):
                    if _is_rclm_command(hook.get("command", "")):
                        hook["command"] = new_command

                # REMOVE any subsequent rclm entries for the same matcher to clean up duplicates
                for idx in reversed(rclm_indices[1:]):
                    existing_entries.pop(idx)
            else:
                # Add new if none existed
                existing_entries.append(new_entry)
    return settings


def _merge_cursor_hooks(data: dict, hooks_to_inject: dict) -> dict:
    """Merge hooks into a Cursor hooks.json dict, replacing ALL existing rclm hooks for an event."""
    data.setdefault("version", 1)
    hooks_section: dict = data.setdefault("hooks", {})
    for event_name, new_entries in hooks_to_inject.items():
        existing_entries: list[dict] = hooks_section.setdefault(event_name, [])
        for new_hook in new_entries:
            new_command = new_hook.get("command", "")

            # Identify all existing rclm hooks for this event
            rclm_indices = [
                i for i, h in enumerate(existing_entries) if _is_rclm_command(h.get("command", ""))
            ]

            if rclm_indices:
                # Update the first one
                existing_entries[rclm_indices[0]]["command"] = new_command
                # Remove any duplicates
                for idx in reversed(rclm_indices[1:]):
                    existing_entries.pop(idx)
            else:
                existing_entries.append(new_hook)
    return data


# ---------------------------------------------------------------------------
# Per-provider install helpers
# ---------------------------------------------------------------------------


def _install_claude(use_global: bool, compress_enabled: bool) -> None:
    path = (
        Path.home() / ".claude" / "settings.json"
        if use_global
        else Path(".claude") / "settings.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    binary = _resolve_binary("rclm-claude-hooks")
    hooks = _with_absolute_binary(_CLAUDE_HOOKS_TO_INJECT, "rclm-claude-hooks", binary)

    settings = _load_json(path)
    if compress_enabled:
        _remove_rtk_hooks(settings)
    _merge_settings_hooks(settings, hooks)
    _write_json(path, settings)
    print(f"rclm hooks installed into {path}")


def _install_gemini(use_global: bool) -> None:
    path = (
        Path.home() / ".gemini" / "settings.json"
        if use_global
        else Path(".gemini") / "settings.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    binary = _resolve_binary("rclm-gemini-hooks")
    hooks = _with_absolute_binary(_GEMINI_HOOKS_TO_INJECT, "rclm-gemini-hooks", binary)

    settings = _load_json(path)
    _merge_settings_hooks(settings, hooks)
    _write_json(path, settings)
    print(f"rclm hooks installed into {path}")


def _install_codex(use_global: bool) -> None:
    path = Path.home() / ".codex" / "hooks.json" if use_global else Path(".codex") / "hooks.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    binary = _resolve_binary("rclm-codex-hooks")
    hooks = _with_absolute_binary(_CODEX_HOOKS_TO_INJECT, "rclm-codex-hooks", binary)

    data = _load_json(path)
    _merge_settings_hooks(data, hooks)
    _write_json(path, data)
    print(f"rclm hooks installed into {path}")


def _install_cursor(use_global: bool) -> None:
    path = Path.home() / ".cursor" / "hooks.json" if use_global else Path(".cursor") / "hooks.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    binary = _resolve_binary("rclm-cursor-hooks")
    hooks = _with_absolute_binary(
        _CURSOR_HOOKS_TO_INJECT, "rclm-cursor-hooks", binary, is_cursor=True
    )

    data = _load_json(path)
    _merge_cursor_hooks(data, hooks)
    _write_json(path, data)
    print(f"rclm hooks installed into {path}")


def _install_openclaw(use_global: bool) -> None:
    if not use_global:
        print(
            "Warning: OpenClaw plugin hooks are installed globally; ignoring --local for OpenClaw.",
            file=sys.stderr,
        )
    from rclm.hooks.openclaw_plugin import install_plugin

    path = install_plugin(use_global=True)
    print(f"rclm OpenClaw plugin installed into {path}")


# ---------------------------------------------------------------------------
# JSON I/O helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        backup = path.with_suffix(".bak")
        try:
            path.rename(backup)
            print(
                f"Warning: {path} contained invalid JSON; backed up to {backup}.",
                file=sys.stderr,
            )
        except OSError:
            print(
                f"Warning: {path} contains invalid JSON and could not be backed up; overwriting.",
                file=sys.stderr,
            )
        return {}


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Browser key flow
# ---------------------------------------------------------------------------


def _wait_for_api_key_via_browser(app_url: str) -> str | None:
    """Open the settings page in a browser and wait for the user to POST back an API key.

    A one-time nonce is embedded in the callback path so that only the rclm
    app (which receives the full URL) can successfully POST to the local server.
    Any request to a different path is rejected with 404.
    """
    received_key: list[str] = []
    nonce = secrets.token_urlsafe(16)

    class _Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            if self.path != f"/{nonce}":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self._cors_headers()
            self.end_headers()

        def do_POST(self) -> None:
            if self.path != f"/{nonce}":
                self.send_response(404)
                self.end_headers()
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                key = body.get("api_key", "").strip()
                if key:
                    received_key.append(key)
            except Exception:
                pass
            self.send_response(200)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def _cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            # Chrome 98+ (Private Network Access spec): HTTPS pages need this
            # header in the preflight response to be allowed to POST to localhost.
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    settings_url = f"{app_url}/settings?cli_callback=http://localhost:{port}/{nonce}"

    print(
        "No API key configured. Opening browser to create one...\n"
        f"  {settings_url}\n\n"
        "Waiting for key from browser... (Ctrl+C to cancel)\n",
        file=sys.stderr,
    )
    webbrowser.open(settings_url)

    deadline = time.monotonic() + _CALLBACK_TIMEOUT_S
    server.timeout = 1.0
    try:
        while not received_key and time.monotonic() < deadline:
            server.handle_request()
    except KeyboardInterrupt:
        print("\nCancelled. To install manually, run:", file=sys.stderr)
        print("  rclm-hooks-install --api-key=<your-key>", file=sys.stderr)
        return None
    finally:
        server.server_close()

    if not received_key:
        print(
            f"Timed out waiting for API key.\n"
            f"Visit {app_url}/settings to create a key, then run:\n"
            "  rclm-hooks-install --api-key=<your-key>",
            file=sys.stderr,
        )
        return None

    return received_key[0]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_flags()

    # Determine which providers to install. --local defaults to file-based providers.
    providers = [p for p in ("claude", "gemini", "codex", "cursor", "openclaw") if getattr(args, p)]
    if not providers:
        providers = (
            ["claude", "gemini", "codex", "cursor"]
            if args.local
            else [
                "claude",
                "gemini",
                "codex",
                "cursor",
                "openclaw",
            ]
        )

    use_global = not args.local

    # Resolve credentials.
    saved = _config.load()
    api_key: str | None = args.api_key or saved.get("api_key")
    server_url: str = args.server_url or saved.get("server_url") or DEFAULT_BACKEND_SERVER_URL

    if not api_key:
        api_key = _wait_for_api_key_via_browser(DEFAULT_FRONTEND_URL)
        if not api_key:
            sys.exit(1)

    server_url = server_url.replace('"', "").replace("'", "").strip()
    api_key = api_key.replace('"', "").replace("'", "").strip()

    compress_enabled = args.compress or saved.get("compress", False)
    dlp_enabled = args.dlp or saved.get("dlp", False)
    _config.save(server_url, api_key, compress=compress_enabled, dlp=dlp_enabled)

    try:
        from rclm.hooks.redaction import sync_remote_settings

        if sync_remote_settings(server_url=server_url, api_key=api_key):
            print("rclm redaction settings synced")
    except Exception:
        pass  # Never let redaction settings sync disrupt hook installation.

    for provider in providers:
        if provider == "claude":
            _install_claude(use_global, compress_enabled)
        elif provider == "gemini":
            _install_gemini(use_global)
        elif provider == "codex":
            _install_codex(use_global)
        elif provider == "cursor":
            _install_cursor(use_global)
        elif provider == "openclaw":
            _install_openclaw(use_global)

    if args.with_mcp:
        try:
            from rclm.mcp_install import install_mcp

            install_mcp(use_global=use_global)
        except Exception as exc:
            print(
                f"Warning: MCP install failed ({exc}). Hooks were installed.",
                file=sys.stderr,
            )

    # Offer to sync existing sessions from all installed providers.
    try:
        from rclm.hooks.historical_sync import prompt_and_run_sync

        prompt_and_run_sync(providers, resync=True)
    except Exception:
        pass  # Never let sync failure disrupt the install.

    # Non-blocking update check — print a notice if a newer version exists.
    try:
        from rclm.hooks.updater import check_for_update, installed_version

        latest = check_for_update()
        if latest:
            current = installed_version()
            print(f"\n✦ rclm {latest} is available (you have {current}). Run: rclm-update")
    except Exception:
        pass
