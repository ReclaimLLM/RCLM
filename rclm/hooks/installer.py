"""Merge rclm hooks into Claude Code, Gemini CLI, and/or Codex CLI settings.

Installs globally (home directory) by default. Pass --local to install into
the current project directory instead.

When no provider flag is given, all three providers are installed.

Usage:
    rclm-hooks-install                            # all providers, global
    rclm-hooks-install --local                    # all providers, current dir
    rclm-hooks-install --claude                   # Claude Code only, global
    rclm-hooks-install --gemini                   # Gemini CLI only, global
    rclm-hooks-install --codex                    # Codex CLI only, global
    rclm-hooks-install --claude --codex           # Claude + Codex, global
    rclm-hooks-install --api-key=<key>            # explicit key (skips browser)
    rclm-hooks-install --compress                 # enable compression (Claude only)

Credentials are stored in ~/.reclaimllm/config.json and reused on subsequent runs.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from rclm import _config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_APP_URL = "https://reclaimllm.com"
DEFAULT_SERVER_URL = "https://api.reclaimllm.com"
SETUP_URL = DEFAULT_APP_URL + "/settings"
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

# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------


def _parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install rclm hooks (all providers by default, global by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s                          # all providers, global (~/.claude, ~/.gemini, ~/.codex)
  %(prog)s --local                  # all providers, current project directory
  %(prog)s --claude                 # Claude Code only
  %(prog)s --gemini                 # Gemini CLI only
  %(prog)s --codex                  # Codex CLI only
  %(prog)s --claude --codex         # Claude Code + Codex CLI
  %(prog)s --api-key=<key>          # explicit key (skips browser prompt)
  %(prog)s --compress               # enable compression for Claude Code

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
        help=f"ReclaimLLM server URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Enable context compression for Claude Code (rewrites Bash/Read/Grep inputs to reduce tokens)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _command_already_present(existing_entries: list[dict], command: str) -> bool:
    for entry in existing_entries:
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
    """Merge hooks into a settings.json dict (Claude Code / Gemini format), skipping duplicates."""
    hooks_section: dict = settings.setdefault("hooks", {})
    for event_name, new_entries in hooks_to_inject.items():
        existing_entries: list[dict] = hooks_section.setdefault(event_name, [])
        for entry in new_entries:
            for hook in entry.get("hooks", []):
                command = hook.get("command", "")
                if not _command_already_present(existing_entries, command):
                    existing_entries.append(entry)
                    break
    return settings


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

    settings = _load_json(path)
    if compress_enabled:
        _remove_rtk_hooks(settings)
    _merge_settings_hooks(settings, _CLAUDE_HOOKS_TO_INJECT)
    _write_json(path, settings)
    print(f"rclm hooks installed into {path}")


def _install_gemini(use_global: bool) -> None:
    path = (
        Path.home() / ".gemini" / "settings.json"
        if use_global
        else Path(".gemini") / "settings.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    settings = _load_json(path)
    _merge_settings_hooks(settings, _GEMINI_HOOKS_TO_INJECT)
    _write_json(path, settings)
    print(f"rclm hooks installed into {path}")


def _install_codex(use_global: bool) -> None:
    path = Path.home() / ".codex" / "hooks.json" if use_global else Path(".codex") / "hooks.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = _load_json(path)
    _merge_settings_hooks(data, _CODEX_HOOKS_TO_INJECT)
    _write_json(path, data)
    print(f"rclm hooks installed into {path}")


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


def _app_url_from_server_url(server_url: str) -> str:
    if "api.reclaimllm.com" in server_url:
        return server_url.replace("api.reclaimllm.com", "app.reclaimllm.com").rstrip("/")
    return DEFAULT_APP_URL


def _wait_for_api_key_via_browser(server_url: str) -> str | None:
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

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    app_url = _app_url_from_server_url(server_url)
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

    # Determine which providers to install. Default: all three.
    providers = [p for p in ("claude", "gemini", "codex") if getattr(args, p)]
    if not providers:
        providers = ["claude", "gemini", "codex"]

    use_global = not args.local

    # Resolve credentials.
    saved = _config.load()
    api_key: str | None = args.api_key or saved.get("api_key")
    server_url: str = args.server_url or saved.get("server_url") or DEFAULT_SERVER_URL

    if not api_key:
        api_key = _wait_for_api_key_via_browser(server_url)
        if not api_key:
            sys.exit(1)

    server_url = server_url.replace('"', "").replace("'", "").strip()
    api_key = api_key.replace('"', "").replace("'", "").strip()

    compress_enabled = args.compress or saved.get("compress", False)
    _config.save(server_url, api_key, compress=compress_enabled)

    for provider in providers:
        if provider == "claude":
            _install_claude(use_global, compress_enabled)
        elif provider == "gemini":
            _install_gemini(use_global)
        elif provider == "codex":
            _install_codex(use_global)

    # Non-blocking update check — print a notice if a newer version exists.
    try:
        from rclm.hooks.updater import check_for_update, installed_version

        latest = check_for_update()
        if latest:
            current = installed_version()
            print(f"\n✦ rclm {latest} is available (you have {current}). Run: rclm-update")
    except Exception:
        pass
