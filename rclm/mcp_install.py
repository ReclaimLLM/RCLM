"""Install ReclaimLLM MCP server configuration into supported local clients."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_SERVER_NAME = "reclaimllm"


def _resolve_binary() -> str:
    resolved = shutil.which("rclm-mcp")
    if resolved:
        return resolved
    candidate = Path(sys.executable).parent / "rclm-mcp"
    if candidate.exists():
        return str(candidate)
    return "rclm-mcp"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _mcp_json_entry(command: str) -> dict:
    return {"command": command, "args": []}


def _install_json_mcp(path: Path, command: str) -> None:
    data = _load_json(path)
    servers = data.setdefault("mcpServers", {})
    servers[_SERVER_NAME] = _mcp_json_entry(command)
    _write_json(path, data)
    print(f"rclm MCP server installed into {path}")


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _codex_block(command: str) -> str:
    return f"[mcp_servers.{_SERVER_NAME}]\ncommand = {_toml_string(command)}\nargs = []\n"


def _install_codex_mcp(path: Path, command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = existing.splitlines()
    output: list[str] = []
    i = 0
    target_prefix = f"[mcp_servers.{_SERVER_NAME}"
    while i < len(lines):
        if lines[i].strip().startswith(target_prefix):
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    if stripped.startswith(target_prefix):
                        i += 1
                        continue
                    break
                i += 1
            continue
        output.append(lines[i])
        i += 1

    text = "\n".join(output).rstrip()
    if text:
        text += "\n\n"
    text += _codex_block(command)
    path.write_text(text, encoding="utf-8")
    print(f"rclm MCP server installed into {path}")


def install_mcp(use_global: bool = True) -> list[Path]:
    """Install ReclaimLLM MCP config for Claude, Gemini, Cursor, and Codex."""
    command = _resolve_binary()
    root = Path.home() if use_global else Path(".")

    claude_path = root / ".claude" / "settings.json"
    gemini_path = root / ".gemini" / "settings.json"
    cursor_path = root / ".cursor" / "mcp.json"
    codex_path = root / ".codex" / "config.toml"

    _install_json_mcp(claude_path, command)
    _install_json_mcp(gemini_path, command)
    _install_json_mcp(cursor_path, command)
    _install_codex_mcp(codex_path, command)
    return [claude_path, gemini_path, cursor_path, codex_path]
