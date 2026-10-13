"""Hook integrity checks across supported coding agents (global + local cwd)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rclm.hooks.installer import (
    _ANTIGRAVITY_HOOK_NAME,
    _ANTIGRAVITY_HOOKS_TO_INJECT,
    _CLAUDE_HOOKS_TO_INJECT,
    _CODEX_HOOKS_TO_INJECT,
    _COPILOT_HOOKS_TO_INJECT,
    _CURSOR_HOOKS_TO_INJECT,
    _GEMINI_HOOKS_TO_INJECT,
    _is_rclm_command,
)

ClientStatus = str  # installed|partial|missing|disabled|modified|not_present


def collect_client_status(*, cwd: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return per-client integrity status for check-in."""
    root = Path(cwd) if cwd else Path.cwd()
    return {
        "claude": _claude_status(root),
        "cursor": _cursor_status(root),
        "codex": _codex_status(root),
        "gemini": _gemini_status(root),
        "copilot": _copilot_status(root),
        "antigravity": _antigravity_status(root),
    }


def integrity_hash(clients: dict[str, dict[str, Any]]) -> str:
    import hashlib

    payload = json.dumps(clients, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _claude_status(cwd: Path) -> dict[str, Any]:
    global_path = Path.home() / ".claude" / "settings.json"
    local_path = cwd / ".claude" / "settings.json"
    return _settings_json_status(
        client="claude",
        expected=_CLAUDE_HOOKS_TO_INJECT,
        global_path=global_path,
        local_path=local_path,
        disabled_checker=_claude_disabled,
        cwd=cwd,
    )


def _cursor_status(cwd: Path) -> dict[str, Any]:
    global_path = Path.home() / ".cursor" / "hooks.json"
    local_path = cwd / ".cursor" / "hooks.json"
    return _cursor_hooks_status(global_path=global_path, local_path=local_path)


def _codex_status(cwd: Path) -> dict[str, Any]:
    global_hooks = Path.home() / ".codex" / "hooks.json"
    local_hooks = cwd / ".codex" / "hooks.json"
    global_toml = Path.home() / ".codex" / "config.toml"
    local_toml = cwd / ".codex" / "config.toml"

    if _codex_hooks_disabled(global_toml) or _codex_hooks_disabled(local_toml):
        present = global_hooks.exists() or local_hooks.exists() or global_toml.exists()
        return _result(
            "disabled"
            if present or global_toml.exists() or local_toml.parent.exists()
            else "not_present",
            scope="none",
            hint=str(global_toml if global_toml.exists() else local_toml),
        )

    return _settings_json_status(
        client="codex",
        expected=_CODEX_HOOKS_TO_INJECT,
        global_path=global_hooks,
        local_path=local_hooks,
        disabled_checker=None,
        cwd=cwd,
    )


def _gemini_status(cwd: Path) -> dict[str, Any]:
    return _settings_json_status(
        client="gemini",
        expected=_GEMINI_HOOKS_TO_INJECT,
        global_path=Path.home() / ".gemini" / "settings.json",
        local_path=cwd / ".gemini" / "settings.json",
        disabled_checker=None,
        cwd=cwd,
    )


def _copilot_status(cwd: Path) -> dict[str, Any]:
    global_path = Path.home() / ".copilot" / "hooks" / "rclm.json"
    local_path = cwd / ".github" / "hooks" / "rclm.json"
    # VS Code can disable via chat.hookFilesLocations mapping path → false.
    if _copilot_hooks_disabled():
        return _result("disabled", scope="none", hint="chat.hookFilesLocations")
    g = _read_json(global_path)
    local_cfg = _read_json(local_path)
    expected_events = set(_COPILOT_HOOKS_TO_INJECT.keys())
    g_ok, g_partial, g_mod, g_present = _match_copilot(expected_events, g)
    l_ok, l_partial, l_mod, l_present = _match_copilot(expected_events, local_cfg)
    if not g_present and not l_present:
        if not global_path.parent.exists() and not local_path.parent.exists():
            return _result("not_present", scope="none", hint=str(global_path))
        return _result("missing", scope="none", hint=str(global_path))
    if g_mod or l_mod:
        return _result(
            "modified",
            scope=_scope_from_flags(g_ok or g_partial, l_ok or l_partial, g_present, l_present),
            hint=str(global_path if g_mod else local_path),
        )
    if g_ok or l_ok:
        scope = "both" if g_ok and l_ok else ("global" if g_ok else "local")
        return _result("installed", scope=scope, hint=str(global_path if g_ok else local_path))
    if g_partial or l_partial:
        scope = "both" if g_partial and l_partial else ("global" if g_partial else "local")
        return _result(
            "partial",
            scope=scope,
            hint=str(global_path if g_partial else local_path),
        )
    return _result("missing", scope="none", hint=str(global_path))


def _antigravity_status(cwd: Path) -> dict[str, Any]:
    global_path = Path.home() / ".gemini" / "config" / "hooks.json"
    local_path = cwd / ".agents" / "hooks.json"
    expected = _ANTIGRAVITY_HOOKS_TO_INJECT.get(_ANTIGRAVITY_HOOK_NAME) or {}
    g = _read_json(global_path)
    local_cfg = _read_json(local_path)
    g_ok, g_partial, g_mod, g_present = _match_antigravity(expected, g)
    l_ok, l_partial, l_mod, l_present = _match_antigravity(expected, local_cfg)
    if not g_present and not l_present:
        if not global_path.parent.exists() and not local_path.parent.exists():
            return _result("not_present", scope="none", hint=str(global_path))
        return _result("missing", scope="none", hint=str(global_path))
    if g_mod or l_mod:
        return _result(
            "modified",
            scope=_scope_from_flags(g_ok or g_partial, l_ok or l_partial, g_present, l_present),
            hint=str(global_path if g_mod else local_path),
        )
    if g_ok or l_ok:
        scope = "both" if g_ok and l_ok else ("global" if g_ok else "local")
        return _result("installed", scope=scope, hint=str(global_path if g_ok else local_path))
    if g_partial or l_partial:
        scope = "both" if g_partial and l_partial else ("global" if g_partial else "local")
        return _result(
            "partial",
            scope=scope,
            hint=str(global_path if g_partial else local_path),
        )
    return _result("missing", scope="none", hint=str(global_path))


def _match_copilot(expected_events: set[str], cfg: dict | None) -> tuple[bool, bool, bool, bool]:
    if not cfg:
        return False, False, False, False
    hooks = cfg.get("hooks")
    if not isinstance(hooks, dict):
        return False, False, False, True
    present = 0
    modified = False
    for event in expected_events:
        entries = hooks.get(event)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cmd = str(entry.get("command") or "")
            if _is_rclm_command(cmd):
                present += 1
                break
            if "rclm" in cmd.lower():
                modified = True
    if modified and present:
        return False, False, True, True
    if present == 0:
        return False, False, False, True
    if present >= len(expected_events):
        return True, False, False, True
    return False, True, False, True


def _match_antigravity(expected: dict, cfg: dict | None) -> tuple[bool, bool, bool, bool]:
    if not cfg:
        return False, False, False, False
    block = cfg.get(_ANTIGRAVITY_HOOK_NAME)
    if not isinstance(block, dict):
        # File present but no rclm block.
        return False, False, False, True
    expected_events = set(expected.keys())
    present = 0
    for event in expected_events:
        entries = block.get(event)
        if not entries:
            continue
        # Flat command objects or nested hooks lists.
        blob = json.dumps(entries)
        if "rclm-antigravity-hooks" in blob:
            present += 1
    if present == 0:
        return False, False, False, True
    if present >= len(expected_events):
        return True, False, False, True
    return False, True, False, True


def _copilot_hooks_disabled() -> bool:
    """Best-effort: VS Code user settings mapping ~/.copilot/hooks → false."""
    candidates = [
        Path.home() / "Library/Application Support/Code/User/settings.json",
        Path.home() / ".config/Code/User/settings.json",
    ]
    for path in candidates:
        data = _read_json(path)
        if not data:
            continue
        mapping = data.get("chat.hookFilesLocations")
        if not isinstance(mapping, dict):
            continue
        for key, value in mapping.items():
            if "copilot/hooks" in str(key).replace("\\", "/") and value is False:
                return True
    return False


def _settings_json_status(
    *,
    client: str,
    expected: dict[str, list[dict]],
    global_path: Path,
    local_path: Path,
    disabled_checker,
    cwd: Path,
) -> dict[str, Any]:
    global_cfg = _read_json(global_path)
    local_cfg = _read_json(local_path)
    if disabled_checker and (disabled_checker(global_cfg, cwd) or disabled_checker(local_cfg, cwd)):
        return _result("disabled", scope=_scope(global_cfg, local_cfg), hint=str(global_path))

    g_ok, g_partial, g_mod, g_present = _match_nested_hooks(expected, global_cfg)
    l_ok, l_partial, l_mod, l_present = _match_nested_hooks(expected, local_cfg)

    if not g_present and not l_present:
        config_dir = global_path.parent
        if not config_dir.exists() and not local_path.parent.exists():
            return _result("not_present", scope="none", hint=str(global_path))
        return _result("missing", scope="none", hint=str(global_path))

    if g_mod or l_mod:
        return _result(
            "modified",
            scope=_scope_from_flags(g_ok or g_partial, l_ok or l_partial, g_present, l_present),
            hint=str(global_path if g_mod else local_path),
        )
    if g_ok or l_ok:
        scope = "both" if g_ok and l_ok else ("global" if g_ok else "local")
        return _result("installed", scope=scope, hint=str(global_path if g_ok else local_path))
    if g_partial or l_partial:
        scope = "both" if g_partial and l_partial else ("global" if g_partial else "local")
        return _result(
            "partial",
            scope=scope,
            hint=str(global_path if g_partial else local_path),
        )
    return _result("missing", scope="none", hint=str(global_path))


def _cursor_hooks_status(*, global_path: Path, local_path: Path) -> dict[str, Any]:
    expected_events = set(_CURSOR_HOOKS_TO_INJECT.keys())
    g = _read_json(global_path)
    local_cfg = _read_json(local_path)
    g_ok, g_partial, g_mod, g_present = _match_cursor(expected_events, g)
    l_ok, l_partial, l_mod, l_present = _match_cursor(expected_events, local_cfg)

    if not g_present and not l_present:
        if not global_path.parent.exists() and not local_path.parent.exists():
            return _result("not_present", scope="none", hint=str(global_path))
        return _result("missing", scope="none", hint=str(global_path))
    if g_mod or l_mod:
        return _result(
            "modified",
            scope=_scope_from_flags(g_ok or g_partial, l_ok or l_partial, g_present, l_present),
            hint=str(global_path if g_mod else local_path),
        )
    if g_ok or l_ok:
        scope = "both" if g_ok and l_ok else ("global" if g_ok else "local")
        return _result("installed", scope=scope, hint=str(global_path if g_ok else local_path))
    if g_partial or l_partial:
        scope = "both" if g_partial and l_partial else ("global" if g_partial else "local")
        return _result(
            "partial",
            scope=scope,
            hint=str(global_path if g_partial else local_path),
        )
    return _result("missing", scope="none", hint=str(global_path))


def _match_nested_hooks(
    expected: dict[str, list[dict]], cfg: dict | None
) -> tuple[bool, bool, bool, bool]:
    """Return (full_ok, partial, modified, present)."""
    if not cfg:
        return False, False, False, False
    hooks = cfg.get("hooks")
    if not isinstance(hooks, dict):
        return False, False, False, True
    present_events = 0
    expected_events = 0
    modified = False
    for event, _entries in expected.items():
        expected_events += 1
        existing = hooks.get(event)
        if not isinstance(existing, list) or not existing:
            continue
        rclm_cmds = []
        for entry in existing:
            if not isinstance(entry, dict):
                continue
            for hook in entry.get("hooks") or []:
                if not isinstance(hook, dict):
                    continue
                cmd = str(hook.get("command") or "")
                if _is_rclm_command(cmd):
                    rclm_cmds.append(cmd)
                elif "rclm" in cmd.lower():
                    modified = True
        if rclm_cmds:
            present_events += 1
            for cmd in rclm_cmds:
                if not _looks_like_rclm_binary(cmd):
                    modified = True
    if modified and present_events:
        return False, False, True, True
    if present_events == 0:
        return False, False, False, True
    if present_events >= expected_events:
        return True, False, False, True
    return False, True, False, True


def _match_cursor(expected_events: set[str], cfg: dict | None) -> tuple[bool, bool, bool, bool]:
    if not cfg:
        return False, False, False, False
    hooks = cfg.get("hooks")
    if not isinstance(hooks, dict):
        return False, False, False, True
    present = 0
    modified = False
    for event in expected_events:
        entries = hooks.get(event)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cmd = str(entry.get("command") or "")
            if _is_rclm_command(cmd):
                present += 1
                if not _looks_like_rclm_binary(cmd):
                    modified = True
                break
            if "rclm" in cmd.lower():
                modified = True
    if modified and present:
        return False, False, True, True
    if present == 0:
        return False, False, False, True
    if present >= len(expected_events):
        return True, False, False, True
    return False, True, False, True


def _looks_like_rclm_binary(command: str) -> bool:
    # Absolute path or bare binary name containing an rclm *-hooks entry is fine.
    return _is_rclm_command(command)


def _claude_disabled(cfg: dict | None, cwd: Path) -> bool:
    if cfg and cfg.get("disableAllHooks") is True:
        return True
    # Project / local settings overlays.
    for rel in (
        cwd / ".claude" / "settings.json",
        cwd / ".claude" / "settings.local.json",
    ):
        data = _read_json(rel)
        if data and data.get("disableAllHooks") is True:
            return True
    return False


def _codex_hooks_disabled(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    # Minimal TOML scan without requiring tomllib for disabled flag.
    in_features = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            in_features = line.lower() in {"[features]"}
            continue
        if not in_features:
            continue
        key, _, value = line.partition("=")
        key = key.strip().lower()
        value = value.strip().lower()
        if key in {"hooks", "codex_hooks"} and value in {"false", "0"}:
            return True
    return False


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _scope(global_cfg: dict | None, local_cfg: dict | None) -> str:
    if global_cfg and local_cfg:
        return "both"
    if global_cfg:
        return "global"
    if local_cfg:
        return "local"
    return "none"


def _scope_from_flags(global_ok: bool, local_ok: bool, g_present: bool, l_present: bool) -> str:
    if global_ok and local_ok:
        return "both"
    if global_ok:
        return "global"
    if local_ok:
        return "local"
    if g_present and l_present:
        return "both"
    if g_present:
        return "global"
    if l_present:
        return "local"
    return "none"


def _result(status: ClientStatus, *, scope: str, hint: str) -> dict[str, Any]:
    return {"status": status, "scope": scope, "hook_file_hint": hint}
