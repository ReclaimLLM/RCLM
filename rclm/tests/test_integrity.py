from __future__ import annotations

import json
from pathlib import Path

from rclm.hooks import integrity
from rclm.hooks.installer import _CLAUDE_HOOKS_TO_INJECT, _CURSOR_HOOKS_TO_INJECT


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_cursor_local_installed_not_missing(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "repo"
    monkeypatch.setattr(Path, "home", lambda: home)
    _write(
        cwd / ".cursor" / "hooks.json",
        {
            "version": 1,
            "hooks": {
                event: [{"command": f"rclm-cursor-hooks {event}"}]
                for event in _CURSOR_HOOKS_TO_INJECT
            },
        },
    )
    status = integrity.collect_client_status(cwd=cwd)["cursor"]
    assert status["status"] == "installed"
    assert status["scope"] == "local"


def test_claude_disabled(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "repo"
    monkeypatch.setattr(Path, "home", lambda: home)
    _write(
        home / ".claude" / "settings.json",
        {"hooks": dict(_CLAUDE_HOOKS_TO_INJECT), "disableAllHooks": True},
    )
    status = integrity.collect_client_status(cwd=cwd)["claude"]
    assert status["status"] == "disabled"


def test_not_present_when_no_config_dirs(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "repo"
    home.mkdir()
    cwd.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    status = integrity.collect_client_status(cwd=cwd)
    assert status["cursor"]["status"] == "not_present"
