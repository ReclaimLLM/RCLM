"""Tests for rclm.hooks.uninstaller."""

from __future__ import annotations

import json

import pytest

from rclm.hooks import uninstaller


def test_openclaw_flag_uninstalls_only_openclaw(monkeypatch):
    removed: list[bool] = []

    def fake_uninstall_openclaw(use_global: bool) -> None:
        removed.append(use_global)

    monkeypatch.setattr(uninstaller, "_uninstall_openclaw", fake_uninstall_openclaw)
    monkeypatch.setattr("sys.argv", ["rclm-hooks-uninstall", "--openclaw"])

    uninstaller.main()

    assert removed == [True]


def test_local_default_does_not_uninstall_openclaw(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(uninstaller, "_uninstall_openclaw", lambda use_global: pytest.fail())
    monkeypatch.setattr("sys.argv", ["rclm-hooks-uninstall", "--local"])

    uninstaller.main()


# ---------------------------------------------------------------------------
# _command_belongs_to_rclm / _is_rclm_hook: must match resolved absolute paths,
# not just the bare binary name. _resolve_binary returns an absolute path (e.g.
# /home/user/.venv/bin/rclm-claude-hooks) whenever the binary is found on PATH,
# which is the common case for a real `pip install rclm`.
# ---------------------------------------------------------------------------


def test_command_belongs_to_rclm_matches_absolute_path():
    assert uninstaller._command_belongs_to_rclm(
        "/home/user/.venv/bin/rclm-claude-hooks SessionStart"
    )


def test_command_belongs_to_rclm_matches_quoted_path_with_spaces():
    assert uninstaller._command_belongs_to_rclm("'/home/user/Local Tools/rclm-cursor-hooks' stop")


def test_command_belongs_to_rclm_matches_bare_name():
    assert uninstaller._command_belongs_to_rclm("rclm-claude-hooks SessionStart")


def test_command_belongs_to_rclm_rejects_foreign_command():
    assert not uninstaller._command_belongs_to_rclm("/usr/local/bin/my-custom-tool")


def test_remove_from_settings_strips_hooks_with_resolved_absolute_paths():
    settings = {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "startup",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "/home/user/.venv/bin/rclm-claude-hooks SessionStart",
                        }
                    ],
                }
            ]
        }
    }
    updated, count = uninstaller._remove_from_settings(settings)
    assert count == 1
    assert "hooks" not in updated


def test_uninstall_cursor_preserves_unrelated_hooks(tmp_path):
    path = tmp_path / "hooks.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "hooks": {
                    "stop": [
                        {"command": "rclm-cursor-hooks stop"},
                        {"command": "my-hook stop"},
                    ]
                },
            }
        )
    )

    uninstaller._uninstall_cursor(path)

    assert json.loads(path.read_text())["hooks"]["stop"] == [{"command": "my-hook stop"}]


def test_uninstall_antigravity_removes_only_owned_namespace(tmp_path):
    path = tmp_path / "hooks.json"
    path.write_text(
        json.dumps(
            {
                "rclm-antigravity-hooks": {"Stop": [{"command": "rclm-antigravity-hooks Stop"}]},
                "company-hook": {"Stop": [{"command": "audit"}]},
            }
        )
    )

    uninstaller._uninstall_antigravity(path)

    assert json.loads(path.read_text()) == {"company-hook": {"Stop": [{"command": "audit"}]}}
