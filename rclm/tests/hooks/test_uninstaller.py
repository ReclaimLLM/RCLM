"""Tests for rclm.hooks.uninstaller."""

from __future__ import annotations

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
