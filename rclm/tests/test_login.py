"""Tests for rclm.login (rclm-login entrypoint)."""

from __future__ import annotations

import json

import pytest

from rclm import _config, auth, login


def _run_login(monkeypatch, tmp_path, *extra_args):
    monkeypatch.setattr("sys.argv", ["rclm-login", *extra_args])
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")


def test_login_saves_key_on_successful_validation(tmp_path, monkeypatch):
    _run_login(monkeypatch, tmp_path)
    monkeypatch.setattr(auth, "wait_for_api_key_via_browser", lambda _url: "sk-good")

    async def fake_validate(server_url, api_key):
        assert api_key == "sk-good"
        return True

    monkeypatch.setattr(auth, "validate_api_key", fake_validate)

    login.main()

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["api_key"] == "sk-good"


def test_login_exits_1_when_browser_flow_cancelled(tmp_path, monkeypatch):
    _run_login(monkeypatch, tmp_path)
    monkeypatch.setattr(auth, "wait_for_api_key_via_browser", lambda _url: None)

    with pytest.raises(SystemExit) as exc_info:
        login.main()

    assert exc_info.value.code == 1
    assert not (tmp_path / "config.json").exists()


def test_login_does_not_save_rejected_key(tmp_path, monkeypatch):
    _run_login(monkeypatch, tmp_path)
    monkeypatch.setattr(auth, "wait_for_api_key_via_browser", lambda _url: "sk-bad")

    async def fake_validate(server_url, api_key):
        return False

    monkeypatch.setattr(auth, "validate_api_key", fake_validate)

    with pytest.raises(SystemExit) as exc_info:
        login.main()

    assert exc_info.value.code == 1
    assert not (tmp_path / "config.json").exists()


def test_login_saves_key_with_warning_on_network_error(tmp_path, monkeypatch, capsys):
    _run_login(monkeypatch, tmp_path)
    monkeypatch.setattr(auth, "wait_for_api_key_via_browser", lambda _url: "sk-maybe")

    async def fake_validate(server_url, api_key):
        return None

    monkeypatch.setattr(auth, "validate_api_key", fake_validate)

    login.main()

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["api_key"] == "sk-maybe"
    assert "network issue" in capsys.readouterr().err.lower()


def test_login_with_api_key_flag_skips_browser(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["rclm-login", "--api-key=sk-explicit", "--server-url=http://test.example.com"],
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(
        auth,
        "wait_for_api_key_via_browser",
        lambda _url: pytest.fail("browser flow should not run"),
    )

    async def fake_validate(server_url, api_key):
        assert api_key == "sk-explicit"
        assert server_url == "http://test.example.com"
        return True

    monkeypatch.setattr(auth, "validate_api_key", fake_validate)

    login.main()

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["api_key"] == "sk-explicit"
    assert config["server_url"] == "http://test.example.com"
