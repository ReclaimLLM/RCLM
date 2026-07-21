"""Tests for rclm.hooks.statusline_handler and its installer/uninstaller wiring."""

import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pytest

from rclm import _config
from rclm.hooks import installer, statusline_handler, uninstaller

# ---------------------------------------------------------------------------
# Handler: rendering logic
# ---------------------------------------------------------------------------


def _no_color(monkeypatch):
    monkeypatch.setattr(statusline_handler, "_colors_enabled", lambda: False)


def test_render_context_shows_bar_and_percentage(monkeypatch):
    _no_color(monkeypatch)
    line = statusline_handler.render_status_line({"context_window": {"used_percentage": 58}})
    assert "ctx" in line
    assert "58%" in line


def test_render_context_absent_when_missing(monkeypatch):
    _no_color(monkeypatch)
    line = statusline_handler.render_status_line({})
    assert "ctx" not in line


def test_rate_limits_present_for_subscriber(monkeypatch):
    _no_color(monkeypatch)
    payload = {
        "rate_limits": {
            "five_hour": {"used_percentage": 42},
            "seven_day": {"used_percentage": 71},
        }
    }
    line = statusline_handler.render_status_line(payload)
    assert "5h 42%" in line
    assert "wk 71%" in line


def test_rate_limits_absent_for_api_key_user(monkeypatch):
    _no_color(monkeypatch)
    line = statusline_handler.render_status_line({"context_window": {"used_percentage": 10}})
    assert "5h" not in line
    assert "wk" not in line


def test_rate_limits_reset_countdown_formatted(monkeypatch):
    _no_color(monkeypatch)
    # Force a fixed "now" so the countdown is deterministic.
    fixed_now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    resets_at = "2026-01-01T13:30:00Z"

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(statusline_handler, "datetime", _FrozenDatetime)
    line = statusline_handler.render_status_line(
        {"rate_limits": {"five_hour": {"used_percentage": 20, "resets_at": resets_at}}}
    )
    assert "resets 1h30m" in line


def test_peak_and_off_peak(monkeypatch):
    _no_color(monkeypatch)
    assert statusline_handler._is_peak(
        datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)  # ~7am Pacific on a Monday
    )
    assert not statusline_handler._is_peak(
        datetime(2026, 1, 5, 5, 0, tzinfo=timezone.utc)  # ~9pm Pacific, prior day, off-peak
    )
    assert not statusline_handler._is_peak(
        datetime(2026, 1, 3, 15, 0, tzinfo=timezone.utc)  # Saturday, off-peak all day
    )


def test_render_peak_labels(monkeypatch):
    _no_color(monkeypatch)
    peak_time = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)
    off_peak_time = datetime(2026, 1, 3, 15, 0, tzinfo=timezone.utc)
    assert statusline_handler._render_peak(peak_time) == "PEAK"
    assert statusline_handler._render_peak(off_peak_time) == "OFF-PEAK"


def test_render_model_and_branch_skips_git_when_no_cwd(monkeypatch):
    _no_color(monkeypatch)
    line = statusline_handler.render_status_line({"model": {"display_name": "Sonnet 4.5"}})
    assert "Sonnet 4.5" in line


def test_git_branch_failure_is_silent(monkeypatch):
    def _boom(*args, **kwargs):
        raise OSError("no git")

    monkeypatch.setattr(statusline_handler.subprocess, "run", _boom)
    assert statusline_handler._git_branch("/tmp") is None


def test_render_lines_added_removed(monkeypatch):
    _no_color(monkeypatch)
    line = statusline_handler.render_status_line(
        {"total_lines_added": 120, "total_lines_removed": 30}
    )
    assert "+120/-30" in line


def test_render_lines_absent_when_zero(monkeypatch):
    _no_color(monkeypatch)
    line = statusline_handler.render_status_line({"total_lines_added": 0, "total_lines_removed": 0})
    assert "+" not in line


def test_no_color_env_disables_ansi(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(statusline_handler.sys.stdout, "isatty", lambda: True)
    line = statusline_handler.render_status_line({"context_window": {"used_percentage": 95}})
    assert "\033[" not in line


def test_main_never_raises_on_malformed_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", StringIO("not json"))
    with pytest.raises(SystemExit) as exc_info:
        statusline_handler.main()
    assert exc_info.value.code == 0
    capsys.readouterr()  # should not raise


def test_main_prints_rendered_line(monkeypatch, capsys):
    _no_color(monkeypatch)
    monkeypatch.setattr(
        "sys.stdin", StringIO(json.dumps({"context_window": {"used_percentage": 12}}))
    )
    with pytest.raises(SystemExit) as exc_info:
        statusline_handler.main()
    assert exc_info.value.code == 0
    assert "12%" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Installer wiring
# ---------------------------------------------------------------------------


def _run_install(
    monkeypatch, tmp_path, *extra_args, api_key="test-key", server_url="http://test.example.com"
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "rclm-hooks-install",
            "--local",
            f"--api-key={api_key}",
            f"--server-url={server_url}",
            *extra_args,
        ],
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(installer, "_resolve_binary", lambda name: name)
    installer.main()


def _read_settings(path: Path) -> dict:
    return json.loads(path.read_text())


def test_statusline_installed_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path)

    settings = _read_settings(tmp_path / ".claude" / "settings.json")
    assert settings["statusLine"] == {
        "type": "command",
        "command": "rclm-claude-statusline",
        "padding": 0,
        "refreshInterval": installer._STATUSLINE_REFRESH_INTERVAL_S,
    }


def test_no_statusline_skips_install(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path, "--no-statusline")

    settings = _read_settings(tmp_path / ".claude" / "settings.json")
    assert "statusLine" not in settings


def test_no_statusline_persists_across_reinstalls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path, "--no-statusline")

    # Reinstall with no flags at all — the opt-out should stick, like --no-compress.
    monkeypatch.setattr("sys.argv", ["rclm-hooks-install", "--local"])
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(installer, "_resolve_binary", lambda name: name)
    installer.main()

    settings = _read_settings(tmp_path / ".claude" / "settings.json")
    assert "statusLine" not in settings
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["statusline"] is False


def test_existing_non_rclm_statusline_is_backed_up(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps({"statusLine": {"type": "command", "command": "my-custom-statusline"}})
    )

    _run_install(monkeypatch, tmp_path)

    settings = _read_settings(settings_path)
    assert settings["statusLine"]["command"] == "rclm-claude-statusline"
    assert "Warning" in capsys.readouterr().err

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["statusline_backup"] == {
        "type": "command",
        "command": "my-custom-statusline",
    }


def test_existing_rclm_statusline_is_not_treated_as_foreign(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path)
    capsys.readouterr()

    _run_install(monkeypatch, tmp_path)  # second install, already ours

    assert "Warning" not in capsys.readouterr().err


def test_reinstall_with_resolved_absolute_path_is_not_treated_as_foreign(
    tmp_path, monkeypatch, capsys
):
    """Regression: _resolve_binary returns an absolute path when found on PATH —
    a reinstall must recognize its own prior statusLine, not back it up as foreign."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "rclm-hooks-install",
            "--local",
            "--api-key=test-key",
            "--server-url=http://test.example.com",
        ],
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(installer, "_resolve_binary", lambda name: f"/home/user/.venv/bin/{name}")
    installer.main()
    capsys.readouterr()

    installer.main()  # reinstall, still resolving to the same absolute path

    assert "Warning" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Uninstaller wiring
# ---------------------------------------------------------------------------


def _run_uninstall(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rclm-hooks-uninstall", "--local"])
    uninstaller.main()


def test_uninstall_removes_rclm_statusline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path)

    _run_uninstall(monkeypatch)

    settings = _read_settings(tmp_path / ".claude" / "settings.json")
    assert "statusLine" not in settings


def test_uninstall_removes_rclm_statusline_with_resolved_absolute_path(tmp_path, monkeypatch):
    """Regression: uninstall must recognize a statusLine command that was resolved
    to an absolute path at install time (the realistic case for a real pip install),
    not just the bare binary name."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "rclm-hooks-install",
            "--local",
            "--api-key=test-key",
            "--server-url=http://test.example.com",
        ],
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(installer, "_resolve_binary", lambda name: f"/home/user/.venv/bin/{name}")
    installer.main()

    _run_uninstall(monkeypatch)

    settings = _read_settings(tmp_path / ".claude" / "settings.json")
    assert "statusLine" not in settings
    assert "hooks" not in settings  # every rclm hook should have been stripped too


def test_uninstall_restores_backed_up_statusline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps({"statusLine": {"type": "command", "command": "my-custom-statusline"}})
    )
    _run_install(monkeypatch, tmp_path)

    _run_uninstall(monkeypatch)

    settings = _read_settings(settings_path)
    assert settings["statusLine"] == {"type": "command", "command": "my-custom-statusline"}
    config = json.loads((tmp_path / "config.json").read_text())
    assert not config.get("statusline_backup")
