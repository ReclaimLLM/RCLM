"""Focused tests for detached session-end update scheduling."""

from __future__ import annotations

from datetime import datetime, timezone

from rclm import _config
from rclm.hooks import updater


def test_schedule_starts_detached_child_once(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    calls: list[tuple[list[str], dict]] = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(updater.subprocess, "Popen", fake_popen)

    assert updater.schedule_session_end_update() is True
    assert calls[0][0] == [
        updater.sys.executable,
        "-m",
        "rclm.hooks.updater",
        "--session-end-update",
    ]
    assert calls[0][1]["start_new_session"] is True
    assert _config.load()["last_session_end_update_attempt"]


def test_schedule_respects_daily_cooldown(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    _config.patch(last_session_end_update_attempt=datetime.now(timezone.utc).isoformat())
    monkeypatch.setattr(updater.subprocess, "Popen", lambda *_args, **_kwargs: AssertionError())

    assert updater.schedule_session_end_update() is False


def test_schedule_records_attempt_and_never_raises_when_spawn_fails(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    def fail_popen(*_args, **_kwargs):
        raise OSError("no process")

    monkeypatch.setattr(updater.subprocess, "Popen", fail_popen)

    assert updater.schedule_session_end_update() is False
    assert _config.load()["last_session_end_update_attempt"]


def test_run_session_end_update_skips_existing_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    lock_path, _ = updater._session_end_paths()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("busy")

    monkeypatch.setattr("rclm.update.main", lambda: AssertionError())

    updater.run_session_end_update()

    assert lock_path.exists()
