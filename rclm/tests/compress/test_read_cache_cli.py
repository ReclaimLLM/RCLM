"""Tests for the rclm-read-cache CLI entry point."""

import json

import pytest

from rclm import _config
from rclm.compress import read_cache_cli
from rclm.hooks import session_store


def _run(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["rclm-read-cache", *argv])
    with pytest.raises(SystemExit) as exc_info:
        read_cache_cli.main()
    return exc_info.value.code


def test_first_read_prints_raw_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sid-cli-1")

    target = tmp_path / "a.py"
    target.write_text("def foo(): pass\n")

    code = _run(monkeypatch, ["cat", str(target)])

    assert code == 0
    assert capsys.readouterr().out == "def foo(): pass\n"
    events = session_store.read_events("sid-cli-1")
    assert events[-1]["event_type"] == "ReadSnapshot"
    assert events[-1]["file_path"] == str(target)


def test_unchanged_reread_replaced_with_notice(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sid-cli-2")

    target = tmp_path / "a.py"
    target.write_text("def foo(): pass\n")

    _run(monkeypatch, ["cat", str(target)])
    capsys.readouterr()

    code = _run(monkeypatch, ["cat", str(target)])

    assert code == 0
    out = capsys.readouterr().out
    assert "Unchanged since the last read" in out
    assert str(target) in out

    events = session_store.read_events("sid-cli-2")
    saving_events = [e for e in events if e.get("event_type") == "MechanismSaving"]
    assert len(saving_events) == 1
    assert saving_events[0]["mechanism"] == "H1_read_cache"
    assert saving_events[0]["applied"] is True


def test_shadow_mode_prints_raw_output_but_records_shadow_saving(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"shadow_mode": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sid-cli-shadow")

    target = tmp_path / "a.py"
    target.write_text("def foo(): pass\n")

    _run(monkeypatch, ["cat", str(target)])
    capsys.readouterr()

    code = _run(monkeypatch, ["cat", str(target)])

    assert code == 0
    out = capsys.readouterr().out
    # Shadow mode: raw content printed, not the "unchanged" notice.
    assert out == "def foo(): pass\n"

    events = session_store.read_events("sid-cli-shadow")
    saving_events = [e for e in events if e.get("event_type") == "MechanismSaving"]
    assert len(saving_events) == 1
    assert saving_events[0]["applied"] is False
    assert saving_events[0]["mechanism"] == "H1_read_cache"


def test_changed_reread_returns_diff(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sid-cli-3")

    target = tmp_path / "a.py"
    target.write_text("line1\nline2\n")
    _run(monkeypatch, ["cat", str(target)])
    capsys.readouterr()

    target.write_text("line1\nCHANGED\n")
    code = _run(monkeypatch, ["cat", str(target)])

    assert code == 0
    out = capsys.readouterr().out
    assert "changed since the last read" in out
    assert "+CHANGED" in out


def test_no_session_id_falls_through_to_raw_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)

    target = tmp_path / "a.py"
    target.write_text("content\n")

    code = _run(monkeypatch, ["cat", str(target)])

    assert code == 0
    assert capsys.readouterr().out == "content\n"


def test_preserves_nonzero_exit_code(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)

    code = _run(monkeypatch, ["cat", str(tmp_path / "does-not-exist.py")])

    assert code != 0
