"""Tests for the rclm-read-cache CLI entry point."""

import json

import pytest

from rclm import _config
from rclm.compress import read_cache_cli
from rclm.hooks import session_store


def _content():
    return "".join(f"line {line}: {'x' * 32}\n" for line in range(1, 81))


def _run(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["rclm-read-cache", *argv])
    with pytest.raises(SystemExit) as exc_info:
        read_cache_cli.main()
    return exc_info.value.code


def _configure_read_cache(monkeypatch, tmp_path, *, shadow=False):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True, "shadow_mode": shadow}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)


def test_first_read_prints_raw_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    _configure_read_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sid-cli-1")

    target = tmp_path / "a.py"
    content = _content()
    target.write_text(content)

    code = _run(monkeypatch, ["cat", str(target)])

    assert code == 0
    assert capsys.readouterr().out == content
    state = session_store.read_read_cache_state("sid-cli-1")
    assert state["files"][str(target)]["spans"] == [{"start": 1, "end": 80, "turn": 1}]


def test_unchanged_reread_replaced_with_notice(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    _configure_read_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sid-cli-2")

    target = tmp_path / "a.py"
    target.write_text(_content())

    _run(monkeypatch, ["cat", str(target)])
    capsys.readouterr()

    code = _run(monkeypatch, ["cat", str(target)])

    assert code == 0
    out = capsys.readouterr().out
    assert "[RCLM] Lines 1-80 of a.py unchanged since turn 1." in out

    events = session_store.read_events("sid-cli-2")
    saving_events = [e for e in events if e.get("event_type") == "MechanismSaving"]
    assert len(saving_events) == 1
    assert saving_events[0]["mechanism"] == "range_cache"
    assert saving_events[0]["applied"] is True
    assert saving_events[0]["measurement_kind"] == "measured"


def test_shadow_mode_prints_raw_output_but_records_shadow_saving(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    _configure_read_cache(monkeypatch, tmp_path, shadow=True)
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sid-cli-shadow")

    target = tmp_path / "a.py"
    content = _content()
    target.write_text(content)

    _run(monkeypatch, ["cat", str(target)])
    capsys.readouterr()

    code = _run(monkeypatch, ["cat", str(target)])

    assert code == 0
    out = capsys.readouterr().out
    # Shadow mode: raw content printed, not the "unchanged" notice.
    assert out == content

    events = session_store.read_events("sid-cli-shadow")
    saving_events = [e for e in events if e.get("event_type") == "MechanismSaving"]
    assert len(saving_events) == 1
    assert saving_events[0]["applied"] is False
    assert saving_events[0]["mechanism"] == "range_cache"


def test_changed_reread_invalidates_and_returns_fresh_content(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    _configure_read_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sid-cli-3")

    target = tmp_path / "a.py"
    original = _content()
    target.write_text(original)
    _run(monkeypatch, ["cat", str(target)])
    capsys.readouterr()

    changed = original.replace("line 40:", "changed 40:")
    target.write_text(changed)
    code = _run(monkeypatch, ["cat", str(target)])

    assert code == 0
    out = capsys.readouterr().out
    assert out == changed


def test_disabled_read_cache_prints_raw_output_without_events(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": False}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sid-cli-disabled")

    target = tmp_path / "a.py"
    content = _content()
    target.write_text(content)

    _run(monkeypatch, ["cat", str(target)])
    capsys.readouterr()
    code = _run(monkeypatch, ["cat", str(target)])

    assert code == 0
    assert capsys.readouterr().out == content
    assert session_store.read_events("sid-cli-disabled") == []


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
