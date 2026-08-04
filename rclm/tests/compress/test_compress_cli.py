"""Tests for the rclm-compress CLI entry point, focused on shadow mode."""

import base64
import json

import pytest

from rclm import _config
from rclm.compress import cli
from rclm.hooks import session_store


def _run(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["rclm-compress", *argv])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    return exc_info.value.code


def _mock_execute(monkeypatch, output: str, exit_code: int = 0) -> None:
    monkeypatch.setattr("rclm.compress.cli.execute", lambda command: (output, "", exit_code))


def test_shadow_mode_prints_original_and_records_shadow_event(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"shadow_mode": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sid-cli-shadow")
    _mock_execute(monkeypatch, "\n".join(f"dir/f{i}.py" for i in range(40)))

    code = _run(monkeypatch, ["ls", "-la"])

    assert code == 0
    out = capsys.readouterr().out
    # Shadow mode: original (unfiltered) output printed, not the compacted listing.
    assert "dir/f0.py" in out
    assert out.count("\n") >= 39  # all 40 original lines present, not collapsed

    events = session_store.read_events("sid-cli-shadow")
    saving_events = [e for e in events if e.get("event_type") == "MechanismSaving"]
    assert len(saving_events) == 1
    assert saving_events[0]["applied"] is False
    assert saving_events[0]["mechanism"] == "legacy_compress"


def test_enforce_mode_prints_compressed_and_records_applied_event(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(
        _config, "CONFIG_PATH", tmp_path / "config.json"
    )  # shadow_mode absent -> False
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sid-cli-enforce")
    _mock_execute(monkeypatch, "\n".join(f"dir/f{i}.py" for i in range(40)))

    code = _run(monkeypatch, ["ls", "-la"])

    assert code == 0
    out = capsys.readouterr().out
    assert "more" in out  # compacted listing summary, not all 40 raw lines

    events = session_store.read_events("sid-cli-enforce")
    saving_events = [e for e in events if e.get("event_type") == "MechanismSaving"]
    assert len(saving_events) == 1
    assert saving_events[0]["applied"] is True


def test_explicit_session_id_records_non_claude_wrapper_savings(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    commands = []

    def _execute(command):
        commands.append(command)
        return "\n".join(f"dir/f{i}.py" for i in range(40)), "", 0

    monkeypatch.setattr("rclm.compress.cli.execute", _execute)

    code = _run(monkeypatch, ["--session-id", "sid-codex", "ls", "-la"])

    assert code == 0
    assert commands == ["ls -la"]
    assert "more" in capsys.readouterr().out
    events = session_store.read_events("sid-codex")
    assert len(events) == 1
    assert events[0]["raw_chars"] > events[0]["compressed_chars"]
    assert events[0]["token_estimator"] == "chars_div_4_v1"


def test_session_id_requires_id_and_command(monkeypatch, capsys):
    code = _run(monkeypatch, ["--session-id", "sid-only"])

    assert code == 1
    assert "requires an ID and a command" in capsys.readouterr().err


def test_session_id_rejects_path_traversal(monkeypatch, capsys):
    code = _run(monkeypatch, ["--session-id", "../../outside", "ls"])

    assert code == 1
    assert "invalid --session-id" in capsys.readouterr().err


def test_encoded_command_preserves_quotes_pipes_and_chains(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    command = "printf 'a b' | sed -n '1p' && git status"
    encoded = base64.urlsafe_b64encode(command.encode()).decode()
    commands = []

    def _execute(received):
        commands.append(received)
        return "ok\n", "", 0

    monkeypatch.setattr("rclm.compress.cli.execute", _execute)

    code = _run(monkeypatch, ["--encoded-command", encoded])

    assert code == 0
    assert commands == [command]


def test_encoded_command_rejects_invalid_payload(monkeypatch, capsys):
    code = _run(monkeypatch, ["--encoded-command", "not-base64!"])

    assert code == 1
    assert "invalid --encoded-command" in capsys.readouterr().err


def test_no_filter_matched_no_saving_event(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sid-cli-nomatch")
    _mock_execute(monkeypatch, "hi\n")

    code = _run(monkeypatch, ["echo", "hi"])

    assert code == 0
    assert capsys.readouterr().out == "hi\n"
    assert session_store.read_events("sid-cli-nomatch") == []


def test_preserves_nonzero_exit_code(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    _mock_execute(monkeypatch, "", exit_code=2)

    code = _run(monkeypatch, ["some-command"])

    assert code == 2
