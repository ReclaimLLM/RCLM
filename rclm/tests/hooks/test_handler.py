"""Tests for rclm.hooks.claude_handler."""

import json

import pytest

from rclm._models import HookSessionRecord
from rclm.hooks import claude_handler as handler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_handler(event_name: str, payload: dict, monkeypatch) -> None:
    """Call handler.main() with event_name as argv[1] and payload as stdin."""
    monkeypatch.setattr("sys.argv", ["rclm-claude-hooks", event_name])
    monkeypatch.setattr("sys.stdin", _make_stdin(json.dumps(payload)))
    with pytest.raises(SystemExit) as exc_info:
        handler.main()
    assert exc_info.value.code == 0


def _make_stdin(text: str):
    from io import StringIO

    return StringIO(text)


# ---------------------------------------------------------------------------
# SessionStart
# ---------------------------------------------------------------------------


def test_session_start_appends_event(monkeypatch, tmp_path):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "session_id": "sid-1",
        "cwd": "/projects/foo",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("SessionStart", payload, monkeypatch)

    events = session_store.read_events("sid-1")
    assert len(events) == 1
    assert events[0]["event_type"] == "SessionStart"
    assert events[0]["cwd"] == "/projects/foo"


# ---------------------------------------------------------------------------
# PreToolUse — compression gating
# ---------------------------------------------------------------------------


def test_pre_tool_use_no_compression_by_default(monkeypatch, tmp_path, capsys):
    """With compress=False (default), PreToolUse should NOT print updatedInput."""
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")

    session_store.append_event(
        "sid-c1", {"event_type": "SessionStart", "cwd": "/x", "timestamp": "2024-01-01T00:00:00Z"}
    )

    payload = {
        "session_id": "sid-c1",
        "tool_name": "Grep",
        "tool_input": {"pattern": "foo"},
        "timestamp": "2024-01-01T00:00:01Z",
    }
    _run_handler("PreToolUse", payload, monkeypatch)

    captured = capsys.readouterr()
    assert "updatedInput" not in captured.out


def test_pre_tool_use_compression_when_enabled(monkeypatch, tmp_path, capsys):
    """With compress=True in config, PreToolUse should print updatedInput for Grep."""
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"compress": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    session_store.append_event(
        "sid-c2", {"event_type": "SessionStart", "cwd": "/x", "timestamp": "2024-01-01T00:00:00Z"}
    )

    payload = {
        "session_id": "sid-c2",
        "tool_name": "Grep",
        "tool_input": {"pattern": "foo"},
        "timestamp": "2024-01-01T00:00:01Z",
    }
    _run_handler("PreToolUse", payload, monkeypatch)

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["hookSpecificOutput"]["updatedInput"]["head_limit"] == 50


# ---------------------------------------------------------------------------
# PostToolUse
# ---------------------------------------------------------------------------


def test_post_tool_use_appends_tool_event(monkeypatch, tmp_path):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    # First set up a SessionStart so the session file exists.
    session_store.append_event(
        "sid-2", {"event_type": "SessionStart", "cwd": "/x", "timestamp": "2024-01-01T00:00:00Z"}
    )

    payload = {
        "session_id": "sid-2",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_response": "file.py\n",
        "timestamp": "2024-01-01T00:00:01Z",
    }
    _run_handler("PostToolUse", payload, monkeypatch)

    events = session_store.read_events("sid-2")
    assert events[-1]["event_type"] == "PostToolUse"
    assert events[-1]["tool_name"] == "Bash"
    assert events[-1]["tool_response"] == "file.py\n"


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


def test_stop_builds_hook_session_record_and_uploads(monkeypatch, tmp_path):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    # Pre-populate session events.
    session_store.append_event(
        "sid-3",
        {"event_type": "SessionStart", "cwd": "/projects/bar", "timestamp": "2024-01-01T00:00:00Z"},
    )

    uploaded_records = []

    async def fake_upload_single(record):
        uploaded_records.append(record)

    from rclm.hooks.transcript import TranscriptData

    monkeypatch.setattr(
        "rclm.hooks.claude_handler.upload_single",
        fake_upload_single,
    )
    monkeypatch.setattr(
        "rclm.hooks.claude_handler.transcript.parse_transcript",
        lambda path: TranscriptData(
            messages=[{"role": "user", "content": "hi", "timestamp": ""}],
            tool_calls=[],
            model="claude-sonnet-4-6",
            total_input_tokens=10,
            total_output_tokens=5,
        ),
    )

    payload = {
        "session_id": "sid-3",
        "cwd": "/projects/bar",
        "transcript_path": "/tmp/fake.jsonl",
        "timestamp": "2024-01-01T00:01:00Z",
    }
    _run_handler("Stop", payload, monkeypatch)

    assert len(uploaded_records) == 1
    record = uploaded_records[0]
    assert isinstance(record, HookSessionRecord)
    assert record.session_id == "sid-3"
    assert record.cwd == "/projects/bar"
    assert record.model == "claude-sonnet-4-6"
    assert record.total_input_tokens == 10
    assert record.total_output_tokens == 5
    assert len(record.messages) == 1

    # Cleanup should have removed the session file.
    assert session_store.read_events("sid-3") == []


def test_stop_without_prior_session_start_uses_fallback(monkeypatch, tmp_path):
    """Stop event with no SessionStart in store must not crash."""
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    uploaded_records = []

    async def fake_upload_single(record):
        uploaded_records.append(record)

    from rclm.hooks.transcript import TranscriptData

    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", fake_upload_single)
    monkeypatch.setattr(
        "rclm.hooks.claude_handler.transcript.parse_transcript",
        lambda path: TranscriptData(),
    )

    payload = {
        "session_id": "sid-4",
        "cwd": "/fallback",
        "transcript_path": None,
        "timestamp": "2024-01-01T00:01:00Z",
    }
    _run_handler("Stop", payload, monkeypatch)

    assert len(uploaded_records) == 1
    record = uploaded_records[0]
    assert record.cwd == "/fallback"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_handler_exits_0_on_exception(monkeypatch, tmp_path):
    """Any exception in a handler must be swallowed; process exits 0."""
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    def boom(session_id, payload):
        raise RuntimeError("catastrophic failure")

    monkeypatch.setitem(handler._HANDLERS, "SessionStart", boom)

    monkeypatch.setattr("sys.argv", ["rclm-claude-hooks", "SessionStart"])
    monkeypatch.setattr("sys.stdin", _make_stdin('{"session_id": "x", "cwd": "/"}'))
    with pytest.raises(SystemExit) as exc_info:
        handler.main()
    assert exc_info.value.code == 0


def test_unknown_event_exits_0(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rclm-claude-hooks", "UnknownEvent"])
    monkeypatch.setattr("sys.stdin", _make_stdin("{}"))
    with pytest.raises(SystemExit) as exc_info:
        handler.main()
    assert exc_info.value.code == 0
