"""Tests for rclm.hooks.session_store."""

import json

import pytest

from rclm.hooks import session_store


@pytest.fixture(autouse=True)
def isolate_sessions_dir(tmp_path, monkeypatch):
    """Redirect _SESSIONS_DIR to a temp directory for each test."""
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", sessions_dir)
    return sessions_dir


def test_append_event_creates_file_and_writes_json():
    session_store.append_event("sess-1", {"event_type": "SessionStart", "cwd": "/tmp"})
    path = session_store._session_path("sess-1")
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["event_type"] == "SessionStart"
    assert data["cwd"] == "/tmp"


def test_append_event_appends_multiple_lines():
    session_store.append_event("sess-2", {"event_type": "SessionStart"})
    session_store.append_event("sess-2", {"event_type": "PreToolUse", "tool_name": "Bash"})
    events = session_store.read_events("sess-2")
    assert len(events) == 2
    assert events[0]["event_type"] == "SessionStart"
    assert events[1]["event_type"] == "PreToolUse"


def test_read_events_returns_empty_list_for_unknown_session():
    result = session_store.read_events("nonexistent-session-id")
    assert result == []


def test_read_events_skips_malformed_lines():
    path = session_store._session_path("sess-3")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"event_type": "SessionStart"}\nNOT_JSON\n{"event_type": "Stop"}\n')
    events = session_store.read_events("sess-3")
    assert len(events) == 2
    assert events[0]["event_type"] == "SessionStart"
    assert events[1]["event_type"] == "Stop"


def test_cleanup_removes_file():
    session_store.append_event("sess-4", {"event_type": "SessionStart"})
    path = session_store._session_path("sess-4")
    assert path.exists()
    session_store.cleanup("sess-4")
    assert not path.exists()


def test_cleanup_noop_if_already_gone():
    # Should not raise.
    session_store.cleanup("session-that-never-existed")
