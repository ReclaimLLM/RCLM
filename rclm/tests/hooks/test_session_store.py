"""Tests for rclm.hooks.session_store."""

import json
import os

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


def test_cleanup_events_preserves_session_sidecars():
    session_store.append_event("sess-4", {"event_type": "SessionStart"})
    session_store.write_dedupe_state("sess-4", {"hash": {}})
    session_store.write_read_cache_state("sess-4", {"files": {}})
    session_store.write_mechanism_savings_state("sess-4", {"range_cache": {}})
    path = session_store._session_path("sess-4")
    assert path.exists()
    session_store.cleanup_events("sess-4")
    assert not path.exists()
    assert session_store._dedupe_path("sess-4").exists()
    assert session_store._read_cache_path("sess-4").exists()
    assert session_store._mechanism_savings_path("sess-4").exists()


def test_cleanup_removes_events_and_session_sidecars():
    session_store.append_event("sess-final", {"event_type": "SessionStart"})
    session_store.write_dedupe_state("sess-final", {"hash": {}})
    session_store.write_read_cache_state("sess-final", {"files": {}})
    session_store.write_mechanism_savings_state("sess-final", {"range_cache": {}})

    session_store.cleanup("sess-final")

    assert not session_store._session_path("sess-final").exists()
    assert not session_store._dedupe_path("sess-final").exists()
    assert not session_store._read_cache_path("sess-final").exists()
    assert not session_store._mechanism_savings_path("sess-final").exists()


def test_cleanup_noop_if_already_gone():
    # Should not raise.
    session_store.cleanup("session-that-never-existed")


def test_hook_health_persists_outside_session_cleanup():
    record = {"session_id": "sess-health", "status": "missing_tool_hooks"}
    session_store.write_hook_health("sess-health", record)

    session_store.append_event("sess-health", {"event_type": "SessionStart"})
    session_store.cleanup("sess-health")

    assert session_store.read_hook_health("sess-health") == record


def test_hook_health_retains_only_newest_bounded_records(monkeypatch):
    monkeypatch.setattr(session_store, "_HOOK_HEALTH_MAX_RECORDS", 2)

    for index in range(3):
        session_store.write_hook_health(
            f"sess-{index}",
            {"session_id": f"sess-{index}", "status": "healthy"},
        )

    records = list(session_store._hook_health_dir().glob("*.json"))
    assert len(records) == 2
    assert session_store.read_hook_health("sess-2")["status"] == "healthy"


def test_prune_stale_sidecars_removes_only_expired_state():
    session_store.write_dedupe_state("stale", {"hash": {}})
    session_store.write_mechanism_savings_state("stale", {"range_cache": {}})
    session_store.write_read_cache_state("fresh", {"files": {}})
    stale = session_store._dedupe_path("stale")
    stale_mechanisms = session_store._mechanism_savings_path("stale")
    os.utime(stale, (0, 0))
    os.utime(stale_mechanisms, (0, 0))

    session_store.prune_stale_sidecars(max_age_s=5)

    assert not stale.exists()
    assert not stale_mechanisms.exists()
    assert session_store._read_cache_path("fresh").exists()
