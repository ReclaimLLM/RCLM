"""Tests for rclm.hooks.copilot_handler."""

import json
from io import StringIO
from unittest.mock import AsyncMock

import pytest

from rclm import _config
from rclm._models import HookSessionRecord
from rclm.hooks import copilot_handler, session_store


def _run_handler(event_name: str, payload: dict, monkeypatch):
    """Call copilot_handler.main() with event_name as argv[1] and payload on stdin."""
    monkeypatch.setattr("sys.argv", ["rclm-copilot-hooks", event_name])
    monkeypatch.setattr("sys.stdin", StringIO(json.dumps(payload)))
    with pytest.raises(SystemExit) as exc_info:
        copilot_handler.main()
    assert exc_info.value.code == 0


def test_session_start_records_event(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")

    payload = {
        "session_id": "copilot-sess-1",
        "cwd": "/test/repo",
        "model": "gpt-4o",
        "source": "new",
        "timestamp": "2026-10-08T12:00:00Z",
    }
    _run_handler("SessionStart", payload, monkeypatch)

    events = session_store.read_events("copilot-sess-1")
    assert len(events) >= 2
    start_ev = next(e for e in events if e.get("event_type") == "SessionStart")
    assert start_ev["cwd"] == "/test/repo"
    assert start_ev["model"] == "gpt-4o"
    assert start_ev["source"] == "new"

    policy_ev = next(e for e in events if e.get("event_type") == "HookPolicySnapshot")
    assert policy_ev["policy"]["capture_provider"] == "copilot"


def test_session_start_context_pack(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    _config.save("http://test.url", "api-key", context_pack=True)

    fake_bootstrap = {
        "context_sessions": [{"title": "Auth setup", "session_summary": "Implemented OAuth tokens"}]
    }

    async def _mock_fetch(*args, **kwargs):
        return fake_bootstrap

    monkeypatch.setattr("rclm.hooks.bootstrap.fetch", _mock_fetch)

    payload = {
        "session_id": "copilot-sess-ctx",
        "cwd": "/test/repo",
        "timestamp": "2026-10-08T12:00:00Z",
    }
    _run_handler("SessionStart", payload, monkeypatch)

    captured = capsys.readouterr()
    output = json.loads(captured.out.strip())
    assert "hookSpecificOutput" in output
    assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "Auth setup" in output["hookSpecificOutput"]["additionalContext"]


def test_user_prompt_submit(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "session_id": "copilot-sess-1",
        "prompt": "Fix bug in parser",
        "timestamp": "2026-10-08T12:01:00Z",
    }
    _run_handler("UserPromptSubmit", payload, monkeypatch)

    events = session_store.read_events("copilot-sess-1")
    assert len(events) == 1
    assert events[0]["event_type"] == "UserPromptSubmit"
    assert events[0]["prompt"] == "Fix bug in parser"


def test_pre_tool_use_recording(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")

    payload = {
        "session_id": "copilot-sess-1",
        "tool_name": "bash",
        "tool_input": {"command": "git status"},
        "tool_use_id": "tool-call-1",
        "timestamp": "2026-10-08T12:02:00Z",
    }
    _run_handler("PreToolUse", payload, monkeypatch)

    events = session_store.read_events("copilot-sess-1")
    assert len(events) == 1
    assert events[0]["event_type"] == "PreToolUse"
    assert events[0]["tool_name"] == "bash"
    assert events[0]["tool_use_id"] == "tool-call-1"


def test_pre_tool_use_dlp_deny(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    _config.save("http://test.url", "api-key", dlp=True)

    from rclm.hooks.dlp import DLPRedactionError

    def _mock_redact(*args, **kwargs):
        raise DLPRedactionError("Leaking .env secrets")

    monkeypatch.setattr("rclm.hooks.dlp.maybe_redact_input", _mock_redact)

    payload = {
        "session_id": "copilot-sess-1",
        "tool_name": "bash",
        "tool_input": {"command": "cat .env"},
        "tool_use_id": "tool-call-dlp",
        "timestamp": "2026-10-08T12:02:00Z",
    }
    _run_handler("PreToolUse", payload, monkeypatch)

    captured = capsys.readouterr()
    output = json.loads(captured.out.strip())
    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "DLP blocked" in output["hookSpecificOutput"]["permissionDecisionReason"]


def test_post_tool_use(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")

    payload = {
        "session_id": "copilot-sess-1",
        "tool_name": "bash",
        "tool_input": {"command": "git status"},
        "tool_use_id": "tool-call-1",
        "tool_response": "On branch main\nnothing to commit",
        "timestamp": "2026-10-08T12:02:05Z",
    }
    _run_handler("PostToolUse", payload, monkeypatch)

    events = session_store.read_events("copilot-sess-1")
    assert len(events) == 1
    assert events[0]["event_type"] == "PostToolUse"
    assert events[0]["tool_name"] == "bash"
    assert events[0]["tool_response"] == "On branch main\nnothing to commit"


def test_subagent_and_compact_events(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    _run_handler(
        "SubagentStart",
        {"session_id": "sub-1", "agent_id": "agent-a", "agent_type": "Plan"},
        monkeypatch,
    )
    _run_handler(
        "SubagentStop",
        {"session_id": "sub-1", "agent_id": "agent-a", "agent_type": "Plan"},
        monkeypatch,
    )
    _run_handler("PreCompact", {"session_id": "sub-1"}, monkeypatch)

    events = session_store.read_events("sub-1")
    event_types = [e["event_type"] for e in events]
    assert event_types == ["SubagentStart", "SubagentStop", "PreCompact"]


def test_stop_assembles_and_uploads_record(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")

    # Simulate sequence: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse
    session_id = "copilot-full-sess"
    _run_handler(
        "SessionStart",
        {
            "session_id": session_id,
            "cwd": "/test/dir",
            "model": "gpt-4o",
            "timestamp": "2026-10-08T12:00:00Z",
        },
        monkeypatch,
    )
    _run_handler(
        "UserPromptSubmit",
        {"session_id": session_id, "prompt": "Run tests", "timestamp": "2026-10-08T12:01:00Z"},
        monkeypatch,
    )
    _run_handler(
        "PreToolUse",
        {
            "session_id": session_id,
            "tool_name": "bash",
            "tool_input": {"command": "pytest"},
            "tool_use_id": "t1",
            "timestamp": "2026-10-08T12:01:05Z",
        },
        monkeypatch,
    )
    _run_handler(
        "PostToolUse",
        {
            "session_id": session_id,
            "tool_name": "bash",
            "tool_input": {"command": "pytest"},
            "tool_response": "5 passed in 0.1s",
            "tool_use_id": "t1",
            "timestamp": "2026-10-08T12:01:10Z",
        },
        monkeypatch,
    )

    uploaded_records: list[HookSessionRecord] = []

    async def _mock_upload(record: HookSessionRecord):
        uploaded_records.append(record)
        return None

    monkeypatch.setattr(copilot_handler, "upload_single", _mock_upload)
    monkeypatch.setattr(copilot_handler, "close_session", AsyncMock())
    monkeypatch.setattr(copilot_handler, "schedule_session_end_update", lambda: None)

    _run_handler(
        "Stop",
        {"session_id": session_id, "timestamp": "2026-10-08T12:02:00Z"},
        monkeypatch,
    )

    assert len(uploaded_records) == 1
    rec = uploaded_records[0]
    assert rec.session_id == session_id
    assert rec.cwd == "/test/dir"
    assert rec.model == "gpt-4o"
    assert rec.agent_client == "copilot"
    assert rec.model_provider == "github"
    assert len(rec.messages) == 1
    assert rec.messages[0]["content"] == "Run tests"
    assert len(rec.tool_calls) == 1
    assert rec.tool_calls[0].tool_name == "bash"
    assert rec.tool_calls[0].tool_result == "5 passed in 0.1s"
    assert session_store.has_marker(session_id, "finalized")


def test_main_error_resilience(monkeypatch):
    """Ensure invalid input or errors exit 0 cleanly without failing."""
    # No arguments
    monkeypatch.setattr("sys.argv", ["rclm-copilot-hooks"])
    with pytest.raises(SystemExit) as exc:
        copilot_handler.main()
    assert exc.value.code == 0

    # Invalid JSON on stdin
    monkeypatch.setattr("sys.argv", ["rclm-copilot-hooks", "SessionStart"])
    monkeypatch.setattr("sys.stdin", StringIO("NOT_JSON"))
    with pytest.raises(SystemExit) as exc:
        copilot_handler.main()
    assert exc.value.code == 0
