"""Tests for rclm.hooks.openclaw_handler."""

from __future__ import annotations

import json
from io import StringIO

import pytest

from rclm._models import HookSessionRecord
from rclm.hooks import openclaw_handler, session_store


def _run_handler(event_name: str, payload: dict, monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["rclm-openclaw-hooks", event_name])
    monkeypatch.setattr("sys.stdin", StringIO(json.dumps(payload)))
    with pytest.raises(SystemExit) as exc_info:
        openclaw_handler.main()
    assert exc_info.value.code == 0


@pytest.fixture(autouse=True)
def isolate_sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")


def test_session_start_appends_event(monkeypatch):
    payload = {
        "hook_name": "session_start",
        "received_at": "2026-04-27T00:00:00+00:00",
        "event": {
            "sessionId": "oc-sid-1",
            "workspaceDir": "/repo",
            "model": "openai/gpt-5.4",
        },
    }

    _run_handler("session_start", payload, monkeypatch)

    events = session_store.read_events("oc-sid-1")
    assert events == [
        {
            "event_type": "SessionStart",
            "cwd": "/repo",
            "timestamp": "2026-04-27T00:00:00+00:00",
            "model": "openai/gpt-5.4",
            "raw": payload["event"],
        }
    ]


def test_llm_and_tool_events_build_uploaded_record(monkeypatch):
    uploaded: list[HookSessionRecord] = []

    async def fake_upload(record, *, max_retries=3):
        uploaded.append(record)
        assert max_retries == 1

    monkeypatch.setattr(openclaw_handler, "upload_single", fake_upload)

    _run_handler(
        "session_start",
        {
            "received_at": "2026-04-27T00:00:00+00:00",
            "event": {"sessionKey": "oc-sid-2", "context": {"workspaceDir": "/work"}},
        },
        monkeypatch,
    )
    _run_handler(
        "llm_input",
        {
            "received_at": "2026-04-27T00:00:01+00:00",
            "event": {
                "sessionKey": "oc-sid-2",
                "messages": [{"role": "user", "content": "hello"}],
            },
        },
        monkeypatch,
    )
    _run_handler(
        "llm_output",
        {
            "received_at": "2026-04-27T00:00:02+00:00",
            "event": {"sessionKey": "oc-sid-2", "message": {"content": "hi back"}},
        },
        monkeypatch,
    )
    _run_handler(
        "before_tool_call",
        {
            "received_at": "2026-04-27T00:00:03+00:00",
            "event": {
                "sessionKey": "oc-sid-2",
                "toolCallId": "tool-1",
                "toolName": "read_file",
                "params": {"path": "README.md"},
            },
        },
        monkeypatch,
    )
    _run_handler(
        "after_tool_call",
        {
            "received_at": "2026-04-27T00:00:04+00:00",
            "event": {
                "sessionKey": "oc-sid-2",
                "toolCallId": "tool-1",
                "toolName": "read_file",
                "result": "contents",
            },
        },
        monkeypatch,
    )
    _run_handler(
        "session_end",
        {
            "received_at": "2026-04-27T00:01:00+00:00",
            "event": {"sessionKey": "oc-sid-2"},
        },
        monkeypatch,
    )

    assert len(uploaded) == 1
    record = uploaded[0]
    assert record.session_id == "oc-sid-2"
    assert record.cwd == "/work"
    assert record.model == "openclaw-unknown"
    assert record.duration_s == 60.0
    assert [msg["content"] for msg in record.messages] == ["hello", "hi back"]
    assert len(record.tool_calls) == 1
    assert record.tool_calls[0].tool_use_id == "tool-1"
    assert record.tool_calls[0].tool_input == {"path": "README.md"}
    assert record.tool_calls[0].tool_result == "contents"
    assert session_store.read_events("oc-sid-2") == []


def test_unknown_event_exits_zero(monkeypatch):
    _run_handler("unknown_hook", {"event": {"sessionKey": "oc-sid-3"}}, monkeypatch)


def test_malformed_json_exits_zero(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rclm-openclaw-hooks", "session_start"])
    monkeypatch.setattr("sys.stdin", StringIO("not json"))
    with pytest.raises(SystemExit) as exc_info:
        openclaw_handler.main()
    assert exc_info.value.code == 0


def test_raw_payload_strings_are_bounded(monkeypatch):
    _run_handler(
        "llm_input",
        {
            "received_at": "2026-04-27T00:00:01+00:00",
            "event": {"sessionKey": "oc-sid-4", "prompt": "x" * 9000},
        },
        monkeypatch,
    )

    events = session_store.read_events("oc-sid-4")
    assert len(events[-1]["raw"]["prompt"]) < 8100
    assert events[-1]["raw"]["prompt"].endswith("...[truncated]")
