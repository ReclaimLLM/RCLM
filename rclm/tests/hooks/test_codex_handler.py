"""Tests for rclm.hooks.codex_handler and codex_transcript."""

from __future__ import annotations

import json

import pytest

from rclm._models import HookSessionRecord
from rclm.hooks import codex_handler, codex_transcript


def _make_stdin(text: str):
    from io import StringIO

    return StringIO(text)


def _run_handler(event_name: str, payload: dict, monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["rclm-codex-hooks", event_name])
    monkeypatch.setattr("sys.stdin", _make_stdin(json.dumps(payload)))
    with pytest.raises(SystemExit) as exc_info:
        codex_handler.main()
    assert exc_info.value.code == 0


def test_codex_transcript_parses_messages_tools_and_diffs(tmp_path):
    transcript_path = tmp_path / "session.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-03-30T12:00:00Z",
                        "type": "session_meta",
                        "payload": {"model_slug": "gpt-5.4"},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-30T12:00:01Z",
                        "type": "event_msg",
                        "payload": {
                            "type": "user_message",
                            "message": "Explain the bug",
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-30T12:00:02Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "Looking now."}
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-30T12:00:03Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "exec_command",
                            "call_id": "call-1",
                            "arguments": json.dumps({"cmd": "pwd"}),
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-30T12:00:04Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call-1",
                            "output": "Command output",
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-30T12:00:05Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "apply_patch",
                            "call_id": "call-2",
                            "arguments": "*** Begin Patch\n*** Add File: foo.txt\n+hello\n*** End Patch\n",
                        },
                    }
                ),
            ]
        )
        + "\n"
    )

    data = codex_transcript.parse_transcript(str(transcript_path))

    assert data.model == "gpt-5.4"
    assert data.messages == [
        {
            "role": "user",
            "content": "Explain the bug",
            "timestamp": "2026-03-30T12:00:01Z",
        },
        {
            "role": "assistant",
            "content": "Looking now.",
            "timestamp": "2026-03-30T12:00:02Z",
        },
    ]
    assert len(data.tool_calls) == 2
    assert data.tool_calls[0].tool_use_id == "call-1"
    assert data.tool_calls[0].tool_name == "exec_command"
    assert data.tool_calls[0].tool_input == {"cmd": "pwd"}
    assert data.tool_calls[0].tool_result == "Command output"
    assert data.tool_calls[1].tool_name == "apply_patch"
    assert len(data.file_diffs) == 1
    assert data.file_diffs[0].path == "foo.txt"
    assert data.file_diffs[0].after == "hello"


def test_codex_stop_prefers_transcript_data(monkeypatch, tmp_path):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    session_store.append_event(
        "sid-codex",
        {
            "event_type": "SessionStart",
            "cwd": "/repo",
            "model": "hook-model",
            "timestamp": "2026-03-30T12:00:00+00:00",
        },
    )
    session_store.append_event(
        "sid-codex",
        {
            "event_type": "UserPromptSubmit",
            "prompt": "fallback user",
            "turn_id": "turn-1",
            "timestamp": "2026-03-30T12:00:01+00:00",
        },
    )

    uploaded_records = []

    async def fake_upload_single(record):
        uploaded_records.append(record)

    monkeypatch.setattr(
        "rclm.hooks.codex_handler.upload_single", fake_upload_single
    )
    monkeypatch.setattr(
        "rclm.hooks.codex_handler.codex_transcript.parse_transcript",
        lambda path: codex_transcript.CodexTranscriptData(
            messages=[
                {
                    "role": "user",
                    "content": "transcript user",
                    "timestamp": "2026-03-30T12:00:02Z",
                },
                {
                    "role": "assistant",
                    "content": "transcript assistant",
                    "timestamp": "2026-03-30T12:00:03Z",
                },
            ],
            tool_calls=[],
            file_diffs=[],
            model="transcript-model",
        ),
    )

    payload = {
        "session_id": "sid-codex",
        "cwd": "/repo",
        "transcript_path": "/tmp/fake.jsonl",
        "last_assistant_message": "fallback assistant",
        "timestamp": "2026-03-30T12:05:00+00:00",
    }

    _run_handler("Stop", payload, monkeypatch)

    assert len(uploaded_records) == 1
    record = uploaded_records[0]
    assert isinstance(record, HookSessionRecord)
    assert record.model == "transcript-model"
    assert [m["content"] for m in record.messages] == [
        "transcript user",
        "transcript assistant",
    ]


def test_codex_stop_falls_back_when_transcript_empty(monkeypatch, tmp_path):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    session_store.append_event(
        "sid-fallback",
        {
            "event_type": "SessionStart",
            "cwd": "/repo",
            "model": "hook-model",
            "timestamp": "2026-03-30T12:00:00+00:00",
        },
    )
    session_store.append_event(
        "sid-fallback",
        {
            "event_type": "UserPromptSubmit",
            "prompt": "hook user",
            "turn_id": "turn-1",
            "timestamp": "2026-03-30T12:00:01+00:00",
        },
    )

    uploaded_records = []

    async def fake_upload_single(record):
        uploaded_records.append(record)

    monkeypatch.setattr(
        "rclm.hooks.codex_handler.upload_single", fake_upload_single
    )
    monkeypatch.setattr(
        "rclm.hooks.codex_handler.codex_transcript.parse_transcript",
        lambda path: codex_transcript.CodexTranscriptData(),
    )

    payload = {
        "session_id": "sid-fallback",
        "cwd": "/repo",
        "transcript_path": None,
        "last_assistant_message": "hook assistant",
        "timestamp": "2026-03-30T12:05:00+00:00",
    }

    _run_handler("Stop", payload, monkeypatch)

    assert len(uploaded_records) == 1
    record = uploaded_records[0]
    assert [m["content"] for m in record.messages] == [
        "hook user",
        "hook assistant",
    ]
    assert record.model == "hook-model"
