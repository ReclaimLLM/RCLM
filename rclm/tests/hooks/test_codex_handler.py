"""Tests for rclm.hooks.codex_handler and codex_transcript."""

from __future__ import annotations

import json

import pytest
from jsonschema import validate

from rclm._models import HookSessionRecord
from rclm.hooks import codex_handler, codex_transcript

CODEX_POST_TOOL_USE_OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "additionalProperties": False,
    "definitions": {
        "BlockDecisionWire": {"enum": ["block"], "type": "string"},
        "HookEventNameWire": {
            "enum": [
                "PreToolUse",
                "PermissionRequest",
                "PostToolUse",
                "SessionStart",
                "UserPromptSubmit",
                "Stop",
            ],
            "type": "string",
        },
        "PostToolUseHookSpecificOutputWire": {
            "additionalProperties": False,
            "properties": {
                "additionalContext": {"default": None, "type": "string"},
                "hookEventName": {"$ref": "#/definitions/HookEventNameWire"},
                "updatedMCPToolOutput": {"default": None},
            },
            "required": ["hookEventName"],
            "type": "object",
        },
    },
    "properties": {
        "continue": {"default": True, "type": "boolean"},
        "decision": {
            "allOf": [{"$ref": "#/definitions/BlockDecisionWire"}],
            "default": None,
        },
        "hookSpecificOutput": {
            "allOf": [{"$ref": "#/definitions/PostToolUseHookSpecificOutputWire"}],
            "default": None,
        },
        "reason": {"default": None, "type": "string"},
        "stopReason": {"default": None, "type": "string"},
        "suppressOutput": {"default": False, "type": "boolean"},
        "systemMessage": {"default": None, "type": "string"},
    },
    "title": "post-tool-use.command.output",
    "type": "object",
}

CODEX_POST_TOOL_USE_INPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "additionalProperties": False,
    "definitions": {"NullableString": {"type": ["string", "null"]}},
    "properties": {
        "cwd": {"type": "string"},
        "hook_event_name": {"const": "PostToolUse", "type": "string"},
        "model": {"type": "string"},
        "permission_mode": {
            "enum": ["default", "acceptEdits", "plan", "dontAsk", "bypassPermissions"],
            "type": "string",
        },
        "session_id": {"type": "string"},
        "tool_input": True,
        "tool_name": {"type": "string"},
        "tool_response": True,
        "tool_use_id": {"type": "string"},
        "transcript_path": {"$ref": "#/definitions/NullableString"},
        "turn_id": {"type": "string"},
    },
    "required": [
        "cwd",
        "hook_event_name",
        "model",
        "permission_mode",
        "session_id",
        "tool_input",
        "tool_name",
        "tool_response",
        "tool_use_id",
        "transcript_path",
        "turn_id",
    ],
    "title": "post-tool-use.command.input",
    "type": "object",
}


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
                            "content": [{"type": "output_text", "text": "Looking now."}],
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


def test_codex_transcript_parses_custom_apply_patch_diffs(tmp_path):
    transcript_path = tmp_path / "session.jsonl"
    transcript_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-07T12:52:34.670Z",
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "status": "completed",
                    "call_id": "call-patch",
                    "name": "apply_patch",
                    "input": (
                        "*** Begin Patch\n"
                        "*** Add File: /repo/new.txt\n"
                        "+first\n"
                        "+second\n"
                        "*** End Patch\n"
                    ),
                },
            }
        )
        + "\n"
    )

    data = codex_transcript.parse_transcript(str(transcript_path))

    assert len(data.tool_calls) == 1
    assert data.tool_calls[0].tool_use_id == "call-patch"
    assert data.tool_calls[0].tool_name == "apply_patch"
    assert data.tool_calls[0].tool_input["input"].startswith("*** Begin Patch")
    assert len(data.file_diffs) == 1
    assert data.file_diffs[0].path == "/repo/new.txt"
    assert data.file_diffs[0].before is None
    assert data.file_diffs[0].after == "first\nsecond"


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

    monkeypatch.setattr("rclm.hooks.codex_handler.upload_single", fake_upload_single)
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

    monkeypatch.setattr("rclm.hooks.codex_handler.upload_single", fake_upload_single)
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


def test_codex_post_tool_use_dlp_output_matches_codex_schema(monkeypatch, tmp_path, capsys):
    from rclm.hooks import dlp, session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    monkeypatch.setattr("rclm._config.load", lambda: {"dlp": True})

    def mock_redact(tool_name, tool_response, cwd):
        assert tool_name == "Bash"
        assert tool_response == "My secret is password123"
        assert cwd == "/repo"
        return "My secret is [REDACTED:PASSWORD]"

    monkeypatch.setattr(dlp, "maybe_redact_output", mock_redact)

    payload = {
        "session_id": "sid-codex",
        "cwd": "/repo",
        "hook_event_name": "PostToolUse",
        "model": "gpt-5.4",
        "permission_mode": "default",
        "tool_name": "Bash",
        "tool_input": {"command": "cat .env"},
        "tool_response": "My secret is password123",
        "tool_use_id": "call-1",
        "transcript_path": None,
        "turn_id": "turn-1",
    }
    validate(instance=payload, schema=CODEX_POST_TOOL_USE_INPUT_SCHEMA)

    _run_handler("PostToolUse", payload, monkeypatch)

    output = capsys.readouterr().out.strip()
    assert output
    parsed = json.loads(output)
    validate(instance=parsed, schema=CODEX_POST_TOOL_USE_OUTPUT_SCHEMA)

    hso = parsed["hookSpecificOutput"]
    assert hso["hookEventName"] == "PostToolUse"
    assert hso["updatedMCPToolOutput"] == "My secret is [REDACTED:PASSWORD]"
    assert "updatedResponse" not in hso


def test_codex_post_tool_use_no_stdout_when_dlp_finds_nothing(monkeypatch, tmp_path, capsys):
    from rclm.hooks import dlp, session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("rclm._config.load", lambda: {"dlp": True})
    monkeypatch.setattr(dlp, "maybe_redact_output", lambda tool_name, response, cwd: None)

    payload = {
        "session_id": "sid-codex-clean",
        "cwd": "/repo",
        "hook_event_name": "PostToolUse",
        "model": "gpt-5.4",
        "permission_mode": "default",
        "tool_name": "Bash",
        "tool_input": {"command": "echo ok"},
        "tool_response": "ok",
        "tool_use_id": "call-2",
        "transcript_path": "/tmp/session.jsonl",
        "turn_id": "turn-2",
    }
    validate(instance=payload, schema=CODEX_POST_TOOL_USE_INPUT_SCHEMA)

    _run_handler("PostToolUse", payload, monkeypatch)

    assert capsys.readouterr().out == ""
