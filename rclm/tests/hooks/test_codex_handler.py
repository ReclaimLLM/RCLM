"""Tests for rclm.hooks.codex_handler and codex_transcript."""

from __future__ import annotations

import base64
import json

import pytest
from jsonschema import validate

from rclm._models import HookSessionRecord
from rclm.hooks import codex_handler, codex_transcript


def _wrapped(command: str, session_id: str) -> str:
    encoded = base64.urlsafe_b64encode(command.encode()).decode()
    return f"rclm-compress --session-id {session_id} --encoded-command {encoded}"


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


def test_codex_pre_tool_use_rewrites_recognized_shell_input(monkeypatch, tmp_path, capsys):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("rclm._config.load", lambda: {"compression": {"enabled": True}})
    monkeypatch.setattr("rclm.hooks.compress._compress_available", lambda: True)

    _run_handler(
        "PreToolUse",
        {
            "session_id": "sid-codex-pre",
            "hook_event_name": "PreToolUse",
            "tool_name": "exec_command",
            "tool_input": {"command": "cat build.log", "yield_time_ms": 1_000},
            "turn_id": "turn-1",
        },
        monkeypatch,
    )

    output = json.loads(capsys.readouterr().out)
    specific = output["hookSpecificOutput"]
    assert specific["hookEventName"] == "PreToolUse"
    assert specific["permissionDecision"] == "allow"
    assert specific["updatedInput"] == {
        "command": _wrapped("cat build.log", "sid-codex-pre"),
        "yield_time_ms": 1_000,
    }


def test_codex_post_tool_use_compacts_recognized_shell_output(monkeypatch, tmp_path, capsys):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("rclm._config.load", lambda: {"compression": {"enabled": True}})
    monkeypatch.setattr("rclm.hooks.compress._compress_available", lambda: False)
    _run_handler(
        "PreToolUse",
        {
            "session_id": "sid-codex-post",
            "hook_event_name": "PreToolUse",
            "tool_name": "exec_command",
            "tool_input": {"command": "cat build.log"},
            "tool_use_id": "call-1",
            "turn_id": "turn-1",
        },
        monkeypatch,
    )
    _run_handler(
        "PreToolUse",
        {
            "session_id": "sid-codex-post",
            "hook_event_name": "PreToolUse",
            "tool_name": "exec_command",
            "tool_input": {"command": "echo unrelated"},
            "tool_use_id": "call-2",
            "turn_id": "turn-1",
        },
        monkeypatch,
    )
    assert capsys.readouterr().out == ""

    _run_handler(
        "PostToolUse",
        {
            "session_id": "sid-codex-post",
            "cwd": "/repo",
            "hook_event_name": "PostToolUse",
            "tool_name": "exec_command",
            "tool_response": "".join(f"line {line}\n" for line in range(100)),
            "tool_use_id": "call-1",
            "turn_id": "turn-1",
        },
        monkeypatch,
    )

    output = json.loads(capsys.readouterr().out)
    assert output["continue"] is False
    assert output["decision"] == "block"
    assert "40 lines omitted" in output["reason"]
    assert output["stopReason"] == (
        "ReclaimLLM compacted this tool result before it entered the model context."
    )
    events = session_store.read_events("sid-codex-post")
    transformation = next(e for e in events if e.get("event_type") == "ToolTransformation")
    assert transformation["compression_strategy"] == "H3_exec_compaction"
    assert transformation["applied"] is True
    assert transformation["raw_chars"] > transformation["compressed_chars"]


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
    assert "+hello" in data.file_diffs[0].unified_diff
    assert data.file_diffs[0].timestamp == "2026-03-30T12:00:05Z"


def test_codex_transcript_captures_developer_role_messages(tmp_path):
    """'developer' is the Responses API's successor to 'system' and carries
    real injected instructions (AGENTS.md, memory-tool guidance, multi-agent
    mode directives) -- it must not be silently dropped alongside user/
    assistant."""
    transcript_path = tmp_path / "session.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-07-29T09:36:34Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "developer",
                            "content": [
                                {"type": "input_text", "text": "## Memory\n\nUse it wisely."}
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-07-29T09:36:35Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "hi"}],
                        },
                    }
                ),
            ]
        )
        + "\n"
    )

    data = codex_transcript.parse_transcript(str(transcript_path))

    assert data.messages == [
        {
            "role": "developer",
            "content": "## Memory\n\nUse it wisely.",
            "timestamp": "2026-07-29T09:36:34Z",
        },
        {
            "role": "user",
            "content": "hi",
            "timestamp": "2026-07-29T09:36:35Z",
        },
    ]


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
    assert "+first" in data.file_diffs[0].unified_diff
    assert "+second" in data.file_diffs[0].unified_diff
    assert data.file_diffs[0].timestamp == "2026-04-07T12:52:34.670Z"


def test_codex_transcript_parses_patch_apply_end_diffs(tmp_path):
    transcript_path = tmp_path / "session.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-08-10T15:32:29.622Z",
                        "type": "event_msg",
                        "payload": {
                            "type": "patch_apply_end",
                            "success": True,
                            "status": "completed",
                            "changes": {
                                "/repo/existing.py": {
                                    "type": "update",
                                    "unified_diff": "@@ -1,2 +1,2 @@\n keep\n-old\n+new\n",
                                    "move_path": None,
                                },
                                "/repo/new.py": {
                                    "type": "add",
                                    "unified_diff": "@@ -0,0 +1,2 @@\n+first\n+second\n",
                                    "move_path": None,
                                },
                                "/repo/old.py": {
                                    "type": "delete",
                                    "unified_diff": "@@ -1 +0,0 @@\n-old\n",
                                    "move_path": None,
                                },
                                "/repo/source.py": {
                                    "type": "update",
                                    "unified_diff": "",
                                    "move_path": "/repo/destination.py",
                                },
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-08-10T15:32:30.000Z",
                        "type": "event_msg",
                        "payload": {
                            "type": "patch_apply_end",
                            "success": False,
                            "changes": {
                                "/repo/failed.py": {
                                    "type": "add",
                                    "unified_diff": "+not-applied\n",
                                }
                            },
                        },
                    }
                ),
            ]
        )
        + "\n"
    )

    data = codex_transcript.parse_transcript(str(transcript_path))

    assert [diff.path for diff in data.file_diffs] == [
        "/repo/existing.py",
        "/repo/new.py",
        "/repo/old.py",
        "/repo/destination.py",
    ]
    assert data.file_diffs[0].before == "keep\nold"
    assert data.file_diffs[0].after == "keep\nnew"
    assert data.file_diffs[0].unified_diff.startswith("@@ -1,2 +1,2 @@")
    assert data.file_diffs[0].timestamp == "2026-08-10T15:32:29.622Z"
    assert data.file_diffs[1].before is None
    assert data.file_diffs[1].after == "first\nsecond"
    assert data.file_diffs[2].before == "old"
    assert data.file_diffs[2].after is None


def test_codex_transcript_parses_model_and_cumulative_usage_with_reset(tmp_path):
    transcript_path = tmp_path / "usage.jsonl"
    entries = [
        {
            "timestamp": "2026-04-29T09:11:18Z",
            "type": "turn_context",
            "payload": {"model": "gpt-5.4"},
        },
        {
            "timestamp": "2026-04-29T09:12:00Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 40,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                    }
                },
            },
        },
        {
            "timestamp": "2026-04-29T09:13:00Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 180,
                        "cached_input_tokens": 80,
                        "output_tokens": 30,
                        "reasoning_output_tokens": 8,
                    }
                },
            },
        },
        {
            "timestamp": "2026-04-29T09:14:00Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 10,
                        "cached_input_tokens": 2,
                        "output_tokens": 4,
                        "reasoning_output_tokens": 1,
                    }
                },
            },
        },
    ]
    transcript_path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")

    data = codex_transcript.parse_transcript(str(transcript_path))

    assert data.model == "gpt-5.4"
    assert data.usage == codex_transcript.CodexUsage(
        input_tokens=190,
        cached_input_tokens=82,
        output_tokens=34,
        reasoning_output_tokens=9,
    )


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


def test_codex_stop_schedules_update_after_upload(monkeypatch, tmp_path):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    calls: list[str] = []

    async def fake_upload_single(_record):
        calls.append("upload")

    monkeypatch.setattr("rclm.hooks.codex_handler.upload_single", fake_upload_single)
    monkeypatch.setattr(
        "rclm.hooks.codex_handler.schedule_session_end_update",
        lambda: calls.append("schedule"),
    )
    monkeypatch.setattr(
        "rclm.hooks.codex_handler.codex_transcript.parse_transcript",
        lambda _path: codex_transcript.CodexTranscriptData(),
    )

    _run_handler(
        "Stop",
        {"session_id": "sid-codex-update", "timestamp": "2026-03-30T12:05:00+00:00"},
        monkeypatch,
    )

    assert calls == ["upload", "schedule"]


def test_codex_stop_closes_uploader_session(monkeypatch, tmp_path):
    """Regression test: Stop must close the module-level aiohttp session in
    the same event loop it uploaded on, or aiohttp emits "Unclosed client
    session"/"Unclosed connector" ResourceWarnings to stderr on every Stop
    (confirmed via a real subprocess repro against a live session
    transcript -- see close_session's docstring in _uploader.py)."""
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    closed = []

    async def fake_upload_single(_record):
        pass

    async def fake_close_session():
        closed.append(True)

    monkeypatch.setattr("rclm.hooks.codex_handler.upload_single", fake_upload_single)
    monkeypatch.setattr("rclm.hooks.codex_handler.close_session", fake_close_session)
    monkeypatch.setattr(
        "rclm.hooks.codex_handler.codex_transcript.parse_transcript",
        lambda _path: codex_transcript.CodexTranscriptData(),
    )

    _run_handler(
        "Stop",
        {"session_id": "sid-codex-close", "timestamp": "2026-03-30T12:06:00+00:00"},
        monkeypatch,
    )

    assert closed == [True]


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

    def mock_redact(tool_name, tool_response, cwd, *, redact_all=False):
        assert tool_name == "Bash"
        assert tool_response == "My secret is password123"
        assert cwd == "/repo"
        assert redact_all is True
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

    assert parsed["continue"] is False
    assert parsed["decision"] == "block"
    assert parsed["reason"] == "My secret is [REDACTED:PASSWORD]"
    assert parsed["stopReason"] == (
        "ReclaimLLM DLP withheld the original tool result because it contained an env-file "
        "secret; a redacted result was returned instead."
    )
    assert "hookSpecificOutput" not in parsed
    event = session_store.read_events("sid-codex")[-1]
    assert event["tool_response"] == "My secret is [REDACTED:PASSWORD]"
    assert "password123" not in json.dumps(event)


def test_codex_post_tool_use_no_stdout_when_dlp_finds_nothing(monkeypatch, tmp_path, capsys):
    from rclm.hooks import dlp, session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("rclm._config.load", lambda: {"dlp": True})
    monkeypatch.setattr(
        dlp,
        "maybe_redact_output",
        lambda tool_name, response, cwd, **kwargs: None,
    )

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


def test_codex_post_tool_use_dedupe_blocks_repeated_result(monkeypatch, tmp_path, capsys):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("rclm._config.load", lambda: {"compression": {"dedupe": True}})

    text = "result line\n" * 100  # > min_dedupe_chars

    first_payload = {
        "session_id": "sid-codex-dedupe",
        "cwd": "/repo",
        "hook_event_name": "PostToolUse",
        "model": "gpt-5.4",
        "permission_mode": "default",
        "tool_name": "Bash",
        "tool_input": {"command": "cat build.log"},
        "tool_response": text,
        "tool_use_id": "call-1",
        "transcript_path": None,
        "turn_id": "turn-1",
    }
    _run_handler("PostToolUse", first_payload, monkeypatch)
    assert capsys.readouterr().out == ""

    second_payload = dict(first_payload, tool_use_id="call-2", turn_id="turn-2")
    _run_handler("PostToolUse", second_payload, monkeypatch)

    output = capsys.readouterr().out.strip()
    parsed = json.loads(output)
    validate(instance=parsed, schema=CODEX_POST_TOOL_USE_OUTPUT_SCHEMA)
    assert parsed["continue"] is False
    assert parsed["decision"] == "block"
    assert "Identical to the result of `Bash`" in parsed["reason"]
    assert parsed["stopReason"] == (
        "ReclaimLLM replaced this repeated tool result with a deduplication notice."
    )


def test_codex_post_tool_use_dedupe_off_by_default(monkeypatch, tmp_path, capsys):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("rclm._config.load", lambda: {})

    text = "result line\n" * 100
    payload = {
        "session_id": "sid-codex-dedupe-off",
        "cwd": "/repo",
        "hook_event_name": "PostToolUse",
        "model": "gpt-5.4",
        "permission_mode": "default",
        "tool_name": "Bash",
        "tool_input": {"command": "cat build.log"},
        "tool_response": text,
        "tool_use_id": "call-1",
        "transcript_path": None,
        "turn_id": "turn-1",
    }
    _run_handler("PostToolUse", payload, monkeypatch)
    _run_handler("PostToolUse", dict(payload, tool_use_id="call-2", turn_id="turn-2"), monkeypatch)

    assert capsys.readouterr().out == ""


def test_codex_post_tool_use_range_cache_blocks_repeated_read(monkeypatch, tmp_path, capsys):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("rclm._config.load", lambda: {"read_cache": True})
    target = tmp_path / "source.py"
    content = "".join(f"line {line}: {'x' * 32}\n" for line in range(1, 81))
    target.write_text(content)

    def _pre(turn_id: str) -> dict:
        return {
            "session_id": "sid-codex-range",
            "cwd": str(tmp_path),
            "hook_event_name": "PreToolUse",
            "tool_input": {"cmd": f"cat {target.name}"},
            "turn_id": turn_id,
            "timestamp": "2026-04-10T00:00:00Z",
        }

    def _post(turn_id: str) -> dict:
        return {
            "session_id": "sid-codex-range",
            "cwd": str(tmp_path),
            "hook_event_name": "PostToolUse",
            "model": "gpt-5.4",
            "permission_mode": "default",
            "tool_name": "exec_command",
            "tool_input": {"cmd": f"cat {target.name}"},
            "tool_response": content,
            "tool_use_id": f"call-{turn_id}",
            "transcript_path": None,
            "turn_id": turn_id,
        }

    _run_handler("PreToolUse", _pre("turn-1"), monkeypatch)
    _run_handler("PostToolUse", _post("turn-1"), monkeypatch)
    assert capsys.readouterr().out == ""
    _run_handler("PreToolUse", _pre("turn-2"), monkeypatch)
    _run_handler("PostToolUse", _post("turn-2"), monkeypatch)

    output = json.loads(capsys.readouterr().out)
    validate(instance=output, schema=CODEX_POST_TOOL_USE_OUTPUT_SCHEMA)
    assert output["continue"] is False
    assert output["decision"] == "block"
    assert "[RCLM] Lines 1-80 of source.py unchanged since turn 1." in output["reason"]
    assert output["stopReason"] == (
        "ReclaimLLM replaced this repeated file read with a range-cache notice."
    )
    events = session_store.read_events("sid-codex-range")
    transformation = next(e for e in events if e.get("event_type") == "ToolTransformation")
    assert transformation["compression_strategy"] == "range_cache"
    assert transformation["measurement_kind"] == "measured"


# ---------------------------------------------------------------------------
# PostToolUse — MCP image-lifecycle measurement (never a rewrite)
# ---------------------------------------------------------------------------


def _b64_noise_image(width: int, height: int) -> str:
    import base64
    import io
    import os

    import PIL.Image

    img = PIL.Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mcp_image_payload(
    session_id, *, turn_id="turn-mcp-1", width=2000, height=2000, url="http://x"
):
    return {
        "session_id": session_id,
        "cwd": "/repo",
        "hook_event_name": "PostToolUse",
        "model": "gpt-5.4",
        "permission_mode": "default",
        "tool_name": "mcp__playwright__browser_take_screenshot",
        "tool_input": {"url": url},
        "tool_response": {
            "content": [
                {"type": "image", "data": _b64_noise_image(width, height), "mimeType": "image/png"}
            ],
            "isError": False,
        },
        "tool_use_id": f"call-{turn_id}",
        "transcript_path": None,
        "turn_id": turn_id,
    }


def test_codex_mcp_image_result_measures_but_never_rewrites(monkeypatch, tmp_path, capsys):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(
        "rclm._config.load", lambda: {"image_lifecycle": True, "image_max_dim": 100}
    )

    payload = _mcp_image_payload("sid-codex-img")
    _run_handler("PostToolUse", payload, monkeypatch)

    # Critical regression guard: Codex must never attempt the rewrite —
    # updatedMCPToolOutput is confirmed broken on Codex CLI, and "decision:
    # block" (the Bash text-replacement mechanism) must not fire either.
    assert capsys.readouterr().out == ""

    events = session_store.read_events("sid-codex-img")
    saving = next(
        e
        for e in events
        if e.get("event_type") == "MechanismSaving" and e.get("mechanism") == "image_downscale"
    )
    assert saving["applied"] is False
    assert saving["measurement_kind"] == "measured"
    assert saving["tokens_saved_estimate"] > 0


def test_codex_mcp_image_eviction_measured_unapplied_on_supersession(monkeypatch, tmp_path, capsys):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(
        "rclm._config.load", lambda: {"image_lifecycle": True, "image_max_dim": 100}
    )

    _run_handler(
        "PostToolUse",
        _mcp_image_payload("sid-codex-evict", turn_id="turn-1", width=800, height=600),
        monkeypatch,
    )
    capsys.readouterr()
    _run_handler(
        "PostToolUse",
        _mcp_image_payload("sid-codex-evict", turn_id="turn-2", width=400, height=300),
        monkeypatch,
    )
    assert capsys.readouterr().out == ""

    events = session_store.read_events("sid-codex-evict")
    eviction_savings = [
        e
        for e in events
        if e.get("event_type") == "MechanismSaving" and e.get("mechanism") == "image_eviction"
    ]
    assert len(eviction_savings) == 1
    assert eviction_savings[0]["applied"] is False
    assert eviction_savings[0]["measurement_kind"] == "estimated"


def test_codex_bash_pipeline_unaffected_by_mcp_branch(monkeypatch, tmp_path, capsys):
    """Regression guard: adding the MCP branch must not disturb the existing
    Bash-shaped pipeline (DLP/read-cache/dedupe), including for the
    "exec_command" tool-name spelling used elsewhere in this file — only the
    unambiguous "mcp__" prefix should ever divert away from it."""
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("rclm._config.load", lambda: {"compression": {"dedupe": True}})

    text = "result line\n" * 100
    payload = {
        "session_id": "sid-codex-examec",
        "cwd": "/repo",
        "hook_event_name": "PostToolUse",
        "model": "gpt-5.4",
        "permission_mode": "default",
        "tool_name": "exec_command",
        "tool_input": {"command": "cat build.log"},
        "tool_response": text,
        "tool_use_id": "call-1",
        "transcript_path": None,
        "turn_id": "turn-1",
    }
    _run_handler("PostToolUse", payload, monkeypatch)
    assert capsys.readouterr().out == ""
    _run_handler("PostToolUse", dict(payload, tool_use_id="call-2", turn_id="turn-2"), monkeypatch)

    output = capsys.readouterr().out.strip()
    parsed = json.loads(output)
    assert parsed["continue"] is False
    assert parsed["decision"] == "block"
    assert "Identical to the result of `Bash`" in parsed["reason"]
    assert parsed["stopReason"] == (
        "ReclaimLLM replaced this repeated tool result with a deduplication notice."
    )


def test_codex_build_tool_calls_preserves_real_tool_name(monkeypatch, tmp_path):
    from rclm.hooks import session_store
    from rclm.hooks.codex_handler import _build_tool_calls

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    events = [
        {"event_type": "PreToolUse", "turn_id": "t1", "tool_input": {"url": "http://x"}},
        {
            "event_type": "PostToolUse",
            "turn_id": "t1",
            "tool_name": "mcp__playwright__browser_take_screenshot",
            "tool_response": {"content": []},
        },
        {"event_type": "PreToolUse", "turn_id": "t2", "tool_input": {"command": "ls"}},
        {
            "event_type": "PostToolUse",
            "turn_id": "t2",
            "tool_name": "Bash",
            "tool_response": "file.py",
        },
        # No tool_name at all -> pre-change session data, defaults to "Bash".
        {"event_type": "PreToolUse", "turn_id": "t3", "tool_input": {"command": "pwd"}},
        {"event_type": "PostToolUse", "turn_id": "t3", "tool_response": "/repo"},
    ]
    tool_calls = _build_tool_calls(events)
    by_turn = {tc.tool_use_id: tc.tool_name for tc in tool_calls}
    assert by_turn["codex-turn-t1"] == "mcp__playwright__browser_take_screenshot"
    assert by_turn["codex-turn-t2"] == "Bash"
    assert by_turn["codex-turn-t3"] == "Bash"
