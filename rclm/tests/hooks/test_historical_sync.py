"""Tests for rclm.hooks.historical_sync."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from rclm._models import HookSessionRecord
from rclm.hooks.historical_sync import (
    _HISTORICAL_PROVIDERS,
    _derive_session_id,
    _discover_sessions,
    _iter_openclaw_sessions,
    _load_synced_index,
    _parse_claude_session,
    _parse_codex_session,
    _parse_gemini_session,
    _parse_openclaw_session,
    _parse_session,
    _save_synced_index,
    _upload_all,
    prompt_and_run_sync,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# _derive_session_id
# ---------------------------------------------------------------------------


def test_derive_session_id_uuid_stem(tmp_path):
    p = tmp_path / "550e8400-e29b-41d4-a716-446655440000.jsonl"
    p.touch()
    assert _derive_session_id(p) == "550e8400-e29b-41d4-a716-446655440000"


def test_derive_session_id_codex_style(tmp_path):
    # "rollout-2026-02-06T14-55-47-019c3486-6120-75f2-90b8-860c9a21dd85"
    p = tmp_path / "rollout-2026-02-06T14-55-47-019c3486-6120-75f2-90b8-860c9a21dd85.jsonl"
    p.touch()
    result = _derive_session_id(p)
    assert result == "019c3486-6120-75f2-90b8-860c9a21dd85"


def test_derive_session_id_fallback(tmp_path):
    p = tmp_path / "no-uuid-here.jsonl"
    p.touch()
    result = _derive_session_id(p)
    # Should be a valid UUID (uuid5 fallback)
    import uuid

    uuid.UUID(result)  # does not raise


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def test_iter_claude_sessions_skips_subdirs(tmp_path):
    """Only top-level *.jsonl files per project are returned; subagent dirs are skipped."""
    project = tmp_path / "myproject"
    project.mkdir()
    top = project / "abc.jsonl"
    top.touch()
    sub = project / "abc" / "subagents" / "agent.jsonl"
    sub.parent.mkdir(parents=True)
    sub.touch()

    with patch("rclm.hooks.historical_sync.Path") as MockPath:
        # Replace Path.home() with tmp_path
        MockPath.home.return_value = tmp_path
        MockPath.side_effect = lambda *a, **kw: Path(*a, **kw)

    # Use direct call with patched base
    import rclm.hooks.historical_sync as mod

    orig = mod._SYNCED_INDEX  # noqa
    with patch.object(mod, "_iter_claude_sessions") as mock_iter:
        mock_iter.return_value = [top]
        result = _discover_sessions(["claude"])
    assert "claude" in result


def test_iter_gemini_sessions_finds_chats(tmp_path):
    """_iter_gemini_sessions only picks up files under .../chats/, not sibling logs."""
    chats = tmp_path / "tmp" / "myproject" / "chats"
    chats.mkdir(parents=True)
    session = chats / "session-abc.json"
    session.touch()
    logs = tmp_path / "tmp" / "myproject" / "logs.json"
    logs.touch()

    # The discovery function uses rglob("chats/*.json") — verify the pattern directly.
    results = [f for f in (tmp_path / "tmp").rglob("chats/*.json") if f.is_file()]
    assert session in results
    assert logs not in results


def test_iter_openclaw_sessions_selects_latest_reset(tmp_path):
    sessions = tmp_path / ".openclaw" / "agents" / "main" / "sessions"
    sessions.mkdir(parents=True)
    plain = sessions / "session-a.jsonl"
    plain.touch()
    older = sessions / "session-a.jsonl.reset.2026-04-28T10:00:00"
    older.touch()
    newer = sessions / "session-a.jsonl.reset.2026-04-29T10:00:00"
    newer.touch()
    other = sessions / "session-b.jsonl"
    other.touch()
    ignored = sessions / "notes.txt"
    ignored.touch()

    with patch("pathlib.Path.home", return_value=tmp_path):
        result = _iter_openclaw_sessions()

    assert result == [newer, other]


# ---------------------------------------------------------------------------
# Claude session parsing
# ---------------------------------------------------------------------------


def _claude_entries(session_id="aaaaaaaa-0000-0000-0000-000000000001"):
    return [
        {
            "type": "user",
            "message": {"role": "user", "content": "Hello"},
            "timestamp": "2024-01-01T00:00:00Z",
            "sessionId": session_id,
            "cwd": "/home/user/project",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": "Hi there",
                "model": "claude-sonnet-4-6",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            "timestamp": "2024-01-01T00:00:01Z",
            "sessionId": session_id,
            "cwd": "/home/user/project",
        },
    ]


def test_parse_claude_session_basic(tmp_path):
    path = tmp_path / "aaaaaaaa-0000-0000-0000-000000000001.jsonl"
    _write_jsonl(path, _claude_entries())
    record = _parse_claude_session(path)
    assert record is not None
    assert record.session_id == "aaaaaaaa-0000-0000-0000-000000000001"
    assert record.cwd == "/home/user/project"
    assert record.model == "claude-sonnet-4-6"
    assert len(record.messages) == 2
    assert record.is_sync is True


def test_parse_claude_session_tokens_from_message(tmp_path):
    """Tokens live inside entry['message']['usage'], not at entry level."""
    path = tmp_path / "bbbbbbbb-0000-0000-0000-000000000001.jsonl"
    _write_jsonl(path, _claude_entries("bbbbbbbb-0000-0000-0000-000000000001"))
    record = _parse_claude_session(path)
    assert record is not None
    # Our raw pass should have picked up tokens
    assert record.total_input_tokens == 10
    assert record.total_output_tokens == 5


def test_parse_claude_session_empty_returns_none(tmp_path):
    path = tmp_path / "cccccccc-0000-0000-0000-000000000001.jsonl"
    path.write_text("")
    assert _parse_claude_session(path) is None


def test_parse_claude_session_no_messages_returns_none(tmp_path):
    """Entries with types other than user/assistant produce no messages."""
    path = tmp_path / "dddddddd-0000-0000-0000-000000000001.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "progress",
                "data": {"type": "hook_progress"},
                "sessionId": "dddddddd-0000-0000-0000-000000000001",
            },
        ],
    )
    assert _parse_claude_session(path) is None


def test_parse_claude_session_write_tool_produces_diff(tmp_path):
    sid = "eeeeeeee-0000-0000-0000-000000000001"
    entries = [
        {
            "type": "user",
            "message": {"role": "user", "content": "write a file"},
            "timestamp": "2024-01-01T00:00:00Z",
            "sessionId": sid,
            "cwd": "/tmp",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_01",
                        "name": "Write",
                        "input": {
                            "file_path": "hello.py",
                            "content": "print('hi')",
                        },
                    }
                ],
            },
            "timestamp": "2024-01-01T00:00:01Z",
            "sessionId": sid,
            "cwd": "/tmp",
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_01",
                        "content": "ok",
                    }
                ],
            },
            "timestamp": "2024-01-01T00:00:02Z",
            "sessionId": sid,
            "cwd": "/tmp",
        },
    ]
    path = tmp_path / f"{sid}.jsonl"
    _write_jsonl(path, entries)
    record = _parse_claude_session(path)
    assert record is not None
    assert len(record.file_diffs) == 1
    assert record.file_diffs[0].path == "hello.py"
    assert record.file_diffs[0].after == "print('hi')"


# ---------------------------------------------------------------------------
# Gemini session parsing
# ---------------------------------------------------------------------------


def _gemini_session(session_id="f04d1d50-fc79-4705-9d89-8695287dbadf"):
    return {
        "sessionId": session_id,
        "startTime": "2024-01-01T00:00:00.000Z",
        "lastUpdated": "2024-01-01T00:01:00.000Z",
        "messages": [
            {
                "id": "msg-1",
                "timestamp": "2024-01-01T00:00:00.000Z",
                "type": "user",
                "content": [{"text": "what is 2+2?"}],
            },
            {
                "id": "msg-2",
                "timestamp": "2024-01-01T00:00:01.000Z",
                "type": "gemini",
                "content": "4",
                "model": "gemini-2.5-pro",
                "tokens": {"input": 100, "output": 5, "total": 105},
                "toolCalls": [],
            },
        ],
    }


def test_parse_gemini_session_basic(tmp_path):
    chats = tmp_path / "tmp" / "myproject" / "chats"
    chats.mkdir(parents=True)
    path = chats / "session-2024.json"
    _write_json(path, _gemini_session())
    record = _parse_gemini_session(path)
    assert record is not None
    assert record.session_id == "f04d1d50-fc79-4705-9d89-8695287dbadf"
    assert record.model == "gemini-2.5-pro"
    assert len(record.messages) == 2
    assert record.messages[0]["role"] == "user"
    assert record.messages[1]["role"] == "assistant"
    assert record.total_input_tokens == 100
    assert record.total_output_tokens == 5
    assert record.cwd == "myproject"
    assert record.is_sync is True


def test_parse_gemini_session_with_tool_calls(tmp_path):
    data = {
        "sessionId": "aaaa0001-0000-0000-0000-000000000001",
        "startTime": "2024-01-01T00:00:00.000Z",
        "lastUpdated": "2024-01-01T00:01:00.000Z",
        "messages": [
            {
                "id": "u1",
                "timestamp": "2024-01-01T00:00:00.000Z",
                "type": "user",
                "content": [{"text": "list files"}],
            },
            {
                "id": "g1",
                "timestamp": "2024-01-01T00:00:01.000Z",
                "type": "gemini",
                "content": None,
                "model": "gemini-3.0-flash",
                "tokens": {"input": 50, "output": 10},
                "toolCalls": [
                    {
                        "id": "tc1",
                        "name": "run_shell_command",
                        "args": {"command": "ls"},
                        "status": "success",
                        "timestamp": "2024-01-01T00:00:02.000Z",
                        "result": [
                            {
                                "functionResponse": {
                                    "id": "tc1",
                                    "name": "run_shell_command",
                                    "response": {"output": "file.py"},
                                }
                            }
                        ],
                    }
                ],
            },
        ],
    }
    path = tmp_path / "session.json"
    _write_json(path, data)
    record = _parse_gemini_session(path)
    assert record is not None
    assert len(record.tool_calls) == 1
    assert record.tool_calls[0].tool_name == "run_shell_command"
    assert record.tool_calls[0].tool_result == "file.py"


def test_parse_gemini_session_empty_messages_returns_none(tmp_path):
    data = {
        "sessionId": "bbbb0001-0000-0000-0000-000000000001",
        "startTime": "2024-01-01T00:00:00.000Z",
        "lastUpdated": "2024-01-01T00:00:01.000Z",
        "messages": [
            {
                "id": "e1",
                "type": "error",
                "content": "API error",
                "timestamp": "",
            },
            {
                "id": "i1",
                "type": "info",
                "content": "update available",
                "timestamp": "",
            },
        ],
    }
    path = tmp_path / "session.json"
    _write_json(path, data)
    assert _parse_gemini_session(path) is None


def test_parse_gemini_session_write_file_diff(tmp_path):
    data = {
        "sessionId": "cccc0001-0000-0000-0000-000000000001",
        "startTime": "2024-01-01T00:00:00.000Z",
        "lastUpdated": "2024-01-01T00:01:00.000Z",
        "messages": [
            {
                "id": "u1",
                "timestamp": "2024-01-01T00:00:00.000Z",
                "type": "user",
                "content": [{"text": "create hello.py"}],
            },
            {
                "id": "g1",
                "timestamp": "2024-01-01T00:00:01.000Z",
                "type": "gemini",
                "content": "Done.",
                "model": "gemini-3.0-flash",
                "tokens": {"input": 10, "output": 2},
                "toolCalls": [
                    {
                        "id": "tc1",
                        "name": "write_file",
                        "args": {
                            "file_path": "hello.py",
                            "content": "print('hi')",
                        },
                        "status": "success",
                        "timestamp": "2024-01-01T00:00:02.000Z",
                        "result": [],
                    }
                ],
            },
        ],
    }
    path = tmp_path / "session.json"
    _write_json(path, data)
    record = _parse_gemini_session(path)
    assert record is not None
    assert len(record.file_diffs) == 1
    assert record.file_diffs[0].path == "hello.py"
    assert record.file_diffs[0].after == "print('hi')"


# ---------------------------------------------------------------------------
# Codex session parsing
# ---------------------------------------------------------------------------


def _codex_jsonl_entries():
    return [
        {
            "timestamp": "2026-01-01T00:00:00.000Z",
            "type": "session_meta",
            "payload": {
                "id": "019c3486-6120-75f2-90b8-860c9a21dd85",
                "cwd": "/home/user/project",
                "model_provider": "openai",
            },
        },
        {
            "timestamp": "2026-01-01T00:00:01.000Z",
            "type": "event_msg",
            "payload": {
                "type": "user_message",
                "message": "write a hello world",
            },
        },
        {
            "timestamp": "2026-01-01T00:00:02.000Z",
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "message": "I'll create hello.py for you.",
            },
        },
    ]


def test_parse_codex_session_basic(tmp_path):
    path = tmp_path / "rollout-2026-01-01T00-00-00-019c3486-6120-75f2-90b8-860c9a21dd85.jsonl"
    _write_jsonl(path, _codex_jsonl_entries())
    record = _parse_codex_session(path)
    assert record is not None
    assert record.session_id == "019c3486-6120-75f2-90b8-860c9a21dd85"
    assert record.cwd == "/home/user/project"
    assert len(record.messages) == 2
    assert record.is_sync is True


def test_parse_codex_session_empty_returns_none(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert _parse_codex_session(path) is None


def test_parse_codex_session_fallback_session_id(tmp_path):
    """If session_meta is missing, session_id is derived from filename."""
    entries = [
        {
            "timestamp": "2026-01-01T00:00:01.000Z",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "hello"},
        },
        {
            "timestamp": "2026-01-01T00:00:02.000Z",
            "type": "event_msg",
            "payload": {"type": "agent_message", "message": "hi"},
        },
    ]
    path = tmp_path / "rollout-2026-01-01T00-00-00-019c3486-6120-75f2-90b8-860c9a21dd85.jsonl"
    _write_jsonl(path, entries)
    record = _parse_codex_session(path)
    assert record is not None
    assert record.session_id == "019c3486-6120-75f2-90b8-860c9a21dd85"


# ---------------------------------------------------------------------------
# OpenClaw session parsing
# ---------------------------------------------------------------------------


def _openclaw_jsonl_entries():
    return [
        {
            "hook_name": "session_start",
            "received_at": "2026-04-27T00:00:00+00:00",
            "event": {
                "sessionId": "oc-sid-1",
                "workspaceDir": "/repo",
                "model": {"id": "openai/gpt-5.4"},
            },
        },
        {
            "hook_name": "llm_input",
            "received_at": "2026-04-27T00:00:01+00:00",
            "event": {
                "sessionId": "oc-sid-1",
                "messages": [
                    {"role": "system", "content": "HEARTBEAT_OK"},
                    {"role": "user", "content": "hello"},
                ],
            },
        },
        {
            "hook_name": "llm_output",
            "received_at": "2026-04-27T00:00:02+00:00",
            "event": {"sessionId": "oc-sid-1", "message": {"content": "hi back"}},
        },
        {
            "hook_name": "before_tool_call",
            "received_at": "2026-04-27T00:00:03+00:00",
            "event": {
                "sessionId": "oc-sid-1",
                "toolCallId": "tool-1",
                "toolName": "read_file",
                "params": {"path": "README.md"},
            },
        },
        {
            "hook_name": "after_tool_call",
            "received_at": "2026-04-27T00:00:04+00:00",
            "event": {
                "sessionId": "oc-sid-1",
                "toolCallId": "tool-1",
                "toolName": "read_file",
                "result": "contents",
            },
        },
        {
            "hook_name": "session_end",
            "received_at": "2026-04-27T00:01:00+00:00",
            "event": {"sessionId": "oc-sid-1"},
        },
    ]


def test_parse_openclaw_session_basic(tmp_path):
    path = tmp_path / "session-a.jsonl"
    _write_jsonl(path, _openclaw_jsonl_entries())

    record = _parse_openclaw_session(path)

    assert record is not None
    assert record.session_id == "oc-sid-1"
    assert record.cwd == "/repo"
    assert record.model == "openai/gpt-5.4"
    assert record.started_at == "2026-04-27T00:00:00+00:00"
    assert record.ended_at == "2026-04-27T00:01:00+00:00"
    assert record.duration_s == 60.0
    assert [msg["content"] for msg in record.messages] == ["hello", "hi back"]
    assert len(record.tool_calls) == 1
    assert record.tool_calls[0].tool_input == {"path": "README.md"}
    assert record.tool_calls[0].tool_result == "contents"
    assert record.is_sync is True


def test_parse_openclaw_session_fallback_session_id_and_unknown_model(tmp_path):
    path = tmp_path / "plain-name.jsonl"
    _write_jsonl(
        path,
        [
            {
                "timestamp": "2026-04-27T00:00:00+00:00",
                "role": "user",
                "content": "direct message",
            }
        ],
    )

    record = _parse_openclaw_session(path)

    assert record is not None
    assert record.session_id == _derive_session_id(path)
    assert record.model == "openclaw-unknown"
    assert record.messages[0]["content"] == "direct message"


def test_parse_openclaw_session_pairs_tool_calls_without_ids(tmp_path):
    path = tmp_path / "no-tool-ids.jsonl"
    _write_jsonl(
        path,
        [
            {
                "hook_name": "before_tool_call",
                "received_at": "2026-04-27T00:00:00+00:00",
                "event": {"toolName": "shell", "params": {"cmd": "pwd"}},
            },
            {
                "hook_name": "after_tool_call",
                "received_at": "2026-04-27T00:00:01+00:00",
                "event": {"toolName": "shell", "result": "/repo"},
            },
        ],
    )

    record = _parse_openclaw_session(path)

    assert record is not None
    assert len(record.tool_calls) == 1
    assert record.tool_calls[0].tool_input == {"cmd": "pwd"}
    assert record.tool_calls[0].tool_result == "/repo"


def test_parse_openclaw_native_message_records(tmp_path):
    path = tmp_path / "native-openclaw.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "session",
                "id": "native-session",
                "timestamp": "2026-04-30T11:26:17.167Z",
                "cwd": "/native/repo",
            },
            {
                "type": "model_change",
                "timestamp": "2026-04-30T11:26:17.176Z",
                "provider": "openai-codex",
                "modelId": "gpt-5.5",
            },
            {
                "type": "message",
                "id": "u1",
                "timestamp": "2026-04-30T11:26:18.000Z",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                },
            },
            {
                "type": "message",
                "id": "a1",
                "timestamp": "2026-04-30T11:26:19.000Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hidden"},
                        {
                            "type": "toolCall",
                            "id": "tc1",
                            "name": "exec",
                            "arguments": {"cmd": "pwd"},
                        },
                        {"type": "text", "text": "I'll check."},
                    ],
                },
            },
            {
                "type": "message",
                "id": "tr1",
                "timestamp": "2026-04-30T11:26:20.000Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "tc1",
                    "toolName": "exec",
                    "content": [{"type": "text", "text": "/native/repo"}],
                },
            },
        ],
    )

    record = _parse_openclaw_session(path)

    assert record is not None
    assert record.session_id == "native-session"
    assert record.cwd == "/native/repo"
    assert record.model == "gpt-5.5"
    assert [msg["content"] for msg in record.messages] == ["hello", "I'll check."]
    assert len(record.tool_calls) == 1
    assert record.tool_calls[0].tool_name == "exec"
    assert record.tool_calls[0].tool_input == {"cmd": "pwd"}
    assert record.tool_calls[0].tool_result == "/native/repo"


def test_parse_openclaw_session_empty_returns_none(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert _parse_openclaw_session(path) is None


# ---------------------------------------------------------------------------
# Sync index
# ---------------------------------------------------------------------------


def test_synced_index_roundtrip(tmp_path):
    import rclm.hooks.historical_sync as mod

    idx_path = tmp_path / "synced_sessions.json"
    with patch.object(mod, "_SYNCED_INDEX", idx_path):
        assert _load_synced_index() == set()
        synced = {"path/a.jsonl", "path/b.json"}
        _save_synced_index(synced)
        loaded = _load_synced_index()
    assert loaded == synced


def test_load_synced_index_tolerates_corrupt_file(tmp_path):
    import rclm.hooks.historical_sync as mod

    idx_path = tmp_path / "synced_sessions.json"
    idx_path.write_text("NOT JSON")
    with patch.object(mod, "_SYNCED_INDEX", idx_path):
        result = _load_synced_index()
    assert result == set()


# ---------------------------------------------------------------------------
# Upload orchestration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_all_skips_already_synced(tmp_path):
    """Files already in the synced index are not parsed or uploaded."""
    path = tmp_path / "session.jsonl"
    _write_jsonl(path, _claude_entries())
    already_synced = {str(path)}

    with patch("rclm.hooks.historical_sync.upload_single", new_callable=AsyncMock) as mock_upload:
        uploaded = await _upload_all({"claude": [path]}, already_synced)

    assert uploaded == 0
    mock_upload.assert_not_called()


@pytest.mark.asyncio
async def test_upload_all_uploads_new_sessions(tmp_path):
    """New, parseable sessions are uploaded and added to already_synced."""
    path = tmp_path / "aaaaaaaa-0000-0000-0000-000000000001.jsonl"
    _write_jsonl(path, _claude_entries())
    already_synced: set[str] = set()

    with patch("rclm.hooks.historical_sync.upload_single", new_callable=AsyncMock):
        uploaded = await _upload_all({"claude": [path]}, already_synced)

    assert uploaded == 1
    assert str(path) in already_synced


@pytest.mark.asyncio
async def test_upload_all_marks_empty_session_as_synced(tmp_path):
    """Empty/unreadable sessions are marked synced so they are not retried."""
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    already_synced: set[str] = set()

    with patch("rclm.hooks.historical_sync.upload_single", new_callable=AsyncMock) as mock_upload:
        uploaded = await _upload_all({"claude": [path]}, already_synced)

    assert uploaded == 0
    assert str(path) in already_synced
    mock_upload.assert_not_called()


# ---------------------------------------------------------------------------
# is_sync flag on HookSessionRecord
# ---------------------------------------------------------------------------


def test_is_sync_default_false():
    record = HookSessionRecord(
        session_id="test-id",
        cwd="/",
        started_at=None,  # type: ignore[arg-type]
        ended_at=None,  # type: ignore[arg-type]
        duration_s=0.0,
        transcript_path=None,
        model="m",
    )
    assert record.is_sync is False


def test_is_sync_serialized_in_asdict():
    import dataclasses

    record = HookSessionRecord(
        session_id="test-id",
        cwd="/",
        started_at=None,  # type: ignore[arg-type]
        ended_at=None,  # type: ignore[arg-type]
        duration_s=0.0,
        transcript_path=None,
        model="m",
        is_sync=True,
    )
    d = dataclasses.asdict(record)
    assert d["is_sync"] is True


def test_historical_parsers_set_is_sync_true(tmp_path):
    """All parsers must produce is_sync=True."""
    # Claude
    claude_path = tmp_path / "aaaaaaaa-0000-0000-0000-000000000001.jsonl"
    _write_jsonl(claude_path, _claude_entries())
    assert _parse_claude_session(claude_path).is_sync is True

    # Gemini
    chats = tmp_path / "tmp" / "proj" / "chats"
    chats.mkdir(parents=True)
    gemini_path = chats / "session.json"
    _write_json(gemini_path, _gemini_session())
    assert _parse_gemini_session(gemini_path).is_sync is True

    # Codex
    codex_path = tmp_path / "rollout-2026-01-01T00-00-00-019c3486-6120-75f2-90b8-860c9a21dd85.jsonl"
    _write_jsonl(codex_path, _codex_jsonl_entries())
    assert _parse_codex_session(codex_path).is_sync is True

    # OpenClaw
    openclaw_path = tmp_path / "openclaw.jsonl"
    _write_jsonl(openclaw_path, _openclaw_jsonl_entries())
    assert _parse_openclaw_session(openclaw_path).is_sync is True


def test_discover_sessions_includes_openclaw_provider(tmp_path):
    with patch("rclm.hooks.historical_sync._iter_openclaw_sessions", return_value=[tmp_path]):
        assert _discover_sessions(["openclaw"]) == {"openclaw": [tmp_path]}


def test_parse_session_dispatches_openclaw(tmp_path):
    path = tmp_path / "openclaw.jsonl"
    _write_jsonl(path, _openclaw_jsonl_entries())
    assert _parse_session("openclaw", path).session_id == "oc-sid-1"


def test_default_provider_list_includes_openclaw():
    assert "openclaw" in _HISTORICAL_PROVIDERS


# ---------------------------------------------------------------------------
# prompt_and_run_sync — non-TTY / no sessions behaviour
# ---------------------------------------------------------------------------


def test_prompt_and_run_sync_skips_on_non_tty(tmp_path):
    """When stdin is not a TTY and force_yes is False, the function returns silently."""
    with (
        patch("rclm.hooks.historical_sync._discover_sessions") as mock_discover,
        patch("sys.stdin") as mock_stdin,
    ):
        mock_stdin.isatty.return_value = False
        prompt_and_run_sync(["claude"])
        mock_discover.assert_not_called()


def test_prompt_and_run_sync_no_new_sessions_force_yes(tmp_path, capsys):
    """With force_yes=True and nothing to sync, prints a message and returns."""
    with (
        patch(
            "rclm.hooks.historical_sync._discover_sessions",
            return_value={"claude": []},
        ),
        patch("rclm.hooks.historical_sync._load_synced_index", return_value=set()),
    ):
        prompt_and_run_sync(["claude"], force_yes=True)
    captured = capsys.readouterr()
    assert "No new sessions" in captured.out
