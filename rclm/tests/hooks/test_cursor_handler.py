"""Tests for rclm.hooks.cursor_handler."""

import base64
import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from rclm.hooks import cursor_handler, session_store


def _wrapped(command: str, session_id: str) -> str:
    encoded = base64.urlsafe_b64encode(command.encode()).decode()
    return f"rclm-compress --session-id {session_id} --encoded-command {encoded}"


def _run_handler(event_name: str, payload: dict, monkeypatch):
    """Call cursor_handler.main() with event_name as argv[1] and payload on stdin."""
    monkeypatch.setattr("sys.argv", ["rclm-cursor-hooks", event_name])
    monkeypatch.setattr("sys.stdin", StringIO(json.dumps(payload)))
    with pytest.raises(SystemExit) as exc_info:
        cursor_handler.main()
    assert exc_info.value.code == 0


def test_before_submit_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "conversation_id": "conv-1",
        "prompt": "Hello world",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("beforeSubmitPrompt", payload, monkeypatch)

    events = session_store.read_events("conv-1")
    assert len(events) == 1
    assert events[0]["event_type"] == "UserPromptSubmit"
    assert events[0]["prompt"] == "Hello world"


def test_before_submit_prompt_strips_cursor_user_wrappers(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "conversation_id": "conv-1",
        "prompt": (
            "<timestamp>Monday, May 4, 2026, 12:38 PM (UTC-4)</timestamp> "
            "<user_query> on EditData, on homepage Bulk Edit section, "
            "make all selection appear in the same line </user_query>"
        ),
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("beforeSubmitPrompt", payload, monkeypatch)

    events = session_store.read_events("conv-1")
    assert events[0]["prompt"] == (
        "on EditData, on homepage Bulk Edit section, make all selection appear in the same line"
    )


def test_before_shell_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "conversation_id": "conv-1",
        "command": "ls -la",
        "timestamp": "2024-01-01T00:00:01Z",
    }
    _run_handler("beforeShellExecution", payload, monkeypatch)

    events = session_store.read_events("conv-1")
    assert len(events) == 1
    assert events[0]["event_type"] == "PreToolUse"
    assert events[0]["tool_name"] == "shell"
    assert events[0]["tool_input"] == {"command": "ls -la"}


def test_after_file_edit(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "conversation_id": "conv-1",
        "file_path": "main.py",
        "edits": [{"old_string": "old\n", "new_string": "new\n"}],
        "timestamp": "2024-01-01T00:00:02Z",
    }
    _run_handler("afterFileEdit", payload, monkeypatch)

    events = session_store.read_events("conv-1")
    assert len(events) == 1
    assert events[0]["event_type"] == "FileEdit"
    assert events[0]["filepath"] == "main.py"
    assert events[0]["edits"] == payload["edits"]

    diffs = cursor_handler._extract_file_diffs_from_events(events)
    assert len(diffs) == 1
    assert diffs[0].path == "main.py"
    assert diffs[0].before == "old\n"
    assert diffs[0].after == "new\n"
    assert "-old" in diffs[0].unified_diff
    assert "+new" in diffs[0].unified_diff
    assert diffs[0].timestamp == "2024-01-01T00:00:02Z"


def test_after_tab_file_edit(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "conversation_id": "conv-1",
        "file_path": "tab.py",
        "edits": [{"old_string": "a = 1", "new_string": "a = 2"}],
        "timestamp": "2024-01-01T00:00:03Z",
    }
    _run_handler("afterTabFileEdit", payload, monkeypatch)

    events = session_store.read_events("conv-1")
    assert len(events) == 1
    assert events[0]["event_type"] == "FileEdit"
    assert events[0]["hook_event"] == "afterTabFileEdit"
    assert events[0]["filepath"] == "tab.py"


def test_generic_cursor_hook_event_is_recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "conversation_id": "conv-1",
        "tool_name": "read_file",
        "timestamp": "2024-01-01T00:00:04Z",
    }
    _run_handler("preToolUse", payload, monkeypatch)

    events = session_store.read_events("conv-1")
    assert len(events) == 1
    assert events[0]["event_type"] == "PreToolUse"
    assert events[0]["tool_name"] == "read_file"
    assert events[0]["tool_input"] == {}


def test_pre_tool_use_rewrites_recognized_shell_input(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("rclm._config.load", lambda: {"compression": {"enabled": True}})
    monkeypatch.setattr("rclm.hooks.compress._compress_available", lambda: True)

    _run_handler(
        "preToolUse",
        {
            "conversation_id": "conv-pre",
            "tool_name": "Shell",
            "tool_input": {"command": "cat build.log", "timeout": 10_000},
            "tool_use_id": "call-1",
        },
        monkeypatch,
    )

    output = json.loads(capsys.readouterr().out)
    assert output["updated_input"] == {
        "command": _wrapped("cat build.log", "conv-pre"),
        "timeout": 10_000,
    }


def test_post_tool_use_dedupes_structured_mcp_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("rclm._config.load", lambda: {"compression": {"dedupe": True}})
    text = "result line\n" * 100
    payload = {
        "conversation_id": "conv-mcp",
        "tool_name": "MCP:filesystem:read_file",
        "tool_input": {"path": "build.log"},
        "tool_output": {
            "content": [{"type": "text", "text": text}],
            "isError": False,
            "metadata": {"source": "filesystem"},
        },
        "tool_use_id": "call-1",
    }

    _run_handler("postToolUse", payload, monkeypatch)
    assert capsys.readouterr().out == ""
    _run_handler("postToolUse", {**payload, "tool_use_id": "call-2"}, monkeypatch)

    output = json.loads(capsys.readouterr().out)
    updated = output["updated_mcp_tool_output"]
    assert "Identical to the result" in updated["content"][0]["text"]
    assert updated["metadata"] == {"source": "filesystem"}


@patch("rclm.hooks.cursor_handler.upload_single")
def test_stop_uploads_record(mock_upload, monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    session_id = "conv-stop"
    # Seed some events
    session_store.append_event(
        session_id,
        {
            "event_type": "UserPromptSubmit",
            "prompt": "Test prompt",
            "timestamp": "2024-01-01T00:00:00Z",
        },
    )
    session_store.append_event(
        session_id,
        {
            "event_type": "PreToolUse",
            "tool_name": "shell",
            "tool_input": {"command": "echo hello"},
            "timestamp": "2024-01-01T00:00:01Z",
        },
    )
    payload = {
        "conversation_id": session_id,
        "cwd": "/test/dir",
        "timestamp": "2024-01-01T00:00:10Z",
    }

    # upload_single is an async function called with asyncio.run
    mock_upload.return_value = MagicMock()  # Mock the coroutine

    _run_handler("stop", payload, monkeypatch)

    assert mock_upload.called
    record = mock_upload.call_args[0][0]
    assert record.session_id == session_id
    assert record.cwd == "/test/dir"
    assert record.duration_s == 10.0
    assert len(record.messages) == 1
    assert record.messages[0]["content"] == "Test prompt"
    assert len(record.tool_calls) == 1
    assert record.tool_calls[0].tool_name == "shell"

    # Store should be cleaned up
    assert len(session_store.read_events(session_id)) == 0


def test_resolve_transcript_path(tmp_path):
    # Setup mock home and projects dir
    mock_home = tmp_path / "home"
    projects_dir = mock_home / ".cursor" / "projects"
    project_slug = "Users-maziz-Desktop-Project"
    session_id = "sid-123"
    transcript_dir = projects_dir / project_slug / "agent-transcripts" / session_id
    transcript_dir.mkdir(parents=True)
    transcript_file = transcript_dir / f"{session_id}.jsonl"
    transcript_file.touch()

    with patch("rclm.hooks.cursor_handler.Path.home", return_value=mock_home):
        # 1. Direct slug match
        cwd = "/Users/maziz/Desktop/Project"
        path = cursor_handler._resolve_transcript_path(cwd, session_id)
        assert path == str(transcript_file)

        # 2. Recursive match if slug is different
        path = cursor_handler._resolve_transcript_path("/different/cwd", session_id)
        assert path == str(transcript_file)

        # 3. No match
        assert cursor_handler._resolve_transcript_path(cwd, "unknown-sid") is None


@patch("rclm.hooks.cursor_handler.upload_single")
@patch("rclm.hooks.cursor_transcript.parse_transcript")
def test_stop_uses_transcript_if_available(mock_parse, mock_upload, monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    from rclm.hooks.cursor_transcript import CursorTranscriptData

    session_id = "conv-transcript"
    cwd = "/test/dir"

    # Mock transcript data
    mock_parse.return_value = CursorTranscriptData(
        messages=[
            {"role": "user", "content": "from transcript", "timestamp": "2024-01-01T00:00:05Z"}
        ],
        tool_calls=[],
        model="gpt-4-parsed",
    )
    session_store.append_event(
        session_id,
        {
            "event_type": "FileEdit",
            "filepath": "from-hook.py",
            "edits": [{"old_string": "old", "new_string": "new"}],
            "timestamp": "2024-01-01T00:00:06Z",
        },
    )

    # Mock _resolve_transcript_path to return a dummy path
    with patch(
        "rclm.hooks.cursor_handler._resolve_transcript_path", return_value="/dummy/path.jsonl"
    ):
        payload = {"conversation_id": session_id, "cwd": cwd, "timestamp": "2024-01-01T00:00:10Z"}
        mock_upload.return_value = MagicMock()

        _run_handler("stop", payload, monkeypatch)

        assert mock_upload.called
        record = mock_upload.call_args[0][0]
        assert record.session_id == session_id
        assert record.transcript_path == "/dummy/path.jsonl"
        assert record.model == "gpt-4-parsed"
        assert len(record.messages) == 1
        assert record.messages[0]["content"] == "from transcript"
        assert len(record.file_diffs) == 1
        assert record.file_diffs[0].path == "from-hook.py"


@patch("rclm.hooks.cursor_handler.upload_single")
@patch("rclm.hooks.cursor_transcript.parse_transcript")
def test_stop_uses_payload_transcript_path_and_derives_session_id(
    mock_parse, mock_upload, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    from rclm.hooks.cursor_transcript import CursorTranscriptData

    transcript_path = (
        "/Users/maziz/.cursor/projects/Users-maziz-Desktop-PreConstructionComparison/"
        "agent-transcripts/366fae37-61cf-4560-a857-d9ea15e4c052/"
        "366fae37-61cf-4560-a857-d9ea15e4c052.jsonl"
    )
    mock_parse.return_value = CursorTranscriptData(
        messages=[
            {
                "role": "user",
                "content": "from payload transcript",
                "timestamp": "2024-01-01T00:00:05Z",
            }
        ],
        tool_calls=[],
        model="cursor-model",
        cwd="/repo/from/transcript",
    )
    mock_upload.return_value = MagicMock()

    _run_handler(
        "stop",
        {"transcript_path": transcript_path, "timestamp": "2024-01-01T00:00:10Z"},
        monkeypatch,
    )

    mock_parse.assert_called_once_with(transcript_path)
    record = mock_upload.call_args[0][0]
    assert record.session_id == "366fae37-61cf-4560-a857-d9ea15e4c052"
    assert record.cwd == "/repo/from/transcript"
    assert record.transcript_path == transcript_path
    assert record.messages[0]["content"] == "from payload transcript"
    assert transcript_path not in capsys.readouterr().err
