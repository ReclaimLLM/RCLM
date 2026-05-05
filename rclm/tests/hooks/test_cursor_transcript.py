"""Tests for rclm.hooks.cursor_transcript."""

import json
from pathlib import Path

from rclm.hooks.cursor_transcript import CursorTranscriptData, parse_transcript


def _write_transcript(path: Path, entries: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return str(path)


def test_returns_empty_for_none_path():
    data = parse_transcript(None)
    assert isinstance(data, CursorTranscriptData)
    assert data.messages == []
    assert data.tool_calls == []
    assert data.model is None


def test_returns_empty_for_missing_file(tmp_path):
    data = parse_transcript(str(tmp_path / "nonexistent.jsonl"))
    assert data.messages == []
    assert data.tool_calls == []


def test_parses_simple_conversation(tmp_path):
    entries = [
        {
            "type": "user",
            "text": "Hello",
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "type": "assistant",
            "text": "Hi there",
            "model": "gpt-4",
            "timestamp": "2024-01-01T00:00:01Z",
            "usage": {"input": 10, "output": 5},
        },
    ]
    path = _write_transcript(tmp_path / "cursor.jsonl", entries)
    data = parse_transcript(path)
    assert len(data.messages) == 2
    assert data.messages[0]["role"] == "user"
    assert data.messages[0]["content"] == "Hello"
    assert data.messages[1]["role"] == "assistant"
    assert data.messages[1]["content"] == "Hi there"
    assert data.model == "gpt-4"
    assert data.total_input_tokens == 10
    assert data.total_output_tokens == 5


def test_parses_tool_calls(tmp_path):
    entries = [
        {
            "type": "tool",
            "name": "shell",
            "input": {"command": "ls"},
            "result": "file1.txt",
            "timestamp": "2024-01-01T00:00:02Z",
            "id": "tool-1",
        }
    ]
    path = _write_transcript(tmp_path / "cursor.jsonl", entries)
    data = parse_transcript(path)
    assert len(data.tool_calls) == 1
    tc = data.tool_calls[0]
    assert tc.tool_name == "shell"
    assert tc.tool_input == {"command": "ls"}
    assert tc.tool_result == "file1.txt"
    assert tc.tool_use_id == "tool-1"


def test_parses_meta_and_alternate_format(tmp_path):
    entries = [
        {
            "type": "session_meta",
            "conversation_id": "conv-123",
            "cwd": "/home/user/project",
            "model": "claude-3-opus",
        },
        {
            "role": "user",
            "message": "Alternative format message",
            "timestamp": "2024-01-01T00:00:05Z",
        },
    ]
    path = _write_transcript(tmp_path / "cursor.jsonl", entries)
    data = parse_transcript(path)
    assert data.session_id == "conv-123"
    assert data.cwd == "/home/user/project"
    assert data.model == "claude-3-opus"
    assert len(data.messages) == 1
    assert data.messages[0]["content"] == "Alternative format message"


def test_removes_timestamp_and_user_query_tags_from_user_messages(tmp_path):
    entries = [
        {
            "role": "user",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<timestamp>Monday, May 4, 2026, 12:38 PM (UTC-4)</timestamp> "
                            "<user_query> on EditData, on homepage Bulk Edit section, "
                            "make all selection appear in the same line </user_query>"
                        ),
                    }
                ]
            },
            "timestamp": "2024-01-01T00:00:05Z",
        }
    ]
    path = _write_transcript(tmp_path / "cursor.jsonl", entries)
    data = parse_transcript(path)

    assert data.messages == [
        {
            "role": "user",
            "content": "on EditData, on homepage Bulk Edit section, make all selection appear in the same line",
            "timestamp": "2024-01-01T00:00:05Z",
        }
    ]


def test_parses_nested_content_and_write_tool(tmp_path):
    entries = [
        {
            "role": "assistant",
            "content": {
                "content": [
                    {"type": "text", "text": "I am writing a file."},
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {"path": "test.py", "content": "print('hello')"},
                        "id": "write-1",
                    },
                ]
            },
            "timestamp": "2024-01-01T00:00:10Z",
        }
    ]
    path = _write_transcript(tmp_path / "cursor.jsonl", entries)
    data = parse_transcript(path)

    # 1. Message text should be flattened
    assert len(data.messages) == 1
    assert data.messages[0]["content"] == "I am writing a file."

    # 2. Tool call should be extracted
    assert len(data.tool_calls) == 1
    assert data.tool_calls[0].tool_name == "Write"
    assert data.tool_calls[0].tool_input["path"] == "test.py"

    # 3. File diff should be generated
    assert len(data.file_diffs) == 1
    assert data.file_diffs[0].path == "test.py"
    assert data.file_diffs[0].after == "print('hello')"
    assert "+++" in data.file_diffs[0].unified_diff


def test_drops_assistant_message_that_is_only_redacted_text(tmp_path):
    entries = [
        {
            "role": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "[REDACTED]"},
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {"path": "kept.py", "content": "print('kept')"},
                        "id": "write-redacted",
                    },
                ]
            },
            "timestamp": "2024-01-01T00:00:10Z",
        },
        {
            "role": "user",
            "message": {"content": [{"type": "text", "text": "[REDACTED]"}]},
            "timestamp": "2024-01-01T00:00:11Z",
        },
        {
            "role": "assistant",
            "message": {"content": [{"type": "text", "text": "[REDACTED] plus context"}]},
            "timestamp": "2024-01-01T00:00:12Z",
        },
    ]
    path = _write_transcript(tmp_path / "cursor.jsonl", entries)
    data = parse_transcript(path)

    assert [msg["content"] for msg in data.messages] == ["[REDACTED]", "[REDACTED] plus context"]
    assert len(data.tool_calls) == 1
    assert data.tool_calls[0].tool_use_id == "write-redacted"
    assert len(data.file_diffs) == 1
    assert data.file_diffs[0].path == "kept.py"
