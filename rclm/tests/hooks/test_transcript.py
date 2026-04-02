"""Tests for rclm.hooks.transcript."""
import json
from pathlib import Path

from rclm.hooks.transcript import TranscriptData, parse_transcript


def _write_transcript(path: Path, entries: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return str(path)


def test_returns_empty_for_none_path():
    data = parse_transcript(None)
    assert isinstance(data, TranscriptData)
    assert data.messages == []
    assert data.tool_calls == []
    assert data.model is None
    assert data.total_input_tokens is None


def test_returns_empty_for_missing_file(tmp_path):
    data = parse_transcript(str(tmp_path / "nonexistent.jsonl"))
    assert data.messages == []
    assert data.tool_calls == []


def test_parses_simple_conversation(tmp_path):
    entries = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Hello"},
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": "Hi there"},
            "timestamp": "2024-01-01T00:00:01Z",
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    ]
    path = _write_transcript(tmp_path / "t.jsonl", entries)
    data = parse_transcript(path)
    assert len(data.messages) == 2
    assert data.messages[0]["role"] == "user"
    assert data.messages[1]["role"] == "assistant"
    assert data.model == "claude-sonnet-4-6"
    assert data.total_input_tokens == 10
    assert data.total_output_tokens == 5


def test_skips_malformed_json_lines(tmp_path):
    raw = '{"type": "user", "message": {"role": "user", "content": "Hi"}, "timestamp": ""}\nNOT_JSON\n'
    p = tmp_path / "t.jsonl"
    p.write_text(raw)
    data = parse_transcript(str(p))
    assert len(data.messages) == 1


def test_pairs_tool_use_with_tool_result(tmp_path):
    entries = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu-123",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ],
            },
            "timestamp": "2024-01-01T00:00:01Z",
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 20, "output_tokens": 8},
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu-123",
                        "content": "file.py\n",
                    }
                ],
            },
            "timestamp": "2024-01-01T00:00:02Z",
        },
    ]
    path = _write_transcript(tmp_path / "t.jsonl", entries)
    data = parse_transcript(path)
    assert len(data.tool_calls) == 1
    tc = data.tool_calls[0]
    assert tc.tool_use_id == "tu-123"
    assert tc.tool_name == "Bash"
    assert tc.tool_input == {"command": "ls"}
    assert tc.tool_result == "file.py\n"


def test_tool_use_without_matching_result_has_none_result(tmp_path):
    entries = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu-999", "name": "Read", "input": {"file_path": "/a"}}
                ],
            },
            "timestamp": "2024-01-01T00:00:01Z",
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        },
    ]
    path = _write_transcript(tmp_path / "t.jsonl", entries)
    data = parse_transcript(path)
    assert len(data.tool_calls) == 1
    assert data.tool_calls[0].tool_result is None


def test_accumulates_tokens_from_multiple_assistant_entries(tmp_path):
    entries = [
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": "First"},
            "timestamp": "2024-01-01T00:00:01Z",
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "More"},
            "timestamp": "2024-01-01T00:00:02Z",
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": "Second"},
            "timestamp": "2024-01-01T00:00:03Z",
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 200, "output_tokens": 75},
        },
    ]
    path = _write_transcript(tmp_path / "t.jsonl", entries)
    data = parse_transcript(path)
    assert data.total_input_tokens == 300
    assert data.total_output_tokens == 125
    assert data.model == "claude-sonnet-4-6"  # from first assistant entry


def test_model_extracted_from_first_assistant_entry(tmp_path):
    entries = [
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": "A"},
            "timestamp": "2024-01-01T00:00:01Z",
            "model": "claude-haiku-4-5",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": "B"},
            "timestamp": "2024-01-01T00:00:02Z",
            "model": "claude-opus-4-6",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    ]
    path = _write_transcript(tmp_path / "t.jsonl", entries)
    data = parse_transcript(path)
    assert data.model == "claude-haiku-4-5"
