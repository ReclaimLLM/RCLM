"""Tests for rclm.hooks.antigravity_transcript."""

from __future__ import annotations

import json

from rclm.hooks.antigravity_transcript import parse_transcript


def _write(tmp_path, lines: list[dict]):
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def test_missing_path_returns_empty():
    data = parse_transcript(None)
    assert data.messages == []
    assert data.tool_calls == []


def test_missing_file_returns_empty(tmp_path):
    data = parse_transcript(str(tmp_path / "does-not-exist.jsonl"))
    assert data.messages == []
    assert data.tool_calls == []


def test_malformed_line_is_skipped(tmp_path):
    path = tmp_path / "transcript.jsonl"
    path.write_text(
        'not json\n{"source": "USER_EXPLICIT", "type": "USER_INPUT", '
        '"created_at": "t0", "content": "hi"}\n',
        encoding="utf-8",
    )
    data = parse_transcript(str(path))
    assert len(data.messages) == 1
    assert data.messages[0]["content"] == "hi"


def test_conversation_history_marker_with_no_content_is_skipped(tmp_path):
    path = _write(
        tmp_path,
        [{"step_index": 0, "source": "SYSTEM", "type": "CONVERSATION_HISTORY", "created_at": "t0"}],
    )
    data = parse_transcript(str(path))
    assert data.messages == []


def test_user_input_becomes_user_message(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "created_at": "2026-08-05T09:31:42Z",
                "content": "<USER_REQUEST>\ngive me a plan\n</USER_REQUEST>",
            }
        ],
    )
    data = parse_transcript(str(path))
    assert data.messages == [
        {
            "role": "user",
            "timestamp": "2026-08-05T09:31:42Z",
            "content": "<USER_REQUEST>\ngive me a plan\n</USER_REQUEST>",
        }
    ]


def test_checkpoint_becomes_system_message(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "step_index": 4,
                "source": "SYSTEM",
                "type": "CHECKPOINT",
                "created_at": "t4",
                "content": "{{ CHECKPOINT 0 }} summary...",
            }
        ],
    )
    data = parse_transcript(str(path))
    assert data.messages == [
        {"role": "system", "timestamp": "t4", "content": "{{ CHECKPOINT 0 }} summary..."}
    ]


def test_tool_call_paired_with_following_result_line(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "step_index": 2,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "t2",
                "thinking": "I should list the directory first.",
                "tool_calls": [
                    {"name": "list_dir", "args": {"DirectoryPath": '"/repo"'}},
                ],
            },
            {
                "step_index": 3,
                "source": "MODEL",
                "type": "LIST_DIRECTORY",
                "created_at": "t3",
                "content": "Summary: 2 files.",
            },
        ],
    )
    data = parse_transcript(str(path))

    # The tool-call step becomes one assistant message (thinking preserved,
    # per the capture requirement) and the following result line is *not*
    # duplicated as its own message -- it's folded into the tool_result.
    assert len(data.messages) == 1
    assert data.messages[0]["role"] == "assistant"
    assert data.messages[0]["thinking"] == "I should list the directory first."

    assert len(data.tool_calls) == 1
    call = data.tool_calls[0]
    assert call.tool_use_id == "2:0"
    assert call.tool_name == "list_dir"
    assert call.tool_input == {"DirectoryPath": '"/repo"'}
    assert call.tool_result == "Summary: 2 files."
    assert call.timestamp == "t3"


def test_multiple_tool_calls_in_one_step_only_pairs_the_first(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "step_index": 0,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "t0",
                "tool_calls": [
                    {"name": "list_dir", "args": {"DirectoryPath": '"/a"'}},
                    {"name": "list_dir", "args": {"DirectoryPath": '"/b"'}},
                ],
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "LIST_DIRECTORY",
                "created_at": "t1",
                "content": "result for /a",
            },
        ],
    )
    data = parse_transcript(str(path))

    assert len(data.tool_calls) == 2
    first, second = data.tool_calls
    assert first.tool_use_id == "0:0"
    assert first.tool_result == "result for /a"
    assert second.tool_use_id == "0:1"
    assert second.tool_result is None


def test_back_to_back_tool_call_steps_are_not_cross_paired(tmp_path):
    """A tool-call entry immediately followed by *another* tool-call entry
    (no plain result line in between) must not treat the second step as the
    first step's result."""
    path = _write(
        tmp_path,
        [
            {
                "step_index": 0,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "t0",
                "tool_calls": [{"name": "list_dir", "args": {}}],
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "t1",
                "tool_calls": [{"name": "view_file", "args": {}}],
            },
        ],
    )
    data = parse_transcript(str(path))

    assert len(data.tool_calls) == 2
    assert data.tool_calls[0].tool_result is None
    assert data.tool_calls[1].tool_result is None
    assert len(data.messages) == 2
