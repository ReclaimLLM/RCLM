"""Tests for parsing historical GitHub Copilot CLI event logs."""

import json

from rclm.hooks.copilot_transcript import parse_transcript


def test_parse_transcript_captures_user_and_assistant_messages_without_system_data(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [
        {
            "type": "session.start",
            "timestamp": "2026-10-08T18:00:00Z",
            "data": {"sessionId": "copilot-session"},
        },
        {
            "type": "system.message",
            "timestamp": "2026-10-08T18:00:01Z",
            "data": {"content": "private system prompt"},
        },
        {
            "type": "user.message",
            "timestamp": "2026-10-08T18:00:02Z",
            "data": {"content": "Explain this change"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-10-08T18:00:03Z",
            "data": {"content": "Here is the explanation.", "model": "gpt-4.1"},
        },
        {
            "type": "assistant.message",
            "timestamp": "2026-10-08T18:00:04Z",
            "data": {"content": ""},
        },
        {
            "type": "model.response",
            "timestamp": "2026-10-08T18:00:05Z",
            "data": {"response": {"content": "internal model response"}},
        },
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    transcript = parse_transcript(str(path))

    assert transcript.session_id == "copilot-session"
    assert transcript.model == "gpt-4.1"
    assert transcript.started_at == "2026-10-08T18:00:00Z"
    assert transcript.ended_at == "2026-10-08T18:00:05Z"
    assert transcript.messages == [
        {
            "role": "user",
            "content": "Explain this change",
            "timestamp": "2026-10-08T18:00:02Z",
        },
        {
            "role": "assistant",
            "content": "Here is the explanation.",
            "timestamp": "2026-10-08T18:00:03Z",
        },
    ]


def test_parse_transcript_pairs_tool_calls_and_reads_hook_metadata(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [
        {
            "type": "hook.start",
            "timestamp": "2026-10-08T18:00:00Z",
            "data": {
                "hookType": "userPromptSubmitted",
                "input": {"sessionId": "copilot-session", "cwd": "/workspace/project"},
            },
        },
        {
            "type": "tool.execution_start",
            "timestamp": "2026-10-08T18:00:01Z",
            "data": {
                "toolCallId": "tool-1",
                "toolName": "read_file",
                "arguments": {"path": "README.md"},
            },
        },
        {
            "type": "tool.execution_complete",
            "timestamp": "2026-10-08T18:00:02Z",
            "data": {
                "toolCallId": "tool-1",
                "success": True,
                "result": {"content": "read result"},
            },
        },
        {
            "type": "tool.execution_start",
            "timestamp": "2026-10-08T18:00:03Z",
            "data": {
                "toolCallId": "tool-2",
                "toolName": "run_command",
                "arguments": {"command": "pytest"},
            },
        },
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    transcript = parse_transcript(str(path))

    assert transcript.session_id == "copilot-session"
    assert transcript.cwd == "/workspace/project"
    assert len(transcript.tool_calls) == 2
    assert transcript.tool_calls[0].tool_use_id == "tool-1"
    assert transcript.tool_calls[0].tool_name == "read_file"
    assert transcript.tool_calls[0].tool_input == {"path": "README.md"}
    assert transcript.tool_calls[0].tool_result == {"content": "read result"}
    assert transcript.tool_calls[1].tool_use_id == "tool-2"
    assert transcript.tool_calls[1].tool_result is None
