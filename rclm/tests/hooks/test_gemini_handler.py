"""Tests for rclm.hooks.gemini_handler."""

import json
from io import StringIO

import pytest
from jsonschema import validate

from rclm._models import HookSessionRecord
from rclm.hooks import gemini_handler, session_store

GEMINI_AFTER_TOOL_INPUT_SCHEMA = {
    "type": "object",
    "required": [
        "session_id",
        "transcript_path",
        "cwd",
        "hook_event_name",
        "timestamp",
        "tool_name",
        "tool_input",
        "tool_response",
    ],
    "properties": {
        "session_id": {"type": "string"},
        "transcript_path": {"type": "string"},
        "cwd": {"type": "string"},
        "hook_event_name": {"const": "AfterTool"},
        "timestamp": {"type": "string"},
        "tool_name": {"type": "string"},
        "tool_input": {"type": "object"},
        "tool_response": {
            "type": "object",
            "properties": {
                "llmContent": {"type": "string"},
                "returnDisplay": {"type": "string"},
                "error": {"type": ["string", "null"]},
            },
        },
        "mcp_context": {"type": "object"},
        "original_request_name": {"type": "string"},
    },
}

GEMINI_COMMON_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "systemMessage": {"type": "string"},
        "suppressOutput": {"type": "boolean"},
        "continue": {"type": "boolean"},
        "stopReason": {"type": "string"},
        "decision": {"enum": ["allow", "deny", "block"]},
        "reason": {"type": "string"},
        "hookSpecificOutput": {"type": "object"},
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_handler(event_name: str, payload: dict, monkeypatch, *, capsys=None) -> str:
    """Call gemini_handler.main() with event_name as argv[1] and payload on stdin.

    Returns captured stdout (should always be '{}\\n').
    """
    monkeypatch.setattr("sys.argv", ["rclm-gemini-hooks", event_name])
    monkeypatch.setattr("sys.stdin", StringIO(json.dumps(payload)))
    with pytest.raises(SystemExit) as exc_info:
        gemini_handler.main()
    assert exc_info.value.code == 0
    if capsys is not None:
        return capsys.readouterr().out
    return ""


# ---------------------------------------------------------------------------
# SessionStart
# ---------------------------------------------------------------------------


def test_session_start_appends_event(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "session_id": "gsid-1",
        "cwd": "/projects/gemini",
        "timestamp": "2024-01-01T00:00:00Z",
        "hook_event_name": "SessionStart",
    }
    _run_handler("SessionStart", payload, monkeypatch)

    events = session_store.read_events("gsid-1")
    assert len(events) == 1
    assert events[0]["event_type"] == "SessionStart"
    assert events[0]["cwd"] == "/projects/gemini"


# ---------------------------------------------------------------------------
# BeforeAgent / AfterAgent
# ---------------------------------------------------------------------------


def test_before_agent_appends_user_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "session_id": "gsid-2",
        "prompt": "refactor this function",
        "timestamp": "2024-01-01T00:00:01Z",
        "hook_event_name": "BeforeAgent",
    }
    _run_handler("BeforeAgent", payload, monkeypatch)

    events = session_store.read_events("gsid-2")
    assert events[-1]["event_type"] == "BeforeAgent"
    assert events[-1]["prompt"] == "refactor this function"


def test_after_agent_appends_assistant_response(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "session_id": "gsid-3",
        "prompt_response": "Here is the refactored code...",
        "timestamp": "2024-01-01T00:00:02Z",
        "hook_event_name": "AfterAgent",
    }
    _run_handler("AfterAgent", payload, monkeypatch)

    events = session_store.read_events("gsid-3")
    assert events[-1]["event_type"] == "AfterAgent"
    assert events[-1]["prompt_response"] == "Here is the refactored code..."


# ---------------------------------------------------------------------------
# AfterTool — response normalisation
# ---------------------------------------------------------------------------


def test_after_tool_normalises_dict_response_prefer_return_display(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "session_id": "gsid-4",
        "tool_name": "read_file",
        "tool_input": {"file_path": "/src/main.py"},
        "tool_response": {
            "llmContent": "raw content",
            "returnDisplay": "display content",
            "error": None,
        },
        "timestamp": "2024-01-01T00:00:03Z",
        "hook_event_name": "AfterTool",
    }
    _run_handler("AfterTool", payload, monkeypatch)

    events = session_store.read_events("gsid-4")
    ev = events[-1]
    assert ev["event_type"] == "AfterTool"
    assert ev["tool_name"] == "read_file"
    assert ev["tool_response"] == "display content"


def test_after_tool_normalises_dict_response_fallback_to_llm_content(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "session_id": "gsid-5",
        "tool_name": "google_web_search",
        "tool_input": {"query": "python asyncio"},
        "tool_response": {"llmContent": "search results", "returnDisplay": ""},
        "timestamp": "2024-01-01T00:00:04Z",
        "hook_event_name": "AfterTool",
    }
    _run_handler("AfterTool", payload, monkeypatch)

    events = session_store.read_events("gsid-5")
    assert events[-1]["tool_response"] == "search results"


def test_after_tool_normalises_error_in_response(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "session_id": "gsid-6",
        "tool_name": "write_file",
        "tool_input": {"file_path": "/ro/file.txt", "content": "data"},
        "tool_response": {
            "llmContent": "",
            "returnDisplay": "",
            "error": "Permission denied",
        },
        "timestamp": "2024-01-01T00:00:05Z",
        "hook_event_name": "AfterTool",
    }
    _run_handler("AfterTool", payload, monkeypatch)

    events = session_store.read_events("gsid-6")
    assert events[-1]["tool_response"] == "Error: Permission denied"


def test_after_tool_dlp_output_matches_gemini_schema(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import dlp

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "load", lambda: {"dlp": True})
    monkeypatch.setattr(
        dlp,
        "maybe_redact_output",
        lambda tool_name, tool_response, cwd: "TOKEN=[REDACTED:TOKEN]",
    )

    payload = {
        "session_id": "gsid-dlp",
        "transcript_path": "/tmp/gemini-session.json",
        "cwd": "/repo",
        "hook_event_name": "AfterTool",
        "timestamp": "2026-04-10T00:00:00Z",
        "tool_name": "read_file",
        "tool_input": {"file_path": "/repo/.env"},
        "tool_response": {
            "llmContent": "TOKEN=secret-token",
            "returnDisplay": "TOKEN=secret-token",
            "error": None,
        },
    }
    validate(instance=payload, schema=GEMINI_AFTER_TOOL_INPUT_SCHEMA)

    output = _run_handler("AfterTool", payload, monkeypatch, capsys=capsys)
    parsed = json.loads(output)
    validate(instance=parsed, schema=GEMINI_COMMON_OUTPUT_SCHEMA)
    assert parsed == {"decision": "deny", "reason": "TOKEN=[REDACTED:TOKEN]"}


# ---------------------------------------------------------------------------
# SessionEnd — record assembly
# ---------------------------------------------------------------------------


def test_session_end_builds_and_uploads_record(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    # Pre-populate with events from a complete session.
    session_store.append_event(
        "gsid-7",
        {
            "event_type": "SessionStart",
            "cwd": "/work",
            "timestamp": "2024-01-01T00:00:00Z",
        },
    )
    session_store.append_event(
        "gsid-7",
        {
            "event_type": "BeforeAgent",
            "prompt": "write a test",
            "timestamp": "2024-01-01T00:00:01Z",
        },
    )
    session_store.append_event(
        "gsid-7",
        {
            "event_type": "AfterAgent",
            "prompt_response": "Sure!",
            "timestamp": "2024-01-01T00:00:10Z",
        },
    )

    uploaded: list[HookSessionRecord] = []

    async def fake_upload(record):
        uploaded.append(record)

    monkeypatch.setattr("rclm.hooks.gemini_handler.upload_single", fake_upload)

    payload = {
        "session_id": "gsid-7",
        "cwd": "/work",
        "transcript_path": "/tmp/gemini-transcript.json",
        "timestamp": "2024-01-01T00:01:00Z",
        "reason": "exit",
        "hook_event_name": "SessionEnd",
    }
    _run_handler("SessionEnd", payload, monkeypatch)

    assert len(uploaded) == 1
    rec = uploaded[0]
    assert isinstance(rec, HookSessionRecord)
    assert rec.session_id == "gsid-7"
    assert rec.cwd == "/work"
    assert rec.started_at == "2024-01-01T00:00:00Z"
    assert rec.ended_at == "2024-01-01T00:01:00Z"
    assert rec.duration_s == 60.0
    assert rec.transcript_path == "/tmp/gemini-transcript.json"
    assert rec.model == "gemini-unknown"  # not set without BeforeModel hook
    assert len(rec.messages) == 2
    assert rec.messages[0] == {
        "role": "user",
        "content": "write a test",
        "timestamp": "2024-01-01T00:00:01Z",
    }
    assert rec.messages[1] == {
        "role": "assistant",
        "content": "Sure!",
        "timestamp": "2024-01-01T00:00:10Z",
    }
    assert rec.tool_calls == []
    assert rec.file_diffs == []

    # Session JSONL cleaned up.
    assert session_store.read_events("gsid-7") == []


def test_session_end_without_session_start_uses_fallback_cwd(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    uploaded: list[HookSessionRecord] = []

    async def fake_upload(record):
        uploaded.append(record)

    monkeypatch.setattr("rclm.hooks.gemini_handler.upload_single", fake_upload)

    payload = {
        "session_id": "gsid-8",
        "cwd": "/fallback-cwd",
        "transcript_path": None,
        "timestamp": "2024-01-01T00:01:00Z",
        "reason": "exit",
        "hook_event_name": "SessionEnd",
    }
    _run_handler("SessionEnd", payload, monkeypatch)

    assert len(uploaded) == 1
    assert uploaded[0].cwd == "/fallback-cwd"


# ---------------------------------------------------------------------------
# SessionEnd — file diff extraction
# ---------------------------------------------------------------------------


def test_session_end_extracts_write_file_diff(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    session_store.append_event(
        "gsid-9",
        {
            "event_type": "SessionStart",
            "cwd": "/w",
            "timestamp": "2024-01-01T00:00:00Z",
        },
    )
    session_store.append_event(
        "gsid-9",
        {
            "event_type": "AfterTool",
            "tool_name": "write_file",
            "tool_input": {
                "file_path": "src/hello.py",
                "content": "print('hello')\n",
            },
            "tool_response": "File written.",
            "timestamp": "2024-01-01T00:00:05Z",
        },
    )

    uploaded: list[HookSessionRecord] = []

    async def fake_upload(record):
        uploaded.append(record)

    monkeypatch.setattr("rclm.hooks.gemini_handler.upload_single", fake_upload)

    _run_handler(
        "SessionEnd",
        {
            "session_id": "gsid-9",
            "cwd": "/w",
            "transcript_path": None,
            "timestamp": "2024-01-01T00:01:00Z",
        },
        monkeypatch,
    )

    rec = uploaded[0]
    assert len(rec.file_diffs) == 1
    diff = rec.file_diffs[0]
    assert diff.path == "src/hello.py"
    assert diff.before is None
    assert diff.after == "print('hello')\n"
    assert "b/src/hello.py" in diff.unified_diff


def test_session_end_extracts_replace_diff(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    session_store.append_event(
        "gsid-10",
        {
            "event_type": "SessionStart",
            "cwd": "/w",
            "timestamp": "2024-01-01T00:00:00Z",
        },
    )
    session_store.append_event(
        "gsid-10",
        {
            "event_type": "AfterTool",
            "tool_name": "replace",
            "tool_input": {
                "file_path": "src/util.py",
                "old_string": "def foo():\n    pass\n",
                "new_string": "def foo():\n    return 42\n",
            },
            "tool_response": "File updated.",
            "timestamp": "2024-01-01T00:00:06Z",
        },
    )

    uploaded: list[HookSessionRecord] = []

    async def fake_upload(record):
        uploaded.append(record)

    monkeypatch.setattr("rclm.hooks.gemini_handler.upload_single", fake_upload)

    _run_handler(
        "SessionEnd",
        {
            "session_id": "gsid-10",
            "cwd": "/w",
            "transcript_path": None,
            "timestamp": "2024-01-01T00:01:00Z",
        },
        monkeypatch,
    )

    rec = uploaded[0]
    assert len(rec.file_diffs) == 1
    diff = rec.file_diffs[0]
    assert diff.path == "src/util.py"
    assert diff.before == "def foo():\n    pass\n"
    assert diff.after == "def foo():\n    return 42\n"
    assert "-    pass" in diff.unified_diff
    assert "+    return 42" in diff.unified_diff


# ---------------------------------------------------------------------------
# Tool calls captured via AfterTool
# ---------------------------------------------------------------------------


def test_session_end_includes_tool_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    session_store.append_event(
        "gsid-11",
        {
            "event_type": "SessionStart",
            "cwd": "/w",
            "timestamp": "2024-01-01T00:00:00Z",
        },
    )
    session_store.append_event(
        "gsid-11",
        {
            "event_type": "AfterTool",
            "tool_name": "run_shell_command",
            "tool_input": {"command": "pytest tests/"},
            "tool_response": "2 passed",
            "timestamp": "2024-01-01T00:00:07Z",
        },
    )

    uploaded: list[HookSessionRecord] = []

    async def fake_upload(record):
        uploaded.append(record)

    monkeypatch.setattr("rclm.hooks.gemini_handler.upload_single", fake_upload)

    _run_handler(
        "SessionEnd",
        {
            "session_id": "gsid-11",
            "cwd": "/w",
            "transcript_path": None,
            "timestamp": "2024-01-01T00:01:00Z",
        },
        monkeypatch,
    )

    rec = uploaded[0]
    assert len(rec.tool_calls) == 1
    tc = rec.tool_calls[0]
    assert tc.tool_name == "run_shell_command"
    assert tc.tool_input == {"command": "pytest tests/"}
    assert tc.tool_result == "2 passed"
    assert tc.tool_use_id.startswith("gemini-tool-")


# ---------------------------------------------------------------------------
# Stdout is always valid JSON (Gemini requirement)
# ---------------------------------------------------------------------------


def test_main_always_outputs_json_on_stdout(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("sys.argv", ["rclm-gemini-hooks", "SessionStart"])
    monkeypatch.setattr(
        "sys.stdin",
        __import__("io").StringIO(
            json.dumps(
                {
                    "session_id": "x",
                    "cwd": "/",
                    "timestamp": "2024-01-01T00:00:00Z",
                }
            )
        ),
    )
    with pytest.raises(SystemExit):
        gemini_handler.main()
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out)  # must not raise
    assert isinstance(parsed, dict)


def test_main_outputs_json_for_unknown_event(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["rclm-gemini-hooks", "BeforeToolSelection"])
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{}"))
    with pytest.raises(SystemExit):
        gemini_handler.main()
    out = capsys.readouterr().out.strip()
    assert json.loads(out) == {}


def test_main_outputs_json_on_missing_argv(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["rclm-gemini-hooks"])
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
    with pytest.raises(SystemExit):
        gemini_handler.main()
    out = capsys.readouterr().out.strip()
    assert json.loads(out) == {}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_handler_exits_0_on_exception(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    def boom(session_id, payload):
        raise RuntimeError("catastrophic failure")

    monkeypatch.setitem(gemini_handler._HANDLERS, "SessionStart", boom)

    monkeypatch.setattr("sys.argv", ["rclm-gemini-hooks", "SessionStart"])
    monkeypatch.setattr(
        "sys.stdin",
        __import__("io").StringIO(json.dumps({"session_id": "x", "cwd": "/"})),
    )
    with pytest.raises(SystemExit) as exc_info:
        gemini_handler.main()
    assert exc_info.value.code == 0


def test_handler_exits_0_on_invalid_json(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rclm-gemini-hooks", "SessionStart"])
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("NOT JSON"))
    with pytest.raises(SystemExit) as exc_info:
        gemini_handler.main()
    assert exc_info.value.code == 0
