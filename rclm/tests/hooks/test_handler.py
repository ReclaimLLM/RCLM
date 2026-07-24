"""Tests for rclm.hooks.claude_handler."""

import json

import pytest
from jsonschema import validate

from rclm._models import HookSessionRecord
from rclm.hooks import claude_handler as handler

CLAUDE_PRE_TOOL_USE_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "continue": {"type": "boolean"},
        "stopReason": {"type": "string"},
        "suppressOutput": {"type": "boolean"},
        "systemMessage": {"type": "string"},
        "hookSpecificOutput": {
            "type": "object",
            "additionalProperties": True,
            "required": ["hookEventName"],
            "properties": {
                "hookEventName": {"const": "PreToolUse"},
                "permissionDecision": {
                    "enum": ["allow", "deny", "ask", "defer", "approve", "block"]
                },
                "permissionDecisionReason": {"type": "string"},
                "updatedInput": {"type": "object"},
                "additionalContext": {"type": "string"},
            },
        },
    },
}

CLAUDE_POST_TOOL_USE_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "continue": {"type": "boolean"},
        "stopReason": {"type": "string"},
        "suppressOutput": {"type": "boolean"},
        "systemMessage": {"type": "string"},
        "decision": {"const": "block"},
        "reason": {"type": "string"},
        "hookSpecificOutput": {
            "type": "object",
            "additionalProperties": False,
            "required": ["hookEventName"],
            "properties": {
                "hookEventName": {"const": "PostToolUse"},
                "additionalContext": {"type": "string"},
                "updatedToolOutput": {"type": "string"},
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_handler(event_name: str, payload: dict, monkeypatch) -> None:
    """Call handler.main() with event_name as argv[1] and payload as stdin."""
    monkeypatch.setattr("sys.argv", ["rclm-claude-hooks", event_name])
    monkeypatch.setattr("sys.stdin", _make_stdin(json.dumps(payload)))
    with pytest.raises(SystemExit) as exc_info:
        handler.main()
    assert exc_info.value.code == 0


def _make_stdin(text: str):
    from io import StringIO

    return StringIO(text)


async def _noop_async(*args, **kwargs) -> None:
    pass


# ---------------------------------------------------------------------------
# SessionStart
# ---------------------------------------------------------------------------


def test_session_start_appends_event(monkeypatch, tmp_path):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "session_id": "sid-1",
        "cwd": "/projects/foo",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("SessionStart", payload, monkeypatch)

    events = session_store.read_events("sid-1")
    assert len(events) == 1
    assert events[0]["event_type"] == "SessionStart"
    assert events[0]["cwd"] == "/projects/foo"


# ---------------------------------------------------------------------------
# SessionStart — context pack
# ---------------------------------------------------------------------------


def test_context_pack_off_by_default(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    payload = {"session_id": "sid-cp1", "cwd": "/repo", "timestamp": "2024-01-01T00:00:00Z"}
    _run_handler("SessionStart", payload, monkeypatch)

    assert capsys.readouterr().out == ""


def test_context_pack_injects_highlights_when_enabled(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"server_url": "https://api.test", "api_key": "key", "context_pack": True})
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    async def fake_request(self, method, path, *, params=None):
        assert path == "/api/sessions/search"
        assert (params or {})["file_path"] == "/repo"
        return {
            "sessions": [
                {"session_id": "s1", "title": "Fixed auth", "session_summary": "Fixed the bug."}
            ]
        }

    monkeypatch.setattr(handler.ReclaimLLMClient, "_request", fake_request)

    payload = {"session_id": "sid-cp2", "cwd": "/repo", "timestamp": "2024-01-01T00:00:00Z"}
    _run_handler("SessionStart", payload, monkeypatch)

    output = json.loads(capsys.readouterr().out)
    assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "Fixed auth" in output["hookSpecificOutput"]["additionalContext"]


def test_context_pack_no_sessions_found_prints_nothing(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"server_url": "https://api.test", "api_key": "key", "context_pack": True})
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    async def fake_request(self, method, path, *, params=None):
        return {"sessions": []}

    monkeypatch.setattr(handler.ReclaimLLMClient, "_request", fake_request)

    payload = {"session_id": "sid-cp3", "cwd": "/repo", "timestamp": "2024-01-01T00:00:00Z"}
    _run_handler("SessionStart", payload, monkeypatch)

    assert capsys.readouterr().out == ""


def test_context_pack_missing_credentials_skips_silently(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"context_pack": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.delenv("RECLAIMLLM_SERVER_URL", raising=False)
    monkeypatch.delenv("RECLAIMLLM_API_KEY", raising=False)

    payload = {"session_id": "sid-cp4", "cwd": "/repo", "timestamp": "2024-01-01T00:00:00Z"}
    _run_handler("SessionStart", payload, monkeypatch)

    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Stop — handoff advisor
# ---------------------------------------------------------------------------


def _stub_transcript(monkeypatch, *, total_input_tokens, total_output_tokens, tool_call_count):
    from rclm._models import ToolCall
    from rclm.hooks.transcript import TranscriptData

    tool_calls = [
        ToolCall(
            tool_use_id=f"tc-{i}",
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_result="",
            timestamp="2024-01-01T00:00:00Z",
        )
        for i in range(tool_call_count)
    ]
    monkeypatch.setattr(
        "rclm.hooks.claude_handler.transcript.parse_transcript",
        lambda path: TranscriptData(
            messages=[],
            tool_calls=tool_calls,
            model="claude-sonnet-4-6",
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        ),
    )


def test_handoff_advisor_off_by_default(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", _noop_async)
    _stub_transcript(
        monkeypatch, total_input_tokens=200_000, total_output_tokens=50_000, tool_call_count=100
    )

    payload = {"session_id": "sid-ha1", "cwd": "/repo", "timestamp": "2024-01-01T00:01:00Z"}
    _run_handler("Stop", payload, monkeypatch)

    assert "handoff" not in capsys.readouterr().out


def test_handoff_advisor_small_session_no_advisory(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"handoff_advisor": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", _noop_async)
    _stub_transcript(
        monkeypatch, total_input_tokens=1000, total_output_tokens=500, tool_call_count=5
    )

    payload = {"session_id": "sid-ha2", "cwd": "/repo", "timestamp": "2024-01-01T00:01:00Z"}
    _run_handler("Stop", payload, monkeypatch)

    assert "handoff" not in capsys.readouterr().out


def test_handoff_advisor_below_threshold_no_advisory(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"handoff_advisor": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", _noop_async)
    _stub_transcript(
        monkeypatch, total_input_tokens=100_000, total_output_tokens=50_000, tool_call_count=119
    )

    payload = {"session_id": "sid-ha-below", "cwd": "/repo", "timestamp": "2024-01-01T00:01:00Z"}
    _run_handler("Stop", payload, monkeypatch)

    assert "handoff" not in capsys.readouterr().out


def test_handoff_advisor_uses_configured_thresholds(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "handoff_advisor": True,
                "handoff_advisor_token_threshold": 300_000,
                "handoff_advisor_tool_call_threshold": 200,
            }
        )
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", _noop_async)
    _stub_transcript(
        monkeypatch, total_input_tokens=200_000, total_output_tokens=50_000, tool_call_count=125
    )

    payload = {
        "session_id": "sid-ha-config-threshold",
        "cwd": "/repo",
        "timestamp": "2024-01-01T00:01:00Z",
    }
    _run_handler("Stop", payload, monkeypatch)

    assert "handoff" not in capsys.readouterr().out


def test_handoff_advisor_fires_on_large_session(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"handoff_advisor": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", _noop_async)
    _stub_transcript(
        monkeypatch, total_input_tokens=200_000, total_output_tokens=50_000, tool_call_count=100
    )

    payload = {"session_id": "sid-ha3", "cwd": "/repo", "timestamp": "2024-01-01T00:01:00Z"}
    _run_handler("Stop", payload, monkeypatch)

    output = json.loads(capsys.readouterr().out)
    assert output["hookSpecificOutput"]["hookEventName"] == "Stop"
    additional_context = output["hookSpecificOutput"]["additionalContext"]
    assert "handoff" in additional_context
    assert "/compact focus on the login bug" in additional_context


def test_handoff_advisor_fires_on_high_tool_call_count(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"handoff_advisor": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", _noop_async)
    _stub_transcript(
        monkeypatch, total_input_tokens=100, total_output_tokens=100, tool_call_count=125
    )

    payload = {"session_id": "sid-ha4", "cwd": "/repo", "timestamp": "2024-01-01T00:01:00Z"}
    _run_handler("Stop", payload, monkeypatch)

    output = json.loads(capsys.readouterr().out)
    assert "handoff" in output["hookSpecificOutput"]["additionalContext"]


def test_handoff_advisor_prints_once_per_session(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"handoff_advisor": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", _noop_async)
    _stub_transcript(
        monkeypatch, total_input_tokens=200_000, total_output_tokens=50_000, tool_call_count=100
    )

    payload = {"session_id": "sid-ha5", "cwd": "/repo", "timestamp": "2024-01-01T00:01:00Z"}
    _run_handler("Stop", payload, monkeypatch)
    first = capsys.readouterr().out

    _run_handler("Stop", payload, monkeypatch)
    second = capsys.readouterr().out

    assert "handoff" in first
    assert second == ""


# ---------------------------------------------------------------------------
# PreToolUse — compression gating
# ---------------------------------------------------------------------------


def test_pre_tool_use_compression_enabled_by_default(monkeypatch, tmp_path, capsys):
    """With no config file, compression defaults to enabled, so PreToolUse prints updatedInput."""
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")

    session_store.append_event(
        "sid-c1", {"event_type": "SessionStart", "cwd": "/x", "timestamp": "2024-01-01T00:00:00Z"}
    )

    payload = {
        "session_id": "sid-c1",
        "tool_name": "Grep",
        "tool_input": {"pattern": "foo"},
        "timestamp": "2024-01-01T00:00:01Z",
    }
    _run_handler("PreToolUse", payload, monkeypatch)

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    validate(instance=output, schema=CLAUDE_PRE_TOOL_USE_OUTPUT_SCHEMA)
    assert output["hookSpecificOutput"]["updatedInput"]["head_limit"] == 50


def test_pre_tool_use_no_compression_when_disabled(monkeypatch, tmp_path, capsys):
    """With compress=False explicitly set, PreToolUse should NOT print updatedInput."""
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"compress": False}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    session_store.append_event(
        "sid-c1", {"event_type": "SessionStart", "cwd": "/x", "timestamp": "2024-01-01T00:00:00Z"}
    )

    payload = {
        "session_id": "sid-c1",
        "tool_name": "Grep",
        "tool_input": {"pattern": "foo"},
        "timestamp": "2024-01-01T00:00:01Z",
    }
    _run_handler("PreToolUse", payload, monkeypatch)

    captured = capsys.readouterr()
    assert "updatedInput" not in captured.out


def test_pre_tool_use_compression_when_enabled(monkeypatch, tmp_path, capsys):
    """With compress=True in config, PreToolUse should print updatedInput for Grep."""
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"compress": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    session_store.append_event(
        "sid-c2", {"event_type": "SessionStart", "cwd": "/x", "timestamp": "2024-01-01T00:00:00Z"}
    )

    payload = {
        "session_id": "sid-c2",
        "tool_name": "Grep",
        "tool_input": {"pattern": "foo"},
        "timestamp": "2024-01-01T00:00:01Z",
    }
    _run_handler("PreToolUse", payload, monkeypatch)

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    validate(instance=output, schema=CLAUDE_PRE_TOOL_USE_OUTPUT_SCHEMA)
    assert output["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert output["hookSpecificOutput"]["updatedInput"]["head_limit"] == 50


# ---------------------------------------------------------------------------
# PreToolUse — loop-breaker gating
# ---------------------------------------------------------------------------


def test_loop_breaker_off_by_default(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")

    for _ in range(4):
        payload = {
            "session_id": "sid-lb1",
            "tool_name": "Bash",
            "tool_input": {"command": "npm test"},
            "timestamp": "2024-01-01T00:00:00Z",
        }
        _run_handler("PreToolUse", payload, monkeypatch)

    captured = capsys.readouterr()
    assert "loop-breaker" not in captured.out


def test_loop_breaker_warns_on_repeated_calls_when_enabled(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"loop_breaker": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    payload = {
        "session_id": "sid-lb2",
        "tool_name": "Bash",
        "tool_input": {"command": "npm test"},
        "timestamp": "2024-01-01T00:00:00Z",
    }
    # First two calls: no signal yet (need 2 prior identical calls to warn).
    _run_handler("PreToolUse", payload, monkeypatch)
    capsys.readouterr()
    _run_handler("PreToolUse", payload, monkeypatch)
    capsys.readouterr()

    # Third call: two prior identical calls now on record -> warn.
    _run_handler("PreToolUse", payload, monkeypatch)
    output = json.loads(capsys.readouterr().out)
    validate(instance=output, schema=CLAUDE_PRE_TOOL_USE_OUTPUT_SCHEMA)
    assert "loop-breaker" in output["hookSpecificOutput"]["additionalContext"]


def test_loop_breaker_shadow_mode_records_but_does_not_warn(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"loop_breaker": True, "shadow_mode": True, "compress": False})
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    payload = {
        "session_id": "sid-lb-shadow",
        "tool_name": "Bash",
        "tool_input": {"command": "npm test"},
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("PreToolUse", payload, monkeypatch)
    capsys.readouterr()
    _run_handler("PreToolUse", payload, monkeypatch)
    capsys.readouterr()

    # Third call would normally warn — in shadow mode, nothing is printed.
    _run_handler("PreToolUse", payload, monkeypatch)
    assert capsys.readouterr().out == ""

    events = session_store.read_events("sid-lb-shadow")
    saving_events = [e for e in events if e.get("event_type") == "MechanismSaving"]
    assert len(saving_events) == 1
    assert saving_events[0]["mechanism"] == "H4_loop_breaker"
    assert saving_events[0]["applied"] is False


def test_loop_breaker_asks_after_repeated_failures(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"loop_breaker": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    edit_input = {"file_path": "src/app.ts", "old_string": "x", "new_string": "y"}
    for _ in range(4):
        failure_payload = {
            "session_id": "sid-lb3",
            "tool_name": "Edit",
            "tool_input": edit_input,
            "tool_output": "Error: old_string not found",
            "timestamp": "2024-01-01T00:00:00Z",
        }
        _run_handler("PostToolUseFailure", failure_payload, monkeypatch)

    payload = {
        "session_id": "sid-lb3",
        "tool_name": "Edit",
        "tool_input": edit_input,
        "timestamp": "2024-01-01T00:00:01Z",
    }
    _run_handler("PreToolUse", payload, monkeypatch)

    output = json.loads(capsys.readouterr().out)
    validate(instance=output, schema=CLAUDE_PRE_TOOL_USE_OUTPUT_SCHEMA)
    assert output["hookSpecificOutput"]["permissionDecision"] == "ask"
    assert "src/app.ts" in output["hookSpecificOutput"]["permissionDecisionReason"]


# ---------------------------------------------------------------------------
# PostToolUseFailure
# ---------------------------------------------------------------------------


def test_post_tool_use_failure_appends_event(monkeypatch, tmp_path):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    payload = {
        "session_id": "sid-fail1",
        "tool_name": "Bash",
        "tool_input": {"command": "pytest"},
        "tool_output": "exit code 1",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("PostToolUseFailure", payload, monkeypatch)

    events = session_store.read_events("sid-fail1")
    assert events[-1]["event_type"] == "ToolFailure"
    assert events[-1]["tool_name"] == "Bash"
    assert events[-1]["tool_output"] == "exit code 1"


# ---------------------------------------------------------------------------
# PostToolUse — read-cache gating
# ---------------------------------------------------------------------------


def test_read_cache_off_by_default(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")

    payload = {
        "session_id": "sid-rc1",
        "tool_name": "Read",
        "tool_input": {"file_path": "/repo/a.py"},
        "tool_response": "def foo(): pass\n",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("PostToolUse", payload, monkeypatch)
    _run_handler("PostToolUse", payload, monkeypatch)

    captured = capsys.readouterr()
    assert "read-cache" not in captured.out


def test_read_cache_first_read_no_output(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    payload = {
        "session_id": "sid-rc2",
        "tool_name": "Read",
        "tool_input": {"file_path": "/repo/a.py"},
        "tool_response": "def foo(): pass\n",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("PostToolUse", payload, monkeypatch)

    assert capsys.readouterr().out == ""
    events = session_store.read_events("sid-rc2")
    assert events[-1]["event_type"] == "ReadSnapshot"
    assert events[-1]["file_path"] == "/repo/a.py"


def test_read_cache_unchanged_reread_replaces_output(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    payload = {
        "session_id": "sid-rc3",
        "tool_name": "Read",
        "tool_input": {"file_path": "/repo/a.py"},
        "tool_response": "def foo(): pass\n",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("PostToolUse", payload, monkeypatch)
    capsys.readouterr()

    _run_handler("PostToolUse", payload, monkeypatch)
    output = json.loads(capsys.readouterr().out)
    validate(instance=output, schema=CLAUDE_POST_TOOL_USE_OUTPUT_SCHEMA)
    assert (
        "Unchanged since the last read of /repo/a.py"
        in output["hookSpecificOutput"]["updatedToolOutput"]
    )

    events = session_store.read_events("sid-rc3")
    saving_events = [e for e in events if e.get("event_type") == "MechanismSaving"]
    assert len(saving_events) == 1
    assert saving_events[0]["mechanism"] == "H1_read_cache"
    assert saving_events[0]["applied"] is True


def test_read_cache_shadow_mode_records_but_does_not_replace_output(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True, "shadow_mode": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    payload = {
        "session_id": "sid-rc-shadow",
        "tool_name": "Read",
        "tool_input": {"file_path": "/repo/a.py"},
        "tool_response": "def foo(): pass\n",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("PostToolUse", payload, monkeypatch)
    capsys.readouterr()

    # Second identical read would normally be replaced — in shadow mode, nothing prints.
    _run_handler("PostToolUse", payload, monkeypatch)
    assert capsys.readouterr().out == ""

    events = session_store.read_events("sid-rc-shadow")
    saving_events = [e for e in events if e.get("event_type") == "MechanismSaving"]
    assert len(saving_events) == 1
    assert saving_events[0]["mechanism"] == "H1_read_cache"
    assert saving_events[0]["applied"] is False


def test_read_cache_changed_reread_returns_diff(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    first_payload = {
        "session_id": "sid-rc4",
        "tool_name": "Read",
        "tool_input": {"file_path": "/repo/a.py"},
        "tool_response": "line1\nline2\n",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("PostToolUse", first_payload, monkeypatch)
    capsys.readouterr()

    second_payload = dict(first_payload, tool_response="line1\nCHANGED\n")
    _run_handler("PostToolUse", second_payload, monkeypatch)
    output = json.loads(capsys.readouterr().out)
    validate(instance=output, schema=CLAUDE_POST_TOOL_USE_OUTPUT_SCHEMA)
    assert "changed since the last read" in output["hookSpecificOutput"]["updatedToolOutput"]


def test_read_cache_scoped_to_offset_limit(monkeypatch, tmp_path, capsys):
    """Reading a different range of the same file is not treated as a repeat."""
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    first_payload = {
        "session_id": "sid-rc5",
        "tool_name": "Read",
        "tool_input": {"file_path": "/repo/a.py", "offset": 0, "limit": 50},
        "tool_response": "chunk one\n",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("PostToolUse", first_payload, monkeypatch)
    capsys.readouterr()

    second_payload = {
        "session_id": "sid-rc5",
        "tool_name": "Read",
        "tool_input": {"file_path": "/repo/a.py", "offset": 50, "limit": 50},
        "tool_response": "chunk two\n",
        "timestamp": "2024-01-01T00:00:01Z",
    }
    _run_handler("PostToolUse", second_payload, monkeypatch)
    assert capsys.readouterr().out == ""


def test_read_cache_and_dlp_compose_scrubbed_content_is_diffed(monkeypatch, tmp_path, capsys):
    """DLP must scrub before read-cache diffs, so a diff can never embed a raw secret."""
    from rclm import _config
    from rclm.hooks import dlp, session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True, "dlp": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        dlp,
        "maybe_redact_output",
        lambda tool_name, tool_response, cwd: tool_response.replace(
            "secret-token", "[REDACTED:TOKEN]"
        ),
    )

    first_payload = {
        "session_id": "sid-rc6",
        "tool_name": "Read",
        "tool_input": {"file_path": "/repo/a.py"},
        "tool_response": "token=secret-token\nline2\n",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("PostToolUse", first_payload, monkeypatch)
    capsys.readouterr()

    second_payload = dict(first_payload, tool_response="token=secret-token\nline2 changed\n")
    _run_handler("PostToolUse", second_payload, monkeypatch)
    output = json.loads(capsys.readouterr().out)
    diff = output["hookSpecificOutput"]["updatedToolOutput"]
    assert "secret-token" not in diff
    assert "[REDACTED:TOKEN]" in diff


# ---------------------------------------------------------------------------
# PostToolUse
# ---------------------------------------------------------------------------


def test_post_tool_use_appends_tool_event(monkeypatch, tmp_path):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    # First set up a SessionStart so the session file exists.
    session_store.append_event(
        "sid-2", {"event_type": "SessionStart", "cwd": "/x", "timestamp": "2024-01-01T00:00:00Z"}
    )

    payload = {
        "session_id": "sid-2",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_response": "file.py\n",
        "timestamp": "2024-01-01T00:00:01Z",
    }
    _run_handler("PostToolUse", payload, monkeypatch)

    events = session_store.read_events("sid-2")
    assert events[-1]["event_type"] == "PostToolUse"
    assert events[-1]["tool_name"] == "Bash"
    assert events[-1]["tool_response"] == "file.py\n"


def test_post_tool_use_dlp_output_matches_claude_schema(monkeypatch, tmp_path, capsys):
    from rclm.hooks import dlp, session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("rclm._config.load", lambda: {"dlp": True})
    monkeypatch.setattr(
        dlp,
        "maybe_redact_output",
        lambda tool_name, tool_response, cwd: "token=[REDACTED:TOKEN]",
    )

    payload = {
        "session_id": "sid-dlp",
        "transcript_path": "/tmp/claude.jsonl",
        "cwd": "/repo",
        "permission_mode": "default",
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "cat .env"},
        "tool_response": "token=secret-token",
        "tool_use_id": "toolu_01ABC123",
        "duration_ms": 12,
    }
    _run_handler("PostToolUse", payload, monkeypatch)

    output = json.loads(capsys.readouterr().out)
    validate(instance=output, schema=CLAUDE_POST_TOOL_USE_OUTPUT_SCHEMA)
    assert output["hookSpecificOutput"] == {
        "hookEventName": "PostToolUse",
        "updatedToolOutput": "token=[REDACTED:TOKEN]",
        "additionalContext": "[rclm DLP] Secrets were redacted from the tool response.",
    }


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


def test_stop_builds_hook_session_record_and_uploads(monkeypatch, tmp_path):
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    # Pre-populate session events.
    session_store.append_event(
        "sid-3",
        {"event_type": "SessionStart", "cwd": "/projects/bar", "timestamp": "2024-01-01T00:00:00Z"},
    )

    uploaded_records = []

    async def fake_upload_single(record):
        uploaded_records.append(record)

    from rclm.hooks.transcript import TranscriptData

    monkeypatch.setattr(
        "rclm.hooks.claude_handler.upload_single",
        fake_upload_single,
    )
    monkeypatch.setattr(
        "rclm.hooks.claude_handler.transcript.parse_transcript",
        lambda path: TranscriptData(
            messages=[{"role": "user", "content": "hi", "timestamp": ""}],
            tool_calls=[],
            model="claude-sonnet-4-6",
            total_input_tokens=10,
            total_output_tokens=5,
            cache_read_tokens=4000,
            cache_creation_tokens=200,
            usage_source="provider",
        ),
    )

    payload = {
        "session_id": "sid-3",
        "cwd": "/projects/bar",
        "transcript_path": "/tmp/fake.jsonl",
        "timestamp": "2024-01-01T00:01:00Z",
    }
    _run_handler("Stop", payload, monkeypatch)

    assert len(uploaded_records) == 1
    record = uploaded_records[0]
    assert isinstance(record, HookSessionRecord)
    assert record.session_id == "sid-3"
    assert record.cwd == "/projects/bar"
    assert record.model == "claude-sonnet-4-6"
    assert record.total_input_tokens == 10
    assert record.total_output_tokens == 5
    assert record.cache_read_tokens == 4000
    assert record.cache_creation_tokens == 200
    assert record.usage_source == "provider"
    assert len(record.messages) == 1

    # Cleanup should have removed the session file.
    assert session_store.read_events("sid-3") == []


def test_primary_stop_schedules_update_after_upload(monkeypatch, tmp_path):
    from rclm.hooks import session_store
    from rclm.hooks.transcript import TranscriptData

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    calls: list[str] = []

    async def fake_upload_single(_record):
        calls.append("upload")

    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", fake_upload_single)
    monkeypatch.setattr(
        "rclm.hooks.claude_handler.transcript.parse_transcript",
        lambda _path: TranscriptData(),
    )
    monkeypatch.setattr(
        "rclm.hooks.claude_handler.schedule_session_end_update",
        lambda: calls.append("schedule"),
    )

    _run_handler(
        "Stop", {"session_id": "sid-update", "timestamp": "2024-01-01T00:01:00Z"}, monkeypatch
    )

    assert calls == ["upload", "schedule"]


def test_stop_without_prior_session_start_uses_fallback(monkeypatch, tmp_path):
    """Stop event with no SessionStart in store must not crash."""
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    uploaded_records = []

    async def fake_upload_single(record):
        uploaded_records.append(record)

    from rclm.hooks.transcript import TranscriptData

    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", fake_upload_single)
    monkeypatch.setattr(
        "rclm.hooks.claude_handler.transcript.parse_transcript",
        lambda path: TranscriptData(),
    )

    payload = {
        "session_id": "sid-4",
        "cwd": "/fallback",
        "transcript_path": None,
        "timestamp": "2024-01-01T00:01:00Z",
    }
    _run_handler("Stop", payload, monkeypatch)

    assert len(uploaded_records) == 1
    record = uploaded_records[0]
    assert record.cwd == "/fallback"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_handler_exits_0_on_exception(monkeypatch, tmp_path):
    """Any exception in a handler must be swallowed; process exits 0."""
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

    def boom(session_id, payload):
        raise RuntimeError("catastrophic failure")

    monkeypatch.setitem(handler._HANDLERS, "SessionStart", boom)

    monkeypatch.setattr("sys.argv", ["rclm-claude-hooks", "SessionStart"])
    monkeypatch.setattr("sys.stdin", _make_stdin('{"session_id": "x", "cwd": "/"}'))
    with pytest.raises(SystemExit) as exc_info:
        handler.main()
    assert exc_info.value.code == 0


def test_unknown_event_exits_0(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rclm-claude-hooks", "UnknownEvent"])
    monkeypatch.setattr("sys.stdin", _make_stdin("{}"))
    with pytest.raises(SystemExit) as exc_info:
        handler.main()
    assert exc_info.value.code == 0
