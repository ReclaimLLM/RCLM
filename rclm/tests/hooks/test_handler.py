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
                # Confirmed empirically that Claude Code (2.1.205) honors an
                # object here for image results (unlike a plain string, which
                # is ignored for native Read) — see image_lifecycle.py.
                "updatedToolOutput": {"type": ["string", "object"]},
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
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")

    payload = {
        "session_id": "sid-1",
        "cwd": "/projects/foo",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("SessionStart", payload, monkeypatch)

    events = session_store.read_events("sid-1")
    assert len(events) == 2
    assert events[0]["event_type"] == "SessionStart"
    assert events[0]["cwd"] == "/projects/foo"
    assert events[1]["event_type"] == "HookPolicySnapshot"


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
        assert path == "/api/settings/bootstrap"
        assert (params or {})["cwd"] == "/repo"
        return {
            "context_sessions": [
                {"session_id": "s1", "title": "Fixed auth", "session_summary": "Fixed the bug."}
            ]
        }

    monkeypatch.setattr(handler.bootstrap.ReclaimLLMClient, "_request", fake_request)

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
        return {"context_sessions": []}

    monkeypatch.setattr(handler.bootstrap.ReclaimLLMClient, "_request", fake_request)

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
# SessionStart / Stop — brevity
# ---------------------------------------------------------------------------


def test_brevity_off_by_default_no_injection(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")

    payload = {"session_id": "sid-brv1", "cwd": "/repo", "timestamp": "2024-01-01T00:00:00Z"}
    _run_handler("SessionStart", payload, monkeypatch)

    assert capsys.readouterr().out == ""


def test_brevity_enabled_injects_and_merges_with_context_pack(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "server_url": "https://api.test",
                "api_key": "key",
                "context_pack": True,
                "brevity": True,
            }
        )
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    async def fake_request(self, method, path, *, params=None):
        return {
            "context_sessions": [
                {"session_id": "s1", "title": "Fixed auth", "session_summary": "Fixed the bug."}
            ]
        }

    monkeypatch.setattr(handler.bootstrap.ReclaimLLMClient, "_request", fake_request)

    payload = {"session_id": "sid-brv2", "cwd": str(tmp_path), "timestamp": "2024-01-01T00:00:00Z"}
    _run_handler("SessionStart", payload, monkeypatch)

    output = json.loads(capsys.readouterr().out)
    additional_context = output["hookSpecificOutput"]["additionalContext"]
    assert "Fixed auth" in additional_context
    assert "[rclm] Be concise" in additional_context

    events = session_store.read_events("sid-brv2")
    brevity_events = [ev for ev in events if ev["event_type"] == "BrevityInjected"]
    assert len(brevity_events) == 1
    assert brevity_events[0]["instruction_tokens"] > 0


def test_brevity_skipped_when_project_has_existing_guidance(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"brevity": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    (tmp_path / "CLAUDE.md").write_text("Be concise in all responses.", encoding="utf-8")

    payload = {"session_id": "sid-brv3", "cwd": str(tmp_path), "timestamp": "2024-01-01T00:00:00Z"}
    _run_handler("SessionStart", payload, monkeypatch)

    assert capsys.readouterr().out == ""


def test_brevity_exception_does_not_block_session_start(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"brevity": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "rclm.hooks.claude_handler.brevity.build_session_start_context",
        lambda cwd, cfg: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    payload = {"session_id": "sid-brv4", "cwd": str(tmp_path), "timestamp": "2024-01-01T00:00:00Z"}
    _run_handler("SessionStart", payload, monkeypatch)

    events = session_store.read_events("sid-brv4")
    assert events[0]["event_type"] == "SessionStart"
    assert capsys.readouterr().out == ""


def test_brevity_records_mechanism_savings_on_stop(monkeypatch, tmp_path):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"brevity": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    captured_records = []

    async def _capture(record):
        captured_records.append(record)

    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", _capture)
    _stub_transcript(
        monkeypatch, total_input_tokens=100, total_output_tokens=100, tool_call_count=1
    )

    start_payload = {
        "session_id": "sid-brv5",
        "cwd": str(tmp_path),
        "timestamp": "2024-01-01T00:00:00Z",
    }
    _run_handler("SessionStart", start_payload, monkeypatch)

    stop_payload = {
        "session_id": "sid-brv5",
        "cwd": str(tmp_path),
        "timestamp": "2024-01-01T00:01:00Z",
    }
    _run_handler("Stop", stop_payload, monkeypatch)

    assert len(captured_records) == 1
    mechanism_savings = captured_records[0].mechanism_savings
    assert mechanism_savings["brevity"]["enabled"] is True
    assert "tokens_saved_estimate" not in mechanism_savings["brevity"]
    assert mechanism_savings["brevity"]["instruction_tokens"] > 0


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


def _read_payload(session_id, target, content, **tool_input):
    return {
        "session_id": session_id,
        "tool_name": "Read",
        "tool_input": {"file_path": str(target), **tool_input},
        "tool_response": content,
        "tool_use_id": f"tool-{session_id}",
        "cwd": str(target.parent),
        "timestamp": "2024-01-01T00:00:00Z",
    }


def _long_lines(start=1, end=80):
    return "".join(f"line {line}: {'x' * 32}\n" for line in range(start, end + 1))


def test_read_cache_off_by_default(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")

    target = tmp_path / "a.py"
    content = _long_lines()
    target.write_text(content)
    payload = _read_payload("sid-rc1", target, content)
    _run_handler("PostToolUse", payload, monkeypatch)
    _run_handler("PostToolUse", payload, monkeypatch)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert not session_store.read_read_cache_state("sid-rc1").get("files")


def test_read_cache_first_read_no_output(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    target = tmp_path / "a.py"
    content = _long_lines()
    target.write_text(content)
    payload = _read_payload("sid-rc2", target, content)
    _run_handler("PostToolUse", payload, monkeypatch)

    assert capsys.readouterr().out == ""
    state = session_store.read_read_cache_state("sid-rc2")
    entry = state["files"][str(target)]
    assert entry["spans"] == [{"start": 1, "end": 80, "turn": 1}]


def test_read_cache_unchanged_reread_records_unenforced_potential(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    target = tmp_path / "a.py"
    content = _long_lines()
    target.write_text(content)
    payload = _read_payload("sid-rc3", target, content)
    _run_handler("PostToolUse", payload, monkeypatch)
    capsys.readouterr()

    _run_handler("PostToolUse", payload, monkeypatch)
    assert capsys.readouterr().out == ""

    events = session_store.read_events("sid-rc3")
    saving_events = [e for e in events if e.get("event_type") == "MechanismSaving"]
    assert len(saving_events) == 1
    assert saving_events[0]["mechanism"] == "range_cache"
    assert saving_events[0]["applied"] is False
    assert saving_events[0]["measurement_kind"] == "measured"
    assert saving_events[0]["file_path"] == "a.py"
    transformations = [e for e in events if e.get("event_type") == "ToolTransformation"]
    assert transformations[-1]["compression_strategy"] == "range_cache"
    assert transformations[-1]["applied"] is False
    assert (
        transformations[-1]["raw_token_estimate"] > transformations[-1]["compressed_token_estimate"]
    )


def test_read_cache_shadow_mode_records_but_does_not_replace_output(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True, "shadow_mode": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    target = tmp_path / "a.py"
    content = _long_lines()
    target.write_text(content)
    payload = _read_payload("sid-rc-shadow", target, content)
    _run_handler("PostToolUse", payload, monkeypatch)
    capsys.readouterr()

    # Second identical read would normally be replaced — in shadow mode, nothing prints.
    _run_handler("PostToolUse", payload, monkeypatch)
    assert capsys.readouterr().out == ""

    events = session_store.read_events("sid-rc-shadow")
    saving_events = [e for e in events if e.get("event_type") == "MechanismSaving"]
    assert len(saving_events) == 1
    assert saving_events[0]["mechanism"] == "range_cache"
    assert saving_events[0]["applied"] is False
    assert saving_events[0]["tokens_saved_estimate"] > 0


def test_read_cache_hash_change_invalidates_and_serves_fresh(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    target = tmp_path / "a.py"
    first = _long_lines()
    target.write_text(first)
    first_payload = _read_payload("sid-rc4", target, first)
    _run_handler("PostToolUse", first_payload, monkeypatch)
    capsys.readouterr()

    second = first.replace("line 40:", "changed 40:")
    target.write_text(second)
    second_payload = dict(first_payload, tool_response=second)
    _run_handler("PostToolUse", second_payload, monkeypatch)
    assert capsys.readouterr().out == ""
    state = session_store.read_read_cache_state("sid-rc4")
    assert state["files"][str(target)]["spans"] == [{"start": 1, "end": 80, "turn": 2}]


def test_read_cache_scoped_to_offset_limit(monkeypatch, tmp_path, capsys):
    """Reading a different range of the same file is not treated as a repeat."""
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    target = tmp_path / "a.py"
    whole = _long_lines(1, 100)
    target.write_text(whole)
    first_payload = _read_payload("sid-rc5", target, _long_lines(1, 50), offset=0, limit=50)
    _run_handler("PostToolUse", first_payload, monkeypatch)
    capsys.readouterr()

    second_payload = _read_payload("sid-rc5", target, _long_lines(51, 100), offset=50, limit=50)
    _run_handler("PostToolUse", second_payload, monkeypatch)
    assert capsys.readouterr().out == ""


def test_read_cache_and_dlp_compose_without_exposing_secret(monkeypatch, tmp_path, capsys):
    """DLP must scrub before cached output is measured or returned."""
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

    target = tmp_path / "a.py"
    content = _long_lines().replace("line 1:", "token=secret-token line 1:")
    target.write_text(content)
    first_payload = _read_payload("sid-rc6", target, content)
    _run_handler("PostToolUse", first_payload, monkeypatch)
    capsys.readouterr()

    _run_handler("PostToolUse", first_payload, monkeypatch)
    output = json.loads(capsys.readouterr().out)
    replacement = output["hookSpecificOutput"]["updatedToolOutput"]
    assert "secret-token" not in replacement
    assert "[RCLM] Lines 1-80" not in replacement


def test_read_cache_edit_invalidates_before_reread(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    target = tmp_path / "a.py"
    content = _long_lines()
    target.write_text(content)
    payload = _read_payload("sid-rc-edit", target, content)
    _run_handler("PostToolUse", payload, monkeypatch)
    capsys.readouterr()

    _run_handler(
        "PreToolUse",
        {
            "session_id": "sid-rc-edit",
            "tool_name": "Edit",
            "tool_input": {"file_path": str(target), "old_string": "x", "new_string": "y"},
            "cwd": str(tmp_path),
        },
        monkeypatch,
    )
    capsys.readouterr()
    _run_handler("PostToolUse", payload, monkeypatch)
    assert capsys.readouterr().out == ""


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


def test_repeated_read_stop_emits_session_and_per_call_range_savings(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm._models import ToolCall
    from rclm.hooks import session_store
    from rclm.hooks.transcript import TranscriptData

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    target = tmp_path / "source.py"
    content = _long_lines()
    target.write_text(content)
    payload = _read_payload("sid-range-stop", target, content)

    _run_handler("PostToolUse", payload, monkeypatch)
    _run_handler("PostToolUse", payload, monkeypatch)
    capsys.readouterr()

    uploaded = []

    async def fake_upload_single(record):
        uploaded.append(record)

    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", fake_upload_single)
    monkeypatch.setattr(
        "rclm.hooks.claude_handler.transcript.parse_transcript",
        lambda _path: TranscriptData(
            tool_calls=[
                ToolCall(
                    tool_use_id="tool-sid-range-stop",
                    tool_name="Read",
                    tool_input={"file_path": str(target)},
                    tool_result=content,
                    timestamp="2024-01-01T00:00:00Z",
                )
            ]
        ),
    )

    _run_handler(
        "Stop",
        {
            "session_id": "sid-range-stop",
            "cwd": str(tmp_path),
            "transcript_path": "/tmp/fake.jsonl",
            "timestamp": "2024-01-01T00:01:00Z",
        },
        monkeypatch,
    )

    record = uploaded[0]
    assert record.mechanism_savings["range_cache"]["measurement_kind"] == "measured"
    assert (
        record.mechanism_savings["range_cache"]["files"]["source.py"]["tokens_saved_estimate"] > 0
    )
    tool_call = record.tool_calls[0]
    assert tool_call.compression_strategy == "range_cache"
    assert tool_call.raw_token_estimate > tool_call.compressed_token_estimate
    assert tool_call.extra_fields["compression_file_path"] == "source.py"


def test_stop_records_missing_tool_hook_health_without_retaining_content(
    monkeypatch, tmp_path, capsys
):
    from rclm import _config
    from rclm._models import ToolCall
    from rclm.hooks import session_store
    from rclm.hooks.transcript import TranscriptData

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    session_store.append_event("sid-missing-hooks", {"event_type": "SessionStart"})
    session_store.append_event("sid-missing-hooks", {"event_type": "BrevityInjected"})

    async def fake_upload_single(_record):
        return None

    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", fake_upload_single)
    monkeypatch.setattr(
        "rclm.hooks.claude_handler.transcript.parse_transcript",
        lambda _path: TranscriptData(
            tool_calls=[
                ToolCall(
                    tool_use_id="tool-secret",
                    tool_name="Bash",
                    tool_input={"command": "printf super-secret"},
                    tool_result="super-secret",
                    timestamp="2024-01-01T00:00:00Z",
                )
            ]
        ),
    )

    _run_handler(
        "Stop",
        {
            "session_id": "sid-missing-hooks",
            "transcript_path": "/tmp/fake.jsonl",
        },
        monkeypatch,
    )

    health = session_store.read_hook_health("sid-missing-hooks")
    assert health["status"] == "missing_tool_hooks"
    assert health["transcript_tool_call_count"] == 1
    assert health["pre_tool_use_count"] == 0
    assert health["post_tool_use_count"] == 0
    assert health["read_cache_enabled"] is True
    assert "super-secret" not in json.dumps(health)
    assert session_store.read_events("sid-missing-hooks") == []
    output = json.loads(capsys.readouterr().out)
    assert (
        "Read cache and other tool-level mechanisms were inactive"
        in output["hookSpecificOutput"]["additionalContext"]
    )


def test_stop_records_healthy_tool_hook_lifecycle(monkeypatch, tmp_path, capsys):
    from rclm._models import ToolCall
    from rclm.hooks import session_store
    from rclm.hooks.transcript import TranscriptData

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    session_store.append_event("sid-healthy-hooks", {"event_type": "SessionStart"})
    session_store.append_event("sid-healthy-hooks", {"event_type": "PreToolUse"})
    session_store.append_event("sid-healthy-hooks", {"event_type": "PostToolUse"})

    async def fake_upload_single(_record):
        return None

    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", fake_upload_single)
    monkeypatch.setattr(
        "rclm.hooks.claude_handler.transcript.parse_transcript",
        lambda _path: TranscriptData(
            tool_calls=[
                ToolCall(
                    tool_use_id="tool-1",
                    tool_name="Read",
                    tool_input={"file_path": "/tmp/example"},
                    tool_result="example",
                    timestamp="2024-01-01T00:00:00Z",
                )
            ]
        ),
    )

    _run_handler(
        "Stop",
        {
            "session_id": "sid-healthy-hooks",
            "transcript_path": "/tmp/fake.jsonl",
        },
        monkeypatch,
    )

    health = session_store.read_hook_health("sid-healthy-hooks")
    assert health["status"] == "healthy"
    assert health["pre_tool_use_count"] == 1
    assert health["post_tool_use_count"] == 1
    assert capsys.readouterr().out == ""


def test_read_cache_survives_stop_until_session_end(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm._models import ToolCall
    from rclm.hooks import session_store
    from rclm.hooks.transcript import TranscriptData

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    target = tmp_path / "source.py"
    content = _long_lines()
    target.write_text(content)
    payload = _read_payload("sid-cross-turn", target, content)

    async def fake_upload_single(_record):
        return None

    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", fake_upload_single)
    monkeypatch.setattr(
        "rclm.hooks.claude_handler.transcript.parse_transcript",
        lambda _path: TranscriptData(
            tool_calls=[
                ToolCall(
                    tool_use_id="tool-sid-cross-turn",
                    tool_name="Read",
                    tool_input={"file_path": str(target)},
                    tool_result=content,
                    timestamp="2024-01-01T00:00:00Z",
                )
            ]
        ),
    )

    _run_handler("PostToolUse", payload, monkeypatch)
    _run_handler(
        "Stop",
        {"session_id": "sid-cross-turn", "transcript_path": "/tmp/fake.jsonl"},
        monkeypatch,
    )
    capsys.readouterr()
    assert session_store.read_read_cache_state("sid-cross-turn").get("files")

    _run_handler("PostToolUse", payload, monkeypatch)
    assert capsys.readouterr().out == ""

    _run_handler("SessionEnd", {"session_id": "sid-cross-turn"}, monkeypatch)
    assert session_store.read_read_cache_state("sid-cross-turn") == {}


def test_mechanism_savings_survive_later_stop_without_cache_hit(monkeypatch, tmp_path):
    from rclm.hooks import session_store
    from rclm.hooks._analytics import mechanism_saving_event
    from rclm.hooks.transcript import TranscriptData

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    uploaded = []

    async def fake_upload_single(record):
        uploaded.append(record)

    monkeypatch.setattr("rclm.hooks.claude_handler.upload_single", fake_upload_single)
    monkeypatch.setattr(
        "rclm.hooks.claude_handler.transcript.parse_transcript",
        lambda _path: TranscriptData(),
    )
    session_store.append_event(
        "sid-cumulative",
        mechanism_saving_event(
            "range_cache",
            applied=True,
            tokens_saved_estimate=686,
            measurement_kind="measured",
            file_path="README.md",
        ),
    )

    stop_payload = {"session_id": "sid-cumulative", "transcript_path": "/tmp/fake.jsonl"}
    _run_handler("Stop", stop_payload, monkeypatch)
    _run_handler("Stop", stop_payload, monkeypatch)

    assert len(uploaded) == 2
    assert uploaded[1].mechanism_savings == uploaded[0].mechanism_savings
    assert uploaded[1].mechanism_savings["range_cache"]["tokens_saved_estimate"] == 686
    assert (
        session_store.read_mechanism_savings_state("sid-cumulative")
        == uploaded[1].mechanism_savings
    )

    _run_handler("SessionEnd", {"session_id": "sid-cumulative"}, monkeypatch)
    assert session_store.read_mechanism_savings_state("sid-cumulative") == {}


def test_read_cache_extracts_structured_claude_bash_response(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    target = tmp_path / "source.py"
    content = _long_lines()
    target.write_text(content)
    payload = _read_payload("sid-structured", target, content)
    payload["tool_response"] = {
        "stdout": content,
        "stderr": "",
        "interrupted": False,
    }

    _run_handler("PostToolUse", payload, monkeypatch)
    assert session_store.read_read_cache_state("sid-structured")["files"][str(target)]["spans"]
    capsys.readouterr()

    _run_handler("PostToolUse", payload, monkeypatch)
    assert capsys.readouterr().out == ""


def test_read_cache_wraps_exact_bash_reads_during_pre_tool_use(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"read_cache": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    target = tmp_path / "source.py"
    target.write_text(_long_lines())

    _run_handler(
        "PreToolUse",
        {
            "session_id": "sid-wrapper",
            "tool_name": "Bash",
            "tool_use_id": "tool-wrapper",
            "cwd": str(tmp_path),
            "tool_input": {"command": "sed -n '1,5p' source.py"},
        },
        monkeypatch,
    )

    output = json.loads(capsys.readouterr().out)
    assert output["hookSpecificOutput"]["permissionDecision"] == "allow"
    command = output["hookSpecificOutput"]["updatedInput"]["command"]
    assert command == ("CLAUDE_SESSION_ID=sid-wrapper rclm-read-cache sed -n '1,5p' source.py")
    events = session_store.read_events("sid-wrapper")
    assert any(
        event.get("event_type") == "ReadCacheWrapped" and event.get("tool_use_id") == "tool-wrapper"
        for event in events
    )


def test_response_text_extracts_structured_native_file_content():
    response = {
        "type": "text",
        "file": {
            "filePath": "/tmp/example.py",
            "content": "line one\nline two",
            "numLines": 2,
        },
    }

    assert handler._response_text(response) == "line one\nline two"


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


# ---------------------------------------------------------------------------
# PostToolUse — image lifecycle (Task 1 downscale + Task 2 eviction)
# ---------------------------------------------------------------------------


def _b64_test_image(width: int, height: int) -> str:
    import base64
    import io
    import os

    import PIL.Image

    # Random noise, not a solid color: PNG compresses a flat color down to a
    # few hundred bytes regardless of dimensions, which would trip the
    # min_size_bytes gate before ever reaching the resize logic.
    img = PIL.Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _image_payload(session_id, *, width=2000, height=2000, tool_use_id="tool-img"):
    return {
        "session_id": session_id,
        "tool_name": "Read",
        "tool_input": {"file_path": "/tmp/screenshot.png"},
        "tool_response": {
            "type": "image",
            "file": {
                "base64": _b64_test_image(width, height),
                "type": "image/png",
                "originalSize": 1,
                "dimensions": {
                    "originalWidth": width,
                    "originalHeight": height,
                    "displayWidth": width,
                    "displayHeight": height,
                },
            },
        },
        "tool_use_id": tool_use_id,
        "cwd": "/tmp",
        "timestamp": "2024-01-01T00:00:00Z",
    }


def test_image_lifecycle_off_by_default_falls_through_to_normal_pipeline(
    monkeypatch, tmp_path, capsys
):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")

    payload = _image_payload("sid-img-off")
    _run_handler("PostToolUse", payload, monkeypatch)

    events = session_store.read_events("sid-img-off")
    assert not any(e.get("event_type") == "MechanismSaving" for e in events)
    # Falls through to the normal (DLP/read-cache/dedupe) pipeline, which
    # produces no output here since none of those flags are on either.
    assert capsys.readouterr().out == ""


def test_image_downscale_emits_measured_saving_and_rewrites_output(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"image_lifecycle": True, "image_max_dim": 100}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    payload = _image_payload("sid-img-1")
    _run_handler("PostToolUse", payload, monkeypatch)

    output = json.loads(capsys.readouterr().out)
    validate(instance=output, schema=CLAUDE_POST_TOOL_USE_OUTPUT_SCHEMA)
    new_tool_output = output["hookSpecificOutput"]["updatedToolOutput"]
    assert new_tool_output["file"]["dimensions"]["displayWidth"] == 100

    events = session_store.read_events("sid-img-1")
    saving = next(e for e in events if e.get("event_type") == "MechanismSaving")
    assert saving["mechanism"] == "image_downscale"
    assert saving["applied"] is True
    assert saving["measurement_kind"] == "measured"
    transformation = next(e for e in events if e.get("event_type") == "ToolTransformation")
    assert transformation["compression_strategy"] == "image_downscale"
    assert transformation["applied"] is True


def test_image_downscale_shadow_mode_measures_but_does_not_rewrite(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"image_lifecycle": True, "image_max_dim": 100, "shadow_mode": True})
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    payload = _image_payload("sid-img-shadow")
    _run_handler("PostToolUse", payload, monkeypatch)

    assert capsys.readouterr().out == ""
    events = session_store.read_events("sid-img-shadow")
    saving = next(e for e in events if e.get("event_type") == "MechanismSaving")
    assert saving["mechanism"] == "image_downscale"
    assert saving["applied"] is False
    assert saving["tokens_saved_estimate"] > 0


def test_image_eviction_measurement_always_unapplied_even_without_shadow_mode(
    monkeypatch, tmp_path, capsys
):
    """Task 2 must never apply, unconditionally — not gated by shadow_mode.

    This is the constraint most likely to regress silently: a future refactor
    that merges the Task 1/Task 2 branches could accidentally apply `not
    shadow` to both. shadow_mode is deliberately OFF here (image_downscale
    above IS applied under these same settings) to prove image_eviction's
    applied=False is independent of that flag.
    """
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"image_lifecycle": True, "image_max_dim": 100}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    # Same file_path (eviction key) across two calls -> second call supersedes the first.
    _run_handler("PostToolUse", _image_payload("sid-evict", width=800, height=600), monkeypatch)
    capsys.readouterr()
    _run_handler("PostToolUse", _image_payload("sid-evict", width=400, height=300), monkeypatch)
    capsys.readouterr()

    events = session_store.read_events("sid-evict")
    eviction_savings = [
        e
        for e in events
        if e.get("event_type") == "MechanismSaving" and e.get("mechanism") == "image_eviction"
    ]
    assert len(eviction_savings) == 1
    assert eviction_savings[0]["applied"] is False
    assert eviction_savings[0]["measurement_kind"] == "estimated"
    assert "modeled_cost_estimate" in eviction_savings[0]


def test_image_branch_skips_dlp_and_dedupe(monkeypatch, tmp_path, capsys):
    from rclm import _config
    from rclm.hooks import session_store

    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "image_lifecycle": True,
                "image_max_dim": 100,
                "dlp": True,
                "compression": {"enabled": True, "dedupe": True},
            }
        )
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    _run_handler("PostToolUse", _image_payload("sid-img-nodlp"), monkeypatch)
    capsys.readouterr()

    # Dedupe state should never have been touched for an image-shaped result.
    assert session_store.read_dedupe_state("sid-img-nodlp") == {}
