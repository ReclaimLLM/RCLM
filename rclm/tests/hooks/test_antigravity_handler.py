"""Tests for rclm.hooks.antigravity_handler."""

from __future__ import annotations

import asyncio
import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rclm import _config
from rclm._models import HookSessionRecord
from rclm.hooks import antigravity_handler, bootstrap, brevity, session_store


@pytest.fixture(autouse=True)
def _isolate_sessions_dir(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", sessions_dir)


def _run_handler(event_name: str, payload: dict, monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["rclm-antigravity-hooks", event_name])
    monkeypatch.setattr("sys.stdin", StringIO(json.dumps(payload)))
    with pytest.raises(SystemExit) as exc_info:
        antigravity_handler.main()
    assert exc_info.value.code == 0


def _write_transcript(tmp_path, lines: list[dict]):
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return str(path)


def test_stop_uploads_record_built_from_transcript(tmp_path, monkeypatch):
    uploaded: list[HookSessionRecord] = []

    async def fake_upload(record, *, max_retries=3):
        uploaded.append(record)

    monkeypatch.setattr(antigravity_handler, "upload_single", fake_upload)

    transcript_path = _write_transcript(
        tmp_path,
        [
            {
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "created_at": "2026-08-05T09:31:42Z",
                "content": "give me a plan",
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "2026-08-05T09:31:43Z",
                "tool_calls": [{"name": "list_dir", "args": {"DirectoryPath": '"/repo"'}}],
            },
            {
                "step_index": 2,
                "source": "MODEL",
                "type": "LIST_DIRECTORY",
                "created_at": "2026-08-05T09:31:44Z",
                "content": "1 file.",
            },
            {
                "step_index": 3,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "created_at": "2026-08-05T09:32:00Z",
                "content": "Here is the plan.",
            },
        ],
    )

    _run_handler(
        "Stop",
        {
            "conversationId": "ag-sid-1",
            "transcriptPath": transcript_path,
            "workspacePaths": ["/repo"],
            "modelName": "Gemini 3.6 Flash (High)",
            "terminationReason": "model_stop",
            "fullyIdle": True,
            "artifactDirectoryPath": "/repo/.gemini/antigravity/artifacts",
        },
        monkeypatch,
    )

    assert len(uploaded) == 1
    record = uploaded[0]
    assert record.capture_source == "native_agent"
    assert record.agent_client == "antigravity"
    assert record.adapter_name == "antigravity_hooks"
    assert record.model_provider == "google"
    assert record.session_id == "ag-sid-1"
    assert record.cwd == "/repo"
    assert record.model == "Gemini 3.6 Flash (High)"
    assert record.transcript_path == transcript_path
    assert record.started_at == "2026-08-05T09:31:42Z"
    assert record.ended_at == "2026-08-05T09:32:00Z"
    assert record.duration_s == pytest.approx(18.0)
    assert len(record.messages) == 3
    assert len(record.tool_calls) == 1
    assert record.tool_calls[0].tool_result == "1 file."
    assert record.tool_call_count == 1
    assert record.dominant_tool == "list_dir"
    assert record.file_diffs == []
    assert record.extra_fields["termination_reason"] == "model_stop"
    assert record.extra_fields["fully_idle"] is True
    assert record.extra_fields["artifact_directory_path"] == "/repo/.gemini/antigravity/artifacts"


def test_stop_closes_uploader_session(tmp_path, monkeypatch):
    closed = []

    async def fake_upload(record, *, max_retries=3):
        pass

    async def fake_close_session():
        closed.append(True)

    monkeypatch.setattr(antigravity_handler, "upload_single", fake_upload)
    monkeypatch.setattr(antigravity_handler, "close_session", fake_close_session)

    _run_handler("Stop", {"conversationId": "ag-sid-close"}, monkeypatch)

    assert closed == [True]


def test_stop_without_conversation_id_does_not_upload(tmp_path, monkeypatch):
    uploaded: list[HookSessionRecord] = []

    async def fake_upload(record, *, max_retries=3):
        uploaded.append(record)

    monkeypatch.setattr(antigravity_handler, "upload_single", fake_upload)

    _run_handler("Stop", {"transcriptPath": None}, monkeypatch)

    assert uploaded == []


def test_stop_falls_back_to_unknown_model_when_missing(tmp_path, monkeypatch):
    uploaded: list[HookSessionRecord] = []

    async def fake_upload(record, *, max_retries=3):
        uploaded.append(record)

    monkeypatch.setattr(antigravity_handler, "upload_single", fake_upload)

    _run_handler(
        "Stop",
        {"conversationId": "ag-sid-2", "transcriptPath": None, "workspacePaths": []},
        monkeypatch,
    )

    assert len(uploaded) == 1
    assert uploaded[0].model == "antigravity-unknown"
    assert uploaded[0].model_provider == "google"
    assert uploaded[0].cwd == ""


def test_pre_invocation_injects_context_and_brevity(monkeypatch, capsys):
    monkeypatch.setattr(
        _config,
        "load",
        lambda: {
            "context_pack": True,
            "brevity": True,
            "handoff_advisor": True,
        },
    )
    monkeypatch.setattr(
        bootstrap,
        "fetch",
        AsyncMock(
            return_value={
                "context_sessions": [
                    {"title": "Prior work", "session_summary": "Fixed connection leak"}
                ]
            }
        ),
    )
    monkeypatch.setattr(
        brevity,
        "build_session_start_context",
        lambda cwd, cfg: {
            "text": "Be concise.",
            "instruction_hash": "hash123",
            "instruction_tokens": 10,
        },
    )

    _run_handler(
        "PreInvocation",
        {
            "conversationId": "ag-pre-inv-1",
            "workspacePaths": ["/workspace/proj"],
            "modelName": "Gemini 3.8 Flash (High)",
            "invocationNum": 1,
            "initialNumSteps": 0,
        },
        monkeypatch,
    )

    captured = json.loads(capsys.readouterr().out)
    assert "injectSteps" in captured
    messages = [s["ephemeralMessage"] for s in captured["injectSteps"]]
    assert any("Prior work" in m for m in messages)
    assert any("Be concise." in m for m in messages)

    events = session_store.read_events("ag-pre-inv-1")
    assert any(ev.get("event_type") == "SessionStart" for ev in events)
    assert any(ev.get("event_type") == "HookPolicySnapshot" for ev in events)
    assert any(ev.get("event_type") == "BrevityInjected" for ev in events)


def test_pre_invocation_handoff_advisor_triggers_when_threshold_exceeded(monkeypatch, capsys):
    monkeypatch.setattr(
        _config,
        "load",
        lambda: {"handoff_advisor": True, "handoff_advisor_tool_call_threshold": 50},
    )

    _run_handler(
        "PreInvocation",
        {
            "conversationId": "ag-handoff-sid",
            "workspacePaths": ["/repo"],
            "invocationNum": 3,
            "initialNumSteps": 60,
        },
        monkeypatch,
    )

    captured = json.loads(capsys.readouterr().out)
    messages = [s["ephemeralMessage"] for s in captured.get("injectSteps", [])]
    assert any("handoff" in m for m in messages)
    assert session_store.has_marker("ag-handoff-sid", "handoff_advised")

    # Subsequent invocation does not re-inject handoff reminder
    _run_handler(
        "PreInvocation",
        {
            "conversationId": "ag-handoff-sid",
            "workspacePaths": ["/repo"],
            "invocationNum": 4,
            "initialNumSteps": 65,
        },
        monkeypatch,
    )
    captured2 = json.loads(capsys.readouterr().out)
    assert captured2 == {"injectSteps": []}


def test_post_invocation_records_event(monkeypatch, capsys):
    _run_handler(
        "PostInvocation",
        {"conversationId": "ag-post-inv-sid", "invocationNum": 2},
        monkeypatch,
    )
    captured = json.loads(capsys.readouterr().out)
    assert captured == {"injectSteps": []}
    events = session_store.read_events("ag-post-inv-sid")
    assert any(
        ev.get("event_type") == "PostInvocation" and ev.get("invocation_num") == 2 for ev in events
    )


def test_pre_tool_dlp_blocks_env_file_view(monkeypatch, capsys):
    monkeypatch.setattr(_config, "load", lambda: {"dlp": True})
    _run_handler(
        "PreToolUse",
        {
            "conversationId": "ag-dlp-sid",
            "toolCall": {
                "name": "view_file",
                "args": {"AbsolutePath": "/project/.env"},
            },
        },
        monkeypatch,
    )
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "deny"
    assert "DLP policy" in out["reason"]


def test_pre_tool_dlp_blocks_env_file_write(monkeypatch, capsys):
    monkeypatch.setattr(_config, "load", lambda: {"dlp": True})
    _run_handler(
        "PreToolUse",
        {
            "conversationId": "ag-dlp-sid-2",
            "toolCall": {
                "name": "write_to_file",
                "args": {"TargetFile": "/project/prod.env", "CodeContent": "KEY=val"},
            },
        },
        monkeypatch,
    )
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "deny"
    assert "DLP policy" in out["reason"]


def test_pre_tool_dlp_blocks_cat_env_command(monkeypatch, capsys):
    monkeypatch.setattr(_config, "load", lambda: {"dlp": True})
    _run_handler(
        "PreToolUse",
        {
            "conversationId": "ag-dlp-sid-3",
            "workspacePaths": ["/project"],
            "toolCall": {
                "name": "run_command",
                "args": {"CommandLine": "cat .env"},
            },
        },
        monkeypatch,
    )
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "deny"
    assert "DLP policy" in out["reason"]


def test_pre_tool_loop_breaker_escalates_on_repeated_calls(monkeypatch, capsys):
    monkeypatch.setattr(_config, "load", lambda: {"loop_breaker": True, "dlp": False})
    sid = "ag-loop-sid"
    # Seed 4 identical PreToolUse events
    for _ in range(4):
        session_store.append_event(
            sid,
            {
                "event_type": "PreToolUse",
                "tool_name": "run_command",
                "tool_input": {"CommandLine": "npm test"},
            },
        )

    _run_handler(
        "PreToolUse",
        {
            "conversationId": sid,
            "toolCall": {
                "name": "run_command",
                "args": {"CommandLine": "npm test"},
            },
        },
        monkeypatch,
    )
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "ask"
    assert "loop-breaker" in out["reason"]


def test_post_tool_use_records_event_and_failure(monkeypatch, capsys):
    sid = "ag-post-tool-sid"
    _run_handler(
        "PostToolUse",
        {
            "conversationId": sid,
            "stepIdx": 10,
            "toolCall": {"name": "run_command", "args": {"CommandLine": "pytest"}},
        },
        monkeypatch,
    )
    assert json.loads(capsys.readouterr().out) == {}

    _run_handler(
        "PostToolUse",
        {
            "conversationId": sid,
            "stepIdx": 11,
            "toolCall": {"name": "run_command", "args": {"CommandLine": "pytest"}},
            "error": "command failed with code 1",
        },
        monkeypatch,
    )
    assert json.loads(capsys.readouterr().out) == {}

    events = session_store.read_events(sid)
    assert any(ev.get("event_type") == "PostToolUse" and ev.get("step_idx") == 10 for ev in events)
    assert any(
        ev.get("event_type") == "ToolFailure"
        and ev.get("step_idx") == 11
        and "failed" in ev.get("error", "")
        for ev in events
    )


def test_pre_tool_caps_unbounded_view_file(monkeypatch, capsys):
    _run_handler(
        "PreToolUse",
        {
            "conversationId": "ag-sid-view",
            "toolCall": {
                "name": "view_file",
                "args": {"AbsolutePath": "/repo/a.py", "StartLine": 401},
            },
        },
        monkeypatch,
    )

    assert json.loads(capsys.readouterr().out) == {
        "decision": "allow",
        "overwrite": {"EndLine": 600},
    }


def test_pre_tool_routes_supported_command_through_compressor(monkeypatch, capsys):
    monkeypatch.setattr(
        antigravity_handler,
        "maybe_compress",
        lambda *args, **kwargs: {"command": "rclm-compress --encoded-command abc"},
    )
    _run_handler(
        "PreToolUse",
        {
            "conversationId": "ag-sid-command",
            "toolCall": {"name": "run_command", "args": {"CommandLine": "pytest -q"}},
        },
        monkeypatch,
    )

    assert json.loads(capsys.readouterr().out)["overwrite"] == {
        "CommandLine": "rclm-compress --encoded-command abc"
    }


def test_malformed_json_exits_zero(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rclm-antigravity-hooks", "Stop"])
    monkeypatch.setattr("sys.stdin", StringIO("not json"))
    with pytest.raises(SystemExit) as exc_info:
        antigravity_handler.main()
    assert exc_info.value.code == 0


def test_unexpected_error_is_swallowed_and_exits_zero(monkeypatch):
    def boom(payload):
        raise RuntimeError("boom")

    monkeypatch.setattr(antigravity_handler, "_handle_stop", boom)
    _run_handler("Stop", {"conversationId": "ag-sid-4"}, monkeypatch)


def test_stop_allows_subsequent_turn_uploads(tmp_path, monkeypatch):
    transcript_path = str(tmp_path / "transcript.jsonl")
    tmp_path.joinpath("transcript.jsonl").write_text(
        json.dumps(
            {
                "step_index": 0,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "created_at": "2026-08-05T09:31:42Z",
                "content": "<USER_REQUEST>turn 1</USER_REQUEST>",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    uploaded: list[HookSessionRecord] = []

    async def fake_upload(record, *, max_retries=3):
        uploaded.append(record)
        return SimpleNamespace(cleanup_safe=True)

    monkeypatch.setattr(antigravity_handler, "upload_single", fake_upload)
    monkeypatch.setattr(antigravity_handler, "close_session", lambda: asyncio.sleep(0))

    # Turn 1 Stop
    _run_handler(
        "Stop",
        {"conversationId": "ag-sid-multi", "transcriptPath": transcript_path},
        monkeypatch,
    )
    assert len(uploaded) == 1
    assert len(uploaded[0].messages) == 1

    # Turn 2 appends to transcript
    with open(transcript_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "step_index": 1,
                    "source": "USER_EXPLICIT",
                    "type": "USER_INPUT",
                    "created_at": "2026-08-05T09:32:00Z",
                    "content": "<USER_REQUEST>turn 2</USER_REQUEST>",
                }
            )
            + "\n"
        )

    # Turn 2 Stop must not be blocked by finalized marker
    _run_handler(
        "Stop",
        {"conversationId": "ag-sid-multi", "transcriptPath": transcript_path},
        monkeypatch,
    )
    assert len(uploaded) == 2
    assert len(uploaded[1].messages) == 2
