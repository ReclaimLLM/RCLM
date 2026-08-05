"""Tests for rclm.hooks.antigravity_handler."""

from __future__ import annotations

import json
from io import StringIO

import pytest

from rclm._models import HookSessionRecord
from rclm.hooks import antigravity_handler


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
        },
        monkeypatch,
    )

    assert len(uploaded) == 1
    record = uploaded[0]
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


def test_stop_closes_uploader_session(tmp_path, monkeypatch):
    """Regression test: Stop must close the module-level aiohttp session in
    the same event loop it uploaded on, or aiohttp emits "Unclosed client
    session"/"Unclosed connector" ResourceWarnings to stderr on every Stop
    (confirmed via a real subprocess repro against a live Claude Code
    session transcript -- see close_session's docstring in _uploader.py)."""
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
    assert uploaded[0].cwd == ""


def test_non_stop_events_are_ignored_without_error(monkeypatch):
    """PreToolUse/PostToolUse/PreInvocation/PostInvocation are never registered
    by the installer, but if Antigravity ever fires one anyway, the handler
    must exit cleanly rather than attempt to build/upload a record."""
    uploaded: list[HookSessionRecord] = []

    async def fake_upload(record, *, max_retries=3):
        uploaded.append(record)

    monkeypatch.setattr(antigravity_handler, "upload_single", fake_upload)

    for event in ("PreToolUse", "PostToolUse", "PreInvocation", "PostInvocation"):
        _run_handler(event, {"conversationId": "ag-sid-3"}, monkeypatch)

    assert uploaded == []


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
