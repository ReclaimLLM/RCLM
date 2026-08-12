"""Tests for caller-owned, ingested-at-based replay corpus selection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rclm.mcp_server import ReclaimLLMClient, _matches_replay_source, _timestamp_in_window


def test_session_inside_window_is_kept():
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    start = now - timedelta(days=30)
    session = {"ingested_at": "2026-07-15T00:00:00Z"}
    assert _timestamp_in_window(session, "ingested_at", start, now)


def test_session_before_window_is_excluded():
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    start = now - timedelta(days=30)
    session = {"ingested_at": "2026-06-01T00:00:00Z"}
    assert not _timestamp_in_window(session, "ingested_at", start, now)


def test_session_after_window_is_excluded():
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    start = now - timedelta(days=30)
    session = {"ingested_at": "2026-08-15T00:00:00Z"}
    assert not _timestamp_in_window(session, "ingested_at", start, now)


def test_missing_ingested_at_is_excluded():
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    start = now - timedelta(days=30)
    assert not _timestamp_in_window({}, "ingested_at", start, now)


def test_naive_datetime_is_treated_as_utc():
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    start = now - timedelta(days=30)
    session = {"ingested_at": "2026-07-15T00:00:00"}  # no timezone suffix
    assert _timestamp_in_window(session, "ingested_at", start, now)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.6-sol", True),
        ("codex-unknown", True),
        ("o3-mini", False),
        ("claude-sonnet-5", False),
        (None, False),
    ],
)
def test_codex_source_accepts_only_gpt_and_codex_prefixes(model, expected):
    assert _matches_replay_source({"model": model}, "codex") is expected


@pytest.mark.asyncio
async def test_corpus_enumeration_scans_four_times_target_and_returns_candidates(monkeypatch):
    client = object.__new__(ReclaimLLMClient)
    captured: dict = {}
    now = datetime.now(timezone.utc).isoformat()

    async def fake_request(method, path, *, params=None):
        captured.update({"method": method, "path": path, "params": params})
        return {
            "sessions": [
                {
                    "session_id": f"session-{index}",
                    "record_type": "session",
                    "ingested_at": now,
                }
                for index in range(75)
            ]
        }

    monkeypatch.setattr(client, "_request", fake_request)

    sessions = await client.enumerate_corpus(
        days=30,
        source=None,
        model_family=None,
        project_name=None,
        session_category=None,
        limit=50,
        min_turns=5,
        min_tool_calls=5,
    )

    assert captured["path"] == "/api/sessions/filter"
    assert "record_type" not in captured["params"]
    assert captured["params"]["include_changed_files"] == "false"
    assert captured["params"]["limit"] == 100
    assert captured["params"]["scope"] == "mine"
    assert captured["params"]["min_turns"] == 5
    assert captured["params"]["min_tool_calls"] == 5
    assert [session["session_id"] for session in sessions] == [
        f"session-{index}" for index in range(75)
    ]
