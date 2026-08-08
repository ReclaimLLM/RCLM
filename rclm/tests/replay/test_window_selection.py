"""Tests for the started_at-based corpus window filter in mcp_server.py.

Report 2's method (docs/whitepaper/report-2-token-savings-data-collection.md,
scripts/report2_collection_status.py): page the indexed `ingested_at`
boundary, then filter the exact study window locally on `started_at`. This
tests only the local-filter half — the pure function with no HTTP involved.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rclm.mcp_server import ReclaimLLMClient, _started_at_in_window


def test_session_inside_window_is_kept():
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    start = now - timedelta(days=30)
    session = {"started_at": "2026-07-15T00:00:00Z"}
    assert _started_at_in_window(session, start, now)


def test_session_before_window_is_excluded():
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    start = now - timedelta(days=30)
    session = {"started_at": "2026-06-01T00:00:00Z"}
    assert not _started_at_in_window(session, start, now)


def test_session_after_window_is_excluded():
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    start = now - timedelta(days=30)
    session = {"started_at": "2026-08-15T00:00:00Z"}
    assert not _started_at_in_window(session, start, now)


def test_missing_started_at_is_excluded():
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    start = now - timedelta(days=30)
    assert not _started_at_in_window({}, start, now)


def test_naive_datetime_is_treated_as_utc():
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    start = now - timedelta(days=30)
    session = {"started_at": "2026-07-15T00:00:00"}  # no timezone suffix
    assert _started_at_in_window(session, start, now)


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
                    "started_at": now,
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
    )

    assert captured["path"] == "/api/sessions/filter"
    assert "record_type" not in captured["params"]
    assert captured["params"]["include_changed_files"] == "false"
    assert captured["params"]["limit"] == 100
    assert [session["session_id"] for session in sessions] == [
        f"session-{index}" for index in range(75)
    ]
