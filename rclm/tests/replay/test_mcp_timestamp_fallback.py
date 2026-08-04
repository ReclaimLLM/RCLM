"""Tests that the mcp_server.py single-session replay paths correctly fall
back to blob-derived started_at/ended_at when the row's own columns are
null (the claude_handler.py Stop-hook bug affecting historical sessions).
"""

from __future__ import annotations

import pytest

from rclm import mcp_server
from rclm.mcp_server import _replay_corpus_pairs, _replay_one_session
from rclm.replay.eligibility import EligibilityResult

pytestmark = pytest.mark.asyncio


def _blob_with_messages(*, turn_count: int = 12, tool_call_count: int = 12) -> dict:
    return {
        "model": "claude-sonnet-4-5",
        "messages": [
            {"role": "user", "timestamp": "2026-07-24T15:45:26.754281+00:00"},
            {"role": "assistant", "timestamp": "2026-07-24T15:47:40.114079+00:00"},
        ],
        "tool_calls": [
            {
                "tool_use_id": f"toolu_{i}",
                "tool_name": "Bash",
                "tool_input": {"command": "cat foo.txt"},
                "tool_result": "\n".join(f"line {j}" for j in range(600)),
            }
            for i in range(tool_call_count)
        ],
    }


class _FakeClient:
    def __init__(self, *, meta: dict, blob: dict | None):
        self._meta = meta
        self._blob = blob
        self.blob_fetch_count = 0

    async def fetch_session_metadata(self, session_id: str) -> dict:
        return self._meta

    async def fetch_blob(self, session_id: str) -> dict | None:
        self.blob_fetch_count += 1
        return self._blob


class TestReplayOneSessionTimestampFallback:
    async def test_null_ended_at_recovers_via_blob_and_replays(self):
        meta = {
            "record_type": "session",
            "ended_at": None,
            "turn_count": 12,
            "tool_call_count": 12,
            "model": "claude-sonnet-4-5",
        }
        blob = _blob_with_messages()
        client = _FakeClient(meta=meta, blob=blob)

        result = await _replay_one_session(
            client, "sess-1", ("shell_compaction",), min_turns=10, min_tool_calls=10
        )

        assert result["verdict"] != "insufficient_data"
        assert client.blob_fetch_count == 1  # fallback fetch reused for the actual replay

    async def test_null_tool_count_recovers_via_blob(self):
        meta = {
            "record_type": "session",
            "ended_at": "2026-07-24T15:47:40.114079+00:00",
            "turn_count": 12,
            "tool_call_count": None,
            "model": "claude-sonnet-4-5",
        }
        client = _FakeClient(meta=meta, blob=_blob_with_messages())

        result = await _replay_one_session(
            client, "sess-null-count", ("shell_compaction",), min_turns=10, min_tool_calls=10
        )

        assert result["verdict"] != "insufficient_data"
        assert client.blob_fetch_count == 1

    async def test_null_ended_at_with_no_recoverable_blob_stays_insufficient_data(self):
        meta = {
            "record_type": "session",
            "ended_at": None,
            "turn_count": 12,
            "tool_call_count": 12,
            "model": "claude-sonnet-4-5",
        }
        client = _FakeClient(meta=meta, blob=None)

        result = await _replay_one_session(
            client, "sess-2", ("shell_compaction",), min_turns=10, min_tool_calls=10
        )

        assert result["verdict"] == "insufficient_data"
        assert result["insufficient_data"]["constraint"] == "session_state"

    async def test_null_ended_at_with_blob_lacking_timestamps_stays_insufficient_data(self):
        meta = {
            "record_type": "session",
            "ended_at": None,
            "turn_count": 12,
            "tool_call_count": 12,
            "model": "claude-sonnet-4-5",
        }
        client = _FakeClient(meta=meta, blob={"messages": [], "tool_calls": []})

        result = await _replay_one_session(
            client, "sess-3", ("shell_compaction",), min_turns=10, min_tool_calls=10
        )

        assert result["verdict"] == "insufficient_data"
        assert result["insufficient_data"]["constraint"] == "session_state"

    async def test_other_gate_failures_never_trigger_a_blob_fetch(self):
        """The fallback is scoped to session_state only — a turn_count
        failure must not spend a blob fetch trying to recover it."""
        meta = {
            "record_type": "session",
            "ended_at": "2026-07-24T15:47:40.114079+00:00",
            "turn_count": 1,
            "tool_call_count": 12,
            "model": "claude-sonnet-4-5",
        }
        client = _FakeClient(meta=meta, blob=_blob_with_messages())

        result = await _replay_one_session(
            client, "sess-4", ("shell_compaction",), min_turns=10, min_tool_calls=10
        )

        assert result["verdict"] == "insufficient_data"
        assert result["insufficient_data"]["constraint"] == "turn_count"
        assert client.blob_fetch_count == 0


class _CorpusClient:
    def __init__(self, count: int):
        self.candidates = [
            {
                "session_id": f"session-{index}",
                "record_type": "session",
                "ended_at": "2026-08-01T00:00:00Z",
                "turn_count": 12,
                "tool_call_count": 12,
                "model": "claude-sonnet-4-5",
            }
            for index in range(count)
        ]
        self.fetch_count = 0

    async def enumerate_corpus(self, **kwargs):
        return self.candidates

    async def fetch_blob(self, session_id: str):
        self.fetch_count += 1
        return {"index": int(session_id.rsplit("-", 1)[1])}


async def test_corpus_limit_counts_fully_eligible_sessions(monkeypatch):
    client = _CorpusClient(60)
    monkeypatch.setattr(mcp_server, "replay_blob", lambda blob, mechanisms: blob["index"])
    monkeypatch.setattr(
        mcp_server.replay_eligibility_mod,
        "blob_eligibility",
        lambda index: EligibilityResult(index >= 10, "text_result_tokens", 0),
    )

    pairs, funnel = await _replay_corpus_pairs(
        client,
        days=30,
        source="all",
        model_family=None,
        project=None,
        session_category=None,
        limit=50,
        min_turns=10,
        min_tool_calls=10,
    )

    assert len(pairs) == 50
    assert funnel == {
        "considered": 60,
        "eligible": 50,
        "excluded": {"text_result_tokens": 10},
    }
    assert client.fetch_count == 60
