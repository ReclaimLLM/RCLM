"""Tests for rclm.hooks.brief_rank."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone

import pytest

from rclm._config import Credentials
from rclm.hooks import brief_cache, brief_rank, session_store
from rclm.hooks._project_key import cache_db_path

NOW = datetime.now(timezone.utc)
RECENT = (NOW - timedelta(days=2)).isoformat()


def _pack(
    *, title="Weekly tokens digest email", summary="Built the weekly digest.", sha_after="deadbeef"
):
    return {
        "etag": "abc123",
        "project_name": "myproj",
        "built_at": NOW.isoformat(),
        "sessions": [
            {
                "session_id": "s1",
                "title": title,
                "category": "feature",
                "agent_client": "claude",
                "ended_at": RECENT,
                "summary": summary,
                "tags": ["email", "digest"],
                "files": [{"path": "server/alerts/digest_task.py", "sha_after": sha_after}],
            }
        ],
        "files": [
            {
                "path": "server/alerts/digest_task.py",
                "last_session_id": "s1",
                "last_edited_at": RECENT,
                "sha_after": sha_after,
                "busyness": 4,
                "cochange": [{"path": "server/alerts/recipients.py", "sessions_coedited": 3}],
            }
        ],
    }


@pytest.fixture(autouse=True)
def isolate_cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr("rclm.hooks._project_key._CACHE_ROOT", tmp_path / "projects")


@pytest.fixture(autouse=True)
def isolate_sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")


@pytest.fixture
def seeded_cwd(tmp_path, monkeypatch):
    """A project dir with a synced cache and credentials resolving to it."""
    cwd = tmp_path / "myproj"
    cwd.mkdir()
    monkeypatch.setattr(
        "rclm.hooks.brief_rank._config.resolve_credentials",
        lambda: Credentials(server_url="https://example.test", api_key="k1"),
    )
    db_path = cache_db_path("myproj", "k1")
    brief_cache.rebuild_cache(db_path, _pack())
    return cwd


# ---------------------------------------------------------------------------
# _tokenize / _is_task_shaped
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_splits_camel_and_snake_case(self):
        tokens = brief_rank._tokenize("digestTask server_alerts digest-task")
        assert "digest" in tokens
        assert "task" in tokens
        assert "server" in tokens
        assert "alerts" in tokens

    def test_drops_stopwords(self):
        tokens = brief_rank._tokenize("can you please fix the digest email")
        assert "the" not in tokens
        assert "please" not in tokens
        assert "digest" in tokens


class TestIsTaskShaped:
    def test_short_prompt_not_task_shaped(self):
        assert brief_rank._is_task_shaped("yes") is False
        assert brief_rank._is_task_shaped("ok thanks") is False

    def test_longer_prompt_is_task_shaped(self):
        assert brief_rank._is_task_shaped("add a weekly digest email for tokens") is True


# ---------------------------------------------------------------------------
# _score_session -- confidence gate + missing-data renormalization
# ---------------------------------------------------------------------------


class TestScoreSession:
    def test_full_match_scores_high(self):
        score = brief_rank._score_session(
            bm25_norm=1.0, path_hits=1.0, ended_at=NOW.isoformat(), agent_client="claude", now=NOW
        )
        assert score >= brief_rank._CONFIDENCE_THRESHOLD

    def test_weak_match_scores_low(self):
        old = (NOW - timedelta(days=400)).isoformat()
        score = brief_rank._score_session(
            bm25_norm=0.1, path_hits=0.0, ended_at=old, agent_client=None, now=NOW
        )
        assert score < brief_rank._CONFIDENCE_THRESHOLD

    def test_missing_agent_term_is_dropped_and_renormalized_not_zeroed(self):
        kwargs = {"bm25_norm": 0.8, "path_hits": 0.5, "ended_at": NOW.isoformat(), "now": NOW}
        missing = brief_rank._score_session(agent_client=None, **kwargs)
        mismatched = brief_rank._score_session(agent_client="codex", **kwargs)
        # Both have identical bm25/path/recency inputs. A session with no
        # agent_client at all must not score the same as (or worse than) one
        # that actively mismatches the current client -- dropping+renormalizing
        # the term must score strictly higher than treating "missing" as "0".
        assert missing > mismatched

    def test_matching_agent_scores_higher_than_missing(self):
        kwargs = {"bm25_norm": 0.8, "path_hits": 0.5, "ended_at": NOW.isoformat(), "now": NOW}
        matching = brief_rank._score_session(agent_client="claude", **kwargs)
        missing = brief_rank._score_session(agent_client=None, **kwargs)
        assert matching > missing


# ---------------------------------------------------------------------------
# build_prompt_context -- end-to-end gate, staleness, turn gating
# ---------------------------------------------------------------------------


class TestBuildPromptContextGating:
    def test_off_by_default(self, seeded_cwd):
        result = brief_rank.build_prompt_context(
            str(seeded_cwd), "sess-1", "add a weekly digest email", {}, prior_events=[]
        )
        assert result is None

    def test_no_cwd_or_session_id(self):
        result = brief_rank.build_prompt_context(
            "", "sess-1", "add a weekly digest email", {"start_brief": True}, prior_events=[]
        )
        assert result is None

    def test_no_cache_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "rclm.hooks.brief_rank._config.resolve_credentials",
            lambda: Credentials(server_url="https://example.test", api_key="unsynced-key"),
        )
        result = brief_rank.build_prompt_context(
            str(tmp_path),
            "sess-1",
            "add a weekly digest email",
            {"start_brief": True},
            prior_events=[],
        )
        assert result is None

    def test_matching_first_prompt_injects(self, seeded_cwd):
        result = brief_rank.build_prompt_context(
            str(seeded_cwd),
            "sess-1",
            "add a weekly tokens digest email",
            {"start_brief": True},
            prior_events=[],
        )
        assert result is not None
        assert "Weekly tokens digest email" in result["text"]
        assert "Files that usually change together" in result["text"]

    def test_unrelated_prompt_does_not_inject(self, seeded_cwd):
        result = brief_rank.build_prompt_context(
            str(seeded_cwd),
            "sess-1",
            "what is the capital of mongolia",
            {"start_brief": True},
            prior_events=[],
        )
        assert result is None


class TestTurnGating:
    def test_short_followup_prompt_skipped(self, seeded_cwd):
        prior = [{"event_type": "UserPromptSubmit", "prompt": "add a weekly tokens digest email"}]
        result = brief_rank.build_prompt_context(
            str(seeded_cwd), "sess-1", "ok", {"start_brief": True}, prior_events=prior
        )
        assert result is None

    def test_followup_with_same_files_is_deduped(self, seeded_cwd):
        prompt = "add a weekly tokens digest email"
        first = brief_rank.build_prompt_context(
            str(seeded_cwd), "sess-1", prompt, {"start_brief": True}, prior_events=[]
        )
        assert first is not None
        prior = [{"event_type": "UserPromptSubmit", "prompt": prompt}]
        second = brief_rank.build_prompt_context(
            str(seeded_cwd),
            "sess-1",
            "fix a bug in the weekly tokens digest email flow",
            {"start_brief": True},
            prior_events=prior,
        )
        # Same underlying file set as the first injection -- Jaccard similarity
        # is high, so this must be suppressed rather than repeated.
        assert second is None


class TestStaleness:
    def test_changed_file_is_flagged(self, tmp_path, monkeypatch):
        cwd = tmp_path / "myproj"
        (cwd / "server" / "alerts").mkdir(parents=True)
        original = b"original content"
        sha = hashlib.sha256(original).hexdigest()
        monkeypatch.setattr(
            "rclm.hooks.brief_rank._config.resolve_credentials",
            lambda: Credentials(server_url="https://example.test", api_key="k2"),
        )
        db_path = cache_db_path("myproj", "k2")
        brief_cache.rebuild_cache(db_path, _pack(sha_after=sha))

        target = cwd / "server" / "alerts" / "digest_task.py"
        target.write_bytes(b"changed content, different from the synced snapshot")
        future = time.time() + 10
        import os

        os.utime(target, (future, future))

        result = brief_rank.build_prompt_context(
            str(cwd),
            "sess-1",
            "add a weekly tokens digest email",
            {"start_brief": True},
            prior_events=[],
        )
        assert result is not None
        assert "Changed since that work" in result["text"]

    def test_unchanged_file_is_not_flagged(self, tmp_path, monkeypatch):
        cwd = tmp_path / "myproj"
        (cwd / "server" / "alerts").mkdir(parents=True)
        content = b"original content"
        sha = hashlib.sha256(content).hexdigest()
        monkeypatch.setattr(
            "rclm.hooks.brief_rank._config.resolve_credentials",
            lambda: Credentials(server_url="https://example.test", api_key="k3"),
        )
        db_path = cache_db_path("myproj", "k3")
        brief_cache.rebuild_cache(db_path, _pack(sha_after=sha))

        target = cwd / "server" / "alerts" / "digest_task.py"
        target.write_bytes(content)
        future = time.time() + 10
        import os

        os.utime(target, (future, future))

        result = brief_rank.build_prompt_context(
            str(cwd),
            "sess-1",
            "add a weekly tokens digest email",
            {"start_brief": True},
            prior_events=[],
        )
        assert result is not None
        assert "Changed since that work" not in result["text"]


class TestBudgetFailOpen:
    def test_deadline_already_passed_injects_nothing(self, seeded_cwd, monkeypatch):
        # Simulate the 150ms budget already being exhausted before ranking
        # starts -- must fail open (return None) rather than crash or block.
        monkeypatch.setattr(brief_rank.time, "monotonic", lambda: 1_000_000.0)
        result = brief_rank._rank(
            cache_db_path("myproj", "k1"),
            prompt="add a weekly tokens digest email",
            cwd=str(seeded_cwd),
            deadline=0.0,
        )
        assert result is None
