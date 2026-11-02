"""Tests for rclm.hooks.brief_cache."""

from __future__ import annotations

import pytest

from rclm._config import Credentials
from rclm.hooks import brief_cache
from rclm.hooks._project_key import cache_db_path

PACK = {
    "etag": "abc123",
    "project_name": "myproj",
    "built_at": "2026-10-30T00:00:00+00:00",
    "sessions": [
        {
            "session_id": "s1",
            "title": "Add weekly digest email",
            "category": "feature",
            "agent_client": "claude",
            "ended_at": "2026-10-20T00:00:00+00:00",
            "summary": "Built the weekly tokens digest.",
            "tags": ["email", "digest"],
            "files": [{"path": "server/alerts/digest_task.py", "sha_after": "deadbeef"}],
        }
    ],
    "files": [
        {
            "path": "server/alerts/digest_task.py",
            "last_session_id": "s1",
            "last_edited_at": "2026-10-20T00:00:00+00:00",
            "sha_after": "deadbeef",
            "busyness": 4,
            "cochange": [{"path": "server/alerts/recipients.py", "sessions_coedited": 3}],
        }
    ],
}


@pytest.fixture(autouse=True)
def isolate_cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr("rclm.hooks._project_key._CACHE_ROOT", tmp_path / "projects")


class TestRebuildCacheAndReadEtag:
    def test_round_trip(self, tmp_path):
        db_path = tmp_path / "brief.sqlite"
        assert brief_cache.read_cached_etag(db_path) is None
        brief_cache.rebuild_cache(db_path, PACK)
        assert brief_cache.read_cached_etag(db_path) == "abc123"

    def test_rebuild_replaces_prior_contents(self, tmp_path):
        db_path = tmp_path / "brief.sqlite"
        brief_cache.rebuild_cache(db_path, PACK)
        second_pack = {**PACK, "etag": "def456", "sessions": [], "files": []}
        brief_cache.rebuild_cache(db_path, second_pack)
        assert brief_cache.read_cached_etag(db_path) == "def456"


class TestScheduleSync:
    def test_no_credentials_does_not_spawn(self, monkeypatch):
        monkeypatch.setattr("rclm.hooks.brief_cache._config.resolve_credentials", lambda: None)
        spawned = []
        monkeypatch.setattr(
            "subprocess.Popen", lambda *a, **kw: spawned.append((a, kw)) or object()
        )
        assert brief_cache.schedule_sync("/tmp/some/repo") is False
        assert not spawned

    def test_spawns_once_then_throttled(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "rclm.hooks.brief_cache._config.resolve_credentials",
            lambda: Credentials(server_url="https://example.test", api_key="k1"),
        )
        spawned = []
        monkeypatch.setattr(
            "subprocess.Popen", lambda *a, **kw: spawned.append((a, kw)) or object()
        )
        cwd = str(tmp_path)
        assert brief_cache.schedule_sync(cwd) is True
        assert len(spawned) == 1
        # Second call within the TTL window must not spawn again.
        assert brief_cache.schedule_sync(cwd) is False
        assert len(spawned) == 1


class TestRunSync:
    @pytest.mark.asyncio
    async def test_304_does_not_rebuild(self, monkeypatch):
        monkeypatch.setattr(
            "rclm.hooks.brief_cache._config.resolve_credentials",
            lambda: Credentials(server_url="https://example.test", api_key="k1"),
        )

        class _FakeClient:
            async def fetch_brief_pack(self, *a, **kw):
                return 304, None, kw.get("if_none_match")

        monkeypatch.setattr("rclm.mcp_server.ReclaimLLMClient", lambda: _FakeClient())
        rebuilt = []
        monkeypatch.setattr(
            "rclm.hooks.brief_cache.rebuild_cache", lambda *a, **kw: rebuilt.append(True)
        )
        result = await brief_cache.run_sync("myproj")
        assert result is False
        assert not rebuilt

    @pytest.mark.asyncio
    async def test_200_rebuilds_cache(self, monkeypatch):
        monkeypatch.setattr(
            "rclm.hooks.brief_cache._config.resolve_credentials",
            lambda: Credentials(server_url="https://example.test", api_key="k1"),
        )

        class _FakeClient:
            async def fetch_brief_pack(self, *a, **kw):
                return 200, PACK, "abc123"

        monkeypatch.setattr("rclm.mcp_server.ReclaimLLMClient", lambda: _FakeClient())
        result = await brief_cache.run_sync("myproj")
        assert result is True
        db_path = cache_db_path("myproj", "k1")
        assert brief_cache.read_cached_etag(db_path) == "abc123"

    @pytest.mark.asyncio
    async def test_no_cwd_skips_resolution_unchanged(self, monkeypatch):
        """run_sync(project_key) with no cwd (the pre-resolve call signature) must
        behave exactly as before -- no resolve attempt, project_key used as-is."""
        monkeypatch.setattr(
            "rclm.hooks.brief_cache._config.resolve_credentials",
            lambda: Credentials(server_url="https://example.test", api_key="k1"),
        )

        class _FakeClient:
            async def resolve_project_name(self, cwd):
                raise AssertionError("must not be called when cwd is empty")

            async def fetch_brief_pack(self, project_name, **kw):
                assert project_name == "myproj"
                return 200, PACK, "abc123"

        monkeypatch.setattr("rclm.mcp_server.ReclaimLLMClient", lambda: _FakeClient())
        assert await brief_cache.run_sync("myproj") is True

    @pytest.mark.asyncio
    async def test_cwd_resolution_redirects_fetch_and_cache(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "rclm.hooks.brief_cache._config.resolve_credentials",
            lambda: Credentials(server_url="https://example.test", api_key="k1"),
        )
        fetched_with = []

        class _FakeClient:
            async def resolve_project_name(self, cwd):
                return "Real Server Name"

            async def fetch_brief_pack(self, project_name, **kw):
                fetched_with.append(project_name)
                return 200, {**PACK, "project_name": project_name}, "abc123"

        monkeypatch.setattr("rclm.mcp_server.ReclaimLLMClient", lambda: _FakeClient())
        cwd = str(tmp_path)
        result = await brief_cache.run_sync("myproj-approx", cwd)
        assert result is True
        assert fetched_with == ["Real Server Name"]
        # Cache lands under the resolved name, not the approximation.
        assert brief_cache.read_cached_etag(cache_db_path("Real Server Name", "k1")) == "abc123"
        assert brief_cache.read_cached_etag(cache_db_path("myproj-approx", "k1")) is None

    @pytest.mark.asyncio
    async def test_resolution_failure_falls_back_to_approximation(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "rclm.hooks.brief_cache._config.resolve_credentials",
            lambda: Credentials(server_url="https://example.test", api_key="k1"),
        )

        class _FakeClient:
            async def resolve_project_name(self, cwd):
                raise RuntimeError("server unreachable")

            async def fetch_brief_pack(self, project_name, **kw):
                assert project_name == "myproj-approx"
                return 200, PACK, "abc123"

        monkeypatch.setattr("rclm.mcp_server.ReclaimLLMClient", lambda: _FakeClient())
        result = await brief_cache.run_sync("myproj-approx", str(tmp_path))
        assert result is True
        assert brief_cache.read_cached_etag(cache_db_path("myproj-approx", "k1")) == "abc123"
