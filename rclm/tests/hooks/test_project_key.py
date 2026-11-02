"""Tests for rclm.hooks._project_key."""

from __future__ import annotations

import pytest

from rclm.hooks._project_key import (
    _git_root,
    _resolve_cache_file,
    api_key_fingerprint,
    cache_db_path,
    derive_project_key,
    read_resolved_project_name,
    write_resolved_project_name,
)


@pytest.fixture(autouse=True)
def isolate_cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr("rclm.hooks._project_key._CACHE_ROOT", tmp_path / "projects")


class TestDeriveProjectKey:
    def test_empty_cwd_returns_unknown(self):
        assert derive_project_key("") == "unknown"

    def test_no_git_root_falls_back_to_cwd_basename(self, tmp_path):
        nested = tmp_path / "myrepo" / "sub"
        nested.mkdir(parents=True)
        assert derive_project_key(str(nested)) == "sub"

    def test_git_root_ancestor_wins(self, tmp_path):
        repo = tmp_path / "myrepo"
        nested = repo / "a" / "b"
        nested.mkdir(parents=True)
        (repo / ".git").mkdir()
        assert derive_project_key(str(nested)) == "myrepo"

    def test_prefers_fresh_server_resolution_over_approximation(self, tmp_path):
        repo = tmp_path / "myrepo"
        repo.mkdir()
        (repo / ".git").mkdir()
        write_resolved_project_name(str(repo), "Real Server Name")
        assert derive_project_key(str(repo)) == "Real Server Name"

    def test_falls_back_when_resolution_is_stale(self, tmp_path, monkeypatch):
        repo = tmp_path / "myrepo"
        repo.mkdir()
        (repo / ".git").mkdir()
        write_resolved_project_name(str(repo), "Real Server Name")
        # Age the cache entry past the TTL.
        monkeypatch.setattr("rclm.hooks._project_key._RESOLVE_TTL_SECONDS", -1)
        assert derive_project_key(str(repo)) == "myrepo"


class TestResolvedProjectNameCache:
    def test_empty_cwd_returns_none(self):
        assert read_resolved_project_name("") is None

    def test_no_resolution_yet_returns_none(self, tmp_path):
        assert read_resolved_project_name(str(tmp_path)) is None

    def test_round_trip(self, tmp_path):
        write_resolved_project_name(str(tmp_path), "Resolved")
        assert read_resolved_project_name(str(tmp_path)) == "Resolved"

    def test_corrupt_cache_file_returns_none(self, tmp_path):
        write_resolved_project_name(str(tmp_path), "Resolved")
        _resolve_cache_file(_git_root(str(tmp_path))).write_text("not json", encoding="utf-8")
        assert read_resolved_project_name(str(tmp_path)) is None

    def test_two_different_cwds_isolated(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        write_resolved_project_name(str(a), "A-name")
        assert read_resolved_project_name(str(b)) is None
        assert read_resolved_project_name(str(a)) == "A-name"

    def test_isolated_from_real_home_directory(self, tmp_path):
        """Guards a real isolation bug caught during review: _resolve_cache_file
        must read _CACHE_ROOT dynamically at call time, not from a constant bound
        once at import time -- otherwise monkeypatching _CACHE_ROOT in tests (the
        isolate_cache_root fixture) would silently miss this cache and write to
        the developer's real home directory instead."""
        write_resolved_project_name(str(tmp_path), "Resolved")
        cache_file = _resolve_cache_file(_git_root(str(tmp_path)))
        assert cache_file.is_relative_to(tmp_path)
        assert "_resolved" in cache_file.parts


class TestApiKeyFingerprint:
    def test_empty_key_returns_sentinel(self):
        assert api_key_fingerprint("") == "no-key"

    def test_deterministic(self):
        assert api_key_fingerprint("key-a") == api_key_fingerprint("key-a")

    def test_differs_for_different_keys(self):
        assert api_key_fingerprint("key-a") != api_key_fingerprint("key-b")


class TestCacheDbPath:
    def test_two_api_keys_same_project_get_different_paths(self):
        a = cache_db_path("myproj", "key-a")
        b = cache_db_path("myproj", "key-b")
        assert a != b
        # Same project_key, different fingerprint directory -- this is the whole
        # point: two developers sharing one $HOME get separate caches (PRD §5.3).
        assert a.parent.parent == b.parent.parent
        assert a.parent != b.parent
