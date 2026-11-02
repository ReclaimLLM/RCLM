"""Tests for rclm.hooks.start_brief."""

from __future__ import annotations

from rclm.hooks.start_brief import build_session_start_context, card_hash

# ---------------------------------------------------------------------------
# card_hash
# ---------------------------------------------------------------------------


class TestCardHash:
    def test_deterministic(self):
        assert card_hash("hello") == card_hash("hello")

    def test_differs_for_different_text(self):
        assert card_hash("hello") != card_hash("world")


# ---------------------------------------------------------------------------
# build_session_start_context
# ---------------------------------------------------------------------------


class TestBuildSessionStartContext:
    def test_off_by_default(self, tmp_path):
        assert build_session_start_context(str(tmp_path), {}) is None

    def test_disabled_explicitly(self, tmp_path):
        assert build_session_start_context(str(tmp_path), {"start_brief": False}) is None

    def test_no_cwd_returns_none(self):
        assert build_session_start_context("", {"start_brief": True}) is None

    def test_nonexistent_cwd_returns_none(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        assert build_session_start_context(str(missing), {"start_brief": True}) is None

    def test_empty_dir_returns_none(self, tmp_path):
        assert build_session_start_context(str(tmp_path), {"start_brief": True}) is None

    def test_top_level_dirs_listed(self, tmp_path):
        (tmp_path / "server").mkdir()
        (tmp_path / "frontend").mkdir()
        result = build_session_start_context(str(tmp_path), {"start_brief": True})
        assert result is not None
        assert "frontend" in result["text"]
        assert "server" in result["text"]

    def test_ignored_dirs_excluded(self, tmp_path):
        (tmp_path / "server").mkdir()
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / ".git").mkdir()
        result = build_session_start_context(str(tmp_path), {"start_brief": True})
        assert result is not None
        assert "node_modules" not in result["text"]
        assert "__pycache__" not in result["text"]
        assert ".git" not in result["text"]

    def test_sibling_repos_detected(self, tmp_path):
        cwd = tmp_path / "my-backend"
        cwd.mkdir()
        (cwd / "src").mkdir()
        sibling = tmp_path / "my-frontend"
        sibling.mkdir()
        (sibling / ".git").mkdir()
        not_a_repo = tmp_path / "docs"
        not_a_repo.mkdir()
        result = build_session_start_context(str(cwd), {"start_brief": True})
        assert result is not None
        assert "my-frontend" in result["text"]
        assert "docs" not in result["text"].split("Sibling repos:")[-1].splitlines()[0]

    def test_latest_migration_detected(self, tmp_path):
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        (migrations / "001_initial.sql").write_text("", encoding="utf-8")
        (migrations / "012_add_index.sql").write_text("", encoding="utf-8")
        (migrations / "007_rename.sql").write_text("", encoding="utf-8")
        result = build_session_start_context(str(tmp_path), {"start_brief": True})
        assert result is not None
        assert "012_add_index.sql" in result["text"]

    def test_result_has_hash_and_tokens(self, tmp_path):
        (tmp_path / "server").mkdir()
        result = build_session_start_context(str(tmp_path), {"start_brief": True})
        assert result is not None
        assert result["card_hash"] == card_hash(result["text"])
        assert result["card_tokens"] > 0
