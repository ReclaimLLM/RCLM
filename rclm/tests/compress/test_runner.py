"""Tests for rclm.compress.runner (apply_filter mechanism tagging, track_savings)."""

from rclm.compress.runner import FilterResult, apply_filter, track_savings
from rclm.hooks import session_store


class TestApplyFilterMechanismTagging:
    def test_git_tagged_legacy_compress(self):
        output = " M src/file.py\n" * 40
        result = apply_filter("git status", output, "")
        assert isinstance(result, FilterResult)
        assert result.mechanism == "legacy_compress"

    def test_rg_tagged_h2_search_shaping(self):
        lines = [f"src/file{i}.py:1:match" for i in range(40)]
        output = "\n".join(lines)
        result = apply_filter("rg -n TODO .", output, "")
        assert result.mechanism == "H2_search_shaping"

    def test_pytest_tagged_test_filter(self):
        output = "\n".join(f"test_{i} PASSED" for i in range(40)) + "\n=== 40 passed ==="
        result = apply_filter("pytest tests/", output, "")
        assert result.mechanism == "test_filter"

    def test_ls_tagged_legacy_compress(self):
        output = "\n".join(f"file{i}.py" for i in range(40))
        result = apply_filter("ls -la", output, "")
        assert result.mechanism == "legacy_compress"

    def test_unmatched_large_output_tagged_h3_exec_compaction(self):
        output = "\n".join(f"unique line {i}" for i in range(100))
        result = apply_filter("some-custom-tool", output, "")
        assert result.mechanism == "H3_exec_compaction"

    def test_small_unmatched_output_no_mechanism(self):
        result = apply_filter("echo hi", "hi\n", "")
        assert result.mechanism is None
        assert result.text == "hi\n"

    def test_empty_command_no_mechanism(self):
        result = apply_filter("", "some output", "")
        assert result.mechanism is None


class TestTrackSavings:
    def test_no_session_id_no_event(self, monkeypatch, tmp_path):
        monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
        monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)

        track_savings("original text", "short", "H3_exec_compaction", applied=True)

        assert not (tmp_path / "sessions").exists()

    def test_no_mechanism_no_event(self, monkeypatch, tmp_path):
        monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

        track_savings("same text", "same text", None, applied=True, session_id="sid-track-1")

        assert session_store.read_events("sid-track-1") == []

    def test_emits_mechanism_saving_event(self, monkeypatch, tmp_path):
        monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

        track_savings(
            "x" * 4000, "x" * 400, "H2_search_shaping", applied=True, session_id="sid-track-2"
        )

        events = session_store.read_events("sid-track-2")
        assert len(events) == 1
        assert events[0]["event_type"] == "MechanismSaving"
        assert events[0]["mechanism"] == "H2_search_shaping"
        assert events[0]["applied"] is True
        assert events[0]["tokens_saved_estimate"] == 900  # (4000-400)//4

    def test_shadow_applied_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")

        track_savings(
            "x" * 400, "x" * 40, "H3_exec_compaction", applied=False, session_id="sid-track-3"
        )

        events = session_store.read_events("sid-track-3")
        assert events[0]["applied"] is False

    def test_uses_env_session_id_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
        monkeypatch.setenv("CLAUDE_SESSION_ID", "sid-track-env")

        track_savings("x" * 400, "x" * 40, "legacy_compress", applied=True)

        events = session_store.read_events("sid-track-env")
        assert len(events) == 1
