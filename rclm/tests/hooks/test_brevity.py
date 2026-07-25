"""Tests for rclm.hooks.brevity."""

from __future__ import annotations

from rclm.hooks.brevity import (
    DEFAULT_INSTRUCTION,
    _has_existing_brevity_guidance,
    build_session_start_context,
    instruction_hash,
)

# ---------------------------------------------------------------------------
# DEFAULT_INSTRUCTION
# ---------------------------------------------------------------------------


class TestDefaultInstruction:
    def test_under_170_tokens(self):
        # ~4 chars/token heuristic, same one used by estimate_tokens elsewhere.
        assert len(DEFAULT_INSTRUCTION) // 4 < 170

    def test_does_not_target_reasoning_or_explicit_requests(self):
        lowered = DEFAULT_INSTRUCTION.lower()
        assert "reasoning" in lowered or "error analysis" in lowered


# ---------------------------------------------------------------------------
# instruction_hash
# ---------------------------------------------------------------------------


class TestInstructionHash:
    def test_deterministic(self):
        assert instruction_hash("hello") == instruction_hash("hello")

    def test_differs_for_different_text(self):
        assert instruction_hash("hello") != instruction_hash("world")


# ---------------------------------------------------------------------------
# _has_existing_brevity_guidance
# ---------------------------------------------------------------------------


class TestHasExistingBrevityGuidance:
    def test_no_cwd_returns_false(self):
        assert _has_existing_brevity_guidance("") is False

    def test_no_guidance_files_returns_false(self, tmp_path):
        assert _has_existing_brevity_guidance(str(tmp_path)) is False

    def test_claude_md_with_brevity_language_detected(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("Be concise in all responses.", encoding="utf-8")
        assert _has_existing_brevity_guidance(str(tmp_path)) is True

    def test_agents_md_with_brevity_language_detected(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text(
            "Keep responses short and avoid preamble.", encoding="utf-8"
        )
        assert _has_existing_brevity_guidance(str(tmp_path)) is True

    def test_claude_md_without_brevity_language_not_detected(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text(
            "This project uses FastAPI and Postgres.", encoding="utf-8"
        )
        assert _has_existing_brevity_guidance(str(tmp_path)) is False

    def test_detected_within_size_cap(self, tmp_path):
        padding = "x" * 50_000
        content = f"{padding}\nBe terse.\n{padding}"
        (tmp_path / "CLAUDE.md").write_text(content, encoding="utf-8")
        assert _has_existing_brevity_guidance(str(tmp_path)) is True

    def test_not_detected_past_size_cap(self, tmp_path):
        padding = "x" * 150_000
        content = f"{padding}\nBe terse.\n"
        (tmp_path / "CLAUDE.md").write_text(content, encoding="utf-8")
        assert _has_existing_brevity_guidance(str(tmp_path)) is False


# ---------------------------------------------------------------------------
# build_session_start_context
# ---------------------------------------------------------------------------


class TestBuildSessionStartContext:
    def test_off_by_default(self, tmp_path):
        assert build_session_start_context(str(tmp_path), {}) is None

    def test_disabled_explicitly(self, tmp_path):
        assert build_session_start_context(str(tmp_path), {"brevity": False}) is None

    def test_enabled_returns_default_instruction(self, tmp_path):
        result = build_session_start_context(str(tmp_path), {"brevity": True})
        assert result is not None
        assert result["text"]
        assert result["instruction_hash"] == instruction_hash(result["text"])
        assert result["instruction_tokens"] > 0

    def test_enabled_but_project_already_has_guidance_skips(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("Be concise.", encoding="utf-8")
        assert build_session_start_context(str(tmp_path), {"brevity": True}) is None

    def test_custom_instruction_from_config(self, tmp_path):
        result = build_session_start_context(
            str(tmp_path),
            {"brevity": True, "brevity_instruction": "Custom brevity text."},
        )
        assert result is not None
        assert result["text"] == "Custom brevity text."
