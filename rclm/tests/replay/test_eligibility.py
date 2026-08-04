"""Tests for rclm.replay.eligibility: session/corpus gates and the funnel."""

from __future__ import annotations

from rclm.replay.eligibility import (
    MIN_TEXT_RESULT_TOKENS,
    MIN_TOOL_CALLS,
    MIN_TURNS,
    blob_eligibility,
    build_funnel,
    corpus_confidence,
    derive_timestamps_from_blob,
    session_metadata_eligibility,
)
from rclm.replay.engine import ReplayResult, ToolCallReplay


def _session(**overrides) -> dict:
    base = {
        "record_type": "session",
        "ended_at": "2026-08-01T00:00:00Z",
        "turn_count": MIN_TURNS,
        "tool_call_count": MIN_TOOL_CALLS,
        "model": "claude-sonnet-4-5",
    }
    base.update(overrides)
    return base


class TestSessionMetadataEligibility:
    def test_passes_when_all_gates_met(self):
        result = session_metadata_eligibility(_session())
        assert result.eligible

    def test_non_session_record_is_refused(self):
        result = session_metadata_eligibility(_session(record_type="proxy"))
        assert not result.eligible
        assert result.failing_constraint == "record_type"
        assert result.actual_value == "proxy"

    def test_missing_record_type_is_refused(self):
        result = session_metadata_eligibility(_session(record_type=None))
        assert not result.eligible
        assert result.failing_constraint == "record_type"
        assert result.actual_value is None

    def test_active_session_refused(self):
        result = session_metadata_eligibility(_session(ended_at=None))
        assert not result.eligible
        assert result.failing_constraint == "session_state"
        assert result.actual_value == "active"

    def test_too_few_turns_names_the_actual_value(self):
        result = session_metadata_eligibility(_session(turn_count=3))
        assert not result.eligible
        assert result.failing_constraint == "turn_count"
        assert result.actual_value == 3

    def test_too_few_tool_calls(self):
        result = session_metadata_eligibility(_session(tool_call_count=2))
        assert not result.eligible
        assert result.failing_constraint == "tool_call_count"

    def test_missing_tool_call_count_falls_back_to_blob(self):
        result = session_metadata_eligibility(
            _session(tool_call_count=None),
            blob={"tool_calls": [{} for _ in range(MIN_TOOL_CALLS)]},
        )
        assert result.eligible

    def test_missing_turn_count_falls_back_to_blob(self):
        result = session_metadata_eligibility(
            _session(turn_count=None),
            blob={"messages": [{"role": "user"} for _ in range(MIN_TURNS)]},
        )
        assert result.eligible

    def test_present_zero_tool_call_count_is_not_overridden_by_blob(self):
        result = session_metadata_eligibility(
            _session(tool_call_count=0),
            blob={"tool_calls": [{} for _ in range(MIN_TOOL_CALLS)]},
        )
        assert not result.eligible
        assert result.actual_value == 0

    def test_missing_model(self):
        result = session_metadata_eligibility(_session(model=None))
        assert not result.eligible
        assert result.failing_constraint == "model"

    def test_never_returns_a_number_alongside_failure(self):
        """PRD §6: fail any gate -> insufficient_data naming the constraint,
        never a number with a warning attached."""
        result = session_metadata_eligibility(_session(turn_count=1))
        assert result.eligible is False
        # EligibilityResult carries no reduction/pct field at all by construction.
        assert not hasattr(result, "pct")

    def test_min_turns_override_lowers_the_floor(self):
        result = session_metadata_eligibility(_session(turn_count=5), min_turns=5)
        assert result.eligible

    def test_min_turns_override_still_excludes_below_the_lowered_floor(self):
        result = session_metadata_eligibility(_session(turn_count=4), min_turns=5)
        assert not result.eligible
        assert result.failing_constraint == "turn_count"
        assert result.actual_value == 4

    def test_min_tool_calls_override_lowers_the_floor(self):
        result = session_metadata_eligibility(_session(tool_call_count=1), min_tool_calls=1)
        assert result.eligible

    def test_min_tool_calls_override_still_excludes_below_the_lowered_floor(self):
        result = session_metadata_eligibility(_session(tool_call_count=0), min_tool_calls=1)
        assert not result.eligible
        assert result.failing_constraint == "tool_call_count"
        assert result.actual_value == 0

    def test_default_min_tool_calls_is_unchanged(self):
        """The library default stays PRD §6's 10; only the MCP tool layer
        exposes a lower default (1) via an explicit min_tool_calls argument."""
        result = session_metadata_eligibility(_session(tool_call_count=9))
        assert not result.eligible
        assert result.failing_constraint == "tool_call_count"

    def test_null_ended_at_falls_back_to_blob_derived_timestamp(self):
        """A hooks bug left many finished sessions with ended_at=None in the
        row even though the blob has real message timestamps — passing the
        blob should let the session pass rather than refuse as 'active'."""
        blob = {
            "messages": [
                {"role": "user", "timestamp": "2026-07-24T15:45:26.754281+00:00"},
                {"role": "assistant", "timestamp": "2026-07-24T15:47:40.114079+00:00"},
            ]
        }
        result = session_metadata_eligibility(_session(ended_at=None), blob=blob)
        assert result.eligible

    def test_null_ended_at_without_blob_still_refused_as_active(self):
        result = session_metadata_eligibility(_session(ended_at=None))
        assert not result.eligible
        assert result.failing_constraint == "session_state"

    def test_null_ended_at_with_blob_lacking_message_timestamps_still_refused(self):
        result = session_metadata_eligibility(_session(ended_at=None), blob={"messages": []})
        assert not result.eligible
        assert result.failing_constraint == "session_state"

    def test_blob_fallback_does_not_override_a_present_ended_at(self):
        """A row that already has ended_at must not be second-guessed by the
        blob fallback, even if it's supplied."""
        blob = {"messages": [{"role": "user", "timestamp": "2020-01-01T00:00:00Z"}]}
        result = session_metadata_eligibility(_session(), blob=blob)
        assert result.eligible

    def test_blob_fallback_does_not_bypass_other_gates(self):
        """The blob fallback only covers session_state — turn_count/etc.
        still come from the row and still gate normally."""
        blob = {
            "messages": [
                {"role": "user", "timestamp": "2026-07-24T15:45:26.754281+00:00"},
                {"role": "assistant", "timestamp": "2026-07-24T15:47:40.114079+00:00"},
            ]
        }
        result = session_metadata_eligibility(_session(ended_at=None, turn_count=1), blob=blob)
        assert not result.eligible
        assert result.failing_constraint == "turn_count"


class TestDeriveTimestampsFromBlob:
    def test_derives_min_max_from_message_timestamps(self):
        blob = {
            "messages": [
                {"role": "user", "timestamp": "2026-07-24T15:47:40.114079+00:00"},
                {"role": "assistant", "timestamp": "2026-07-24T15:45:26.754281+00:00"},
                {"role": "user", "timestamp": "2026-07-24T15:46:00+00:00"},
            ]
        }
        started, ended = derive_timestamps_from_blob(blob)
        assert started == "2026-07-24T15:45:26.754281+00:00"
        assert ended == "2026-07-24T15:47:40.114079+00:00"

    def test_no_messages_returns_none_none(self):
        assert derive_timestamps_from_blob({"messages": []}) == (None, None)
        assert derive_timestamps_from_blob({}) == (None, None)

    def test_ignores_messages_missing_or_unparseable_timestamps(self):
        blob = {
            "messages": [
                {"role": "user", "timestamp": ""},
                {"role": "user"},
                {"role": "user", "timestamp": "not-a-date"},
                {"role": "assistant", "timestamp": "2026-07-24T15:45:26+00:00"},
            ]
        }
        started, ended = derive_timestamps_from_blob(blob)
        assert started == ended == "2026-07-24T15:45:26+00:00"

    def test_handles_z_suffix(self):
        blob = {"messages": [{"role": "user", "timestamp": "2026-07-24T15:45:26Z"}]}
        started, _ = derive_timestamps_from_blob(blob)
        assert started == "2026-07-24T15:45:26+00:00"

    def test_default_min_turns_is_unchanged(self):
        """The library default stays PRD §6's 10; only the MCP tool layer
        exposes a lower default (5) via an explicit min_turns argument."""
        result = session_metadata_eligibility(_session(turn_count=9))
        assert not result.eligible
        assert result.failing_constraint == "turn_count"


class TestBlobEligibility:
    @staticmethod
    def _result(text_tokens: int) -> ReplayResult:
        tokens_per_call, remainder = divmod(text_tokens, 5)
        return ReplayResult(
            calls=[
                ToolCallReplay(
                    index=index,
                    tool_name="Bash",
                    classification="uncovered",
                    mechanism=None,
                    original_tokens=tokens_per_call + (1 if index < remainder else 0),
                    compressed_tokens=tokens_per_call,
                    is_reachable=True,
                )
                for index in range(5)
            ]
        )

    def test_five_thousand_text_tokens_is_eligible(self):
        assert blob_eligibility(self._result(MIN_TEXT_RESULT_TOKENS)).eligible

    def test_below_five_thousand_text_tokens_is_refused(self):
        result = blob_eligibility(self._result(MIN_TEXT_RESULT_TOKENS - 1))
        assert not result.eligible
        assert result.failing_constraint == "text_result_tokens"
        assert result.actual_value == 4_999


class TestFunnel:
    def test_eligible_count_is_considered_minus_excluded(self):
        funnel = build_funnel(100, {"turn_count": 10, "tool_call_count": 5})
        assert funnel == {
            "considered": 100,
            "eligible": 85,
            "excluded": {"turn_count": 10, "tool_call_count": 5},
        }

    def test_no_exclusions_means_all_eligible(self):
        funnel = build_funnel(10, {})
        assert funnel["eligible"] == 10


class TestCorpusConfidence:
    def test_under_twenty_is_insufficient(self):
        confidence = corpus_confidence(5)
        assert confidence.level == "insufficient"

    def test_twenty_to_fifty_is_wide_distribution(self):
        confidence = corpus_confidence(30)
        assert confidence.level == "wide_distribution"

    def test_fifty_plus_is_full_confidence(self):
        confidence = corpus_confidence(50)
        assert confidence.level == "full"
