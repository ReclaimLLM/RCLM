"""Tests for rclm.hooks._analytics."""

from rclm._models import FileDiff, ToolCall
from rclm.hooks._analytics import (
    FALLBACK_IMAGE_TOKENS,
    aggregate_mechanism_savings,
    compute_session_analytics,
    estimate_image_tokens,
    estimate_tokens,
    mechanism_saving_event,
)

# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_none_returns_zero(self):
        assert estimate_tokens(None) == 0

    def test_short_string(self):
        assert estimate_tokens("hi") == 1  # min 1

    def test_longer_string(self):
        text = "a" * 100
        assert estimate_tokens(text) == 25  # 100 / 4

    def test_dict(self):
        result = estimate_tokens({"key": "value"})
        assert result > 0

    def test_list(self):
        result = estimate_tokens([1, 2, 3])
        assert result > 0

    def test_empty_string(self):
        assert estimate_tokens("") == 0


# ---------------------------------------------------------------------------
# estimate_image_tokens
# ---------------------------------------------------------------------------


class TestEstimateImageTokens:
    def test_unknown_dimensions_falls_back(self):
        assert estimate_image_tokens(None, None) == FALLBACK_IMAGE_TOKENS
        assert estimate_image_tokens(0, 100) == FALLBACK_IMAGE_TOKENS
        assert estimate_image_tokens(100, -5) == FALLBACK_IMAGE_TOKENS

    def test_anthropic_formula_matches_published_width_times_height_over_750(self):
        # https://docs.anthropic.com/en/docs/build-with-claude/vision
        assert estimate_image_tokens(1000, 750, provider="anthropic") == 1000

    def test_anthropic_is_default_and_fallback_for_unknown_providers(self):
        assert estimate_image_tokens(1000, 750) == 1000
        assert estimate_image_tokens(1000, 750, provider="some-other-vendor") == 1000

    def test_openai_low_detail_is_flat_85_regardless_of_size(self):
        assert estimate_image_tokens(4000, 4000, provider="openai", detail="low") == 85
        assert estimate_image_tokens(10, 10, provider="openai", detail="low") == 85

    def test_openai_high_detail_small_image_one_tile(self):
        # 512x512 needs no scaling (under 2048 fit, under 768 shortest-side) —
        # exactly one 512x512 tile: 85 base + 170*1*1.
        assert estimate_image_tokens(512, 512, provider="openai", detail="high") == 255

    def test_openai_high_detail_scales_shortest_side_then_tiles(self):
        # 1024x1024: already under the 2048 fit; shortest side 1024 > 768, so
        # scaled to 768x768 -> ceil(768/512)=2 tiles per side -> 85 + 170*4.
        assert estimate_image_tokens(1024, 1024, provider="openai", detail="high") == 765

    def test_openai_high_detail_never_upscales_a_small_image(self):
        # Already-small image (both scale steps are no-ops) stays a single tile.
        assert estimate_image_tokens(100, 100, provider="openai", detail="high") == 255


# ---------------------------------------------------------------------------
# compute_session_analytics
# ---------------------------------------------------------------------------


class TestComputeSessionAnalytics:
    def test_empty_inputs(self):
        result = compute_session_analytics([], [])
        assert result["tool_token_stats"] is None
        assert result["tool_call_count"] is None
        assert result["unique_files_modified"] is None
        assert result["dominant_tool"] is None

    def test_counts_tools(self):
        tool_calls = [
            ToolCall("t1", "Bash", {"command": "ls"}, "output", "2024-01-01T00:00:00Z"),
            ToolCall("t2", "Bash", {"command": "git status"}, "output", "2024-01-01T00:00:01Z"),
            ToolCall("t3", "Read", {"file_path": "/foo"}, "content", "2024-01-01T00:00:02Z"),
        ]
        result = compute_session_analytics(tool_calls, [])
        assert result["tool_call_count"] == 3
        assert result["dominant_tool"] == "Bash"
        assert result["tool_token_stats"]["Bash"]["count"] == 2
        assert result["tool_token_stats"]["Read"]["count"] == 1

    def test_unique_files(self):
        diffs = [
            FileDiff("a.py", None, "content", "+content"),
            FileDiff("b.py", "old", "new", "-old\n+new"),
            FileDiff("a.py", "v1", "v2", "-v1\n+v2"),  # duplicate path
        ]
        result = compute_session_analytics([], diffs)
        assert result["unique_files_modified"] == 2

    def test_uses_existing_token_estimates(self):
        tc = ToolCall(
            "t1",
            "Bash",
            {"command": "ls"},
            "output",
            "2024-01-01T00:00:00Z",
            input_token_estimate=10,
            output_token_estimate=20,
        )
        result = compute_session_analytics([tc], [])
        assert result["tool_token_stats"]["Bash"]["input_tokens"] == 10
        assert result["tool_token_stats"]["Bash"]["output_tokens"] == 20


# ---------------------------------------------------------------------------
# mechanism_saving_event / aggregate_mechanism_savings
# ---------------------------------------------------------------------------


class TestMechanismSavingEvent:
    def test_shape(self):
        ev = mechanism_saving_event(
            "range_cache",
            applied=True,
            tokens_saved_estimate=42,
            measurement_kind="measured",
            file_path="src/api.py",
            raw_token_estimate=50,
            compressed_token_estimate=8,
        )
        assert ev["event_type"] == "MechanismSaving"
        assert ev["mechanism"] == "range_cache"
        assert ev["applied"] is True
        assert ev["tokens_saved_estimate"] == 42
        assert ev["measurement_kind"] == "measured"
        assert ev["file_path"] == "src/api.py"
        assert "timestamp" in ev

    def test_modeled_cost_estimate_included_only_when_set(self):
        with_cost = mechanism_saving_event(
            "image_eviction",
            applied=False,
            tokens_saved_estimate=0,
            measurement_kind="estimated",
            raw_token_estimate=800,
            modeled_cost_estimate=1000,
        )
        assert with_cost["modeled_cost_estimate"] == 1000

        without_cost = mechanism_saving_event("hash_dedupe", applied=True, tokens_saved_estimate=42)
        assert "modeled_cost_estimate" not in without_cost


class TestAggregateMechanismSavings:
    def test_no_savings_events(self):
        events = [{"event_type": "SessionStart"}, {"event_type": "PreToolUse"}]
        assert aggregate_mechanism_savings(events) is None

    def test_empty_events(self):
        assert aggregate_mechanism_savings([]) is None

    def test_aggregates_by_mechanism(self):
        events = [
            {"event_type": "SessionStart"},
            mechanism_saving_event("H1_read_cache", applied=True, tokens_saved_estimate=1000),
            mechanism_saving_event("H1_read_cache", applied=True, tokens_saved_estimate=500),
            mechanism_saving_event("H3_exec_compaction", applied=True, tokens_saved_estimate=200),
        ]
        result = aggregate_mechanism_savings(events)
        assert result is not None
        assert result["H1_read_cache"] == {
            "applied_count": 2,
            "shadow_count": 0,
            "tokens_saved_estimate": 1500,
            "measurement_kind": "estimated",
        }
        assert result["H3_exec_compaction"] == {
            "applied_count": 1,
            "shadow_count": 0,
            "tokens_saved_estimate": 200,
            "measurement_kind": "estimated",
        }

    def test_aggregates_measured_file_attribution(self):
        events = [
            mechanism_saving_event(
                "range_cache",
                applied=True,
                tokens_saved_estimate=120,
                measurement_kind="measured",
                file_path="src/api.py",
            ),
            mechanism_saving_event(
                "range_cache",
                applied=True,
                tokens_saved_estimate=30,
                measurement_kind="measured",
                file_path="src/api.py",
            ),
        ]
        result = aggregate_mechanism_savings(events)
        assert result["range_cache"] == {
            "applied_count": 2,
            "shadow_count": 0,
            "tokens_saved_estimate": 150,
            "measurement_kind": "measured",
            "files": {
                "src/api.py": {
                    "applied_count": 2,
                    "shadow_count": 0,
                    "tokens_saved_estimate": 150,
                }
            },
        }

    def test_shadow_vs_applied_counts_separately(self):
        events = [
            mechanism_saving_event("H4_loop_breaker", applied=False, tokens_saved_estimate=100),
            mechanism_saving_event("H4_loop_breaker", applied=True, tokens_saved_estimate=100),
        ]
        result = aggregate_mechanism_savings(events)
        assert result is not None
        assert result["H4_loop_breaker"]["applied_count"] == 1
        assert result["H4_loop_breaker"]["shadow_count"] == 1
        assert result["H4_loop_breaker"]["tokens_saved_estimate"] == 200

    def test_unknown_mechanism_field_defaults(self):
        events = [{"event_type": "MechanismSaving", "applied": True, "tokens_saved_estimate": 5}]
        result = aggregate_mechanism_savings(events)
        assert result is not None
        assert "unknown" in result

    def test_modeled_cost_estimate_is_not_summed_into_the_rollup(self):
        events = [
            mechanism_saving_event(
                "image_eviction",
                applied=False,
                tokens_saved_estimate=0,
                measurement_kind="estimated",
                raw_token_estimate=800,
                modeled_cost_estimate=1000,
            ),
        ]
        result = aggregate_mechanism_savings(events)
        assert result is not None
        assert "modeled_cost_estimate" not in result["image_eviction"]
        assert result["image_eviction"]["tokens_saved_estimate"] == 0
