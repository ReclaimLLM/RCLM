"""Tests for rclm.hooks._analytics."""
from rclm._models import FileDiff, ToolCall
from rclm.hooks._analytics import (
    aggregate_compression_savings,
    compute_session_analytics,
    estimate_tokens,
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
            "t1", "Bash", {"command": "ls"}, "output", "2024-01-01T00:00:00Z",
            input_token_estimate=10, output_token_estimate=20,
        )
        result = compute_session_analytics([tc], [])
        assert result["tool_token_stats"]["Bash"]["input_tokens"] == 10
        assert result["tool_token_stats"]["Bash"]["output_tokens"] == 20


# ---------------------------------------------------------------------------
# aggregate_compression_savings
# ---------------------------------------------------------------------------

class TestAggregateCompressionSavings:
    def test_no_savings_events(self):
        events = [{"event_type": "SessionStart"}, {"event_type": "PreToolUse"}]
        assert aggregate_compression_savings(events) is None

    def test_aggregates_savings(self):
        events = [
            {"event_type": "SessionStart"},
            {
                "event_type": "CompressionSaving",
                "original_chars": 1000,
                "compressed_chars": 200,
            },
            {
                "event_type": "CompressionSaving",
                "original_chars": 500,
                "compressed_chars": 100,
            },
        ]
        result = aggregate_compression_savings(events)
        assert result is not None
        assert result["total_original_chars"] == 1500
        assert result["total_compressed_chars"] == 300
        assert result["savings_pct"] == 80.0
        assert result["command_count"] == 2

    def test_empty_events(self):
        assert aggregate_compression_savings([]) is None
