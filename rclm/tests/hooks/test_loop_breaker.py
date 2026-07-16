"""Tests for rclm.hooks.loop_breaker (PreToolUse repeat/failure detection)."""

from rclm.hooks.loop_breaker import analyze

# ---------------------------------------------------------------------------
# No signal
# ---------------------------------------------------------------------------


def test_no_history_returns_none():
    assert analyze("Bash", {"command": "ls"}, []) is None


def test_unrelated_history_returns_none():
    events = [
        {"event_type": "PreToolUse", "tool_name": "Read", "tool_input": {"file_path": "a.py"}},
        {"event_type": "PostToolUse", "tool_name": "Read", "tool_input": {"file_path": "a.py"}},
    ]
    assert analyze("Bash", {"command": "ls"}, events) is None


# ---------------------------------------------------------------------------
# Repeated identical calls
# ---------------------------------------------------------------------------


def _pre(tool_name: str, tool_input: dict) -> dict:
    return {"event_type": "PreToolUse", "tool_name": tool_name, "tool_input": tool_input}


class TestRepeatedCalls:
    def test_one_prior_identical_call_is_below_threshold(self):
        events = [_pre("Bash", {"command": "npm test"})]
        assert analyze("Bash", {"command": "npm test"}, events) is None

    def test_two_prior_identical_calls_warns(self):
        events = [_pre("Bash", {"command": "npm test"})] * 2
        result = analyze("Bash", {"command": "npm test"}, events)
        assert result is not None
        assert "additionalContext" in result
        assert "retry loop" in result["additionalContext"]

    def test_four_prior_identical_calls_asks(self):
        events = [_pre("Bash", {"command": "npm test"})] * 4
        result = analyze("Bash", {"command": "npm test"}, events)
        assert result == {
            "permissionDecision": "ask",
            "permissionDecisionReason": (
                "4 consecutive identical Bash calls with the same input. "
                "This looks like a retry loop — check whether the input actually needs to change."
            ),
        }

    def test_non_consecutive_repeats_dont_count(self):
        events = [
            _pre("Bash", {"command": "npm test"}),
            _pre("Bash", {"command": "ls"}),
            _pre("Bash", {"command": "npm test"}),
        ]
        assert analyze("Bash", {"command": "npm test"}, events) is None

    def test_different_input_breaks_streak(self):
        events = [
            _pre("Bash", {"command": "npm test -- foo"}),
            _pre("Bash", {"command": "npm test -- foo"}),
        ]
        assert analyze("Bash", {"command": "npm test -- bar"}, events) is None


# ---------------------------------------------------------------------------
# Consecutive failures on the same target
# ---------------------------------------------------------------------------


def _failure(tool_name: str, tool_input: dict) -> dict:
    return {"event_type": "ToolFailure", "tool_name": tool_name, "tool_input": tool_input}


def _success(tool_name: str, tool_input: dict) -> dict:
    return {"event_type": "PostToolUse", "tool_name": tool_name, "tool_input": tool_input}


_EDIT_INPUT = {"file_path": "src/app.ts", "old_string": "x", "new_string": "y"}


class TestConsecutiveFailures:
    def test_two_failures_warns(self):
        events = [_failure("Edit", _EDIT_INPUT), _failure("Edit", _EDIT_INPUT)]
        result = analyze("Edit", _EDIT_INPUT, events)
        assert result is not None
        assert "additionalContext" in result
        assert "2 consecutive failures" in result["additionalContext"]

    def test_four_failures_asks(self):
        events = [_failure("Edit", _EDIT_INPUT)] * 4
        result = analyze("Edit", _EDIT_INPUT, events)
        assert result is not None
        assert result["permissionDecision"] == "ask"
        assert "src/app.ts" in result["permissionDecisionReason"]

    def test_success_resets_streak(self):
        events = [
            _failure("Edit", _EDIT_INPUT),
            _failure("Edit", _EDIT_INPUT),
            _success("Edit", {"file_path": "src/app.ts", "old_string": "a", "new_string": "b"}),
        ]
        assert analyze("Edit", _EDIT_INPUT, events) is None

    def test_failures_on_different_file_dont_count(self):
        events = [
            _failure("Edit", {"file_path": "other.ts", "old_string": "a", "new_string": "b"}),
            _failure("Edit", {"file_path": "other.ts", "old_string": "a", "new_string": "b"}),
        ]
        assert analyze("Edit", _EDIT_INPUT, events) is None

    def test_interleaved_unrelated_tool_does_not_break_streak(self):
        """A Read between two failing Edits on the same file (the Gemini re-read pattern)
        should not reset the failure streak."""
        events = [
            _failure("Edit", _EDIT_INPUT),
            _pre("Read", {"file_path": "src/app.ts"}),
            _success("Read", {"file_path": "src/app.ts"}),
            _failure("Edit", _EDIT_INPUT),
        ]
        result = analyze("Edit", _EDIT_INPUT, events)
        assert result is not None
        assert "2 consecutive failures" in result["additionalContext"]

    def test_bash_target_is_command(self):
        cmd = {"command": "pytest tests/test_foo.py"}
        events = [_failure("Bash", cmd)] * 2
        result = analyze("Bash", cmd, events)
        assert result is not None
        assert "pytest tests/test_foo.py" in result["additionalContext"]

    def test_grep_has_no_failure_target(self):
        """Grep isn't a file/command-target tool, so failure-streak tracking doesn't apply
        (repeat-call detection still would, covered separately)."""
        events = [_failure("Grep", {"pattern": "foo"})] * 4
        assert analyze("Grep", {"pattern": "foo"}, events) is None


# ---------------------------------------------------------------------------
# Whichever signal is stronger wins the reason text
# ---------------------------------------------------------------------------


def test_failure_reason_used_when_failures_exceed_repeats():
    events = [_failure("Edit", _EDIT_INPUT)] * 4
    result = analyze("Edit", _EDIT_INPUT, events)
    assert "consecutive failures" in result["permissionDecisionReason"]
