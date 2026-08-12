"""Direct tests for conservative PostToolUse result dedupe."""

from rclm.hooks.dedupe import maybe_dedupe, normalize_result


def test_normalization_removes_only_expected_noise():
    result = normalize_result(
        "\x1b[31mPID=1234\x1b[0m at 2026-07-23T10:14:22Z after 0.42s\n/repo/src/app.py   \n",
        cwd="/repo",
    )
    assert result == "pid=<pid> at <timestamp> after <duration>\n./src/app.py"


def test_semantically_different_test_counts_do_not_collide():
    state = {}
    first = "=" * 200 + "\n3 failed, 10 passed in 0.42s\n"
    second = "=" * 200 + "\n4 failed, 10 passed in 0.42s\n"
    _, state, _ = maybe_dedupe(first, state, tool_name="Bash", turn=1)
    replacement, _, _ = maybe_dedupe(second, state, tool_name="Bash", turn=2)
    assert replacement is None


def test_identical_result_returns_pointer_and_lru_is_bounded():
    text = "result line\n" * 100
    _, state, _ = maybe_dedupe(text, {}, tool_name="git status", turn=14, max_entries=1)
    replacement, state, match = maybe_dedupe(
        text, state, tool_name="git status", turn=15, max_entries=1
    )
    assert "git status" in replacement
    assert "turn 14" in replacement
    assert match is not None
    assert len(state) == 1


def test_identical_result_pointer_is_deterministic_for_replay():
    text = "result line\n" * 100
    _, state, _ = maybe_dedupe(text, {}, tool_name="Read", turn=1)

    replacements = {maybe_dedupe(text, state, tool_name="Read", turn=2)[0] for _ in range(10)}

    assert len(replacements) == 1


def test_error_dict_is_never_deduplicated():
    replacement, state, match = maybe_dedupe(
        {"is_error": True, "content": "x" * 600}, {}, tool_name="Bash", turn=1
    )
    assert replacement is None
    assert state == {}
    assert match is None
