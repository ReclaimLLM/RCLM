from rclm.hooks.result_delta import process_delta


def test_repeated_command_returns_only_state_delta() -> None:
    first = "Created At: one\n" + "\n".join(f"test_{i} PASSED" for i in range(100))
    second = "Created At: two\n" + "\n".join(
        [*(f"test_{i} PASSED" for i in range(99)), "test_99 FAILED: assertion"]
    )
    tool_input = {"CommandLine": "pytest -q", "Cwd": "/repo"}

    initial = process_delta("run_command", tool_input, first, {})
    changed = process_delta("run_command", tool_input, second, initial.state)

    assert initial.replacement is None
    assert changed.replacement is not None
    assert "+ test_99 FAILED: assertion" in changed.replacement
    assert "- test_99 PASSED" in changed.replacement


def test_failed_command_is_never_rewritten() -> None:
    text = "The command exited with code 1.\n" + "failure\n" * 200
    first = process_delta("run_command", {"CommandLine": "pytest"}, text, {})
    second = process_delta("run_command", {"CommandLine": "pytest"}, text, first.state)
    assert second.replacement is None
