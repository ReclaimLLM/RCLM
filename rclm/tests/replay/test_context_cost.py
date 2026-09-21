from rclm.replay.context_cost import replay_recallable_context


def test_superseded_result_becomes_small_recallable_stub() -> None:
    large = "result line\n" * 200
    blob = {
        "model": "test-model",
        "tool_calls": [
            {
                "tool_use_id": "1",
                "tool_name": "grep_search",
                "tool_input": {"Query": "needle", "SearchPath": "/repo"},
                "tool_result": large,
            },
            {"tool_name": "unrelated", "tool_input": {}, "tool_result": "small"},
            {
                "tool_use_id": "2",
                "tool_name": "grep_search",
                "tool_input": {"Query": "needle", "SearchPath": "/repo"},
                "tool_result": large + "new\n",
            },
            {"tool_name": "unrelated", "tool_input": {}, "tool_result": "small"},
        ],
    }

    result = replay_recallable_context(blob)

    assert result.evicted_results == 1
    assert result.tokens_removed > 0
    assert result.optimized_tokens < result.baseline_tokens


def test_unique_results_are_not_claimed_as_evictions() -> None:
    blob = {
        "tool_calls": [
            {
                "tool_name": "view_file",
                "tool_input": {"AbsolutePath": "/repo/a.py"},
                "tool_result": "x" * 1000,
            },
            {"tool_name": "unrelated", "tool_input": {}, "tool_result": "small"},
        ]
    }
    result = replay_recallable_context(blob)
    assert result.evicted_results == 0
    assert result.tokens_removed == 0
