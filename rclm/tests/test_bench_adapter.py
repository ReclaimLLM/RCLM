from rclm.bench_adapter import RCLMBenchAdapter


def test_bench_adapter_uses_shipped_shell_compaction() -> None:
    adapter = RCLMBenchAdapter({"mechanisms": ["shell_compaction"]})
    output = "\n".join([f"line {index}" for index in range(200)])

    result = adapter.transform(
        {
            "sequence": 1,
            "tool_name": "exec_command",
            "tool_input": {"cmd": "cat build.log"},
            "tool_result": output,
            "is_error": False,
        }
    )

    assert result["status"] == "applied"
    assert result["mechanism"]
    assert len(result["transformed_result"]) < len(output)


def test_bench_adapter_reports_mcp_result_as_ineligible() -> None:
    adapter = RCLMBenchAdapter({"mechanisms": ["shell_compaction"]})

    result = adapter.transform(
        {
            "sequence": 1,
            "tool_name": "mcp__example",
            "tool_input": {},
            "tool_result": "large result",
            "is_error": False,
        }
    )

    assert result["status"] == "ineligible"
    assert result["transformed_result"] == "large result"
