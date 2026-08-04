"""Tests for the provider-neutral tool-result transform core."""

import json

import pytest

from rclm.hooks.tool_result_transform import (
    analytics_events,
    compact_tool_result,
    extract_text_envelope,
)


def _large_output() -> str:
    return "".join(f"unique line {line}\n" for line in range(100))


def test_compacts_recognized_shell_output():
    decision = compact_tool_result(
        "exec_command",
        {"command": "cat build.log"},
        _large_output(),
    )

    assert decision is not None
    assert decision.mechanism == "H3_exec_compaction"
    assert "40 lines omitted" in decision.compressed_text
    assert decision.raw_chars > decision.compressed_chars


def test_preserves_structured_result_metadata():
    response = {
        "stdout": _large_output(),
        "exit_code": 0,
        "metadata": {"duration_ms": 12},
    }
    decision = compact_tool_result("Bash", {"command": "cat build.log"}, response)

    assert decision is not None
    assert decision.structured_replacement["metadata"] == {"duration_ms": 12}
    assert decision.structured_replacement["exit_code"] == 0
    assert "lines omitted" in decision.structured_replacement["stdout"]


def test_preserves_outer_json_string_envelope():
    response = json.dumps(
        {
            "output": _large_output(),
            "exitCode": 0,
            "metadata": {"source": "shell"},
        }
    )
    decision = compact_tool_result("shell", {"command": "cat build.log"}, response)

    assert decision is not None
    assert isinstance(decision.wire_replacement, str)
    rebuilt = json.loads(decision.wire_replacement)
    assert rebuilt["metadata"] == {"source": "shell"}
    assert "lines omitted" in rebuilt["output"]


@pytest.mark.parametrize(
    "tool_name,tool_input,response",
    [
        ("Bash", {"command": "cat build.log"}, {"stdout": _large_output(), "exit_code": 1}),
        ("Bash", {"command": "cat build.log"}, {"content": _large_output(), "isError": True}),
        ("Bash", {"command": "echo hello"}, _large_output()),
        ("MCP:filesystem:read", {"command": "cat build.log"}, _large_output()),
        ("Bash", {"command": "cat image.txt"}, "data:image/png;base64," + "a" * 10_000),
    ],
)
def test_unsafe_or_unknown_results_pass_through(tool_name, tool_input, response):
    assert compact_tool_result(tool_name, tool_input, response) is None


def test_multiple_mcp_text_blocks_are_ambiguous():
    response = {
        "content": [
            {"type": "text", "text": _large_output()},
            {"type": "text", "text": "independent second block"},
        ]
    }
    assert extract_text_envelope(response) is None


def test_oversized_content_list_is_not_scanned_or_changed():
    response = {"content": [{"type": "text", "text": "line"}] * 65}
    assert compact_tool_result("Bash", {"command": "cat build.log"}, response) is None


def test_analytics_include_char_counts_and_explicit_estimator():
    decision = compact_tool_result("Bash", {"command": "cat build.log"}, _large_output())
    assert decision is not None

    saving, transformation = analytics_events(
        decision,
        tool_use_id="call-1",
        applied=True,
    )

    assert saving["mechanism"] == "H3_exec_compaction"
    assert transformation["raw_chars"] == len(decision.original_text)
    assert transformation["compressed_chars"] == len(decision.compressed_text)
    assert transformation["token_estimator"] == "chars_div_4_v1"
