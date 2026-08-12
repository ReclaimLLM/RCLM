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


def test_losslessly_folds_chained_search_output_when_command_parser_cannot() -> None:
    output = "".join(f"src/api/handler.py:{line}:match {line}\n" for line in range(1, 41))

    decision = compact_tool_result(
        "Bash",
        {"command": "cd src && /usr/bin/grep -rn match ."},
        output,
    )

    assert decision is not None
    assert decision.mechanism == "lossless_search_path_folding"
    assert decision.compressed_text.startswith("src/api/handler.py\n1:match 1\n")
    assert "40:match 40" in decision.compressed_text


def test_lossless_path_fallback_preserves_structured_metadata() -> None:
    response = {
        "stdout": "src/api/a.py\nsrc/api/b.py\nsrc/web/c.ts\n",
        "exit_code": 0,
        "metadata": {"duration_ms": 12},
    }

    decision = compact_tool_result("exec_command", {"cmd": "cd repo && find src"}, response)

    assert decision is not None
    assert decision.mechanism == "lossless_search_path_folding"
    assert decision.structured_replacement["metadata"] == {"duration_ms": 12}
    assert decision.structured_replacement["stdout"] == ("src/api/\na.py\nb.py\nsrc/web/\nc.ts\n")


def test_lossless_fold_runs_after_existing_exec_compaction() -> None:
    output = "".join(f"src/api/handler.py:{line}:match {line}\n" for line in range(1, 101))

    decision = compact_tool_result("Bash", {"command": "cat build.log"}, output)

    assert decision is not None
    assert decision.mechanism == "H3_exec_compaction"
    assert [step.mechanism for step in decision.savings_steps] == [
        "H3_exec_compaction",
        "lossless_search_path_folding",
    ]
    assert decision.compressed_text.startswith("src/api/handler.py\n1:match 1\n")
    assert "lines omitted" in decision.compressed_text


def test_analytics_split_combined_transform_savings_without_changing_total() -> None:
    output = "".join(f"src/api/handler.py:{line}:match {line}\n" for line in range(1, 101))
    decision = compact_tool_result("Bash", {"command": "cat build.log"}, output)
    assert decision is not None

    *saving_events, transformation = analytics_events(
        decision,
        tool_use_id="call-combined",
        applied=True,
    )

    assert [event["mechanism"] for event in saving_events] == [
        "H3_exec_compaction",
        "lossless_search_path_folding",
    ]
    assert (
        sum(event["tokens_saved_estimate"] for event in saving_events)
        == (transformation["tokens_saved_estimate"])
    )


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


def test_top_level_single_text_block_preserves_list_envelope():
    response = [{"type": "text", "text": _large_output()}, {"type": "resource", "uri": "x"}]

    envelope = extract_text_envelope(response)

    assert envelope is not None
    wire, structured, _model_text = envelope.replace("short")
    assert wire == structured
    assert structured[0] == {"type": "text", "text": "short"}
    assert structured[1] == {"type": "resource", "uri": "x"}


def test_top_level_multiple_text_blocks_are_ambiguous():
    response = [
        {"type": "text", "text": _large_output()},
        {"type": "text", "text": "independent second block"},
    ]
    assert extract_text_envelope(response) is None


def test_compacts_large_playwright_snapshot_in_top_level_text_block():
    snapshot = "\n".join(f"- generic [ref=e{line}]: item {line}" for line in range(100))

    decision = compact_tool_result(
        "mcp__playwright__browser_snapshot",
        {},
        [{"type": "text", "text": snapshot}],
    )

    assert decision is not None
    assert decision.mechanism == "H3_exec_compaction"
    assert "40 lines omitted" in decision.compressed_text
    assert decision.structured_replacement[0]["type"] == "text"


def test_compacts_large_read_only_supabase_result_with_explicit_row_marker():
    rows = [{"id": index, "value": "x" * 80} for index in range(50)]
    tag = "untrusted-data-1234"
    result_text = (
        "Below is the result of the SQL query.\n\n"
        f"<{tag}>\n{json.dumps(rows, separators=(',', ':'))}\n</{tag}>\n"
    )
    response = [{"type": "text", "text": json.dumps({"result": result_text})}]

    decision = compact_tool_result(
        "mcp__supabase_local__execute_sql",
        {"query": "SELECT id, value FROM large_table"},
        response,
    )

    assert decision is not None
    assert decision.mechanism == "H3_exec_compaction"
    assert "_rclm_omitted_rows" in decision.compressed_text
    assert decision.raw_chars > decision.compressed_chars


def test_supabase_mutation_and_unknown_sql_envelopes_pass_through():
    rows = [{"id": index} for index in range(50)]
    tag = "untrusted-data-1234"
    text = json.dumps({"result": f"<{tag}>\n{json.dumps(rows)}\n</{tag}>"})
    response = [{"type": "text", "text": text}]

    assert (
        compact_tool_result(
            "mcp__supabase_local__execute_sql",
            {"query": "DELETE FROM large_table RETURNING id"},
            response,
        )
        is None
    )
    assert compact_tool_result("mcp__other__execute_sql", {"query": "SELECT 1"}, response) is None


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
