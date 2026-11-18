"""Tests for the residual minifiers (repeated-line collapse, blob placeholders)."""

import base64

from rclm.compress.filters.minify import (
    BLOB_MECHANISM,
    REPEATED_LINE_MECHANISM,
    collapse_repeated_lines,
    minify,
    replace_blobs,
)
from rclm.hooks.tool_result_transform import compact_tool_result


def _retry_log() -> str:
    lines = []
    for attempt in range(1, 41):
        lines.append(f"WARN retry attempt {attempt} for job sync-users after {attempt * 50}ms")
        lines.append(f"step {attempt} done")
    lines.append("ERROR job sync-users gave up: connection refused")
    return "\n".join(lines)


def test_collapse_keeps_first_occurrence_and_distinct_lines():
    collapsed = collapse_repeated_lines(_retry_log())

    assert collapsed is not None
    assert collapsed.count("WARN retry attempt") == 1
    assert "+39 similar lines differing only in numbers" in collapsed
    assert "ERROR job sync-users gave up: connection refused" in collapsed
    # "step N done" is under the minimum length, so every copy is kept.
    assert collapsed.count("done") == 40


def test_collapse_keeps_lines_that_differ_in_words():
    report = "\n".join(
        f"[{i:04d}] check_component_{i} ............ {'FAIL' if i == 137 else 'ok'} (latency {i % 17} ms)"
        for i in range(250)
    )
    collapsed = collapse_repeated_lines(report)

    assert collapsed is not None
    assert "check_component_137 ............ FAIL" in collapsed
    assert collapsed.count(" ok (latency") == 1


def test_collapse_never_folds_tables_or_key_value_data():
    table = "\n".join(
        f"4000 | 0 | {threshold} | {cost:.2f} | {pct:+.1f}% | {n} | 180.8"
        for threshold, cost, pct, n in [
            (200_000 - i * 1000, 87.24 + i, -42.7 + i, i) for i in range(40)
        ]
    )
    csv_rows = "\n".join(
        f"session-{i},claude-sonnet,{i * 997},{i * 13},0.{i:02d}" for i in range(40)
    )
    json_lines = "\n".join(f'   "tokens_saved_estimate": {i * 37},' for i in range(40))

    for text in (table, csv_rows, json_lines):
        assert collapse_repeated_lines(text) is None


def test_collapse_ignores_small_inputs():
    assert collapse_repeated_lines("line 1 of a short output here\n" * 2) is None


def test_blob_placeholder_replaces_base64_and_long_hex_only():
    payload = base64.b64encode(bytes(range(256)) * 4).decode()
    sha256 = "ab" * 32
    text = f'response: {{"image": "data:image/png;base64,{payload}", "digest": "{sha256}"}}'

    replaced = replace_blobs(text)

    assert replaced is not None
    assert payload not in replaced
    assert "[rclm: base64 blob," in replaced
    # sha256 digests are quoted by agents and must survive.
    assert sha256 in replaced


def test_minify_reports_each_step_separately():
    payload = base64.b64encode(b"x" * 600).decode()
    text = _retry_log() + "\n" + payload

    steps = minify(text)

    assert [mechanism for mechanism, _, _ in steps] == [BLOB_MECHANISM, REPEATED_LINE_MECHANISM]
    assert steps[0][1] == text
    assert steps[1][1] == steps[0][2]


def test_compaction_applies_minifiers_to_unrecognized_shell_output():
    decision = compact_tool_result(
        "Bash",
        {"command": "./scripts/sync_users.sh"},
        {"stdout": _retry_log(), "stderr": "", "interrupted": False},
    )

    assert decision is not None
    assert REPEATED_LINE_MECHANISM in [step.mechanism for step in decision.savings_steps]
    assert decision.structured_replacement["stdout"] == decision.compressed_text


def test_compaction_never_minifies_file_content():
    content = _retry_log()
    minifiers = {REPEATED_LINE_MECHANISM, BLOB_MECHANISM}

    # `cat` keeps its existing command-aware filter, but no lossy minifier.
    shell_print = compact_tool_result("Bash", {"command": "cat sync.log"}, content)
    assert shell_print is None or not minifiers & {s.mechanism for s in shell_print.savings_steps}
    assert compact_tool_result("Read", {"file_path": "/repo/sync.log"}, content) is None


def test_compaction_never_minifies_failed_commands():
    response = {"stdout": _retry_log(), "stderr": "", "exit_code": 1}

    assert compact_tool_result("Bash", {"command": "./scripts/sync_users.sh"}, response) is None
