"""Tests for eliding read-back of text the model itself just wrote."""

from __future__ import annotations

from rclm.hooks import authored_reads as ar
from rclm.replay.engine import EXPERIMENTAL_MECHANISMS, replay_blob


def _body(count: int, label: str = "row") -> str:
    return "\n".join(f"    {label}_{n} = compute({n})" for n in range(count))


def _arrow(lines: list[str], start: int = 1) -> str:
    return "".join(f"{n:>6}→{text}\n" for n, text in enumerate(lines, start))


def test_authored_text_reads_edit_and_write_shapes() -> None:
    edit = {"file_path": "/a", "new_string": "x", "old_string": "y"}
    assert ar.authored_text(edit) == ("/a", "x")
    assert ar.authored_text({"path": "/a", "content": "body"}) == ("/a", "body")
    assert ar.authored_text({"TargetFile": "/a", "ReplacementContent": "z"}) == ("/a", "z")
    assert ar.authored_text({"file_path": "/a"}) is None
    assert ar.authored_text("not a dict") is None


def test_full_readback_of_own_write_is_elided() -> None:
    text = _body(40)
    state = ar.record_authored(None, "/a.py", text, turn=3)
    block = _arrow(text.splitlines())

    decision = ar.elide_authored_runs(state, "/a.py", block, numbered=True, turn=5)

    assert decision.replacement is not None
    assert decision.elided_lines == 40
    assert "Lines 1-40 of /a.py" in decision.replacement
    assert "turn 3" in decision.replacement
    assert len(decision.replacement) < len(block) // 4


def test_lines_around_the_edit_are_kept_verbatim() -> None:
    authored = _body(12, "new")
    state = ar.record_authored(None, "/a.py", authored, turn=1)
    lines = ["import os", "", *authored.splitlines(), "", "def tail(): ..."]
    block = _arrow(lines)

    decision = ar.elide_authored_runs(state, "/a.py", block, numbered=True, turn=2)

    assert decision.replacement is not None
    assert "     1→import os" in decision.replacement
    assert "def tail(): ..." in decision.replacement
    assert "new_5" not in decision.replacement


def test_short_or_changed_runs_are_never_elided() -> None:
    state = ar.record_authored(None, "/a.py", _body(30), turn=1)
    short = _arrow(_body(4).splitlines())
    changed = _arrow([line.replace("compute", "other") for line in _body(30).splitlines()])

    assert ar.elide_authored_runs(state, "/a.py", short, numbered=True, turn=2).replacement is None
    assert (
        ar.elide_authored_runs(state, "/a.py", changed, numbered=True, turn=2).replacement is None
    )


def test_blank_only_runs_do_not_match() -> None:
    state = ar.record_authored(None, "/a.py", "\n" * 20 + "x", turn=1)

    decision = ar.elide_authored_runs(state, "/a.py", "\n" * 20, numbered=False, turn=2)

    assert decision.replacement is None


def test_old_authored_text_is_still_referenced() -> None:
    text = _body(30)
    state = ar.record_authored(None, "/a.py", text, turn=1)
    block = _arrow(text.splitlines())

    decision = ar.elide_authored_runs(state, "/a.py", block, numbered=True, turn=5_000)

    assert decision.replacement is not None


def test_other_paths_and_plain_blocks() -> None:
    text = _body(30)
    state = ar.record_authored(None, "/a.py", text, turn=1)

    other = ar.elide_authored_runs(state, "/b.py", text + "\n", numbered=False, turn=2)
    plain = ar.elide_authored_runs(state, "/a.py", text + "\n", numbered=False, turn=2)

    assert other.replacement is None
    assert plain.replacement is not None and "30 lines of /a.py" in plain.replacement


def test_state_is_bounded() -> None:
    state: dict = {}
    for n in range(ar.MAX_TRACKED_FILES + 10):
        state = ar.record_authored(state, f"/f{n}.py", _body(8), turn=n)
    assert len(state["files"]) == ar.MAX_TRACKED_FILES
    for turn in range(ar.MAX_BLOCKS_PER_FILE + 5):
        state = ar.record_authored(state, "/f_hot.py", _body(8), turn=1000 + turn)
    assert len(state["files"]["/f_hot.py"]) == ar.MAX_BLOCKS_PER_FILE
    oversized = _body(ar.MAX_AUTHORED_LINES + 1)
    assert ar.record_authored(state, "/big.py", oversized, turn=1) == state


def test_replay_counts_authored_reads_only_when_opted_in() -> None:
    text = _body(40)
    write = {"file_path": "/a.py", "content": text}
    blob = {
        "tool_calls": [
            {"tool_name": "Write", "tool_input": write, "tool_result": "ok"},
            {
                "tool_name": "Read",
                "tool_input": {"file_path": "/a.py"},
                "tool_result": _arrow(text.splitlines()),
            },
        ]
    }

    default = replay_blob(blob, mechanisms=("range_cache",))
    opted_in = replay_blob(blob, mechanisms=EXPERIMENTAL_MECHANISMS)

    assert default.tokens_removed == 0
    assert opted_in.tokens_removed > 0
    assert opted_in.shaped_calls[0].mechanism == "authored_read"


def test_replay_elision_carries_a_recall_handle_and_still_saves() -> None:
    import hashlib

    from rclm.replay.engine import authored_recall_stub

    text = _body(40)
    read_result = _arrow(text.splitlines())
    blob = {
        "tool_calls": [
            {
                "tool_name": "Write",
                "tool_input": {"file_path": "/a.py", "content": text},
                "tool_result": "ok",
            },
            {
                "tool_name": "Read",
                "tool_input": {"file_path": "/a.py"},
                "tool_result": read_result,
            },
        ]
    }

    call = replay_blob(blob, mechanisms=EXPERIMENTAL_MECHANISMS).shaped_calls[0]
    stub = authored_recall_stub(read_result, "Read", "/a.py", "toolu_1")

    assert call.compressed_tokens < call.original_tokens
    assert f"rclm://artifact/sha256/{hashlib.sha256(read_result.encode()).hexdigest()}" in stub
    assert f"chars={len(read_result)}" in stub


def test_elision_is_dropped_when_the_handle_would_cost_more_than_it_saves() -> None:
    text = _body(ar.MIN_RUN_LINES)
    tiny = _arrow([f"{line[:6]}" for line in text.splitlines()])
    write = {"file_path": "/a.py", "content": "\n".join(line[:6] for line in text.splitlines())}
    blob = {
        "tool_calls": [
            {"tool_name": "Write", "tool_input": write, "tool_result": "ok"},
            {"tool_name": "Read", "tool_input": {"file_path": "/a.py"}, "tool_result": tiny},
        ]
    }

    assert replay_blob(blob, mechanisms=EXPERIMENTAL_MECHANISMS).tokens_removed == 0
