"""Range-cache parser, interval-state, and fail-open tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rclm.hooks import read_cache

FIXTURES = Path(__file__).parents[1] / "fixtures" / "read_commands.json"


def _long_lines(start: int, end: int, marker: str = "line") -> str:
    return "".join(f"{marker}-{line}-" + ("x" * 100) + "\n" for line in range(start, end + 1))


def _write_lines(path: Path, count: int, marker: str = "line") -> None:
    path.write_text(_long_lines(1, count, marker), encoding="utf-8")


def _request(tmp_path: Path, command: str) -> read_cache.ReadRequest:
    request = read_cache.parse_shell_read(command, cwd=str(tmp_path), shell="posix")
    assert request is not None
    return request


def test_real_captured_posix_commands_match_expected_syntax() -> None:
    fixture = json.loads(FIXTURES.read_text(encoding="utf-8"))
    assert "production session_tool_calls" in fixture["source"]
    for case in fixture["commands"]:
        parsed = read_cache._parse_posix(case["command"])
        assert (parsed is not None) is case["recognized"], case["command"]
        if parsed is not None:
            path, start, end, style = parsed
            assert path == case["path"]
            assert start == case["start"]
            assert end == case["end"]
            if case.get("tail_count"):
                assert style == "tail"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("sed -n '10,40p' source.py", (10, 40)),
        ("sed -n '50p' source.py", (50, 50)),
        ("head -50 source.py", (1, 50)),
        ("head -n 50 source.py", (1, 50)),
        ("tail -n 30 source.py", (71, 100)),
        ("cat source.py", (1, 100)),
        ("nl source.py", (1, 100)),
        ("nl -ba source.py | sed -n '1,80p'", (1, 80)),
        ("awk 'NR>=10 && NR<=40' source.py", (10, 40)),
    ],
)
def test_parse_supported_posix_ranges(
    tmp_path: Path, command: str, expected: tuple[int, int]
) -> None:
    _write_lines(tmp_path / "source.py", 100)
    request = read_cache.parse_shell_read(command, cwd=str(tmp_path), shell="zsh")
    assert request is not None
    assert (request.start_line, request.end_line) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("Get-Content source.ps1", (1, 100)),
        ("Get-Content -Path source.ps1", (1, 100)),
        ("Get-Content -LiteralPath source.ps1", (1, 100)),
        ("Get-Content -TotalCount 25 source.ps1", (1, 25)),
        ("Get-Content -Tail 20 source.ps1", (81, 100)),
    ],
)
def test_parse_supported_powershell_ranges(
    tmp_path: Path, command: str, expected: tuple[int, int]
) -> None:
    _write_lines(tmp_path / "source.ps1", 100)
    request = read_cache.parse_shell_read(command, cwd=str(tmp_path), shell="pwsh")
    assert request is not None
    assert (request.start_line, request.end_line) == expected


@pytest.mark.parametrize(
    "command",
    [
        "cat *.py",
        "cat a.py b.py",
        "cat a.py > out.txt",
        "cat a.py | grep token",
        "sed -n '1,20p' a.py | grep token",
        "awk '{print $1}' a.py",
        "nl -v 10 a.py",
    ],
)
def test_ambiguous_posix_commands_passthrough(tmp_path: Path, command: str) -> None:
    _write_lines(tmp_path / "a.py", 100)
    assert read_cache.parse_shell_read(command, cwd=str(tmp_path), shell="posix") is None


def test_full_coverage_returns_turn_notice(tmp_path: Path) -> None:
    path = tmp_path / "source.py"
    _write_lines(path, 10)
    request = _request(tmp_path, "cat source.py")
    content = path.read_text(encoding="utf-8")

    first = read_cache.process_read(request, content, {}, turn=4)
    second = read_cache.process_read(request, content, first.state, turn=5)

    assert first.replacement is None
    assert second.cache_hit is True
    assert second.replacement == "[RCLM] Lines 1-10 of source.py unchanged since turn 4.\n"


def test_partial_overlap_keeps_only_uncovered_lines(tmp_path: Path) -> None:
    path = tmp_path / "source.py"
    _write_lines(path, 8)
    first_request = _request(tmp_path, "sed -n '1,4p' source.py")
    second_request = _request(tmp_path, "sed -n '3,8p' source.py")

    first = read_cache.process_read(first_request, _long_lines(1, 4), {}, turn=1)
    second = read_cache.process_read(second_request, _long_lines(3, 8), first.state, turn=2)

    assert second.replacement is not None
    assert "Lines 3-4 of source.py unchanged since turn 1" in second.replacement
    assert "line-5-" in second.replacement
    assert "line-8-" in second.replacement
    assert "line-3-" not in second.replacement


def test_hash_change_invalidates_all_intervals(tmp_path: Path) -> None:
    path = tmp_path / "source.py"
    _write_lines(path, 6, "old")
    first_request = _request(tmp_path, "cat source.py")
    first = read_cache.process_read(first_request, path.read_text(), {}, turn=1)

    _write_lines(path, 6, "new")
    changed_request = _request(tmp_path, "cat source.py")
    changed = read_cache.process_read(changed_request, path.read_text(), first.state, turn=2)

    assert changed.replacement is None
    entry = changed.state["files"][changed_request.path]
    assert entry["spans"] == [{"start": 1, "end": 6, "turn": 2}]


def test_edit_invalidation_forces_fresh_reread(tmp_path: Path) -> None:
    path = tmp_path / "source.py"
    _write_lines(path, 6)
    request = _request(tmp_path, "cat source.py")
    first = read_cache.process_read(request, path.read_text(), {}, turn=1)

    invalidated = read_cache.invalidate_tool_path(
        first.state, "Edit", {"file_path": "source.py"}, cwd=str(tmp_path)
    )
    reread = read_cache.process_read(request, path.read_text(), invalidated, turn=2)

    assert reread.replacement is None
    assert reread.state["files"][request.path]["spans"][0]["turn"] == 2


def test_shadow_mode_measures_without_returning_replacement(tmp_path: Path) -> None:
    path = tmp_path / "source.py"
    _write_lines(path, 10)
    request = _request(tmp_path, "cat source.py")
    content = path.read_text()
    first = read_cache.apply_range_cache(
        request, content, {}, turn=1, tool_use_id="tool-1", shadow=True
    )
    second = read_cache.apply_range_cache(
        request, content, first.state, turn=2, tool_use_id="tool-2", shadow=True
    )

    assert second.replacement is None
    assert [event["event_type"] for event in second.events] == [
        "MechanismSaving",
        "ToolTransformation",
    ]
    assert second.events[0]["measurement_kind"] == "measured"
    assert second.events[0]["applied"] is False


def test_pagination_advances_to_first_unseen_line(tmp_path: Path) -> None:
    path = tmp_path / "large.py"
    _write_lines(path, 600)
    first_request = _request(tmp_path, "sed -n '1,200p' large.py")
    first = read_cache.process_read(first_request, _long_lines(1, 200), {}, turn=1)

    offset, state = read_cache.next_unseen_offset(
        {"file_path": "large.py"}, first.state, cwd=str(tmp_path)
    )

    assert offset == 200
    assert state["files"][first_request.path]["line_count"] == 600


def test_binary_missing_and_malformed_state_fail_open(tmp_path: Path) -> None:
    (tmp_path / "binary.bin").write_bytes(b"text\x00binary")
    assert read_cache.parse_shell_read("cat binary.bin", cwd=str(tmp_path)) is None
    assert read_cache.parse_shell_read("cat missing.py", cwd=str(tmp_path)) is None

    path = tmp_path / "source.py"
    _write_lines(path, 3)
    request = _request(tmp_path, "cat source.py")
    decision = read_cache.process_read(request, path.read_text(), {"files": "bad"}, turn=1)
    assert decision.replacement is None
    assert decision.state["version"] == read_cache.STATE_VERSION


def test_file_and_span_caps_evict_oldest_entries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(read_cache, "MAX_TRACKED_FILES", 2)
    state: dict = {}
    for index in range(3):
        path = tmp_path / f"file-{index}.py"
        _write_lines(path, 2)
        request = _request(tmp_path, f"cat file-{index}.py")
        state = read_cache.process_read(request, path.read_text(), state, turn=index + 1).state
    assert len(state["files"]) == 2
    assert str((tmp_path / "file-0.py").resolve()) not in state["files"]
