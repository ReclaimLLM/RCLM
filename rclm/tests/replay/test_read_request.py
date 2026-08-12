"""Tests for rclm.replay.read_request: capture-derived H1 ReadRequest building."""

from __future__ import annotations

from rclm.hooks.read_cache import FileMetadata, ReadRequest, process_read, serialize_read_request
from rclm.replay.read_request import (
    build_native_read_request,
    build_read_request,
    build_shell_read_request,
    split_native_read_block,
)


def _numbered(start: int, count: int) -> str:
    return "\n".join(f"{n:>6}→line {n}" for n in range(start, start + count)) + "\n"


class TestSplitNativeReadBlock:
    def test_pure_numbered_block_has_no_trailing(self):
        content = _numbered(1, 5)
        split = split_native_read_block(content)
        assert split is not None
        block, trailing = split
        assert block == content
        assert trailing == ""

    def test_system_reminder_suffix_is_split_off(self):
        content = _numbered(1, 5) + "<system-reminder>note</system-reminder>\n"
        split = split_native_read_block(content)
        assert split is not None
        block, trailing = split
        assert block == _numbered(1, 5)
        assert "system-reminder" in trailing

    def test_non_numbered_content_returns_none(self):
        assert split_native_read_block("plain text, no line numbers") is None


class TestBuildNativeReadRequest:
    def test_offset_derived_from_first_line_number_not_tool_input(self):
        content = _numbered(38, 10)  # lines 38..47
        built = build_native_read_request({"file_path": "/a.py"}, content)
        assert built is not None
        request, trailing = built
        assert request.start_line == 38
        assert request.end_line == 47
        assert trailing == ""

    def test_missing_file_path_is_unresolvable(self):
        assert build_native_read_request({}, _numbered(1, 5)) is None

    def test_non_numbered_content_is_unresolvable(self):
        assert build_native_read_request({"file_path": "/a.py"}, "no line numbers here") is None


class TestBuildShellReadRequest:
    def test_sed_range_parses(self):
        content = "\n".join(f"line {i}" for i in range(10, 21)) + "\n"  # 11 lines
        built = build_shell_read_request("sed -n '10,20p' file.txt", content)
        assert built is not None
        request, trailing = built
        assert request.start_line == 10
        assert request.end_line == 20
        assert trailing == ""

    def test_tail_style_is_unsupported(self):
        assert build_shell_read_request("tail -n 5 file.txt", "a\nb\nc\nd\ne\n") is None

    def test_mismatched_output_line_count_is_unresolvable(self):
        content = "only one line\n"
        assert build_shell_read_request("sed -n '10,20p' file.txt", content) is None


class TestDispatchAndInterop:
    def test_dispatches_native_for_read_tool(self):
        content = _numbered(1, 10)
        built = build_read_request("Read", {"file_path": "/a.py"}, content)
        assert built is not None

    def test_dispatches_shell_for_bash_tool(self):
        content = "\n".join(f"l{i}" for i in range(1, 6)) + "\n"
        built = build_read_request("Bash", {"command": "sed -n '1,5p' f.txt"}, content)
        assert built is not None

    def test_unknown_tool_returns_none(self):
        assert build_read_request("Edit", {"file_path": "/a.py"}, "x") is None

    def test_prefers_valid_capture_metadata_for_plain_native_output(self):
        request = ReadRequest(
            FileMetadata("/past/a.py", "a.py", "a" * 64, 2, 12),
            1,
            2,
            "native",
        )

        built = build_read_request(
            "Read",
            {"file_path": "/past/a.py"},
            "plain first line\nplain second line\n",
            serialize_read_request(request),
        )

        assert built is not None
        captured, trailing = built
        assert captured == request
        assert trailing == ""

    def test_built_request_feeds_process_read_cleanly(self):
        content = _numbered(1, 20)
        built = build_native_read_request({"file_path": "/a.py"}, content)
        assert built is not None
        request, trailing = built
        block = content[: len(content) - len(trailing)] if trailing else content
        decision = process_read(request, block, {}, turn=1)
        assert decision.reliable
