"""Tests for shell output filters (ls/find listing, ANSI strip, generic fallback)."""

from rclm.compress.filters.shell import (
    collapse_repeats,
    filter_generic,
    filter_shell,
    strip_ansi,
)

# ---------------------------------------------------------------------------
# strip_ansi
# ---------------------------------------------------------------------------


class TestStripAnsi:
    def test_strips_color_codes(self):
        text = "\x1b[32mPASS\x1b[0m tests/test_foo.py"
        assert strip_ansi(text) == "PASS tests/test_foo.py"

    def test_strips_cursor_movement(self):
        text = "loading\x1b[2K\x1b[1Gdone"
        assert strip_ansi(text) == "loadingdone"

    def test_strips_osc_title_sequence(self):
        text = "\x1b]0;my terminal title\x07prompt$ "
        assert strip_ansi(text) == "prompt$ "

    def test_plain_text_unchanged(self):
        text = "no escape codes here\nsecond line"
        assert strip_ansi(text) == text


# ---------------------------------------------------------------------------
# collapse_repeats
# ---------------------------------------------------------------------------


class TestCollapseRepeats:
    def test_collapses_long_runs(self):
        lines = ["Downloading..."] * 10
        result = collapse_repeats(lines)
        assert result == ["Downloading...", "... (repeated 9 more times)"]

    def test_short_runs_kept_verbatim(self):
        lines = ["a", "a", "b"]
        assert collapse_repeats(lines) == ["a", "a", "b"]

    def test_mixed_content(self):
        lines = ["start", *(["dot"] * 5), "end"]
        result = collapse_repeats(lines)
        assert result == ["start", "dot", "... (repeated 4 more times)", "end"]

    def test_empty_input(self):
        assert collapse_repeats([]) == []


# ---------------------------------------------------------------------------
# filter_generic
# ---------------------------------------------------------------------------


class TestFilterGeneric:
    def test_small_output_returns_none(self):
        output = "\n".join(f"line {i}" for i in range(10))
        assert filter_generic(output) is None

    def test_caps_large_unrepeated_output(self):
        output = "\n".join(f"unique line {i}" for i in range(200))
        result = filter_generic(output)
        assert result is not None
        assert "lines omitted" in result
        assert "unique line 0" in result  # head preserved
        assert "unique line 199" in result  # tail preserved
        assert len(result.splitlines()) < 200

    def test_collapses_repeats_before_capping(self):
        output = "\n".join(["progress..."] * 200)
        result = filter_generic(output)
        assert result is not None
        assert "repeated" in result
        assert len(result.splitlines()) < 10


# ---------------------------------------------------------------------------
# filter_shell (ls/find)
# ---------------------------------------------------------------------------


class TestFilterShellListing:
    def test_non_listing_command_returns_none(self):
        assert filter_shell("echo hi", "hi\n") is None

    def test_small_listing_returned_as_is(self):
        output = "a.py\nb.py\n"
        assert filter_shell("ls", output) == output

    def test_empty_listing(self):
        assert filter_shell("ls", "") == "(empty)"

    def test_groups_large_listing_by_directory(self):
        lines = [f"src/file{i}.py" for i in range(40)]
        output = "\n".join(lines)
        result = filter_shell("find . -name '*.py'", output)
        assert result is not None
        assert "src/ (40 files)" in result
        assert "more" in result
