"""Tests for search (rg/grep) output filters."""

from rclm.compress.filters.search import filter_search

# ---------------------------------------------------------------------------
# Non-matching / pass-through cases
# ---------------------------------------------------------------------------


def test_unknown_command_returns_none():
    assert filter_search("cat file.txt", "some output\n" * 40) is None


def test_empty_output_returns_none():
    assert filter_search("rg foo", "") is None


def test_small_output_returns_none():
    output = "\n".join(f"file{i}.py:{i}:match" for i in range(10))
    assert filter_search("rg foo", output) is None


def test_files_with_matches_flag_untouched():
    output = "\n".join(f"file{i}.py" for i in range(40))
    assert filter_search("rg -l foo", output) is None


def test_count_flag_untouched():
    output = "\n".join(f"file{i}.py:3" for i in range(40))
    assert filter_search("rg --count foo", output) is None


# ---------------------------------------------------------------------------
# Flat `path:line:content` shaping (grep -rn / rg --no-heading)
# ---------------------------------------------------------------------------


class TestFlatFormat:
    def test_groups_by_file_with_counts_and_first_match(self):
        lines = []
        for i in range(5):
            for j in range(8):
                lines.append(f"src/file{i}.py:{j + 1}:match number {j} in file {i}")
        output = "\n".join(lines)

        result = filter_search("rg -n TODO .", output)

        assert result is not None
        assert "40 matches across 5 files" in result
        assert "src/file0.py (8 matches)" in result
        assert "1: match number 0 in file 0" in result
        # Only the first match per file is retained
        assert "match number 1 in file 0" not in result

    def test_caps_number_of_files_shown(self):
        lines = [f"src/file{i}.py:1:match" for i in range(40)]
        output = "\n".join(lines)

        result = filter_search("grep -rn TODO .", output)

        assert result is not None
        assert "40 matches across 40 files" in result
        assert "more files with matches" in result

    def test_significant_compression(self):
        lines = []
        for i in range(30):
            for j in range(10):
                lines.append(f"src/file{i}.py:{j}:{'x' * 80}")
        output = "\n".join(lines)

        result = filter_search("rg -n pattern .", output)

        assert result is not None
        assert len(result) < len(output)


# ---------------------------------------------------------------------------
# Ripgrep headed format (bare path, then `line:content` blocks)
# ---------------------------------------------------------------------------


class TestHeadingFormat:
    def test_groups_by_file(self):
        blocks = []
        for i in range(5):
            blocks.append(f"src/file{i}.py")
            for j in range(8):
                blocks.append(f"{j + 1}:match {j} in {i}")
            blocks.append("")
        output = "\n".join(blocks)

        result = filter_search("rg TODO .", output)

        assert result is not None
        assert "40 matches across 5 files" in result
        assert "src/file0.py (8 matches)" in result


# ---------------------------------------------------------------------------
# Single-file `line:content` (no file grouping)
# ---------------------------------------------------------------------------


class TestSingleFileFormat:
    def test_caps_matches(self):
        lines = [f"{i}:some matching line {i}" for i in range(1, 41)]
        output = "\n".join(lines)

        result = filter_search("grep -n pattern single_file.py", output)

        assert result is not None
        assert "40 matching lines" in result
        assert "more matches" in result
