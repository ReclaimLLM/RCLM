"""Tests for rclm.hooks.read_cache (diff-on-change for repeated reads)."""

from unittest.mock import patch

from rclm.hooks.read_cache import build_delta, maybe_wrap_dump_command, snapshot_event

# ---------------------------------------------------------------------------
# First read of a range — nothing to compare against
# ---------------------------------------------------------------------------


def test_first_read_returns_none():
    assert build_delta("a.py", None, None, "content", []) is None


def test_only_other_files_in_history_returns_none():
    events = [snapshot_event("b.py", None, None, "content", "t1")]
    assert build_delta("a.py", None, None, "content", events) is None


# ---------------------------------------------------------------------------
# Unchanged re-read
# ---------------------------------------------------------------------------


def test_identical_reread_returns_unchanged_notice():
    events = [snapshot_event("a.py", None, None, "def foo(): pass\n", "t1")]
    delta = build_delta("a.py", None, None, "def foo(): pass\n", events)
    assert delta is not None
    assert "Unchanged since the last read of a.py" in delta["updatedToolOutput"]
    assert "t1" in delta["updatedToolOutput"]


def test_uses_most_recent_matching_snapshot():
    events = [
        snapshot_event("a.py", None, None, "old content\n", "t1"),
        snapshot_event("a.py", None, None, "new content\n", "t2"),
    ]
    delta = build_delta("a.py", None, None, "new content\n", events)
    assert "Unchanged" in delta["updatedToolOutput"]
    assert "t2" in delta["updatedToolOutput"]


# ---------------------------------------------------------------------------
# Changed re-read — unified diff
# ---------------------------------------------------------------------------


def test_changed_reread_returns_diff():
    before = "line1\nline2\nline3\n"
    after = "line1\nCHANGED\nline3\n"
    events = [snapshot_event("a.py", None, None, before, "t1")]
    delta = build_delta("a.py", None, None, after, events)
    assert delta is not None
    output = delta["updatedToolOutput"]
    assert "a.py changed since the last read" in output
    assert "-line2" in output
    assert "+CHANGED" in output


def test_large_diff_is_capped():
    before = "\n".join(f"line{i}" for i in range(200)) + "\n"
    after = "\n".join(f"changed{i}" for i in range(200)) + "\n"
    events = [snapshot_event("a.py", None, None, before, "t1")]
    delta = build_delta("a.py", None, None, after, events)
    output = delta["updatedToolOutput"]
    assert "more diff lines omitted" in output
    assert len(output.splitlines()) < 200


# ---------------------------------------------------------------------------
# Different ranges of the same file are tracked independently
# ---------------------------------------------------------------------------


def test_different_offset_limit_not_compared():
    events = [snapshot_event("a.py", 0, 100, "content A", "t1")]
    # Same file, different range -> no history for this specific key.
    assert build_delta("a.py", 100, 200, "content A", events) is None


def test_same_offset_limit_compared():
    events = [snapshot_event("a.py", 0, 100, "content A", "t1")]
    delta = build_delta("a.py", 0, 100, "content A", events)
    assert delta is not None
    assert "Unchanged" in delta["updatedToolOutput"]


# ---------------------------------------------------------------------------
# snapshot_event shape
# ---------------------------------------------------------------------------


def test_snapshot_event_shape():
    ev = snapshot_event("a.py", 1, 2, "hello", "t1")
    assert ev["event_type"] == "ReadSnapshot"
    assert ev["file_path"] == "a.py"
    assert ev["key"] == "a.py::1::2"
    assert ev["content"] == "hello"
    assert ev["timestamp"] == "t1"
    assert "content_hash" in ev


# ---------------------------------------------------------------------------
# maybe_wrap_dump_command
# ---------------------------------------------------------------------------


class TestMaybeWrapDumpCommand:
    @patch("rclm.hooks.read_cache.shutil.which", return_value="/usr/local/bin/rclm-read-cache")
    def test_cat_is_wrapped(self, mock_which):
        result = maybe_wrap_dump_command({"command": "cat src/a.py"})
        assert result == {"command": "rclm-read-cache cat src/a.py"}

    @patch("rclm.hooks.read_cache.shutil.which", return_value="/usr/local/bin/rclm-read-cache")
    def test_head_is_wrapped(self, mock_which):
        result = maybe_wrap_dump_command({"command": "head -50 src/a.py"})
        assert result == {"command": "rclm-read-cache head -50 src/a.py"}

    @patch("rclm.hooks.read_cache.shutil.which", return_value="/usr/local/bin/rclm-read-cache")
    def test_powershell_get_content_is_wrapped(self, mock_which):
        result = maybe_wrap_dump_command({"command": "Get-Content src/a.ps1"})
        assert result == {"command": "rclm-read-cache Get-Content src/a.ps1"}

    @patch("rclm.hooks.read_cache.shutil.which", return_value="/usr/local/bin/rclm-read-cache")
    def test_already_wrapped_skipped(self, mock_which):
        assert maybe_wrap_dump_command({"command": "rclm-read-cache cat src/a.py"}) is None

    @patch("rclm.hooks.read_cache.shutil.which", return_value="/usr/local/bin/rclm-read-cache")
    def test_piped_command_not_wrapped(self, mock_which):
        assert maybe_wrap_dump_command({"command": "cat src/a.py | grep foo"}) is None

    @patch("rclm.hooks.read_cache.shutil.which", return_value="/usr/local/bin/rclm-read-cache")
    def test_redirected_command_not_wrapped(self, mock_which):
        assert maybe_wrap_dump_command({"command": "cat src/a.py > out.txt"}) is None

    @patch("rclm.hooks.read_cache.shutil.which", return_value="/usr/local/bin/rclm-read-cache")
    def test_chained_command_not_wrapped(self, mock_which):
        assert maybe_wrap_dump_command({"command": "cat src/a.py && echo done"}) is None

    @patch("rclm.hooks.read_cache.shutil.which", return_value=None)
    def test_binary_unavailable_not_wrapped(self, mock_which):
        assert maybe_wrap_dump_command({"command": "cat src/a.py"}) is None

    @patch("rclm.hooks.read_cache.shutil.which", return_value="/usr/local/bin/rclm-read-cache")
    def test_non_dump_command_not_wrapped(self, mock_which):
        assert maybe_wrap_dump_command({"command": "python script.py"}) is None

    def test_empty_command(self):
        assert maybe_wrap_dump_command({"command": ""}) is None
