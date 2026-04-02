"""Tests for rclm.hooks.compress (PreToolUse compression engine)."""
from unittest.mock import patch

from rclm.hooks.compress import maybe_compress


# ---------------------------------------------------------------------------
# Read compression
# ---------------------------------------------------------------------------

class TestCompressRead:
    def test_no_limit_set_small_file(self, tmp_path):
        small_file = tmp_path / "small.py"
        small_file.write_text("line\n" * 100)
        result = maybe_compress("Read", {"file_path": str(small_file)})
        assert result is None

    def test_no_limit_set_large_file(self, tmp_path):
        large_file = tmp_path / "large.py"
        large_file.write_text("line\n" * 1000)
        result = maybe_compress("Read", {"file_path": str(large_file)})
        assert result is not None
        assert "limit" in result
        assert result["limit"] == 200

    def test_limit_already_set(self, tmp_path):
        large_file = tmp_path / "large.py"
        large_file.write_text("line\n" * 1000)
        result = maybe_compress("Read", {"file_path": str(large_file), "limit": 50})
        assert result is None

    def test_missing_file(self):
        result = maybe_compress("Read", {"file_path": "/nonexistent/file.py"})
        assert result is None

    def test_no_file_path(self):
        result = maybe_compress("Read", {})
        assert result is None


# ---------------------------------------------------------------------------
# Grep compression
# ---------------------------------------------------------------------------

class TestCompressGrep:
    def test_no_head_limit_injects_default(self):
        result = maybe_compress("Grep", {"pattern": "foo"})
        assert result is not None
        assert result["head_limit"] == 50

    def test_head_limit_already_set(self):
        result = maybe_compress("Grep", {"pattern": "foo", "head_limit": 10})
        assert result is None


# ---------------------------------------------------------------------------
# Bash compression
# ---------------------------------------------------------------------------

class TestCompressBash:
    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_git_status_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "git status"})
        assert result is not None
        assert result["command"] == "rclm-compress git status"

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_git_diff_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "git diff --staged"})
        assert result is not None
        assert "rclm-compress" in result["command"]

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_pytest_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "python -m pytest tests/ -v"})
        assert result is not None
        assert "rclm-compress" in result["command"]

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_npm_test_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "npm test"})
        assert result is not None
        assert "rclm-compress" in result["command"]

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_ls_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "ls -la"})
        assert result is not None
        assert "rclm-compress" in result["command"]

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_already_wrapped_skipped(self, mock_avail):
        result = maybe_compress("Bash", {"command": "rclm-compress git status"})
        assert result is None

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_rtk_wrapped_skipped(self, mock_avail):
        result = maybe_compress("Bash", {"command": "rtk git status"})
        assert result is None

    @patch("rclm.hooks.compress._compress_available", return_value=False)
    def test_compress_not_available(self, mock_avail):
        result = maybe_compress("Bash", {"command": "git status"})
        assert result is None

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_non_matching_command_not_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "echo hello"})
        assert result is None

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_python_non_test_not_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "python script.py"})
        assert result is None

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_npm_non_test_not_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "npm install"})
        assert result is None

    def test_empty_command(self):
        result = maybe_compress("Bash", {"command": ""})
        assert result is None


# ---------------------------------------------------------------------------
# Other tools
# ---------------------------------------------------------------------------

def test_unknown_tool_returns_none():
    assert maybe_compress("Write", {"file_path": "/foo"}) is None
    assert maybe_compress("Edit", {"file_path": "/foo"}) is None
    assert maybe_compress("Agent", {}) is None
