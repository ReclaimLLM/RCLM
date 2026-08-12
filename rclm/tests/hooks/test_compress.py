"""Tests for rclm.hooks.compress (PreToolUse compression engine)."""

import base64
import shlex
from unittest.mock import patch

from rclm.hooks.compress import (
    extract_base_command,
    maybe_compress,
    split_command_segments,
)


def _wrapped(command: str, session_id: str | None = None) -> str:
    encoded = base64.urlsafe_b64encode(command.encode()).decode()
    session_arg = f" --session-id {shlex.quote(session_id)}" if session_id else ""
    return f"rclm-compress{session_arg} --encoded-command {encoded}"


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

    def test_shadow_suppresses_read_shaping(self, tmp_path):
        large_file = tmp_path / "large.py"
        large_file.write_text("line\n" * 1000)
        result = maybe_compress("Read", {"file_path": str(large_file)}, shadow=True)
        assert result is None


# ---------------------------------------------------------------------------
# Grep compression
# ---------------------------------------------------------------------------


class TestCompressGrep:
    def test_no_params_defaults_both(self):
        result = maybe_compress("Grep", {"pattern": "foo"})
        assert result is not None
        assert result["head_limit"] == 50
        assert result["output_mode"] == "count"

    def test_head_limit_set_still_defaults_output_mode(self):
        result = maybe_compress("Grep", {"pattern": "foo", "head_limit": 10})
        assert result is not None
        assert result["output_mode"] == "count"
        assert "head_limit" not in result

    def test_output_mode_set_still_defaults_head_limit(self):
        result = maybe_compress("Grep", {"pattern": "foo", "output_mode": "content"})
        assert result is not None
        assert result["head_limit"] == 50
        assert "output_mode" not in result

    def test_both_already_set(self):
        result = maybe_compress(
            "Grep", {"pattern": "foo", "head_limit": 10, "output_mode": "content"}
        )
        assert result is None

    def test_shadow_suppresses_grep_shaping(self):
        result = maybe_compress("Grep", {"pattern": "foo"}, shadow=True)
        assert result is None


# ---------------------------------------------------------------------------
# Bash compression
# ---------------------------------------------------------------------------


class TestCompressBash:
    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_git_status_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "git status"})
        assert result is not None
        assert result["command"] == _wrapped("git status")

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_bash_rewrite_unaffected_by_shadow(self, mock_avail):
        """Bash always routes through rclm-compress — shadow decision is made there."""
        result = maybe_compress("Bash", {"command": "git status"}, shadow=True)
        assert result is not None
        assert result["command"] == _wrapped("git status")

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
    def test_python3_pytest_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "python3 -m pytest tests/ -v"})
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
    def test_rg_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "rg -n TODO ."})
        assert result is not None
        assert result["command"] == _wrapped("rg -n TODO .")

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_grep_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "grep -rn TODO src/"})
        assert result is not None
        assert "rclm-compress" in result["command"]

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_pipeline_rewritten_when_later_segment_matches(self, mock_avail):
        result = maybe_compress("Bash", {"command": "echo TODO | grep TODO"})
        assert result is not None
        assert result["command"] == _wrapped("echo TODO | grep TODO")

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_chain_rewritten_when_later_segment_matches(self, mock_avail):
        result = maybe_compress("Bash", {"command": "echo first && git status"})
        assert result is not None
        assert result["command"] == _wrapped("echo first && git status")

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_semicolon_chain_rewritten_when_later_segment_matches(self, mock_avail):
        result = maybe_compress("Bash", {"command": "echo first; git status"})
        assert result is not None
        assert result["command"] == _wrapped("echo first; git status")

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_nl_sed_pipeline_rewritten_after_allowlist_expansion(self, mock_avail):
        result = maybe_compress("Bash", {"command": "nl f | sed -n '1,80p'"})
        assert result == {"command": _wrapped("nl f | sed -n '1,80p'")}

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_cat_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "cat build.log"})
        assert result == {"command": _wrapped("cat build.log")}

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_session_id_is_forwarded_and_shell_quoted(self, mock_avail):
        result = maybe_compress(
            "Bash",
            {"command": "cat build.log"},
            session_id="session with spaces",
        )
        assert result == {"command": _wrapped("cat build.log", "session with spaces")}

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_unsafe_session_id_is_omitted_without_blocking_command(self, mock_avail):
        result = maybe_compress(
            "Bash",
            {"command": "cat build.log"},
            session_id="../../outside",
        )
        assert result == {"command": _wrapped("cat build.log")}

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_posix_shell_from_payload_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "git status", "shell": "/bin/zsh"})
        assert result is not None
        assert result["command"] == _wrapped("git status")

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_non_posix_shell_from_payload_not_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "git status", "shell": "powershell"})
        assert result is None

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    def test_powershell_get_content_rewritten(self, mock_avail):
        result = maybe_compress(
            "Bash",
            {"command": "Get-Content build.log", "shell": "powershell"},
        )
        assert result == {"command": _wrapped("Get-Content build.log")}

    @patch("rclm.hooks.compress._compress_available", return_value=True)
    @patch("rclm.hooks.compress.os.name", "nt")
    def test_non_posix_os_fallback_not_rewritten(self, mock_avail):
        result = maybe_compress("Bash", {"command": "git status"})
        assert result is None

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


class TestShellCommandParsing:
    def test_plain_command_base(self):
        assert split_command_segments("git status") == ["git status"]
        assert extract_base_command("git status") == "git"

    def test_env_var_prefix_base(self):
        assert extract_base_command("FOO=bar BAR=baz git status") == "git"

    def test_flag_containing_equals_is_not_skipped(self):
        assert extract_base_command("--color=always git status") == "--color=always"

    def test_pipeline_segments(self):
        assert split_command_segments("nl f | sed -n '1,80p'") == [
            "nl f",
            "sed -n '1,80p'",
        ]

    def test_non_posix_shell_returns_empty_segments(self):
        assert split_command_segments("git status | grep clean", shell="powershell") == []
        assert split_command_segments("git status | findstr clean", shell="cmd") == []

    def test_and_chain_segments(self):
        assert split_command_segments("echo first && git status") == [
            "echo first",
            "git status",
        ]

    def test_semicolon_chain_segments(self):
        assert split_command_segments("echo first; git status") == [
            "echo first",
            "git status",
        ]

    def test_separator_inside_single_quotes_not_split(self):
        assert split_command_segments("printf 'a|b && c; d' | grep a") == [
            "printf 'a|b && c; d'",
            "grep a",
        ]

    def test_separator_inside_double_quotes_not_split(self):
        assert split_command_segments('printf "a|b && c; d" | grep a') == [
            'printf "a|b && c; d"',
            "grep a",
        ]

    def test_subshell_separators_not_split(self):
        assert split_command_segments("echo $(git status | head -1) && grep x f") == [
            "echo $(git status | head -1)",
            "grep x f",
        ]

    def test_or_chain_segments(self):
        assert split_command_segments("echo first || git status") == [
            "echo first",
            "git status",
        ]

    def test_parse_failure_returns_empty(self):
        assert split_command_segments("echo 'unterminated") == []
        assert split_command_segments("echo $(git status | head -1") == []

    def test_empty_or_whitespace_input(self):
        assert split_command_segments("") == []
        assert split_command_segments("   ") == []
        assert extract_base_command("") == ""


# ---------------------------------------------------------------------------
# Other tools
# ---------------------------------------------------------------------------


def test_unknown_tool_returns_none():
    assert maybe_compress("Write", {"file_path": "/foo"}) is None
    assert maybe_compress("Edit", {"file_path": "/foo"}) is None
    assert maybe_compress("Agent", {}) is None
