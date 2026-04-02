"""Tests for git output filters."""

from rclm.compress.filters.git import filter_git

# ---------------------------------------------------------------------------
# git status
# ---------------------------------------------------------------------------


class TestFilterGitStatus:
    def test_clean_repo(self):
        result = filter_git("status", "")
        assert "clean" in result
        assert "nothing to commit" in result

    def test_short_format_modified(self):
        output = " M src/main.py\n M src/utils.py\n"
        result = filter_git("status", output)
        assert "2 modified" in result
        assert "src/main.py" in result
        assert "src/utils.py" in result

    def test_short_format_untracked(self):
        output = "?? new_file.py\n?? another.py\n"
        result = filter_git("status", output)
        assert "2 untracked" in result

    def test_short_format_mixed(self):
        output = " M changed.py\n?? new.py\nA  added.py\n"
        result = filter_git("status", output)
        assert "modified" in result
        assert "untracked" in result

    def test_long_format_modified(self):
        output = (
            "On branch main\n"
            "Changes not staged for commit:\n"
            '  (use "git add <file>..." to update what will be committed)\n'
            "\n"
            "\tmodified:   src/main.py\n"
            "\tmodified:   src/utils.py\n"
        )
        result = filter_git("status", output)
        assert "modified" in result

    def test_significant_compression(self):
        """Status output should be significantly smaller than input."""
        lines = [f" M src/file_{i}.py" for i in range(50)]
        output = "\n".join(lines)
        result = filter_git("status", output)
        assert len(result) < len(output)


# ---------------------------------------------------------------------------
# git diff
# ---------------------------------------------------------------------------


class TestFilterGitDiff:
    def test_no_changes(self):
        result = filter_git("diff", "")
        assert "no changes" in result

    def test_keeps_diff_headers(self):
        output = (
            "diff --git a/file.py b/file.py\n"
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-old line\n"
            "+new line\n"
        )
        result = filter_git("diff", output)
        assert "diff --git" in result
        assert "-old line" in result
        assert "+new line" in result

    def test_truncates_long_hunks(self):
        """Hunks with many lines should be truncated."""
        lines = ["diff --git a/f.py b/f.py", "--- a/f.py", "+++ b/f.py", "@@ -1,50 +1,50 @@"]
        lines += [f"+added line {i}" for i in range(50)]
        output = "\n".join(lines)
        result = filter_git("diff", output)
        assert "more lines" in result
        assert len(result) < len(output)


# ---------------------------------------------------------------------------
# git log
# ---------------------------------------------------------------------------


class TestFilterGitLog:
    def test_no_commits(self):
        result = filter_git("log", "")
        assert "no commits" in result

    def test_condenses_verbose_log(self):
        output = (
            "commit abc1234567890abcdef1234567890abcdef12345\n"
            "Author: Test <test@test.com>\n"
            "Date:   Mon Jan 1 00:00:00 2024 +0000\n"
            "\n"
            "    First commit message\n"
            "\n"
            "commit def4567890abcdef1234567890abcdef12345678\n"
            "Author: Test <test@test.com>\n"
            "Date:   Mon Jan 2 00:00:00 2024 +0000\n"
            "\n"
            "    Second commit message\n"
        )
        result = filter_git("log", output)
        assert "abc1234" in result
        assert "First commit" in result
        assert "def4567" in result
        assert "Author:" not in result  # Should be stripped


# ---------------------------------------------------------------------------
# git action commands
# ---------------------------------------------------------------------------


class TestFilterGitAction:
    def test_commit_extracts_summary(self):
        output = "[main abc1234] Fix the bug\n 1 file changed, 2 insertions(+), 1 deletion(-)\n"
        result = filter_git("commit", output)
        assert "[main abc1234]" in result

    def test_push_extracts_arrow(self):
        output = (
            "Enumerating objects: 5, done.\n"
            "Counting objects: 100% (5/5), done.\n"
            "Writing objects: 100% (3/3), 300 bytes | 300.00 KiB/s, done.\n"
            "   abc1234..def5678  main -> main\n"
        )
        result = filter_git("push", output)
        assert "->" in result
        assert "Enumerating" not in result

    def test_empty_output_returns_ok(self):
        result = filter_git("add", "")
        assert result == "ok"

    def test_already_up_to_date(self):
        result = filter_git("pull", "Already up to date.\n")
        assert "Already up to date" in result


# ---------------------------------------------------------------------------
# Unknown subcommands
# ---------------------------------------------------------------------------


def test_unknown_subcommand_returns_none():
    assert filter_git("unknown-subcommand", "some output") is None
