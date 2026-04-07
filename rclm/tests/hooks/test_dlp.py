"""Tests for rclm.hooks.dlp (DLP engine)."""

from __future__ import annotations

import os
from pathlib import Path

from rclm.hooks.dlp import (
    _build_scrub_set,
    _is_env_file,
    _parse_env_file,
    maybe_redact_input,
    maybe_redact_output,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_env(directory: Path, name: str, content: str) -> Path:
    p = directory / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _is_env_file
# ---------------------------------------------------------------------------


class TestIsEnvFile:
    def test_plain_dotenv(self):
        assert _is_env_file(".env")

    def test_dotenv_with_suffix(self):
        assert _is_env_file(".env.local")
        assert _is_env_file(".env.production")

    def test_prefixed_dotenv(self):
        assert _is_env_file("dev.env")
        assert _is_env_file("prod.env")
        assert _is_env_file("llm.env")

    def test_envrc(self):
        assert _is_env_file(".envrc")

    def test_regular_files_not_matched(self):
        assert not _is_env_file("config.json")
        assert not _is_env_file("settings.py")
        assert not _is_env_file("environment.txt")
        assert not _is_env_file("myenv")


# ---------------------------------------------------------------------------
# _parse_env_file
# ---------------------------------------------------------------------------


class TestParseEnvFile:
    def test_key_equals_value(self, tmp_path):
        p = _write_env(tmp_path, "dev.env", "SECRET_KEY=supersecretvalue123\n")
        result = _parse_env_file(p)
        assert result["SECRET_KEY"] == "supersecretvalue123"

    def test_export_prefix(self, tmp_path):
        p = _write_env(tmp_path, "dev.env", "export API_TOKEN=tok_abcdef1234\n")
        result = _parse_env_file(p)
        assert result["API_TOKEN"] == "tok_abcdef1234"

    def test_double_quoted_value(self, tmp_path):
        p = _write_env(tmp_path, "dev.env", 'DB_URL="postgres://user:pass@host/db"\n')
        result = _parse_env_file(p)
        assert result["DB_URL"] == "postgres://user:pass@host/db"

    def test_single_quoted_value(self, tmp_path):
        p = _write_env(tmp_path, "dev.env", "SECRET='my secret value'\n")
        result = _parse_env_file(p)
        assert result["SECRET"] == "my secret value"

    def test_space_separated(self, tmp_path):
        p = _write_env(tmp_path, "dev.env", "MY_KEY my_space_separated_value\n")
        result = _parse_env_file(p)
        assert result["MY_KEY"] == "my_space_separated_value"

    def test_full_line_comment_skipped(self, tmp_path):
        p = _write_env(tmp_path, "dev.env", "# this is a comment\nKEY=value12345\n")
        result = _parse_env_file(p)
        assert "# this is a comment" not in result
        assert result["KEY"] == "value12345"

    def test_inline_comment_stripped(self, tmp_path):
        p = _write_env(tmp_path, "dev.env", "KEY=secretvalue123  # this is inline\n")
        result = _parse_env_file(p)
        assert result["KEY"] == "secretvalue123"

    def test_mixed_formats(self, tmp_path):
        content = (
            "# config\n"
            "export TOKEN=tok-xyz789abc\n"
            'DB_URL="postgres://u:p@h/db"\n'
            "PORT=8080\n"
            "DEBUG=false\n"
        )
        p = _write_env(tmp_path, ".env", content)
        result = _parse_env_file(p)
        assert result["TOKEN"] == "tok-xyz789abc"
        assert result["DB_URL"] == "postgres://u:p@h/db"
        assert result["PORT"] == "8080"
        assert result["DEBUG"] == "false"

    def test_missing_file_returns_empty(self, tmp_path):
        result = _parse_env_file(tmp_path / "nonexistent.env")
        assert result == {}


# ---------------------------------------------------------------------------
# _build_scrub_set
# ---------------------------------------------------------------------------


class TestBuildScrubSet:
    def test_short_values_excluded(self):
        # Values < 5 chars must not enter the scrub set
        secrets = {"PORT": "8080", "V": "ab"}
        scrub = _build_scrub_set(secrets)
        values = {v for v, _ in scrub}
        assert "8080" not in values
        assert "ab" not in values

    def test_safe_values_excluded(self):
        secrets = {
            "FLAG": "true",
            "ENABLED": "false",
            "HOST": "localhost",
            "ADDR": "0.0.0.0",
        }
        scrub = _build_scrub_set(secrets)
        values = {v for v, _ in scrub}
        assert not values  # all filtered

    def test_pure_integers_excluded(self):
        secrets = {"TIMEOUT": "30000"}
        scrub = _build_scrub_set(secrets)
        assert not scrub

    def test_real_secrets_included(self):
        secrets = {"API_KEY": "sk-ant-longkey123456"}
        scrub = _build_scrub_set(secrets)
        assert len(scrub) == 1
        val, placeholder = scrub[0]
        assert val == "sk-ant-longkey123456"
        assert placeholder == "[REDACTED:API_KEY]"

    def test_sorted_longest_first(self):
        secrets = {"SHORT": "abcde", "LONG": "abcdefghijklmn"}
        scrub = _build_scrub_set(secrets)
        assert scrub[0][0] == "abcdefghijklmn"


# ---------------------------------------------------------------------------
# maybe_redact_input — Read tool
# ---------------------------------------------------------------------------


class TestMaybeRedactInputRead:
    def test_env_file_redirected_to_sanitised_temp(self, tmp_path):
        _write_env(tmp_path, "dev.env", "API_KEY=sk-supersecretvalue\nPORT=8080\n")
        env_file = tmp_path / "dev.env"

        result = maybe_redact_input("Read", {"file_path": str(env_file)}, str(tmp_path))

        assert result is not None
        temp_path = result["file_path"]
        assert temp_path != str(env_file)
        content = Path(temp_path).read_text()
        assert "[REDACTED:API_KEY]" in content
        assert "sk-supersecretvalue" not in content
        # PORT=8080 value "8080" is a pure integer — not redacted
        assert "8080" in content
        # Clean up
        os.unlink(temp_path)

    def test_non_env_file_not_redirected(self, tmp_path):
        src = tmp_path / "main.py"
        src.write_text("print('hello')")
        result = maybe_redact_input("Read", {"file_path": str(src)}, str(tmp_path))
        assert result is None

    def test_self_scrubs_when_cwd_has_no_env_files(self, tmp_path):
        # Even if cwd has no sibling env files, the file scrubs itself.
        env_file = tmp_path / "dev.env"
        env_file.write_text("API_KEY=sk-supersecretvalue\n")
        empty_dir = tmp_path / "subdir"
        empty_dir.mkdir()
        result = maybe_redact_input("Read", {"file_path": str(env_file)}, str(empty_dir))
        # Should still redirect — the file is its own secret source
        assert result is not None
        content = Path(result["file_path"]).read_text()
        assert "[REDACTED:API_KEY]" in content
        os.unlink(result["file_path"])

    def test_empty_env_file_returns_none(self, tmp_path):
        # A file with no parseable secrets (e.g. only comments) → no redirect needed
        env_file = tmp_path / "dev.env"
        env_file.write_text("# just a comment\nDEBUG=false\nPORT=8080\n")
        result = maybe_redact_input("Read", {"file_path": str(env_file)}, str(tmp_path))
        assert result is None  # all values filtered by scrub-set rules

    def test_track_temp_callback_invoked(self, tmp_path):
        _write_env(tmp_path, ".env", "SECRET=verylongsecretvalue\n")
        env_file = tmp_path / ".env"

        tracked: list[str] = []
        maybe_redact_input(
            "Read",
            {"file_path": str(env_file)},
            str(tmp_path),
            track_temp=tracked.append,
        )

        assert len(tracked) == 1
        assert os.path.exists(tracked[0])
        os.unlink(tracked[0])

    def test_unknown_tool_returns_none(self, tmp_path):
        result = maybe_redact_input("Write", {"file_path": "/foo"}, str(tmp_path))
        assert result is None

    def test_env_file_updated_between_calls_is_fresh(self, tmp_path):
        env_path = tmp_path / "dev.env"
        env_path.write_text("ORIGINAL_KEY=originalvalue123\n")

        # First call — original secret in scrub set
        result1 = maybe_redact_input("Read", {"file_path": str(env_path)}, str(tmp_path))
        assert result1 is not None
        content1 = Path(result1["file_path"]).read_text()
        os.unlink(result1["file_path"])

        # Update the env file mid-session
        env_path.write_text("NEW_KEY=brandnewsecretvalue456\n")

        # Second call — must pick up the new secret
        result2 = maybe_redact_input("Read", {"file_path": str(env_path)}, str(tmp_path))
        assert result2 is not None
        content2 = Path(result2["file_path"]).read_text()
        os.unlink(result2["file_path"])

        assert "[REDACTED:NEW_KEY]" in content2
        assert "brandnewsecretvalue456" not in content2
        # Old key should not be redacted in the second call (it's gone from the file)
        assert "originalvalue123" not in content1


# ---------------------------------------------------------------------------
# maybe_redact_input — Bash tool
# ---------------------------------------------------------------------------


class TestMaybeRedactInputBash:
    def test_cat_env_file_blocked(self, tmp_path):
        result = maybe_redact_input("Bash", {"command": f"cat {tmp_path}/dev.env"}, str(tmp_path))
        assert result is not None
        assert "echo" in result["command"]
        assert "DLP" in result["command"]

    def test_cat_non_env_file_not_blocked(self, tmp_path):
        result = maybe_redact_input("Bash", {"command": "cat main.py"}, str(tmp_path))
        assert result is None

    def test_non_cat_command_not_blocked(self, tmp_path):
        result = maybe_redact_input("Bash", {"command": "git status"}, str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# maybe_redact_output
# ---------------------------------------------------------------------------


class TestMaybeRedactOutput:
    def test_secret_in_output_is_scrubbed(self, tmp_path):
        _write_env(tmp_path, "dev.env", "DB_PASS=supersecretpassword123\n")
        response = "connecting with supersecretpassword123 to db"
        result = maybe_redact_output("Bash", response, str(tmp_path))
        assert result is not None
        assert "[REDACTED:DB_PASS]" in result
        assert "supersecretpassword123" not in result

    def test_no_secrets_in_output_returns_none(self, tmp_path):
        _write_env(tmp_path, "dev.env", "API_KEY=secretkey123456\n")
        response = "build succeeded in 2.1s"
        result = maybe_redact_output("Bash", response, str(tmp_path))
        assert result is None

    def test_no_env_files_returns_none(self, tmp_path):
        # Empty directory — no env files
        result = maybe_redact_output("Bash", "some output with data", str(tmp_path))
        assert result is None

    def test_non_string_response_handled(self, tmp_path):
        _write_env(tmp_path, "dev.env", "TOKEN=tok_abcde12345\n")
        result = maybe_redact_output("Bash", None, str(tmp_path))
        assert result is None  # None converts to "None", no match

    def test_multiple_secrets_all_scrubbed(self, tmp_path):
        _write_env(
            tmp_path,
            "dev.env",
            "KEY1=firstsecretvalue1\nKEY2=secondsecretvalue2\n",
        )
        response = "key1=firstsecretvalue1 key2=secondsecretvalue2"
        result = maybe_redact_output("Bash", response, str(tmp_path))
        assert result is not None
        assert "firstsecretvalue1" not in result
        assert "secondsecretvalue2" not in result
