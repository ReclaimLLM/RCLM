"""Tests for rclm.hooks.dlp (DLP engine)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from rclm.hooks.dlp import (
    DLPRedactionError,
    _build_scrub_set,
    _find_env_files,
    _is_env_file,
    _parse_env_file,
    input_may_read_env,
    maybe_redact_input,
    maybe_redact_output,
    maybe_redact_value,
    reconcile_captured_tool_inputs,
    reconcile_captured_tool_results,
    redact_high_confidence_value,
    redact_json_payload,
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

    def test_missing_file_fails_closed(self, tmp_path):
        with pytest.raises(DLPRedactionError):
            _parse_env_file(tmp_path / "nonexistent.env")

    def test_multiline_quoted_value(self, tmp_path):
        p = _write_env(tmp_path, ".env", 'PRIVATE_KEY="line-one\nline-two"\n')
        assert _parse_env_file(p)["PRIVATE_KEY"] == "line-one\nline-two"


class TestFindEnvFiles:
    def test_discovers_nested_env_files_recursively(self, tmp_path):
        nested = tmp_path / "services" / "api"
        nested.mkdir(parents=True)
        root_env = _write_env(tmp_path, ".env.local", "ROOT_SECRET=root-secret-value\n")
        nested_env = _write_env(nested, "dev.env", "NESTED_SECRET=nested-secret-value\n")

        assert _find_env_files(str(tmp_path)) == [root_env, nested_env]

    def test_does_not_follow_symlinked_env_files(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        target = _write_env(outside, ".env", "SECRET=outside-secret-value\n")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".env").symlink_to(target)

        assert _find_env_files(str(workspace)) == []


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

    def test_ordinary_configuration_values_excluded(self):
        secrets = {
            "QDRANT_COLLECTION": "sessions",
            "ENTERPRISE_PROXY_DEFAULT_SLUG": "default",
            "AZURE_OPENAI_LIST_MODE": "config",
            "AWS_REGION": "us-east-1",
            "OPENAI_MODEL": "gpt-5-mini",
            "POSTHOG_HOST": "https://analytics.example.test",
        }

        assert _build_scrub_set(secrets) == []

    def test_sensitive_key_and_opaque_value_are_included(self):
        secrets = {
            "DB_PASSWORD": "ordinary-looking-password",
            "CUSTOM_VALUE": "AbCdEfGhIjKlMnOpQrStUv123456",
        }

        placeholders = {placeholder for _, placeholder in _build_scrub_set(secrets)}

        assert placeholders == {
            "[REDACTED:DB_PASSWORD]",
            "[REDACTED:CUSTOM_VALUE]",
        }

    def test_explicitly_public_keys_are_excluded(self):
        secrets = {
            "NEXT_PUBLIC_API_KEY": "AbCdEfGhIjKlMnOpQrStUv123456",
            "SUPABASE_ANON_KEY": "AbCdEfGhIjKlMnOpQrStUv123456",
        }

        assert _build_scrub_set(secrets) == []

    def test_direct_access_includes_every_assignment(self):
        secrets = {"PORT": "8080", "DEBUG": "false", "REGION": "us-east-1"}

        placeholders = {
            placeholder for _, placeholder in _build_scrub_set(secrets, include_all_values=True)
        }

        assert placeholders == {
            "[REDACTED:PORT]",
            "[REDACTED:DEBUG]",
            "[REDACTED:REGION]",
        }

    def test_sorted_longest_first(self):
        secrets = {"SHORT_TOKEN": "abcde", "LONG_TOKEN": "abcdefghijklmn"}
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
        assert "[REDACTED:PORT]" in content
        assert "8080" not in content
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

    def test_comment_only_env_file_returns_none(self, tmp_path):
        env_file = tmp_path / "dev.env"
        env_file.write_text("# just a comment\n")
        result = maybe_redact_input("Read", {"file_path": str(env_file)}, str(tmp_path))
        assert result is None

    def test_config_only_env_file_is_fully_sanitized(self, tmp_path):
        env_file = tmp_path / ".env.example"
        env_file.write_text("DEBUG=false\nPORT=8080\nREGION=us-east-1\n")

        result = maybe_redact_input("Read", {"file_path": str(env_file)}, str(tmp_path))

        assert result is not None
        content = Path(result["file_path"]).read_text()
        assert "false" not in content
        assert "8080" not in content
        assert "us-east-1" not in content
        os.unlink(result["file_path"])

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

    def test_oversized_env_file_fails_closed(self, tmp_path, monkeypatch):
        env_file = _write_env(tmp_path, ".env", "TOKEN=oversized-secret-value\n")
        monkeypatch.setattr("rclm.hooks.dlp.MAX_ENV_FILE_BYTES", 1)

        with pytest.raises(DLPRedactionError):
            maybe_redact_input("Read", {"file_path": str(env_file)}, str(tmp_path))

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

    def test_supported_sed_read_of_nested_env_file_is_blocked(self, tmp_path):
        nested = tmp_path / "service"
        nested.mkdir()
        _write_env(nested, ".env", "TOKEN=sed-secret-value\n")

        result = maybe_redact_input(
            "Bash",
            {"command": "sed -n '1p' service/.env"},
            str(tmp_path),
        )

        assert result is not None
        assert "DLP" in result["command"]


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

    def test_nested_env_secret_is_scrubbed(self, tmp_path):
        nested = tmp_path / "apps" / "api"
        nested.mkdir(parents=True)
        _write_env(nested, ".env.production", "TOKEN=nested-production-secret\n")

        result = maybe_redact_output("Bash", "nested-production-secret", str(tmp_path))

        assert result == "[REDACTED:TOKEN]"

    def test_common_config_values_do_not_redact_unrelated_output(self, tmp_path):
        _write_env(
            tmp_path,
            "local.env",
            "QDRANT_COLLECTION=sessions\n"
            "ENTERPRISE_PROXY_DEFAULT_SLUG=default\n"
            "AZURE_OPENAI_LIST_MODE=config\n",
        )

        result = maybe_redact_output(
            "Bash",
            "sessions use the default config",
            str(tmp_path),
        )

        assert result is None

    def test_direct_env_access_redacts_common_config_values(self, tmp_path):
        _write_env(tmp_path, "local.env", "QDRANT_COLLECTION=sessions\n")

        result = maybe_redact_output(
            "Bash",
            "QDRANT_COLLECTION=sessions",
            str(tmp_path),
            redact_all=True,
        )

        assert result == "QDRANT_COLLECTION=[REDACTED:QDRANT_COLLECTION]"

    def test_example_and_test_env_files_are_not_ambient_secret_sources(self, tmp_path):
        _write_env(tmp_path, ".env.example", "API_KEY=example-secret-value\n")
        _write_env(tmp_path, "pytest.env", "PASSWORD=test-secret-value\n")

        result = maybe_redact_output(
            "Bash",
            "example-secret-value test-secret-value",
            str(tmp_path),
        )

        assert result is None

    def test_shape_preserving_redaction(self, tmp_path):
        _write_env(tmp_path, ".env", "TOKEN=structured-secret-value\n")
        response = {"content": [{"type": "text", "text": "structured-secret-value"}]}

        result = maybe_redact_value(response, str(tmp_path))

        assert result == {"content": [{"type": "text", "text": "[REDACTED:TOKEN]"}]}


def test_reconcile_replaces_only_explicitly_redacted_transcript_results():
    redacted = SimpleNamespace(tool_use_id="call-1", tool_result="raw-secret")
    untouched = SimpleNamespace(tool_use_id="call-2", tool_result="keep-structured-result")
    events = [
        {
            "event_type": "PostToolUse",
            "tool_use_id": "call-1",
            "tool_response": "[REDACTED:TOKEN]",
            "dlp_redacted": True,
        },
        {
            "event_type": "PostToolUse",
            "tool_use_id": "call-2",
            "tool_response": "different-capture-shape",
            "dlp_redacted": False,
        },
    ]

    reconcile_captured_tool_results([redacted, untouched], events)

    assert redacted.tool_result == "[REDACTED:TOKEN]"
    assert untouched.tool_result == "keep-structured-result"


def test_serialized_multiline_env_value_is_redacted(tmp_path):
    _write_env(tmp_path, ".env", 'PRIVATE_KEY="line-one\nline-two"\n')
    payload = json.dumps({"content": "line-one\nline-two"})

    result = redact_json_payload(payload, str(tmp_path))

    assert "line-one" not in result
    assert json.loads(result)["content"] == "[REDACTED:PRIVATE_KEY]"


def test_inline_jwt_is_redacted_without_matching_env_file(tmp_path):
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJyb2xlIjoic2VydmljZV9yb2xlIn0.signature_value_12345"

    assert maybe_redact_output("Bash", f"token={jwt}", str(tmp_path)) == "token=[REDACTED:JWT]"
    assert redact_high_confidence_value({"command": f"curl -H 'Bearer {jwt}'"}) == {
        "command": "curl -H 'Bearer [REDACTED:JWT]'"
    }
    payload = redact_json_payload(json.dumps({"tool_input": {"token": jwt}}), str(tmp_path))
    assert jwt not in payload
    assert json.loads(payload)["tool_input"]["token"] == "[REDACTED:JWT]"


def test_reconcile_replaces_transcript_tool_input_with_sanitized_capture():
    call = SimpleNamespace(tool_use_id="call-1", tool_input={"command": "raw-secret"})
    events = [
        {
            "event_type": "PreToolUse",
            "tool_use_id": "call-1",
            "tool_input": {"command": "[REDACTED:JWT]"},
        }
    ]

    reconcile_captured_tool_inputs([call], events)

    assert call.tool_input == {"command": "[REDACTED:JWT]"}


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        ("Read", {"file_path": "services/api/.env.local"}),
        ("run_shell_command", {"command": "cat config/dev.env"}),
        ("Bash", {"command": "printenv"}),
        ("exec", {"input": 'tools.exec_command({cmd:"sed -n 1,20p services/api/.env"})'}),
    ],
)
def test_recognizes_env_access_for_targeted_fail_closed_behavior(tool_name, tool_input):
    assert input_may_read_env(tool_name, tool_input)


def test_unrelated_shell_command_remains_fail_open():
    assert not input_may_read_env("Bash", {"command": "ls"})
