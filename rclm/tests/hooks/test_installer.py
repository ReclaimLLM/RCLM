"""Tests for rclm.hooks.installer."""

import json
from pathlib import Path

import pytest

from rclm import _config, auth
from rclm.hooks import installer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_install(
    monkeypatch, tmp_path, *extra_args, api_key="test-key", server_url="http://test.example.com"
):
    """Call installer.main() with credentials pre-supplied and config path isolated."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "rclm-hooks-install",
            "--local",
            f"--api-key={api_key}",
            f"--server-url={server_url}",
            *extra_args,
        ],
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    installer.main()


def _read_settings(path: Path) -> dict:
    return json.loads(path.read_text())


def _hook_commands_for_event(settings: dict, event: str) -> list[str]:
    commands = []
    for entry in settings.get("hooks", {}).get(event, []):
        for hook in entry.get("hooks", []):
            commands.append(hook.get("command", ""))
    return commands


# ---------------------------------------------------------------------------
# Basic install (Claude Code)
# ---------------------------------------------------------------------------


def test_creates_new_settings_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path)

    settings_path = tmp_path / ".claude" / "settings.json"
    assert settings_path.exists()
    settings = _read_settings(settings_path)
    assert "hooks" in settings
    for event in installer._CLAUDE_HOOKS_TO_INJECT:
        assert event in settings["hooks"]


def test_all_expected_events_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path)

    settings = _read_settings(tmp_path / ".claude" / "settings.json")
    expected = {
        "SessionStart",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "UserPromptSubmit",
        "Stop",
        "SubagentStop",
    }
    assert set(settings["hooks"].keys()) == expected


def test_commands_are_clean_without_credential_prefix(tmp_path, monkeypatch):
    """Hook commands must not embed credentials — those live in the config file."""
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path, api_key="sk-secret")

    settings = _read_settings(tmp_path / ".claude" / "settings.json")
    for event in installer._CLAUDE_HOOKS_TO_INJECT:
        for cmd in _hook_commands_for_event(settings, event):
            assert "sk-secret" not in cmd
            assert "RECLAIMLLM" not in cmd


def test_merges_into_existing_settings_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = {
        "someOtherKey": "value",
        "hooks": {"MyCustomHook": [{"hooks": [{"type": "command", "command": "my-tool"}]}]},
    }
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(existing))

    _run_install(monkeypatch, tmp_path)

    settings = _read_settings(settings_path)
    assert settings["someOtherKey"] == "value"
    assert "MyCustomHook" in settings["hooks"]
    assert "SessionStart" in settings["hooks"]


def test_idempotent_running_twice_does_not_duplicate_hooks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(installer, "_resolve_binary", lambda name: name)
    _run_install(monkeypatch, tmp_path)
    _run_install(monkeypatch, tmp_path)

    settings = _read_settings(tmp_path / ".claude" / "settings.json")
    for event, entries in installer._CLAUDE_HOOKS_TO_INJECT.items():
        for entry in entries:
            matcher = entry.get("matcher", "")
            for hook in entry.get("hooks", []):
                command = hook["command"]
                count = sum(
                    1
                    for e in settings["hooks"].get(event, [])
                    if e.get("matcher", "") == matcher
                    for h in e.get("hooks", [])
                    if h.get("command") == command
                )
                assert count == 1, (
                    f"Expected exactly 1 entry for '{command}' (matcher={matcher!r}), got {count}"
                )


def test_handles_invalid_json_in_existing_settings(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("THIS IS NOT JSON")

    _run_install(monkeypatch, tmp_path)

    settings = _read_settings(settings_path)
    assert "hooks" in settings
    assert "Warning" in capsys.readouterr().err


def test_global_flag_targets_home_dot_claude(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "rclm-hooks-install",
            "--api-key=test-key",
        ],
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")

    installer.main()

    assert (tmp_path / ".claude" / "settings.json").exists()


# ---------------------------------------------------------------------------
# Config file persistence
# ---------------------------------------------------------------------------


def test_credentials_saved_to_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path, api_key="sk-save-me", server_url="http://saved-server.com")

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["api_key"] == "sk-save-me"
    assert config["server_url"] == "http://saved-server.com"


def test_saved_config_used_when_no_flags_provided(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # First install saves credentials.
    _run_install(monkeypatch, tmp_path, api_key="sk-saved", server_url="http://saved.com")

    # Second install with no --api-key flag reuses saved config.
    monkeypatch.setattr("sys.argv", ["rclm-hooks-install"])
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    installer.main()  # must not exit(1)

    settings = _read_settings(tmp_path / ".claude" / "settings.json")
    assert "SessionStart" in settings["hooks"]


def test_with_mcp_reuses_saved_api_key_without_browser_prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    _config.save("http://saved.com", "sk-saved")
    monkeypatch.setattr(
        "sys.argv",
        [
            "rclm-hooks-install",
            "--local",
            "--with-mcp",
        ],
    )
    monkeypatch.setattr(
        auth,
        "wait_for_api_key_via_browser",
        lambda _url: pytest.fail("browser API-key flow should not run"),
    )
    monkeypatch.setattr("rclm.mcp_install._resolve_binary", lambda: "/bin/rclm-mcp")

    installer.main()

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["api_key"] == "sk-saved"
    assert config["server_url"] == "http://saved.com"
    assert (tmp_path / ".codex" / "config.toml").exists()


# ---------------------------------------------------------------------------
# Missing credentials
# ---------------------------------------------------------------------------


def test_missing_api_key_exits_1_and_shows_setup_url(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["rclm-hooks-install"])
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    # Patch browser flow to return None immediately (simulates user cancelling).
    monkeypatch.setattr(auth, "wait_for_api_key_via_browser", lambda server_url: None)

    with pytest.raises(SystemExit) as exc_info:
        installer.main()

    assert exc_info.value.code == 1


def test_missing_api_key_does_not_write_settings_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["rclm-hooks-install"])
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(auth, "wait_for_api_key_via_browser", lambda server_url: None)

    with pytest.raises(SystemExit):
        installer.main()

    assert not (tmp_path / ".claude" / "settings.json").exists()


# ---------------------------------------------------------------------------
# Gemini CLI
# ---------------------------------------------------------------------------


def test_gemini_flag_writes_to_gemini_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path, "--gemini")

    assert (tmp_path / ".gemini" / "settings.json").exists()
    assert not (tmp_path / ".claude" / "settings.json").exists()


def test_gemini_all_expected_events_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path, "--gemini")

    settings = _read_settings(tmp_path / ".gemini" / "settings.json")
    expected = {"SessionStart", "BeforeAgent", "AfterAgent", "AfterTool", "SessionEnd"}
    assert set(settings["hooks"].keys()) == expected


def test_gemini_global_flag_targets_home_dot_gemini(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "rclm-hooks-install",
            "--gemini",
            "--api-key=sk-gemini",
        ],
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")

    installer.main()

    assert (tmp_path / ".gemini" / "settings.json").exists()
    assert not (tmp_path / ".claude" / "settings.json").exists()


def test_openclaw_flag_installs_only_openclaw(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    installed: list[bool] = []

    def fake_install_openclaw(use_global: bool) -> None:
        installed.append(use_global)

    monkeypatch.setattr(installer, "_install_openclaw", fake_install_openclaw)
    _run_install(monkeypatch, tmp_path, "--openclaw")

    assert installed == [False]
    assert not (tmp_path / ".claude" / "settings.json").exists()
    assert not (tmp_path / ".gemini" / "settings.json").exists()
    assert not (tmp_path / ".codex" / "hooks.json").exists()


def test_cursor_all_expected_events_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(installer, "_resolve_binary", lambda name: name)

    _run_install(monkeypatch, tmp_path, "--cursor")

    settings = _read_settings(tmp_path / ".cursor" / "hooks.json")
    assert set(settings["hooks"]) == set(installer._CURSOR_HOOKS_TO_INJECT)
    for event, entries in installer._CURSOR_HOOKS_TO_INJECT.items():
        assert settings["hooks"][event] == entries


def test_local_default_does_not_install_openclaw(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(installer, "_install_openclaw", lambda use_global: pytest.fail())

    _run_install(monkeypatch, tmp_path)

    assert (tmp_path / ".claude" / "settings.json").exists()


def test_with_mcp_installs_local_mcp_configs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rclm.mcp_install._resolve_binary", lambda: "/bin/rclm-mcp")

    _run_install(monkeypatch, tmp_path, "--with-mcp")

    claude = _read_settings(tmp_path / ".claude" / "mcp.json")
    assert claude["mcpServers"]["reclaimllm"] == {
        "command": "/bin/rclm-mcp",
        "args": [],
    }

    cursor = _read_settings(tmp_path / ".cursor" / "mcp.json")
    assert cursor["mcpServers"]["reclaimllm"] == {
        "command": "/bin/rclm-mcp",
        "args": [],
    }

    gemini = _read_settings(tmp_path / ".gemini" / "settings.json")
    assert gemini["mcpServers"]["reclaimllm"] == {
        "command": "/bin/rclm-mcp",
        "args": [],
    }

    codex = (tmp_path / ".codex" / "config.toml").read_text()
    assert "[mcp_servers.reclaimllm]" in codex
    assert 'command = "/bin/rclm-mcp"' in codex
    assert "args = []" in codex


# ---------------------------------------------------------------------------
# Compression flag
# ---------------------------------------------------------------------------


def test_compress_flag_saves_to_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path, "--compress")

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["compress"] is True


def test_compress_on_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["compress"] is True


def test_no_compress_disables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path, "--no-compress")

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["compress"] is False


def test_compress_flag_persists_across_reinstalls(tmp_path, monkeypatch):
    """Once --compress is set, subsequent installs without it preserve the setting."""
    monkeypatch.chdir(tmp_path)

    # First install enables compression.
    _run_install(monkeypatch, tmp_path, "--compress")

    # Second install without --compress should keep it enabled.
    monkeypatch.setattr("sys.argv", ["rclm-hooks-install"])
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    installer.main()

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["compress"] is True


def test_no_compress_persists_across_reinstalls(tmp_path, monkeypatch):
    """Once --no-compress is set, subsequent installs without any flag preserve the opt-out."""
    monkeypatch.chdir(tmp_path)

    # First install disables compression explicitly.
    _run_install(monkeypatch, tmp_path, "--no-compress")

    # Second install with no flags should keep it disabled, not fall back to the new default.
    monkeypatch.setattr("sys.argv", ["rclm-hooks-install"])
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    installer.main()

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["compress"] is False


def _seed_rtk_hook(tmp_path: Path) -> Path:
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [{"type": "command", "command": "rtk bash-tool"}],
                        },
                    ],
                },
            }
        )
    )
    return settings_path


def test_rtk_survives_with_no_compress(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings_path = _seed_rtk_hook(tmp_path)

    _run_install(monkeypatch, tmp_path, "--no-compress")

    settings = _read_settings(settings_path)
    rtk_commands = _hook_commands_for_event(settings, "PreToolUse")
    assert "rtk bash-tool" in rtk_commands


def test_rtk_removed_by_default(tmp_path, monkeypatch):
    """Compress is on by default, so RTK hooks are removed even without --compress."""
    monkeypatch.chdir(tmp_path)
    settings_path = _seed_rtk_hook(tmp_path)

    _run_install(monkeypatch, tmp_path)

    settings = _read_settings(settings_path)
    rtk_commands = _hook_commands_for_event(settings, "PreToolUse")
    assert "rtk bash-tool" not in rtk_commands


# ---------------------------------------------------------------------------
# read-cache / loop-breaker / with-mcp defaults
# ---------------------------------------------------------------------------


def test_read_cache_on_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["read_cache"] is True


def test_no_read_cache_disables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path, "--no-read-cache")

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["read_cache"] is False


def test_loop_breaker_on_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["loop_breaker"] is True


def test_no_loop_breaker_disables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path, "--no-loop-breaker")

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["loop_breaker"] is False


def test_with_mcp_on_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rclm.mcp_install._resolve_binary", lambda: "/bin/rclm-mcp")

    _run_install(monkeypatch, tmp_path)

    assert (tmp_path / ".claude" / "mcp.json").exists()


def test_no_with_mcp_skips_install(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "rclm.mcp_install.install_mcp",
        lambda **kwargs: pytest.fail("MCP install should be skipped"),
    )

    _run_install(monkeypatch, tmp_path, "--no-with-mcp")

    assert not (tmp_path / ".claude" / "mcp.json").exists()
