"""Tests for rclm.hooks.installer."""

import json
from pathlib import Path

import pytest

from rclm import _config
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
    _run_install(monkeypatch, tmp_path)
    _run_install(monkeypatch, tmp_path)

    settings = _read_settings(tmp_path / ".claude" / "settings.json")
    for event, entries in installer._CLAUDE_HOOKS_TO_INJECT.items():
        for entry in entries:
            for hook in entry.get("hooks", []):
                command = hook["command"]
                commands = _hook_commands_for_event(settings, event)
                assert commands.count(command) == 1, (
                    f"Expected exactly 1 entry for '{command}', got {commands.count(command)}"
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


# ---------------------------------------------------------------------------
# Missing credentials
# ---------------------------------------------------------------------------


def test_missing_api_key_exits_1_and_shows_setup_url(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["rclm-hooks-install"])
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    # Patch browser flow to return None immediately (simulates user cancelling).
    monkeypatch.setattr(installer, "_wait_for_api_key_via_browser", lambda server_url: None)

    with pytest.raises(SystemExit) as exc_info:
        installer.main()

    assert exc_info.value.code == 1


def test_missing_api_key_does_not_write_settings_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["rclm-hooks-install"])
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(installer, "_wait_for_api_key_via_browser", lambda server_url: None)

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


# ---------------------------------------------------------------------------
# Compression flag
# ---------------------------------------------------------------------------


def test_compress_flag_saves_to_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path, "--compress")

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["compress"] is True


def test_compress_off_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_install(monkeypatch, tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config.get("compress") is False


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


def test_rtk_only_removed_when_compress_enabled(tmp_path, monkeypatch):
    """RTK hooks should only be removed when --compress is passed."""
    monkeypatch.chdir(tmp_path)

    # Pre-populate settings with a fake RTK hook.
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

    # Install without --compress — RTK should survive.
    _run_install(monkeypatch, tmp_path)
    settings = _read_settings(settings_path)
    rtk_commands = _hook_commands_for_event(settings, "PreToolUse")
    assert "rtk bash-tool" in rtk_commands

    # Install with --compress — RTK should be removed.
    _run_install(monkeypatch, tmp_path, "--compress")
    settings = _read_settings(settings_path)
    rtk_commands = _hook_commands_for_event(settings, "PreToolUse")
    assert "rtk bash-tool" not in rtk_commands
