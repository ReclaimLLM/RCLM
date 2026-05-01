"""Tests for OpenClaw plugin generation."""

from __future__ import annotations

import json

from rclm.hooks import openclaw_plugin


def _redirect_openclaw_home(tmp_path, monkeypatch):
    openclaw_dir = tmp_path / ".openclaw"
    monkeypatch.setattr(openclaw_plugin, "OPENCLAW_DIR", openclaw_dir)
    monkeypatch.setattr(openclaw_plugin, "OPENCLAW_CONFIG_PATH", openclaw_dir / "openclaw.json")
    monkeypatch.setattr(
        openclaw_plugin,
        "PLUGIN_DIR",
        openclaw_dir / "extensions" / openclaw_plugin.PLUGIN_ID,
    )
    monkeypatch.setattr(openclaw_plugin, "_resolve_binary", lambda name: f"/bin/{name}")
    return openclaw_dir


def test_install_plugin_writes_files_and_config(tmp_path, monkeypatch):
    openclaw_dir = _redirect_openclaw_home(tmp_path, monkeypatch)

    plugin_dir = openclaw_plugin.install_plugin()

    assert plugin_dir == openclaw_dir / "extensions" / "reclaimllm"
    index_ts = (plugin_dir / "index.ts").read_text()
    assert "api.on(hookName as any" in index_ts
    assert '"session_start"' in index_ts
    assert '"before_tool_call"' in index_ts
    assert "/bin/rclm-openclaw-hooks" in index_ts

    manifest = json.loads((plugin_dir / "openclaw.plugin.json").read_text())
    assert manifest["id"] == "reclaimllm"
    assert manifest["configSchema"] == {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }

    package = json.loads((plugin_dir / "package.json").read_text())
    assert package["type"] == "module"
    assert package["openclaw"]["extensions"] == ["./index.ts"]

    config = json.loads((openclaw_dir / "openclaw.json").read_text())
    assert config["plugins"]["enabled"] is True
    assert str(plugin_dir) in config["plugins"]["load"]["paths"]
    assert config["plugins"]["entries"]["reclaimllm"] == {
        "enabled": True,
        "hooks": {"allowConversationAccess": True},
    }


def test_install_plugin_preserves_allowlist_and_adds_reclaimllm(tmp_path, monkeypatch):
    openclaw_dir = _redirect_openclaw_home(tmp_path, monkeypatch)
    openclaw_dir.mkdir()
    (openclaw_dir / "openclaw.json").write_text(
        json.dumps({"plugins": {"allow": ["existing"], "entries": {"existing": {"enabled": True}}}})
    )

    openclaw_plugin.install_plugin()

    config = json.loads((openclaw_dir / "openclaw.json").read_text())
    assert config["plugins"]["allow"] == ["existing", "reclaimllm"]
    assert config["plugins"]["entries"]["existing"] == {"enabled": True}


def test_install_plugin_does_not_modify_non_json_config(tmp_path, monkeypatch, capsys):
    openclaw_dir = _redirect_openclaw_home(tmp_path, monkeypatch)
    openclaw_dir.mkdir()
    config_path = openclaw_dir / "openclaw.json"
    config_path.write_text("{ // json5 comment\n plugins: {} }")

    openclaw_plugin.install_plugin()

    assert config_path.read_text() == "{ // json5 comment\n plugins: {} }"
    assert "not strict JSON" in capsys.readouterr().err


def test_uninstall_plugin_removes_only_reclaimllm(tmp_path, monkeypatch):
    openclaw_dir = _redirect_openclaw_home(tmp_path, monkeypatch)
    plugin_dir = openclaw_plugin.install_plugin()
    config_path = openclaw_dir / "openclaw.json"
    config = json.loads(config_path.read_text())
    config["plugins"]["entries"]["other"] = {"enabled": True}
    config["plugins"]["load"]["paths"].append("/tmp/other")
    config_path.write_text(json.dumps(config))

    removed_files, removed_config = openclaw_plugin.uninstall_plugin()

    assert removed_files is True
    assert removed_config is True
    assert not plugin_dir.exists()

    config = json.loads(config_path.read_text())
    assert "reclaimllm" not in config["plugins"].get("entries", {})
    assert config["plugins"]["entries"]["other"] == {"enabled": True}
    assert "/tmp/other" in config["plugins"]["load"]["paths"]
