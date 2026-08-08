from rclm import mcp_install


def test_codex_install_replaces_existing_reclaimllm_sections(tmp_path):
    config = tmp_path / "config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "\n".join(
            [
                'model = "gpt-5"',
                "",
                "[mcp_servers.reclaimllm]",
                'command = "old"',
                "args = []",
                "",
                "[mcp_servers.reclaimllm.env]",
                'OLD = "1"',
                "",
                "[mcp_servers.supabase]",
                'url = "https://example.test/mcp"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    mcp_install._install_codex_mcp(config, "/bin/rclm-mcp")

    text = config.read_text(encoding="utf-8")
    assert text.count("[mcp_servers.reclaimllm]") == 1
    assert "[mcp_servers.reclaimllm.env]" not in text
    assert 'command = "/bin/rclm-mcp"' in text
    assert "tool_timeout_sec = 600" in text
    assert "[mcp_servers.supabase]" in text


def test_install_mcp_adds_gemini_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mcp_install, "_resolve_binary", lambda: "/bin/rclm-mcp")
    gemini_settings = tmp_path / ".gemini" / "settings.json"
    gemini_settings.parent.mkdir(parents=True, exist_ok=True)
    gemini_settings.write_text(
        '{"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "keep"}]}]}}',
        encoding="utf-8",
    )

    paths = mcp_install.install_mcp(use_global=False)

    assert gemini_settings.resolve() in [path.resolve() for path in paths]
    data = mcp_install._load_json(gemini_settings)
    assert data["hooks"]["SessionStart"][0]["hooks"][0]["command"] == "keep"
    assert data["mcpServers"]["reclaimllm"] == {
        "command": "/bin/rclm-mcp",
        "args": [],
        "timeout": 600_000,
    }


def test_install_mcp_respects_providers_filter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mcp_install, "_resolve_binary", lambda: "/bin/rclm-mcp")

    paths = mcp_install.install_mcp(use_global=False, providers=["claude"])

    assert len(paths) == 1
    assert paths[0].resolve() == (tmp_path / ".claude" / "mcp.json").resolve()
    assert (tmp_path / ".claude" / "mcp.json").exists()
    assert not (tmp_path / ".gemini" / "settings.json").exists()
    assert not (tmp_path / ".cursor" / "mcp.json").exists()
    assert not (tmp_path / ".codex" / "config.toml").exists()
