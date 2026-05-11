import json

import pytest

from rclm import _config, mcp_server


def test_load_credentials_prefers_config_over_env(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"server_url": "https://config.test", "api_key": "config-key"})
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "https://env.test")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "env-key")

    creds = mcp_server._load_credentials()

    assert creds.server_url == "https://config.test"
    assert creds.api_key == "config-key"


def test_load_credentials_uses_env_as_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "missing.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "https://env.test")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "env-key")

    creds = mcp_server._load_credentials()

    assert creds.server_url == "https://env.test"
    assert creds.api_key == "env-key"


def test_highlight_prefers_markdown_highlights():
    session = {
        "session_summary": (
            "## What Happened\nLess specific.\n\n"
            "## Highlights\n- Fixed auth fallback.\n- Added tests.\n\n"
            "## Improvements\n- None."
        )
    }

    assert mcp_server._highlight_for_session(session) == "- Fixed auth fallback.\n- Added tests."


def test_frontend_session_url_uses_frontend_url_env(monkeypatch):
    monkeypatch.setenv("FRONTEND_URL", "https://app.example.test/")

    assert (
        mcp_server._frontend_session_url("session-123")
        == "https://app.example.test/sessions/session-123"
    )


@pytest.mark.asyncio
async def test_get_session_returns_summary_and_link(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("FRONTEND_URL", "https://app.test")

    async def fake_request(self, method, path, *, params=None):
        assert method == "GET"
        assert path == "/api/sessions/session-1"
        assert params == {"include_blob": "false"}
        return {
            "session_id": "session-1",
            "title": "Auth fix",
            "session_summary": "Fixed auth token refresh.",
            "project_name": "ReclaimLLM",
        }

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    result = await mcp_server.ReclaimLLMClient().get_session("session-1")

    assert result["summary"] == "Fixed auth token refresh."
    assert result["link"] == "https://app.test/sessions/session-1"
    assert "blob" not in result


@pytest.mark.asyncio
async def test_search_by_filename_path_does_not_send_text_query(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    seen_params = {}

    async def fake_request(self, method, path, *, params=None):
        assert method == "GET"
        assert path == "/api/sessions/search"
        seen_params.update(params or {})
        return {
            "sessions": [
                {
                    "session_id": "session-1",
                    "title": "Touched auth.tsx",
                    "session_summary": "Changed auth UI.",
                }
            ]
        }

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    result = await mcp_server.ReclaimLLMClient().search_sessions(
        None,
        project_name=None,
        file_path="auth.tsx",
        record_type="session",
        limit=8,
    )

    assert "text_query" not in seen_params
    assert seen_params["file_path"] == "auth.tsx"
    assert result["sessions"][0]["title"] == "Touched auth.tsx"


def test_missing_credentials_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "missing.json")
    monkeypatch.delenv("RECLAIMLLM_SERVER_URL", raising=False)
    monkeypatch.delenv("RECLAIMLLM_API_KEY", raising=False)

    with pytest.raises(mcp_server.ReclaimLLMError):
        mcp_server._load_credentials()
