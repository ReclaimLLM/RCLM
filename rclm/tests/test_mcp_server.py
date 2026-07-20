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


@pytest.mark.asyncio
async def test_search_sessions_passes_scope_param(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    seen_params = {}

    async def fake_request(self, method, path, *, params=None):
        seen_params.update(params or {})
        return {"sessions": [], "scope": "team"}

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    result = await mcp_server.ReclaimLLMClient().search_sessions(
        "auth bug",
        project_name=None,
        file_path=None,
        record_type="session",
        limit=5,
        scope="team",
    )

    assert seen_params["scope"] == "team"
    assert result["scope"] == "team"


@pytest.mark.asyncio
async def test_search_sessions_omits_scope_param_by_default(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    seen_params = {}

    async def fake_request(self, method, path, *, params=None):
        seen_params.update(params or {})
        return {"sessions": []}

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    await mcp_server.ReclaimLLMClient().search_sessions(
        "auth bug",
        project_name=None,
        file_path=None,
        record_type="session",
        limit=5,
    )

    assert "scope" not in seen_params


@pytest.mark.asyncio
async def test_search_sessions_rejects_invalid_scope(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    with pytest.raises(mcp_server.ReclaimLLMError):
        await mcp_server.ReclaimLLMClient().search_sessions(
            "auth bug",
            project_name=None,
            file_path=None,
            record_type="session",
            limit=5,
            scope="everyone",
        )


def test_session_search_result_includes_owner_fields_when_shared():
    session = {
        "session_id": "session-1",
        "title": "Auth fix",
        "user_email": "teammate@example.com",
        "user_display_name": "Teammate",
    }

    result = mcp_server._session_search_result(session)

    assert result["owner_email"] == "teammate@example.com"
    assert result["owner_name"] == "Teammate"


def test_session_search_result_omits_owner_fields_when_own_session():
    session = {"session_id": "session-1", "title": "Auth fix"}

    result = mcp_server._session_search_result(session)

    assert "owner_email" not in result
    assert "owner_name" not in result


@pytest.mark.asyncio
async def test_file_brief_reshapes_search_result(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    seen_params = {}

    async def fake_request(self, method, path, *, params=None):
        seen_params.update(params or {})
        return {
            "sessions": [
                {"session_id": "s1", "title": "Fixed auth", "session_summary": "..."},
                {"session_id": "s2", "title": "Refactored auth", "session_summary": "..."},
            ],
            "scope": "team",
        }

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    result = await mcp_server.ReclaimLLMClient().file_brief("src/auth.tsx", limit=5, scope="team")

    assert seen_params["file_path"] == "src/auth.tsx"
    assert seen_params["scope"] == "team"
    assert "text_query" not in seen_params
    assert result["path"] == "src/auth.tsx"
    assert result["touch_count"] == 2
    assert result["scope"] == "team"
    assert len(result["sessions"]) == 2


@pytest.mark.asyncio
async def test_file_brief_no_touches(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    async def fake_request(self, method, path, *, params=None):
        return {"sessions": []}

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    result = await mcp_server.ReclaimLLMClient().file_brief("new_file.py", limit=5, scope=None)

    assert result["touch_count"] == 0
    assert result["sessions"] == []


@pytest.mark.asyncio
async def test_handoff_wraps_export_context(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    seen = {}

    async def fake_request(self, method, path, *, params=None):
        seen["method"] = method
        seen["path"] = path
        seen["params"] = params
        return {"context_document": "## Task Overview\n...", "token_estimate": 1200}

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    result = await mcp_server.ReclaimLLMClient().handoff(
        "session-1", include_diffs=True, max_diff_lines=50
    )

    assert seen["path"] == "/api/sessions/session-1/export-context"
    assert result["session_id"] == "session-1"
    assert result["handoff_document"] == "## Task Overview\n..."
    assert result["token_estimate"] == 1200
    assert "instructions" in result


def test_missing_credentials_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "missing.json")
    monkeypatch.delenv("RECLAIMLLM_SERVER_URL", raising=False)
    monkeypatch.delenv("RECLAIMLLM_API_KEY", raising=False)

    with pytest.raises(mcp_server.ReclaimLLMError) as exc_info:
        mcp_server._load_credentials()

    assert "rclm-login" in str(exc_info.value)


class _FakeAuthFailureResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    async def text(self) -> str:
        return "unauthorized"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeAuthFailureSession:
    def __init__(self, status: int) -> None:
        self._status = status

    def request(self, method, url, params=None):
        return _FakeAuthFailureResponse(self._status)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_request_raises_clear_message_on_401(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "revoked-key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        mcp_server.aiohttp, "ClientSession", lambda **kwargs: _FakeAuthFailureSession(401)
    )

    client = mcp_server.ReclaimLLMClient()
    with pytest.raises(mcp_server.ReclaimLLMError) as exc_info:
        await client._request("GET", "/api/sessions")

    assert "rclm-login" in str(exc_info.value)
