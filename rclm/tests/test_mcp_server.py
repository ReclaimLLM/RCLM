import hashlib
import json

import pytest

from rclm import _config, mcp_server


def test_load_credentials_prefers_env_over_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"server_url": "https://config.test", "api_key": "config-key"})
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "https://env.test")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "env-key")

    creds = mcp_server._load_credentials()

    assert creds.server_url == "https://env.test"
    assert creds.api_key == "env-key"


def test_load_credentials_reads_config_when_env_unset(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"server_url": "https://config.test", "api_key": "config-key"})
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.delenv("RECLAIMLLM_SERVER_URL", raising=False)
    monkeypatch.delenv("RECLAIMLLM_API_KEY", raising=False)

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
            "record_type": "session",
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
async def test_get_session_rejects_non_session_record(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    async def fake_request(self, method, path, *, params=None):
        return {"session_id": "proxy-1", "record_type": "proxy"}

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    with pytest.raises(mcp_server.ReclaimLLMError, match="record_type=session only"):
        await mcp_server.ReclaimLLMClient().get_session("proxy-1")


@pytest.mark.asyncio
async def test_filter_sessions_uses_postgres_without_text_or_record_type_params(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    seen_params = {}

    async def fake_request(self, method, path, *, params=None):
        assert method == "GET"
        assert path == "/api/sessions/filter"
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

    result = await mcp_server.ReclaimLLMClient().filter_sessions(
        project_name=None,
        file_path="auth.tsx",
        limit=8,
        min_turns=5,
        min_tool_calls=5,
        include_changed_files=True,
    )

    assert "text_query" not in seen_params
    assert seen_params["file_path"] == "auth.tsx"
    assert seen_params["include_changed_files"] == "true"
    assert seen_params["min_turns"] == 5
    assert seen_params["min_tool_calls"] == 5
    assert "record_type" not in seen_params
    assert result["sessions"][0]["title"] == "Touched auth.tsx"


@pytest.mark.asyncio
async def test_search_sessions_passes_scope_param(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    seen_params = {}

    async def fake_request(self, method, path, *, params=None):
        assert path == "/api/sessions/search"
        seen_params.update(params or {})
        return {"sessions": [], "scope": "team"}

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    result = await mcp_server.ReclaimLLMClient().search_sessions(
        "auth bug",
        project_name=None,
        file_path=None,
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
        limit=5,
    )

    assert "scope" not in seen_params


@pytest.mark.asyncio
async def test_search_sessions_passes_ingestion_date_range(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    seen_params = {}

    async def fake_request(self, method, path, *, params=None):
        seen_params.update(params or {})
        return {"sessions": []}

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    await mcp_server.ReclaimLLMClient().filter_sessions(
        project_name=None,
        file_path="src/auth.tsx",
        limit=5,
        date_from="2026-07-08",
        date_to="2026-07-30",
    )

    assert seen_params["date_from"] == "2026-07-08"
    assert seen_params["date_to"] == "2026-07-30"


@pytest.mark.asyncio
async def test_search_sessions_rejects_empty_text_query(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    with pytest.raises(mcp_server.ReclaimLLMError, match="query is required"):
        await mcp_server.ReclaimLLMClient().search_sessions(
            "   ",
            project_name=None,
            file_path=None,
            limit=5,
        )


@pytest.mark.parametrize(
    ("date_from", "date_to", "message"),
    [
        ("07/08/2026", None, "date_from must be an ISO date"),
        ("2026-07-30", "2026-07-30", "date_from must be earlier"),
        ("2026-07-31", "2026-07-30", "date_from must be earlier"),
    ],
)
def test_validate_date_range_rejects_invalid_or_empty_windows(date_from, date_to, message):
    with pytest.raises(mcp_server.ReclaimLLMError, match=message):
        mcp_server._validated_date_range(date_from, date_to)


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


def test_session_search_result_includes_changed_files_for_history_followup():
    session = {
        "session_id": "session-1",
        "title": "Auth fix",
        "changed_files": [
            {
                "file_path": "src/auth.tsx",
                "operation": "edit",
                "is_new_file": False,
                "lines_added": 8,
                "lines_removed": 2,
            }
        ],
        "changed_files_total": 4,
        "changed_files_truncated": True,
    }

    result = mcp_server._session_search_result(session)

    assert result["changed_files"][0]["file_path"] == "src/auth.tsx"
    assert result["changed_files_total"] == 4
    assert result["changed_files_truncated"] is True


@pytest.mark.asyncio
async def test_file_brief_reshapes_search_result(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    seen_params = {}

    async def fake_request(self, method, path, *, params=None):
        if path == "/api/signals/file-brief":
            return {"signal": None}
        assert path == "/api/sessions/filter"
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
async def test_most_recent_complete_session_uses_filter_endpoint(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    seen_paths = []

    async def fake_request(self, method, path, *, params=None):
        seen_paths.append(path)
        if path == "/api/sessions/filter":
            return {
                "sessions": [
                    {
                        "session_id": "session-1",
                        "record_type": "session",
                    }
                ]
            }
        return {
            "session_id": "session-1",
            "record_type": "session",
            "ended_at": "2026-08-01T12:00:00Z",
        }

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    session_id = await mcp_server.ReclaimLLMClient().most_recent_complete_session_id()

    assert session_id == "session-1"
    assert seen_paths[0] == "/api/sessions/filter"


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
async def test_file_brief_carries_open_p2_signal(tmp_path, monkeypatch):
    """PRD §6.5.1: file_brief enrichment -- the agent sees "read in 31
    sessions across 4 contributors" exactly when it's about to re-read the
    file itself."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    async def fake_request(self, method, path, *, params=None):
        if path == "/api/signals/file-brief":
            return {
                "signal": {
                    "path": "server/routes/sessions.py",
                    "session_count": 31,
                    "contributor_count": 4,
                    "summary": "read in 31 sessions across 4 contributor(s) -- a structural summary in CLAUDE.md is recommended.",
                }
            }
        return {"sessions": [], "scope": None}

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    result = await mcp_server.ReclaimLLMClient().file_brief(
        "server/routes/sessions.py", limit=5, scope=None
    )

    assert result["signal"]["session_count"] == 31
    assert result["signal"]["contributor_count"] == 4


@pytest.mark.asyncio
async def test_file_brief_survives_signal_lookup_failure(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    async def fake_request(self, method, path, *, params=None):
        if path == "/api/signals/file-brief":
            raise mcp_server.ReclaimLLMError("boom")
        return {"sessions": [], "scope": None}

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    result = await mcp_server.ReclaimLLMClient().file_brief("file.py", limit=5, scope=None)

    assert result["signal"] is None
    assert result["touch_count"] == 0


@pytest.mark.asyncio
async def test_signals_tool_returns_shaped_list(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    seen = {}

    async def fake_request(self, method, path, *, params=None):
        seen["path"] = path
        seen["params"] = params
        return {
            "items": [
                {
                    "pattern": "P2",
                    "scope_type": "project",
                    "evidence": {"top_files": []},
                    "projected_savings": None,
                }
            ]
        }

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    result = await mcp_server.ReclaimLLMClient().signals(cwd="/some/project")

    assert seen["path"] == "/api/signals/for-session"
    assert seen["params"] == {"cwd": "/some/project"}
    assert result["signals"][0]["pattern"] == "P2"
    assert result["signals"][0]["scope"] == "project"


@pytest.mark.asyncio
async def test_handoff_wraps_export_context(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    calls = []

    async def fake_request(self, method, path, *, params=None):
        calls.append({"method": method, "path": path, "params": params})
        if method == "GET" and path == "/api/sessions/session-1":
            return {"session_id": "session-1", "record_type": "session"}
        if path == "/api/signals/mark-acted":
            return {}
        return {"context_document": "## Task Overview\n...", "token_estimate": 1200}

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    result = await mcp_server.ReclaimLLMClient().handoff(
        "session-1", include_diffs=True, max_diff_lines=50
    )

    assert calls[1]["path"] == "/api/sessions/session-1/export-context"
    assert result["session_id"] == "session-1"
    assert result["handoff_document"] == "## Task Overview\n..."
    assert result["token_estimate"] == 1200
    assert "instructions" in result


@pytest.mark.asyncio
async def test_handoff_marks_signals_acted(tmp_path, monkeypatch):
    """handoff() closes the loop on whether its prescribed fix got used
    (PRD §6.5.2): after export-context succeeds, it fires a mark-acted call
    for the same session_id."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    calls = []

    async def fake_request(self, method, path, *, params=None):
        calls.append({"method": method, "path": path, "params": params})
        if method == "GET" and path == "/api/sessions/session-2":
            return {"session_id": "session-2", "record_type": "session"}
        if path == "/api/signals/mark-acted":
            return {}
        return {"context_document": "doc", "token_estimate": 10}

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    await mcp_server.ReclaimLLMClient().handoff("session-2", include_diffs=False, max_diff_lines=10)

    mark_acted_calls = [c for c in calls if c["path"] == "/api/signals/mark-acted"]
    assert len(mark_acted_calls) == 1
    assert mark_acted_calls[0]["method"] == "POST"
    assert mark_acted_calls[0]["params"] == {"session_id": "session-2"}


@pytest.mark.asyncio
async def test_handoff_survives_mark_acted_failure(tmp_path, monkeypatch):
    """A failing mark-acted call must never break the handoff response the
    user is waiting on."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)

    async def fake_request(self, method, path, *, params=None):
        if method == "GET" and path == "/api/sessions/session-3":
            return {"session_id": "session-3", "record_type": "session"}
        if path == "/api/signals/mark-acted":
            raise mcp_server.ReclaimLLMError("boom")
        return {"context_document": "doc", "token_estimate": 10}

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "_request", fake_request)

    result = await mcp_server.ReclaimLLMClient().handoff(
        "session-3", include_diffs=False, max_diff_lines=10
    )

    assert result["session_id"] == "session-3"
    assert result["handoff_document"] == "doc"


class _FakeTransferContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def iter_chunked(self, _size: int):
        async def iterator():
            for chunk in self._chunks:
                yield chunk

        return iterator()


class _FakeTransferResponse:
    def __init__(self, content: bytes) -> None:
        digest = hashlib.sha256(content).hexdigest()
        self.status = 200
        self.content_length = len(content)
        self.content = _FakeTransferContent([content[:7], content[7:]])
        self.headers = {
            "X-ReclaimLLM-Transfer-Schema": "reclaimllm.session-transfer.v1",
            "X-ReclaimLLM-Transfer-SHA256": digest,
            "X-ReclaimLLM-Transfer-Bytes": str(len(content)),
            "X-ReclaimLLM-Transfer-Token-Estimate": "42",
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeTransferSession:
    def __init__(self, content: bytes) -> None:
        self._content = content

    def get(self, _url: str):
        return _FakeTransferResponse(self._content)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_transfer_session_streams_verified_artifact(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "https://api.test", "api_key": "key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    content = b'{"schema_version":"reclaimllm.session-transfer.v1","payload":{}}'
    session_kwargs = {}

    def fake_client_session(**kwargs):
        session_kwargs.update(kwargs)
        return _FakeTransferSession(content)

    monkeypatch.setattr(mcp_server.aiohttp, "ClientSession", fake_client_session)
    real_write_transfer_stream = mcp_server.write_transfer_stream

    async def write_to_test_root(chunks, *, max_bytes):
        return await real_write_transfer_stream(
            chunks,
            root=tmp_path / "transfers",
            max_bytes=max_bytes,
        )

    monkeypatch.setattr(mcp_server, "write_transfer_stream", write_to_test_root)

    async def session_metadata(self, session_id: str):
        return {"session_id": session_id, "record_type": "session"}

    monkeypatch.setattr(mcp_server.ReclaimLLMClient, "fetch_session_metadata", session_metadata)

    result = await mcp_server.ReclaimLLMClient().transfer_session("session-1")

    assert result["complete"] is True
    assert result["schema_version"] == "reclaimllm.session-transfer.v1"
    assert result["token_estimate"] == 42
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert session_kwargs["timeout"].total == 600
    with open(result["artifact_path"], "rb") as artifact_file:
        assert artifact_file.read() == content


@pytest.mark.asyncio
async def test_transfer_session_is_registered_as_mcp_tool():
    tools = await mcp_server.build_mcp_server().list_tools()

    assert "transfer_session" in {tool.name for tool in tools}


@pytest.mark.asyncio
async def test_search_tools_do_not_expose_record_type_override():
    tools = {tool.name: tool for tool in await mcp_server.build_mcp_server().list_tools()}

    for name in ("search_sessions", "filter_sessions", "search_by_filename"):
        assert "record_type" not in tools[name].inputSchema.get("properties", {})


@pytest.mark.asyncio
async def test_filter_sessions_is_registered_as_no_query_postgres_tool():
    tools = {tool.name: tool for tool in await mcp_server.build_mcp_server().list_tools()}

    schema = tools["filter_sessions"].inputSchema
    properties = schema["properties"]
    assert "query" not in properties
    assert "text_query" not in properties
    assert properties["limit"]["default"] == 50
    assert {
        "provider",
        "model",
        "model_family",
        "date_from",
        "date_to",
        "min_turns",
        "min_tool_calls",
    } <= properties.keys()


@pytest.mark.asyncio
async def test_filter_sessions_tool_calls_postgres_client_with_metadata_filters(monkeypatch):
    captured = {}
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "https://api.test")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "test-key")

    async def fake_filter_sessions(self, **kwargs):
        captured.update(kwargs)
        return {"sessions": [], "scope": "mine"}

    monkeypatch.setattr(
        mcp_server.ReclaimLLMClient,
        "filter_sessions",
        fake_filter_sessions,
    )

    await mcp_server.build_mcp_server().call_tool(
        "filter_sessions",
        {
            "provider": "codex",
            "date_from": "2026-07-24",
            "date_to": "2026-08-09",
            "scope": "mine",
            "limit": 100,
            "min_turns": 5,
            "min_tool_calls": 5,
        },
    )

    assert captured["provider"] == "codex"
    assert captured["date_from"] == "2026-07-24"
    assert captured["date_to"] == "2026-08-09"
    assert captured["scope"] == "mine"
    assert captured["limit"] == 100
    assert captured["min_turns"] == 5
    assert captured["min_tool_calls"] == 5
    assert captured["include_changed_files"] is False


@pytest.mark.asyncio
async def test_replay_tools_default_minimums_to_five():
    tools = {tool.name: tool for tool in await mcp_server.build_mcp_server().list_tools()}

    for name in ("replay_eligibility", "replay_session", "replay_corpus", "replay_compare"):
        properties = tools[name].inputSchema["properties"]
        assert properties["min_turns"]["default"] == 5
        assert properties["min_tool_calls"]["default"] == 5


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
    session_kwargs = {}

    def fake_client_session(**kwargs):
        session_kwargs.update(kwargs)
        return _FakeAuthFailureSession(401)

    monkeypatch.setattr(mcp_server.aiohttp, "ClientSession", fake_client_session)

    client = mcp_server.ReclaimLLMClient()
    with pytest.raises(mcp_server.ReclaimLLMError) as exc_info:
        await client._request("GET", "/api/sessions")

    assert "rclm-login" in str(exc_info.value)
    assert session_kwargs["timeout"].total == 600
