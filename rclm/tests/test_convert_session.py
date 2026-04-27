"""Tests for rclm.convert (rclm convert-session command)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from rclm import _config
from rclm.convert import convert_session

_SESSION_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
_CONTEXT_DOC = "# Prior Session\n\nDo the thing."
_SUCCESS_BODY = json.dumps({"context_document": _CONTEXT_DOC})


# ---------------------------------------------------------------------------
# Minimal aiohttp mocks (avoid AsyncMock magic for clarity)
# ---------------------------------------------------------------------------


class _MockResp:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.ok = 200 <= status < 400
        self._body = body
        self.calls: list[dict] = []  # unused here, but symmetrical with session

    async def text(self) -> str:
        return self._body

    async def __aenter__(self) -> _MockResp:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass


class _MockSession:
    def __init__(self, resp: _MockResp) -> None:
        self._resp = resp
        self.calls: list[dict] = []

    def post(self, url: str, **kwargs: object) -> _MockResp:
        self.calls.append({"url": url, **kwargs})
        return self._resp

    async def __aenter__(self) -> _MockSession:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass


def _mock_http(status: int, body: str):
    """Return (patch-ctx, session) for a single HTTP response."""
    resp = _MockResp(status, body)
    session = _MockSession(resp)
    ctx = patch("rclm.convert.aiohttp.ClientSession", return_value=session)
    return ctx, session


# ---------------------------------------------------------------------------
# Success: stdout and file output
# ---------------------------------------------------------------------------


def test_prints_context_to_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://test.local")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "test-key")

    ctx, _ = _mock_http(200, _SUCCESS_BODY)
    with ctx:
        convert_session(_SESSION_ID, "generic")

    assert _CONTEXT_DOC in capsys.readouterr().out


def test_writes_context_to_file(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://test.local")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "test-key")
    dest = tmp_path / "session.md"

    ctx, _ = _mock_http(200, _SUCCESS_BODY)
    with ctx:
        convert_session(_SESSION_ID, "claude", output_path=str(dest))

    assert dest.read_text(encoding="utf-8") == _CONTEXT_DOC


def test_stdout_path_does_not_create_files(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://test.local")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "test-key")

    ctx, _ = _mock_http(200, _SUCCESS_BODY)
    with ctx:
        convert_session(_SESSION_ID, "generic")

    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


def test_reads_credentials_from_config_file(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"server_url": "http://from-config.local", "api_key": "cfg-key"})
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.delenv("RECLAIMLLM_SERVER_URL", raising=False)
    monkeypatch.delenv("RECLAIMLLM_API_KEY", raising=False)

    ctx, _ = _mock_http(200, _SUCCESS_BODY)
    with ctx:
        convert_session(_SESSION_ID, "generic")

    assert _CONTEXT_DOC in capsys.readouterr().out


def test_env_var_overrides_config_server_url(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"server_url": "http://should-not-use.local", "api_key": "cfg-key"})
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://env-wins.local")
    monkeypatch.delenv("RECLAIMLLM_API_KEY", raising=False)

    ctx, session = _mock_http(200, _SUCCESS_BODY)
    with ctx:
        convert_session(_SESSION_ID, "generic")

    assert session.calls[0]["url"].startswith("http://env-wins.local")


def test_env_var_overrides_config_api_key(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "http://test.local", "api_key": "old-key"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.delenv("RECLAIMLLM_SERVER_URL", raising=False)
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "new-key")

    ctx, session = _mock_http(200, _SUCCESS_BODY)
    with ctx:
        convert_session(_SESSION_ID, "generic")

    assert session.calls[0]["headers"]["X-API-Key"] == "new-key"


# ---------------------------------------------------------------------------
# Missing credentials
# ---------------------------------------------------------------------------


def test_no_server_url_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.delenv("RECLAIMLLM_SERVER_URL", raising=False)
    monkeypatch.delenv("RECLAIMLLM_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc:
        convert_session(_SESSION_ID, "generic")

    assert exc.value.code == 1
    assert "server URL" in capsys.readouterr().err


def test_no_api_key_exits_1(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"server_url": "http://test.local"}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.delenv("RECLAIMLLM_SERVER_URL", raising=False)
    monkeypatch.delenv("RECLAIMLLM_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc:
        convert_session(_SESSION_ID, "generic")

    assert exc.value.code == 1
    assert "API key" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------


def test_404_exits_1_with_not_found_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://test.local")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "test-key")

    ctx, _ = _mock_http(404, "")
    with ctx, pytest.raises(SystemExit) as exc:
        convert_session(_SESSION_ID, "generic")

    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().err


def test_503_with_json_detail_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://test.local")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "test-key")

    ctx, _ = _mock_http(503, json.dumps({"detail": "No LLM backend configured"}))
    with ctx, pytest.raises(SystemExit) as exc:
        convert_session(_SESSION_ID, "generic")

    assert exc.value.code == 1
    assert "No LLM backend configured" in capsys.readouterr().err


def test_503_non_json_body_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://test.local")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "test-key")

    ctx, _ = _mock_http(503, "Service Unavailable")
    with ctx, pytest.raises(SystemExit) as exc:
        convert_session(_SESSION_ID, "generic")

    assert exc.value.code == 1


def test_401_exits_1_with_detail(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://test.local")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "test-key")

    ctx, _ = _mock_http(401, json.dumps({"detail": "Unauthorized"}))
    with ctx, pytest.raises(SystemExit) as exc:
        convert_session(_SESSION_ID, "generic")

    assert exc.value.code == 1
    assert "Unauthorized" in capsys.readouterr().err


def test_invalid_json_response_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://test.local")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "test-key")

    ctx, _ = _mock_http(200, "THIS IS NOT JSON")
    with ctx, pytest.raises(SystemExit) as exc:
        convert_session(_SESSION_ID, "generic")

    assert exc.value.code == 1
    assert "unexpected response" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Query param and header passthrough
# ---------------------------------------------------------------------------


def test_url_contains_session_id(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://test.local")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "test-key")

    ctx, session = _mock_http(200, _SUCCESS_BODY)
    with ctx:
        convert_session(_SESSION_ID, "generic")

    assert _SESSION_ID in session.calls[0]["url"]


def test_passes_target_tool_param(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://test.local")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "test-key")

    ctx, session = _mock_http(200, _SUCCESS_BODY)
    with ctx:
        convert_session(_SESSION_ID, "gemini")

    assert session.calls[0]["params"]["target_tool"] == "gemini"


def test_passes_include_diffs_false(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://test.local")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "test-key")

    ctx, session = _mock_http(200, _SUCCESS_BODY)
    with ctx:
        convert_session(_SESSION_ID, "generic", include_diffs=False)

    assert session.calls[0]["params"]["include_diffs"] == "false"


def test_passes_force_regenerate_true(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://test.local")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "test-key")

    ctx, session = _mock_http(200, _SUCCESS_BODY)
    with ctx:
        convert_session(_SESSION_ID, "generic", force_regenerate=True)

    assert session.calls[0]["params"]["force_regenerate"] == "true"


def test_passes_max_diff_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://test.local")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "test-key")

    ctx, session = _mock_http(200, _SUCCESS_BODY)
    with ctx:
        convert_session(_SESSION_ID, "generic", max_diff_lines=100)

    assert session.calls[0]["params"]["max_diff_lines"] == "100"


def test_sets_api_key_header(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "http://test.local")
    monkeypatch.setenv("RECLAIMLLM_API_KEY", "my-secret-key")

    ctx, session = _mock_http(200, _SUCCESS_BODY)
    with ctx:
        convert_session(_SESSION_ID, "generic")

    assert session.calls[0]["headers"]["X-API-Key"] == "my-secret-key"
