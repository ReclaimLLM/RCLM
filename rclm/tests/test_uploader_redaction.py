from __future__ import annotations

import json

import pytest

from rclm import _config
from rclm._models import HookSessionRecord
from rclm._uploader import upload


def _record(cwd: str = "/tmp/project") -> HookSessionRecord:
    return HookSessionRecord(
        session_id="00000000-0000-0000-0000-000000000001",
        cwd=cwd,
        started_at="2026-04-27T00:00:00+00:00",
        ended_at="2026-04-27T00:01:00+00:00",
        duration_s=60.0,
        transcript_path=None,
        model="test-model",
        messages=[{"role": "user", "content": "secret"}],
    )


class _Response:
    status = 201

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Session:
    def __init__(self) -> None:
        self.posts: list[dict] = []

    def post(self, url: str, *, data: str, headers: dict):
        self.posts.append({"url": url, "data": data, "headers": headers})
        return _Response()


@pytest.mark.asyncio
async def test_upload_redacts_payload_before_post(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    _config.patch(
        server_url="https://api.example.test",
        api_key="key",
        redaction={
            "enabled": True,
            "remote_substitutions": {"secret": "[REDACTED]"},
            "local_substitutions": {},
            "exclude_folders": [],
            "last_sync": None,
        },
    )
    session = _Session()

    await upload(_record(), session)

    sent = json.loads(session.posts[0]["data"])
    assert sent["messages"][0]["content"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_upload_skips_excluded_folder(tmp_path, monkeypatch):
    project = tmp_path / "private"
    project.mkdir()
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    _config.patch(
        server_url="https://api.example.test",
        api_key="key",
        redaction={
            "enabled": True,
            "remote_substitutions": {"secret": "[REDACTED]"},
            "local_substitutions": {},
            "exclude_folders": [str(project)],
            "last_sync": None,
        },
    )
    session = _Session()

    await upload(_record(str(project / "repo")), session)

    assert session.posts == []


@pytest.mark.asyncio
async def test_upload_prefers_config_server_url_over_env(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "https://env.example.test")
    monkeypatch.setenv("BACKEND_SERVER", "https://legacy-env.example.test")
    _config.patch(
        server_url="https://config.example.test",
        api_key="key",
        redaction={
            "enabled": True,
            "remote_substitutions": {},
            "local_substitutions": {},
            "exclude_folders": [],
            "last_sync": None,
        },
    )
    session = _Session()

    await upload(_record(), session)

    assert session.posts[0]["url"] == "https://config.example.test/api/ingest"
