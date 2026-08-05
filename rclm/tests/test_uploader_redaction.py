from __future__ import annotations

import json

import pytest

from rclm import _config, _uploader
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
async def test_upload_skips_outside_include_folder(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    _config.patch(
        server_url="https://api.example.test",
        api_key="key",
        redaction={
            "enabled": True,
            "remote_substitutions": {},
            "local_substitutions": {},
            "include_folders": [str(allowed)],
            "exclude_folders": [],
            "last_sync": None,
        },
    )
    session = _Session()

    await upload(_record(str(tmp_path / "other")), session)

    assert session.posts == []


@pytest.mark.asyncio
async def test_upload_include_folder_supersedes_exclude_folder(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    _config.patch(
        server_url="https://api.example.test",
        api_key="key",
        redaction={
            "enabled": True,
            "remote_substitutions": {},
            "local_substitutions": {},
            "include_folders": [str(allowed)],
            "exclude_folders": [str(allowed)],
            "last_sync": None,
        },
    )
    session = _Session()

    await upload(_record(str(allowed / "repo")), session)

    assert len(session.posts) == 1


@pytest.mark.asyncio
async def test_upload_prefers_env_server_url_over_config(tmp_path, monkeypatch):
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

    assert session.posts[0]["url"] == "https://env.example.test/api/ingest"


@pytest.mark.asyncio
async def test_upload_missing_credentials_quarantines_with_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.delenv("RECLAIMLLM_SERVER_URL", raising=False)
    monkeypatch.delenv("RECLAIMLLM_API_KEY", raising=False)
    monkeypatch.setattr(_uploader, "_FAILED_UPLOADS_DIR", tmp_path / "failed_uploads")
    session = _Session()
    record = _record()

    await upload(record, session)

    assert session.posts == []
    quarantined = tmp_path / "failed_uploads" / f"{record.session_id}.json"
    assert quarantined.exists()
    err = capsys.readouterr().err
    assert "not authenticated" in err
    assert "rclm-login" in err
