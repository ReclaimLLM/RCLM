from __future__ import annotations

import json
from pathlib import Path

from rclm import _config
from rclm._models import HookSessionRecord
from rclm.hooks import redaction


def _record(cwd: str) -> HookSessionRecord:
    return HookSessionRecord(
        session_id="00000000-0000-0000-0000-000000000001",
        cwd=cwd,
        started_at="2026-04-27T00:00:00+00:00",
        ended_at="2026-04-27T00:01:00+00:00",
        duration_s=60.0,
        transcript_path=None,
        model="test-model",
        messages=[{"role": "user", "content": "secret and local-secret"}],
    )


def test_local_substitutions_override_remote():
    settings = redaction.RedactionSettings(
        enabled=True,
        remote_substitutions={"secret": "[REMOTE]", "remote-only": "[REMOTE_ONLY]"},
        local_substitutions={"secret": "[LOCAL]"},
        exclude_folders=[],
    )

    result = redaction.redact_json_payload(
        "secret remote-only local-secret",
        settings,
    )

    assert result == "[LOCAL] [REMOTE_ONLY] local-[LOCAL]"


def test_redaction_disabled_returns_payload_unchanged():
    settings = redaction.RedactionSettings(
        enabled=False,
        remote_substitutions={"secret": "[REDACTED]"},
        local_substitutions={},
        exclude_folders=[],
    )

    assert redaction.redact_json_payload("secret", settings) == "secret"


def test_should_skip_record_inside_excluded_folder(tmp_path: Path):
    project = tmp_path / "private"
    project.mkdir()
    settings = redaction.RedactionSettings(
        enabled=True,
        remote_substitutions={},
        local_substitutions={},
        exclude_folders=[str(project)],
    )

    assert redaction.should_skip_record(_record(str(project / "repo")), settings)


def test_should_not_skip_record_outside_excluded_folder(tmp_path: Path):
    settings = redaction.RedactionSettings(
        enabled=True,
        remote_substitutions={},
        local_substitutions={},
        exclude_folders=[str(tmp_path / "private")],
    )

    assert not redaction.should_skip_record(_record(str(tmp_path / "public")), settings)


def test_ensure_settings_writes_missing_default_keys(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    _config.patch(server_url="https://api.example.test", api_key="key")

    redaction.ensure_settings()

    cfg = _config.load()["redaction"]
    assert cfg == {
        "enabled": True,
        "remote_substitutions": {},
        "local_substitutions": {},
        "exclude_folders": [],
        "last_sync": None,
    }


def test_sync_remote_settings_seeds_defaults_when_credentials_missing(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    _config.patch(server_url="https://api.example.test")

    assert not redaction.sync_remote_settings()

    cfg = _config.load()["redaction"]
    assert cfg["enabled"] is True
    assert cfg["remote_substitutions"] == {}
    assert cfg["local_substitutions"] == {}
    assert cfg["exclude_folders"] == []
    assert cfg["last_sync"] is None


def test_sync_remote_settings_preserves_local_fields(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    _config.patch(
        server_url="https://api.example.test",
        api_key="key",
        redaction={
            "enabled": True,
            "remote_substitutions": {"old": "[OLD]"},
            "local_substitutions": {"local": "[LOCAL]"},
            "exclude_folders": ["/private"],
            "last_sync": None,
        },
    )

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "enabled": False,
                    "substitutions": {"remote": "[REMOTE]"},
                    "updated_at": "2026-04-27T00:00:00+00:00",
                }
            ).encode("utf-8")

    monkeypatch.setattr(redaction.urllib.request, "urlopen", lambda req, timeout: _Response())

    assert redaction.sync_remote_settings()

    cfg = _config.load()["redaction"]
    assert cfg["enabled"] is False
    assert cfg["remote_substitutions"] == {"remote": "[REMOTE]"}
    assert cfg["local_substitutions"] == {"local": "[LOCAL]"}
    assert cfg["exclude_folders"] == ["/private"]
    assert cfg["last_sync"]


def test_sync_remote_settings_saves_remote_substitutions(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    _config.patch(
        server_url="https://api.example.test",
        api_key="key",
    )

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "enabled": True,
                    "substitutions": {"secret": "[REDACTED]"},
                    "updated_at": "2026-04-27T00:00:00+00:00",
                }
            ).encode("utf-8")

    monkeypatch.setattr(redaction.urllib.request, "urlopen", lambda req, timeout: _Response())

    assert redaction.sync_remote_settings()

    cfg = _config.load()["redaction"]
    assert cfg["remote_substitutions"] == {"secret": "[REDACTED]"}


def test_sync_remote_settings_accepts_wrapped_redaction_payload(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    _config.patch(
        server_url="https://api.example.test",
        api_key="key",
    )

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "redaction": {
                        "enabled": False,
                        "substitutions": {"secret": "[REDACTED]"},
                    }
                }
            ).encode("utf-8")

    monkeypatch.setattr(redaction.urllib.request, "urlopen", lambda req, timeout: _Response())

    assert redaction.sync_remote_settings()

    cfg = _config.load()["redaction"]
    assert cfg["enabled"] is False
    assert cfg["remote_substitutions"] == {"secret": "[REDACTED]"}


def test_sync_remote_settings_keeps_remote_substitutions_when_payload_omits_them(
    tmp_path: Path, monkeypatch
):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    _config.patch(
        server_url="https://api.example.test",
        api_key="key",
        redaction={
            "enabled": True,
            "remote_substitutions": {"existing": "[EXISTING]"},
            "local_substitutions": {},
            "exclude_folders": [],
            "last_sync": None,
        },
    )

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"enabled": False}).encode("utf-8")

    monkeypatch.setattr(redaction.urllib.request, "urlopen", lambda req, timeout: _Response())

    assert redaction.sync_remote_settings()

    cfg = _config.load()["redaction"]
    assert cfg["enabled"] is False
    assert cfg["remote_substitutions"] == {"existing": "[EXISTING]"}


def test_sync_remote_settings_prefers_config_server_url_over_env(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("RECLAIMLLM_SERVER_URL", "https://env.example.test")
    monkeypatch.setenv("BACKEND_SERVER", "https://legacy-env.example.test")
    _config.patch(
        server_url="https://config.example.test",
        api_key="key",
        redaction=redaction.default_redaction_config(),
    )
    seen_urls: list[str] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"enabled": True, "substitutions": {}}).encode("utf-8")

    def _urlopen(req, timeout):
        seen_urls.append(req.full_url)
        return _Response()

    monkeypatch.setattr(redaction.urllib.request, "urlopen", _urlopen)

    assert redaction.sync_remote_settings()
    assert seen_urls == ["https://config.example.test/api/settings/redaction"]
