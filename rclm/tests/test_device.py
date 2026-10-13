from __future__ import annotations

import json
from pathlib import Path

from rclm import _device


def test_ensure_device_id_stable(tmp_path: Path, monkeypatch):
    path = tmp_path / "device.json"
    monkeypatch.setattr(_device, "DEVICE_PATH", path)

    first = _device.ensure_device_id()
    second = _device.ensure_device_id()
    assert first == second
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["device_id"] == first
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600 or mode == 0o644  # some CI umasks loosen chmod


def test_hostname_hash_changes_with_salt(monkeypatch):
    monkeypatch.setattr(_device.socket, "gethostname", lambda: "dev-box")
    a = _device.hostname_hash()
    b = _device.hostname_hash(org_salt="salt")
    assert a != b
    assert len(a) == 64


def test_environment_ephemeral(monkeypatch):
    monkeypatch.delenv("CODESPACE_NAME", raising=False)
    assert _device.environment_class() == "durable"
    monkeypatch.setenv("CODESPACE_NAME", "nice-space")
    assert _device.environment_class() == "ephemeral"
