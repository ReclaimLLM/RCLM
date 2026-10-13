from __future__ import annotations

from pathlib import Path

from rclm.hooks import bootstrap


def test_policy_snapshot_v2_fields(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("rclm._device.DEVICE_PATH", tmp_path / "device.json")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    snap = bootstrap.policy_snapshot("claude")
    assert snap["snapshot_schema_version"] == 2
    assert "dlp_enabled" in snap
    assert "dlp_local_setting" in snap
    assert "device_id" in snap
    assert "integrity_hash" in snap
    assert "routing" in snap
