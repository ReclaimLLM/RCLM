from __future__ import annotations

import json

from rclm import _config


def test_compression_config_reads_legacy_flat_keys():
    settings = _config.compression_config(
        {"compress": False, "dedupe": True, "compression_thresholds": {"min_dedupe_chars": 700}}
    )
    assert settings["enabled"] is False
    assert settings["dedupe"] is True
    assert settings["min_dedupe_chars"] == 700


def test_save_migrates_flat_compression_keys(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"compress": True, "dedupe": True}))
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    _config.save(
        "https://example.test",
        "key",
        compression={
            "enabled": True,
            "dedupe": False,
            "test_filter": True,
            "min_dedupe_chars": 500,
            "test_filter_max_chars": 8000,
        },
    )
    saved = json.loads(config_path.read_text())
    assert "compress" not in saved
    assert "dedupe" not in saved
    assert saved["compression"]["dedupe"] is False
