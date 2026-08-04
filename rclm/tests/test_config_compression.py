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


def test_org_force_enforcement_overrides_legacy_shadow_mode():
    policy = _config.effective_hook_policy(
        {
            "shadow_mode": True,
            "read_cache": True,
            "compression": {"enabled": True, "dedupe": True, "test_filter": True},
            "org_hook_policy": {
                "force_compression_enforcement": True,
                "policy_version": 4,
                "compression_modes": {},
            },
        },
        provider="claude",
    )

    assert policy.legacy_shadow is False
    assert policy.policy_version == 4
    assert policy.mechanisms["range_cache"]["mode"] == "enforce"
    assert policy.mechanisms["hash_dedupe"]["mode"] == "enforce"


def test_codex_unsupported_image_rewrite_stays_observe_only():
    policy = _config.effective_hook_policy(
        {
            "image_lifecycle": True,
            "org_hook_policy": {"force_compression_enforcement": True},
        },
        provider="codex",
    )

    assert policy.mechanisms["image_downscale"] == {
        "enabled": True,
        "mode": "observe",
        "supported": False,
    }


def test_codex_and_cursor_support_exec_compaction():
    config = {"compression": {"enabled": True}}

    for provider in ("codex", "cursor"):
        policy = _config.effective_hook_policy(config, provider=provider)
        assert policy.mechanisms["exec_compaction"] == {
            "enabled": True,
            "mode": "enforce",
            "supported": True,
        }
