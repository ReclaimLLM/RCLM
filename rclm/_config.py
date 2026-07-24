"""Shared access to ~/.reclaimllm/config.json.

Written by rclm-hooks-install; read by the uploader at upload time.
Env vars RECLAIMLLM_SERVER_URL / RECLAIMLLM_API_KEY always take precedence.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".reclaimllm" / "config.json"

DEFAULT_COMPRESSION_CONFIG = {
    "enabled": True,
    "dedupe": False,
    "test_filter": True,
    "min_dedupe_chars": 500,
    "test_filter_max_chars": 8_000,
}


def load() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def compression_config(cfg: dict | None = None) -> dict:
    """Return nested compression settings with flat-key compatibility.

    Existing installations used ``compress``, ``dedupe``, and
    ``compression_thresholds`` at the top level. Read them only as fallbacks;
    new installs persist the single ``compression`` object.
    """
    cfg = cfg or load()
    nested = cfg.get("compression")
    result = dict(DEFAULT_COMPRESSION_CONFIG)
    if isinstance(nested, dict):
        result.update({key: value for key, value in nested.items() if key in result})
        return result
    result["enabled"] = bool(cfg.get("compress", result["enabled"]))
    result["dedupe"] = bool(cfg.get("dedupe", result["dedupe"]))
    thresholds = cfg.get("compression_thresholds")
    if isinstance(thresholds, dict):
        for key in ("min_dedupe_chars", "test_filter_max_chars"):
            if key in thresholds:
                result[key] = thresholds[key]
    return result


def save(server_url: str, api_key: str, **extra: object) -> None:
    existing = load()
    if "compression" in extra:
        for key in ("compress", "dedupe", "compression_thresholds"):
            existing.pop(key, None)
    existing.update({"server_url": server_url, "api_key": api_key, **extra})
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(existing, indent=2),
        encoding="utf-8",
    )


def patch(**fields: object) -> None:
    """Update specific fields in config.json without requiring server_url/api_key.

    Used by the update checker to persist last_update_check and latest_version.
    """
    existing = load()
    existing.update(fields)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(existing, indent=2),
        encoding="utf-8",
    )
