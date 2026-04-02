"""Shared access to ~/.reclaimllm/config.json.

Written by rclm-hooks-install; read by the uploader at upload time.
Env vars RECLAIMLLM_SERVER_URL / RECLAIMLLM_API_KEY always take precedence.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".reclaimllm" / "config.json"


def load() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save(server_url: str, api_key: str, **extra: object) -> None:
    existing = load()
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
