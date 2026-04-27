"""Local hook upload redaction settings and helpers."""

from __future__ import annotations

import dataclasses
import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rclm import _config
from rclm._endpoints import REDACTION_SETTINGS_PATH

logger = logging.getLogger(__name__)

_NETWORK_TIMEOUT_S = 30


@dataclasses.dataclass(frozen=True)
class RedactionSettings:
    enabled: bool
    remote_substitutions: dict[str, Any]
    local_substitutions: dict[str, Any]
    exclude_folders: list[str]
    last_sync: str | None = None

    @property
    def substitutions(self) -> dict[str, Any]:
        merged = dict(self.remote_substitutions)
        merged.update(self.local_substitutions)
        return merged


def default_redaction_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "remote_substitutions": {},
        "local_substitutions": {},
        "exclude_folders": [],
        "last_sync": None,
    }


def _normalise_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, str) and k}


def _normalise_folders(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def load_settings(cfg: dict | None = None) -> RedactionSettings:
    if cfg is None:
        cfg = _config.load()
    raw = cfg.get("redaction")
    if not isinstance(raw, dict):
        raw = {}
    defaults = default_redaction_config()
    return RedactionSettings(
        enabled=bool(raw.get("enabled", defaults["enabled"])),
        remote_substitutions=_normalise_mapping(raw.get("remote_substitutions")),
        local_substitutions=_normalise_mapping(raw.get("local_substitutions")),
        exclude_folders=_normalise_folders(raw.get("exclude_folders")),
        last_sync=(raw.get("last_sync") if isinstance(raw.get("last_sync"), str) else None),
    )


def _settings_to_config(settings: RedactionSettings) -> dict[str, Any]:
    return {
        "enabled": settings.enabled,
        "remote_substitutions": settings.remote_substitutions,
        "local_substitutions": settings.local_substitutions,
        "exclude_folders": settings.exclude_folders,
        "last_sync": settings.last_sync,
    }


def _save_settings(settings: RedactionSettings) -> None:
    _config.patch(redaction=_settings_to_config(settings))


def ensure_settings(cfg: dict | None = None) -> RedactionSettings:
    """Persist default redaction config keys while preserving existing local fields."""
    settings = load_settings(cfg)
    _save_settings(settings)
    return settings


def _redaction_response_payload(data: object) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    nested = data.get("redaction")
    if isinstance(nested, dict):
        return nested
    return data


def sync_remote_settings(
    *,
    server_url: str | None = None,
    api_key: str | None = None,
) -> bool:
    """Fetch remote redaction settings and merge them into local config.

    Returns True when sync succeeded. All network/API failures are swallowed so
    update/install flows do not break if the settings endpoint is unavailable.
    """
    cfg = _config.load()
    current = ensure_settings(cfg)
    base = (server_url or cfg.get("server_url") or "").strip()
    key = (api_key or cfg.get("api_key") or "").strip()
    if not base or not key:
        return False

    url = base.rstrip("/") + REDACTION_SETTINGS_PATH
    req = urllib.request.Request(url, headers={"X-API-Key": key, "User-Agent": "rclm-hooks"})

    try:
        with urllib.request.urlopen(req, timeout=_NETWORK_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        logger.debug("rclm redaction settings sync failed", exc_info=True)
        return False

    payload = _redaction_response_payload(data)
    remote_substitutions = (
        _normalise_mapping(payload["substitutions"])
        if isinstance(payload.get("substitutions"), dict)
        else current.remote_substitutions
    )
    _save_settings(
        RedactionSettings(
            enabled=bool(payload.get("enabled", True)),
            remote_substitutions=remote_substitutions,
            local_substitutions=current.local_substitutions,
            exclude_folders=current.exclude_folders,
            last_sync=datetime.now(timezone.utc).isoformat(),
        )
    )
    return True


def _resolve_path(path: str) -> Path | None:
    if not path:
        return None
    try:
        return Path(path).expanduser().resolve(strict=False)
    except OSError:
        return None


def _is_relative_to(path: Path, folder: Path) -> bool:
    try:
        path.relative_to(folder)
        return True
    except ValueError:
        return False


def should_skip_record(record: object, settings: RedactionSettings | None = None) -> bool:
    """Return True when record belongs to a locally excluded folder."""
    settings = settings or load_settings()
    if not settings.exclude_folders:
        return False

    candidates = [
        getattr(record, "cwd", None),
        getattr(record, "transcript_path", None),
    ]
    resolved_candidates = [p for p in (_resolve_path(str(c or "")) for c in candidates) if p]
    if not resolved_candidates:
        return False

    for raw_folder in settings.exclude_folders:
        folder = _resolve_path(raw_folder)
        if folder is None:
            continue
        for candidate in resolved_candidates:
            if _is_relative_to(candidate, folder):
                return True
    return False


def apply_substitutions(text: str, substitutions: dict[str, Any]) -> str:
    """Apply substitutions with longest keys first for deterministic redaction."""
    if not substitutions:
        return text
    pairs = [(str(k), str(v)) for k, v in substitutions.items() if str(k)]
    for key, value in sorted(pairs, key=lambda item: len(item[0]), reverse=True):
        text = text.replace(key, value)
    return text


def redact_json_payload(payload: str, settings: RedactionSettings | None = None) -> str:
    settings = settings or load_settings()
    if not settings.enabled:
        return payload
    return apply_substitutions(payload, settings.substitutions)
