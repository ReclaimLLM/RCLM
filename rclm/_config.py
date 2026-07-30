"""Shared access to ~/.reclaimllm/config.json.

Written by rclm-hooks-install; read by the uploader at upload time.
Env vars RECLAIMLLM_SERVER_URL / RECLAIMLLM_API_KEY always take precedence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path.home() / ".reclaimllm" / "config.json"

DEFAULT_COMPRESSION_CONFIG = {
    "enabled": True,
    "dedupe": False,
    "test_filter": True,
    "min_dedupe_chars": 500,
    "test_filter_max_chars": 8_000,
}

_POLICY_MODES = frozenset({"off", "observe", "enforce"})


@dataclass(frozen=True)
class EffectiveHookPolicy:
    force_compression_enforcement: bool
    policy_version: int
    mechanisms: dict[str, dict[str, object]]
    legacy_shadow: bool

    def enabled(self, mechanism: str) -> bool:
        value = self.mechanisms.get(mechanism) or {}
        return bool(value.get("enabled"))

    def shadow_for(self, mechanism: str) -> bool:
        value = self.mechanisms.get(mechanism) or {}
        return value.get("mode") != "enforce"

    def snapshot(self) -> dict:
        return {
            "force_compression_enforcement": self.force_compression_enforcement,
            "policy_version": self.policy_version,
            "mechanisms": self.mechanisms,
        }


def effective_hook_policy(
    cfg: dict | None = None, *, provider: str | None = None
) -> EffectiveHookPolicy:
    """Resolve local feature switches and the cached org policy once.

    Hooks remain network-free. SessionStart refreshes ``org_hook_policy`` in
    config.json; every provider consumes the same explicit snapshot here.
    """
    cfg = cfg or load()
    compression = compression_config(cfg)
    remote = cfg.get("org_hook_policy")
    remote = remote if isinstance(remote, dict) else {}
    force = bool(remote.get("force_compression_enforcement", False))
    legacy_shadow = bool(cfg.get("shadow_mode", False)) and not force
    raw_modes = remote.get("compression_modes")
    raw_modes = raw_modes if isinstance(raw_modes, dict) else {}

    local_enabled = {
        "exec_compaction": bool(compression["enabled"]),
        "hash_dedupe": bool(compression["dedupe"]),
        "test_filter": bool(compression["enabled"] and compression["test_filter"]),
        "range_cache": bool(cfg.get("read_cache", False)),
        "image_downscale": bool(cfg.get("image_lifecycle", False)),
    }
    mechanisms: dict[str, dict[str, object]] = {}
    unsupported = {
        "codex": {"exec_compaction", "test_filter", "image_downscale"},
        "gemini": {"exec_compaction", "test_filter", "image_downscale"},
    }.get(provider or "", set())
    for name, enabled in local_enabled.items():
        supported = name not in unsupported
        configured_mode = raw_modes.get(name)
        if configured_mode not in _POLICY_MODES:
            configured_mode = None
        if not enabled or configured_mode == "off":
            mode = "off"
        elif not supported:
            mode = "observe"
        elif force:
            mode = "enforce"
        elif configured_mode:
            mode = configured_mode
        else:
            mode = "observe" if legacy_shadow else "enforce"
        mechanisms[name] = {
            "enabled": enabled and configured_mode != "off",
            "mode": mode,
            "supported": supported,
        }

    version = remote.get("policy_version", 0)
    return EffectiveHookPolicy(
        force_compression_enforcement=force,
        policy_version=version if isinstance(version, int) and version >= 0 else 0,
        mechanisms=mechanisms,
        legacy_shadow=legacy_shadow,
    )


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
