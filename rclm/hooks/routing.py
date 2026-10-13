"""Classify agent provider routing as gateway / direct / unknown (no URLs)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rclm import _config

_ANTHROPIC_ENV = ("ANTHROPIC_BASE_URL",)
_OPENAI_ENV = ("OPENAI_BASE_URL",)
_GEMINI_ENV = ("GOOGLE_GEMINI_BASE_URL", "GEMINI_API_BASE", "GEMINI_BASE_URL")


def classify_routing(*, cfg: dict | None = None) -> dict[str, str]:
    """Return {provider: gateway|direct|unknown} for required/known providers."""
    cfg = cfg if cfg is not None else _config.load()
    policy = cfg.get("org_hook_policy") if isinstance(cfg.get("org_hook_policy"), dict) else {}
    hosts = {
        _normalize_host(h)
        for h in (policy.get("gateway_hosts") or [])
        if isinstance(h, str) and h.strip()
    }
    required = {
        str(p).lower()
        for p in (policy.get("gateway_required_providers") or [])
        if isinstance(p, str)
    }
    result: dict[str, str] = {
        "anthropic": _class_from_env(_ANTHROPIC_ENV, hosts) or _class_from_claude_settings(hosts),
        "openai": _class_from_env(_OPENAI_ENV, hosts) or _class_from_codex_config(hosts),
        "gemini": _class_from_env(_GEMINI_ENV, hosts) or "unknown",
        "cursor": "unknown",  # Cursor-hosted models; bypass N/A
        "copilot": "unknown",
    }
    # Drop providers not configured/required unless we have a signal.
    out: dict[str, str] = {}
    for provider, classification in result.items():
        if provider in required or classification != "unknown":
            out[provider] = classification
        elif required:
            continue
        else:
            out[provider] = classification
    return out


def _class_from_env(keys: tuple[str, ...], gateway_hosts: set[str]) -> str | None:
    for key in keys:
        raw = os.environ.get(key)
        if not raw:
            continue
        return _classify_url(raw, gateway_hosts)
    return None


def _classify_url(raw: str, gateway_hosts: set[str]) -> str:
    host = _normalize_host(raw)
    if not host:
        return "unknown"
    if not gateway_hosts:
        # No org gateway configured — never call this "bypass".
        return "unknown"
    if host in gateway_hosts or any(host.endswith("." + h) for h in gateway_hosts):
        return "gateway"
    return "direct"


def _normalize_host(value: str) -> str:
    text = value.strip().lower()
    if not text:
        return ""
    if "://" not in text:
        text = "https://" + text
    try:
        parsed = urlparse(text)
        return (parsed.hostname or "").lower()
    except Exception:
        return ""


def _class_from_claude_settings(gateway_hosts: set[str]) -> str:
    path = Path.home() / ".claude" / "settings.json"
    data = _read_json(path)
    if not data:
        return "unknown"
    env = data.get("env") if isinstance(data.get("env"), dict) else {}
    for key in _ANTHROPIC_ENV:
        if isinstance(env.get(key), str) and env[key].strip():
            return _classify_url(env[key], gateway_hosts)
    # apiKeyHelper presence without gateway URL → likely direct provider auth helper.
    if data.get("apiKeyHelper") and not gateway_hosts:
        return "unknown"
    if data.get("apiKeyHelper") and gateway_hosts:
        # Helper alone is inconclusive.
        return "unknown"
    return "unknown"


def _class_from_codex_config(gateway_hosts: set[str]) -> str:
    path = Path.home() / ".codex" / "config.toml"
    if not path.exists():
        return "unknown"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    # Minimal scan for base_url = "..."
    for line in text.splitlines():
        stripped = line.strip()
        if "base_url" not in stripped.lower():
            continue
        _, _, rhs = stripped.partition("=")
        value = rhs.strip().strip("\"'")
        if value:
            return _classify_url(value, gateway_hosts)
    return "unknown"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None
