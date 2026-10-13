"""Device check-in upload (SessionStart + rclm-update enrollment)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiohttp

from rclm import _config, _device
from rclm._http import create_tcp_connector
from rclm.hooks import integrity, routing
from rclm.hooks.updater import installed_version

logger = logging.getLogger(__name__)

_CHECKIN_PATH = "/api/devices/checkin"
_THROTTLE_SECONDS = 6 * 3600
_STATE_PATH = Path.home() / ".reclaimllm" / "checkin_state.json"
_TIMEOUT_S = 5.0


def maybe_checkin(
    *,
    cwd: str | None = None,
    event: str | None = None,
    force: bool = False,
    timeout_s: float | None = None,
) -> None:
    """Best-effort sync wrapper for SessionStart / update paths."""
    try:
        asyncio.run(async_checkin(cwd=cwd, event=event, force=force, timeout_s=timeout_s))
    except Exception as exc:  # noqa: BLE001 — never break hooks/update
        logger.debug("device check-in skipped: %s", exc)


async def async_checkin(
    *,
    cwd: str | None = None,
    event: str | None = None,
    force: bool = False,
    timeout_s: float | None = None,
) -> dict[str, Any] | None:
    creds = _config.resolve_credentials()
    if creds is None:
        return None

    clients = integrity.collect_client_status(cwd=cwd)
    cfg = _config.load()
    local_dlp = bool(cfg.get("dlp", _config.DEFAULT_DLP_ENABLED))
    enabled = _config.dlp_enabled(cfg)
    pending = list(cfg.get("pending_events") or [])
    dlp_payload = {
        "enabled": enabled,
        "local_setting": local_dlp,
        "required": bool(
            isinstance(cfg.get("org_hook_policy"), dict)
            and cfg["org_hook_policy"].get("dlp_required")
        ),
    }
    exception_id = _active_exception_id(cfg)
    if exception_id:
        dlp_payload["exception_id"] = exception_id

    policy = cfg.get("org_hook_policy") if isinstance(cfg.get("org_hook_policy"), dict) else {}
    policy_hash = str(policy.get("policy_version") or "")
    payload = {
        "payload_version": 1,
        "device": {
            "device_id": _device.device_id(),
            "hostname_hash": _device.hostname_hash(
                org_salt=str(policy.get("org_salt") or "") or None
            ),
            "os": _os_name(),
            "rclm_version": installed_version(),
            "environment": _device.environment_class(),
        },
        "clients": clients,
        "dlp": dlp_payload,
        "policy_hash": policy_hash,
        "policy_signature_valid": True,
        "routing": routing.classify_routing(cfg=cfg),
        "integrity_hash": integrity.integrity_hash(clients),
        "heartbeat_enabled": bool(cfg.get("heartbeat_enabled")),
    }
    if event:
        payload["event"] = event
    if pending:
        payload["pending_events"] = pending

    fingerprint = json.dumps(
        {"clients": clients, "dlp": dlp_payload, "event": event},
        sort_keys=True,
        separators=(",", ":"),
    )
    if not force and not event and not _should_send(fingerprint):
        return None

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": creds.api_key,
        "X-RCLM-Device": _device.device_id(),
    }
    device_key = cfg.get("device_api_key")
    if isinstance(device_key, str) and device_key.strip():
        headers["X-RCLM-Device-Key"] = device_key.strip()
    url = creds.server_url.rstrip("/") + _CHECKIN_PATH
    timeout = float(timeout_s) if timeout_s is not None else _TIMEOUT_S
    try:
        async with (
            aiohttp.ClientSession(
                connector=create_tcp_connector(),
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as session,
            session.post(url, json=payload, headers=headers) as resp,
        ):
            body = await resp.text()
            if resp.status >= 400:
                logger.debug("check-in failed %s: %s", resp.status, body[:200])
                return None
            if pending:
                _config.patch(pending_events=[])
            parsed: dict[str, Any]
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {"ok": True}
            issued = parsed.get("device_api_key")
            if isinstance(issued, str) and issued.strip():
                _config.patch(device_api_key=issued.strip())
            _remember(fingerprint)
            return parsed
    except (aiohttp.ClientError, OSError, TimeoutError) as exc:
        logger.debug("check-in network error: %s", exc)
        return None


def _should_send(fingerprint: str) -> bool:
    state = _load_state()
    now = time.time()
    last_at = float(state.get("last_at") or 0)
    last_fp = state.get("fingerprint")
    if last_fp != fingerprint:
        return True
    return (now - last_at) >= _THROTTLE_SECONDS


def _remember(fingerprint: str) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(
        json.dumps({"fingerprint": fingerprint, "last_at": time.time()}),
        encoding="utf-8",
    )


def _load_state() -> dict:
    if not _STATE_PATH.exists():
        return {}
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _active_exception_id(cfg: dict) -> str | None:
    policy = cfg.get("org_hook_policy")
    if not isinstance(policy, dict):
        return None
    exceptions = policy.get("dlp_exceptions")
    if not isinstance(exceptions, list):
        return None
    now = time.time()
    for item in exceptions:
        if not isinstance(item, dict):
            continue
        expires = item.get("expires_at")
        if not expires:
            continue
        try:
            # Accept Z or offset ISO strings via fromisoformat-ish parse
            from datetime import datetime

            exp = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
            if exp.timestamp() > now:
                eid = item.get("id")
                return str(eid) if eid else None
        except ValueError:
            continue
    return None


def _os_name() -> str:
    import platform

    return platform.system().lower() or "unknown"
