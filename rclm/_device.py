"""Stable per-machine device identity for coverage / MCP check-ins."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEVICE_PATH = Path.home() / ".reclaimllm" / "device.json"

_EPHEMERAL_ENV_KEYS = (
    "CODESPACE_NAME",
    "REMOTE_CONTAINERS",
    "DEVCONTAINER",
    "GITPOD_WORKSPACE_ID",
    "GITPOD_INSTANCE_ID",
)


def ensure_device_id() -> str:
    """Return the local device_id, creating ~/.reclaimllm/device.json if needed."""
    data = _load()
    device_id = data.get("device_id")
    if isinstance(device_id, str) and _is_uuid(device_id):
        return device_id
    device_id = str(uuid.uuid4())
    _write({"device_id": device_id, "created_at": _now_iso()})
    return device_id


def device_id() -> str:
    return ensure_device_id()


def hostname_hash(*, org_salt: str | None = None) -> str:
    """SHA256 of hostname, optionally salted once org salt is known."""
    host = socket.gethostname() or "unknown"
    material = f"{org_salt}:{host}" if org_salt else host
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def environment_class() -> str:
    """Classify durable vs ephemeral host for orphan / silence handling."""
    for key in _EPHEMERAL_ENV_KEYS:
        if os.environ.get(key):
            return "ephemeral"
    return "durable"


def _load() -> dict:
    if not DEVICE_PATH.exists():
        return {}
    try:
        data = json.loads(DEVICE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write(data: dict) -> None:
    DEVICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEVICE_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    with contextlib.suppress(OSError):
        DEVICE_PATH.chmod(0o600)


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
