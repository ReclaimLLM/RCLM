"""Update-check logic for rclm CLI tools.

check_for_update() is called from rclm-hooks-install and rclm-update.
It fetches the latest version from PyPI at most once every 24 hours,
caching the result in ~/.reclaimllm/config.json.

All network errors are swallowed — a failed check is always silent.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from rclm import _config

_PYPI_URL = "https://pypi.org/pypi/rclm/json"
_CHECK_TTL_HOURS = 24
_NETWORK_TIMEOUT_S = 2


def installed_version() -> str:
    """Return the currently installed rclm version, or '0.0.0' if undetectable."""
    try:
        return _pkg_version("rclm")
    except PackageNotFoundError:
        return "0.0.0"


def _is_newer(candidate: str, installed: str) -> bool:
    """Return True if candidate is strictly newer than installed (simple tuple compare)."""
    def _parse(v: str) -> tuple[int, ...]:
        parts = []
        for part in v.split("."):
            digits = "".join(c for c in part if c.isdigit())
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

    try:
        return _parse(candidate) > _parse(installed)
    except Exception:
        return False


def _fetch_pypi_latest() -> str | None:
    """GET the latest version from PyPI. Returns None on any error."""
    try:
        req = urllib.request.Request(
            _PYPI_URL,
            headers={"User-Agent": f"rclm/{installed_version()}"},
        )
        with urllib.request.urlopen(req, timeout=_NETWORK_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
            return data["info"]["version"]
    except Exception:
        return None


def check_for_update(force: bool = False) -> str | None:
    """Return the latest version string if it is newer than installed, else None.

    Uses a 24-hour cache stored in ~/.reclaimllm/config.json under
    ``last_update_check`` (ISO-8601 timestamp) and ``latest_version`` (str).

    Pass force=True to skip the cache and always hit PyPI (used by rclm-update).
    All errors are swallowed — callers should never crash due to an update check.
    """
    try:
        cfg = _config.load()
        now = datetime.now(timezone.utc)
        current = installed_version()

        if not force:
            last_check_raw = cfg.get("last_update_check")
            cached_latest = cfg.get("latest_version")
            if last_check_raw and cached_latest:
                try:
                    last_check = datetime.fromisoformat(last_check_raw)
                    if (now - last_check) < timedelta(hours=_CHECK_TTL_HOURS):
                        # Cache is fresh — use it
                        return cached_latest if _is_newer(cached_latest, current) else None
                except (ValueError, TypeError):
                    pass  # Bad timestamp in cache — fall through to a fresh fetch

        latest = _fetch_pypi_latest()
        if latest:
            _config.patch(
                last_update_check=now.isoformat(),
                latest_version=latest,
            )
        return latest if (latest and _is_newer(latest, current)) else None
    except Exception:
        return None


def apply_update() -> bool:
    """Run ``pip install --upgrade rclm`` in the active Python environment.

    Uses sys.executable so it targets the correct interpreter regardless of
    virtualenv or system Python. Returns True on success.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "rclm"],
    )
    return result.returncode == 0
