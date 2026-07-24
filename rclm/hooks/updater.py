"""Update-check logic for rclm CLI tools.

check_for_update() is called from rclm-hooks-install and rclm-update.
It fetches the latest version from PyPI at most once every 24 hours,
caching the result in ~/.reclaimllm/config.json.

All network errors are swallowed — a failed check is always silent.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

from rclm import _config

_PYPI_URL = "https://pypi.org/pypi/rclm/json"
_CHECK_TTL_HOURS = 24
_NETWORK_TIMEOUT_S = 2
_SESSION_END_UPDATE_TTL_HOURS = 24
_SESSION_END_UPDATE_ATTEMPT_KEY = "last_session_end_update_attempt"
_SESSION_END_UPDATE_LOCK_NAME = "session-end-update.lock"
_SESSION_END_UPDATE_LOG_NAME = "session-end-update.log"


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


def _session_end_paths() -> tuple[Path, Path]:
    config_dir = _config.CONFIG_PATH.parent
    return (
        config_dir / _SESSION_END_UPDATE_LOCK_NAME,
        config_dir / _SESSION_END_UPDATE_LOG_NAME,
    )


def _session_end_attempt_is_recent(cfg: dict, now: datetime) -> bool:
    value = cfg.get(_SESSION_END_UPDATE_ATTEMPT_KEY)
    if not isinstance(value, str):
        return False
    try:
        attempted_at = datetime.fromisoformat(value)
        return now - attempted_at < timedelta(hours=_SESSION_END_UPDATE_TTL_HOURS)
    except (TypeError, ValueError):
        return False


def schedule_session_end_update() -> bool:
    """Start one detached full update per day after a primary session ends.

    The terminal hook only reads local configuration and starts a child process.
    Network access, pip, and hook reinstallation run outside the hook's critical
    path. The attempt is recorded before spawning so a transient failure does
    not retry on every subsequent session end.
    """
    try:
        cfg = _config.load()
        now = datetime.now(timezone.utc)
        if _session_end_attempt_is_recent(cfg, now):
            return False

        _config.patch(**{_SESSION_END_UPDATE_ATTEMPT_KEY: now.isoformat()})
        _lock_path, log_path = _session_end_paths()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_file:
            subprocess.Popen(
                [sys.executable, "-m", "rclm.hooks.updater", "--session-end-update"],
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return True
    except Exception:
        return False


def run_session_end_update() -> None:
    """Run a scheduled update once, guarded against concurrent hook processes."""
    lock_path, _log_path = _session_end_paths()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return
    except OSError:
        return

    try:
        from rclm import update

        original_argv = sys.argv[:]
        sys.argv = ["rclm-update"]
        try:
            update.main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv
    finally:
        os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


if __name__ == "__main__" and sys.argv[1:] == ["--session-end-update"]:
    run_session_end_update()
