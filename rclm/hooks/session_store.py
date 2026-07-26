"""Per-session JSONL accumulation in ~/.reclaimllm/sessions/{session_id}.jsonl.

Claude Code runs hooks sequentially per session, so no file locking is needed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SESSIONS_DIR = Path.home() / ".reclaimllm" / "sessions"
_HOOK_HEALTH_MAX_RECORDS = 100
_ORPHANED_SIDECAR_MAX_AGE_S = 7 * 24 * 60 * 60


def _session_path(session_id: str) -> Path:
    return _SESSIONS_DIR / f"{session_id}.jsonl"


def _dedupe_path(session_id: str) -> Path:
    return _SESSIONS_DIR / f"{session_id}.dedupe.json"


def _read_cache_path(session_id: str) -> Path:
    return _SESSIONS_DIR / f"{session_id}.readcache.json"


def _mechanism_savings_path(session_id: str) -> Path:
    return _SESSIONS_DIR / f"{session_id}.mechanisms.json"


def _image_eviction_path(session_id: str) -> Path:
    return _SESSIONS_DIR / f"{session_id}.imageeviction.json"


def _hook_health_dir() -> Path:
    """Return the persistent Claude hook-health directory.

    Deriving this from ``_SESSIONS_DIR`` keeps tests isolated when they redirect
    the session store and keeps health metadata outside normal session cleanup.
    """
    return _SESSIONS_DIR.parent / "hook-health" / "claude"


def _hook_health_path(session_id: str) -> Path:
    return _hook_health_dir() / f"{session_id}.json"


def _marker_path(session_id: str, marker: str) -> Path:
    return _SESSIONS_DIR / f"{session_id}.{marker}.marker"


def has_marker(session_id: str, marker: str) -> bool:
    return _marker_path(session_id, marker).exists()


def write_marker(session_id: str, marker: str) -> None:
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    _marker_path(session_id, marker).write_text("1", encoding="utf-8")


def read_dedupe_state(session_id: str) -> dict[str, dict]:
    try:
        data = json.loads(_dedupe_path(session_id).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_dedupe_state(session_id: str, state: dict[str, dict]) -> None:
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    _dedupe_path(session_id).write_text(json.dumps(state), encoding="utf-8")


def read_read_cache_state(session_id: str) -> dict:
    try:
        data = json.loads(_read_cache_path(session_id).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_read_cache_state(session_id: str, state: dict) -> None:
    """Atomically persist range-cache state for one session."""
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    destination = _read_cache_path(session_id)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, destination)


def read_mechanism_savings_state(session_id: str) -> dict:
    try:
        data = json.loads(_mechanism_savings_path(session_id).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_mechanism_savings_state(session_id: str, state: dict) -> None:
    """Atomically persist cumulative mechanism savings for one host session."""
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    destination = _mechanism_savings_path(session_id)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, destination)


def read_image_eviction_state(session_id: str) -> dict:
    try:
        data = json.loads(_image_eviction_path(session_id).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_image_eviction_state(session_id: str, state: dict) -> None:
    """Atomically persist image-eviction LRU state for one session."""
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    destination = _image_eviction_path(session_id)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, destination)


def write_hook_health(session_id: str, record: dict) -> None:
    """Atomically persist one metadata-only hook-health record and bound history."""
    directory = _hook_health_dir()
    directory.mkdir(parents=True, exist_ok=True)
    destination = _hook_health_path(session_id)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, destination)

    try:
        records = sorted(
            directory.glob("*.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return
    for stale in records[_HOOK_HEALTH_MAX_RECORDS:]:
        with contextlib.suppress(OSError):
            stale.unlink()


def read_hook_health(session_id: str) -> dict:
    """Read one hook-health record, returning an empty dict when unavailable."""
    try:
        data = json.loads(_hook_health_path(session_id).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def prune_stale_sidecars(*, max_age_s: int = _ORPHANED_SIDECAR_MAX_AGE_S) -> None:
    """Remove session sidecars abandoned by sessions that never emitted SessionEnd."""
    cutoff = time.time() - max_age_s
    for pattern in (
        "*.dedupe.json",
        "*.readcache.json",
        "*.mechanisms.json",
        "*.imageeviction.json",
    ):
        for path in _SESSIONS_DIR.glob(pattern):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue


def append_event(session_id: str, event: dict) -> None:
    """Append one JSON event dict as a line to the session file."""
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_path(session_id)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


def read_events(session_id: str) -> list[dict]:
    """Read all JSON event lines for a session. Returns [] if file missing."""
    path = _session_path(session_id)
    if not path.exists():
        return []
    events: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("session_store: malformed JSON line in %s, skipping", path)
    return events


def cleanup_events(session_id: str) -> None:
    """Delete turn-scoped events while preserving session-scoped optimization state."""
    path = _session_path(session_id)
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def cleanup(session_id: str) -> None:
    """Delete all local state after the host's final session-end event."""
    cleanup_events(session_id)
    with contextlib.suppress(FileNotFoundError):
        _dedupe_path(session_id).unlink()
    with contextlib.suppress(FileNotFoundError):
        _read_cache_path(session_id).unlink()
    with contextlib.suppress(FileNotFoundError):
        _mechanism_savings_path(session_id).unlink()
    with contextlib.suppress(FileNotFoundError):
        _image_eviction_path(session_id).unlink()
