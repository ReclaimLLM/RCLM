"""Shared (project_key, cache-identity) derivation for the Start Brief ranked cache.

project_key is primarily the server's real project_name now, fetched via
GET /api/projects/resolve (server.stats_task.derive_project_name) and cached
in _RESOLVE_CACHE_ROOT -- see rclm.hooks.brief_cache.run_sync, which performs
that network call and calls write_resolved_project_name() on success. This
module's own derive_project_key() stays network-free by design (it can run on
the UserPromptSubmit hot path), so it reads whatever resolution is already
cached and only falls back to a local approximation (git-root basename, or
cwd basename if not a git repo) on a cold cache, a stale one, or a project the
server has never resolved -- e.g. before the first background sync completes.
A project renamed or re-mapped server-side converges to the real name within
one resolve cycle; until then the approximation may not match.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

_CACHE_ROOT = Path.home() / ".reclaimllm" / "projects"
_MAX_ANCESTOR_WALK = 50

# Same cadence as rclm.hooks.brief_cache._SYNC_TTL_SECONDS by convention (one
# resolve per background sync pass) -- kept as a separate constant to avoid a
# circular import between the two modules, not a deliberately different value.
_RESOLVE_TTL_SECONDS = 900


def _git_root(cwd: str) -> Path:
    base = Path(cwd)
    current = base
    for _ in range(_MAX_ANCESTOR_WALK):
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return base


def _resolve_cache_file(root: Path) -> Path:
    # Reads module-global _CACHE_ROOT at call time (not a bound constant) so tests
    # that monkeypatch it (e.g. isolate_cache_root in test_brief_cache.py) isolate
    # this cache too, not just the main project cache.
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
    return _CACHE_ROOT / "_resolved" / f"{digest}.json"


def read_resolved_project_name(cwd: str) -> str | None:
    """A fresh, server-resolved project_name for cwd, or None (cold/stale/unresolved).

    Local file read only -- safe to call from the UserPromptSubmit hot path.
    """
    if not cwd:
        return None
    path = _resolve_cache_file(_git_root(cwd))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    resolved_at = data.get("resolved_at")
    if not isinstance(resolved_at, (int, float)) or time.time() - resolved_at > (
        _RESOLVE_TTL_SECONDS
    ):
        return None
    name = data.get("project_name")
    return name if isinstance(name, str) and name else None


def write_resolved_project_name(cwd: str, project_name: str) -> None:
    """Persist a server-resolved project_name for cwd's git root. Network-call site only."""
    path = _resolve_cache_file(_git_root(cwd))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"project_name": project_name, "resolved_at": time.time()}),
        encoding="utf-8",
    )


def derive_project_key(cwd: str) -> str:
    if not cwd:
        return "unknown"
    resolved = read_resolved_project_name(cwd)
    if resolved:
        return resolved
    root = _git_root(cwd)
    return root.name or Path(cwd).name or "unknown"


def api_key_fingerprint(api_key: str) -> str:
    """Short stable hash of the active API key.

    The local cache is keyed by (project_key, this fingerprint), not
    project_key alone: config.json and device.json are scoped by $HOME, not
    by API key, so two developers sharing one machine and switching keys get
    the same local state regardless of whose key is active. Keying the cache
    by the fingerprint means re-syncing under a different key naturally lands
    in a different cache path -- there is nothing to invalidate (PRD §5
    point 3).
    """
    if not api_key:
        return "no-key"
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def cache_dir(project_key: str, api_key: str) -> Path:
    return _CACHE_ROOT / project_key / api_key_fingerprint(api_key)


def cache_db_path(project_key: str, api_key: str) -> Path:
    return cache_dir(project_key, api_key) / "brief.sqlite"
