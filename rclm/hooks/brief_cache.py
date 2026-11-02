"""Local SQLite cache for the Start Brief ranked "similar past work" brief.

Synced in the background (SessionStart spawns a detached subprocess, mirroring
rclm.hooks.updater.schedule_session_end_update) from the server's
GET /api/projects/{project_name}/brief-pack endpoint. rclm.hooks.brief_rank
queries this cache synchronously at UserPromptSubmit time -- it never fetches
over the network on that path.

Keyed by (project_key, api_key fingerprint) via rclm.hooks._project_key, not
project_key alone -- see that module's docstring for why.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from rclm import _config
from rclm.hooks._project_key import cache_db_path, derive_project_key, write_resolved_project_name

_SYNC_TTL_SECONDS = 900  # don't spawn more than one sync subprocess per project per 15 min
_FETCH_TIMEOUT_SECONDS = 10.0

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT,
    category TEXT,
    agent_client TEXT,
    ended_at TEXT,
    summary TEXT
);
CREATE VIRTUAL TABLE sessions_fts USING fts5(
    session_id UNINDEXED, title, summary, tags, paths
);
CREATE TABLE session_files (
    session_id TEXT,
    path TEXT,
    sha_after TEXT
);
CREATE TABLE files (
    path TEXT PRIMARY KEY,
    last_session_id TEXT,
    last_edited_at TEXT,
    sha_after TEXT,
    busyness INTEGER
);
CREATE TABLE cochange (
    path TEXT,
    neighbour_path TEXT,
    sessions_coedited INTEGER
);
CREATE INDEX session_files_session_idx ON session_files(session_id);
CREATE INDEX session_files_path_idx ON session_files(path);
CREATE INDEX cochange_path_idx ON cochange(path);
"""

_DROP_ALL = """
DROP TABLE IF EXISTS meta;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS sessions_fts;
DROP TABLE IF EXISTS session_files;
DROP TABLE IF EXISTS files;
DROP TABLE IF EXISTS cochange;
"""


def _ensure_private_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)


def _last_sync_marker(db_path: Path) -> Path:
    return db_path.with_suffix(".lastsync")


def _sync_is_recent(db_path: Path) -> bool:
    try:
        age = time.time() - _last_sync_marker(db_path).stat().st_mtime
    except OSError:
        return False
    return age < _SYNC_TTL_SECONDS


def schedule_sync(cwd: str) -> bool:
    """Spawn one detached background sync per project, throttled to _SYNC_TTL_SECONDS.

    Mirrors rclm.hooks.updater.schedule_session_end_update: the calling hook
    only reads local config and starts a child process -- network access runs
    outside the hook's critical path. The attempt is recorded before spawning
    so a transient failure does not retry on every subsequent SessionStart.
    """
    try:
        creds = _config.resolve_credentials()
        if creds is None:
            return False
        project_key = derive_project_key(cwd)
        db_path = cache_db_path(project_key, creds.api_key)
        if _sync_is_recent(db_path):
            return False
        _ensure_private_dir(db_path.parent)
        _last_sync_marker(db_path).touch()

        import subprocess

        log_path = db_path.with_suffix(".synclog")
        with log_path.open("ab") as log_file:
            subprocess.Popen(
                [sys.executable, "-m", "rclm.hooks.brief_cache", "--sync", project_key, cwd],
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return True
    except Exception:
        return False


def rebuild_cache(db_path: Path, pack: dict[str, Any]) -> None:
    """Replace the local cache's contents with a freshly fetched pack.

    Drop-and-recreate inside one connection/transaction: the only reader is
    this same developer's short-lived, single-threaded hook process, so there
    is no concurrent-reader hazard to design around.
    """
    _ensure_private_dir(db_path.parent)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_DROP_ALL)
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('etag', ?), ('project_name', ?), ('built_at', ?)",
            (
                str(pack.get("etag") or ""),
                str(pack.get("project_name") or ""),
                str(pack.get("built_at") or ""),
            ),
        )
        for session in pack.get("sessions") or []:
            sid = session.get("session_id")
            if not sid:
                continue
            files = session.get("files") or []
            conn.execute(
                "INSERT INTO sessions (session_id, title, category, agent_client, ended_at, "
                "summary) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    sid,
                    session.get("title") or "",
                    session.get("category") or "",
                    session.get("agent_client") or "",
                    session.get("ended_at") or "",
                    session.get("summary") or "",
                ),
            )
            conn.execute(
                "INSERT INTO sessions_fts (session_id, title, summary, tags, paths) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    sid,
                    session.get("title") or "",
                    session.get("summary") or "",
                    " ".join(session.get("tags") or []),
                    " ".join(f.get("path", "") for f in files),
                ),
            )
            for f in files:
                path = f.get("path")
                if not path:
                    continue
                conn.execute(
                    "INSERT INTO session_files (session_id, path, sha_after) VALUES (?, ?, ?)",
                    (sid, path, f.get("sha_after")),
                )
        for f in pack.get("files") or []:
            path = f.get("path")
            if not path:
                continue
            conn.execute(
                "INSERT INTO files (path, last_session_id, last_edited_at, sha_after, busyness) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    path,
                    f.get("last_session_id"),
                    f.get("last_edited_at"),
                    f.get("sha_after"),
                    int(f.get("busyness") or 0),
                ),
            )
            for neighbour in f.get("cochange") or []:
                npath = neighbour.get("path")
                if not npath:
                    continue
                conn.execute(
                    "INSERT INTO cochange (path, neighbour_path, sessions_coedited) "
                    "VALUES (?, ?, ?)",
                    (path, npath, int(neighbour.get("sessions_coedited") or 0)),
                )
        conn.commit()
    finally:
        conn.close()


def read_cached_etag(db_path: Path) -> str | None:
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'etag'").fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return None


async def run_sync(project_key: str, cwd: str = "") -> bool:
    """Resolve the real project_name, fetch the latest pack, and rebuild the
    local cache if it changed.

    Runs inside the detached subprocess schedule_sync() spawns -- deferred
    import of ReclaimLLMClient keeps mcp_server.py out of the main hook
    process's import graph.

    project_key is schedule_sync()'s best-effort local key (used to throttle
    and to find the previous cache if resolution fails this time too). If the
    server resolves a real project_name for cwd, that name -- not
    project_key -- is what gets fetched and cached, and it's persisted via
    write_resolved_project_name() so later UserPromptSubmit calls (which
    never touch the network) pick it up through derive_project_key().
    """
    from rclm.mcp_server import ReclaimLLMClient, ReclaimLLMError

    creds = _config.resolve_credentials()
    if creds is None:
        return False

    client = ReclaimLLMClient()
    effective_key = project_key
    if cwd:
        with suppress(Exception):
            resolved = await client.resolve_project_name(cwd)
            if resolved:
                write_resolved_project_name(cwd, resolved)
                effective_key = resolved

    db_path = cache_db_path(effective_key, creds.api_key)
    local_etag = read_cached_etag(db_path)
    try:
        status, pack, _etag = await client.fetch_brief_pack(
            effective_key, if_none_match=local_etag, timeout_seconds=_FETCH_TIMEOUT_SECONDS
        )
    except ReclaimLLMError:
        return False
    if status == 304 or pack is None:
        return False
    rebuild_cache(db_path, pack)
    return True


def main() -> None:
    """Entry point for the detached sync subprocess.

    Invoked as: python -m rclm.hooks.brief_cache --sync <project_key> <cwd>
    -- cwd is passed through to run_sync() for the project_name resolve call.
    """
    import asyncio

    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "--sync":
        project_key = args[1]
        cwd = args[2] if len(args) >= 3 else ""
        with suppress(Exception):
            asyncio.run(run_sync(project_key, cwd))


if __name__ == "__main__":
    main()
