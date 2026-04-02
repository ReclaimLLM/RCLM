"""Shared async upload logic used by both rclm-proxy and rclm-wrap."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

import aiohttp

_FAILED_UPLOADS_DIR = Path.home() / ".reclaimllm" / "failed_uploads"

from rclm import _config  # noqa: E402
from rclm._models import HookSessionRecord, ProxyRecord, SessionRecord  # noqa: E402

logger = logging.getLogger(__name__)

_RETRY_DELAYS = (0.5, 1.0, 2.0)  # seconds, exponential backoff


AnyRecord = ProxyRecord | SessionRecord | HookSessionRecord


def _to_json(record: AnyRecord) -> str:
    return json.dumps(dataclasses.asdict(record))


async def upload(
    record: AnyRecord,
    session: aiohttp.ClientSession,
) -> None:
    """POST record as JSON to BACKEND_SERVER/api/ingest.

    Retries 3x with exponential backoff.
    Quarantines to ~/.reclaimllm/failed_uploads/ if server URL is unset or all retries fail.
    """
    cfg = _config.load()
    base = os.environ.get("BACKEND_SERVER") or cfg.get("server_url")
    if not base:
        _quarantine(record)
        return
    url = base.rstrip("/") + "/api/ingest"
    # if len(record.messages) == 0:
    #     logger.warning("Record has empty messages; skipping upload")
    #     return
    payload = _to_json(record)
    headers = {"Content-Type": "application/json"}
    # get from config first, then environment
    api_key = cfg.get("api_key")
    if api_key:
        headers["X-API-Key"] = api_key

    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        try:
            async with session.post(url, data=payload, headers=headers) as resp:
                if resp.status < 500:
                    # 2xx/4xx: do not retry (4xx is a client-config error, not transient)
                    if resp.status >= 400:
                        logger.warning(
                            "rclm upload got %s from server; giving up",
                            resp.status,
                        )
                    return
                logger.warning(
                    "rclm upload attempt %d/%d got %s; retrying in %.1fs",
                    attempt,
                    len(_RETRY_DELAYS),
                    resp.status,
                    delay,
                )
        except (aiohttp.ClientError, OSError) as exc:
            logger.warning(
                "rclm upload attempt %d/%d failed: %s; retrying in %.1fs",
                attempt,
                len(_RETRY_DELAYS),
                exc,
                delay,
            )
        await asyncio.sleep(delay)

    logger.error("rclm upload failed after all retries; quarantining record locally")
    _quarantine(record)


def _quarantine(record: AnyRecord) -> None:
    """Write the failed record to ~/.reclaimllm/failed_uploads/ with owner-only permissions.

    Emits a one-line stderr notice pointing to the file. The directory and file
    are created with mode 0o700/0o600 so other local users cannot read them.
    """
    try:
        _FAILED_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        os.chmod(_FAILED_UPLOADS_DIR, 0o700)
        path = _FAILED_UPLOADS_DIR / f"{record.session_id}.json"
        path.write_text(_to_json(record), encoding="utf-8")
        os.chmod(path, 0o600)
        print(
            f"rclm: upload failed; record saved to {path}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"rclm: upload failed and could not quarantine record: {exc}",
            file=sys.stderr,
        )


async def run_upload_worker(queue: asyncio.Queue) -> None:
    """Background asyncio task. Drains the queue and uploads each record."""
    async with aiohttp.ClientSession() as session:
        while True:
            record = await queue.get()
            if record is None:  # sentinel: shut down
                queue.task_done()
                return
            try:
                await upload(record, session)
            except Exception:
                logger.exception("Unexpected error in upload worker")
            finally:
                queue.task_done()


_session: aiohttp.ClientSession | None = None


async def _get_session() -> aiohttp.ClientSession:
    """Return a persistent module-level ClientSession, creating it if needed."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def upload_single(record: AnyRecord) -> None:
    """Upload one record, reusing the module-level session."""
    session = await _get_session()
    await upload(record, session)
