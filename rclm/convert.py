"""CLI client for the session context export endpoint."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp

from rclm import _config


async def _fetch_context(
    server_url: str,
    api_key: str,
    session_id: str,
    target_tool: str,
    include_diffs: bool,
    max_diff_lines: int,
    force_regenerate: bool,
) -> str:
    params = {
        "target_tool": target_tool,
        "include_diffs": str(include_diffs).lower(),
        "max_diff_lines": str(max_diff_lines),
        "force_regenerate": str(force_regenerate).lower(),
    }
    url = f"{server_url.rstrip('/')}/api/sessions/{session_id}/export-context"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    async with (
        aiohttp.ClientSession() as session,
        session.post(url, params=params, headers=headers) as resp,
    ):
        body = await resp.text()
        if resp.status == 404:
            print(f"rclm: session {session_id!r} not found", file=sys.stderr)
            sys.exit(1)
        if resp.status == 503:
            try:
                detail = json.loads(body).get("detail", body)
            except Exception:
                detail = body
            print(f"rclm: server error: {detail}", file=sys.stderr)
            sys.exit(1)
        if not resp.ok:
            try:
                detail = json.loads(body).get("detail", body)
            except Exception:
                detail = body
            print(
                f"rclm: export-context failed (HTTP {resp.status}): {detail}",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            print(f"rclm: unexpected response from server: {body[:200]}", file=sys.stderr)
            sys.exit(1)

    return data["context_document"]


def convert_session(
    session_id: str,
    target_tool: str,
    *,
    output_path: str | None = None,
    include_diffs: bool = True,
    max_diff_lines: int = 50,
    force_regenerate: bool = False,
) -> None:
    """Fetch a context export document from the server and write it to stdout or a file."""
    cfg = _config.load()
    server_url = os.environ.get("RECLAIMLLM_SERVER_URL") or cfg.get("server_url")
    api_key = os.environ.get("RECLAIMLLM_API_KEY") or cfg.get("api_key")

    if not server_url:
        print(
            "rclm: server URL not configured. Run 'rclm-hooks-install' or set RECLAIMLLM_SERVER_URL.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not api_key:
        print(
            "rclm: API key not configured. Run 'rclm-hooks-install' or set RECLAIMLLM_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    document = asyncio.run(
        _fetch_context(
            server_url,
            api_key,
            session_id,
            target_tool,
            include_diffs,
            max_diff_lines,
            force_regenerate,
        )
    )

    if output_path:
        dest = Path(output_path)
        dest.write_text(document, encoding="utf-8")
        print(f"rclm: context saved to {dest}", file=sys.stderr)
    else:
        print(document)
