"""MCP server exposing ReclaimLLM session recall tools."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp

from rclm import _config

_DEFAULT_LIMIT = 5
_MAX_LIMIT = 25
_MAX_HIGHLIGHT_CHARS = 500
_VALID_RECORD_TYPES = {"session", "proxy", "browser-chat"}
_MCP_INSTRUCTIONS = (
    "ReclaimLLM tools recall prior captured AI sessions. Use them sparingly and only when prior "
    "session context is likely useful. Prefer normal reasoning and local repo inspection for the "
    "current task. "
    "Tool selection rules: "
    "1. search_sessions: use only when the user hints that similar prior work may exist, asks how "
    "something was handled before, or the task is a bug fix/performance improvement where past "
    "context may help. If the prompt has semantic terms plus a file/folder path, pass that path as "
    "file_path. Run at most one search round; if results are weak or the user is unsatisfied, do not "
    "keep changing terms and searching again. Ask whether they want to skip or provide a more specific "
    "session clue. "
    "2. search_by_filename: use for file/folder-only history requests such as 'show changes in "
    "`auth.tsx`' or 'show changes under `/api/auth`'. Do not use it when semantic intent is present; "
    "then use search_sessions with file_path. "
    "3. get_session: use only when the user asks to inspect a specific session ID. It returns summary "
    "metadata and a frontend link, not context to inject. "
    "4. summarize_session: never call immediately after search_sessions or search_by_filename. Search "
    "results are enough for the user to decide next steps. Use summarize_session only after an explicit "
    "user follow-up such as 'summarize <session-id>', 'use this session', or 'add <session-id> as context'. "
    "5. list_projects: use only when the user asks what projects are available or asks to choose a "
    "project filter."
)


class ReclaimLLMError(RuntimeError):
    """Raised when the ReclaimLLM backend request fails."""


@dataclass(frozen=True)
class Credentials:
    server_url: str
    api_key: str


def _load_credentials() -> Credentials:
    """Load MCP credentials, matching hook uploader precedence: config first."""
    cfg = _config.load()
    server_url = (cfg.get("server_url") or os.environ.get("RECLAIMLLM_SERVER_URL") or "").strip()
    api_key = (cfg.get("api_key") or os.environ.get("RECLAIMLLM_API_KEY") or "").strip()

    if not server_url:
        raise ReclaimLLMError("ReclaimLLM server_url missing. Run `rclm-hooks-install --with-mcp`.")
    if not api_key:
        raise ReclaimLLMError("ReclaimLLM api_key missing. Run `rclm-hooks-install --with-mcp`.")
    return Credentials(server_url=server_url.rstrip("/"), api_key=api_key)


def _truncate(value: Any, max_chars: int) -> Any:
    if not isinstance(value, str) or len(value) <= max_chars:
        return value
    return value[:max_chars] + f"\n\n[truncated {len(value) - max_chars} chars]"


def _extract_section(markdown: str | None, heading: str) -> str:
    if not markdown:
        return ""
    marker = f"## {heading}".lower()
    text = markdown.replace("\r\n", "\n")
    lower = text.lower()
    start = lower.find(marker)
    if start < 0:
        return ""
    body_start = text.find("\n", start)
    if body_start < 0:
        return ""
    next_heading = lower.find("\n## ", body_start + 1)
    body = text[body_start + 1 : next_heading if next_heading >= 0 else len(text)]
    return body.strip()


def _highlight_for_session(session: dict[str, Any]) -> str:
    summary = session.get("session_summary")
    highlights = _extract_section(summary, "Highlights")
    if highlights:
        return _truncate(highlights, _MAX_HIGHLIGHT_CHARS)

    happened = _extract_section(summary, "What Happened")
    if happened:
        return _truncate(happened, _MAX_HIGHLIGHT_CHARS)

    description = session.get("description")
    if isinstance(description, str) and description.strip():
        return _truncate(description.strip(), _MAX_HIGHLIGHT_CHARS)

    if isinstance(summary, str) and summary.strip():
        return _truncate(summary.strip(), _MAX_HIGHLIGHT_CHARS)

    return ""


def _frontend_session_url(session_id: str) -> str:
    base = (
        os.environ.get("FRONTEND_URL")
        or os.environ.get("RECLAIMLLM_FRONTEND_URL")
        or "https://reclaimllm.com"
    )
    return f"{base.rstrip('/')}/sessions/{session_id}"


def _session_search_result(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session.get("session_id"),
        "title": session.get("title") or "Untitled session",
        "project_name": session.get("project_name"),
        "model": session.get("model"),
        "started_at": session.get("started_at"),
        "ingested_at": session.get("ingested_at"),
        "highlight": _highlight_for_session(session),
    }


class ReclaimLLMClient:
    def __init__(self) -> None:
        creds = _load_credentials()
        self.server_url = creds.server_url
        self.headers = {
            "X-API-Key": creds.api_key,
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.server_url}{path}"
        async with (
            aiohttp.ClientSession(headers=self.headers) as session,
            session.request(method, url, params=params) as resp,
        ):
            body = await resp.text()
            if resp.status >= 400:
                detail = body
                with suppress(Exception):
                    detail = json.loads(body).get("detail", body)
                raise ReclaimLLMError(f"{method} {path} failed ({resp.status}): {detail}")
            try:
                return json.loads(body)
            except json.JSONDecodeError as exc:
                raise ReclaimLLMError(
                    f"{method} {path} returned non-JSON response: {body[:200]}"
                ) from exc

    async def search_sessions(
        self,
        query: str | None,
        *,
        project_name: str | None,
        file_path: str | None,
        record_type: str | None,
        limit: int,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": max(1, min(limit, _MAX_LIMIT))}
        if query:
            params["text_query"] = query
        if project_name:
            params["project_name"] = project_name
        if file_path:
            params["file_path"] = file_path
        if record_type:
            if record_type not in _VALID_RECORD_TYPES:
                raise ReclaimLLMError(
                    f"record_type must be one of: {', '.join(sorted(_VALID_RECORD_TYPES))}"
                )
            params["record_type"] = record_type

        data = await self._request("GET", "/api/sessions/search", params=params)
        sessions = [_session_search_result(session) for session in data.get("sessions", [])]
        return {"sessions": sessions, "total_returned": len(sessions)}

    async def get_session(self, session_id: str) -> dict[str, Any]:
        data = await self._request(
            "GET",
            f"/api/sessions/{session_id}",
            params={"include_blob": "false"},
        )
        summary = data.get("session_summary") or data.get("description") or ""
        return {
            "session_id": data.get("session_id") or session_id,
            "title": data.get("title") or "Untitled session",
            "project_name": data.get("project_name"),
            "model": data.get("model"),
            "started_at": data.get("started_at"),
            "ingested_at": data.get("ingested_at"),
            "summary": _truncate(summary, 2000),
            "link": _frontend_session_url(session_id),
        }

    async def summarize_session(
        self,
        session_id: str,
        *,
        include_diffs: bool,
        max_diff_lines: int,
        force_regenerate: bool,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/sessions/{session_id}/export-context",
            params={
                "target_tool": "generic",
                "include_diffs": str(include_diffs).lower(),
                "max_diff_lines": max(10, min(max_diff_lines, 200)),
                "force_regenerate": str(force_regenerate).lower(),
            },
        )

    async def list_projects(self) -> dict[str, Any]:
        return await self._request("GET", "/api/sessions/projects")


def build_mcp_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "ReclaimLLM",
        instructions=_MCP_INSTRUCTIONS,
    )

    @mcp.tool()
    async def search_sessions(
        query: str,
        project_name: str | None = None,
        file_path: str | None = None,
        record_type: (Literal["session", "proxy", "browser-chat"] | None) = "session",
        limit: int = _DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        """Search prior ReclaimLLM sessions by intent using backend hybrid semantic plus BM25 search.

        Use only when the user hints that similar prior work may exist, or when the task is a bug fix
        or performance improvement where prior context may help. If the user is unsatisfied with the
        first result set, do not keep retrying with changed search terms.

        If the prompt includes both a file/folder path and semantic terms, pass the file/folder as
        file_path. If the prompt is only about a file/folder history, use search_by_filename instead.
        Use project_name to narrow results only when current project is known.
        Returns session IDs, short titles, and short highlights so the user can decide next steps.
        Do not automatically call summarize_session after this tool.
        """
        client = ReclaimLLMClient()
        return await client.search_sessions(
            query,
            project_name=project_name,
            file_path=file_path,
            record_type=record_type,
            limit=limit,
        )

    @mcp.tool()
    async def search_by_filename(
        file_path: str,
        project_name: str | None = None,
        record_type: (Literal["session", "proxy", "browser-chat"] | None) = "session",
        limit: int = _DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        """Find prior sessions that touched a file or folder path.

        Use when the user asks for changes/history for a specific file or folder and does not provide
        separate semantic search terms, for example "show me all changes in `auth.tsx`" or
        "show me changes under `/somefolder`". If the user includes semantic terms too, such as
        "show all auth fixes in `auth.tsx`", use search_sessions with file_path instead.
        Do not automatically call summarize_session after this tool.
        """
        client = ReclaimLLMClient()
        return await client.search_sessions(
            None,
            project_name=project_name,
            file_path=file_path,
            record_type=record_type,
            limit=limit,
        )

    @mcp.tool()
    async def get_session(session_id: str) -> dict[str, Any]:
        """Return summary metadata and a frontend link for a specific ReclaimLLM session ID.

        Use only when the user asks to look at a particular session by ID.
        """
        client = ReclaimLLMClient()
        return await client.get_session(session_id)

    @mcp.tool()
    async def summarize_session(
        session_id: str,
        include_diffs: bool = True,
        max_diff_lines: int = 50,
        force_regenerate: bool = False,
    ) -> dict[str, Any]:
        """Return reusable markdown context for a ReclaimLLM session using backend export-context flow.

        Use only after an explicit user request such as "summarize <session-id>", "use this session",
        or "add <session-id> as context". Do not infer this request from search results alone.
        """
        client = ReclaimLLMClient()
        return await client.summarize_session(
            session_id,
            include_diffs=include_diffs,
            max_diff_lines=max_diff_lines,
            force_regenerate=force_regenerate,
        )

    @mcp.tool()
    async def list_projects() -> dict[str, Any]:
        """List deterministic ReclaimLLM CLI project names.

        Use only when the user asks to see available projects.
        """
        client = ReclaimLLMClient()
        return await client.list_projects()

    return mcp


def main() -> None:
    try:
        mcp = build_mcp_server()
    except ImportError:
        print(
            'rclm-mcp: missing MCP SDK. Install with `pip install "mcp>=1,<2"`.',
            file=sys.stderr,
        )
        sys.exit(1)
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
