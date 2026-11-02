"""Query-time ranking for the Start Brief "similar past work" brief.

Reads the local SQLite cache rclm.hooks.brief_cache syncs in the background --
never fetches over the network on this path, since UserPromptSubmit is on
Claude Code's critical path. Scoring follows
docs/work_context/PRD_agent_start_here.md §3.2. The layout card
(rclm.hooks.start_brief) is a separate, unconditional mechanism gated only by
the shared `start_brief` config flag -- it keeps firing independently of this
module's confidence gate.

Simplification from the PRD's "same_category/agent" term: there is no current
notion of the prompt's own category to compare against, so this only checks
agent_client against the current client ("claude", since this module is
Claude Code only).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rclm import _config
from rclm.hooks import session_store
from rclm.hooks._analytics import estimate_tokens
from rclm.hooks._project_key import cache_db_path, derive_project_key

_CONFIDENCE_THRESHOLD = 0.35  # tau; starting value, tune in shadow mode (PRD §4)
_MIN_FILES = 2
_SEED_SESSIONS = 3
_MAX_DISPLAY_SESSIONS = 3
_MAX_COCHANGE_DISPLAY = 6
_MAX_CHANGED_SINCE_DISPLAY = 3
_BUDGET_SECONDS = 0.150  # hard deadline; fail open past this (PRD §3.2/§7)
_MAX_CHARS = 2500
_RECENCY_HALF_LIFE_DAYS = 30.0
_MIN_TASK_SHAPED_TOKENS = 4
_JACCARD_REINJECT_THRESHOLD = 0.3
_CURRENT_AGENT = "claude"

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "to",
        "of",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "this",
        "that",
        "it",
        "i",
        "you",
        "we",
        "can",
        "do",
        "does",
        "did",
        "please",
        "yes",
        "no",
        "ok",
        "okay",
        "thanks",
        "thank",
        "it's",
        "lets",
        "let's",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_/.\-]+")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[_\-/]")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        for piece in _CAMEL_SPLIT_RE.split(raw):
            piece = piece.strip().lower()
            if len(piece) < 2 or piece in _STOPWORDS:
                continue
            tokens.append(piece)
    return tokens


def _is_task_shaped(prompt: str) -> bool:
    return len(_tokenize(prompt)) >= _MIN_TASK_SHAPED_TOKENS


def _recency_score(ended_at: str | None, now: datetime) -> float:
    if not ended_at:
        return 0.0
    try:
        ended = datetime.fromisoformat(str(ended_at).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if ended.tzinfo is None:
        ended = ended.replace(tzinfo=timezone.utc)
    age_days = max((now - ended).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (age_days / _RECENCY_HALF_LIFE_DAYS)


def _path_hits(tokens: set[str], paths_text: str) -> float:
    if not paths_text or not tokens:
        return 0.0
    path_tokens = set(_tokenize(paths_text))
    if not path_tokens:
        return 0.0
    hits = len(tokens & path_tokens)
    return min(hits / max(len(tokens), 1), 1.0)


def _score_session(
    *,
    bm25_norm: float,
    path_hits: float,
    ended_at: str | None,
    agent_client: str | None,
    now: datetime,
) -> float:
    """0.55*bm25 + 0.20*path_hits + 0.15*recency + 0.10*same_agent.

    Missing terms are dropped and the remaining weights renormalized (PRD
    open-question Q10) rather than scored as zero, so sessions with gaps in
    their data (e.g. no agent_client) aren't penalized relative to sessions
    with complete data.
    """
    terms: list[tuple[float, float]] = [
        (0.55, bm25_norm),
        (0.20, path_hits),
        (0.15, _recency_score(ended_at, now)),
    ]
    if agent_client:
        terms.append((0.10, 1.0 if agent_client == _CURRENT_AGENT else 0.0))
    total_weight = sum(w for w, _ in terms)
    if total_weight <= 0:
        return 0.0
    return sum(w * v for w, v in terms) / total_weight


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rank(
    db_path: Path, *, prompt: str, cwd: str, deadline: float
) -> tuple[str, set[str], int] | None:
    if time.monotonic() > deadline:
        return None
    tokens = _tokenize(prompt)
    if not tokens:
        return None
    token_set = set(tokens)

    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.DatabaseError:
        return None
    conn.row_factory = sqlite3.Row
    try:
        match_query = " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))
        try:
            fts_rows = conn.execute(
                "SELECT session_id, paths, bm25(sessions_fts) AS rank FROM sessions_fts "
                "WHERE sessions_fts MATCH ? ORDER BY rank LIMIT 20",
                (match_query,),
            ).fetchall()
        except sqlite3.OperationalError:
            return None
        if not fts_rows or time.monotonic() > deadline:
            return None

        max_abs_rank = max(abs(r["rank"]) for r in fts_rows) or 1.0
        session_rows = conn.execute(
            "SELECT session_id, title, category, agent_client, ended_at, summary FROM sessions"
        ).fetchall()
        session_by_id = {r["session_id"]: r for r in session_rows}
        now = datetime.now(timezone.utc)

        scored: list[tuple[float, sqlite3.Row, str]] = []
        for r in fts_rows:
            sess = session_by_id.get(r["session_id"])
            if sess is None:
                continue
            bm25_norm = min(abs(r["rank"]) / max_abs_rank, 1.0)
            score = _score_session(
                bm25_norm=bm25_norm,
                path_hits=_path_hits(token_set, r["paths"] or ""),
                ended_at=sess["ended_at"],
                agent_client=sess["agent_client"],
                now=now,
            )
            scored.append((score, sess, r["paths"] or ""))
        if not scored:
            return None
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored[0][0] < _CONFIDENCE_THRESHOLD:
            return None
        if time.monotonic() > deadline:
            return None

        seeds = scored[:_SEED_SESSIONS]
        seed_ids = [s[1]["session_id"] for s in seeds]
        seed_scores = {s[1]["session_id"]: s[0] for s in seeds}

        placeholders = ",".join("?" * len(seed_ids))
        file_rows = conn.execute(
            f"SELECT session_id, path, sha_after FROM session_files "
            f"WHERE session_id IN ({placeholders})",
            seed_ids,
        ).fetchall()

        file_scores: dict[str, float] = {}
        file_session: dict[str, str] = {}
        file_sha: dict[str, str | None] = {}
        for row in file_rows:
            path = row["path"]
            file_scores[path] = file_scores.get(path, 0.0) + seed_scores.get(row["session_id"], 0.0)
            file_session.setdefault(path, row["session_id"])
            file_sha[path] = row["sha_after"]

        if time.monotonic() > deadline:
            return None

        if file_scores:
            base_paths = list(file_scores.keys())
            cochange_placeholders = ",".join("?" * len(base_paths))
            cochange_rows = conn.execute(
                f"SELECT path, neighbour_path, sessions_coedited FROM cochange "
                f"WHERE path IN ({cochange_placeholders})",
                base_paths,
            ).fetchall()
            for row in cochange_rows:
                npath = row["neighbour_path"]
                coedited = int(row["sessions_coedited"] or 0)
                file_scores[npath] = file_scores.get(npath, 0.0) + 0.5 * coedited

        if file_scores:
            busy_placeholders = ",".join("?" * len(file_scores))
            busy_rows = conn.execute(
                f"SELECT path, busyness FROM files WHERE path IN ({busy_placeholders})",
                list(file_scores.keys()),
            ).fetchall()
            busyness_by_path = {r["path"]: int(r["busyness"] or 0) for r in busy_rows}
            for path in list(file_scores.keys()):
                file_scores[path] += 0.15 * math.log1p(busyness_by_path.get(path, 0))
                if any(t in path.lower() for t in token_set):
                    file_scores[path] += 0.2

        ranked_files = sorted(file_scores.items(), key=lambda kv: kv[1], reverse=True)
        if len(ranked_files) < _MIN_FILES:
            return None
        if time.monotonic() > deadline:
            return None

        top_files = [p for p, _ in ranked_files[:8]]

        changed_since: list[str] = []
        for path in top_files:
            sha = file_sha.get(path)
            if not sha:
                continue
            seed_sid = file_session.get(path)
            seed_sess = session_by_id.get(seed_sid) if seed_sid else None
            ended_at = seed_sess["ended_at"] if seed_sess else None
            if not ended_at:
                continue
            try:
                ended_dt = datetime.fromisoformat(str(ended_at).replace("Z", "+00:00"))
            except ValueError:
                continue
            if ended_dt.tzinfo is None:
                ended_dt = ended_dt.replace(tzinfo=timezone.utc)
            full_path = Path(path) if os.path.isabs(path) else Path(cwd) / path
            try:
                st = full_path.stat()
            except OSError:
                continue
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
            if mtime <= ended_dt:
                continue
            try:
                actual_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()
            except OSError:
                continue
            if actual_hash != sha:
                changed_since.append(path)

        if time.monotonic() > deadline:
            return None

        text = _format_brief(
            seeds=seeds, top_files=top_files, file_sha=file_sha, changed_since=changed_since
        )
        return text, set(top_files), len(seeds)
    finally:
        conn.close()


def _format_brief(
    *,
    seeds: list[tuple[float, sqlite3.Row, str]],
    top_files: list[str],
    file_sha: dict[str, str | None],
    changed_since: list[str],
) -> str:
    lines = ["[rclm] Similar past work in this project (verify before editing):"]
    for _score, sess, _paths in seeds[:_MAX_DISPLAY_SESSIONS]:
        date = (sess["ended_at"] or "")[:10]
        title = sess["title"] or "Untitled session"
        meta = ", ".join(p for p in (sess["category"] or "", sess["agent_client"] or "") if p)
        summary = (sess["summary"] or "")[:140]
        line = f'- {date} "{title}"'
        if meta:
            line += f" ({meta})"
        if summary:
            line += f": {summary}"
        lines.append(line)

    cochange_only = [p for p in top_files if p not in file_sha][:_MAX_COCHANGE_DISPLAY]
    if cochange_only:
        lines.append("Files that usually change together: " + ", ".join(cochange_only))
    if changed_since:
        lines.append(
            "Changed since that work: " + ", ".join(changed_since[:_MAX_CHANGED_SINCE_DISPLAY])
        )

    text = "\n".join(lines)
    if len(text) > _MAX_CHARS:
        text = text[: _MAX_CHARS - 1] + "…"
    return text


def build_prompt_context(
    cwd: str, session_id: str, prompt: str, cfg: dict, *, prior_events: list[dict]
) -> dict[str, Any] | None:
    """Return {"text"}, or None if not injecting.

    Callers must wrap this in try/except -- like every other mechanism here,
    a failure must never block Claude Code's UserPromptSubmit, and a timeout
    past the self-imposed budget must fail open (inject nothing), not block.
    """
    if not cfg.get("start_brief", False):
        return None
    if not cwd or not session_id:
        return None

    deadline = time.monotonic() + _BUDGET_SECONDS

    prior_prompts = sum(
        1 for e in prior_events if e.get("event_type") == "UserPromptSubmit" and e.get("prompt")
    )
    is_first_prompt = prior_prompts == 0
    if not is_first_prompt and not _is_task_shaped(prompt):
        return None

    creds = _config.resolve_credentials()
    if creds is None:
        return None
    project_key = derive_project_key(cwd)
    db_path = cache_db_path(project_key, creds.api_key)
    if not db_path.exists():
        return None

    result = _rank(db_path, prompt=prompt, cwd=cwd, deadline=deadline)
    if result is None:
        return None
    text, file_paths, session_count = result

    if not is_first_prompt:
        state = session_store.read_start_brief_state(session_id)
        last_files = set(state.get("last_files") or [])
        if _jaccard(last_files, file_paths) >= _JACCARD_REINJECT_THRESHOLD:
            return None  # too similar to the last injected set -- don't repeat

    session_store.write_start_brief_state(
        session_id, {"last_files": sorted(file_paths), "updated_at": _now_iso()}
    )
    session_store.append_event(
        session_id,
        {
            "event_type": "StartBriefRankInjected",
            "brief_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "brief_tokens": estimate_tokens(text),
            "session_count": session_count,
            "file_count": len(file_paths),
        },
    )
    return {"text": text}
