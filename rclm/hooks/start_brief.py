"""Local-only layout card for Claude Code's SessionStart hook.

Answers the first question a coding agent asks before any tool call: what is
the repo layout, what sibling repos exist under the same workspace, and what
is the latest migration number. Computed entirely from the local filesystem —
no network call, no server dependency — so it is safe to run synchronously
inside SessionStart's existing budget alongside brevity and the context pack.

This ships only the layout card described in
docs/work_context/PRD_agent_start_here.md §3 — not the ranked "similar past
work" brief, which needs the server-side pack builder and is a later slice.

Opt-in (`start_brief` config flag / --start-brief install flag) and off by
default, same posture as `context_pack`, until shadow-mode data justifies
default-on.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from rclm.hooks._analytics import estimate_tokens

_IGNORE_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "target",
    "vendor",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
    ".vscode",
}

_MIGRATION_DIR_NAMES = ("migrations",)
_MIGRATION_RE = re.compile(r"^(\d{3,})_")

# Defensive caps to keep the card short — this is the cheapest slice, not the
# ranked brief, so it should read as a few lines, not a directory dump.
_MAX_TOP_DIRS = 12
_MAX_SIBLING_REPOS = 8


def _visible_dirs(path: Path) -> list[str]:
    try:
        return sorted(
            p.name
            for p in path.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name not in _IGNORE_DIR_NAMES
        )
    except OSError:
        return []


def _top_level_dirs(cwd: Path) -> list[str]:
    return _visible_dirs(cwd)[:_MAX_TOP_DIRS]


def _sibling_repos(cwd: Path) -> list[str]:
    parent = cwd.parent
    if parent == cwd:
        return []
    try:
        siblings = sorted(
            p.name
            for p in parent.iterdir()
            if p.is_dir()
            and p != cwd
            and not p.name.startswith(".")
            and p.name not in _IGNORE_DIR_NAMES
            and (p / ".git").exists()
        )
    except OSError:
        return []
    return siblings[:_MAX_SIBLING_REPOS]


def _latest_migration(cwd: Path) -> str | None:
    best: tuple[int, str] | None = None
    for dirname in _MIGRATION_DIR_NAMES:
        migrations_dir = cwd / dirname
        try:
            if not migrations_dir.is_dir():
                continue
            for entry in migrations_dir.iterdir():
                if not entry.is_file():
                    continue
                match = _MIGRATION_RE.match(entry.name)
                if not match:
                    continue
                num = int(match.group(1))
                if best is None or num > best[0]:
                    best = (num, entry.name)
        except OSError:
            continue
    return best[1] if best else None


def card_hash(text: str) -> str:
    """Short stable hash identifying a layout-card variant, for tuning-change tracking."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_session_start_context(cwd: str, cfg: dict) -> dict | None:
    """Return {"text", "card_hash", "card_tokens"}, or None if not injecting.

    Callers must wrap this in try/except — like every other mechanism here, a
    failure must never block Claude Code's SessionStart.
    """
    if not cfg.get("start_brief", False):
        return None
    if not cwd:
        return None
    base = Path(cwd)
    if not base.is_dir():
        return None

    top_dirs = _top_level_dirs(base)
    siblings = _sibling_repos(base)
    migration = _latest_migration(base)

    if not top_dirs and not siblings and not migration:
        return None

    lines = ["[rclm] Layout (from your local filesystem — verify before editing):"]
    if top_dirs:
        lines.append(f"Top-level: {', '.join(top_dirs)}")
    if siblings:
        lines.append(f"Sibling repos: {', '.join(siblings)}")
    if migration:
        lines.append(f"Latest migration: {migration}")
    text = "\n".join(lines)

    return {
        "text": text,
        "card_hash": card_hash(text),
        "card_tokens": estimate_tokens(text),
    }
