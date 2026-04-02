"""Per-session JSONL accumulation in ~/.reclaimllm/sessions/{session_id}.jsonl.

Claude Code runs hooks sequentially per session, so no file locking is needed.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SESSIONS_DIR = Path.home() / ".reclaimllm" / "sessions"


def _session_path(session_id: str) -> Path:
    return _SESSIONS_DIR / f"{session_id}.jsonl"


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


def cleanup(session_id: str) -> None:
    """Delete the session file. No-ops if already gone."""
    path = _session_path(session_id)
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
