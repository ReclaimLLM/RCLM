"""Brevity instruction injection for Claude Code's SessionStart hook.

Advisory only — nothing enforces model compliance, and every request in a session
carries the injected instruction, so this is opt-in (`brevity` config flag / --brevity
install flag) and off by default. Savings are not measurable per-call or per-session
(there's no before/after to compare against); the session record carries `enabled`,
`instruction_hash`, and `instruction_tokens` only, and any cost-reduction estimate is a
cohort-level comparison computed later from `output_tokens_per_turn` across sessions.

Claude Code only. Gemini CLI and Codex CLI's SessionStart hooks have no equivalent
`additionalContext` injection surface in this codebase today (see gemini_handler.py and
codex_handler.py) — this module is not wired into either, so brevity is a no-op there
until such a surface exists.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from rclm.hooks._analytics import estimate_tokens

BREVITY_INSTRUCTION_KEY = "brevity_instruction"

DEFAULT_INSTRUCTION = (
    "[rclm] Be concise in your responses.\n"
    "\n"
    "## Skip\n"
    "- Preamble that restates the request before acting\n"
    "- Summarizing a change when the diff already shows it\n"
    '- Unrequested alternatives, caveats, or "next steps" suggestions\n'
    '- Pleasantries and filler: "Sure," "I can help," "Hope this helps"\n'
    '- Meta-talk: "Let\'s break this down," "First, I will think about..."\n'
    "- Markdown headers or bullet lists on short answers — plain prose is enough\n"
    "\n"
    "## Keep\n"
    "- Reasoning and error analysis, as thorough as they need to be\n"
    "- Anything the user explicitly asked for, in the depth they asked for\n"
    "- Direct output: start with the core facts or steps, not a lead-in\n"
)

# Existing project instructions containing any of these are assumed to already cover
# brevity; injecting on top would just restate the same guidance twice. Tuned to skip
# when in doubt — double instructions waste tokens, a missed real instruction wastes more.
_EXISTING_GUIDANCE_MARKERS = (
    "concise",
    "conciseness",
    "brevity",
    "be brief",
    "terse",
    "no preamble",
    "minimal output",
    "short answers",
    "keep responses short",
)

_GUIDANCE_FILENAMES = ("CLAUDE.md", "AGENTS.md")

# Defensive cap only — real brevity guidance always appears well within this range.
# Guards against pathological file sizes, not a tuned "read enough to detect" budget.
_MAX_GUIDANCE_FILE_BYTES = 100_000


def _has_existing_brevity_guidance(cwd: str) -> bool:
    if not cwd:
        return False
    base = Path(cwd)
    for filename in _GUIDANCE_FILENAMES:
        path = base / filename
        try:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")[:_MAX_GUIDANCE_FILE_BYTES]
        except OSError:
            continue
        lowered = text.lower()
        if any(marker in lowered for marker in _EXISTING_GUIDANCE_MARKERS):
            return True
    return False


def instruction_hash(text: str) -> str:
    """Short stable hash identifying an instruction variant, for tuning-change tracking."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# Collapses runs of spaces/tabs only — newlines are preserved so DEFAULT_INSTRUCTION's
# markdown (headers, bullet lists) survives injection instead of being flattened to one line.
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def build_session_start_context(cwd: str, cfg: dict) -> dict | None:
    """Return {"text", "instruction_hash", "instruction_tokens"}, or None if not injecting.

    Callers must wrap this in try/except — like every other mechanism here, a failure
    must never block Claude Code's SessionStart.
    """
    if not cfg.get("brevity", False):
        return None
    if _has_existing_brevity_guidance(cwd):
        return None

    text = cfg.get(BREVITY_INSTRUCTION_KEY) or DEFAULT_INSTRUCTION
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text).strip()
    if not text:
        return None

    return {
        "text": text,
        "instruction_hash": instruction_hash(text),
        "instruction_tokens": estimate_tokens(text),
    }
