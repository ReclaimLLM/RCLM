"""Entry point for Claude Code's statusLine: rclm-claude-statusline.

Claude Code invokes this binary on every statusline render (new assistant message,
/compact, permission-mode change, or the configured refreshInterval), piping a JSON
payload on stdin and printing whatever this script writes to stdout as the status
line. Rendering must never fail visibly — a bad or missing field is simply omitted
from the line, and this process always exits 0.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

SEGMENT_SEP = "  "

_BAR_WIDTH = 5
_BAR_FILLED = "▰"  # ▰
_BAR_EMPTY = "▱"  # ▱

_PCT_WARN = 70
_PCT_CRIT = 90

# Anthropic's published peak window for Claude.ai Pro/Max: weekdays, Pacific time.
_PEAK_TZ = ZoneInfo("America/Los_Angeles")
_PEAK_START_HOUR = 5
_PEAK_END_HOUR = 11  # exclusive
_PEAK_WEEKDAYS = range(0, 5)  # datetime.weekday(): 0=Mon .. 4=Fri

_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RED = "\033[31m"
_ANSI_DIM = "\033[2m"
_ANSI_RESET = "\033[0m"

GIT_TIMEOUT_S = 0.5


def _colors_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _colorize(text: str, color: str) -> str:
    if not _colors_enabled():
        return text
    return f"{color}{text}{_ANSI_RESET}"


def _pct_color(pct: float) -> str:
    if pct >= _PCT_CRIT:
        return _ANSI_RED
    if pct >= _PCT_WARN:
        return _ANSI_YELLOW
    return _ANSI_GREEN


def _bar(pct: float) -> str:
    filled = max(0, min(_BAR_WIDTH, round((pct / 100) * _BAR_WIDTH)))
    return _BAR_FILLED * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)


def _render_context(payload: dict) -> str | None:
    ctx = payload.get("context_window")
    if not isinstance(ctx, dict):
        return None
    pct = ctx.get("used_percentage")
    if pct is None:
        return None
    pct = float(pct)
    color = _pct_color(pct)
    return f"ctx {_colorize(_bar(pct), color)} {_colorize(f'{pct:.0f}%', color)}"


def _format_reset(resets_at: object) -> str:
    if not isinstance(resets_at, str) or not resets_at:
        return ""
    try:
        reset_dt = datetime.fromisoformat(resets_at.replace("Z", "+00:00"))
    except ValueError:
        return ""
    total_minutes = int((reset_dt - datetime.now(timezone.utc)).total_seconds() // 60)
    if total_minutes <= 0:
        return ""
    hours, minutes = divmod(total_minutes, 60)
    return f" (resets {hours}h{minutes}m)" if hours else f" (resets {minutes}m)"


def _render_rate_limits(payload: dict) -> str | None:
    """Only present for Claude.ai Pro/Max subscribers; absent for API-key billing."""
    limits = payload.get("rate_limits")
    if not isinstance(limits, dict):
        return None
    parts = []
    for key, label in (("five_hour", "5h"), ("seven_day", "wk")):
        window = limits.get(key)
        if not isinstance(window, dict):
            continue
        pct = window.get("used_percentage")
        if pct is None:
            continue
        pct = float(pct)
        color = _pct_color(pct)
        reset = _format_reset(window.get("resets_at"))
        parts.append(f"{label} {_colorize(f'{pct:.0f}%', color)}{reset}")
    return " ".join(parts) if parts else None


def _is_peak(now: datetime) -> bool:
    local = now.astimezone(_PEAK_TZ)
    if local.weekday() not in _PEAK_WEEKDAYS:
        return False
    return _PEAK_START_HOUR <= local.hour < _PEAK_END_HOUR


def _render_peak(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return _colorize("PEAK", _ANSI_YELLOW) if _is_peak(now) else _colorize("OFF-PEAK", _ANSI_GREEN)


def _git_branch(cwd: str) -> str | None:
    if not cwd:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _render_model_branch(payload: dict) -> str | None:
    model = payload.get("model")
    model_name = model.get("display_name") or model.get("id") if isinstance(model, dict) else None

    cwd = payload.get("cwd") or (payload.get("workspace") or {}).get("current_dir", "")
    branch = _git_branch(cwd)

    if model_name and branch:
        return f"{model_name} {_colorize(f'({branch})', _ANSI_DIM)}"
    if model_name:
        return model_name
    if branch:
        return _colorize(f"({branch})", _ANSI_DIM)
    return None


def _render_lines(payload: dict) -> str | None:
    added = payload.get("total_lines_added") or 0
    removed = payload.get("total_lines_removed") or 0
    if not added and not removed:
        return None
    return f"{_colorize(f'+{added}', _ANSI_GREEN)}/{_colorize(f'-{removed}', _ANSI_RED)}"


def render_status_line(payload: dict) -> str:
    segments = [
        _render_context(payload),
        _render_rate_limits(payload),
        _render_peak(),
        _render_model_branch(payload),
        _render_lines(payload),
    ]
    return SEGMENT_SEP.join(segment for segment in segments if segment)


def main() -> None:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {}

    try:
        line = render_status_line(payload)
    except Exception:
        line = ""

    print(line)
    sys.exit(0)


if __name__ == "__main__":
    main()
