"""Shared handoff-advisor defaults and config parsing."""

from __future__ import annotations

DEFAULT_TOKEN_THRESHOLD = 160_000
DEFAULT_TOOL_CALL_THRESHOLD = 120
TOKEN_THRESHOLD_KEY = "handoff_advisor_token_threshold"
TOOL_CALL_THRESHOLD_KEY = "handoff_advisor_tool_call_threshold"


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def thresholds_from_config(cfg: dict) -> tuple[int, int]:
    return (
        _positive_int(cfg.get(TOKEN_THRESHOLD_KEY), DEFAULT_TOKEN_THRESHOLD),
        _positive_int(cfg.get(TOOL_CALL_THRESHOLD_KEY), DEFAULT_TOOL_CALL_THRESHOLD),
    )
