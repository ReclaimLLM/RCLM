"""Provenance for a replay result: what produced it, so identical inputs
produce byte-identical output (PRD_Replay_MCP.md §11.4) and so a session's
`replay_corpus`/`replay_eligibility` line can state which mechanisms and
tokenizer were simulated, rather than implying it observed the session's
actual recorded mode (which isn't reachable client-side; see
docs/work_context/PRD_Replay_MCP.md, "Effective-mode note").
"""

from __future__ import annotations

from rclm.hooks.updater import installed_version
from rclm.replay.eligibility import MIN_TOOL_CALLS, MIN_TURNS
from rclm.replay.engine import Mechanism
from rclm.replay.tokenizer import TOKENIZER_NAME


def build_provenance(
    mechanisms: tuple[Mechanism, ...],
    *,
    min_turns: int = MIN_TURNS,
    min_tool_calls: int = MIN_TOOL_CALLS,
) -> dict:
    return {
        "rclm_version": installed_version(),
        "tokenizer": TOKENIZER_NAME,
        "mechanisms": list(mechanisms),
        "range_cache_mode": "simulated" if "range_cache" in mechanisms else "not_simulated",
        "min_turns_applied": min_turns,
        "min_tool_calls_applied": min_tool_calls,
    }
