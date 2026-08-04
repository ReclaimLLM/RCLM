"""Eligibility gates for Replay (PRD_Replay_MCP.md §6), grounded in the
whitepaper reports rather than invented thresholds.

Two tiers, both client-side:

- Tier 1 (metadata-only, cheap): checkable from session metadata alone
  (`SessionOut`-shaped dict) without fetching the blob — record type, turns,
  tool-call count, completion state, model present. This is what `replay_eligibility`
  answers cheaply, per the PRD's "call this first" guidance. The
  model-present gate and the requirement that `started_at` (not `ingested_at`)
  define the study window both come from Report 2's live-cohort eligibility
  criteria (docs/whitepaper/report-2-token-savings-data-collection.md §"Eligibility").
- Tier 2 (needs the blob, computed from a `replay.engine.ReplayResult`):
  eligible-call count, the text-token denominator, and the unresolvable
  share.

A session that passes both tiers but has no reduction opportunity is still
eligible — Report 2 (line 66): "A session with no compaction opportunity
remains valid evidence about actual token and turn outcomes." Only genuine
gate failures are excluded; zero-opportunity sessions pass through and let
the verdict layer report `no_effect` plainly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rclm.replay.engine import ReplayResult, ToolCallReplay

MIN_TURNS = 10
MIN_TOOL_CALLS = 10
MIN_ELIGIBLE_CALLS = 5
MIN_TEXT_RESULT_TOKENS = 5_000
MAX_UNRESOLVABLE_SHARE = 0.30

MIN_CORPUS_ELIGIBLE_SESSIONS = 20
MIN_CORPUS_ELIGIBLE_SESSIONS_FULL_CONFIDENCE = 50
MIN_CORPUS_ELIGIBLE_CALLS = 500
MIN_CORPUS_DISTINCT_PROJECTS_CROSS = 2


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    failing_constraint: str | None = None
    actual_value: object = None


def is_eligible_call(call: ToolCallReplay) -> bool:
    """A call the transform core can reach at all — shaped, unresolvable
    (attempted but undetermined), or a call some mechanism looked at but
    didn't reduce. Excludes images and uncoverable tool types."""
    if call.classification in ("shaped", "unresolvable"):
        return True
    return call.classification == "uncovered" and call.is_reachable


def derive_timestamps_from_blob(blob: dict) -> tuple[str | None, str | None]:
    """Fallback started_at/ended_at when the row's own columns are null.

    Mirrors the frontend's client-side fallback
    (ReclaimLLM-frontend/src/components/sessions/MergedSessionDetail.tsx):
    take the min/max of every message's own `timestamp` field. Exists
    because a hooks bug (fixed going forward, see claude_handler.py
    _handle_stop) left many historical rows with null started_at/ended_at
    even though the session, and its blob, are genuinely complete.
    """
    timestamps: list[datetime] = []
    for message in blob.get("messages") or []:
        raw = message.get("timestamp") if isinstance(message, dict) else None
        if not raw:
            continue
        try:
            timestamps.append(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
        except ValueError:
            continue
    if not timestamps:
        return None, None
    return min(timestamps).isoformat(), max(timestamps).isoformat()


def session_metadata_eligibility(
    session: dict,
    *,
    min_turns: int = MIN_TURNS,
    min_tool_calls: int = MIN_TOOL_CALLS,
    blob: dict | None = None,
) -> EligibilityResult:
    """Tier 1: cheap checks against a `SessionOut`-shaped dict, no blob fetch
    required by default.

    `min_turns`/`min_tool_calls` override the default turn-count and
    tool-call-count floors (PRD §6's 10/10, themselves provisional —
    calibrated on a ~2-user corpus). Callers that lower them are trading
    evidence quality for sample size on a small or short-session corpus —
    the caller is responsible for surfacing that choice, not this function;
    it just applies whatever floors it's given.

    `blob` is optional and only fills missing row metadata. Completion comes
    from message timestamps, turn count from user messages, and tool-call
    count from the captured tool-call list. Present row values are never
    second-guessed.
    """
    record_type = session.get("record_type")
    if record_type != "session":
        return EligibilityResult(False, "record_type", record_type)
    ended_at = session.get("ended_at")
    if ended_at is None and blob is not None:
        _derived_started, ended_at = derive_timestamps_from_blob(blob)
    if ended_at is None:
        return EligibilityResult(False, "session_state", "active")
    turn_count = session.get("turn_count")
    if turn_count is None and blob is not None:
        turn_count = sum(
            1
            for message in blob.get("messages") or []
            if isinstance(message, dict) and message.get("role") == "user"
        )
    if turn_count is None:
        return EligibilityResult(False, "turn_count", None)
    if turn_count < min_turns:
        return EligibilityResult(False, "turn_count", turn_count)
    tool_call_count = session.get("tool_call_count")
    if tool_call_count is None and blob is not None:
        tool_call_count = len(blob.get("tool_calls") or [])
    if tool_call_count is None:
        return EligibilityResult(False, "tool_call_count", None)
    if tool_call_count < min_tool_calls:
        return EligibilityResult(False, "tool_call_count", tool_call_count)
    if not session.get("model"):
        return EligibilityResult(False, "model", None)
    return EligibilityResult(True)


def blob_eligibility(result: ReplayResult) -> EligibilityResult:
    """Tier 2: checks requiring the fetched blob's replay result."""
    eligible_calls = [c for c in result.calls if is_eligible_call(c)]
    if len(eligible_calls) < MIN_ELIGIBLE_CALLS:
        return EligibilityResult(False, "eligible_calls", len(eligible_calls))
    text_tokens = result.text_result_tokens
    if text_tokens < MIN_TEXT_RESULT_TOKENS:
        return EligibilityResult(False, "text_result_tokens", text_tokens)
    unresolvable_share = len(result.unresolvable_calls) / len(eligible_calls)
    if unresolvable_share > MAX_UNRESOLVABLE_SHARE:
        return EligibilityResult(False, "unresolvable_share", round(unresolvable_share, 4))
    return EligibilityResult(True)


def session_eligibility(
    session: dict,
    result: ReplayResult | None,
    *,
    min_turns: int = MIN_TURNS,
    min_tool_calls: int = MIN_TOOL_CALLS,
) -> EligibilityResult:
    """Full session-level gate: Tier 1, then Tier 2 if a blob was fetched."""
    tier1 = session_metadata_eligibility(
        session, min_turns=min_turns, min_tool_calls=min_tool_calls
    )
    if not tier1.eligible:
        return tier1
    if result is None:
        return EligibilityResult(True)
    return blob_eligibility(result)


@dataclass(frozen=True)
class CorpusConfidence:
    eligible_sessions: int
    level: str  # "insufficient" | "wide_distribution" | "full"
    note: str


def corpus_confidence(eligible_sessions: int) -> CorpusConfidence:
    """PRD §6 corpus-level guidance: under 20 -> report the count, no
    recommendation; 20-50 -> report the median with an explicit wide-
    distribution note; 50+ -> full confidence."""
    if eligible_sessions < MIN_CORPUS_ELIGIBLE_SESSIONS:
        return CorpusConfidence(
            eligible_sessions,
            "insufficient",
            f"only {eligible_sessions} eligible sessions; count stated, no recommendation",
        )
    if eligible_sessions < MIN_CORPUS_ELIGIBLE_SESSIONS_FULL_CONFIDENCE:
        return CorpusConfidence(
            eligible_sessions,
            "wide_distribution",
            "reporting the median; the distribution is wide at this sample size",
        )
    return CorpusConfidence(eligible_sessions, "full", "sufficient sample for a stated ratio")


def build_funnel(considered: int, excluded: dict[str, int]) -> dict:
    """Every result states sessions considered, eligible, and why the rest
    were excluded (PRD §6, "Always report the funnel")."""
    eligible = considered - sum(excluded.values())
    return {"considered": considered, "eligible": eligible, "excluded": excluded}
