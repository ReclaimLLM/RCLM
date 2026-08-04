"""Verdict thresholds and PRD §9 output assembly.

Aggregates one or more `replay.engine.ReplayResult`s (one per session blob)
into the published output shape: reduction, coverage, by_source,
concentration, unresolvable, reconciliation, cost, provenance,
`cannot_tell_you`. Also handles the `insufficient_data` shape for sessions
or corpora that fail eligibility (PRD §6: "never a number with a warning
attached").
"""

from __future__ import annotations

from collections import Counter

from rclm.replay.eligibility import EligibilityResult, is_eligible_call
from rclm.replay.engine import ReplayResult

CLAIM_BEING_VERIFIED = (
    "replayed tool-result token reduction; not total model input, billing, or live savings"
)

CANNOT_TELL_YOU = (
    "Replay never invokes a model. It cannot observe path changes, retries, or turn-count "
    "effects, which may make real-world impact lower — or negative."
)

RUNTIME_ESTIMATOR_UNDERCOUNT_PCT = 13.56  # Report 1: chars/4 undercounts the real tokenizer
COST_UNAVAILABLE_REASON = "provider usage coverage 2.6% of sessions"

VERDICT_HELPS_MIN_PCT = 15.0
VERDICT_HELPS_MIN_COVERAGE_PCT = 25.0
VERDICT_MARGINAL_MIN_PCT = 5.0
VERDICT_NO_EFFECT_MAX_COVERAGE_PCT = 10.0

_PROVIDER_TOOL_ID_PREFIXES = (
    ("gemini-", "gemini"),
    ("codex-", "codex"),
    ("call_", "codex"),
    ("cursor-", "cursor"),
    ("openclaw-", "openclaw"),
)


def infer_source(blob: dict) -> str:
    """Provider inference mirroring server/tool_call_stats.py's `_infer_provider`
    pattern. Not a shared import — DC-hooks-proxy and ReclaimLLM-server are
    separate deployables — but the same precedence: tool_use_id prefix, then
    blob.provider, then a model-name substring match."""
    for tool_call in blob.get("tool_calls") or []:
        tool_id = str(tool_call.get("tool_use_id") or "")
        for prefix, source in _PROVIDER_TOOL_ID_PREFIXES:
            if tool_id.startswith(prefix):
                return source
    provider = blob.get("provider")
    if isinstance(provider, str) and provider:
        return provider
    model = str(blob.get("model") or "").lower()
    if "claude" in model:
        return "claude"
    if any(marker in model for marker in ("gpt", "codex", "o1", "o3")):
        return "codex"
    if "gemini" in model:
        return "gemini"
    if "cursor" in model:
        return "cursor"
    return "unattributed"


def compute_verdict(reduction_pct: float, coverage_pct: float) -> str:
    """PRD §9 thresholds, anchored to the corrected 25.38% session-only sample."""
    if coverage_pct < VERDICT_NO_EFFECT_MAX_COVERAGE_PCT:
        return "no_effect"
    if reduction_pct < VERDICT_MARGINAL_MIN_PCT:
        return "no_effect"
    if reduction_pct >= VERDICT_HELPS_MIN_PCT and coverage_pct >= VERDICT_HELPS_MIN_COVERAGE_PCT:
        return "helps"
    return "marginal"


def _command_family(tool_call: dict) -> str | None:
    tool_input = tool_call.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    command = tool_input.get("command") or tool_input.get("cmd")
    if not isinstance(command, str) or not command.strip():
        return None
    first = command.strip().split()[0]
    return first.rsplit("/", 1)[-1]


def _file_path(tool_call: dict) -> str | None:
    tool_input = tool_call.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    for key in ("file_path", "path", "filePath"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _concentration(blobs_and_results: list[tuple[dict, ReplayResult]], top_n: int = 5) -> dict:
    families: Counter[str] = Counter()
    files: Counter[str] = Counter()
    for blob, result in blobs_and_results:
        tool_calls = blob.get("tool_calls") or []
        for call in result.shaped_calls:
            if call.index >= len(tool_calls):
                continue
            raw = tool_calls[call.index]
            saved = max(0, call.original_tokens - call.compressed_tokens)
            family = _command_family(raw)
            if family:
                families[family] += saved
            path = _file_path(raw)
            if path:
                files[path] += saved
    return {
        "top_command_families": [
            {"name": name, "tokens_removed": tokens} for name, tokens in families.most_common(top_n)
        ],
        "top_files": [
            {"path": path, "tokens_removed": tokens} for path, tokens in files.most_common(top_n)
        ],
    }


def _by_source(blobs_and_results: list[tuple[dict, ReplayResult]]) -> dict:
    by_source: dict[str, dict] = {}
    for blob, result in blobs_and_results:
        source = infer_source(blob)
        bucket = by_source.setdefault(source, {"tokens_removed": 0, "text_result_tokens": 0})
        bucket["tokens_removed"] += result.tokens_removed
        bucket["text_result_tokens"] += result.text_result_tokens
    for bucket in by_source.values():
        denom = bucket["text_result_tokens"]
        bucket["pct"] = round(100 * bucket["tokens_removed"] / denom, 2) if denom else 0.0
    return by_source


def build_insufficient_data_output(
    funnel: dict, eligibility: EligibilityResult, provenance: dict
) -> dict:
    """PRD §6: fail any gate -> insufficient_data naming the constraint and
    its actual value, never a number with a warning attached."""
    return {
        "verdict": "insufficient_data",
        "claim_being_verified": CLAIM_BEING_VERIFIED,
        "eligibility": funnel,
        "insufficient_data": {
            "constraint": eligibility.failing_constraint,
            "actual_value": eligibility.actual_value,
        },
        "provenance": provenance,
        "cannot_tell_you": CANNOT_TELL_YOU,
    }


def build_result_output(
    blobs_and_results: list[tuple[dict, ReplayResult]],
    funnel: dict,
    provenance: dict,
) -> dict:
    """Assemble the full PRD §9 shape from one or more replayed sessions."""
    tokens_removed = sum(r.tokens_removed for _, r in blobs_and_results)
    text_result_tokens = sum(r.text_result_tokens for _, r in blobs_and_results)
    reduction_pct = (
        round(100 * tokens_removed / text_result_tokens, 2) if text_result_tokens else 0.0
    )

    coverage_by_class = Counter()
    for _, result in blobs_and_results:
        for key, value in result.coverage_by_class().items():
            coverage_by_class[key] += value
    reachable = coverage_by_class["shaped"] + coverage_by_class["uncovered_shell"]
    total_reach = reachable + coverage_by_class["uncoverable"]
    coverage_pct = round(100 * reachable / total_reach, 2) if total_reach else 0.0

    unresolvable_calls = sum(len(r.unresolvable_calls) for _, r in blobs_and_results)
    eligible_calls = sum(1 for _, r in blobs_and_results for c in r.calls if is_eligible_call(c))

    verdict = compute_verdict(reduction_pct, coverage_pct)

    return {
        "verdict": verdict,
        "claim_being_verified": CLAIM_BEING_VERIFIED,
        "eligibility": funnel,
        "reduction": {
            "tokens_removed": tokens_removed,
            "text_result_tokens": text_result_tokens,
            "pct": reduction_pct,
            "unit": "text tool-result tokens",
        },
        "coverage": {
            "pct_of_result_chars_reachable": coverage_pct,
            "by_class": dict(coverage_by_class),
        },
        "by_source": _by_source(blobs_and_results),
        "concentration": _concentration(blobs_and_results),
        "unresolvable": {
            "calls": unresolvable_calls,
            "pct_of_eligible": round(100 * unresolvable_calls / eligible_calls, 2)
            if eligible_calls
            else 0.0,
        },
        "reconciliation": {
            "runtime_estimator": "chars_div_4_v1",
            "measured_undercount_pct": RUNTIME_ESTIMATOR_UNDERCOUNT_PCT,
            "note": "replay figures exceed dashboard figures for this reason",
        },
        "cost": {"available": False, "reason": COST_UNAVAILABLE_REASON},
        "provenance": provenance,
        "cannot_tell_you": CANNOT_TELL_YOU,
    }
