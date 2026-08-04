"""Anchor test: replay must reproduce Report 1's corrected figures.

The report now admits only blobs whose parent row was reverified as
``record_type='session'``. Of the historical 49 files, 39 still resolve and
produce 417,536/1,645,171 = 25.38% with shell compaction alone.
If this test's numbers drift materially from those, either replay or the
published claim is wrong — this is deliberately the highest-stakes test in
this suite (PRD_Replay_MCP.md §11.3).

The tokens_removed numerator must match exactly: it comes from the same
shipped `compact_tool_result`/`apply_filter` functions Report 1 used, with
no room for methodology drift. The denominator is allowed a small tolerance
band: Report 1's exact rule for a handful of ambiguous/error-shaped results
isn't recoverable from the report text alone.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rclm.replay.engine import replay_blob

_EXAMPLE_SESSIONS_DIR = Path(__file__).resolve().parents[4] / "data" / "example_sessions"

pytestmark = pytest.mark.skip(
    reason="data/example_sessions is a sibling monorepo directory, never present in CI; "
    "run manually against a local checkout to reverify Report 1"
)

REPORT1_TOKENS_REMOVED = 417_536
REPORT1_TEXT_RESULT_TOKENS = 1_645_171
REPORT1_PCT = 25.38
REPORT1_CLAUDE_TOKENS_REMOVED = 23_107
REPORT1_CODEX_TOKENS_REMOVED = 394_429

_UNVERIFIED_PARENT_SESSION_IDS = frozenset(
    {
        "019f61ae-631f-76c0-9e2b-902c0f62b512",
        "019f6986-c13d-7ae0-b115-1a420d0c9771",
        "019f6a74-b0bf-7fe1-9565-fb9a03d7b63d",
        "019f6a87-2ab0-78f3-8157-58d8f114386a",
        "019f74c2-2b4a-75d1-be91-1b5512745c80",
        "019f816e-0975-7670-b8f5-6d6d5124f3e3",
        "019f8af1-6a99-7472-9fa6-efd3ee7e3ea3",
        "019f9383-96b8-7270-a34f-df7f4bda4552",
        "019f93af-024e-72d2-9c4b-4b6c2151bf1d",
        "fe9543b4-a730-49db-b5ef-66d6e7ae6777",
    }
)

_DENOMINATOR_TOLERANCE_PCT = 0.5  # see module docstring


def _session_id(path: Path) -> str:
    value = path.stem
    for prefix in ("cursor_output_", "gemini_cli_", "codex_", "claude_", "image_"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _load_verified_session_blobs() -> list[tuple[str, dict]]:
    return [
        (str(path), json.load(open(path)))
        for path in sorted(_EXAMPLE_SESSIONS_DIR.rglob("*.json"))
        if _session_id(path) not in _UNVERIFIED_PARENT_SESSION_IDS
    ]


def _provider_of(path: str) -> str:
    base = os.path.basename(path)
    parts = Path(path).parts
    if "claude" in parts or base.startswith("claude_"):
        return "claude"
    if "codex" in parts or base.startswith("codex_"):
        return "codex"
    if base.startswith("cursor"):
        return "cursor"
    if base.startswith("gemini"):
        return "gemini"
    return "unknown"


def test_overall_reproduces_report1_within_tolerance():
    blobs = _load_verified_session_blobs()
    assert len(blobs) == 39, "expected the corrected 39-session Report 1 sample"

    total_removed = 0
    total_tokens = 0
    for _, blob in blobs:
        result = replay_blob(blob, mechanisms=("shell_compaction",))
        total_removed += result.tokens_removed
        total_tokens += result.text_result_tokens

    # Numerator must match exactly: same shipped compaction functions, same inputs.
    assert total_removed == REPORT1_TOKENS_REMOVED

    denom_diff_pct = (
        100 * abs(total_tokens - REPORT1_TEXT_RESULT_TOKENS) / REPORT1_TEXT_RESULT_TOKENS
    )
    assert denom_diff_pct < _DENOMINATOR_TOLERANCE_PCT, (
        f"denominator {total_tokens} diverged {denom_diff_pct:.3f}% from "
        f"Report 1's {REPORT1_TEXT_RESULT_TOKENS}"
    )

    pct = 100 * total_removed / total_tokens
    assert abs(pct - REPORT1_PCT) < 0.1


def test_per_provider_numerators_match_report1_exactly():
    """Report 1 also states per-provider numerators (23,107 Claude / 394,429
    Codex) — these should match exactly since they're the same underlying
    per-call reductions, just bucketed differently."""
    blobs = _load_verified_session_blobs()
    removed_by_provider: dict[str, int] = {}
    for path, blob in blobs:
        provider = _provider_of(path)
        result = replay_blob(blob, mechanisms=("shell_compaction",))
        removed_by_provider[provider] = removed_by_provider.get(provider, 0) + result.tokens_removed

    assert removed_by_provider.get("claude", 0) == REPORT1_CLAUDE_TOKENS_REMOVED
    assert removed_by_provider.get("codex", 0) == REPORT1_CODEX_TOKENS_REMOVED
