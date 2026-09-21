"""Model repeated context reads when older tool results become recallable stubs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from rclm.hooks.recallable_artifacts import ArtifactProvenance, build_stub
from rclm.hooks.tool_result_transform import extract_tool_text_envelope
from rclm.hooks.tool_semantics import semantic_target
from rclm.replay.tokenizer import count_tokens

MIN_EVICTABLE_CHARS = 600


@dataclass(frozen=True)
class ContextCostResult:
    """Tool-step proxy for cumulative LLM input context, not provider billing."""

    baseline_tokens: int
    optimized_tokens: int
    tokens_removed: int
    evicted_results: int

    @property
    def reduction_pct(self) -> float:
        if not self.baseline_tokens:
            return 0.0
        return round(100 * self.tokens_removed / self.baseline_tokens, 2)


def replay_recallable_context(blob: dict) -> ContextCostResult:
    """Estimate repeated reads of superseded results at each later tool step.

    A result stays verbatim until another call with the same explicit semantic
    target completes.  Thereafter, subsequent steps carry a compact immutable
    reference.  This deliberately excludes first-delivery savings and model
    output tokens.
    """
    calls = blob.get("tool_calls") or []
    results: list[tuple[int, int, str | None]] = []
    next_by_key: dict[str, int] = {}
    next_index: list[int | None] = [None] * len(calls)

    for index in range(len(calls) - 1, -1, -1):
        call = calls[index]
        name = call.get("tool_name") or ""
        key = semantic_target(name, call.get("tool_input"))
        if key is not None:
            scoped_key = f"{name.lower()}:{key}"
            next_index[index] = next_by_key.get(scoped_key)
            next_by_key[scoped_key] = index

    for index, call in enumerate(calls):
        envelope = extract_tool_text_envelope(call.get("tool_name") or "", call.get("tool_result"))
        if envelope is None:
            continue
        text = envelope.text
        raw_tokens = count_tokens(text)
        successor = next_index[index]
        if successor is None or len(text) < MIN_EVICTABLE_CHARS:
            results.append((raw_tokens * max(0, len(calls) - index - 1), 0, None))
            continue
        raw_reads = successor - index
        later_reads = max(0, len(calls) - successor - 1)
        digest = hashlib.sha256(text.encode("utf-8", errors="surrogateescape")).hexdigest()
        handle = f"rclm://artifact/sha256/{digest}"
        stub = build_stub(
            handle,
            len(text),
            ArtifactProvenance(
                provider=str(blob.get("model") or "unknown"),
                tool_name=name,
                tool_use_id=call.get("tool_use_id"),
                target=key,
            ).bounded(),
        )
        baseline = raw_tokens * max(0, len(calls) - index - 1)
        optimized = raw_tokens * raw_reads + count_tokens(stub) * later_reads
        results.append((baseline, optimized, key))

    baseline_tokens = sum(baseline for baseline, _optimized, _key in results)
    optimized_tokens = sum(
        optimized if key is not None else baseline for baseline, optimized, key in results
    )
    return ContextCostResult(
        baseline_tokens=baseline_tokens,
        optimized_tokens=optimized_tokens,
        tokens_removed=max(0, baseline_tokens - optimized_tokens),
        evicted_results=sum(1 for _baseline, _optimized, key in results if key is not None),
    )


__all__ = ["ContextCostResult", "replay_recallable_context"]
