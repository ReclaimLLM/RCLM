#!/usr/bin/env python3
"""Replay local session JSON files through the shipped reduction mechanisms."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from rclm.hooks.codex_transcript import parse_transcript
from rclm.hooks.tool_result_transform import extract_tool_text_envelope
from rclm.replay.context_cost import replay_recallable_context
from rclm.replay.engine import ALL_MECHANISMS, replay_blob
from rclm.replay.read_request import split_antigravity_view_block
from rclm.replay.tokenizer import count_tokens


def _load(path: Path, *, repair_codex: bool) -> tuple[dict, int, int]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    recovered_calls = 0
    recovered_chars = 0
    if repair_codex and str(blob.get("model") or "").lower().startswith(("gpt-", "codex")):
        transcript_path = blob.get("transcript_path")
        if isinstance(transcript_path, str):
            candidate = Path(transcript_path)
            if not candidate.exists() and transcript_path.startswith("/home-dir/"):
                candidate = Path.home() / transcript_path.removeprefix("/home-dir/")
            if candidate.exists():
                parsed = parse_transcript(str(candidate))
                stored_by_id = {
                    call.get("tool_use_id"): call
                    for call in blob.get("tool_calls") or []
                    if isinstance(call.get("tool_use_id"), str) and call.get("tool_use_id")
                }
                recovered_ids: set[str] = set()
                for call in parsed.tool_calls:
                    stored = stored_by_id.get(call.tool_use_id)
                    if (
                        call.tool_use_id
                        and call.tool_use_id not in recovered_ids
                        and stored is not None
                        and stored.get("tool_result") is None
                    ):
                        stored["tool_result"] = call.tool_result
                        if call.tool_result is not None:
                            recovered_ids.add(call.tool_use_id)
                            recovered_calls += 1
                            recovered_chars += len(
                                call.tool_result
                                if isinstance(call.tool_result, str)
                                else json.dumps(call.tool_result, ensure_ascii=False)
                            )
                # Include transcript-only calls only when the stored blob had
                # no tool calls; otherwise preserve its captured ordering.
                if not stored_by_id and parsed.tool_calls:
                    blob["tool_calls"] = [asdict(call) for call in parsed.tool_calls]
    return blob, recovered_calls, recovered_chars


def benchmark(paths: list[Path], *, repair_codex: bool) -> dict:
    seen: set[str] = set()
    blobs: list[dict] = []
    repaired_calls = 0
    repaired_chars = 0
    for path in paths:
        blob, recovered_calls, recovered_chars = _load(path, repair_codex=repair_codex)
        session_id = str(blob.get("session_id") or path.resolve())
        if session_id in seen:
            continue
        seen.add(session_id)
        blobs.append(blob)
        repaired_calls += recovered_calls
        repaired_chars += recovered_chars

    configurations = {
        "provider_compaction": ("shell_compaction",),
        "range_cache": ("range_cache",),
        "stateful_delta": ("stateful_delta",),
        "hash_dedupe": ("hash_dedupe",),
        "combined": ALL_MECHANISMS,
    }
    reports: dict[str, dict] = {}
    for label, mechanisms in configurations.items():
        denominator = 0
        removed = 0
        shaped_calls = 0
        mechanism_tokens: Counter[str] = Counter()
        tool_tokens: Counter[str] = Counter()
        for blob in blobs:
            replay = replay_blob(blob, mechanisms=mechanisms)
            denominator += replay.text_result_tokens
            removed += replay.tokens_removed
            for call in replay.shaped_calls:
                saved = max(0, call.original_tokens - call.compressed_tokens)
                shaped_calls += 1
                mechanism_tokens[call.mechanism or "unknown"] += saved
                tool_tokens[call.tool_name] += saved
        reports[label] = {
            "text_result_tokens": denominator,
            "tokens_removed": removed,
            "reduction_pct": round(100 * removed / denominator, 2) if denominator else 0.0,
            "shaped_calls": shaped_calls,
            "tokens_removed_by_mechanism": dict(mechanism_tokens.most_common()),
            "tokens_removed_by_tool": dict(tool_tokens.most_common()),
        }

    context_baseline = 0
    context_optimized = 0
    evicted_results = 0
    for blob in blobs:
        context = replay_recallable_context(blob)
        context_baseline += context.baseline_tokens
        context_optimized += context.optimized_tokens
        evicted_results += context.evicted_results
    context_removed = max(0, context_baseline - context_optimized)
    provider_input_tokens = sum(
        value for blob in blobs if isinstance((value := blob.get("total_input_tokens")), int)
    )
    provider_output_tokens = sum(
        value for blob in blobs if isinstance((value := blob.get("total_output_tokens")), int)
    )
    provider_usage_sessions = sum(
        1
        for blob in blobs
        if isinstance(blob.get("total_input_tokens"), int)
        or isinstance(blob.get("total_output_tokens"), int)
    )

    capped_calls = 0
    cap_tokens_removed = 0
    capped_blobs: list[dict] = []
    for blob in blobs:
        capped_blob = copy.deepcopy(blob)
        capped_blobs.append(capped_blob)
        for call in capped_blob.get("tool_calls") or []:
            if str(call.get("tool_name") or "").lower() != "view_file":
                continue
            envelope = extract_tool_text_envelope(
                call.get("tool_name") or "", call.get("tool_result")
            )
            if envelope is None:
                continue
            split = split_antigravity_view_block(envelope.text)
            if split is None:
                continue
            _prefix, block, _suffix = split
            lines = block.splitlines(keepends=True)
            if len(lines) <= 200:
                continue
            capped_calls += 1
            truncated_block = "".join(lines[:200])
            cap_tokens_removed += count_tokens(block) - count_tokens(truncated_block)
            prefix, suffix = split[0], split[2]
            first_line = int(lines[0].split(":", 1)[0])
            capped_end = first_line + 199
            prefix = re.sub(
                r"(?m)^Showing lines \d+ to \d+$",
                f"Showing lines {first_line} to {capped_end}",
                prefix,
            )
            updated_text = prefix + truncated_block + suffix
            _wire, structured, _model_text = envelope.replace(updated_text)
            call["tool_result"] = structured
            if isinstance(call.get("tool_input"), dict):
                call["tool_input"]["EndLine"] = capped_end

    capped_denominator = 0
    capped_removed = 0
    for blob in capped_blobs:
        replay = replay_blob(blob)
        capped_denominator += replay.text_result_tokens
        capped_removed += replay.tokens_removed
    original_denominator = reports["combined"]["text_result_tokens"]
    combined_optimized = max(0, capped_denominator - capped_removed)

    return {
        "sessions": len(blobs),
        "codex_results_recovered": repaired_calls,
        "codex_result_chars_recovered": repaired_chars,
        "captured_provider_usage": {
            "sessions_with_usage": provider_usage_sessions,
            "input_tokens": provider_input_tokens,
            "output_tokens": provider_output_tokens,
        },
        "immediate_tool_result_replay": reports,
        "recallable_context_model": {
            "unit": "tool-step context-token proxy; not provider billing",
            "baseline_tokens": context_baseline,
            "optimized_tokens": context_optimized,
            "tokens_removed": context_removed,
            "reduction_pct": (
                round(100 * context_removed / context_baseline, 2) if context_baseline else 0.0
            ),
            "evicted_results": evicted_results,
        },
        "pretool_view_cap": {
            "max_lines": 200,
            "calls_reduced": capped_calls,
            "immediate_tokens_removed": cap_tokens_removed,
        },
        "combined_with_pretool_view_cap": {
            "original_text_result_tokens": original_denominator,
            "optimized_text_result_tokens": combined_optimized,
            "tokens_removed": max(0, original_denominator - combined_optimized),
            "reduction_pct": (
                round(100 * (original_denominator - combined_optimized) / original_denominator, 2)
                if original_denominator
                else 0.0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--repair-codex", action="store_true")
    args = parser.parse_args()
    files: list[Path] = []
    for path in args.paths:
        files.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
    print(json.dumps(benchmark(files, repair_codex=args.repair_codex), indent=2))


if __name__ == "__main__":
    main()
