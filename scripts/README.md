# Session corpus benchmark

`benchmark_session_corpus.py` replays captured session JSON files through the shipped ReclaimLLM compression core. Use it to compare individual mechanisms, measure their combined reduction, and check whether a change improves both recent and older session cohorts.

The benchmark is offline and read-only. It does not call a model, re-execute historical tools, update captured sessions, or write to the ReclaimLLM database.

## Requirements

- Run commands from `DC-hooks-proxy`.
- Install the project dependencies with `uv sync`.
- Provide one or more captured session JSON files or directories containing JSON files.
- If you use `--repair-codex`, the transcript path referenced by each Codex session must still exist locally.

## Prepare a corpus

Keep downloaded sessions outside the package directory. For example:

```bash
mkdir -p ../data/example_sessions/recent50
aws s3 cp s3://<bucket>/<session-key>.json ../data/example_sessions/recent50/<session-id>.json --profile root
```

Repeat the copy for each selected session. The script searches directories recursively and ignores duplicate `session_id` values, so nested provider or cohort directories are supported.

Do not commit downloaded sessions unless they have been reviewed and intentionally approved as fixtures. Session blobs can contain source code, prompts, tool output, file paths, and other sensitive data.

## Run the benchmark

Benchmark one directory:

```bash
uv run python scripts/benchmark_session_corpus.py ../data/example_sessions/recent50
```

Benchmark multiple cohorts together:

```bash
uv run python scripts/benchmark_session_corpus.py ../data/example_sessions/recent50 ../data/example_sessions/older
```

Repair missing Codex results from locally available transcripts before replay:

```bash
uv run python scripts/benchmark_session_corpus.py ../data/example_sessions/recent50 --repair-codex
```

The repair happens only in memory. It pairs stored calls with standard and custom Codex tool-call outputs but does not modify the input JSON or transcript files. Sessions whose transcript path is unavailable remain unchanged.

To retain the JSON report:

```bash
uv run python scripts/benchmark_session_corpus.py ../data/example_sessions/recent50 --repair-codex > /tmp/reclaimllm-recent50-benchmark.json
```

## What it compares

The report evaluates these configurations independently:

| Report key | Replay mechanisms |
| --- | --- |
| `provider_compaction` | `shell_compaction`, including recognized provider shaping and edit receipts |
| `range_cache` | Range-aware repeated-read reduction |
| `stateful_delta` | Changes between repeated commands, searches, and listings |
| `hash_dedupe` | Exact repeated-result references |
| `combined` | All shipped replay mechanisms in their normal precedence order |

It also models the 200-line Antigravity `view_file` pre-tool cap and the repeated-context effect of replacing superseded results with recallable artifact stubs.

## Reading the output

Important top-level fields:

- `sessions`: unique sessions included in the report.
- `codex_results_recovered` and `codex_result_chars_recovered`: results restored in memory by `--repair-codex`.
- `captured_provider_usage`: provider-reported input and output totals when present, plus the number of sessions with that data.
- `immediate_tool_result_replay`: original text-result tokens, removed tokens, percentage reduction, affected calls, and breakdowns by mechanism and tool.
- `pretool_view_cap`: calls and immediate tokens affected by the modeled 200-line Antigravity cap.
- `combined_with_pretool_view_cap`: combined result reduction after adding that pre-tool model.
- `recallable_context_model`: a tool-step proxy for repeatedly carrying older results through later context.

Compare `text_result_tokens`, `tokens_removed`, and `reduction_pct` within the same corpus. Use `tokens_removed_by_mechanism` and `tokens_removed_by_tool` to find where a change produced its reduction.

## Interpretation limits

The benchmark measures text tool-result reduction. It does not prove:

- lower billed provider input or output tokens;
- unchanged model decisions or task success;
- fewer turns, retries, or follow-up reads;
- prompt-cache savings;
- equivalent results on a different provider or workload.

`recallable_context_model` is explicitly a repeated tool-step context proxy, not provider billing telemetry. Use shadow-mode production telemetry and paired live task evaluations before enabling a new lossy mechanism broadly.

Historical Cursor sessions without captured result fields cannot be repaired safely. Unsupported tools, failures, images, and ambiguous structured results remain unchanged by design.

## Validate benchmark changes

```bash
uv run pytest rclm/tests/replay rclm/tests/hooks/test_tool_result_transform.py rclm/tests/hooks/test_recallable_artifacts.py rclm/tests/hooks/test_result_delta.py
```

```bash
uv run ruff check scripts/benchmark_session_corpus.py rclm/replay rclm/hooks
```
