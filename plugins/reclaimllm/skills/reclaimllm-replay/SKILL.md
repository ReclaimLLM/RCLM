---
name: reclaimllm-replay
description: >
  Read-only verification of ReclaimLLM's published tool-result token-reduction
  claim (25.08%) against the user's own captured sessions. TRIGGER when the
  user asks: "would compression have helped here", "verify the token savings
  claim on my sessions", "does compression actually help me", "replay this
  session", "check if the 25% number holds for my data", "is compression
  worth turning on for me", or otherwise wants to independently confirm the
  compression mechanism's effect rather than take the published number on
  faith. Covers the four rclm-mcp replay tools: replay_eligibility,
  replay_session, replay_corpus, replay_compare.
  DO NOT TRIGGER for general session search or memory recall — use the
  reclaimllm-memory skill for that. Do not trigger to explain what
  compression is in the abstract; only for verifying its effect on real
  captured data.
license: Apache-2.0
metadata:
  author: ReclaimLLM
  version: "0.1.0"
  category: ai-observability
  tags: "compression, verification, token-reduction, replay, mcp"
compatibility: Requires Python 3.10+, pip install rclm, authenticated ReclaimLLM credentials in ~/.reclaimllm/config.json or RECLAIMLLM_SERVER_URL and RECLAIMLLM_API_KEY environment variables, and an MCP-capable host.
---

# ReclaimLLM Replay

Replay reproduces RCLM's compression claim against the user's own captured
sessions, on their machine, in the same units the claim was published in:
**text tool-result tokens removed**. It is strictly read-only — it never
writes to the database, never re-executes a historical command, and never
calls a model. The claim being verified is narrow and must be stated exactly
this way when relaying results:

> a replayed **tool-result token reduction** — not a claim about total model
> input, billing, or live user savings.

## Core Pattern

1. Always call `replay_eligibility` first. It is a cheap, metadata-only
   check — no blob fetch — and answers "is this worth replaying" before any
   real computation runs.
2. If eligible, call `replay_session` (one session) or `replay_corpus` (a
   filtered window) depending on what the user asked about.
3. Only call `replay_compare` when the user explicitly wants multiple
   mechanism configurations compared against the same corpus in one call.
4. Narrate the result plainly. Do not decide for the user whether the number
   is "good" — state it, state the funnel, state `cannot_tell_you`.

## Tool Routing

- `replay_eligibility(session_id?, days?, source?, model_family?, project?, session_category?, limit?, min_turns?, min_tool_calls?)` —
  call this before either tool below. Pass `session_id` to check one
  session; omit it to check a corpus window. Returns `eligible`, the failing
  constraint if not, and the funnel (`considered`/`eligible`/`excluded`).
- `replay_session(session_id?, mechanisms?, min_turns?, min_tool_calls?)` —
  one session. Defaults to the caller's most recent complete session.
  `mechanisms` defaults to all three (`range_cache`, `shell_compaction`,
  `hash_dedupe`); pass a subset only if the user wants to isolate one
  mechanism.
- `replay_corpus(days=30, source="all", model_family?, project?, session_category?, mechanisms?, limit?, min_turns?, min_tool_calls?)` —
  a filtered set of sessions. `days` is a session-activity window (measured
  on when each session started, not when it was ingested). `limit` is the
  target fully eligible session count. Replay fetches up to four times that
  many recent `session` records (capped at 200), applies both eligibility
  tiers in order, and stops at `limit` eligible sessions or scan exhaustion.
- `replay_compare(days=30, source="all", ..., configs?, min_turns?, min_tool_calls?)` —
  same corpus, multiple mechanism sets, one call, so bundles stay
  attributable. Only use when the user is comparing configurations, not for
  a single verification.

All four tools accept `min_turns` (default **5**) and `min_tool_calls`
(default **5**). PRD §6's documented floors are 10/10; the tool layer
defaults lower so short-session users still get a number. This is a
visible, stated deviation — every result's `provenance` carries
`min_turns_applied`/`min_tool_calls_applied` (or the equivalent
`*_applied` fields on `replay_eligibility`'s response), and it must be
surfaced to the user whenever a result is reported: "at min_turns=1,
min_tool_calls=1 (both far below the documented floor of 10)". At these
defaults the two gates barely filter anything — treat a low `considered`/
`eligible` count as itself informative (most sessions are short or
tool-call-light), not as evidence of a broken query. Do not treat requests
to lower these further as routine — flag the tradeoff (smaller/shorter
sessions are noisier evidence) before applying it, the same way you would
for any other loosened filter.

Single-session lookups (`replay_eligibility(session_id=...)`,
`replay_session`) also fill missing completion, turn-count, and tool-count
metadata from the captured blob. Corpus replay uses the same fallback after
a record enters the activity window. This fallback is *not* applied during
the initial corpus-window screening
(`replay_corpus`/`replay_compare`/corpus-mode `replay_eligibility`), which
still requires the row's own `started_at` to consider a session a
candidate at all — a low corpus count can still mean "row not backfilled
yet," not "genuinely no eligible sessions."

## Honesty Rules — do not soften these

- The field is `reduction`, never "savings". Always state the unit
  ("text tool-result tokens") alongside any number.
- A `no_effect` or `insufficient_data` verdict is reported **flatly**. An LLM
  narrating a negative result tends to hedge it into sounding positive —
  resist that. `no_effect` is a good outcome to report: it saves the user
  from enabling a transform on their tool calls for nothing.
- Always surface the response's `cannot_tell_you` line to the user whenever
  the verdict is `helps` — it states what replay cannot observe (path
  changes, retries, turn-count effects) that could make real-world impact
  lower, or negative.
- Never state or imply a dollar figure. `cost.available` is always `false`
  in this version; explain why if asked (provider usage coverage is too low
  to be defensible).
- If `reconciliation` is present, mention it when the user compares replay's
  number against anything the dashboard showed them — replay's figure is
  expected to be *higher* because the dashboard's runtime estimator
  (`chars/4`) undercounts the real tokenizer; that is not a bug.
- Always state the eligibility funnel (sessions considered vs eligible vs
  excluded, and why) alongside any reduction number. A low eligible count is
  itself the finding — it means compression is largely irrelevant to how
  this user works, and they should know that before enabling anything.

## Setup

Replay uses the same local `rclm-mcp` server as ReclaimLLM memory:

```bash
pip install rclm
rclm-hooks-install --with-mcp
```

The MCP server reads credentials from `~/.reclaimllm/config.json`, falling
back to `RECLAIMLLM_SERVER_URL` and `RECLAIMLLM_API_KEY`.

## Available MCP Tools

| Tool | Use |
|------|-----|
| `replay_eligibility` | Cheap pre-check — call first, always. |
| `replay_session` | Replay one captured session. |
| `replay_corpus` | Replay a filtered window of sessions, aggregated. |
| `replay_compare` | Replay the same corpus under multiple mechanism sets. |

## Guardrails

- Do not call `replay_session`/`replay_corpus` without first calling
  `replay_eligibility` — an ineligible session or corpus should be refused
  with the specific constraint, never handed a number.
- Do not average, sum, or otherwise combine Replay's output with live
  telemetry from the dashboard — they measure different things.
- Do not lower the eligibility bar or keep re-querying with looser filters
  to force a number into existence. If nothing is eligible, that is the
  answer.
- Do not treat Replay as proof of live savings. It is a deterministic
  counterfactual over already-captured tool results, not an observation of
  the agent's actual behavior under compression.
