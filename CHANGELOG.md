# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v0.1.26] — 2026-08-07

### Added
- Added recursive env-file DLP scanning with size limits, symlink skipping, and fail-closed `DLPRedactionError` when a recognized secret source cannot be inspected (`rclm/hooks/dlp.py`)
- Added secret-shape heuristics (key markers, known token prefixes, entropy, credential URLs, PEM private keys) and public-key exclusions for ambient output scrubbing (`rclm/hooks/dlp.py`)
- Added shape-preserving `maybe_redact_value`, upload-time `redact_json_payload`, and `reconcile_captured_tool_results` so redacted hook output is what gets captured and uploaded (`rclm/hooks/dlp.py`, `rclm/_uploader.py`, provider handlers)
- Added shared `/api/sessions/filter` client path (`filter_sessions`) for non-semantic listing used by `search_by_filename`, `file_brief`, handoff, and replay candidate scans (`rclm/mcp_server.py`)
- Added 600s MCP tool timeouts on install for JSON MCP configs and Codex `tool_timeout_sec` (`rclm/mcp_install.py`)
- Added `rclm-bench-adapter` console script entry (`pyproject.toml`)
- Added `--no-dlp` / BooleanOptionalAction for DLP so fresh installs default on while preserving an explicit saved opt-out (`rclm/hooks/installer.py`, `rclm/_config.py`)

### Changed
- Env-file DLP is now on by default (`DEFAULT_DLP_ENABLED`); README documents `--dlp` / `--no-dlp` behavior
- Direct env-file reads redact every assignment (`include_all_values`); ambient scans stay conservative and skip fixture/template env filenames
- Bash DLP now uses `read_cache.parse_shell_read` plus `shlex` tokenization, and blocks `env`/`printenv`/`set` dumps that reference env paths (`rclm/hooks/dlp.py`)
- Provider PostToolUse DLP now fails closed on env-file access (withhold/redact) instead of silently passing through (`claude_handler.py`, `codex_handler.py`, `cursor_handler.py`, `gemini_handler.py`, `openclaw_handler.py`)
- `search_sessions` requires a non-empty semantic `query`; empty-query callers move to `filter_sessions` (`rclm/mcp_server.py`)
- Replay MCP defaults raised from `min_turns`/`min_tool_calls` 1/1 to 5/5; candidate scan cap lowered to 100 via the filter endpoint (`rclm/mcp_server.py`, replay skill)
- Failed-upload quarantine now writes the already-redacted payload instead of re-serializing without DLP (`rclm/_uploader.py`)
- MCP HTTP client calls use a 600s total timeout (`rclm/mcp_server.py`)

### Fixed
- Fixed multiline quoted env values and escaped characters inside inline comments being misparsed during DLP env loading (`rclm/hooks/dlp.py`)
- Fixed relative env-file Read paths resolving incorrectly when cwd was present (`rclm/hooks/dlp.py`)

### Security
- Unreadable or oversized env trees no longer fail open: recognized env-file tool access is blocked or withheld, and upload raises `DLPRedactionError` without uploading or quarantining unredacted data (`rclm/hooks/dlp.py`, `rclm/_uploader.py`, provider handlers)

---

## [v0.1.25] — 2026-08-05

### Added
- Added Antigravity CLI hook integration: capture-only `Stop` hook via a new `rclm-antigravity-hooks` command, installing into `.agents/hooks.json` (`--local`) or `~/.gemini/config/hooks.json` (global) (`rclm/hooks/antigravity_handler.py`, `rclm/hooks/antigravity_transcript.py`, `rclm/hooks/installer.py`, `pyproject.toml`)
- Added `_config.resolved_server_url()`, `resolved_api_key()`, and `resolve_credentials()` as a single shared credential-resolution helper (`rclm/_config.py`)
- Added an MCP Tools reference table and a "Replay: verifying token savings" section to the README

### Changed
- Antigravity is now part of the default `rclm-hooks-install` provider set alongside Claude, Gemini, Codex, and Cursor; OpenClaw remains opt-in (`rclm/hooks/installer.py`)
- `codex_transcript.py` now captures `developer` and `system` role messages instead of silently dropping them alongside user/assistant — `developer` carries real injected instructions (AGENTS.md, memory-tool guidance, multi-agent-mode directives), not boilerplate

### Fixed
- Fixed rclm hook entries at a matcher an older version used (e.g. Codex `PreToolUse`/`PostToolUse` `"Bash"` → `""`) surviving reinstall as orphaned duplicates instead of being migrated in place (`rclm/hooks/installer.py`)
- Fixed every provider's Stop/SessionEnd handler leaving the module-level aiohttp session open across the event loop's teardown, which printed an "Unclosed client session"/"Unclosed connector" `ResourceWarning` to stderr on every single Stop invocation — confirmed via a real subprocess repro against a live session transcript (`rclm/hooks/claude_handler.py`, `codex_handler.py`, `gemini_handler.py`, `cursor_handler.py`, `openclaw_handler.py`, `antigravity_handler.py`)

### Security
- Unified `server_url`/`api_key` resolution (env var takes precedence over `config.json`) across `mcp_server.py`, `convert.py`, `proxy/start.py`, and `_uploader.py` — previously inconsistent, with some call sites resolving precedence the opposite way
- A missing `api_key` no longer results in silently sending an unauthenticated upload request; `_uploader.py` now prints a clear "not authenticated" message and quarantines the record locally instead

---

## [v0.1.24] — 2026-08-03

### Added
- Added `rclm-mcp` replay tools (`replay_eligibility`, `replay_session`, `replay_corpus`, `replay_compare`) that reproduce the shipped compression mechanisms over captured sessions or a filtered corpus and report real tool-result token reduction — read-only, no model calls, no re-execution of historical commands (`rclm/mcp_server.py`)
- Added a shared, fail-open tool-result transform core and native Claude Code, Codex, and Cursor adapters for recognized shell-output compaction without requiring gateway traffic (`rclm/hooks/tool_result_transform.py`, provider handlers)
- Added structured-result preservation, image/error/ambiguity gates, provider contract tests, and recorded-session replay evidence for the expanded token-reduction surface
- Added provider-neutral session attribution for `rclm-compress` wrappers via `--session-id`/`--encoded-command`, including raw/compressed character telemetry and an explicitly labelled runtime token estimator (`rclm/compress/cli.py`, `rclm/compress/runner.py`)
- Added native Cursor `PreToolUse`/`PostToolUse`/`SessionStart` handling — shell-input compaction, MCP text-result dedupe, and hook-policy bootstrap/snapshot — replacing the prior generic passthrough recording (`rclm/hooks/cursor_handler.py`)

### Changed
- Expanded conservative command coverage to POSIX `cat`, `nl`, and `sed`, plus simple PowerShell `Get-Content`; Codex and Cursor now treat exec compaction as supported by their effective hook policy (`rclm/_config.py`, `rclm/hooks/compress.py`)
- Preserved exact shell command text across wrapper execution by passing URL-safe base64-encoded UTF-8 instead of allowing the outer shell to consume quotes, pipes, or chains
- Codex `PostToolUse` output replacement now pairs `continue: false` with `decision: block` instead of `decision: block` alone, matching Codex's documented model-visible feedback contract (`rclm/hooks/codex_handler.py`)
- Codex `PreToolUse`/`PostToolUse` correlation now keys on `tool_use_id`, falling back to the legacy `turn_id` match
- `--shadow-mode` is now a proper on/off flag (`BooleanOptionalAction`) instead of `store_true`, so `--no-shadow-mode` can return to enforcement; help text updated to describe cross-client compression rather than Claude-only (`rclm/hooks/installer.py`)
- Removed the `record_type` parameter from `search_sessions`/`search_by_filename`; the MCP surface is fixed to `record_type="session"` and now raises on direct session lookups for any other record type (`rclm/mcp_server.py`)
- Varied the canned elision/cap/dedupe messages (test-output capping, pytest/JS/Go pass summaries, dedupe pointers) with randomly chosen plain-language phrasings instead of one fixed string each time (`rclm/compress/filters/test.py`, `rclm/hooks/dedupe.py`)
- `rclm-update` now also refreshes and reports the cached organization hook policy, not just redaction settings (`rclm/update.py`)
- `bootstrap.policy_snapshot()` now includes schema version, capture client version, and capture provider fields

### Fixed
- Fixed `Stop` (which fires at the end of every turn, not just session end) wiping the whole session event log each time, so any turn after the first lost `SessionStart` and fell back to `started_at=now()`, corrupting stored duration and start/end timestamps on multi-turn sessions; full cleanup now happens once, in `SessionEnd` (`rclm/hooks/claude_handler.py`)
- Fixed the same handler double-counting `pre_tool_use`/`post_tool_use`/`tool_failure` counters and mechanism-savings totals across turns, since the event log is now session-cumulative rather than per-turn
- Fixed `bootstrap.fetch()` treating an explicit `org_hook_policy: null` the same as a missing key, so a user who lost org membership no longer keeps a stale cached policy

### Removed
- Removed the leftover `rclm-cursor-hooks TEMP event=... payload=...` debug print that logged full hook payloads to stderr on every Cursor hook call (`rclm/hooks/cursor_handler.py`)

### Performance
- Changed generic repeat-collapse and head/tail compaction to retain bounded line state instead of materializing a second full output-sized line list (`rclm/compress/filters/shell.py`)

### Deps
- Added `tiktoken>=0.8,<1` as a direct dependency (`pyproject.toml`, `uv.lock`)

---

## [v0.1.23] — 2026-07-29

### Added
- Added bounded `changed_files` metadata to MCP session-search results so agents can identify likely implementation files and explicitly follow their latest history with `search_by_filename` (`rclm/mcp_server.py`)
- Added inclusive `date_from` and exclusive `date_to` ingestion-date filters to both `search_sessions` and `search_by_filename`, with shared `YYYY-MM-DD` validation and invalid-range handling (`rclm/mcp_server.py`)
- Added focused MCP tests for changed-file result shaping, backend filter forwarding, valid date windows, malformed dates, and reversed or empty ranges (`rclm/tests/test_mcp_server.py`)

- Added the `transfer_session` MCP tool for moving a complete captured session between Claude Code and Codex as a versioned local JSON artifact, preserving captured messages, tool calls/results, file diffs, and metadata without re-executing historical actions (`rclm/mcp_server.py`, `rclm/_session_transfer.py`)
- Added the `signals` MCP tool plus best-effort Signal enrichment for `file_brief` and automatic acted-state reporting after successful handoffs, so agents can surface workflow-efficiency evidence and close the loop when a prescribed handoff is used (`rclm/mcp_server.py`)
- Added tests for streamed transfer integrity, size limits, secure artifact lifecycle, MCP registration, Signal shaping and failure isolation, and handoff acted-state reporting (`rclm/tests/test_session_transfer.py`, `rclm/tests/test_mcp_server.py`)

### Changed
- Updated MCP routing instructions and the bundled ReclaimLLM memory skill to use semantic search first, then filename history for likely changed files, and to translate relative periods such as “last 3 weeks” into concrete ingestion-date bounds (`rclm/mcp_server.py`, `plugins/reclaimllm/skills/reclaimllm-memory/SKILL.md`)
- Clarified that `convert-session` and `handoff` produce compact context documents while `transfer_session` loads the full captured record, and documented transfer limits, artifact expiry, image-lifecycle installation, and provider-specific rewrite behavior (`README.md`, `plugins/reclaimllm/skills/reclaimllm-memory/SKILL.md`)
- Expanded MCP routing instructions to reserve full-session transfer and workflow Signals for explicit user requests (`rclm/mcp_server.py`, `plugins/reclaimllm/skills/reclaimllm-memory/SKILL.md`)

### Security
- Wrote transfer artifacts atomically with owner-only permissions, enforced configurable byte ceilings without silent truncation, verified backend byte counts and SHA-256 manifests, and added bounded expiry cleanup (`rclm/_session_transfer.py`, `rclm/mcp_server.py`)

### Performance
- Streamed full-session downloads in bounded chunks instead of buffering complete captures in memory (`rclm/mcp_server.py`, `rclm/_session_transfer.py`)

### Deps
- Updated the lockfile with Pillow 11.3.0 for the existing optional `images` and development extras (`uv.lock`)

---

## [v0.1.22] — 2026-07-26

### Added
- Added shared, provider-agnostic image detection and downscale-at-capture for oversized tool-result images — native file reads, MCP `ImageContent` blocks, and MCP `CallToolResult` wrappers with `structuredContent` mirrors — gated behind `--image-lifecycle`/`--image-max-dim`, off by default (`rclm/hooks/image_lifecycle.py`, `rclm/hooks/installer.py`)
- Added `estimate_image_tokens()` using published Anthropic and OpenAI per-provider image-token formulas, isolated from the existing `estimate_tokens()` relied on by other mechanisms (`rclm/hooks/_analytics.py`)
- Added session-scoped, LRU-capped stale-image eviction tracking keyed by (tool, page/URL, viewport), reporting shadow-only estimated savings via a modeled prompt-cache-write cost — never applied, regardless of `shadow_mode` (`rclm/hooks/image_eviction.py`, `rclm/hooks/session_store.py`)
- Added the `image_downscale` (measured) and `image_eviction` (estimated) mechanisms to the `PostToolUse` pipeline, with a real output rewrite on Claude and measurement-only reporting on Codex, whose own MCP output-rewrite contract does not apply the change (`rclm/hooks/claude_handler.py`, `rclm/hooks/codex_handler.py`)
- Added tests covering image detection/downscale across all supported wire shapes, eviction-tracking supersession logic, Claude/Codex measurement paths, and the measured-vs-estimated mechanism classification (`rclm/tests/hooks/test_image_lifecycle.py`, `rclm/tests/hooks/test_image_eviction.py`, `rclm/tests/hooks/test_handler.py`, `rclm/tests/hooks/test_codex_handler.py`, `rclm/tests/hooks/test_analytics.py`)

### Changed
- Widened Codex's `PreToolUse`/`PostToolUse` hook matcher from `Bash`-only to unrestricted, and switched Codex's tool routing from an `!= "Bash"` check to matching the unambiguous `mcp__<server>__<tool>` prefix, so shell tool-name variants (e.g. `exec_command`) can't be misrouted away from the existing DLP/read-cache/dedupe pipeline (`rclm/hooks/installer.py`, `rclm/hooks/codex_handler.py`)
- Changed Codex's `_build_tool_calls()` to preserve the real recorded `tool_name` instead of hardcoding `"Bash"` for every tool call (`rclm/hooks/codex_handler.py`)
- Reformatted the default brevity instruction into a structured `## Skip` / `## Keep` markdown list and changed whitespace collapsing to preserve newlines instead of flattening the instruction to one line (`rclm/hooks/brevity.py`)

### Performance
- Bounded worst-case image-decode latency with a header-only pixel-count check before raster decode, instead of a wall-clock timer (`rclm/hooks/image_lifecycle.py`)

### Deps
- Added Pillow as an optional `images` extra (`rclm[images]`) and to the `dev` extras, rather than a hard dependency, since most installs never enable `--image-lifecycle` (`pyproject.toml`)

---

## [v0.1.21] — 2026-07-24

### Added
- Added conservative, range-aware read caching for native and exact shell file reads across Claude, Codex, and Gemini, with interval tracking, edit invalidation, bounded session state, shadow mode, and per-file measured savings (`rclm/hooks/read_cache.py`, `rclm/compress/read_cache_cli.py`, `rclm/hooks/{claude,codex,gemini}_handler.py`)
- Added bounded Claude hook-health diagnostics that compare transcript tool calls with observed lifecycle events and warn once when tool hooks are missing (`rclm/hooks/claude_handler.py`, `rclm/hooks/session_store.py`)
- Added opt-in Claude brevity instruction wiring and `SessionEnd` registration for final session-state cleanup (`rclm/hooks/claude_handler.py`, `rclm/hooks/installer.py`)

### Changed
- Extended mechanism telemetry with measured/estimated classification, raw and compressed token counts, per-file attribution, and provider tool-transformation attachment (`rclm/hooks/_analytics.py`, `rclm/hooks/{claude,codex,gemini}_handler.py`, `rclm/compress/runner.py`)
- Made large native reads advance to the first unseen range and made shell-command detection explicitly parse supported POSIX segments instead of using substring matching (`rclm/hooks/compress.py`, `rclm/hooks/read_cache.py`)
- Expanded package compatibility through Python 3.14 (`pyproject.toml`, `uv.lock`)

### Fixed
- Fixed Claude structured tool responses and exact Bash reads so range-cache replacements reach the model despite Claude ignoring `PostToolUse.updatedToolOutput` (`rclm/hooks/claude_handler.py`, `rclm/compress/read_cache_cli.py`)
- Fixed Claude `Stop` cleanup erasing cache state and earlier savings by separating turn cleanup from `SessionEnd` cleanup and persisting cumulative mechanism rollups (`rclm/hooks/claude_handler.py`, `rclm/hooks/session_store.py`)
- Fixed range cache and hash deduplication from double-claiming the same result, while preserving transformation telemetry on uploaded Codex and Gemini tool calls (`rclm/hooks/{claude,codex,gemini}_handler.py`)

### Performance
- Replaced full repeated file output with compact unchanged-range notices while passing ambiguous commands, changed files, binary files, and unsupported output through unchanged (`rclm/hooks/read_cache.py`)

### Deps
- Refreshed `uv.lock` to the current lock format and Python 3.14-compatible artifact metadata without changing declared runtime dependencies (`uv.lock`)

---

## [v0.1.20] — 2026-07-24

### Added
- Added session-scoped, conservative PostToolUse hash deduplication for large identical tool results, including ANSI/path/timestamp normalization, error bypasses, bounded LRU state, and per-call compression telemetry (`rclm/hooks/dedupe.py`, `rclm/hooks/{claude,codex,gemini}_handler.py`)
- Added per-mechanism compression savings events and extended tool-call payloads with compression strategy, before/after token estimates, savings, ratios, and shadow-mode status (`rclm/_models.py`, `rclm/hooks/_analytics.py`)
- Added conservative pytest, Jest/Vitest, Go, and Cargo test-output filters with failure preservation, ambiguity passthrough, output caps, and `test_filter` telemetry (`rclm/compress/filters/test.py`, `rclm/compress/runner.py`)
- Added `--dedupe` installation flag and nested `compression` settings for deduplication and test-filter thresholds (`rclm/_config.py`, `rclm/hooks/installer.py`)
- Added upload include/exclude folder filters, including a default exclusion for Codex memory files (`rclm/hooks/redaction.py`, `rclm/hooks/installer.py`)
- Added default-on Claude handoff advisor with configurable token/tool-call thresholds and once-per-session guidance (`rclm/hooks/handoff_advisor.py`, `rclm/hooks/claude_handler.py`)
- Added a once-daily, detached `rclm-update` scheduler after successful primary Claude, Codex, and Gemini session uploads, with local locking and an update log (`rclm/hooks/updater.py`, `rclm/hooks/{claude,codex,gemini}_handler.py`)

### Changed
- Expanded package Python compatibility to include Python 3.14 (`pyproject.toml`, `uv.lock`)
- Migrated legacy flat compression configuration keys to the nested `compression` object while retaining read compatibility for existing installations (`rclm/_config.py`, `rclm/update.py`)
- Routed `go test` through the command wrapper and made test filtering exit-code-aware so unknown non-zero runner output passes through unchanged (`rclm/hooks/compress.py`, `rclm/compress/cli.py`)
- Moved automatic update execution from login/install completion to primary session completion, so session hooks never wait for PyPI, pip, or hook reinstallation (`rclm/login.py`, `rclm/hooks/installer.py`, `rclm/hooks/updater.py`)

### Fixed
- Fixed Codex and Gemini PostToolUse replacement responses to use their supported block/deny contracts when DLP or deduplication rewrites output (`rclm/hooks/{codex,gemini}_handler.py`)

### Security
- Prevented uploads from the local Codex memories directory by default and allowlisted folder capture when configured (`rclm/hooks/redaction.py`, `rclm/_uploader.py`)

---

## [v0.1.19] — 2026-07-21

### Added
- Added `rclm-claude-statusline` entry point: a Claude Code statusline showing context window usage, five-hour/weekly rate limits (Claude.ai subscribers only), PEAK/OFF-PEAK status against Anthropic's published weekday 5–11am Pacific window, model name, git branch, and lines changed — sourced entirely from Claude Code's `statusLine` stdin payload with no network calls (`rclm/hooks/statusline_handler.py`, `pyproject.toml`)
- Added `--statusline`/`--no-statusline` flag to `rclm-hooks-install`, on by default and persisted across reinstalls the same way `--compress` is (`rclm/hooks/installer.py`)

### Changed
- `rclm-hooks-install` now backs up any pre-existing non-rclm Claude Code `statusLine` into `~/.reclaimllm/config.json` before replacing it, and `rclm-hooks-uninstall` restores it automatically (`rclm/hooks/installer.py`, `rclm/hooks/uninstaller.py`)
- `rclm-hooks-install --codex` now prints a one-line hint to enable Codex's native `context-used`/`context-remaining`/`five-hour-limit`/`weekly-limit` status line items via `/statusline`, since Codex has no external-script hook to target (`rclm/hooks/installer.py`)

### Fixed
- Fixed `rclm-hooks-uninstall` silently removing nothing in any real (`pip install rclm`) installation: hook and statusline commands were matched by literal string prefix (`command.startswith("rclm-")`), but `_resolve_binary` returns an absolute path (e.g. `/home/user/.venv/bin/rclm-claude-hooks`) whenever the binary is found on PATH; matching is now done against the binary's basename via a shared `_command_belongs_to_rclm` helper (`rclm/hooks/installer.py`, `rclm/hooks/uninstaller.py`)

---

## [v0.1.18] — 2026-07-20

### Added
- Added `rclm-login` CLI entry point for standalone sign-in without installing hooks (`rclm/login.py`, `pyproject.toml`)
- Added shared `rclm/auth.py` module consolidating the browser device-flow key handoff and `/api/whoami` key validation, used by `rclm-hooks-install`, `rclm-login`, and `rclm-mcp`
- Added support for pasting an API key directly into the terminal while waiting for the browser callback, via `select()`-based stdin multiplexing (POSIX only; falls back to browser-only wait on Windows) (`rclm/auth.py`)
- Added tests for the new auth module and `rclm-login` entrypoint, including the stdin-paste flow (`rclm/tests/test_auth.py`, `rclm/tests/test_login.py`)

### Changed
- Changed `--with-mcp`, `--read-cache`, `--loop-breaker`, and `--compress` to be enabled by default on `rclm-hooks-install`; pass `--no-<flag>` to opt out, and the opt-out persists across reinstalls (`rclm/hooks/installer.py`)
- Changed `rclm-mcp` auth-required and 401/403 error messages to point at `rclm-login` instead of the retired `rclm-hooks-install --with-mcp` credential path (`rclm/mcp_server.py`)

### Fixed
- Fixed installer tests that asserted compression/RTK-removal were off by default, which no longer matched the new default-on behavior; added matching coverage for the `--no-compress`/`--no-read-cache`/`--no-loop-breaker`/`--no-with-mcp` opt-outs (`rclm/tests/hooks/test_installer.py`)

### Security
- Moved the local CLI callback URL out of the query string and into the URL fragment so it's never transmitted to the server or captured by page-load analytics (`rclm/auth.py`)

---

## [v0.1.16] — 2026-07-07

### Added
- Added provider-scoped MCP installation so `rclm-hooks-install --with-mcp` only writes MCP configs for the selected providers (`rclm/mcp_install.py`, `rclm/hooks/installer.py`)

### Changed
- Updated the bundled ReclaimLLM plugin MCP config to expose `reclaimllm` at the top level instead of wrapping it in `mcpServers` (`plugins/reclaimllm/.mcp.json`)

### Fixed
- Fixed Claude MCP install paths to use `~/.claude.json` globally and `.claude/mcp.json` locally instead of Claude settings files ignored by Claude Code CLI (`rclm/mcp_install.py`, `rclm/tests/hooks/test_installer.py`)
- Fixed Codex `apply_patch` parsing to retain unified diff content for added, deleted, and updated files (`rclm/hooks/codex_transcript.py`)

### Deps
- Refreshed `uv.lock` metadata and dev-extra serialization without changing project dependency declarations (`uv.lock`)

---

## [v0.1.15] — 2026-06-15

### Added
- Added shared aiohttp TLS helpers that build a verified SSL context, honor explicit CA bundle environment variables, and add certifi roots when local Python CA discovery is incomplete (`rclm/_http.py`)
- Added focused tests for explicit CA bundle selection and certifi fallback behavior (`rclm/tests/test_http.py`)

### Changed
- Updated uploader, session conversion, and MCP client HTTP calls to use the shared verified TLS connector (`rclm/_uploader.py`, `rclm/convert.py`, `rclm/mcp_server.py`)
- Expanded package Python compatibility to include Python 3.13 (`pyproject.toml`, `uv.lock`)

### Fixed
- Fixed upload failures on machines whose Python/OpenSSL environment could not find a valid local issuer certificate for `api.reclaimllm.com` (`rclm/_uploader.py`, `rclm/_http.py`)

### Security
- Preserved certificate verification while improving CA root reliability, avoiding unsafe `ssl=False` fallbacks for ReclaimLLM API calls (`rclm/_http.py`)

### Deps
- Added `certifi>=2026.2.25` and refreshed lockfile metadata with Python 3.13 wheel support (`pyproject.toml`, `uv.lock`)

---

## [v0.1.14] — 2026-06-10

### Added
- Added ReclaimLLM plugin marketplace catalogs for Codex, Claude, and Cursor, plus a root marketplace catalog for sharing the `reclaimllm` plugin from `DC-hooks-proxy` (`.agents/plugins/marketplace.json`, `.codex-plugin/marketplace.json`, `.claude-plugin/marketplace.json`, `.cursor-plugin/marketplace.json`, `marketplace.json`)
- Added the `reclaimllm` agent plugin with Codex, Claude, and Cursor manifests, local MCP server configuration, and a ReclaimLLM memory skill for persistent captured-session search (`plugins/reclaimllm/**`)

### Changed
- Documented agent plugin installation and MCP authentication workflow in the package README (`README.md`)

### Removed
- Removed obsolete standalone debug/reproduction scripts for proxy and Codex hook experiments (`debug_proxy.py`, `repro_codex.py`, `test_codex_schema.py`, `test_main_codex.py`)

---

## [v0.1.13] — 2026-05-11

### Added
- Added ReclaimLLM MCP server (`rclm-mcp`) providing `search_sessions`, `search_by_filename`, `get_session`, `summarize_session`, and `list_projects` tools for AI session recall (`rclm/mcp_server.py`, `pyproject.toml`)
- Added MCP server installation and registration logic for Claude, Gemini, Cursor, and Codex CLI clients (`rclm/mcp_install.py`)
- Added `--with-mcp` flag to `rclm-hooks-install` and `rclm-update` to automate MCP server registration (`rclm/hooks/installer.py`, `rclm/update.py`)
- Added `timestamp` field to `FileDiff` model to capture exact timing of file edits across all supported providers (`rclm/_models.py`, `rclm/hooks/**`)

### Changed
- Refactored hook installer to replace ALL existing rclm hook entries for a given matcher/event instead of appending, preventing duplicate registrations during re-installs (`rclm/hooks/installer.py`)
- Improved rclm command detection in the installer to handle absolute paths and spaces more robustly (`rclm/hooks/installer.py`)

### Fixed
- Fixed duplicate rclm hook commands being registered when the installer was run multiple times or with different path styles (`rclm/hooks/installer.py`)

### Deps
- Added `mcp>=1.0,<2` dependency to support the ReclaimLLM MCP server (`pyproject.toml`, `uv.lock`)

---

## [v0.1.12] — 2026-05-05

### Added
- Added Cursor hook CLI entry point and installer support for `.cursor/hooks.json` (`pyproject.toml`, `rclm/hooks/installer.py`)
- Added registration for all documented Cursor hook events, including agent, tool, shell, MCP, file edit, Tab edit, session, and compaction hooks (`rclm/hooks/installer.py`)
- Added Cursor historical sync discovery and parsing for `~/.cursor/projects/*/agent-transcripts/*/*.jsonl` (`rclm/hooks/historical_sync.py`)

### Changed
- Updated default hook installation and CLI help to include Cursor alongside Claude, Gemini, Codex, and OpenClaw (`rclm/hooks/installer.py`)
- Extended hook command absolute-path rewriting and merge logic to support Cursor's flat hook schema (`rclm/hooks/installer.py`)
- Extended historical sync provider dispatch and CLI flags to include `--cursor` (`rclm/hooks/historical_sync.py`)

### Fixed
- Fixed Cursor historical sync fallback model to use `cursor-unknown` instead of a Claude-derived default (`rclm/hooks/historical_sync.py`)

---

## [v0.1.11] — 2026-05-01

### Added
- Added OpenClaw capture support with a new `rclm-openclaw-hooks` entry point, plugin generator, and transcript parser (`rclm/hooks/openclaw_handler.py`, `rclm/hooks/openclaw_plugin.py`, `rclm/hooks/openclaw_transcript.py`, `pyproject.toml`)
- Added OpenClaw historical sync support, including `rclm-sync --openclaw` and provider discovery for `~/.openclaw/agents/main/sessions` (`rclm/hooks/historical_sync.py`)
- Added OpenClaw install/uninstall wiring plus tests for the new plugin and hook paths (`rclm/hooks/installer.py`, `rclm/hooks/uninstaller.py`, `rclm/tests/hooks/**`)

### Changed
- Updated installer defaults and docs so OpenClaw is treated as a first-class provider alongside Claude, Gemini, and Codex (`rclm/hooks/installer.py`, `README.md`, `architecture.md`)
- Extended historical sync dispatch and CLI help to include OpenClaw session backfill (`rclm/hooks/historical_sync.py`)
- Expanded architecture docs to describe the OpenClaw plugin lifecycle and config layout (`architecture.md`)

## [v0.1.10] — 2026-04-27

### Added
- Added local hook upload redaction with default-on settings, remote substitution sync, local-only substitutions, folder exclusions, and longest-first payload replacement (`rclm/hooks/redaction.py`, `rclm/_uploader.py`)
- Added redaction settings sync during hook install and `rclm-update`, including remote substitution count output after successful sync (`rclm/hooks/installer.py`, `rclm/update.py`)
- Added shared endpoint constants for ingest and redaction settings API paths (`rclm/_endpoints.py`)
- Added tests for redaction sync, upload-time redaction, excluded-folder skips, provider hook schemas, Codex transcript parsing, and session conversion failure paths (`rclm/tests/**`)

### Changed
- Updated hook upload path to use saved config server URL, apply local redaction before POST, and redact quarantined failed-upload payloads (`rclm/_uploader.py`)
- Updated Claude, Codex, and Gemini DLP hook responses to match provider-specific hook output contracts (`rclm/hooks/claude_handler.py`, `rclm/hooks/codex_handler.py`, `rclm/hooks/gemini_handler.py`)
- Updated Codex transcript parsing to support `custom_tool_call` entries and parse tool input from either `arguments` or `input` (`rclm/hooks/codex_transcript.py`)
- Refreshed README coverage for install, hook setup, historical sync, session conversion, compression, DLP, proxy capture, and development workflows (`README.md`)

### Security
- Redacted configured sensitive values before normal hook uploads leave the machine, with local exclusions that skip upload entirely for configured folders (`rclm/hooks/redaction.py`, `rclm/_uploader.py`)

---

## [v0.1.9] — 2026-04-16

### Added
- Added `rclm convert-session <session_id> <target_tool>` subcommand: exports a captured session as a markdown context document for continuing work in a different AI tool. Supports `claude`, `gemini`, `codex`, and `generic` target formats; `-o/--output` for file output; `--no-diffs`; `--max-diff-lines N`; `--force-regenerate` to invoke LLM even when annotations are cached. Fast path (no LLM) used by default when existing annotations are available (`rclm/cli.py`, `rclm/convert.py`)
- Documented `rclm convert-session` in README with full usage examples, fast/full path explanation, and config note (`README.md`)

### Fixed
- Fixed hook binaries written as bare names (e.g. `rclm-claude-hooks`) in provider config files — installer now resolves the absolute path via `shutil.which()` so hooks fire correctly when the virtualenv is not on `PATH` at hook invocation time (`rclm/hooks/installer.py`)
- Fixed `_command_already_present()` incorrectly matching hook commands across different `matcher` values, causing duplicate-check false positives when the same binary handles multiple matchers (`rclm/hooks/installer.py`)

---

## [v0.1.8] — 2026-04-09

### Fixed
- Fixed `Unclosed client session` / `Unclosed connector` warnings after `rclm-sync` — added `close_session()` to `_uploader.py` and called it in `finally` blocks in both `_run()` and `_run_failed()` coroutines so the module-level aiohttp session is always closed before the event loop exits (`rclm/_uploader.py`, `rclm/hooks/historical_sync.py`)
- Fixed `rclm-hooks-install` browser API key callback silently failing on Chrome 98+ — added `Access-Control-Allow-Private-Network: true` to the local HTTP server's CORS preflight response, required by Chrome's Private Network Access spec for HTTPS→localhost requests (`rclm/hooks/installer.py`)

---

## [v0.1.7] — 2026-04-09

### Fixed
- Fixed model name always resolving to `"claude-unknown"` — `model` and `usage` are nested inside `message{}` in Claude Code's JSONL transcript, not at the top level; extraction now checks both locations (`rclm/hooks/transcript.py`)
- Fixed token counts always being 0 for the same reason — `usage` lookup now falls back to `msg.get("usage")` (`rclm/hooks/transcript.py`)

### Changed
- Updated Claude `PostToolUse` DLP handler to emit `hookEventName` + `additionalContext` metadata instead of rewriting the tool response directly, conforming to Claude Code's hook contract (`rclm/hooks/claude_handler.py`)
- Corrected transcript module docstring to reflect actual JSONL shape (model and usage live inside `message{}`) (`rclm/hooks/transcript.py`)
- Added a top-level `rclm` CLI entry point for version checks and future features.

---

## v0.1.6

### Added
- Added historical session backfill via `rclm-sync`, with discovery support for Claude Code, Gemini CLI, and Codex CLI transcripts.
- Added installer-driven sync onboarding so `rclm-hooks-install` can offer an immediate upload of existing sessions after hooks are installed.
- Added DLP support for `.env`-style files, including env-file detection, secret parsing, input redaction for Claude reads, shell-read blocking for Claude Bash usage, output scrubbing across supported providers, and temp-file cleanup for sanitized reads.
- Added sync-aware uploads with `HookSessionRecord.is_sync` so the server can distinguish historical imports from live captures.
- Added tests for DLP behavior and historical sync discovery/parsing flows.

### Changed
- Updated Claude, Gemini, and Codex hook handlers to support DLP-driven response rewriting without breaking provider hook contracts.
- Updated Claude post-tool DLP handling to emit hook metadata for redacted responses instead of returning a rewritten tool payload directly.
- Updated Gemini hook output handling so hook-specific JSON can be returned when a tool response is scrubbed.
- Extended uploader retry configuration so sync paths can cap retries independently of live-session uploads.
- Updated installer flags and persisted config handling to support `--dlp` alongside existing hook installation options.
- Expanded project documentation and architecture notes to cover historical sync, DLP behavior, provider coverage, and the new `rclm-sync` entry point.

### Internal
- Added the `rclm-sync` console script to package entry points.
- Renamed installer URL constants for clearer frontend/backend separation.
