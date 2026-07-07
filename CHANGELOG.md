# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
