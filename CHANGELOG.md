# Changelog

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
