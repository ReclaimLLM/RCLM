# Changelog

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
