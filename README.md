# RCLM — Data Capture for AI Tools

Every time you use an AI coding assistant, you produce valuable reasoning and code. **RCLM** (ReclaimLLM) ensures that data belongs to you. It is a lightweight capture layer that records your AI sessions from Claude Code, Gemini CLI, Codex CLI, and OpenClaw, shipping them to your personal ReclaimLLM account for search, analysis, and continuation.

## Key Features

- **Native Hooks:** Zero-config integration into Claude Code, Gemini CLI, Codex CLI, and OpenClaw.
- **Historical Sync:** One-command backfill for all your past AI sessions.
- **DLP & Privacy:** Automatic redaction of secrets from `.env` files before they reach the model.
- **Context Compression:** Read caching, result dedup, exec-output compaction, and automatic image downscaling cut token usage without losing information — see [Context Compression & DLP](#context-compression--dlp).
- **Context Conversion:** Export any captured session as a Markdown context document to continue work in a different tool.
- **Local Proxy:** Experimental LiteLLM-based proxy for OpenAI-compatible tools.

---

## Quick Start

### 1. Install
```bash
pip install rclm
# Or for proxy support: pip install 'rclm[proxy]'
```

### 2. Setup Hooks
```bash
# Integrates with Claude Code, Gemini CLI, Codex CLI, and OpenClaw
rclm-hooks-install
```
This will open a browser to `reclaimllm.com` to link your account. Once linked, every session is automatically captured.

### 3. Sync History
```bash
# Upload sessions that predated the installation
rclm-sync
```

---

## Session Conversion (New!)

`rclm convert-session` generates a compact context document for starting a new session in another tool. It does not restore the source tool's private runtime state.

```bash
# Export a session for Claude Code
rclm convert-session <session_id> claude -o CLAUDE.md

# Export for Gemini CLI
rclm convert-session <session_id> gemini -o .gemini

# Options
rclm convert-session <session_id> generic --no-diffs          # Omit file diffs
rclm convert-session <session_id> claude  --force-regenerate  # Use LLM for a fresh summary
```
*Supported targets:* `claude`, `gemini`, `codex`, `generic`.

---

## Agent Plugins

RCLM includes local plugin marketplace entries for Codex, Claude, and Cursor. Each plugin exposes ReclaimLLM as persistent memory for AI agents through the bundled `rclm-mcp` server.

```bash
codex plugin marketplace add /path/to/DC-hooks-proxy
codex plugin add reclaimllm@reclaimllm-plugins
```

For Claude and Cursor, add the matching marketplace file from this repo:

- `DC-hooks-proxy/.claude-plugin/marketplace.json`
- `DC-hooks-proxy/.cursor-plugin/marketplace.json`

Then authenticate the local MCP server if you have not already:

```bash
rclm-hooks-install --with-mcp
```

Start a new agent thread and confirm the `reclaimllm` plugin and MCP server are enabled.

### MCP Tools

| Tool | Description |
|---|---|
| `search_sessions` | Hybrid semantic + keyword search across captured sessions by topic, error, file, or date range. |
| `filter_sessions` | Authoritative Postgres session listing for metadata/date filters when there is no semantic text query. |
| `search_by_filename` | File/folder-scoped session history, e.g. "what changed in `auth.tsx`". |
| `get_session` | Summary metadata and a frontend link for one session ID. |
| `summarize_session` | Pull a specific session's summary in as working context, on explicit request. |
| `list_projects` | List available project filters. |
| `file_brief` | Recent-history brief for a file before making a non-trivial edit to it. |
| `handoff` | Generate a continuation document to start a fresh session without losing context. |
| `transfer_session` | Stream a complete captured session (messages, tool calls/results, file diffs) as a versioned JSON artifact. |
| `signals` | Up to 5 open workflow-efficiency signals (evidence + prescribed fix) for the current project. |
| `replay_eligibility` | Cheap, metadata-only check of whether replaying compression mechanisms is worth doing. |
| `replay_session` | Reproduce shipped compression mechanisms over one captured session and report the real tool-result token reduction. |
| `replay_corpus` | Same as `replay_session`, aggregated across a filtered window of sessions. |
| `replay_compare` | Replay the same corpus under multiple mechanism configurations in one call. |

All tools are read-only: none re-execute historical commands, call a model, or modify captured data.

When you need the complete captured session instead of a summary, ask the target agent to call `transfer_session` with the ReclaimLLM session ID. The tool streams a versioned JSON artifact containing every captured message, tool call/result, file diff, and metadata field into an owner-only temporary file. The target agent reads that file as historical context; recorded tool calls are never re-executed automatically.

`SESSION_TRANSFER_MAX_BYTES` controls the backend and local download ceiling and defaults to 100 MiB. Transfers are never silently truncated. `SESSION_TRANSFER_TTL_SECONDS` controls when local artifacts become eligible for bounded opportunistic cleanup and defaults to one hour.

### Replay: verifying token savings

`replay_eligibility`, `replay_session`, `replay_corpus`, and `replay_compare` reproduce RCLM's shipped compression mechanisms (`range_cache`, `shell_compaction`, `hash_dedupe`) over already-captured sessions and report the real tool-result token reduction, without calling a model or re-running any historical command:

- "Would compression help on my last 50 sessions?" → `replay_eligibility`
- "How much did compression save on session `<id>`?" → `replay_session`
- "What's the aggregate savings across my Codex sessions this month?" → `replay_corpus`
- "Compare shell compaction alone vs. combined with range cache" → `replay_compare`

Every result states sessions considered vs. eligible vs. excluded; a session or corpus below the turn/tool-call thresholds is refused with the specific failing constraint rather than given an unstable number.

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `rclm-hooks-install` | Install/configure native hooks for local LLM CLIs. |
| `rclm-sync` | Discover and upload historical transcripts. |
| `rclm convert-session` | Export a session to Markdown context for tool switching. |
| `rclm-proxy` | Start/setup a LiteLLM proxy for OpenAI-compatible capture. |
| `rclm-update` | Check for and apply updates to the `rclm` package. |

---

## Advanced Usage

### Context Compression & DLP
Env-file DLP is enabled on fresh installs and preserves an explicit saved opt-out:
```bash
rclm-hooks-install --compress                          # Reduces tool-result tokens in Claude Code, Codex, and Cursor
rclm-hooks-install --dlp                                # Explicitly enable or re-enable DLP
rclm-hooks-install --no-dlp                             # Explicitly disable DLP
rclm-hooks-install --image-lifecycle                    # Downscales oversized screenshots/images before they reach the model
rclm-hooks-install --image-lifecycle --image-max-dim=1280  # Set the max image dimension in pixels (default 1280)
```

Image downscaling (`--image-lifecycle`) resizes and re-encodes oversized tool-result images — full-page screenshots, MCP screenshot-tool output — before they enter the model's context, and never upscales. It applies for real on Claude Code sessions; on Codex it currently reports measured before/after savings only, since Codex CLI does not yet apply hook-driven rewrites of MCP tool output. Requires the optional `images` extra: `pip install 'rclm[images]'`.

Text compression uses each coding client's native hooks; it does not require proxy or LLM-gateway traffic. Claude Code and Codex support recognized shell-output compaction. Cursor wraps recognized shell commands before execution and limits post-result replacement to structured MCP output. Unknown commands, failures, images, and ambiguous structured results pass through unchanged. Identical-result dedupe remains off by default (`--dedupe`).

### Folder Capture Filters
Limit uploads to specific project folders during installation:
```bash
rclm-hooks-install --include-folder=/path/to/project
rclm-hooks-install --include-folder=/work/app --include-folder=/work/infra
```

Use `--exclude-folder=/path/to/private` to skip specific folders when no include allowlist is configured.

### Proxy Capture (Experimental)
Point your tools at `http://localhost:4000` to capture raw API interactions:
```bash
rclm-proxy setup
rclm-proxy start
```

---

## Technical Details

For information on data models, hook internals, and the DLP engine, see [**architecture.md**](architecture.md).

## Development

```bash
uv sync --extra dev          # Install dev dependencies
uv run pre-commit install    # Setup linting/formatting hooks
uv run pytest rclm/tests     # Run the test suite
```

License: Apache-2.0
