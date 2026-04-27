# RCLM — Data Capture for AI Tools

Every time you use an AI coding assistant, you produce valuable reasoning and code. **RCLM** (ReclaimLLM) ensures that data belongs to you. It is a lightweight capture layer that records your AI sessions from Claude Code, Gemini CLI, and Codex CLI, shipping them to your personal ReclaimLLM account for search, analysis, and continuation.

## Key Features

- **Native Hooks:** Zero-config integration into Claude Code, Gemini CLI, and Codex CLI.
- **Historical Sync:** One-command backfill for all your past AI sessions.
- **DLP & Privacy:** Automatic redaction of secrets from `.env` files before they reach the model.
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
# Integrates with Claude Code, Gemini CLI, and Codex CLI
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

`rclm convert-session` allows you to take a session captured in one tool (e.g., Claude Code) and instantly resume it in another (e.g., Gemini CLI) by generating a structured context document.

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
Enable advanced features during installation:
```bash
rclm-hooks-install --compress  # Reduces token usage for Claude Code
rclm-hooks-install --dlp       # Enables Data Loss Prevention for .env files
```

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
