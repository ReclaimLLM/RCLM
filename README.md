# rclm — data capture

Every time you use an AI coding assistant, you produce something valuable: real problems solved, real reasoning applied, real code written and debugged. That data belongs to you — not to the provider running the API.

**ReclaimLLM** is building the infrastructure for people to own, search, and optionally monetise their AI interaction history. `rclm` is the capture layer: a lightweight Python package that sits between you and your LLM tools and silently records every session, then ships it to your personal ReclaimLLM account.

**Python package** (`pip install rclm`) that intercepts and records LLM sessions from two capture modes, then uploads them to the ReclaimLLM server.

> **Repo:** `DC-hooks-proxy/` — one of three repos in the ReclaimLLM monorepo.
> The other two are `DC-browser-extension/` (Chrome extension) and `ReclaimLLM-server/` (backend).

## Capture modes

- **Hooks** — native integrations into Claude Code, Gemini CLI, and Codex CLI. Captures structured session data: messages, tool calls, file diffs, token counts. Upload happens at session end.
- **Proxy** — (experimental) a local LiteLLM proxy that sits in front of provider APIs. Captures raw request/response payloads for any tool that speaks the OpenAI-compatible API. Upload happens per request.

Records are POSTed as JSON to the configured ReclaimLLM server (saved in `~/.reclaimllm/config.json`, or overridden by `BACKEND_SERVER`). If the server is unreachable, failed records are quarantined to `~/.reclaimllm/failed_uploads/` with owner-only permissions.

See [`architecture.md`](architecture.md) for data models and flow diagrams.

---

## Installation

```bash
# Hooks only (Claude Code, Gemini CLI, Codex CLI)
pip install rclm

# Hooks + proxy
pip install 'rclm[proxy]'
```

### Hooks setup

```bash
# Install for all providers (Claude Code + Gemini CLI + Codex CLI), global
rclm-hooks-install

# Install for a single provider
rclm-hooks-install --claude
rclm-hooks-install --gemini
rclm-hooks-install --codex

# Install into the current project directory instead of home
rclm-hooks-install --local

# Skip the browser flow and supply the key directly
rclm-hooks-install --api-key=<key>

# Enable context compression for Claude Code (reduces token usage on Bash/Read/Grep)
rclm-hooks-install --compress

# Remove hooks from all providers
rclm-hooks-uninstall
```

`rclm-hooks-install` opens a browser to `reclaimllm.com/settings` so you can create an API key and have it sent back automatically. The key and server URL are saved to `~/.reclaimllm/config.json` and reused on subsequent installs.

### Proxy setup

```bash
# Interactive setup — writes ~/.reclaimllm/litellm_config.yaml
rclm-proxy setup

# Start the LiteLLM proxy (default port 4000)
rclm-proxy start

# Pass extra LiteLLM flags
rclm-proxy start --port 8080
```

Point your tools at `http://localhost:4000` and set provider keys as environment variables before starting:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
rclm-proxy start
```

---

## How hooks work

Hooks integrate directly into each CLI's lifecycle via its settings file. `rclm-hooks-install` merges hook commands into the provider's config idempotently (skips duplicates, backs up invalid JSON). Each CLI calls the relevant `rclm-*-hooks` binary for every event, passing a JSON payload on stdin. All handlers exit 0 and swallow exceptions — hook failures never disrupt the underlying CLI.

Events are accumulated per-session in `~/.reclaimllm/sessions/{session_id}.jsonl`. At session end, the spool file is read, the transcript is parsed, a `HookSessionRecord` is assembled, uploaded, and the spool file is deleted.

### Claude Code

**Binary:** `rclm-claude-hooks`
**Config:** `~/.claude/settings.json` (or `.claude/settings.json` for `--local`)

| Event | What's captured |
|-------|----------------|
| `SessionStart` | `cwd`, `model`, `timestamp` |
| `UserPromptSubmit` | user prompt text |
| `PreToolUse` | tool name, tool input; runs compression if enabled |
| `PostToolUse` | tool name, tool input, tool response |
| `Stop` / `SubagentStop` | triggers assembly and upload |

On `Stop`, the Claude Code transcript JSONL at `transcript_path` is parsed to extract:
- Full conversation messages (role, content, timestamp)
- Tool calls with input/output and per-tool token estimates
- File diffs from `Write`, `Edit`, and `MultiEdit` tool invocations (unified diff format)
- Token usage (`input_tokens`, `output_tokens` from usage blocks)
- Session analytics: total tool call count, unique files modified, dominant tool

Sessions shorter than 5 seconds are recorded with `duration_s=0` and no timestamps (treats them as noise).

### Gemini CLI

**Binary:** `rclm-gemini-hooks`
**Config:** `~/.gemini/settings.json`

Every hook invocation must print a JSON object to stdout (Gemini requirement) — the handler always prints `{}`.

| Event | What's captured |
|-------|----------------|
| `SessionStart` | `cwd`, `timestamp` |
| `BeforeAgent` | user prompt (one per agentic turn) |
| `AfterAgent` | assistant response (one per agentic turn) |
| `AfterTool` | tool name, input, normalised response |
| `SessionEnd` | triggers assembly and upload |

On `SessionEnd`, messages are reconstructed from `BeforeAgent`/`AfterAgent` pairs. File diffs are extracted from `write_file` and `replace` tool events. Token counts and model name are read from Gemini's session JSON at `transcript_path` (each assistant turn carries a `tokens` block).

### Codex CLI

**Binary:** `rclm-codex-hooks`
**Config:** `~/.codex/hooks.json`

Only `Bash` tool events are captured (Codex's primary tool). File diffs are extracted from `apply_patch` tool calls in the transcript JSONL. Hook-event reconstruction (from accumulated spool events) is used as a fallback if the transcript is missing or unreadable.

| Event | What's captured |
|-------|----------------|
| `SessionStart` | `cwd`, `model`, `timestamp` |
| `UserPromptSubmit` | user prompt text, `turn_id` |
| `PreToolUse` | Bash command input, `turn_id` |
| `PostToolUse` | Bash tool response, `turn_id` |
| `Stop` | triggers assembly and upload |

`PreToolUse` / `PostToolUse` pairs are matched by `turn_id` to build `ToolCall` records. Unmatched `PreToolUse` events (session killed mid-tool) are recorded with `tool_result=None`.

---

## How the proxy works

The proxy runs LiteLLM in front of any provider API. A `ReclaimLLMLogger` (LiteLLM `CustomLogger`) hooks into `async_log_success_event` and `async_log_failure_event`. Each API call — success or failure — produces one `ProxyRecord` and is uploaded immediately.

Each `ProxyRecord` contains:
- `model` and `provider` (inferred from the LiteLLM model prefix, e.g. `anthropic/claude-sonnet-4-5` → `anthropic`)
- `request_body` — full messages array sent to the provider plus optional params
- `response_body` — full parsed response or `{"error": "..."}` on failure
- `messages` — synthesised `[{role, content, timestamp}]` list (request history + new assistant turn)
- `is_streaming`, `duration_ms`, `response_cost`, `total_input_tokens`, `total_output_tokens`

Image content blocks in messages are normalised to `"[image]"`. Tool use / tool result / thinking blocks are stripped from the synthesised message list (raw `request_body`/`response_body` preserve everything).

Each proxy call gets a fresh `session_id` (uuid4) — there is no session grouping at the proxy layer.

---

## Update module

The updater (`rclm/hooks/updater.py`) checks PyPI for a newer version of `rclm` and is called non-blockingly from `rclm-hooks-install` and `rclm-update`. All network errors are swallowed — a failed check is always silent and never crashes the caller.

**Check behaviour:**
- Fetches `https://pypi.org/pypi/rclm/json` with a 2-second timeout
- Caches the result (`last_update_check`, `latest_version`) in `~/.reclaimllm/config.json`
- Skips the network call if the cache is less than 24 hours old
- Returns the latest version string only if it is strictly newer than installed; otherwise `None`

**Update command:**

```bash
rclm-update
```

Runs `pip install --upgrade rclm` against the active Python interpreter (`sys.executable`), so it always targets the correct virtualenv or system Python.

---

## Package layout

```
rclm/
├── _config.py              # reads/writes ~/.reclaimllm/config.json
├── _models.py              # shared dataclasses: ProxyRecord, HookSessionRecord, ToolCall, FileDiff
├── _uploader.py            # async upload with 3-retry exp backoff; quarantine on failure
├── update.py               # rclm-update entry point
├── proxy/
│   ├── start.py            # rclm-proxy CLI (setup / start subcommands)
│   ├── litellm_callback.py # ReclaimLLMLogger: LiteLLM CustomLogger → ProxyRecord
│   └── config_template.yaml
└── hooks/
    ├── claude_handler.py   # Claude Code hook handler (rclm-claude-hooks)
    ├── gemini_handler.py   # Gemini CLI hook handler (rclm-gemini-hooks)
    ├── codex_handler.py    # Codex CLI hook handler (rclm-codex-hooks)
    ├── transcript.py       # Claude Code JSONL transcript parser
    ├── codex_transcript.py # Codex JSONL transcript + apply_patch diff extractor
    ├── session_store.py    # per-session JSONL spool (~/.reclaimllm/sessions/)
    ├── _analytics.py       # token estimation, tool stats, compression savings
    ├── compress.py         # PreToolUse compression gate (reads config.compress flag)
    ├── installer.py        # rclm-hooks-install
    ├── uninstaller.py      # rclm-hooks-uninstall
    └── updater.py          # PyPI version check + apply_update()
```

```
rclm/tests/
├── proxy/    # LiteLLM callback tests
├── hooks/    # handler, installer, session store, transcript, analytics tests
└── compress/ # compression filter tests
```

---

## Testing ingestion

Replace `YOUR_API_KEY` and `http://localhost:8000` with your actual key and server URL.

### Hook session record

```bash
echo '{"session_id":"00000000-0000-0000-0000-000000000001","source":"claude-code","model":"claude-sonnet-4-6","started_at":"2024-01-01T00:00:00Z","ended_at":"2024-01-01T00:01:00Z","duration_s":60,"messages":[{"role":"user","content":"hello","timestamp":"2024-01-01T00:00:00Z"},{"role":"assistant","content":"hi there","timestamp":"2024-01-01T00:00:01Z"}],"tool_calls":[],"file_diffs":[]}' \
  | curl -s -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -H "X-API-Key: --api-key=LnzdYCz27mEcMKQdHR12rFq66vRODVok9HTfRfjXpQU" -d @- | jq .
```

### Proxy record (OpenAI-compatible)

```bash
echo '{"record_type":"proxy","model":"gpt-4o","method":"POST","url":"https://api.openai.com/v1/chat/completions","request_body":{"messages":[{"role":"user","content":"what is 2+2?"}]},"response_body":{"choices":[{"message":{"role":"assistant","content":"4"}}]},"response_status":200,"is_streaming":false,"duration_ms":312}' \
  | curl -s -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -H "X-API-Key: YOUR_API_KEY" -d @- | jq .
```

### Browser-chat record

```bash
echo '{"source":"browser-chatgpt","model":"gpt-4o","messages":[{"role":"user","content":"explain closures","timestamp":"2024-01-01T00:00:00Z"},{"role":"assistant","content":"a closure captures its enclosing scope","timestamp":"2024-01-01T00:00:02Z"}]}' \
  | curl -s -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -H "X-API-Key: YOUR_API_KEY" -d @- | jq .
```

### Pipe the hook binary directly (Claude Code `Stop` event)

```bash
echo '{"hook_event_name":"Stop","session_id":"00000000-0000-0000-0000-000000000002","transcript_path":"/dev/null"}' \
  | rclm-claude-hooks
```

### Verify the session landed

```bash
curl -s http://localhost:8000/sessions/00000000-0000-0000-0000-000000000001 \
  -H "X-API-Key: YOUR_API_KEY" | jq '{session_id, record_type, model, session_summary}'
```

---

## Tests

```bash
pytest rclm/tests -v
```
