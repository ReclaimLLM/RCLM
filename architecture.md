# ReclaimLLM — Architecture

## Package layout

```
rclm/
├── _config.py        # ~/.reclaimllm/config.json read/write (shared by installer + uploader)
├── _models.py        # shared dataclasses (ProxyRecord, HookSessionRecord, ToolCall, FileDiff)
├── _uploader.py      # async upload + retry logic (used by both proxy and hooks)
├── update.py         # package update entry point
├── proxy/
│   ├── start.py            # proxy CLI entry point: rclm-proxy
│   ├── litellm_callback.py # LiteLLM callback -> ProxyRecord adapter
│   └── config_template.yaml
└── hooks/
    ├── claude_handler.py   # Claude Code lifecycle event handler, entry point: rclm-claude-hooks
    ├── gemini_handler.py   # Gemini CLI lifecycle event handler, entry point: rclm-gemini-hooks
    ├── codex_handler.py    # Codex CLI lifecycle event handler, entry point: rclm-codex-hooks
    ├── codex_transcript.py # Codex transcript parser -> normalized session data
    ├── compress.py         # PreToolUse compression engine (Read/Grep/Bash token reduction)
    ├── dlp.py              # DLP engine (secret redaction from env files before model context)
    ├── installer.py        # writes hook config into provider settings files, entry point: rclm-hooks-install
    ├── session_store.py    # per-session JSONL accumulator (~/.reclaimllm/sessions/)
    └── transcript.py       # Claude Code JSONL transcript parser
```

---

## Credential flow

```
rclm-hooks-install --api-key=<key>
        │
        ├── saves to ~/.reclaimllm/config.json
        │       { "server_url": "...", "api_key": "..." }
        │
        └── writes clean hook commands into .claude/settings.json or .gemini/settings.json
                "command": "rclm-claude-hooks SessionStart"   (no inline credentials)

At upload time (_uploader.upload):
        env var BACKEND_SERVER          ──┐
                                        ──┤─ env vars take precedence
                                          │
        _config.load()["server_url"]    ──┤─ config file fallback
        _config.load()["api_key"]       ──┘
```

Env vars still work for the proxy (or for manual overrides). The config file is the primary mechanism for hooks since hook processes inherit no special environment.

---

## rclm-proxy

```
rclm-proxy start
        │
        ├── ensures ~/.reclaimllm/litellm_config.yaml exists
        ├── writes a callback shim under ~/.reclaimllm/rclm/proxy/
        └── launches LiteLLM with rclm.proxy.litellm_callback.proxy_handler_instance
                              │
                              └── LiteLLM callback builds ProxyRecord
                                      └── _uploader.upload_single()
```

**Implementation note:** the proxy path is now built on LiteLLM rather than a custom aiohttp reverse proxy.

---

## rclm-claude-hooks (Claude Code)

```
Claude Code calls: rclm-claude-hooks <EventName>   (stdin: JSON payload)
                        │
          ┌─────────────┼──────────────────────────────────┐
          │             │                                   │
    SessionStart   PreToolUse /          Stop / SubagentStop
    UserPromptSubmit  PostToolUse              │
          │             │                      ├── read session JSONL
          └─────────────┘                      ├── parse transcript file
                │                              │     (messages, tool_calls, tokens, model)
        append event to                        ├── extract FileDiffs from
        ~/.reclaimllm/sessions/{sid}.jsonl     │     Write / Edit / MultiEdit tool inputs
                                               ├── build HookSessionRecord
                                               ├── _uploader.upload_single()
                                               └── cleanup session JSONL
```

**session_store:** one JSONL file per session under `~/.reclaimllm/sessions/`. Claude Code runs hooks sequentially — no concurrent writes, no locking needed.

**transcript.py:** parses Claude Code's JSONL transcript to extract structured messages, paired tool calls (input + result via `tool_use_id`), model name, and cumulative token counts.

---

## rclm-gemini-hooks (Gemini CLI)

```
Gemini CLI calls: rclm-gemini-hooks <EventName>   (stdin: JSON payload)
                        │
          ┌─────────────┼─────────────────────────────────┐
          │             │                                  │
    SessionStart   BeforeAgent /        SessionEnd
    BeforeAgent    AfterAgent /              │
    AfterAgent     AfterTool                ├── read session JSONL
          │             │                   ├── build messages from BeforeAgent/AfterAgent events
          └─────────────┘                   ├── build tool_calls from AfterTool events
                │                           ├── extract FileDiffs from
        append event to                     │     write_file / replace tool inputs
        ~/.reclaimllm/sessions/{sid}.jsonl  ├── build HookSessionRecord
                                            ├── _uploader.upload_single()
                                            └── cleanup session JSONL
```

**Stdout requirement:** Gemini CLI expects a JSON object on stdout for every hook call. `main()` prints the return value of the handler function if it is a dict (e.g. `{"hookSpecificOutput": {...}}` from DLP), otherwise `{}`.

**No transcript parsing:** Gemini's transcript format is not documented. Messages and tool calls are assembled purely from accumulated hook events.

**Gemini tool name mapping:**

| Gemini tool | Claude equivalent | Captured as |
|---|---|---|
| `write_file` | `Write` | `FileDiff` (before=None) |
| `replace` | `Edit` | `FileDiff` (before/after strings) |

---

## DLP (Data Loss Prevention)

`hooks/dlp.py` intercepts tool calls before secret values from `.env`-style files reach
the model context. Enabled opt-in via `rclm-hooks-install --dlp`; stored as
`"dlp": true` in `~/.reclaimllm/config.json`.

### How it works

```
PreToolUse (Read tool)
    agent tries to read dev.env
          │
          ├── dlp.maybe_redact_input()
          │       ├── detect env file by name (_is_env_file)
          │       ├── parse all *.env / .env* files in CWD (_load_secrets)
          │       ├── build scrub set: filter short values, safe values, pure integers
          │       ├── write sanitised copy to /tmp/rclm_dlp_*.env  ← track path in session_store
          │       └── return {"file_path": "/tmp/rclm_dlp_*.env"}
          │
          └── Claude Code sees updatedInput → reads sanitised file instead
                  MODEL SEES: API_KEY=[REDACTED:API_KEY]
                  MODEL NEVER SEES: actual secret values

PreToolUse (Bash tool)
    agent runs: cat dev.env
          │
          ├── dlp.maybe_redact_input()
          │       ├── detect cat/less/more/head/tail targeting an env file
          │       └── return {"command": "echo '[rclm DLP] Blocked: ...'"}
          │
          └── Claude Code executes the echo instead of the cat

PostToolUse (Bash / shell tool output)
    tool output contains a secret value
          │
          ├── dlp.maybe_redact_output()
          │       ├── load secrets from CWD env files (re-parsed, always fresh)
          │       ├── replace all matching secret values with [REDACTED:VAR_NAME]
          │       └── return scrubbed string (or None if nothing changed)
          │
          └── handler prints hookSpecificOutput.updatedResponse → model sees scrubbed output

Stop
    ├── iterate session events for DLPTempFile entries
    └── os.unlink() each tracked temp path
```

### Per-provider coverage

| Provider | Read redirect (PreToolUse) | Bash block (PreToolUse) | Output scrub (PostToolUse) |
|---|---|---|---|
| Claude Code | ✓ | ✓ | ✓ |
| Gemini CLI | — (AfterTool fires post-execution) | — | ✓ (`run_shell_command`, `read_file`) |
| Codex CLI | — (Bash-only hooks) | — | ✓ |

### Env file detection

Files are matched if their **basename** satisfies any of:
- equals `.env` or `.envrc`
- starts with `.env.` — e.g. `.env.local`, `.env.production`
- ends with `.env` — e.g. `dev.env`, `prod.env`, `llm.env`

Scanning is non-recursive (CWD only). Re-parsed on every hook invocation
so changes to env files mid-session are picked up immediately.

### Env file parsing

Supports all common formats in a single pass:

| Format | Example |
|---|---|
| `KEY=VALUE` | `API_KEY=sk-ant-abc123` |
| `export KEY=VALUE` | `export DB_PASS=s3cret` |
| `KEY="quoted value"` | `URL="postgres://u:p@h/db"` |
| `KEY='single quoted'` | `TOKEN='abc def'` |
| `KEY VALUE` (space-sep) | `SECRET myvalue` |
| Inline comments | `KEY=value  # note` → value is `value` |
| Full-line comments | `# ignored entirely` |

### Scrub set filtering

Values are **excluded** from the scrub set (will not be redacted) if they:
- are shorter than 5 characters
- appear in the safe-value allowlist: `true`, `false`, `yes`, `no`, `null`, `none`,
  `localhost`, `0.0.0.0`, `127.0.0.1`, `::1` (and case variants)
- consist entirely of digits (e.g. `PORT=8080`)

Remaining values are sorted **longest-first** before substitution to prevent
a shorter secret from partially replacing a longer one that shares a prefix.

### Temp file lifecycle

| Event | Action |
|---|---|
| `PreToolUse` (Read on env file) | `tempfile.NamedTemporaryFile(delete=False, prefix="rclm_dlp_")` written; path stored as `{"event_type": "DLPTempFile", "path": "..."}` in session_store |
| `Stop` / `SubagentStop` | Iterate session events; `os.unlink()` each `DLPTempFile` path |
| Session crash (no `Stop`) | Temp files remain in `/tmp/` — pre-sanitised content only, no security risk; evicted by OS eventually |

### Config key

```json
{ "dlp": true }
```

Stored in `~/.reclaimllm/config.json` alongside `compress`, `server_url`, and `api_key`.
Persisted by `rclm-hooks-install --dlp`; can also be set manually via `rclm._config.patch(dlp=True)`.

---

## rclm-hooks-install

```
rclm-hooks-install [--claude] [--gemini] [--codex] [--local] [--api-key=<key>] [--server-url=<url>] [--compress] [--dlp]
        │
        ├── resolve credentials: --api-key flag → saved config → prompt with SETUP_URL + exit 1
        ├── _config.save(server_url, api_key)     # persist for uploader + future installs
        │
        ├── [--claude]  → target .claude/settings.json, inject rclm-claude-hooks commands
        ├── [--gemini]  → target .gemini/settings.json, inject rclm-gemini-hooks commands
        └── [--codex]   → target .codex/hooks.json,     inject rclm-codex-hooks commands
                │
                └── _merge_hooks(): deep-merge, skip duplicate commands (idempotent)
```

---

## Data models

### ProxyRecord
| Field | Type | Notes |
|---|---|---|
| `session_id` | str | uuid4 |
| `timestamp` | str | ISO-8601, request start |
| `method` / `url` | str | |
| `request_headers` | dict | Host stripped |
| `request_body` | dict \| str | parsed JSON or raw |
| `response_status` | int | |
| `response_headers` | dict | Transfer-Encoding stripped |
| `response_body` | dict \| list | list of events for SSE |
| `is_streaming` | bool | |
| `duration_ms` | float | |
| `model` | str \| None | from request body |

### HookSessionRecord
| Field | Type | Notes |
|---|---|---|
| `session_id` | str | uuid4 |
| `cwd` | str | working directory at session start |
| `started_at` / `ended_at` | str | ISO-8601 |
| `duration_s` | float | |
| `transcript_path` | str \| None | path to agent transcript file |
| `model` | str \| None | from transcript (Claude) or None (Gemini) |
| `messages` | list[dict] | `{role, content, timestamp}` |
| `tool_calls` | list[ToolCall] | |
| `file_diffs` | list[FileDiff] | |
| `total_input_tokens` | int \| None | from transcript (Claude) or None (Gemini) |
| `total_output_tokens` | int \| None | from transcript (Claude) or None (Gemini) |

### ToolCall
| Field | Type | Notes |
|---|---|---|
| `tool_use_id` | str | from transcript (Claude) or `gemini-tool-{i}` (Gemini) |
| `tool_name` | str | e.g. `Write`, `Bash`, `write_file` |
| `tool_input` | dict | |
| `tool_result` | str \| dict \| list \| None | |
| `timestamp` | str | ISO-8601 |

### FileDiff
| Field | Type | Notes |
|---|---|---|
| `path` | str | |
| `before` | str \| None | None for new files |
| `after` | str \| None | None for deleted files |
| `unified_diff` | str | output of `difflib.unified_diff` |

---

## Error handling

| Location | Error | Handling |
|---|---|---|
| Proxy upstream | Connection error | 502 to client |
| Proxy upstream | Non-2xx | Forwarded as-is, still recorded |
| SSE reassembler | Malformed JSON | Silently discarded |
| Uploader | Network failure | 3 retries (0.5s, 1s, 2s), then stderr |
| Uploader | No server URL (env + config both unset) | Immediate stderr dump |
| Hook handler | Any exception | Swallowed; process always exits 0 (must not disrupt Claude Code) |
| Gemini hook handler | Any exception | Swallowed; process always exits 0, always prints `{}` to stdout |
| Installer | No API key | Prints setup URL to stderr, exits 1 |
| Installer | Invalid JSON in existing settings | Warning to stderr, overwrites with empty `{}` |
| Config file | Malformed JSON or missing | `_config.load()` returns `{}` silently |
| Transcript | Missing or unreadable file | Returns empty `TranscriptData` |
| Transcript | Malformed JSON lines | Skipped silently |
