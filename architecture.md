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

**Stdout requirement:** Gemini CLI expects a JSON object on stdout for every hook call. The handler always prints `{}` before exiting regardless of event or outcome.

**No transcript parsing:** Gemini's transcript format is not documented. Messages and tool calls are assembled purely from accumulated hook events.

**Gemini tool name mapping:**

| Gemini tool | Claude equivalent | Captured as |
|---|---|---|
| `write_file` | `Write` | `FileDiff` (before=None) |
| `replace` | `Edit` | `FileDiff` (before/after strings) |

---

## rclm-hooks-install

```
rclm-hooks-install [--claude] [--gemini] [--codex] [--local] [--api-key=<key>] [--server-url=<url>]
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
