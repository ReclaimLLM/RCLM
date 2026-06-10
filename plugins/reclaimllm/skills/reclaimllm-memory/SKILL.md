---
name: reclaimllm-memory
description: >
  ReclaimLLM app for adding persistent memory to AI applications and retrieving
  arbitrary prior memories through MCP. TRIGGER when: user mentions
  "ReclaimLLM", "rclm", "persistent memory", "memory layer", "agent memory",
  "search my memory", "prior sessions", "captured sessions", "session recall",
  "reuse context", "find previous work", or needs arbitrary memory search for
  Codex, Claude Code, Gemini CLI, Cursor, OpenClaw, browser chat, or proxy
  traffic. Covers the local rclm MCP server, semantic session search, file-based
  memory search, project filtering, session metadata lookup, and context export.
  DO NOT TRIGGER for generic web search or current-code questions unless prior
  session memory is likely useful.
license: Apache-2.0
metadata:
  author: ReclaimLLM
  version: "0.1.0"
  category: ai-memory
  tags: "memory, persistent-memory, mcp, codex, session-search, context-export"
compatibility: Requires Python 3.10+, pip install rclm, authenticated ReclaimLLM credentials in ~/.reclaimllm/config.json or RECLAIMLLM_SERVER_URL and RECLAIMLLM_API_KEY environment variables, and an MCP-capable host.
---

# ReclaimLLM Memory

ReclaimLLM is an app that adds persistent memory to AI applications. It captures AI sessions from coding agents, browser chat, and proxy traffic, then exposes those captured sessions as searchable memory through the local `rclm-mcp` server. Use it when an agent needs arbitrary memory search, prior implementation context, file history, reusable session context, or cross-agent continuity.

## Core Pattern

Every ReclaimLLM memory workflow follows the same pattern: retrieve, reason, and optionally expand.

1. Retrieve relevant memories with `search_sessions`, `search_by_filename`, or `list_projects`.
2. Reason over the result titles, timestamps, projects, models, and highlights together with the current repo or user prompt.
3. Expand a chosen memory with `get_session` or `summarize_session` only when the user asks to inspect or reuse a specific session.

## Tool Routing

- Use `search_sessions` for arbitrary memory search by topic, intent, feature, bug, architecture decision, performance issue, prior implementation, or user preference. If the prompt also includes a file or folder path, pass that path as `file_path`.
- Use `search_by_filename` for file/folder-only memory requests such as "what changed in auth.tsx" or "show history under /api/auth".
- Use `list_projects` when the user asks which project memories exist, wants to choose a project filter, or the same query may span unrelated projects.
- Use `get_session` only when the user asks to inspect a specific ReclaimLLM session ID. This returns metadata, a short summary, and a frontend link.
- Use `summarize_session` only after an explicit instruction such as "summarize this session", "use this session", "add this session as context", or "export context for this session".

## Search Guidance

- Start broad enough to catch useful memories, then use project or file filters when the prompt gives them.
- Prefer `record_type="session"` for coding-agent memory. Use `record_type="browser-chat"` for browser conversation memory and `record_type="proxy"` for captured API/proxy traffic when the user asks for those sources.
- Keep `limit` small by default. Increase it only when the user asks for a broader recall pass.
- Run at most one broad semantic search round before asking for a better clue. Do not repeatedly mutate search terms without user input.
- Treat ReclaimLLM memory as prior context that may be stale. Verify current behavior in the live repo before making code changes or strong claims.

## Setup

Install and authenticate ReclaimLLM before using the bundled MCP server:

```bash
pip install rclm
rclm-hooks-install --with-mcp
```

The MCP server reads credentials from `~/.reclaimllm/config.json`, falling back to `RECLAIMLLM_SERVER_URL` and `RECLAIMLLM_API_KEY`.

## Available MCP Tools

| Tool | Use |
|------|-----|
| `search_sessions` | Arbitrary semantic memory search across captured sessions. |
| `search_by_filename` | File or folder history search. |
| `list_projects` | Discover available project filters. |
| `get_session` | Retrieve metadata and a link for a known session ID. |
| `summarize_session` | Export reusable markdown context for a known session ID. |

## Guardrails

- Do not call `summarize_session` immediately after search results unless the user already gave a specific session ID or asked to use the top match.
- Do not use ReclaimLLM as a substitute for reading current local files when the task depends on current code.
- Mention when an answer depends on ReclaimLLM memory and note that prior sessions can be outdated.
- Do not expose raw API keys, local credential paths beyond setup guidance, or unrelated private session details in user-facing output.
