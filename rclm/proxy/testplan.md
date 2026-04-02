# Proxy Test Plan — Manual / End-to-End

Focus: every LLM call gets captured as a complete, correct `ProxyRecord` and reaches the server.

---

## Prerequisites

```bash
# Terminal 1: proxy running
rclm-proxy start

# Terminal 2: mock server to inspect what arrives (or use the real ReclaimLLM server)
python3 -m http.server 8888
```

Set env so uploads go somewhere inspectable:

```bash
export BACKEND_SERVER=http://localhost:8888
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export LITELLM_MASTER_KEY=sk-reclaim-local   # used as the Bearer token for all proxy calls
```

---

## T1 — Non-streaming request is fully captured

**Goal:** A single chat completion produces one `ProxyRecord` with correct fields.

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "anthropic/claude-3-haiku-20240307",
    "messages": [{"role": "user", "content": "Say hello in one word"}]
  }'
```

**Verify on the server (`/api/ingest` payload):**
- `model` = `"anthropic/claude-haiku-4-5"`
- `provider` = `"anthropic"`
- `is_streaming` = `false`
- `request_body.messages` contains the user message
- `response_body` is the full response object (not null)
- `prompt_tokens` and `completion_tokens` are non-zero integers
- `response_cost` is a positive float
- `duration_ms` is a positive number
- `session_id` is a valid UUID
- `timestamp` is a valid ISO-8601 string

---

## T2 — Streaming request captures the full response

**Goal:** Streaming doesn't cause a partial or missing capture.

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "anthropic/claude-haiku-4-5",
    "stream": true,
    "messages": [{"role": "user", "content": "Count to 5"}]
  }'
```

**Verify:**
- `is_streaming` = `true`
- `response_body` is present (not null, not empty) — LiteLLM reassembles the stream before calling the callback
- `prompt_tokens` and `completion_tokens` populated (LiteLLM computes these post-stream)
- The curl output streams chunks to the terminal normally — proxy adds no visible latency

---

## T3 — Each request gets its own `session_id`

**Goal:** Two back-to-back calls produce two distinct records, not one merged session.

Run T1 twice. Inspect the two ingest payloads on the server — `session_id` values must be different UUIDs.

---

## T4 — OpenAI provider is captured with correct provider label

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

**Verify:**
- `provider` = `"openai"`
- `model` = `"openai/gpt-4o-mini"`

---

## T5 — Failed LLM call (bad API key) still produces a record

**Goal:** Errors are logged, not silently dropped.

```bash
ANTHROPIC_API_KEY=invalid curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "anthropic/claude-haiku-4-5",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**Verify on server:**
- A record was still uploaded
- `response_body` = `{"error": "...auth..."}`
- `response_cost` = `0.0`
- `prompt_tokens` / `completion_tokens` may be null — acceptable

---

## T6 — Upload reaches the server with the API key header

**Goal:** Auth header is forwarded correctly.

```bash
export RECLAIMLLM_API_KEY=test-key-123
```

Run T1 again. Inspect the raw HTTP request on the mock server:
- `X-API-Key: test-key-123` header is present on the `POST /api/ingest`

---

## T7 — No server configured: record falls back to stderr, proxy still works

**Goal:** A misconfigured upload must not crash the proxy or block the response.

```bash
unset BACKEND_SERVER
unset RECLAIMLLM_API_KEY
# Remove server_url from ~/.reclaimllm/config.json temporarily
```

Run T1. **Verify:**
- Curl still gets a valid LLM response (proxy is not broken)
- The proxy terminal output contains a JSON blob (the `ProxyRecord` dumped to stderr)
- No exception or crash in the proxy logs

Restore config afterward.

---

## T8 — System prompt is captured in `request_body`

**Goal:** System prompts (the most valuable part of a session for ReclaimLLM) are logged.

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "anthropic/claude-haiku-4-5",
    "messages": [
      {"role": "system", "content": "You are a pirate. Respond only in pirate speak."},
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

**Verify:**
- `request_body.messages` contains both the system and user messages
- The system message content is not stripped or truncated

---

## T9 — Multi-turn conversation is captured as a single record

**Goal:** All messages in a multi-turn request body are stored together.

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "anthropic/claude-haiku-4-5",
    "messages": [
      {"role": "user", "content": "My name is Alice"},
      {"role": "assistant", "content": "Hello Alice!"},
      {"role": "user", "content": "What is my name?"}
    ]
  }'
```

**Verify:**
- `request_body.messages` contains all 3 messages in order
- `prompt_tokens` reflects the full context (should be larger than T1)

---

## T10 — Shim file is regenerated on restart

**Goal:** `_ensure_callback_shim()` is idempotent.

```bash
rm ~/.reclaimllm/rclm/proxy/litellm_callback.py
rclm-proxy start
```

**Verify:** Proxy starts without error and the shim file is recreated at `~/.reclaimllm/rclm/proxy/litellm_callback.py`.

---

## T11 — Gemini non-streaming request is captured

**Prerequisite:** `export GEMINI_API_KEY=...`

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "gemini/gemini-1.5-flash",
    "messages": [{"role": "user", "content": "Say hello in one word"}]
  }'
```

**Verify:**
- `provider` = `"gemini"`
- `model` = `"gemini/gemini-2.5-flash"`
- `response_body` is the full response (not null)
- `prompt_tokens` and `completion_tokens` are non-zero integers
- `response_cost` is a positive float

---

## T12 — Gemini streaming request captures the full response

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "gemini/gemini-2.5-flash",
    "stream": true,
    "messages": [{"role": "user", "content": "Count to 5"}]
  }'
```

**Verify:**
- `is_streaming` = `true`
- `response_body` is present and non-empty — Gemini SSE stream is reassembled by LiteLLM before the callback fires
- `prompt_tokens` and `completion_tokens` populated post-stream
- Chunks stream to terminal normally

---

## T13 — Gemini system prompt is captured

**Goal:** Gemini routes system messages differently (as `system_instruction`) — verify ReclaimLLM captures the original messages array before LiteLLM transforms it.

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "gemini/gemini-2.5-flash",
    "messages": [
      {"role": "system", "content": "You are a pirate. Respond only in pirate speak."},
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

**Verify:**
- `request_body.messages` contains both the system and user messages as sent — not Gemini's transformed format
- System message content is not stripped

---

## Known gaps (not covered here)

| Gap | Why it matters |
|-----|----------------|
| Upload retry on 5xx | Need a mock server returning 500 then 200 to confirm backoff logic |
| Large context (>100K tokens) | `request_body` serialisation size / upload timeout behaviour |
| Concurrent requests | Two simultaneous calls — both records arrive, no race on `session_id` |
| Token counts on streaming | LiteLLM may not always populate these post-stream depending on provider |
