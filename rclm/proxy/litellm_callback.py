"""LiteLLM CustomLogger that maps StandardLoggingPayload → ProxyRecord and uploads it."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

from litellm.integrations.custom_logger import CustomLogger

from rclm._models import ProxyRecord
from rclm._uploader import upload_single

logger = logging.getLogger(__name__)


def _infer_provider(model: str | None) -> str | None:
    """Extract provider from a fully-qualified LiteLLM model string.

    "anthropic/claude-sonnet-4-5" → "anthropic"
    "gpt-4o"                      → None  (no prefix)
    """
    if not model or "/" not in model:
        return None
    return model.split("/")[0]


def _extract_text_content(content: object) -> str:
    """Normalise message content to a plain string.

    Handles:
    - str: returned as-is
    - list of content blocks: text blocks joined, image blocks → "[image]",
      other block types (tool_use, tool_result, thinking) skipped
    - anything else: str() fallback
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "text":
                parts.append(block.get("text") or "")
            elif t == "image":
                parts.append("[image]")
            # tool_use / tool_result / thinking blocks are skipped
        return "\n".join(p for p in parts if p)
    return str(content)


def _synthesise_messages(
    request_body: dict | str,
    response_body: dict | list | str | None,
    timestamp: str,
) -> list[dict]:
    """Build a clean [{role, content, timestamp}] list from a LiteLLM request/response pair.

    - Extracts each turn from request_body["messages"] (history sent to the model)
    - Appends the new assistant turn from response_body["choices"][0]["message"]["content"]
    """
    messages: list[dict] = []

    # ── conversation history from the request ────────────────────────────────
    req_messages: list = []
    if isinstance(request_body, dict):
        req_messages = request_body.get("messages") or []
    for msg in req_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "user"
        content = _extract_text_content(msg.get("content"))
        messages.append({"role": role, "content": content, "timestamp": timestamp})

    # ── new assistant turn from the response ─────────────────────────────────
    if isinstance(response_body, dict):
        if response_body.get("error"):
            messages.append({
                "role": "assistant",
                "content": f"[error] {response_body['error']}",
                "timestamp": timestamp,
            })
        else:
            choices = response_body.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg_obj = choices[0].get("message") or {}
                content = _extract_text_content(msg_obj.get("content"))
                if content:
                    messages.append({"role": "assistant", "content": content, "timestamp": timestamp})

    return messages


def _build_record(
    kwargs: dict,
    response_obj: object,
    start_time: datetime,
    end_time: datetime,
    error: str | None = None,
) -> ProxyRecord:
    slo: dict = kwargs.get("standard_logging_object") or {}
    model: str | None = kwargs.get("model")

    request_body: dict = {"messages": kwargs.get("messages", []), "model": model}
    extra = kwargs.get("optional_params")
    if extra:
        request_body.update(extra)

    if error is not None:
        response_body: dict | list | str | None = {"error": error}
        cost: float | None = 0.0
    else:
        if hasattr(response_obj, "model_dump"):
            response_body = response_obj.model_dump()  # type: ignore[union-attr]
        else:
            response_body = response_obj  # type: ignore[assignment]
        cost = slo.get("response_cost")

    duration_ms = (end_time - start_time).total_seconds() * 1000
    timestamp = start_time.isoformat()

    return ProxyRecord(
        session_id=str(uuid.uuid4()),
        timestamp=timestamp,
        request_body=request_body,
        response_body=response_body,
        is_streaming=bool(kwargs.get("stream", False)),
        duration_ms=duration_ms,
        model=model,
        messages=_synthesise_messages(request_body, response_body, timestamp),
        provider=_infer_provider(model),
        response_cost=cost,
        total_input_tokens=slo.get("prompt_tokens"),
        total_output_tokens=slo.get("completion_tokens"),
    )


class ReclaimLLMLogger(CustomLogger):
    async def async_log_success_event(
        self, kwargs: dict, response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        try:
            record = _build_record(kwargs, response_obj, start_time, end_time)
            await upload_single(record)
        except Exception:
            logger.exception("ReclaimLLM: failed to log success event")

    async def async_log_failure_event(
        self, kwargs: dict, response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        try:
            err = str(kwargs.get("exception", "unknown error"))
            record = _build_record(kwargs, response_obj, start_time, end_time, error=err)
            await upload_single(record)
        except Exception:
            logger.exception("ReclaimLLM: failed to log failure event")


# LiteLLM config.yaml references this instance by name:
# litellm_settings:
#   callbacks: rclm.proxy.litellm_callback.proxy_handler_instance
proxy_handler_instance = ReclaimLLMLogger()
