"""Unit tests for litellm_callback: StandardLoggingPayload → ProxyRecord."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from rclm._models import ProxyRecord
from rclm.proxy.litellm_callback import (
    ReclaimLLMLogger,
    _build_record,
    _infer_provider,
)

# ---------------------------------------------------------------------------
# _infer_provider
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, expected",
    [
        ("anthropic/claude-sonnet-4-5", "anthropic"),
        ("openai/gpt-4o", "openai"),
        ("gemini/gemini-1.5-pro", "gemini"),
        ("gpt-4o", None),  # bare name, no prefix
        (None, None),
    ],
)
def test_provider_inference(model, expected):
    assert _infer_provider(model) == expected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2026, 3, 14, 12, 0, 1, tzinfo=timezone.utc)  # 1 000 ms later


def _make_kwargs(model="anthropic/claude-opus-4-5", stream=False, exception=None):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
        "standard_logging_object": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "response_cost": 0.0025,
        },
        "exception": exception,
    }


# ---------------------------------------------------------------------------
# _build_record — success path
# ---------------------------------------------------------------------------


def test_build_record_success():
    record = _build_record(_make_kwargs(), response_obj=None, start_time=_T0, end_time=_T1)

    assert isinstance(record, ProxyRecord)
    assert record.model == "anthropic/claude-opus-4-5"
    assert record.provider == "anthropic"
    assert record.response_cost == pytest.approx(0.0025)
    assert record.duration_ms == pytest.approx(1000.0)
    assert record.is_streaming is False
    assert isinstance(record.request_body, dict)
    assert record.request_body["messages"] == [{"role": "user", "content": "hello"}]


def test_build_record_streaming_flag():
    record = _build_record(
        _make_kwargs(stream=True),
        response_obj=None,
        start_time=_T0,
        end_time=_T1,
    )
    assert record.is_streaming is True


# ---------------------------------------------------------------------------
# _build_record — failure path
# ---------------------------------------------------------------------------


def test_build_record_failure():
    record = _build_record(
        _make_kwargs(exception=ValueError("rate limit")),
        response_obj=None,
        start_time=_T0,
        end_time=_T1,
        error="rate limit",
    )

    assert record.response_body == {"error": "rate limit"}
    assert record.response_cost == pytest.approx(0.0)
    assert record.model == "anthropic/claude-opus-4-5"


# ---------------------------------------------------------------------------
# ReclaimLLMLogger — upload_single is called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_called_on_success():
    handler = ReclaimLLMLogger()
    with patch(
        "rclm.proxy.litellm_callback.upload_single",
        new_callable=AsyncMock,
    ) as mock_upload:
        await handler.async_log_success_event(_make_kwargs(), None, _T0, _T1)

    mock_upload.assert_awaited_once()
    assert isinstance(mock_upload.call_args[0][0], ProxyRecord)


@pytest.mark.asyncio
async def test_upload_called_on_failure():
    handler = ReclaimLLMLogger()
    kwargs = _make_kwargs(exception=ConnectionError("upstream timeout"))

    with patch(
        "rclm.proxy.litellm_callback.upload_single",
        new_callable=AsyncMock,
    ) as mock_upload:
        await handler.async_log_failure_event(kwargs, None, _T0, _T1)

    mock_upload.assert_awaited_once()
    record: ProxyRecord = mock_upload.call_args[0][0]
    assert record.response_body == {"error": "upstream timeout"}
