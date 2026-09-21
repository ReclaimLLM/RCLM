"""Canonical metadata shared by native coding-agent capture adapters."""

from __future__ import annotations

ADAPTER_VERSION = "1"


def model_provider(model: str | None) -> str | None:
    normalized = (model or "").lower()
    if "claude" in normalized:
        return "anthropic"
    if "gemini" in normalized:
        return "google"
    if any(token in normalized for token in ("gpt", "o1", "o3", "o4", "codex")):
        return "openai"
    return None


def native_metadata(
    agent_client: str,
    *,
    adapter_name: str,
    model: str | None = None,
    provider: str | None = None,
    agent_client_version: str | None = None,
    capabilities: dict | None = None,
    warnings: list[str] | None = None,
    extra_fields: dict | None = None,
) -> dict:
    """Return canonical HookSessionRecord keyword arguments for one adapter."""
    return {
        "capture_schema_version": 1,
        "capture_source": "native_agent",
        "agent_client": agent_client,
        "agent_client_version": agent_client_version,
        "adapter_name": adapter_name,
        "adapter_version": ADAPTER_VERSION,
        "model_provider": provider or model_provider(model),
        "capture_capabilities": capabilities or {},
        "capture_warnings": (warnings or [])[:32],
        "extra_fields": extra_fields or {},
    }
