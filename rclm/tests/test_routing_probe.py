from __future__ import annotations

import json

from rclm.hooks import routing


def test_gateway_vs_direct_classification(monkeypatch):
    cfg = {
        "org_hook_policy": {
            "gateway_hosts": ["gateway.example.com"],
            "gateway_required_providers": ["anthropic", "openai"],
        }
    }
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.com/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_API_BASE", raising=False)
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)

    result = routing.classify_routing(cfg=cfg)
    assert result["anthropic"] == "gateway"
    assert result["openai"] == "direct"
    blob = json.dumps(result)
    assert "gateway.example.com" not in blob
    assert "api.openai.com" not in blob
    assert "https://" not in blob


def test_unknown_without_gateway_hosts(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    result = routing.classify_routing(cfg={"org_hook_policy": {}})
    assert result.get("anthropic") == "unknown"
