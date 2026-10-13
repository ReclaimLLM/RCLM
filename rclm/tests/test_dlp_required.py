from __future__ import annotations

from datetime import datetime, timedelta, timezone

from rclm import _config


def test_dlp_required_overrides_local_false():
    cfg = {
        "dlp": False,
        "org_hook_policy": {"dlp_required": True, "dlp_exceptions": []},
    }
    assert _config.dlp_enabled(cfg) is True


def test_unexpired_exception_allows_disabled():
    future = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    cfg = {
        "dlp": False,
        "org_hook_policy": {
            "dlp_required": True,
            "dlp_exceptions": [{"id": "e1", "expires_at": future}],
        },
    }
    assert _config.dlp_enabled(cfg) is False


def test_expired_exception_enforces_required():
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    cfg = {
        "dlp": False,
        "org_hook_policy": {
            "dlp_required": True,
            "dlp_exceptions": [{"id": "e1", "expires_at": past}],
        },
    }
    assert _config.dlp_enabled(cfg) is True


def test_local_when_not_required():
    assert _config.dlp_enabled({"dlp": False, "org_hook_policy": {}}) is False
    assert _config.dlp_local_setting({"dlp": False}) is False
