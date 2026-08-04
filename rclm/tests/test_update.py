from __future__ import annotations

from argparse import Namespace

from rclm import update
from rclm.hooks import bootstrap


def test_sync_org_hook_policy_reuses_bootstrap(monkeypatch, capsys):
    calls: list[tuple[str, bool]] = []

    async def fake_fetch(cwd: str, *, include_context: bool) -> dict:
        calls.append((cwd, include_context))
        return {"org_hook_policy": {"policy_version": 7}}

    monkeypatch.setattr(bootstrap, "fetch", fake_fetch)

    update._sync_org_hook_policy()

    assert calls == [("", False)]
    assert capsys.readouterr().out == "Synced organization hook policy (v7).\n"


def test_sync_org_hook_policy_failure_does_not_raise(monkeypatch, capsys):
    async def fail_fetch(_cwd: str, *, include_context: bool) -> dict:
        raise RuntimeError("offline")

    monkeypatch.setattr(bootstrap, "fetch", fail_fetch)

    update._sync_org_hook_policy()

    assert "Organization hook policy sync failed: offline" in capsys.readouterr().err


def test_main_syncs_org_policy_when_package_is_current(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(update, "_parse_flags", lambda: Namespace(check=False, with_mcp=False))
    monkeypatch.setattr(update, "installed_version", lambda: "1.2.3")
    monkeypatch.setattr(update, "check_for_update", lambda *, force: None)
    monkeypatch.setattr(update, "_sync_redaction_settings", lambda: calls.append("redaction"))
    monkeypatch.setattr(update, "_sync_org_hook_policy", lambda: calls.append("hook_policy"))

    update.main()

    assert calls == ["redaction", "hook_policy"]
