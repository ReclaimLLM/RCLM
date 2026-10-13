from __future__ import annotations

from rclm.hooks import uninstaller


def test_uninstall_ping_before_provider_work(monkeypatch):
    calls: list[dict] = []

    def fake_checkin(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("rclm.hooks.checkin.maybe_checkin", fake_checkin)
    monkeypatch.setattr(uninstaller, "_uninstall_settings_provider", lambda path: None)
    monkeypatch.setattr(uninstaller, "_uninstall_codex", lambda path: None)
    monkeypatch.setattr(uninstaller, "_uninstall_cursor", lambda path: None)
    monkeypatch.setattr(uninstaller, "_uninstall_antigravity", lambda path: None)
    monkeypatch.setattr(uninstaller, "_uninstall_openclaw", lambda use_global: None)
    monkeypatch.setattr(uninstaller, "_uninstall_copilot", lambda path: None)
    monkeypatch.setattr("sys.argv", ["rclm-hooks-uninstall", "--claude"])

    uninstaller.main()

    assert calls == [{"event": "uninstall", "force": True, "timeout_s": 2.0}]


def test_uninstall_continues_when_ping_fails(monkeypatch):
    removed: list[str] = []

    def boom(**kwargs):
        raise RuntimeError("network down")

    def track_claude(path):
        removed.append(str(path))

    monkeypatch.setattr("rclm.hooks.checkin.maybe_checkin", boom)
    monkeypatch.setattr(uninstaller, "_uninstall_settings_provider", track_claude)
    monkeypatch.setattr("sys.argv", ["rclm-hooks-uninstall", "--claude"])

    uninstaller.main()
    assert removed  # uninstall still ran
