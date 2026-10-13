from __future__ import annotations

from pathlib import Path

from rclm import heartbeat


def test_write_launch_agent_content(tmp_path: Path, monkeypatch):
    agent = tmp_path / "com.reclaimllm.heartbeat.plist"
    monkeypatch.setattr(heartbeat, "_LAUNCH_AGENT_PATH", agent)
    monkeypatch.setattr(heartbeat, "_rclm_heartbeat_command", lambda: "/usr/bin/rclm-heartbeat")
    monkeypatch.setattr(heartbeat.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(heartbeat._config, "patch", lambda **kw: None)

    heartbeat._write_launch_agent()
    text = agent.read_text(encoding="utf-8")
    assert "com.reclaimllm.heartbeat" in text
    assert "/usr/bin/rclm-heartbeat" in text
    assert "<integer>21600</integer>" in text
    assert "<true/>" in text  # RunAtLoad


def test_write_systemd_timer_content(tmp_path: Path, monkeypatch):
    systemd = tmp_path / "systemd"
    service = systemd / "rclm-heartbeat.service"
    timer = systemd / "rclm-heartbeat.timer"
    monkeypatch.setattr(heartbeat, "_SYSTEMD_DIR", systemd)
    monkeypatch.setattr(heartbeat, "_SYSTEMD_SERVICE", service)
    monkeypatch.setattr(heartbeat, "_SYSTEMD_TIMER", timer)
    monkeypatch.setattr(heartbeat, "_rclm_heartbeat_command", lambda: "/usr/bin/rclm-heartbeat")

    heartbeat._write_systemd_timer()
    assert "ExecStart=/usr/bin/rclm-heartbeat" in service.read_text(encoding="utf-8")
    assert "OnUnitActiveSec=6h" in timer.read_text(encoding="utf-8")


def test_install_timer_idempotent(tmp_path: Path, monkeypatch):
    agent = tmp_path / "com.reclaimllm.heartbeat.plist"
    patches: list[dict] = []
    monkeypatch.setattr(heartbeat, "_LAUNCH_AGENT_PATH", agent)
    monkeypatch.setattr(heartbeat, "_rclm_heartbeat_command", lambda: "rclm-heartbeat")
    monkeypatch.setattr(heartbeat.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(heartbeat._config, "patch", lambda **kw: patches.append(kw))

    heartbeat.install_timer()
    heartbeat.install_timer()
    assert agent.exists()
    assert patches[-1] == {"heartbeat_enabled": True}
