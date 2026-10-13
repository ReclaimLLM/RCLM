"""rclm heartbeat — integrity + check-in, optional OS timer install."""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

from rclm import _config, _device
from rclm.hooks import bootstrap, checkin, integrity
from rclm.hooks.updater import installed_version

_LAUNCH_AGENT_LABEL = "com.reclaimllm.heartbeat"
_LAUNCH_AGENT_PATH = Path.home() / "Library/LaunchAgents" / f"{_LAUNCH_AGENT_LABEL}.plist"
_SYSTEMD_DIR = Path.home() / ".config/systemd/user"
_SYSTEMD_SERVICE = _SYSTEMD_DIR / "rclm-heartbeat.service"
_SYSTEMD_TIMER = _SYSTEMD_DIR / "rclm-heartbeat.timer"


def _parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run coverage heartbeat check-in, or install/uninstall the OS timer",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install macOS LaunchAgent or systemd user timer (every 6h)",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove the installed heartbeat timer",
    )
    return parser.parse_args()


def _rclm_heartbeat_command() -> str:
    exe = shutil.which("rclm-heartbeat")
    if exe:
        return exe
    return f"{sys.executable} -m rclm.heartbeat"


def _write_launch_agent() -> None:
    cmd = _rclm_heartbeat_command()
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{_LAUNCH_AGENT_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{cmd}</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>StartInterval</key>
  <integer>21600</integer>
  <key>StandardOutPath</key>
  <string>{Path.home() / ".reclaimllm/heartbeat.log"}</string>
  <key>StandardErrorPath</key>
  <string>{Path.home() / ".reclaimllm/heartbeat.err.log"}</string>
</dict>
</plist>
"""
    _LAUNCH_AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LAUNCH_AGENT_PATH.write_text(plist, encoding="utf-8")
    try:
        import subprocess

        subprocess.run(["launchctl", "unload", str(_LAUNCH_AGENT_PATH)], check=False)
        subprocess.run(["launchctl", "load", str(_LAUNCH_AGENT_PATH)], check=False)
    except OSError:
        pass


def _write_systemd_timer() -> None:
    cmd = _rclm_heartbeat_command()
    _SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
    _SYSTEMD_SERVICE.write_text(
        f"""[Unit]
Description=ReclaimLLM coverage heartbeat
[Service]
Type=oneshot
ExecStart={cmd}
""",
        encoding="utf-8",
    )
    _SYSTEMD_TIMER.write_text(
        """[Unit]
Description=ReclaimLLM coverage heartbeat timer
[Timer]
OnBootSec=2min
OnUnitActiveSec=6h
Persistent=true
[Install]
WantedBy=timers.target
""",
        encoding="utf-8",
    )
    try:
        import subprocess

        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", "rclm-heartbeat.timer"],
            check=False,
        )
    except OSError:
        pass


def install_timer() -> None:
    system = platform.system().lower()
    if system == "darwin":
        _write_launch_agent()
        print(f"Installed LaunchAgent at {_LAUNCH_AGENT_PATH}")
    elif system == "linux":
        _write_systemd_timer()
        print(f"Installed systemd user timer at {_SYSTEMD_TIMER}")
    else:
        print(f"Heartbeat timer not supported on {system}", file=sys.stderr)
        sys.exit(1)
    _config.patch(heartbeat_enabled=True)


def uninstall_timer() -> None:
    system = platform.system().lower()
    if system == "darwin" and _LAUNCH_AGENT_PATH.exists():
        try:
            import subprocess

            subprocess.run(["launchctl", "unload", str(_LAUNCH_AGENT_PATH)], check=False)
        except OSError:
            pass
        _LAUNCH_AGENT_PATH.unlink(missing_ok=True)
        print("Removed LaunchAgent heartbeat.")
    elif system == "linux":
        try:
            import subprocess

            subprocess.run(
                ["systemctl", "--user", "disable", "--now", "rclm-heartbeat.timer"],
                check=False,
            )
        except OSError:
            pass
        _SYSTEMD_SERVICE.unlink(missing_ok=True)
        _SYSTEMD_TIMER.unlink(missing_ok=True)
        print("Removed systemd heartbeat timer.")
    _config.patch(heartbeat_enabled=False)


def run_once() -> int:
    """Refresh policy, run integrity, force check-in."""
    _device.ensure_device_id()
    try:
        import asyncio

        asyncio.run(bootstrap.fetch("", include_context=False))
    except Exception as exc:  # noqa: BLE001
        print(f"Policy refresh skipped: {exc}", file=sys.stderr)
    clients = integrity.collect_client_status()
    print(f"rclm {installed_version()} heartbeat — {len(clients)} clients checked")
    for name, status in sorted(clients.items()):
        print(f"  {name}: {status.get('status')} ({status.get('scope')})")
    checkin.maybe_checkin(force=True)
    return 0


def main() -> None:
    args = _parse_flags()
    if args.install and args.uninstall:
        print("Use only one of --install / --uninstall", file=sys.stderr)
        sys.exit(2)
    if args.install:
        install_timer()
        run_once()
        return
    if args.uninstall:
        uninstall_timer()
        return
    sys.exit(run_once())


if __name__ == "__main__":
    main()
