"""Entry point for rclm-update.

Checks PyPI for the latest rclm version, upgrades via pip if a newer
version is available, then re-installs hooks for all configured providers so
any new hook commands take effect immediately.

Usage:
    rclm-update          # check and upgrade if needed
    rclm-update --check  # print status only, do not upgrade
"""

from __future__ import annotations

import argparse
import sys

from rclm import _config
from rclm.hooks.updater import apply_update, check_for_update, installed_version


def _parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upgrade rclm to the latest version from PyPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s           # upgrade if a newer version is available
  %(prog)s --check   # print status only, do not upgrade""",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print the current and latest version without upgrading",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_flags()
    current = installed_version()

    print("Checking for updates...")
    latest = check_for_update(force=True)

    if latest is None:
        print(f"rclm is up to date ({current}).")
        return

    print(f"Current: {current}  →  Latest: {latest}")

    if args.check:
        print("\nRun rclm-update to upgrade.")
        return

    print()
    print("Upgrading rclm...")
    success = apply_update()

    if not success:
        print(
            "\nUpgrade failed. Run manually:",
            file=sys.stderr,
        )
        print(
            f"  {sys.executable} -m pip install --upgrade rclm",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n✓ Upgraded to {latest}")

    # Re-install hooks so any changed hook commands are written to config files.
    # Uses saved config (no credential prompt) and defaults to global scope
    # to match the install default.
    cfg = _config.load()
    compress = cfg.get("compress", False)

    try:
        from rclm.hooks.installer import (
            _install_claude,
            _install_codex,
            _install_gemini,
        )

        print("Re-installing hooks (global)...")
        _install_claude(use_global=True, compress_enabled=compress)
        _install_gemini(use_global=True)
        _install_codex(use_global=True)
    except Exception as exc:
        # Hook reinstall failing should not abort the update — the pip upgrade succeeded.
        print(
            f"\nNote: hook reinstall encountered an error ({exc}).\n"
            "Run rclm-hooks-install to refresh hook configs manually.",
            file=sys.stderr,
        )
