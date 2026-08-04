"""Entry point for rclm-update.

Checks PyPI for the latest rclm version, refreshes remote settings, upgrades
via pip if a newer version is available, then re-installs hooks for all
configured providers so any new hook commands take effect immediately.

Usage:
    rclm-update          # check and upgrade if needed
    rclm-update --check  # print status only, do not upgrade
    rclm-update --with-mcp  # upgrade if needed and install MCP config
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from rclm import _config
from rclm.hooks.updater import apply_update, check_for_update, installed_version


def _sync_redaction_settings() -> None:
    try:
        from rclm.hooks.redaction import load_settings, sync_remote_settings

        if sync_remote_settings():
            settings = load_settings()
            count = len(settings.remote_substitutions)
            print(f"Synced redaction settings ({count} remote substitutions).")
        else:
            print("Redaction settings sync skipped or unavailable.")
    except Exception as exc:
        print(f"Redaction settings sync failed: {exc}", file=sys.stderr)


def _sync_org_hook_policy() -> None:
    """Refresh the cached organization hook policy without blocking updates."""
    try:
        from rclm.hooks import bootstrap

        result = asyncio.run(bootstrap.fetch("", include_context=False))
        policy = result.get("org_hook_policy")
        if isinstance(policy, dict):
            version = policy.get("policy_version")
            suffix = f" (v{version})" if isinstance(version, int) else ""
            print(f"Synced organization hook policy{suffix}.")
        elif "org_hook_policy" in result:
            print("No organization hook policy assigned.")
        else:
            print("Organization hook policy sync skipped or unavailable.")
    except Exception as exc:
        print(f"Organization hook policy sync failed: {exc}", file=sys.stderr)


def _parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upgrade rclm to the latest version from PyPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s           # upgrade if a newer version is available
  %(prog)s --check   # print status only, do not upgrade
  %(prog)s --with-mcp # also install ReclaimLLM MCP server config""",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print the current and latest version without upgrading",
    )
    parser.add_argument(
        "--with-mcp",
        action="store_true",
        help="Install ReclaimLLM MCP server config after update check/upgrade",
    )
    return parser.parse_args()


def _install_mcp_config() -> None:
    try:
        from rclm.mcp_install import install_mcp

        install_mcp(use_global=True)
    except Exception as exc:
        print(f"MCP install failed: {exc}", file=sys.stderr)


def main() -> None:
    args = _parse_flags()
    current = installed_version()

    print("Checking for updates...")
    latest = check_for_update(force=True)
    _sync_redaction_settings()
    _sync_org_hook_policy()

    if latest is None:
        print(f"rclm is up to date ({current}).")
        if args.with_mcp:
            _install_mcp_config()
        return

    print(f"Current: {current}  →  Latest: {latest}")

    if args.check:
        print("\nRun rclm-update to upgrade.")
        if args.with_mcp:
            print("Skipping MCP install because --check was used.")
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
    compress = _config.compression_config(cfg)["enabled"]

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

    if args.with_mcp:
        _install_mcp_config()
