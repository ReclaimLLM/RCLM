"""Auth-only entrypoint: rclm-login.

Obtains and validates a ReclaimLLM API key, then saves it to
~/.reclaimllm/config.json for use by rclm-mcp, the hooks, and the proxy.
Does not touch any hook/MCP configuration — use `rclm-hooks-install` for that.

Usage:
    rclm-login                     # opens a browser to create/fetch a key
    rclm-login --api-key=<key>     # explicit key (skips browser)
    rclm-login --server-url=<url>  # override the default backend URL
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from rclm import _config, auth


def _parse_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sign in to ReclaimLLM and save credentials for rclm-mcp / hooks / proxy",
    )
    parser.add_argument("--api-key", help="API key for ReclaimLLM server authentication")
    parser.add_argument(
        "--server-url",
        default=None,
        help=f"ReclaimLLM server URL (default: {auth.DEFAULT_BACKEND_SERVER_URL})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_flags()

    saved = _config.load()
    server_url = args.server_url or saved.get("server_url") or auth.DEFAULT_BACKEND_SERVER_URL
    server_url = server_url.replace('"', "").replace("'", "").strip()

    api_key = args.api_key
    if not api_key:
        api_key = auth.wait_for_api_key_via_browser(auth.DEFAULT_FRONTEND_URL)
        if not api_key:
            sys.exit(1)
    api_key = api_key.replace('"', "").replace("'", "").strip()

    result = asyncio.run(auth.validate_api_key(server_url, api_key))

    if result is False:
        print(
            f"That API key was rejected by {server_url}.\n"
            f"Create a new one at {auth.SETUP_URL} and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    if result is None:
        print(
            f"Warning: couldn't reach {server_url} to verify the key (network issue). "
            "Saved anyway — run `rclm-login` again if problems persist.",
            file=sys.stderr,
        )

    _config.save(server_url, api_key)
    print(f"Logged in. Credentials saved to {_config.CONFIG_PATH}.")


if __name__ == "__main__":
    main()
