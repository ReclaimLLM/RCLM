"""Shared ReclaimLLM CLI auth: browser device-flow + key validation.

Used by both `rclm-hooks-install` (installer.py) and `rclm-login` (login.py),
and referenced by `rclm-mcp` (mcp_server.py) for its auth-required error text.
"""

from __future__ import annotations

import json
import secrets
import select
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import quote, urlsplit

import aiohttp

from rclm._http import create_tcp_connector

DEFAULT_FRONTEND_URL = "https://reclaimllm.com"
DEFAULT_BACKEND_SERVER_URL = "https://api.reclaimllm.com"
SETUP_URL = DEFAULT_FRONTEND_URL + "/settings#api_key"

AUTH_REQUIRED_MESSAGE = (
    f"ReclaimLLM: not authenticated. Run `rclm-login` to sign in (create a key at {SETUP_URL})."
)

_CALLBACK_TIMEOUT_S = 300  # 5 minutes
_VALIDATE_TIMEOUT_S = 10


def _can_watch_stdin() -> bool:
    """Whether we can multiplex a pasted API key alongside the browser wait.

    select() on stdin only works for regular files/TTYs on POSIX; on Windows
    select() is socket-only, so pasting falls back to being unavailable there.
    """
    return sys.stdin.isatty() and sys.platform != "win32"


def wait_for_api_key_via_browser(app_url: str) -> str | None:
    """Open the settings page in a browser and wait for the user to POST back an API key.

    A one-time nonce is embedded in the callback path so that only the rclm
    app (which receives the full URL) can successfully POST to the local server.
    Any request to a different path is rejected with 404.
    """
    received_key: list[str] = []
    nonce = secrets.token_urlsafe(16)
    # Origin the local server will accept POSTs from — the app's scheme+host
    # only (no path), which is what browsers send in the `Origin` header.
    allowed_origin = urlsplit(app_url)._replace(path="", query="", fragment="").geturl()

    class _Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            if self.path != f"/{nonce}":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self._cors_headers()
            self.end_headers()

        def do_POST(self) -> None:
            if self.path != f"/{nonce}":
                self.send_response(404)
                self.end_headers()
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                key = body.get("api_key", "").strip()
                if key:
                    received_key.append(key)
            except Exception:
                pass
            self.send_response(200)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def _cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            # Chrome 98+ (Private Network Access spec): HTTPS pages need this
            # header in the preflight response to be allowed to POST to localhost.
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    # cli_callback lives entirely in the fragment, not the query string:
    # fragments are never sent to the server and aren't captured by page
    # analytics that log window.location.href on load.
    callback_url = f"http://localhost:{port}/{nonce}"
    settings_url = f"{app_url}/settings#cli_callback={quote(callback_url, safe='')}"

    watch_stdin = _can_watch_stdin()

    print(
        f"No API key configured. Opening browser to create one...\n  {settings_url}\n",
        file=sys.stderr,
    )
    if watch_stdin:
        print(
            f"Waiting for key from browser, or paste one from {SETUP_URL} "
            "and press Enter... (Ctrl+C to cancel)\n",
            file=sys.stderr,
        )
    else:
        print("Waiting for key from browser... (Ctrl+C to cancel)\n", file=sys.stderr)
    webbrowser.open(settings_url)

    deadline = time.monotonic() + _CALLBACK_TIMEOUT_S
    server.timeout = 1.0
    try:
        while not received_key and time.monotonic() < deadline:
            if not watch_stdin:
                server.handle_request()
                continue

            remaining = max(0.0, min(1.0, deadline - time.monotonic()))
            readable, _, _ = select.select([server.fileno(), sys.stdin.fileno()], [], [], remaining)
            if sys.stdin.fileno() in readable:
                pasted = sys.stdin.readline().strip().strip("\"'")
                if pasted:
                    received_key.append(pasted)
                    break
            if server.fileno() in readable:
                server.handle_request()
    except KeyboardInterrupt:
        print("\nCancelled. To install manually, run:", file=sys.stderr)
        print("  rclm-login --api-key=<your-key>", file=sys.stderr)
        return None
    finally:
        server.server_close()

    if not received_key:
        print(
            f"Timed out waiting for API key.\n"
            f"Visit {app_url}/settings to create a key, then run:\n"
            "  rclm-login --api-key=<your-key>",
            file=sys.stderr,
        )
        return None

    return received_key[0]


async def validate_api_key(server_url: str, api_key: str) -> bool | None:
    """Check a key against the backend's /api/whoami endpoint.

    Returns True if accepted, False if explicitly rejected (401/403), or
    None if the check itself failed (timeout/network error) — callers should
    treat None as "couldn't verify," not as "key is bad."
    """
    url = server_url.rstrip("/") + "/api/whoami"
    timeout = aiohttp.ClientTimeout(total=_VALIDATE_TIMEOUT_S)
    try:
        async with (
            aiohttp.ClientSession(
                headers={"X-API-Key": api_key},
                connector=create_tcp_connector(),
                timeout=timeout,
            ) as session,
            session.get(url) as resp,
        ):
            if resp.status in (401, 403):
                return False
            if resp.status < 400:
                return True
            return None  # unexpected status (e.g. 5xx) — couldn't verify either way
    except (aiohttp.ClientError, TimeoutError, OSError):
        return None
