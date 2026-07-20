"""Tests for rclm.auth: the browser device-flow and key validation."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from urllib.parse import unquote

import aiohttp
import pytest

from rclm import auth

# ---------------------------------------------------------------------------
# wait_for_api_key_via_browser: real HTTPServer, driven by real HTTP requests.
# Only webbrowser.open is mocked — everything else (nonce check, socket,
# handler) is the real implementation, never exercised directly before.
# ---------------------------------------------------------------------------


def _post_key(url: str, api_key: str) -> int:
    req = urllib.request.Request(
        url,
        data=json.dumps({"api_key": api_key}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _extract_callback_url(settings_url: str) -> str:
    return unquote(settings_url.split("#cli_callback=", 1)[1])


def _post_key_with_headers(url: str, api_key: str) -> tuple[int, dict[str, str]]:
    req = urllib.request.Request(
        url,
        data=json.dumps({"api_key": api_key}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {})


def test_wait_for_api_key_via_browser_returns_key_from_valid_nonce_post(monkeypatch):
    captured = {}

    def fake_open(url):
        captured["url"] = url
        callback = _extract_callback_url(url)
        threading.Thread(target=_post_key, args=(callback, "sk-test-123"), daemon=True).start()

    monkeypatch.setattr(auth.webbrowser, "open", fake_open)

    key = auth.wait_for_api_key_via_browser("https://app.test")

    assert key == "sk-test-123"
    # The callback URL (which is the only access control on the local
    # server) must live in the fragment, not the query string — fragments
    # aren't sent to the server or logged by page analytics on load.
    assert captured["url"].startswith("https://app.test/settings#cli_callback=")
    assert "?" not in captured["url"]


def test_wait_for_api_key_via_browser_scopes_cors_to_app_origin(monkeypatch):
    """The local callback server must not accept cross-origin POSTs from arbitrary sites."""
    result = {}
    threads: list[threading.Thread] = []

    def fake_open(url):
        callback = _extract_callback_url(url)

        def drive_request():
            status, headers = _post_key_with_headers(callback, "sk-test-456")
            result["status"] = status
            result["origin_header"] = headers.get("Access-Control-Allow-Origin")

        t = threading.Thread(target=drive_request, daemon=True)
        threads.append(t)
        t.start()

    monkeypatch.setattr(auth.webbrowser, "open", fake_open)

    key = auth.wait_for_api_key_via_browser("https://app.test")
    for t in threads:
        t.join(timeout=5)

    assert key == "sk-test-456"
    assert result["status"] == 200
    assert result["origin_header"] == "https://app.test"


def test_wait_for_api_key_via_browser_rejects_wrong_nonce_path(monkeypatch):
    """A POST to the wrong path must 404 and must not satisfy the wait loop."""

    def fake_open(url):
        callback = _extract_callback_url(url)
        wrong_path_url = callback.rsplit("/", 1)[0] + "/not-the-real-nonce"

        def drive_requests():
            status = _post_key(wrong_path_url, "sk-should-not-be-used")
            assert status == 404
            _post_key(callback, "sk-correct")

        threading.Thread(target=drive_requests, daemon=True).start()

    monkeypatch.setattr(auth.webbrowser, "open", fake_open)

    key = auth.wait_for_api_key_via_browser("https://app.test")

    assert key == "sk-correct"


# ---------------------------------------------------------------------------
# wait_for_api_key_via_browser: pasting a key into stdin instead of using
# the browser. Uses a real os.pipe() as stdin so select()/readline() behave
# exactly as they would against a real terminal.
# ---------------------------------------------------------------------------


def test_pasted_key_short_circuits_the_browser_wait(monkeypatch):
    monkeypatch.setattr(auth, "_can_watch_stdin", lambda: True)
    read_fd, write_fd = os.pipe()
    fake_stdin = os.fdopen(read_fd)
    monkeypatch.setattr(auth.sys, "stdin", fake_stdin)

    def fake_open(_url):
        os.write(write_fd, b"sk-pasted-key\n")

    monkeypatch.setattr(auth.webbrowser, "open", fake_open)

    try:
        key = auth.wait_for_api_key_via_browser("https://app.test")
    finally:
        fake_stdin.close()
        os.close(write_fd)

    assert key == "sk-pasted-key"


def test_pasted_key_strips_surrounding_quotes(monkeypatch):
    monkeypatch.setattr(auth, "_can_watch_stdin", lambda: True)
    read_fd, write_fd = os.pipe()
    fake_stdin = os.fdopen(read_fd)
    monkeypatch.setattr(auth.sys, "stdin", fake_stdin)

    def fake_open(_url):
        os.write(write_fd, b'"sk-quoted-key"\n')

    monkeypatch.setattr(auth.webbrowser, "open", fake_open)

    try:
        key = auth.wait_for_api_key_via_browser("https://app.test")
    finally:
        fake_stdin.close()
        os.close(write_fd)

    assert key == "sk-quoted-key"


def test_browser_callback_still_wins_when_stdin_is_watched(monkeypatch):
    """The browser POST path must still work when stdin-watching is enabled but unused."""
    monkeypatch.setattr(auth, "_can_watch_stdin", lambda: True)
    read_fd, write_fd = os.pipe()
    fake_stdin = os.fdopen(read_fd)
    monkeypatch.setattr(auth.sys, "stdin", fake_stdin)

    def fake_open(url):
        callback = _extract_callback_url(url)
        threading.Thread(target=_post_key, args=(callback, "sk-from-browser"), daemon=True).start()

    monkeypatch.setattr(auth.webbrowser, "open", fake_open)

    try:
        key = auth.wait_for_api_key_via_browser("https://app.test")
    finally:
        fake_stdin.close()
        os.close(write_fd)

    assert key == "sk-from-browser"


# ---------------------------------------------------------------------------
# validate_api_key
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int | None) -> None:
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, *, status: int | None = None, error: Exception | None = None) -> None:
        self._status = status
        self._error = error

    def get(self, url):
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._status)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _patch_session(monkeypatch, *, status=None, error=None):
    monkeypatch.setattr(
        auth.aiohttp,
        "ClientSession",
        lambda **kwargs: _FakeSession(status=status, error=error),
    )


@pytest.mark.asyncio
async def test_validate_api_key_returns_true_on_200(monkeypatch):
    _patch_session(monkeypatch, status=200)

    assert await auth.validate_api_key("https://api.test", "sk-key") is True


@pytest.mark.asyncio
async def test_validate_api_key_returns_false_on_401(monkeypatch):
    _patch_session(monkeypatch, status=401)

    assert await auth.validate_api_key("https://api.test", "sk-key") is False


@pytest.mark.asyncio
async def test_validate_api_key_returns_none_on_network_error(monkeypatch):
    _patch_session(monkeypatch, error=aiohttp.ClientConnectionError())

    assert await auth.validate_api_key("https://api.test", "sk-key") is None


@pytest.mark.asyncio
async def test_validate_api_key_returns_none_on_unexpected_server_error(monkeypatch):
    _patch_session(monkeypatch, status=500)

    assert await auth.validate_api_key("https://api.test", "sk-key") is None
