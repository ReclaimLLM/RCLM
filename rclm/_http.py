"""Shared HTTP client helpers for ReclaimLLM API calls."""

from __future__ import annotations

import os
import ssl

import aiohttp
import certifi


def create_ssl_context() -> ssl.SSLContext:
    """Return an SSL context that preserves verification and has reliable roots.

    Some Python installs cannot discover a local CA bundle even when the OS has
    one. Add certifi's Mozilla roots to the normal default context so public
    endpoints validate consistently without disabling TLS verification.
    """
    ca_file = (
        os.environ.get("SSL_CERT_FILE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("CURL_CA_BUNDLE")
    )
    if ca_file:
        return ssl.create_default_context(cafile=ca_file)

    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    return context


def create_tcp_connector() -> aiohttp.TCPConnector:
    """Create an aiohttp connector with the ReclaimLLM TLS policy."""
    return aiohttp.TCPConnector(ssl=create_ssl_context())
