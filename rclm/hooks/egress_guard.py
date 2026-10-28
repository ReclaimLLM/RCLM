"""Pre-execution hook guard for external network egress (URL / IP / Country policy).

Only makes a backend lookup (`/api/settings/egress-check`) when a tool call
initiates an external network egress request. Non-egress tool calls return
immediately with zero network overhead.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import shlex
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import aiohttp

from rclm import _config
from rclm._http import create_tcp_connector

logger = logging.getLogger(__name__)

_EGRESS_CHECK_PATH = "/api/settings/egress-check"
_TIMEOUT_S = 2.5

_SHELL_TOOLS = frozenset(
    {
        "bash",
        "shell",
        "sh",
        "zsh",
        "terminal",
        "run_command",
        "runcommand",
        "run_shell_command",
        "exec",
        "exec_command",
    }
)

_WEB_TOOLS = frozenset(
    {
        "webfetch",
        "web_fetch",
        "fetch_url",
        "http_request",
        "read_url_content",
        "open_url",
    }
)

_EGRESS_BINARIES = frozenset(
    {
        "curl",
        "wget",
        "http",
        "https",
        "xh",
        "curlie",
        "aria2c",
        "fetch",
        "iwr",
        "invoke-webrequest",
        "irm",
        "invoke-restmethod",
        "scp",
        "sftp",
        "rsync",
        "nc",
        "ncat",
        "netcat",
        "socat",
        "telnet",
        "ftp",
        "tftp",
    }
)

_SCRIPT_BINARIES = frozenset(
    {"python", "python3", "node", "nodejs", "ruby", "perl", "php", "deno", "bun"}
)

_URL_RE = re.compile(
    r"\b(?:https?|ftp|sftp|scp|ws|wss|hxxps?)://[^\s'\"<>`|;)]+",
    re.IGNORECASE,
)
_DEV_TCP_RE = re.compile(r"/dev/(?:tcp|udp)/([A-Za-z0-9._-]+|\[[0-9a-fA-F:]+\])/\d+")
_SCP_REMOTE_RE = re.compile(
    r"^(?:[A-Za-z0-9._-]+@)?([A-Za-z0-9.-]+\.[A-Za-z]{2,}|\d{1,3}(?:\.\d{1,3}){3}):.+$"
)
_BARE_IPV4_RE = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")

_LOCAL_HOSTS = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "host.docker.internal",
        "gateway.docker.internal",
        "kubernetes.default",
        "kubernetes.default.svc",
    }
)

_METADATA_HOSTS = frozenset(
    {
        "169.254.169.254",
        "169.254.170.2",
        "100.100.100.200",
        "fd00:ec2::254",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
        "instance-data.ec2.internal",
    }
)


@dataclass(frozen=True)
class EgressVerdict:
    allowed: int  # 1 = allowed, 0 = blocked
    reason: str | None = None
    blocked_targets: tuple[dict[str, Any], ...] = ()


def _is_local_target(host_or_ip: str) -> bool:
    h = host_or_ip.strip().strip("[]").lower().rstrip(".")
    if not h:
        return True
    if h in _METADATA_HOSTS:
        return False
    if h in _LOCAL_HOSTS or h.endswith((".localhost", ".local")):
        return True
    try:
        ip = ipaddress.ip_address(h)
        if str(ip) in _METADATA_HOSTS:
            return False
        if ip.version == 4 and ip in ipaddress.ip_network("169.254.0.0/16"):
            return False
        return bool(ip.is_loopback or ip.is_unspecified)
    except ValueError:
        return False


def _normalize_candidate(raw_url_or_host: str) -> str | None:
    val = raw_url_or_host.strip().rstrip(".,;:'\")>\\]")
    if not val:
        return None
    if val.lower().startswith(("hxxp://", "hxxps://")):
        val = "http" + val[4:]
    if "://" in val:
        try:
            parsed = urlparse(val)
            host = (parsed.hostname or "").strip().lower().rstrip(".")
            if not host or _is_local_target(host):
                return None
            return val
        except ValueError:
            return None

    host = val.split("/", 1)[0]
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if host.startswith("[") and "]" in host:
        host = host[1 : host.index("]")]
    elif host.count(":") == 1:
        h_part, p_part = host.rsplit(":", 1)
        if p_part.isdigit():
            host = h_part
    host = host.strip().lower().rstrip(".")
    if not host or _is_local_target(host):
        return None
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    if "." in host and re.match(r"^[a-z0-9.-]+\.[a-z]{2,}$", host):
        return host
    return None


def _extract_from_shell_command(command: str) -> list[str]:
    if not command or not isinstance(command, str):
        return []

    cmd_lower = command.lower()
    has_egress_hint = (
        any(bin_name in cmd_lower for bin_name in _EGRESS_BINARIES)
        or "://" in cmd_lower
        or "/dev/tcp/" in cmd_lower
        or "/dev/udp/" in cmd_lower
        or "169.254.169.254" in cmd_lower
    )
    if not has_egress_hint:
        return []

    targets: list[str] = []
    seen: set[str] = set()

    def _add(raw_item: str) -> None:
        norm = _normalize_candidate(raw_item)
        if norm and norm not in seen:
            seen.add(norm)
            targets.append(norm)

    for m in _URL_RE.finditer(command):
        _add(m.group(0))

    for m in _DEV_TCP_RE.finditer(command):
        _add(m.group(1))

    # Tokenize command segments to catch bare hosts/IPs on curl/wget/scp/nc/etc.
    cleaned = command.replace("\\\n", " ")
    for segment in re.split(r"(?:\|\||&&|[|;\n])", cleaned):
        segment = segment.strip()
        if not segment:
            continue
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            tokens = segment.split()
        if not tokens:
            continue
        idx = 0
        while idx < len(tokens) and (
            "=" in tokens[idx]
            or os.path.basename(tokens[idx]).lower() in {"sudo", "env", "nohup", "time", "command"}
        ):
            idx += 1
        if idx >= len(tokens):
            continue
        binary = os.path.basename(tokens[idx]).lower()
        rest = tokens[idx + 1 :]
        if binary in _EGRESS_BINARIES:
            for tok in rest:
                if tok.startswith("-"):
                    continue
                scp_m = _SCP_REMOTE_RE.match(tok)
                if scp_m:
                    _add(scp_m.group(1))
                else:
                    _add(tok)
        elif binary in _SCRIPT_BINARIES:
            for m in _BARE_IPV4_RE.finditer(segment):
                _add(m.group(1))

    return targets


def extract_egress_targets(tool_name: str, tool_input: dict[str, Any]) -> list[str]:
    """Extract non-local external URLs/hosts/IPs from a tool invocation, or `[]`."""
    if not isinstance(tool_input, dict):
        return []
    norm_tool = (tool_name or "").strip().lower()

    if norm_tool in _SHELL_TOOLS:
        cmd = (
            tool_input.get("command")
            or tool_input.get("CommandLine")
            or tool_input.get("cmd")
            or ""
        )
        return _extract_from_shell_command(str(cmd)) if cmd else []

    if norm_tool in _WEB_TOOLS:
        targets: list[str] = []
        for key in ("url", "uri", "target", "href"):
            val = tool_input.get(key)
            if isinstance(val, str) and val.strip():
                norm = _normalize_candidate(val)
                if norm:
                    targets.append(norm)
        return targets

    return []


async def _async_query_backend(
    targets: list[str],
    *,
    cfg: dict[str, Any] | None = None,
    timeout_s: float = _TIMEOUT_S,
) -> EgressVerdict:
    creds = _config.resolve_credentials(cfg)
    if creds is None:
        return EgressVerdict(allowed=1)

    url = creds.server_url.rstrip("/") + _EGRESS_CHECK_PATH
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": creds.api_key,
    }
    payload = {"targets": targets}
    try:
        async with (
            aiohttp.ClientSession(
                connector=create_tcp_connector(),
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as session,
            session.post(url, json=payload, headers=headers) as resp,
        ):
            if resp.status >= 400:
                logger.debug("egress-check returned %s; failing open", resp.status)
                return EgressVerdict(allowed=1)
            data = await resp.json()
    except Exception as exc:  # noqa: BLE001 — fail open if backend is unreachable
        logger.debug("egress-check network error; failing open: %s", exc)
        return EgressVerdict(allowed=1)

    allowed_bit = 0 if data.get("allowed") in (0, False, "0") else 1
    if allowed_bit == 1:
        return EgressVerdict(allowed=1)

    results = data.get("results") or []
    blocked = [r for r in results if isinstance(r, dict) and r.get("allowed") in (0, False, "0")]
    parts: list[str] = []
    for item in blocked[:3]:
        host = item.get("host") or item.get("target") or "unknown"
        ips = item.get("ips") or []
        countries = item.get("country_codes") or []
        reason_code = str(item.get("reason") or "policy_blocked")
        detail_bits: list[str] = []
        if ips:
            detail_bits.append(f"IP: {ips[0]}")
        if countries:
            detail_bits.append(f"country: {', '.join(countries)}")
        detail_bits.append(reason_code)
        parts.append(f"{host} ({', '.join(detail_bits)})")

    summary = ", ".join(parts) if parts else ", ".join(targets[:3])
    reason = f"ReclaimLLM Egress Policy blocked outbound connection to {summary}."
    return EgressVerdict(allowed=0, reason=reason, blocked_targets=tuple(blocked))


def check_tool_egress(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
    timeout_s: float = _TIMEOUT_S,
) -> EgressVerdict:
    """Synchronous pre-tool hook helper.

    Returns immediately (`allowed=1`) without making any network call if the
    tool call does not contain an external network egress target.
    """
    targets = extract_egress_targets(tool_name, tool_input)
    if not targets:
        return EgressVerdict(allowed=1)
    try:
        return asyncio.run(_async_query_backend(targets, cfg=cfg, timeout_s=timeout_s))
    except Exception as exc:  # noqa: BLE001
        logger.debug("egress check failed open: %s", exc)
        return EgressVerdict(allowed=1)
