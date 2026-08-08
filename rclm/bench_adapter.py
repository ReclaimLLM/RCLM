"""Vendor-neutral RCLM Bench adapter over the exact shipped transform core.

This executable has no dependency on RCLM Bench. It implements Bench protocol
v1 over stdin/stdout JSONL so any compatible harness can exercise the runtime
compression implementation without a backend, gateway, or model call.
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from rclm.hooks import dedupe
from rclm.hooks.read_cache import process_read
from rclm.hooks.tool_result_transform import (
    compact_tool_result,
    decision_from_replacement,
    extract_text_envelope,
)
from rclm.replay.read_request import build_read_request

PROTOCOL_VERSION = 1
_KNOWN_MECHANISMS = frozenset({"range_cache", "shell_compaction", "hash_dedupe"})
_SHELL_TOOLS = frozenset({"bash", "exec", "exec_command", "shell"})
_READ_TOOLS = frozenset({"read", *_SHELL_TOOLS})


class RCLMBenchAdapter:
    def __init__(self, settings: dict[str, Any]) -> None:
        requested = settings.get("mechanisms", ["shell_compaction"])
        if not isinstance(requested, list) or any(not isinstance(item, str) for item in requested):
            raise ValueError("settings.mechanisms must be a list of strings")
        unknown = sorted(set(requested) - _KNOWN_MECHANISMS)
        if unknown:
            raise ValueError(f"Unknown RCLM mechanism(s): {', '.join(unknown)}")
        self.mechanisms = frozenset(requested)
        self.read_state: dict[str, Any] = {}
        self.dedupe_state: dict[str, Any] = {}

    def transform(self, frame: dict[str, Any]) -> dict[str, Any]:
        sequence = frame.get("sequence")
        tool_name = frame.get("tool_name")
        tool_input = frame.get("tool_input")
        tool_result = frame.get("tool_result")
        if not isinstance(sequence, int) or not isinstance(tool_name, str):
            raise ValueError("unit requires integer sequence and string tool_name")
        if frame.get("is_error") is True:
            return _result(sequence, "ineligible", tool_result, warning="error result")
        envelope = extract_text_envelope(tool_result)
        if envelope is None:
            return _result(sequence, "ineligible", tool_result, warning="no safe text envelope")

        lowered = tool_name.lower()
        if "range_cache" in self.mechanisms and lowered in _READ_TOOLS:
            built = build_read_request(tool_name, tool_input, envelope.text)
            if built is not None:
                request, trailing = built
                block = (
                    envelope.text[: len(envelope.text) - len(trailing)]
                    if trailing
                    else envelope.text
                )
                decision = process_read(request, block, self.read_state, turn=sequence)
                self.read_state = decision.state
                if decision.cache_hit and decision.replacement is not None:
                    transformed = decision_from_replacement(
                        envelope,
                        decision.replacement + trailing,
                        mechanism="range_cache",
                    )
                    if transformed is not None:
                        return _result(
                            sequence,
                            "applied",
                            transformed.wire_replacement,
                            mechanism="range_cache",
                        )
                return _result(sequence, "ineligible", tool_result)
            if lowered == "read":
                return _result(
                    sequence,
                    "ineligible",
                    tool_result,
                    warning="read range could not be resolved",
                )

        if "shell_compaction" in self.mechanisms and lowered in _SHELL_TOOLS:
            decision = compact_tool_result(tool_name, tool_input, tool_result)
            if decision is not None:
                return _result(
                    sequence,
                    "applied",
                    decision.wire_replacement,
                    mechanism=decision.mechanism,
                )
            return _result(sequence, "ineligible", tool_result)

        if "hash_dedupe" in self.mechanisms:
            replacement, self.dedupe_state, match = dedupe.maybe_dedupe(
                envelope.text,
                self.dedupe_state,
                tool_name=tool_name,
                turn=sequence,
            )
            if replacement and match:
                decision = decision_from_replacement(envelope, replacement, mechanism="hash_dedupe")
                if decision is not None:
                    return _result(
                        sequence,
                        "applied",
                        decision.wire_replacement,
                        mechanism="hash_dedupe",
                    )
        return _result(sequence, "ineligible", tool_result)


def main() -> None:
    adapter: RCLMBenchAdapter | None = None
    for line in sys.stdin:
        try:
            frame = json.loads(line)
            if not isinstance(frame, dict):
                raise ValueError("frame must be a JSON object")
            frame_type = frame.get("type")
            if frame_type == "handshake":
                if frame.get("protocol_version") != PROTOCOL_VERSION:
                    raise ValueError("unsupported protocol version")
                if frame.get("scope") != "tool_result":
                    raise ValueError("RCLM adapter supports tool_result scope only")
                settings = frame.get("settings", {})
                if not isinstance(settings, dict):
                    raise ValueError("handshake settings must be an object")
                adapter = RCLMBenchAdapter(settings)
                response = {
                    "type": "handshake",
                    "protocol_version": PROTOCOL_VERSION,
                    "name": "rclm-runtime",
                    "version": _package_version(),
                    "capabilities": {
                        "scopes": ["tool_result"],
                        "mechanisms": sorted(_KNOWN_MECHANISMS),
                    },
                }
            elif frame_type == "unit":
                if adapter is None:
                    raise ValueError("handshake required before units")
                response = adapter.transform(frame)
            elif frame_type == "close":
                return
            else:
                raise ValueError(f"unknown frame type {frame_type!r}")
        except Exception as exc:
            print(f"rclm-bench-adapter: {exc}", file=sys.stderr, flush=True)
            raise SystemExit(2) from exc
        print(json.dumps(response, separators=(",", ":"), ensure_ascii=False), flush=True)


def _result(
    sequence: int,
    status: str,
    transformed_result: object,
    *,
    mechanism: str | None = None,
    warning: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "result",
        "sequence": sequence,
        "status": status,
        "transformed_result": transformed_result,
        "mechanism": mechanism,
        "warning": warning,
        "metadata": {},
    }


def _package_version() -> str:
    try:
        return version("rclm")
    except PackageNotFoundError:
        return "development"


if __name__ == "__main__":
    main()
