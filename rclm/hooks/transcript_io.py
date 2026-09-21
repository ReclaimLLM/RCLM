"""Bounded local transcript readers for native coding-agent adapters."""

from __future__ import annotations

import json
import logging
from pathlib import Path

MAX_TRANSCRIPT_BYTES = 64 * 1024 * 1024
MAX_TRANSCRIPT_LINES = 100_000
MAX_LINE_BYTES = 2 * 1024 * 1024


def read_jsonl(
    transcript_path: str | None,
    *,
    logger: logging.Logger,
) -> tuple[list[dict], list[str]]:
    """Read bounded JSONL records, rejecting symlinked or non-regular files."""
    if not transcript_path:
        return [], ["transcript_path_missing"]

    path = Path(transcript_path)
    try:
        if path.is_symlink() or not path.is_file():
            logger.warning("transcript file is missing, symlinked, or not regular: %s", path)
            return [], ["transcript_unavailable"]
        if path.stat().st_size > MAX_TRANSCRIPT_BYTES:
            logger.warning("transcript exceeds %d bytes: %s", MAX_TRANSCRIPT_BYTES, path)
            return [], ["transcript_size_limit_exceeded"]
    except OSError:
        logger.warning("could not inspect transcript: %s", path)
        return [], ["transcript_unavailable"]

    records: list[dict] = []
    warnings: list[str] = []
    try:
        with path.open("rb") as fh:
            for line_number, raw_line in enumerate(fh, start=1):
                if line_number > MAX_TRANSCRIPT_LINES:
                    warnings.append("transcript_line_limit_reached")
                    break
                if len(raw_line) > MAX_LINE_BYTES:
                    warnings.append("transcript_line_size_limit_exceeded")
                    continue
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.warning("malformed JSON at %s:%d; skipping", path, line_number)
                    if "malformed_transcript_line" not in warnings:
                        warnings.append("malformed_transcript_line")
                    continue
                if isinstance(value, dict):
                    records.append(value)
    except OSError:
        logger.warning("could not read transcript: %s", path)
        return [], ["transcript_unavailable"]
    return records, warnings


def read_json_document(
    transcript_path: str | None,
    *,
    logger: logging.Logger,
) -> tuple[dict | None, list[str]]:
    """Read one bounded JSON object from a regular, non-symlink file."""
    if not transcript_path:
        return None, ["transcript_path_missing"]
    path = Path(transcript_path)
    try:
        if path.is_symlink() or not path.is_file():
            return None, ["transcript_unavailable"]
        size = path.stat().st_size
        if size > MAX_TRANSCRIPT_BYTES:
            logger.warning("transcript exceeds %d bytes: %s", MAX_TRANSCRIPT_BYTES, path)
            return None, ["transcript_size_limit_exceeded"]
        with path.open("rb") as fh:
            raw = fh.read(MAX_TRANSCRIPT_BYTES + 1)
        if len(raw) > MAX_TRANSCRIPT_BYTES:
            return None, ["transcript_size_limit_exceeded"]
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("could not parse transcript: %s", path)
        return None, ["malformed_or_unavailable_transcript"]
    return (value, []) if isinstance(value, dict) else (None, ["unexpected_transcript_shape"])
