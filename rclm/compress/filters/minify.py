"""Residual minifiers for tool output that command-aware filters leave untouched.

Two lossy folds, each attributed separately and always paired with a recall
pointer by the caller:

* ``repeated_line_collapse`` keeps the first of 3+ non-adjacent lines that differ
  only in their digits (retry loops, progress logs, polling output).
* ``blob_placeholder`` replaces long base64 or hex runs with their size and a
  short digest, since the encoded bytes carry no meaning for the model.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

REPEATED_LINE_MECHANISM = "repeated_line_collapse"
BLOB_MECHANISM = "blob_placeholder"

_MAX_INPUT_CHARS = 2_000_000
_MIN_REPEATS = 5
# Short lines such as "}" or "ok" repeat legitimately and cost almost nothing.
_MIN_REPEATED_LINE_CHARS = 20
_MIN_SAVED_CHARS = 200
# Only prose-like log lines are collapsed. In tables, CSV and key/value data the
# numbers are the payload, so those lines are never treated as repeats.
_MIN_WORDS = 3
_MAX_DIGIT_SHARE = 0.3
_DIGITS_RE = re.compile(r"\d+")
_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_KEY_VALUE_RE = re.compile(r"""^\s*["']?[\w.\-]+["']?\s*[:=]\s*\S""")
_TABLE_SEPARATORS = ("|", "\t")
# Base64 needs a long run to rule out identifiers and paths; hex needs 128+ so
# sha256 digests and commit ids, which agents do quote, are never touched.
_BASE64_RE = re.compile(r"(?:data:[\w/+.-]+;base64,)?[A-Za-z0-9+/]{160,}={0,2}")
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{128,}\b")


def collapse_repeated_lines(text: str) -> str | None:
    """Drop later copies of digit-normalized repeated lines, or return None."""
    if not text or len(text) > _MAX_INPUT_CHARS:
        return None
    lines = text.split("\n")
    keys = [_DIGITS_RE.sub("#", line.strip()) for line in lines]
    counts = Counter(key for key, line in zip(keys, lines, strict=True) if _is_log_like(line))
    repeated = {key for key, count in counts.items() if count >= _MIN_REPEATS}
    if not repeated:
        return None

    output: list[str] = []
    first_index: dict[str, int] = {}
    for key, line in zip(keys, lines, strict=True):
        if key not in repeated:
            output.append(line)
        elif key not in first_index:
            first_index[key] = len(output)
            output.append(line)
    for key, index in first_index.items():
        output[index] = (
            f"{output[index]}  [rclm: +{counts[key] - 1} similar lines differing only in numbers]"
        )
    collapsed = "\n".join(output)
    return collapsed if len(text) - len(collapsed) >= _MIN_SAVED_CHARS else None


def _is_log_like(line: str) -> bool:
    """True for prose-like lines whose numbers are incidental (attempts, timings)."""
    stripped = line.strip()
    if len(stripped) < _MIN_REPEATED_LINE_CHARS or _KEY_VALUE_RE.match(stripped):
        return False
    if any(separator in stripped for separator in _TABLE_SEPARATORS) or stripped.count(",") >= 2:
        return False
    compact = "".join(stripped.split())
    digit_share = sum(ch.isdigit() for ch in compact) / max(1, len(compact))
    return digit_share <= _MAX_DIGIT_SHARE and len(_WORD_RE.findall(stripped)) >= _MIN_WORDS


def replace_blobs(text: str) -> str | None:
    """Replace long base64/hex runs with a size-and-digest placeholder, or None."""
    if not text or len(text) > _MAX_INPUT_CHARS:
        return None

    def placeholder(kind: str, match: re.Match[str]) -> str:
        blob = match.group(0)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
        return f"[rclm: {kind} blob, {len(blob):,} chars, sha256:{digest}]"

    replaced = _BASE64_RE.sub(lambda m: placeholder("base64", m), text)
    replaced = _HEX_RE.sub(lambda m: placeholder("hex", m), replaced)
    return replaced if len(text) - len(replaced) >= _MIN_SAVED_CHARS else None


def minify(text: str) -> list[tuple[str, str, str]]:
    """Apply both folds in order; return (mechanism, before, after) per applied step."""
    steps: list[tuple[str, str, str]] = []
    current = text
    for mechanism, fold in (
        (BLOB_MECHANISM, replace_blobs),
        (REPEATED_LINE_MECHANISM, collapse_repeated_lines),
    ):
        folded = fold(current)
        if folded is not None:
            steps.append((mechanism, current, folded))
            current = folded
    return steps
