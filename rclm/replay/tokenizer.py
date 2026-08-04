"""Real-tokenizer counting for replay, matching Report 1's method.

Report 1 tokenized all text tool results with `o200k_base` and excluded
recognized image payloads from the text denominator (see
docs/whitepaper/report-1-token-savings-verification.md, line 27). Runtime
telemetry uses a `chars/4` heuristic (`rclm.hooks._analytics.estimate_tokens`)
instead; replay must not reuse that heuristic; the whole point of replay is
counting with a real tokenizer.
"""

from __future__ import annotations

import tiktoken

TOKENIZER_NAME = "o200k_base"

_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(TOKENIZER_NAME)
    return _encoding


def count_tokens(text: str | None) -> int:
    """Count tokens in `text` with the real tokenizer. Empty/None -> 0."""
    if not text:
        return 0
    return len(_get_encoding().encode(text, disallowed_special=()))
