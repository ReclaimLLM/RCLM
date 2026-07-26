"""Session-scoped stale-image eviction *measurement* — never mutates any tool result.

Tracks images per (tool, page/viewport) key in a dedicated LRU sidecar (see
session_store.read_image_eviction_state/write_image_eviction_state). When a newer
image supersedes an older tracked one, this computes what evicting the old one
WOULD save (its token footprint) and WOULD cost (a modeled prompt-cache
prefix-invalidation estimate) and returns a measurement for the caller to log as a
MechanismSaving event under "image_eviction" — always measurement_kind="estimated"
(the cost side is inherently modeled, never observed) and always applied=False,
unconditionally, regardless of the global shadow_mode flag: this module has no
code path that mutates a tool result, and callers must not wire it to do so.

The real per-call prompt-cache-invalidation cost isn't available at PostToolUse
time — cache_read_tokens/cache_creation_tokens are only assembled from the
transcript at Stop (see transcript.py), long after any individual image would
need to be evicted. So every measurement here uses the modeled formula; there is
no "use real data when available" branch to fall back from. See
PRD_project_cost_analytics.md for the broader mechanism vocabulary this extends.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

from rclm.hooks import image_lifecycle
from rclm.hooks._analytics import FALLBACK_IMAGE_TOKENS, estimate_image_tokens

MECHANISM = "image_eviction"

MAX_EVICTION_ENTRIES = 200  # LRU cap, same order of magnitude as dedupe.py's bound

# Deliberately provisional and narrow — NOT a general modeled-usage utility.
# Anthropic prices cache writes at roughly 1.25x the base input-token price
# (https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching). Expressed
# as a token-equivalent multiplier (this product's currency is tokens, not
# dollars — see PRD_project_cost_analytics.md §1), so it needs no pricing config.
# Delete/replace once PRD_project_cost_analytics.md's R1 general modeled-usage
# reconstruction ships and can supply a real, validated figure instead.
_MODELED_CACHE_WRITE_MULTIPLIER = 1.25

# Argument names commonly used by MCP screenshot/browser tools for the page
# identity. Unknown tools fall back to "" for this component (and for
# viewport, handled separately below), which still produces a stable (if
# coarser) key rather than skipping tracking.
_PAGE_ARG_NAMES = ("url", "page_url", "href", "target")


def _modeled_cache_write_cost(invalidated_prefix_tokens: int) -> int:
    """Token-equivalent modeled cost of re-establishing an invalidated cache prefix."""
    return max(0, round(invalidated_prefix_tokens * _MODELED_CACHE_WRITE_MULTIPLIER))


def _page_component(tool_input: dict) -> str:
    file_path = tool_input.get("file_path")
    if isinstance(file_path, str) and file_path:
        return file_path
    for name in _PAGE_ARG_NAMES:
        value = tool_input.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


def _viewport_component(tool_input: dict) -> str:
    viewport = tool_input.get("viewport")
    if isinstance(viewport, dict):
        width = viewport.get("width", "")
        height = viewport.get("height", "")
        return f"{width}x{height}"
    if isinstance(viewport, str) and viewport:
        return viewport
    width = tool_input.get("width")
    height = tool_input.get("height")
    if width or height:
        return f"{width}x{height}"
    return ""


def eviction_key(tool_name: str, tool_input: dict) -> str:
    """Stable string key for (tool, page/URL, viewport).

    Tools that don't expose a recognizable page/viewport argument still key on
    tool_name alone (page/viewport components are ""), which is conservative
    (may treat unrelated same-tool images as superseding each other) but never
    raises and never silently drops tracking.
    """
    page = _page_component(tool_input)
    viewport = _viewport_component(tool_input)
    return f"{tool_name}::{page}::{viewport}"


def maybe_track_eviction(
    tool_response: object,
    state: dict[str, dict],
    *,
    tool_name: str,
    tool_input: dict,
    turn: int,
    provider: str = "anthropic",
    max_entries: int = MAX_EVICTION_ENTRIES,
) -> tuple[dict | None, dict[str, dict]]:
    """Track one image sighting; return (measurement, updated_state). Never raises.

    `measurement` is None when: there's no recognized image, this is the first
    sighting of this (tool, page, viewport) key (nothing to evict yet — state is
    still seeded), or the image dimensions can't be determined (state is still
    seeded, using the fallback token constant, so later calls can still detect
    supersession — just without a precisely measured prior size).

    When not None, `measurement` has the shape:
      {
        "would_save_tokens": int,    # the superseded image's resident token estimate
        "modeled_cost_tokens": int,  # _modeled_cache_write_cost(would_save_tokens)
        "net_tokens": int,           # max(0, would_save_tokens - modeled_cost_tokens)
        "superseded_turn": int,
      }
    """
    try:
        ref = image_lifecycle.find_image(tool_response)
        if ref is None:
            return None, state

        key = eviction_key(tool_name, tool_input)
        dims = image_lifecycle.peek_image_dimensions(ref)
        new_tokens = (
            estimate_image_tokens(dims[0], dims[1], provider=provider)
            if dims is not None
            else FALLBACK_IMAGE_TOKENS
        )
        content_hash = hashlib.sha256(ref.base64_data.encode("ascii")).hexdigest()[:16]

        lru: OrderedDict[str, dict] = OrderedDict(state)
        prior = lru.get(key)

        lru[key] = {
            "content_hash": content_hash,
            "token_estimate": new_tokens,
            "turn": turn,
        }
        lru.move_to_end(key)
        while len(lru) > max_entries:
            lru.popitem(last=False)

        if prior is None or prior.get("content_hash") == content_hash:
            # First sighting, or the same image seen again — nothing superseded.
            return None, dict(lru)

        would_save = int(prior.get("token_estimate") or 0)
        modeled_cost = _modeled_cache_write_cost(would_save)
        measurement = {
            "would_save_tokens": would_save,
            "modeled_cost_tokens": modeled_cost,
            "net_tokens": max(0, would_save - modeled_cost),
            "superseded_turn": prior.get("turn"),
        }
        return measurement, dict(lru)
    except Exception:
        return None, state
