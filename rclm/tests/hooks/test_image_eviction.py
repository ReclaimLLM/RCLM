"""Tests for rclm.hooks.image_eviction — shadow-only stale-image measurement."""

from __future__ import annotations

import base64
import io

import PIL.Image

from rclm.hooks._analytics import FALLBACK_IMAGE_TOKENS
from rclm.hooks.image_eviction import (
    _modeled_cache_write_cost,
    eviction_key,
    maybe_track_eviction,
)


def _b64_image(width: int, height: int) -> str:
    img = PIL.Image.new("RGB", (width, height), 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mcp_result(b64: str) -> dict:
    return {"content": [{"type": "image", "data": b64, "mimeType": "image/png"}], "isError": False}


# ---------------------------------------------------------------------------
# eviction_key
# ---------------------------------------------------------------------------


class TestEvictionKey:
    def test_same_inputs_produce_same_key(self):
        a = eviction_key(
            "browser_screenshot", {"url": "http://x", "viewport": {"width": 800, "height": 600}}
        )
        b = eviction_key(
            "browser_screenshot", {"url": "http://x", "viewport": {"width": 800, "height": 600}}
        )
        assert a == b

    def test_different_url_produces_different_key(self):
        a = eviction_key("browser_screenshot", {"url": "http://x"})
        b = eviction_key("browser_screenshot", {"url": "http://y"})
        assert a != b

    def test_different_viewport_produces_different_key(self):
        a = eviction_key(
            "browser_screenshot", {"url": "http://x", "viewport": {"width": 800, "height": 600}}
        )
        b = eviction_key(
            "browser_screenshot", {"url": "http://x", "viewport": {"width": 1024, "height": 768}}
        )
        assert a != b

    def test_claude_read_uses_file_path(self):
        a = eviction_key("Read", {"file_path": "/tmp/screenshot.png"})
        b = eviction_key("Read", {"file_path": "/tmp/other.png"})
        assert a != b

    def test_missing_page_and_viewport_args_still_produces_stable_key(self):
        a = eviction_key("mcp__x__shot", {})
        b = eviction_key("mcp__x__shot", {})
        assert a == b  # doesn't raise, doesn't crash, stable


# ---------------------------------------------------------------------------
# _modeled_cache_write_cost
# ---------------------------------------------------------------------------


class TestModeledCacheWriteCost:
    def test_zero_in_zero_out(self):
        assert _modeled_cache_write_cost(0) == 0

    def test_scales_by_multiplier(self):
        assert _modeled_cache_write_cost(1000) == 1250

    def test_never_negative(self):
        assert _modeled_cache_write_cost(-100) == 0


# ---------------------------------------------------------------------------
# maybe_track_eviction
# ---------------------------------------------------------------------------


class TestMaybeTrackEviction:
    def test_first_sighting_returns_no_measurement_but_seeds_state(self):
        tr = _mcp_result(_b64_image(800, 600))
        measurement, state = maybe_track_eviction(
            tr, {}, tool_name="shot", tool_input={"url": "http://x"}, turn=1
        )
        assert measurement is None
        assert len(state) == 1

    def test_second_image_at_same_key_returns_measurement(self):
        state = {}
        _, state = maybe_track_eviction(
            _mcp_result(_b64_image(800, 600)),
            state,
            tool_name="shot",
            tool_input={"url": "http://x"},
            turn=1,
        )
        measurement, state = maybe_track_eviction(
            _mcp_result(_b64_image(400, 300)),
            state,
            tool_name="shot",
            tool_input={"url": "http://x"},
            turn=2,
        )
        assert measurement is not None
        assert measurement["would_save_tokens"] > 0
        assert measurement["modeled_cost_tokens"] == _modeled_cache_write_cost(
            measurement["would_save_tokens"]
        )
        assert measurement["net_tokens"] == max(
            0, measurement["would_save_tokens"] - measurement["modeled_cost_tokens"]
        )
        assert measurement["superseded_turn"] == 1

    def test_different_key_on_second_call_tracks_independently(self):
        state = {}
        _, state = maybe_track_eviction(
            _mcp_result(_b64_image(800, 600)),
            state,
            tool_name="shot",
            tool_input={"url": "http://x"},
            turn=1,
        )
        measurement, state = maybe_track_eviction(
            _mcp_result(_b64_image(400, 300)),
            state,
            tool_name="shot",
            tool_input={"url": "http://y"},
            turn=2,
        )
        assert measurement is None
        assert len(state) == 2

    def test_identical_image_seen_again_is_not_a_supersession(self):
        b64 = _b64_image(800, 600)
        state = {}
        _, state = maybe_track_eviction(
            _mcp_result(b64), state, tool_name="shot", tool_input={"url": "http://x"}, turn=1
        )
        measurement, state = maybe_track_eviction(
            _mcp_result(b64), state, tool_name="shot", tool_input={"url": "http://x"}, turn=2
        )
        assert measurement is None

    def test_lru_eviction_bounded_by_max_entries(self):
        state = {}
        for i in range(5):
            _, state = maybe_track_eviction(
                _mcp_result(_b64_image(100, 100)),
                state,
                tool_name="shot",
                tool_input={"url": f"http://{i}"},
                turn=i,
                max_entries=2,
            )
        assert len(state) == 2

    def test_non_image_tool_response_returns_none_state_unchanged(self):
        state = {"existing": {"content_hash": "x", "token_estimate": 1, "turn": 0}}
        measurement, new_state = maybe_track_eviction(
            "plain bash output", state, tool_name="Bash", tool_input={}, turn=1
        )
        assert measurement is None
        assert new_state == state

    def test_corrupt_image_still_tracks_using_fallback_tokens(self):
        bad_tr = {
            "content": [{"type": "image", "data": "not-real-image-data", "mimeType": "image/png"}]
        }
        state = {}
        _, state = maybe_track_eviction(
            bad_tr, state, tool_name="shot", tool_input={"url": "http://x"}, turn=1
        )
        assert list(state.values())[0]["token_estimate"] == FALLBACK_IMAGE_TOKENS

        measurement, state = maybe_track_eviction(
            bad_tr, state, tool_name="shot", tool_input={"url": "http://x"}, turn=2
        )
        # Same (corrupt) data each time -> same content hash -> not a supersession.
        assert measurement is None

    def test_never_raises_on_arbitrary_garbage(self):
        for garbage in [object(), 12345, None, {"content": [None]}]:
            measurement, _ = maybe_track_eviction(garbage, {}, tool_name="x", tool_input={}, turn=1)
            assert measurement is None
