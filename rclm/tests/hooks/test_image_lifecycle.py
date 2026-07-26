"""Tests for rclm.hooks.image_lifecycle.

Shapes exercised here (Claude Code native Read-of-image, MCP ImageContent,
MCP CallToolResult with a structuredContent mirror) were all confirmed against
real captured payloads — see image_lifecycle.py's module/function docstrings.
"""

from __future__ import annotations

import base64
import io

import PIL.Image
import pytest

from rclm.hooks import image_lifecycle
from rclm.hooks.image_lifecycle import (
    find_image,
    maybe_downscale_image_result,
    peek_image_dimensions,
)


def _b64_image(width: int, height: int, *, mode: str = "RGB", fmt: str = "PNG") -> str:
    img = PIL.Image.new(mode, (width, height), (255, 0, 0, 128) if mode == "RGBA" else 0)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _claude_read_shape(b64: str, media_type: str = "image/png") -> dict:
    return {
        "type": "image",
        "file": {
            "base64": b64,
            "type": media_type,
            "originalSize": len(b64),
            "dimensions": {
                "originalWidth": 100,
                "originalHeight": 100,
                "displayWidth": 100,
                "displayHeight": 100,
            },
        },
    }


def _mcp_image_block(b64: str, media_type: str = "image/png") -> dict:
    return {"type": "image", "data": b64, "mimeType": media_type}


def _mcp_call_tool_result(b64: str, *, with_structured_mirror: bool = True) -> dict:
    result = {
        "content": [_mcp_image_block(b64)],
        "isError": False,
    }
    if with_structured_mirror:
        result["structuredContent"] = _mcp_image_block(b64)
    return result


# ---------------------------------------------------------------------------
# find_image
# ---------------------------------------------------------------------------


class TestFindImage:
    def test_claude_read_shape(self):
        b64 = _b64_image(10, 10)
        ref = find_image(_claude_read_shape(b64))
        assert ref is not None
        assert ref.base64_data == b64

    def test_bare_mcp_image_block(self):
        b64 = _b64_image(10, 10)
        ref = find_image(_mcp_image_block(b64))
        assert ref is not None
        assert ref.base64_data == b64

    def test_mcp_call_tool_result_with_structured_mirror(self):
        b64 = _b64_image(10, 10)
        ref = find_image(_mcp_call_tool_result(b64, with_structured_mirror=True))
        assert ref is not None
        assert ref.base64_data == b64

    def test_mcp_call_tool_result_without_structured_mirror(self):
        b64 = _b64_image(10, 10)
        ref = find_image(_mcp_call_tool_result(b64, with_structured_mirror=False))
        assert ref is not None
        assert ref.base64_data == b64

    def test_bare_list_of_content_blocks(self):
        b64 = _b64_image(10, 10)
        ref = find_image([{"type": "text", "text": "hi"}, _mcp_image_block(b64)])
        assert ref is not None
        assert ref.base64_data == b64

    @pytest.mark.parametrize(
        "tool_response",
        [
            None,
            "plain string result",
            {},
            {"type": "image"},  # missing file/data entirely
            {"type": "image", "file": {}},  # missing base64
            {"type": "image", "data": ""},  # empty data
            {"content": "not a list"},
            {"content": [{"type": "text", "text": "no image here"}]},
            [1, 2, 3],
            {"stdout": "regular bash output", "stderr": ""},
        ],
    )
    def test_unrecognized_shapes_return_none(self, tool_response):
        assert find_image(tool_response) is None


# ---------------------------------------------------------------------------
# peek_image_dimensions
# ---------------------------------------------------------------------------


class TestPeekImageDimensions:
    def test_valid_image(self):
        b64 = _b64_image(123, 45)
        ref = find_image(_mcp_image_block(b64))
        assert peek_image_dimensions(ref) == (123, 45)

    def test_corrupt_data_returns_none(self):
        ref = find_image(_mcp_image_block("not-valid-base64-image-data!!"))
        assert peek_image_dimensions(ref) is None

    def test_oversized_pixel_count_returns_none(self, monkeypatch):
        monkeypatch.setattr(image_lifecycle, "MAX_DECODE_PIXELS", 50)
        b64 = _b64_image(100, 100)
        ref = find_image(_mcp_image_block(b64))
        assert peek_image_dimensions(ref) is None


# ---------------------------------------------------------------------------
# maybe_downscale_image_result
# ---------------------------------------------------------------------------


class TestMaybeDownscaleImageResult:
    def test_under_min_size_bytes_returns_none(self):
        b64 = _b64_image(50, 50)
        result = maybe_downscale_image_result(_mcp_image_block(b64), min_size_bytes=10_000_000)
        assert result is None

    def test_already_within_bounds_no_alpha_returns_none(self):
        # Small enough to pass the size-bytes gate at a low threshold, but
        # already within max_dim and no alpha channel -> nothing to do.
        b64 = _b64_image(500, 500)
        result = maybe_downscale_image_result(_mcp_image_block(b64), min_size_bytes=1, max_dim=1280)
        assert result is None

    def test_oversized_rgb_downscales_to_jpeg(self):
        b64 = _b64_image(2000, 2000)
        result = maybe_downscale_image_result(_mcp_image_block(b64), min_size_bytes=1, max_dim=100)
        assert result is not None
        assert result.tokens_saved_estimate > 0
        assert result.compression_ratio < 1.0
        new_block = result.new_tool_response
        assert new_block["mimeType"] == "image/jpeg"
        img = PIL.Image.open(io.BytesIO(base64.b64decode(new_block["data"])))
        assert img.size == (100, 100)
        assert img.format == "JPEG"

    def test_alpha_channel_keeps_png(self):
        b64 = _b64_image(2000, 2000, mode="RGBA")
        result = maybe_downscale_image_result(_mcp_image_block(b64), min_size_bytes=1, max_dim=100)
        assert result is not None
        new_block = result.new_tool_response
        assert new_block["mimeType"] == "image/png"
        img = PIL.Image.open(io.BytesIO(base64.b64decode(new_block["data"])))
        assert img.format == "PNG"

    def test_never_upscales(self):
        b64 = _b64_image(50, 50)
        result = maybe_downscale_image_result(_mcp_image_block(b64), min_size_bytes=1, max_dim=5000)
        # Already smaller than max_dim -> no resize should occur (None: no benefit).
        assert result is None

    def test_structured_content_mirror_rewritten_in_lockstep(self):
        b64 = _b64_image(2000, 2000)
        tool_response = _mcp_call_tool_result(b64, with_structured_mirror=True)
        result = maybe_downscale_image_result(tool_response, min_size_bytes=1, max_dim=100)
        assert result is not None
        new_tr = result.new_tool_response
        assert new_tr["content"][0]["data"] != b64
        assert new_tr["structuredContent"]["data"] != b64
        assert new_tr["content"][0]["data"] == new_tr["structuredContent"]["data"]

    def test_corrupt_image_payload_returns_none_never_raises(self):
        result = maybe_downscale_image_result(
            _mcp_image_block("this-is-not-a-real-image-!!!" * 20_000), min_size_bytes=1
        )
        assert result is None

    def test_non_image_result_returns_none(self):
        assert maybe_downscale_image_result("regular bash stdout output") is None
        assert maybe_downscale_image_result({"stdout": "ls output", "stderr": ""}) is None

    def test_oversized_pixel_count_returns_none(self, monkeypatch):
        monkeypatch.setattr(image_lifecycle, "MAX_DECODE_PIXELS", 50)
        b64 = _b64_image(500, 500)
        result = maybe_downscale_image_result(_mcp_image_block(b64), min_size_bytes=1)
        assert result is None

    def test_never_raises_on_arbitrary_garbage(self):
        for garbage in [object(), 12345, {"content": [None, 1, "x"]}, {"file": None}]:
            assert maybe_downscale_image_result(garbage) is None
