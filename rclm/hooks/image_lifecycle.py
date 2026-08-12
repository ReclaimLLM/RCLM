"""Downscale base64 images embedded in tool results before they enter the transcript.

Shared across provider handlers (Claude, Codex, ...) — see PRD_project_cost_analytics.md
for the H1-H4 mechanism vocabulary this extends. Every public function here is guaranteed
to never raise: any decode failure, unrecognized format, or internal error results in
`None` (or the original text is deemed unmodifiable elsewhere in this module), meaning
"pass the original tool result through unmodified."
"""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Callable
from dataclasses import dataclass

from rclm.hooks._analytics import estimate_image_tokens

logger = logging.getLogger(__name__)

MECHANISM = "image_downscale"

DEFAULT_MAX_DIM = 1280
DEFAULT_JPEG_QUALITY = 70
DEFAULT_MIN_SIZE_BYTES = 100_000

# PIL decode/resize/encode time scales with pixel count. Rejecting oversized
# images before decode bounds worst-case latency on this hot path without a
# wall-clock timer, which would need a helper thread to interrupt safely and
# wouldn't behave the same on the Windows/PowerShell paths this codebase
# already supports for shell read-cache (see read_cache.py).
MAX_DECODE_PIXELS = 40_000_000  # ~40 megapixels, e.g. a 6000x6600 full-page capture


@dataclass
class ImageDownscaleResult:
    new_tool_response: object
    raw_token_estimate: int
    compressed_token_estimate: int
    tokens_saved_estimate: int
    compression_ratio: float


@dataclass
class _ImageRef:
    """Points at one base64 image inside a tool_response, with enough context
    to write a resized replacement back into the original structure."""

    base64_data: str
    rebuild: Callable[[str, str, int, tuple[int, int]], object]


def _find_mcp_image_block(block: object) -> _ImageRef | None:
    """MCP protocol's ImageContent shape: {"type": "image", "data": <base64>, "mimeType": "image/png"}.

    Confirmed empirically against a real MCP tool call (Codex CLI 0.145.0,
    `mcp.server.fastmcp` + `mcp.types.ImageContent`): this is what MCP servers
    actually put on the wire, not the Anthropic Messages API's nested
    `{"source": {"type": "base64", "data": ...}}` shape — the two are easy to
    conflate but are genuinely different schemas.
    """
    if not isinstance(block, dict) or block.get("type") != "image":
        return None
    data = block.get("data")
    media_type = block.get("mimeType")
    if not isinstance(data, str) or not data or not isinstance(media_type, str):
        return None

    def _rebuild(
        new_b64: str, new_media_type: str, _size_bytes: int, _dims: tuple[int, int], *, _block=block
    ) -> dict:
        new_block = dict(_block)
        new_block["data"] = new_b64
        new_block["mimeType"] = new_media_type
        return new_block

    return _ImageRef(base64_data=data, rebuild=_rebuild)


def _find_anthropic_image_block(block: object) -> _ImageRef | None:
    """Anthropic content block: ``{type: image, source: {type: base64, ...}}``.

    Claude transcripts store native image results in this Messages API shape,
    commonly as the sole item in a top-level content list. Keep it distinct
    from MCP's flat ``data``/``mimeType`` block so each wire envelope is
    rebuilt without renaming provider fields.
    """
    if not isinstance(block, dict) or block.get("type") != "image":
        return None
    source = block.get("source")
    if not isinstance(source, dict) or source.get("type") != "base64":
        return None
    data = source.get("data")
    media_type = source.get("media_type")
    if not isinstance(data, str) or not data or not isinstance(media_type, str):
        return None

    def _rebuild(
        new_b64: str,
        new_media_type: str,
        _size_bytes: int,
        _dims: tuple[int, int],
        *,
        _block=block,
        _source=source,
    ) -> dict:
        new_source = dict(_source)
        new_source["data"] = new_b64
        new_source["media_type"] = new_media_type
        new_block = dict(_block)
        new_block["source"] = new_source
        return new_block

    return _ImageRef(base64_data=data, rebuild=_rebuild)


def _find_claude_read_image(tool_response: object) -> _ImageRef | None:
    """Claude Code's native `Read`-of-an-image-file shape:
    {"type": "image", "file": {"base64": ..., "type": "image/png", "dimensions": {...}}}.
    """
    if not isinstance(tool_response, dict) or tool_response.get("type") != "image":
        return None
    file_block = tool_response.get("file")
    if not isinstance(file_block, dict):
        return None
    data = file_block.get("base64")
    if not isinstance(data, str) or not data:
        return None

    def _rebuild(
        new_b64: str,
        media_type: str,
        size_bytes: int,
        dims: tuple[int, int],
        *,
        _tool_response=tool_response,
        _file_block=file_block,
    ) -> dict:
        new_file = dict(_file_block)
        new_file["base64"] = new_b64
        new_file["type"] = media_type
        new_file["originalSize"] = size_bytes
        old_dims = _file_block.get("dimensions")
        new_file["dimensions"] = {
            **(old_dims if isinstance(old_dims, dict) else {}),
            "displayWidth": dims[0],
            "displayHeight": dims[1],
        }
        new_tool_response = dict(_tool_response)
        new_tool_response["file"] = new_file
        return new_tool_response

    return _ImageRef(base64_data=data, rebuild=_rebuild)


def _find_in_list(items: list) -> _ImageRef | None:
    for index, item in enumerate(items):
        inner = _find_mcp_image_block(item) or _find_anthropic_image_block(item)
        if inner is None:
            continue

        def _rebuild_in_list(
            new_b64: str,
            media_type: str,
            size_bytes: int,
            dims: tuple[int, int],
            *,
            _items=items,
            _index=index,
            _inner=inner,
        ) -> list:
            new_items = list(_items)
            new_items[_index] = _inner.rebuild(new_b64, media_type, size_bytes, dims)
            return new_items

        return _ImageRef(base64_data=inner.base64_data, rebuild=_rebuild_in_list)
    return None


def _find_mcp_call_tool_result(tool_response: object) -> _ImageRef | None:
    """Standard MCP `CallToolResult` shape: {"content": [...], "structuredContent": {...}, "isError": ...}.

    Codex (and, since this is the same MCP wire protocol, presumably any MCP
    client) surfaces a captured screenshot/image tool result with the image
    duplicated in two places: once inside the `content` list (what a
    protocol-level client renders) and again, flattened, under
    `structuredContent` (confirmed empirically against Codex CLI 0.145.0).
    Both must be rewritten together — patching only `content` would leave a
    full-size duplicate sitting in `structuredContent`, undoing the savings.
    """
    if not isinstance(tool_response, dict):
        return None
    content = tool_response.get("content")
    if not isinstance(content, list):
        return None
    inner = _find_in_list(content)
    if inner is None:
        return None

    structured = tool_response.get("structuredContent")
    structured_mirror = _find_mcp_image_block(structured) if isinstance(structured, dict) else None

    def _rebuild(
        new_b64: str,
        media_type: str,
        size_bytes: int,
        dims: tuple[int, int],
        *,
        _tool_response=tool_response,
        _inner=inner,
        _structured_mirror=structured_mirror,
    ) -> dict:
        new_tool_response = dict(_tool_response)
        new_tool_response["content"] = _inner.rebuild(new_b64, media_type, size_bytes, dims)
        if _structured_mirror is not None:
            new_tool_response["structuredContent"] = _structured_mirror.rebuild(
                new_b64, media_type, size_bytes, dims
            )
        return new_tool_response

    return _ImageRef(base64_data=inner.base64_data, rebuild=_rebuild)


def find_image(tool_response: object) -> _ImageRef | None:
    """Locate the first recognized base64 image in a tool_response, if any.

    Recognizes Claude Code's native Read-of-image shape, a bare MCP
    ImageContent block, and the standard MCP CallToolResult shape (a
    `content` list, optionally mirrored in `structuredContent`). Anything
    else is unrecognized and returns None, which callers treat as "pass
    through unmodified."
    """
    ref = _find_claude_read_image(tool_response)
    if ref is not None:
        return ref

    ref = _find_mcp_call_tool_result(tool_response)
    if ref is not None:
        return ref

    ref = _find_mcp_image_block(tool_response)
    if ref is not None:
        return ref

    ref = _find_anthropic_image_block(tool_response)
    if ref is not None:
        return ref

    if isinstance(tool_response, list):
        return _find_in_list(tool_response)

    return None


def peek_image_dimensions(ref: _ImageRef) -> tuple[int, int] | None:
    """Return (width, height) for a recognized image ref without a full raster decode.

    For callers that only need a token estimate (e.g. image_eviction.py's
    supersession tracking), not a resized copy. `PIL.Image.open()` reads only
    the header before `.size` is accessed, so this is cheap even for large
    images — bounded by the same MAX_DECODE_PIXELS ceiling as the full
    downscale path. Never raises: any decode failure, oversized image, or
    missing Pillow returns None.
    """
    try:
        import PIL.Image

        raw_bytes = base64.b64decode(ref.base64_data, validate=False)
        with PIL.Image.open(io.BytesIO(raw_bytes)) as img:
            width, height = img.size
            if width * height > MAX_DECODE_PIXELS:
                return None
            return width, height
    except Exception:
        return None


def maybe_downscale_image_result(
    tool_response: object,
    *,
    max_dim: int = DEFAULT_MAX_DIM,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    min_size_bytes: int = DEFAULT_MIN_SIZE_BYTES,
    provider: str = "anthropic",
) -> ImageDownscaleResult | None:
    """Downscale the base64 image embedded in a tool result, if there is one worth shrinking.

    Returns None if there's no recognized image, it's already under
    `min_size_bytes`, decoding/re-encoding fails, or the re-encode wouldn't
    actually be smaller — every one of those is "pass the original through
    unmodified," never an error surfaced to the caller.
    """
    try:
        ref = find_image(tool_response)
        if ref is None:
            return None

        # Cheap check on encoded size before touching the decoder at all —
        # base64 is ~4/3 the size of the decoded bytes.
        if len(ref.base64_data) * 3 // 4 < min_size_bytes:
            return None

        import PIL.Image

        raw_bytes = base64.b64decode(ref.base64_data, validate=False)

        with PIL.Image.open(io.BytesIO(raw_bytes)) as img:
            width, height = img.size  # header read only; no raster decode yet
            if width * height > MAX_DECODE_PIXELS:
                logger.warning(
                    "image exceeds %d px, skipping downscale (got %dx%d)",
                    MAX_DECODE_PIXELS,
                    width,
                    height,
                )
                return None

            img.load()  # decode now that size is bounded

            has_alpha = img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            )
            scale = min(1.0, max_dim / max(width, height))  # never upscale
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))

            if new_size == (width, height) and not has_alpha:
                return None  # already within bounds, no format change needed

            resized = (
                img.resize(new_size, PIL.Image.Resampling.LANCZOS)
                if new_size != (width, height)
                else img
            )

            buf = io.BytesIO()
            if has_alpha:
                resized.save(buf, format="PNG", optimize=True)
                new_media_type = "image/png"
            else:
                resized.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
                new_media_type = "image/jpeg"

        new_bytes = buf.getvalue()
        if len(new_bytes) >= len(raw_bytes):
            return None  # re-encode didn't help; not worth the format/content change

        raw_tokens = estimate_image_tokens(width, height, provider=provider)
        compressed_tokens = estimate_image_tokens(new_size[0], new_size[1], provider=provider)
        new_b64 = base64.b64encode(new_bytes).decode("ascii")

        return ImageDownscaleResult(
            new_tool_response=ref.rebuild(new_b64, new_media_type, len(new_bytes), new_size),
            raw_token_estimate=raw_tokens,
            compressed_token_estimate=compressed_tokens,
            tokens_saved_estimate=max(0, raw_tokens - compressed_tokens),
            compression_ratio=len(new_bytes) / max(1, len(raw_bytes)),
        )
    except Exception:
        logger.exception("image downscale failed; passing through original tool result")
        return None
