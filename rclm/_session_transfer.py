from __future__ import annotations

import hashlib
import os
import tempfile
import time
import uuid
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_BYTES = 100 * 1024 * 1024
DEFAULT_TTL_SECONDS = 60 * 60
DEFAULT_CLEANUP_LIMIT = 100
_FILE_PREFIX = "transfer-"


class SessionTransferTooLarge(ValueError):
    pass


@dataclass(frozen=True)
class LocalTransferArtifact:
    path: Path
    byte_size: int
    sha256: str


def default_transfer_root() -> Path:
    return Path(tempfile.gettempdir()) / "reclaimllm-transfers"


def _configured_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be greater than zero")
    return value


def configured_max_bytes() -> int:
    return _configured_positive_int("SESSION_TRANSFER_MAX_BYTES", DEFAULT_MAX_BYTES)


def configured_ttl_seconds() -> int:
    return _configured_positive_int("SESSION_TRANSFER_TTL_SECONDS", DEFAULT_TTL_SECONDS)


def _ensure_transfer_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Unsafe session transfer directory: {root}")
    os.chmod(root, 0o700)


def cleanup_expired_transfers(
    root: Path,
    *,
    now: float | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    max_entries: int = DEFAULT_CLEANUP_LIMIT,
) -> int:
    if not root.exists():
        return 0
    current_time = time.time() if now is None else now
    removed = 0
    inspected = 0
    for path in root.iterdir():
        if inspected >= max_entries:
            break
        inspected += 1
        if not path.is_file() or not path.name.startswith(_FILE_PREFIX):
            continue
        try:
            if current_time - path.stat().st_mtime >= ttl_seconds:
                path.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


async def write_transfer_stream(
    chunks: AsyncIterable[bytes],
    *,
    root: Path | None = None,
    max_bytes: int | None = None,
) -> LocalTransferArtifact:
    transfer_root = root or default_transfer_root()
    limit = configured_max_bytes() if max_bytes is None else max_bytes
    if limit < 1:
        raise ValueError("max_bytes must be greater than zero")

    _ensure_transfer_root(transfer_root)
    cleanup_expired_transfers(
        transfer_root,
        ttl_seconds=configured_ttl_seconds(),
    )

    transfer_id = uuid.uuid4().hex
    partial_path = transfer_root / f"{_FILE_PREFIX}{transfer_id}.partial"
    final_path = transfer_root / f"{_FILE_PREFIX}{transfer_id}.json"
    digest = hashlib.sha256()
    total = 0

    fd = os.open(partial_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            async for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError("Session transfer stream yielded a non-bytes chunk")
                total += len(chunk)
                if total > limit:
                    raise SessionTransferTooLarge(
                        f"Session transfer exceeds the {limit}-byte local limit"
                    )
                handle.write(chunk)
                digest.update(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial_path, final_path)
        os.chmod(final_path, 0o600)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise

    return LocalTransferArtifact(
        path=final_path,
        byte_size=total,
        sha256=digest.hexdigest(),
    )


def delete_transfer_artifact(path: Path) -> None:
    path.unlink(missing_ok=True)
