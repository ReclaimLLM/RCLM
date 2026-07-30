from __future__ import annotations

import hashlib
import os
import stat

import pytest

from rclm._session_transfer import (
    SessionTransferTooLarge,
    cleanup_expired_transfers,
    write_transfer_stream,
)


async def _chunks(*values: bytes):
    for value in values:
        yield value


@pytest.mark.asyncio
async def test_write_transfer_stream_is_lossless_and_owner_only(tmp_path) -> None:
    content = b'{"schema_version":"v1","payload":{"messages":["exact"]}}'

    artifact = await write_transfer_stream(
        _chunks(content[:13], content[13:31], content[31:]),
        root=tmp_path,
        max_bytes=1024,
    )

    assert artifact.path.read_bytes() == content
    assert artifact.byte_size == len(content)
    assert artifact.sha256 == hashlib.sha256(content).hexdigest()
    assert stat.S_IMODE(artifact.path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.asyncio
async def test_oversized_transfer_removes_partial_file(tmp_path) -> None:
    with pytest.raises(SessionTransferTooLarge):
        await write_transfer_stream(
            _chunks(b"1234", b"5678"),
            root=tmp_path,
            max_bytes=5,
        )

    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_interrupted_transfer_removes_partial_file(tmp_path) -> None:
    async def interrupted_chunks():
        yield b"partial"
        raise RuntimeError("connection dropped")

    with pytest.raises(RuntimeError, match="connection dropped"):
        await write_transfer_stream(
            interrupted_chunks(),
            root=tmp_path,
            max_bytes=1024,
        )

    assert not list(tmp_path.iterdir())


def test_cleanup_removes_only_expired_transfer_files(tmp_path) -> None:
    expired = tmp_path / "transfer-expired.json"
    fresh = tmp_path / "transfer-fresh.json"
    unrelated = tmp_path / "keep-me.txt"
    for path in (expired, fresh, unrelated):
        path.write_text("data", encoding="utf-8")
    os.utime(expired, (100, 100))
    os.utime(fresh, (950, 950))
    os.utime(unrelated, (100, 100))

    removed = cleanup_expired_transfers(
        tmp_path,
        now=1000,
        ttl_seconds=100,
        max_entries=100,
    )

    assert removed == 1
    assert not expired.exists()
    assert fresh.exists()
    assert unrelated.exists()


def test_cleanup_respects_scan_bound(tmp_path) -> None:
    for index in range(3):
        path = tmp_path / f"transfer-{index}.json"
        path.write_text("data", encoding="utf-8")
        os.utime(path, (100, 100))

    removed = cleanup_expired_transfers(
        tmp_path,
        now=1000,
        ttl_seconds=100,
        max_entries=1,
    )

    assert removed <= 1
    assert len(list(tmp_path.iterdir())) >= 2
