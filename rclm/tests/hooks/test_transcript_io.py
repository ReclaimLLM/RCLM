from __future__ import annotations

import logging

from rclm.hooks import transcript_io


def test_read_jsonl_rejects_symlink(tmp_path):
    target = tmp_path / "target.jsonl"
    target.write_text('{"role":"user"}\n')
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)

    records, warnings = transcript_io.read_jsonl(str(link), logger=logging.getLogger(__name__))

    assert records == []
    assert warnings == ["transcript_unavailable"]


def test_read_jsonl_bounds_line_count(tmp_path, monkeypatch):
    path = tmp_path / "transcript.jsonl"
    path.write_text('{"n":1}\n{"n":2}\n')
    monkeypatch.setattr(transcript_io, "MAX_TRANSCRIPT_LINES", 1)

    records, warnings = transcript_io.read_jsonl(str(path), logger=logging.getLogger(__name__))

    assert records == [{"n": 1}]
    assert warnings == ["transcript_line_limit_reached"]
