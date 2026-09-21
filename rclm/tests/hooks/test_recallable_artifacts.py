import json

from rclm.hooks.recallable_artifacts import (
    ArtifactLimits,
    ArtifactProvenance,
    RecallableArtifactStore,
)


def _store(tmp_path, **limits):
    return RecallableArtifactStore(
        tmp_path / "artifacts",
        limits=ArtifactLimits(**limits) if limits else None,
    )


def test_content_addressed_handle_and_stub_include_bounded_provenance(tmp_path):
    store = _store(tmp_path)
    provenance = ArtifactProvenance(
        provider="gemini",
        tool_name="view_file",
        tool_use_id="call-7",
        target="src/main.py",
    )

    first = store.store("session-a", "alpha\nbeta\n", provenance)
    second = store.store("session-a", "alpha\nbeta\n", provenance)

    assert first is not None
    assert second is not None
    assert first.handle == second.handle
    assert first.stub == second.stub
    assert "provider=gemini" in first.stub
    assert "tool=view_file" in first.stub
    assert "target=src/main.py" in first.stub
    assert "recall=chars[0:11)" in first.stub
    assert len(store.manifest("session-a")["artifacts"]) == 1


def test_recall_is_exact_for_full_text_and_unicode_character_ranges(tmp_path):
    store = _store(tmp_path)
    reference = store.store(
        "session-a",
        "a🙂bcé",
        ArtifactProvenance(provider="claude", tool_name="Read"),
    )
    assert reference is not None

    full = store.recall("session-a", reference.handle)
    partial = store.recall("session-a", reference.handle, start=1, end=4)

    assert full is not None and full.text == "a🙂bcé" and full.complete
    assert partial is not None and partial.text == "🙂bc"
    assert (partial.start, partial.end, partial.total_chars) == (1, 4, 5)
    assert store.recall("session-a", reference.handle, start=0, end=6) is None


def test_handle_recall_is_session_scoped(tmp_path):
    store = _store(tmp_path)
    reference = store.store(
        "session-a",
        "private result",
        ArtifactProvenance(provider="codex", tool_name="exec_command"),
    )
    assert reference is not None
    assert store.recall("session-b", reference.handle) is None


def test_count_and_byte_limits_evict_oldest_artifacts(tmp_path):
    store = _store(
        tmp_path,
        max_artifacts=2,
        max_session_bytes=8,
        max_artifact_bytes=8,
    )
    provenance = ArtifactProvenance(provider="test", tool_name="read")
    first = store.store("session", "1111", provenance)
    second = store.store("session", "2222", provenance)
    third = store.store("session", "3333", provenance)

    assert first is not None and second is not None and third is not None
    assert store.recall("session", first.handle) is None
    assert store.recall("session", second.handle).text == "2222"
    assert store.recall("session", third.handle).text == "3333"
    manifest = store.manifest("session")
    assert [item["handle"] for item in manifest["artifacts"]] == [second.handle, third.handle]


def test_oversized_or_failed_persistence_returns_none_and_preserves_old_manifest(
    tmp_path, monkeypatch
):
    store = _store(
        tmp_path,
        max_artifacts=2,
        max_session_bytes=20,
        max_artifact_bytes=10,
    )
    provenance = ArtifactProvenance(provider="test", tool_name="read")
    original = store.store("session", "original", provenance)
    assert original is not None
    assert store.store("session", "x" * 11, provenance) is None

    monkeypatch.setattr(store, "_write_manifest", lambda *_args: (_ for _ in ()).throw(OSError()))
    assert store.store("session", "new", provenance) is None
    assert store.recall("session", original.handle).text == "original"


def test_manifest_is_uploadable_and_cleanup_removes_session_state(tmp_path):
    store = _store(tmp_path)
    reference = store.store(
        "../provider-controlled",
        "result",
        ArtifactProvenance(provider="cursor", tool_name="mcp", target="resource://one"),
    )
    assert reference is not None
    manifest = store.manifest("../provider-controlled")

    assert manifest["version"] == 1
    assert manifest["session_id"] == "../provider-controlled"
    assert json.dumps(manifest)
    assert store._session_dir("../provider-controlled").parent == store.root

    store.cleanup("../provider-controlled")
    assert store.recall("../provider-controlled", reference.handle) is None
