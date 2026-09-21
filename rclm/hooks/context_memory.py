"""Integration helpers for recallable artifacts and the typed task ledger."""

from __future__ import annotations

from rclm.hooks import session_store
from rclm.hooks.recallable_artifacts import (
    ArtifactProvenance,
    ArtifactReference,
    RecallableArtifactStore,
)
from rclm.hooks.task_ledger import LedgerWriteResult, TaskLedgerStore
from rclm.hooks.tool_result_transform import (
    TransformDecision,
    attach_recall_stub,
)
from rclm.hooks.tool_semantics import semantic_target


def _artifacts() -> RecallableArtifactStore:
    return RecallableArtifactStore(session_store._SESSIONS_DIR / "artifacts")


def _ledgers() -> TaskLedgerStore:
    return TaskLedgerStore(session_store._SESSIONS_DIR / "task-ledgers")


def store_result(
    session_id: str,
    text: str,
    *,
    provider: str,
    tool_name: str,
    tool_input: object,
    tool_use_id: str | None,
) -> ArtifactReference | None:
    """Persist one secret-safe result before replacing it in model context."""
    return _artifacts().store(
        session_id,
        text,
        ArtifactProvenance(
            provider=provider,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            target=semantic_target(tool_name, tool_input),
        ),
    )


def observe_tool(
    session_id: str,
    tool_name: str,
    tool_input: object,
    tool_response: object,
    *,
    tool_use_id: str | None,
    artifact_handle: str | None = None,
    cwd: str | None = None,
) -> LedgerWriteResult:
    """Update the bounded deterministic task ledger from one tool event."""
    return _ledgers().observe_tool(
        session_id,
        tool_name,
        tool_input,
        tool_response,
        tool_use_id=tool_use_id,
        artifact_handle=artifact_handle,
        cwd=cwd,
    )


def make_text_recallable(
    session_id: str,
    original_text: str,
    replacement: str,
    *,
    provider: str,
    tool_name: str,
    tool_input: object,
    tool_use_id: str | None,
) -> tuple[str, str | None]:
    """Persist an exact result and append its handle when that still saves tokens."""
    reference = store_result(
        session_id,
        original_text,
        provider=provider,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_use_id=tool_use_id,
    )
    if reference is None:
        return replacement, None
    recallable = replacement.rstrip("\n") + "\n" + reference.stub + "\n"
    if len(recallable) >= len(original_text):
        return replacement, None
    return recallable, reference.handle


def make_decision_recallable(
    session_id: str,
    source: object,
    decision: TransformDecision,
    *,
    provider: str,
    tool_name: str,
    tool_input: object,
    tool_use_id: str | None,
) -> tuple[TransformDecision, str | None]:
    """Persist and attach an exact-result reference to a structured transform."""
    reference = store_result(
        session_id,
        decision.original_text,
        provider=provider,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_use_id=tool_use_id,
    )
    if reference is None:
        return decision, None
    recalled = attach_recall_stub(source, decision, reference.stub, tool_name=tool_name)
    if reference.stub not in recalled.compressed_text:
        return decision, None
    return recalled, reference.handle


__all__ = [
    "make_decision_recallable",
    "make_text_recallable",
    "observe_tool",
    "store_result",
]
