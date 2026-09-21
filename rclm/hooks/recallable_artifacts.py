"""Session-scoped, recallable storage for evicted tool-result text.

The hook path deliberately performs local filesystem work only.  Provider
handlers may replace a large model-visible result with ``ArtifactReference.stub``
only when ``store`` succeeds; a ``None`` result means pass the original result
through unchanged.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = 1
HANDLE_PREFIX = "rclm://artifact/sha256/"
_HANDLE_RE = re.compile(r"^rclm://artifact/sha256/(?P<digest>[0-9a-f]{64})$")
_DEFAULT_ROOT = Path.home() / ".reclaimllm" / "sessions" / "artifacts"
_MAX_PROVENANCE_FIELD_CHARS = 160


@dataclass(frozen=True)
class ArtifactLimits:
    """Hard per-session limits for local recallable state."""

    max_artifacts: int = 32
    max_session_bytes: int = 32 * 1024 * 1024
    max_artifact_bytes: int = 8 * 1024 * 1024


@dataclass(frozen=True)
class ArtifactProvenance:
    """Origin metadata safe to repeat in a compact model-visible stub."""

    provider: str
    tool_name: str
    tool_use_id: str | None = None
    target: str | None = None

    def bounded(self) -> ArtifactProvenance:
        return ArtifactProvenance(
            provider=_compact_field(self.provider),
            tool_name=_compact_field(self.tool_name),
            tool_use_id=_compact_optional(self.tool_use_id),
            target=_compact_optional(self.target),
        )


@dataclass(frozen=True)
class ArtifactReference:
    handle: str
    digest: str
    chars: int
    bytes: int
    provenance: ArtifactProvenance
    stub: str


@dataclass(frozen=True)
class RecallResult:
    """An exact, end-exclusive character range recalled from one artifact."""

    handle: str
    text: str
    start: int
    end: int
    total_chars: int

    @property
    def complete(self) -> bool:
        return self.start == 0 and self.end == self.total_chars


class RecallableArtifactStore:
    """Persist immutable text artifacts without network access.

    Manifests are replaced atomically before stale artifact files are removed.
    If any persistence step fails, ``store`` returns ``None`` so callers can
    leave the original tool result model-visible.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        limits: ArtifactLimits | None = None,
    ) -> None:
        self.root = root or _DEFAULT_ROOT
        self.limits = limits or ArtifactLimits()

    def store(
        self,
        session_id: str,
        text: str,
        provenance: ArtifactProvenance,
    ) -> ArtifactReference | None:
        """Store text and return a compact recall reference, or fail open."""
        if not session_id or not isinstance(text, str) or not text:
            return None
        payload = text.encode("utf-8")
        if (
            self.limits.max_artifacts < 1
            or self.limits.max_session_bytes < 1
            or len(payload) > self.limits.max_artifact_bytes
            or len(payload) > self.limits.max_session_bytes
        ):
            return None

        digest = hashlib.sha256(payload).hexdigest()
        handle = f"{HANDLE_PREFIX}{digest}"
        bounded_provenance = provenance.bounded()
        directory = self._session_dir(session_id)
        destination = directory / f"{digest}.txt"
        created_artifact = False

        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            manifest = self._read_manifest(session_id)
            entries = manifest["artifacts"]
            existing = next((entry for entry in entries if entry.get("digest") == digest), None)

            if existing is None:
                self._atomic_write_bytes(destination, payload)
                created_artifact = True
                sequence = int(manifest.get("next_sequence", 1))
                entries.append(
                    {
                        "digest": digest,
                        "handle": handle,
                        "chars": len(text),
                        "bytes": len(payload),
                        "sequence": sequence,
                        "provenance": asdict(bounded_provenance),
                    }
                )
                manifest["next_sequence"] = sequence + 1
            elif not self._artifact_matches(destination, digest):
                # A content-addressed entry must never point at altered bytes.
                return None

            retained, removed = self._bounded_entries(entries)
            if not any(entry.get("digest") == digest for entry in retained):
                if created_artifact:
                    with contextlib.suppress(OSError):
                        destination.unlink()
                return None

            manifest["artifacts"] = retained
            self._write_manifest(session_id, manifest)

            # Remove old content only after the new manifest is durable.  This
            # preserves every reference from the old manifest if replacement
            # fails and leaves at worst an unreferenced new file.
            for entry in removed:
                stale_digest = entry.get("digest")
                if isinstance(stale_digest, str) and stale_digest != digest:
                    with contextlib.suppress(OSError):
                        (directory / f"{stale_digest}.txt").unlink()
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

        stub = build_stub(handle, len(text), bounded_provenance)
        return ArtifactReference(
            handle=handle,
            digest=digest,
            chars=len(text),
            bytes=len(payload),
            provenance=bounded_provenance,
            stub=stub,
        )

    def recall(
        self,
        session_id: str,
        handle: str,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> RecallResult | None:
        """Recall an exact ``[start:end)`` character range from this session."""
        digest = parse_handle(handle)
        if (
            not session_id
            or digest is None
            or isinstance(start, bool)
            or not isinstance(start, int)
        ):
            return None
        if end is not None and (isinstance(end, bool) or not isinstance(end, int)):
            return None
        if start < 0 or (end is not None and end < start):
            return None

        path = self._session_dir(session_id) / f"{digest}.txt"
        try:
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                return None
            text = payload.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        resolved_end = len(text) if end is None else end
        if start > len(text) or resolved_end > len(text):
            return None
        return RecallResult(
            handle=handle,
            text=text[start:resolved_end],
            start=start,
            end=resolved_end,
            total_chars=len(text),
        )

    def manifest(self, session_id: str) -> dict:
        """Return a copy of the bounded upload manifest for a session."""
        try:
            return json.loads(json.dumps(self._read_manifest(session_id)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return self._empty_manifest(session_id)

    def cleanup(self, session_id: str) -> None:
        """Best-effort removal of one session's local artifacts."""
        directory = self._session_dir(session_id)
        try:
            paths = list(directory.iterdir())
        except OSError:
            return
        for path in paths:
            if path.is_file():
                with contextlib.suppress(OSError):
                    path.unlink()
        with contextlib.suppress(OSError):
            directory.rmdir()

    def _session_dir(self, session_id: str) -> Path:
        # Hashing prevents provider-controlled IDs from becoming paths while the
        # manifest retains the original ID for later upload association.
        key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
        return self.root / key

    def _manifest_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "manifest.json"

    def _empty_manifest(self, session_id: str) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "session_id": session_id,
            "next_sequence": 1,
            "artifacts": [],
        }

    def _read_manifest(self, session_id: str) -> dict:
        path = self._manifest_path(session_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self._empty_manifest(session_id)
        if (
            not isinstance(data, dict)
            or data.get("version") != SCHEMA_VERSION
            or data.get("session_id") != session_id
            or not isinstance(data.get("artifacts"), list)
        ):
            raise ValueError("invalid artifact manifest")
        return data

    def _write_manifest(self, session_id: str, manifest: dict) -> None:
        payload = json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._atomic_write_bytes(self._manifest_path(session_id), payload)

    def _bounded_entries(self, entries: list[dict]) -> tuple[list[dict], list[dict]]:
        ordered = sorted(entries, key=lambda entry: int(entry.get("sequence", 0)))
        retained = list(ordered)
        total_bytes = sum(max(0, int(entry.get("bytes", 0))) for entry in retained)
        removed: list[dict] = []
        while (
            len(retained) > self.limits.max_artifacts or total_bytes > self.limits.max_session_bytes
        ):
            stale = retained.pop(0)
            total_bytes -= max(0, int(stale.get("bytes", 0)))
            removed.append(stale)
        return retained, removed

    @staticmethod
    def _artifact_matches(path: Path, digest: str) -> bool:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest() == digest
        except OSError:
            return False

    @staticmethod
    def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with open(temporary, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
        finally:
            with contextlib.suppress(OSError):
                temporary.unlink()


def parse_handle(handle: str) -> str | None:
    if not isinstance(handle, str):
        return None
    match = _HANDLE_RE.fullmatch(handle)
    return match.group("digest") if match else None


def build_stub(handle: str, chars: int, provenance: ArtifactProvenance) -> str:
    """Render deterministic one-line provenance and recall instructions."""
    fields = [
        "rclm artifact evicted",
        f"provider={provenance.provider or 'unknown'}",
        f"tool={provenance.tool_name or 'unknown'}",
    ]
    if provenance.tool_use_id:
        fields.append(f"call={provenance.tool_use_id}")
    if provenance.target:
        fields.append(f"target={provenance.target}")
    fields.extend((f"chars={chars}", f"handle={handle}", f"recall=chars[0:{chars})"))
    return "[" + " | ".join(fields) + "]"


def _compact_optional(value: str | None) -> str | None:
    if value is None:
        return None
    compacted = _compact_field(value)
    return compacted or None


def _compact_field(value: str) -> str:
    if not isinstance(value, str):
        return ""
    # Avoid multiline or delimiter injection into the model-visible stub.
    compacted = " ".join(value.split()).replace("|", "/")
    if len(compacted) <= _MAX_PROVENANCE_FIELD_CHARS:
        return compacted
    return compacted[: _MAX_PROVENANCE_FIELD_CHARS - 1] + "…"
