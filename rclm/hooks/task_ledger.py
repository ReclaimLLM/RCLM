"""Deterministic, bounded task-state extraction from provider tool semantics.

This module does not summarize arbitrary model text.  It records only explicit
plan fields, known file-operation inputs, recognized test commands, and
structured failure status so the ledger remains auditable and replayable.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = 1
_DEFAULT_ROOT = Path.home() / ".reclaimllm" / "sessions" / "task-ledgers"
_TEST_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:"
    r"(?:uv\s+run\s+)?(?:python(?:3)?\s+-m\s+)?pytest\b|"
    r"(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test\b|"
    r"(?:npx\s+)?(?:jest|vitest)\b|"
    r"cargo\s+test\b|go\s+test\b|dotnet\s+test\b|"
    r"(?:mvnw?|gradlew?|gradle)\b[^;&|]*(?:test|check)\b"
    r")",
    re.IGNORECASE,
)
_FILE_FIELDS = (
    "file_path",
    "path",
    "absolute_path",
    "target_file",
    "notebook_path",
    "AbsolutePath",
    "TargetFile",
    "SearchPath",
    "SearchDirectory",
    "DirectoryPath",
)
_READ_TOOLS = frozenset(
    {
        "read",
        "read_file",
        "read_text_file",
        "view_file",
        "get_file_contents",
    }
)
_SEARCH_TOOLS = frozenset({"grep", "grep_search", "search_files", "find_by_name"})
_WRITE_TOOLS = frozenset(
    {
        "write",
        "write_file",
        "replace",
        "replace_file_content",
        "multi_replace_file_content",
        "edit",
        "multiedit",
        "multi_edit",
        "notebookedit",
        "notebook_edit",
        "apply_patch",
        "applypatch",
        "write_to_file",
    }
)
_CREATE_TOOLS = frozenset({"create_file"})
_DELETE_TOOLS = frozenset({"delete_file", "remove_file"})
_PLAN_TOOLS = frozenset(
    {
        "update_plan",
        "updateplan",
        "write_todos",
        "writetodos",
        "todo_write",
        "todowrite",
        "set_task_state",
        "settaskstate",
    }
)
_PATCH_PATH = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)

FileOperation = Literal["read", "searched", "modified", "created", "deleted"]
TestStatus = Literal["passed", "failed", "unknown"]


@dataclass(frozen=True)
class LedgerLimits:
    max_goal_chars: int = 512
    max_item_chars: int = 512
    max_path_chars: int = 768
    max_constraints: int = 32
    max_decisions: int = 32
    max_files: int = 64
    max_tests: int = 32
    max_errors: int = 32
    max_next_actions: int = 32


@dataclass(frozen=True)
class FileState:
    path: str
    operation: FileOperation
    tool_name: str
    tool_use_id: str | None = None
    artifact_handle: str | None = None


@dataclass(frozen=True)
class TestState:
    command: str
    status: TestStatus
    summary: str | None = None
    tool_use_id: str | None = None
    artifact_handle: str | None = None


@dataclass(frozen=True)
class ErrorState:
    source: str
    message: str
    tool_use_id: str | None = None
    artifact_handle: str | None = None


@dataclass(frozen=True)
class TaskLedger:
    version: int = SCHEMA_VERSION
    revision: int = 0
    goal: str | None = None
    constraints: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    files: tuple[FileState, ...] = ()
    tests: tuple[TestState, ...] = ()
    errors: tuple[ErrorState, ...] = ()
    next_actions: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Return a stable, fully typed JSON shape, including empty fields."""
        return {
            "version": self.version,
            "revision": self.revision,
            "goal": self.goal,
            "constraints": list(self.constraints),
            "decisions": list(self.decisions),
            "files": [asdict(item) for item in self.files],
            "tests": [asdict(item) for item in self.tests],
            "errors": [asdict(item) for item in self.errors],
            "next_actions": list(self.next_actions),
        }


@dataclass(frozen=True)
class LedgerPatch:
    goal: str | None = None
    constraints: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    files: tuple[FileState, ...] = ()
    tests: tuple[TestState, ...] = ()
    errors: tuple[ErrorState, ...] = ()
    next_actions: tuple[str, ...] | None = None

    @property
    def empty(self) -> bool:
        return not any(
            (
                self.goal,
                self.constraints,
                self.decisions,
                self.files,
                self.tests,
                self.errors,
                self.next_actions is not None,
            )
        )


@dataclass(frozen=True)
class LedgerWriteResult:
    ledger: TaskLedger
    persisted: bool


class TaskLedgerStore:
    """Atomically persist bounded typed state for one host session."""

    def __init__(self, root: Path | None = None, *, limits: LedgerLimits | None = None) -> None:
        self.root = root or _DEFAULT_ROOT
        self.limits = limits or LedgerLimits()

    def read(self, session_id: str) -> TaskLedger:
        if not session_id:
            return TaskLedger()
        try:
            raw = json.loads(self._path(session_id).read_text(encoding="utf-8"))
            return _ledger_from_dict(raw, self.limits)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return TaskLedger()

    def observe_tool(
        self,
        session_id: str,
        tool_name: str,
        tool_input: object,
        tool_response: object,
        *,
        tool_use_id: str | None = None,
        artifact_handle: str | None = None,
        cwd: str | None = None,
    ) -> LedgerWriteResult:
        """Infer and persist auditable state, failing open on filesystem errors."""
        current = self.read(session_id)
        if not session_id:
            return LedgerWriteResult(current, persisted=False)
        patch = infer_ledger_patch(
            tool_name,
            tool_input,
            tool_response,
            tool_use_id=tool_use_id,
            artifact_handle=artifact_handle,
            cwd=cwd,
            limits=self.limits,
        )
        if patch.empty:
            return LedgerWriteResult(current, persisted=True)
        updated = apply_patch(current, patch, self.limits)
        try:
            self._write(session_id, updated)
        except (OSError, TypeError, ValueError):
            return LedgerWriteResult(current, persisted=False)
        return LedgerWriteResult(updated, persisted=True)

    def cleanup(self, session_id: str) -> None:
        with contextlib.suppress(OSError):
            self._path(session_id).unlink()

    def _path(self, session_id: str) -> Path:
        import hashlib

        key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
        return self.root / f"{key}.json"

    def _write(self, session_id: str, ledger: TaskLedger) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = self._path(session_id)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        payload = json.dumps(
            ledger.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
        finally:
            with contextlib.suppress(OSError):
                temporary.unlink()


def infer_ledger_patch(
    tool_name: str,
    tool_input: object,
    tool_response: object,
    *,
    tool_use_id: str | None = None,
    artifact_handle: str | None = None,
    cwd: str | None = None,
    limits: LedgerLimits | None = None,
) -> LedgerPatch:
    """Extract only state directly supported by one tool observation."""
    limits = limits or LedgerLimits()
    normalized_name = _canonical_tool_name(tool_name)
    data = tool_input if isinstance(tool_input, dict) else {}
    bounded_tool_name = _bounded(tool_name, limits.max_item_chars)
    bounded_call_id = _bounded_optional(tool_use_id, limits.max_item_chars)
    bounded_handle = _bounded_optional(artifact_handle, limits.max_item_chars)

    goal: str | None = None
    constraints: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    next_actions: tuple[str, ...] | None = None
    if normalized_name in _PLAN_TOOLS:
        goal = _bounded_optional(data.get("goal"), limits.max_goal_chars)
        constraints = _bounded_string_list(data.get("constraints"), limits.max_item_chars)
        decisions = _bounded_string_list(data.get("decisions"), limits.max_item_chars)
        next_actions = _pending_plan_items(data)
        next_actions = tuple(_bounded(item, limits.max_item_chars) for item in next_actions)

    error_message = _error_message(tool_response, limits.max_item_chars)
    exit_code = _exit_code(tool_response)
    failed = error_message is not None or (exit_code is not None and exit_code != 0)
    operation = _file_operation(normalized_name)
    files: list[FileState] = []
    if operation is not None and not failed:
        for path in _tool_paths(normalized_name, data, cwd=cwd):
            files.append(
                FileState(
                    path=_bounded(path, limits.max_path_chars),
                    operation=operation,
                    tool_name=bounded_tool_name,
                    tool_use_id=bounded_call_id,
                    artifact_handle=bounded_handle,
                )
            )

    tests: list[TestState] = []
    command = data.get("command") or data.get("cmd") or data.get("CommandLine")
    if isinstance(command, str) and _TEST_COMMAND.search(command):
        status: TestStatus = "unknown"
        if error_message is not None or (exit_code is not None and exit_code != 0):
            status = "failed"
        elif exit_code == 0:
            status = "passed"
        tests.append(
            TestState(
                command=_bounded(command, limits.max_item_chars),
                status=status,
                summary=_result_summary(tool_response, limits.max_item_chars),
                tool_use_id=bounded_call_id,
                artifact_handle=bounded_handle,
            )
        )

    errors: list[ErrorState] = []
    if failed:
        message = (
            error_message
            or _result_summary(tool_response, limits.max_item_chars)
            or f"tool exited with status {exit_code}"
        )
        errors.append(
            ErrorState(
                source=bounded_tool_name or "tool",
                message=message,
                tool_use_id=bounded_call_id,
                artifact_handle=bounded_handle,
            )
        )

    return LedgerPatch(
        goal=goal,
        constraints=constraints,
        decisions=decisions,
        files=tuple(files),
        tests=tuple(tests),
        errors=tuple(errors),
        next_actions=next_actions,
    )


def apply_patch(ledger: TaskLedger, patch: LedgerPatch, limits: LedgerLimits) -> TaskLedger:
    """Merge a patch with deterministic de-duplication and oldest-first bounds."""
    goal = patch.goal or ledger.goal
    constraints = _merge_strings(ledger.constraints, patch.constraints, limits.max_constraints)
    decisions = _merge_strings(ledger.decisions, patch.decisions, limits.max_decisions)
    files = _merge_records(
        ledger.files,
        patch.files,
        key=lambda item: item.path,
        limit=limits.max_files,
    )
    tests = _merge_records(
        ledger.tests,
        patch.tests,
        key=lambda item: item.command,
        limit=limits.max_tests,
    )
    errors = _merge_records(
        ledger.errors,
        patch.errors,
        key=lambda item: (item.source, item.message),
        limit=limits.max_errors,
    )
    next_actions = ledger.next_actions
    if patch.next_actions is not None:
        next_actions = _tail(_unique(patch.next_actions), limits.max_next_actions)
    return TaskLedger(
        revision=ledger.revision + 1,
        goal=goal,
        constraints=constraints,
        decisions=decisions,
        files=files,
        tests=tests,
        errors=errors,
        next_actions=next_actions,
    )


def _ledger_from_dict(raw: object, limits: LedgerLimits) -> TaskLedger:
    if not isinstance(raw, dict) or raw.get("version") != SCHEMA_VERSION:
        raise ValueError("invalid task-ledger schema")
    ledger = TaskLedger(
        revision=max(0, int(raw.get("revision", 0))),
        goal=_bounded_optional(raw.get("goal"), limits.max_goal_chars),
        constraints=_bounded_string_list(raw.get("constraints"), limits.max_item_chars),
        decisions=_bounded_string_list(raw.get("decisions"), limits.max_item_chars),
        files=tuple(_file_state(item, limits) for item in _dict_items(raw.get("files"))),
        tests=tuple(_test_state(item, limits) for item in _dict_items(raw.get("tests"))),
        errors=tuple(_error_state(item, limits) for item in _dict_items(raw.get("errors"))),
        next_actions=_bounded_string_list(raw.get("next_actions"), limits.max_item_chars),
    )
    return TaskLedger(
        revision=ledger.revision,
        goal=ledger.goal,
        constraints=_tail(ledger.constraints, limits.max_constraints),
        decisions=_tail(ledger.decisions, limits.max_decisions),
        files=_tail(ledger.files, limits.max_files),
        tests=_tail(ledger.tests, limits.max_tests),
        errors=_tail(ledger.errors, limits.max_errors),
        next_actions=_tail(ledger.next_actions, limits.max_next_actions),
    )


def _file_state(item: dict, limits: LedgerLimits) -> FileState:
    operation = item.get("operation")
    if operation not in {"read", "searched", "modified", "created", "deleted"}:
        raise ValueError("invalid file operation")
    return FileState(
        path=_bounded(item.get("path"), limits.max_path_chars),
        operation=operation,
        tool_name=_bounded(item.get("tool_name"), limits.max_item_chars),
        tool_use_id=_bounded_optional(item.get("tool_use_id"), limits.max_item_chars),
        artifact_handle=_bounded_optional(item.get("artifact_handle"), limits.max_item_chars),
    )


def _test_state(item: dict, limits: LedgerLimits) -> TestState:
    status = item.get("status")
    if status not in {"passed", "failed", "unknown"}:
        raise ValueError("invalid test status")
    return TestState(
        command=_bounded(item.get("command"), limits.max_item_chars),
        status=status,
        summary=_bounded_optional(item.get("summary"), limits.max_item_chars),
        tool_use_id=_bounded_optional(item.get("tool_use_id"), limits.max_item_chars),
        artifact_handle=_bounded_optional(item.get("artifact_handle"), limits.max_item_chars),
    )


def _error_state(item: dict, limits: LedgerLimits) -> ErrorState:
    return ErrorState(
        source=_bounded(item.get("source"), limits.max_item_chars),
        message=_bounded(item.get("message"), limits.max_item_chars),
        tool_use_id=_bounded_optional(item.get("tool_use_id"), limits.max_item_chars),
        artifact_handle=_bounded_optional(item.get("artifact_handle"), limits.max_item_chars),
    )


def _dict_items(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _canonical_tool_name(tool_name: str) -> str:
    name = tool_name.rsplit("__", 1)[-1].rsplit(":", 1)[-1]
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _file_operation(tool_name: str) -> FileOperation | None:
    if tool_name in _READ_TOOLS:
        return "read"
    if tool_name in _SEARCH_TOOLS:
        return "searched"
    if tool_name in _CREATE_TOOLS:
        return "created"
    if tool_name in _DELETE_TOOLS:
        return "deleted"
    if tool_name in _WRITE_TOOLS:
        return "modified"
    return None


def _tool_paths(tool_name: str, data: dict, *, cwd: str | None) -> tuple[str, ...]:
    values: list[str] = []
    for field_name in _FILE_FIELDS:
        value = data.get(field_name)
        if isinstance(value, str) and value.strip():
            values.append(_normalize_path(value, cwd))
    if tool_name == "apply_patch":
        patch = data.get("patch") or data.get("input")
        if isinstance(patch, str):
            values.extend(_normalize_path(path, cwd) for path in _PATCH_PATH.findall(patch))
    return _unique(tuple(values))


def _normalize_path(path: str, cwd: str | None) -> str:
    path = path.strip()
    if cwd and not os.path.isabs(path):
        path = os.path.join(cwd, path)
    return os.path.normpath(path)


def _pending_plan_items(data: dict) -> tuple[str, ...]:
    raw = data.get("plan")
    if raw is None:
        raw = data.get("todos")
    if not isinstance(raw, list):
        return ()
    items: list[str] = []
    for item in raw:
        if isinstance(item, str):
            items.append(item)
            continue
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "pending")).lower()
        if status in {"completed", "complete", "done", "cancelled", "canceled"}:
            continue
        value = item.get("step") or item.get("content") or item.get("task") or item.get("text")
        if isinstance(value, str) and value.strip():
            items.append(value)
    return _unique(tuple(items))


def _exit_code(response: object) -> int | None:
    if isinstance(response, str):
        match = re.search(r"The command exited with code (\d+)\.", response)
        return int(match.group(1)) if match is not None else None
    if not isinstance(response, dict):
        return None
    for key in ("exit_code", "exitCode", "status_code"):
        value = response.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _error_message(response: object, max_chars: int) -> str | None:
    if not isinstance(response, dict):
        return None
    error = response.get("error")
    if isinstance(error, str) and error.strip():
        return _bounded(error, max_chars)
    if error is not None and error is not False:
        return _bounded(json.dumps(error, ensure_ascii=False, sort_keys=True), max_chars)
    if response.get("isError") is True or response.get("is_error") is True:
        return _result_summary(response, max_chars) or "tool reported an error"
    return None


def _result_summary(response: object, max_chars: int) -> str | None:
    candidates: list[str] = []
    if isinstance(response, str):
        candidates.append(response)
    elif isinstance(response, dict):
        for key in ("stderr", "stdout", "output", "content", "message", "reason"):
            value = response.get(key)
            if isinstance(value, str):
                candidates.append(value)
            elif isinstance(value, list):
                candidates.extend(
                    block.get("text", "")
                    for block in value
                    if isinstance(block, dict) and isinstance(block.get("text"), str)
                )
    for candidate in candidates:
        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        if lines:
            return _bounded(lines[-1], max_chars)
    return None


def _bounded_string_list(value: object, max_chars: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    bounded = tuple(
        _bounded(item, max_chars) for item in value if isinstance(item, str) and item.strip()
    )
    return _unique(tuple(item for item in bounded if item))


def _bounded_optional(value: object, max_chars: int) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    bounded = _bounded(value, max_chars)
    return bounded or None


def _bounded(value: object, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    if max_chars <= 0:
        return ""
    compacted = " ".join(value.split())
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max(0, max_chars - 1)] + "…"


def _merge_strings(current: tuple[str, ...], new: tuple[str, ...], limit: int) -> tuple[str, ...]:
    return _tail(_unique(current + new), limit)


def _merge_records(current: tuple, new: tuple, *, key, limit: int) -> tuple:
    merged = list(current)
    for item in new:
        item_key = key(item)
        merged = [existing for existing in merged if key(existing) != item_key]
        merged.append(item)
    return _tail(tuple(merged), limit)


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _tail(values: tuple, limit: int) -> tuple:
    return values[-limit:] if limit > 0 else ()
