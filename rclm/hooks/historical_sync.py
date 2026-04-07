"""Historical session sync: discover and upload existing sessions from all providers.

Called during install when the user opts in. Discovers existing session files from
Claude Code (~/.claude/projects/**/*.jsonl), Gemini CLI (~/.gemini/tmp/**/chats/*.json),
and Codex CLI (~/.codex/sessions/**/*.jsonl), parses them into HookSessionRecord
objects, and uploads them via the same mechanism as live sessions.

A sync index at ~/.reclaimllm/synced_sessions.json tracks which files have already
been uploaded so re-running the installer never duplicates records.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from rclm._models import (
    FileDiff,
    FileEvent,
    HookSessionRecord,
    ProxyRecord,
    SessionRecord,
    ToolCall,
)
from rclm._uploader import _FAILED_UPLOADS_DIR, AnyRecord, upload_single
from rclm.hooks import codex_transcript
from rclm.hooks import transcript as claude_transcript
from rclm.hooks._analytics import compute_session_analytics

# Number of retry attempts when reprocessing quarantined failed uploads.
# Set low (1) so a persistently-unreachable server doesn't block the user for long.
_FAILED_UPLOAD_MAX_RETRIES = 1

# ---------------------------------------------------------------------------
# Sync index — tracks which files have already been uploaded
# ---------------------------------------------------------------------------

_SYNCED_INDEX = Path.home() / ".reclaimllm" / "synced_sessions.json"


def _load_synced_index() -> set[str]:
    if not _SYNCED_INDEX.exists():
        return set()
    try:
        data = json.loads(_SYNCED_INDEX.read_text(encoding="utf-8"))
        return set(data.get("synced", []))
    except Exception:
        return set()


def _save_synced_index(synced: set[str]) -> None:
    _SYNCED_INDEX.parent.mkdir(parents=True, exist_ok=True)
    _SYNCED_INDEX.write_text(
        json.dumps({"synced": sorted(synced)}, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _iter_claude_sessions() -> list[Path]:
    """Yield top-level Claude Code JSONL transcript files.

    Only includes direct children of each project directory; subagent transcripts
    (nested under <session-uuid>/subagents/) are intentionally excluded.
    """
    base = Path.home() / ".claude" / "projects"
    if not base.exists():
        return []
    files = []
    for project_dir in base.iterdir():
        if not project_dir.is_dir():
            continue
        for f in project_dir.glob("*.jsonl"):
            if f.is_file():
                files.append(f)
    return files


def _iter_gemini_sessions() -> list[Path]:
    """Yield Gemini CLI session JSON files from ~/.gemini/tmp/**/chats/."""
    base = Path.home() / ".gemini" / "tmp"
    if not base.exists():
        return []
    return [f for f in base.rglob("chats/*.json") if f.is_file()]


def _iter_codex_sessions() -> list[Path]:
    """Yield Codex CLI JSONL session files from ~/.codex/sessions/."""
    base = Path.home() / ".codex" / "sessions"
    if not base.exists():
        return []
    return [f for f in base.rglob("*.jsonl") if f.is_file()]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _timestamps_to_duration(started_at: str | None, ended_at: str | None) -> float:
    if not started_at or not ended_at:
        return 0.0
    try:
        return (
            datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)
        ).total_seconds()
    except (ValueError, TypeError):
        return 0.0


def _z_to_utc(ts: str) -> str:
    """Convert ISO-8601 timestamp with trailing 'Z' to '+00:00' offset."""
    return ts.replace("Z", "+00:00") if ts.endswith("Z") else ts


def _derive_session_id(path: Path) -> str:
    """Return a stable session_id for a file path.

    Tries filename stem → UUID parse → trailing UUID extraction → uuid5 fallback.
    """
    stem = path.stem
    try:
        return str(uuid.UUID(stem))
    except ValueError:
        pass
    # e.g. "rollout-2026-02-06T14-55-47-019c3486-6120-75f2-90b8-860c9a21dd85"
    parts = stem.split("-")
    if len(parts) >= 5:
        candidate = "-".join(parts[-5:])
        try:
            return str(uuid.UUID(candidate))
        except ValueError:
            pass
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(path)))


# ---------------------------------------------------------------------------
# Claude parsing
# ---------------------------------------------------------------------------


def _extract_claude_file_diffs(tool_calls: list[ToolCall]) -> list[FileDiff]:
    diffs: list[FileDiff] = []
    for tc in tool_calls:
        name = tc.tool_name
        inp = tc.tool_input
        if name == "Write":
            file_path = inp.get("file_path", "")
            content = inp.get("content", "")
            unified = "".join(
                difflib.unified_diff(
                    [],
                    content.splitlines(keepends=True),
                    fromfile=f"a/{file_path}",
                    tofile=f"b/{file_path}",
                )
            )
            diffs.append(
                FileDiff(
                    path=file_path,
                    before=None,
                    after=content,
                    unified_diff=unified,
                )
            )
        elif name == "Edit":
            file_path = inp.get("file_path", "")
            old = inp.get("old_string", "")
            new = inp.get("new_string", "")
            unified = "".join(
                difflib.unified_diff(
                    old.splitlines(keepends=True),
                    new.splitlines(keepends=True),
                    fromfile=f"a/{file_path}",
                    tofile=f"b/{file_path}",
                )
            )
            diffs.append(FileDiff(path=file_path, before=old, after=new, unified_diff=unified))
        elif name == "MultiEdit":
            file_path = inp.get("file_path", "")
            for edit in inp.get("edits", []):
                old = edit.get("old_string", "")
                new = edit.get("new_string", "")
                unified = "".join(
                    difflib.unified_diff(
                        old.splitlines(keepends=True),
                        new.splitlines(keepends=True),
                        fromfile=f"a/{file_path}",
                        tofile=f"b/{file_path}",
                    )
                )
                diffs.append(
                    FileDiff(
                        path=file_path,
                        before=old,
                        after=new,
                        unified_diff=unified,
                    )
                )
    return diffs


def _parse_claude_session(path: Path) -> HookSessionRecord | None:
    """Parse a Claude Code JSONL transcript file into a HookSessionRecord.

    Uses the existing transcript parser for messages and tool calls, then does
    a separate pass over the raw entries to extract session_id, cwd, model,
    and token usage (which live in entry["message"] rather than the top level).
    """
    raw_entries: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                with contextlib.suppress(json.JSONDecodeError):
                    raw_entries.append(json.loads(line))
    except OSError:
        return None

    if not raw_entries:
        return None

    # First pass: extract session_id, cwd, model, and tokens from raw entries.
    session_id: str | None = None
    cwd = ""
    model: str | None = None
    total_input = 0
    total_output = 0
    has_tokens = False

    for entry in raw_entries:
        if session_id is None:
            session_id = entry.get("sessionId")
        if not cwd:
            cwd = entry.get("cwd", "")
        if entry.get("type") == "assistant":
            msg = entry.get("message", {})
            if isinstance(msg, dict):
                if model is None and msg.get("model"):
                    model = msg["model"]
                usage = msg.get("usage", {})
                if isinstance(usage, dict) and usage:
                    has_tokens = True
                    total_input += usage.get("input_tokens", 0)
                    total_output += usage.get("output_tokens", 0)

    if not session_id:
        session_id = _derive_session_id(path)

    # Second pass: use the existing transcript parser for messages + tool calls.
    transcript_data = claude_transcript.parse_transcript(str(path))
    if not transcript_data.messages and not transcript_data.tool_calls:
        return None

    timestamps = [msg["timestamp"] for msg in transcript_data.messages if msg.get("timestamp")]
    started_at = min(timestamps) if timestamps else None
    ended_at = max(timestamps) if timestamps else None
    duration_s = _timestamps_to_duration(started_at, ended_at)

    file_diffs = _extract_claude_file_diffs(transcript_data.tool_calls)
    analytics = compute_session_analytics(transcript_data.tool_calls, file_diffs)

    # Prefer transcript-level tokens if the parser found them, else use our pass.
    final_input = transcript_data.total_input_tokens or (total_input if has_tokens else None)
    final_output = transcript_data.total_output_tokens or (total_output if has_tokens else None)
    final_model = transcript_data.model or model or "claude-unknown"

    return HookSessionRecord(
        session_id=session_id,
        cwd=cwd,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=duration_s,
        transcript_path=str(path),
        model=final_model,
        messages=transcript_data.messages,
        tool_calls=transcript_data.tool_calls,
        file_diffs=file_diffs,
        total_input_tokens=final_input,
        total_output_tokens=final_output,
        tool_token_stats=analytics.get("tool_token_stats"),
        tool_call_count=analytics.get("tool_call_count"),
        unique_files_modified=analytics.get("unique_files_modified"),
        dominant_tool=analytics.get("dominant_tool"),
        is_sync=True,
    )


# ---------------------------------------------------------------------------
# Gemini parsing
# ---------------------------------------------------------------------------


def _extract_gemini_text(content: object) -> str:
    """Flatten Gemini's content field (list of {text: ...} or plain string) to str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            (
                str(item.get("text") or item.get("content") or "")
                if isinstance(item, dict)
                else str(item)
            )
            for item in content
            if item
        ]
        return "\n".join(p for p in parts if p)
    return ""


def _extract_gemini_tool_result(result: object) -> str | None:
    """Extract output text from a Gemini functionResponse result array."""
    if not isinstance(result, list) or not result:
        return None
    first = result[0]
    if not isinstance(first, dict):
        return None
    resp = first.get("functionResponse", {})
    if isinstance(resp, dict):
        response = resp.get("response", {})
        if isinstance(response, dict):
            return response.get("output")
    return None


def _extract_gemini_file_diffs(tool_name: str, args: object) -> list[FileDiff]:
    if not isinstance(args, dict):
        return []
    diffs: list[FileDiff] = []
    if tool_name == "write_file":
        file_path = args.get("file_path", "")
        content = args.get("content", "")
        unified = "".join(
            difflib.unified_diff(
                [],
                content.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
            )
        )
        diffs.append(FileDiff(path=file_path, before=None, after=content, unified_diff=unified))
    elif tool_name == "replace":
        file_path = args.get("file_path", "")
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        unified = "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
            )
        )
        diffs.append(FileDiff(path=file_path, before=old, after=new, unified_diff=unified))
    return diffs


def _parse_gemini_session(path: Path) -> HookSessionRecord | None:
    """Parse a Gemini CLI JSON chat file into a HookSessionRecord.

    Gemini session format:
      {sessionId, startTime, lastUpdated, messages: [{id, timestamp, type, content,
        toolCalls?: [{id, name, args, result, timestamp}], tokens?, model?}]}

    Tool calls and their results are embedded together in type=="gemini" messages.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    session_id = data.get("sessionId") or _derive_session_id(path)
    start_time = _z_to_utc(data.get("startTime", ""))
    end_time = _z_to_utc(data.get("lastUpdated", ""))
    duration_s = _timestamps_to_duration(start_time or None, end_time or None)

    # cwd is not stored in Gemini session files; use the project directory name as a hint.
    cwd = path.parent.parent.name  # ~/.gemini/tmp/<project-name>/chats/session.json

    messages: list[dict] = []
    tool_calls: list[ToolCall] = []
    file_diffs: list[FileDiff] = []
    model: str | None = None
    total_input = 0
    total_output = 0
    has_tokens = False
    tool_counter = 0

    for msg in data.get("messages") or []:
        msg_type = msg.get("type", "")
        timestamp = _z_to_utc(msg.get("timestamp", ""))

        if msg_type == "user":
            text = _extract_gemini_text(msg.get("content"))
            if text:
                messages.append({"role": "user", "content": text, "timestamp": timestamp})

        elif msg_type == "gemini":
            if model is None and msg.get("model"):
                model = msg["model"]
            tokens = msg.get("tokens") or {}
            if isinstance(tokens, dict) and tokens:
                has_tokens = True
                total_input += tokens.get("input", 0)
                total_output += tokens.get("output", 0)

            # Assistant text response (may be absent when only tool calls are made).
            text = _extract_gemini_text(msg.get("content"))
            if text:
                messages.append(
                    {
                        "role": "assistant",
                        "content": text,
                        "timestamp": timestamp,
                    }
                )

            # Tool calls are embedded alongside the assistant turn.
            for tc_raw in msg.get("toolCalls") or []:
                call_id = tc_raw.get("id") or f"gemini-tool-{tool_counter}"
                tool_name = tc_raw.get("name", "")
                args = tc_raw.get("args") or {}
                tool_result = _extract_gemini_tool_result(tc_raw.get("result"))
                tc_timestamp = _z_to_utc(tc_raw.get("timestamp", timestamp))

                tool_calls.append(
                    ToolCall(
                        tool_use_id=call_id,
                        tool_name=tool_name,
                        tool_input=(args if isinstance(args, dict) else {"input": args}),
                        tool_result=tool_result,
                        timestamp=tc_timestamp,
                    )
                )
                file_diffs.extend(_extract_gemini_file_diffs(tool_name, args))
                tool_counter += 1

    if not messages and not tool_calls:
        return None

    analytics = compute_session_analytics(tool_calls, file_diffs)

    return HookSessionRecord(
        session_id=session_id,
        cwd=cwd,
        started_at=start_time or None,
        ended_at=end_time or None,
        duration_s=duration_s,
        transcript_path=str(path),
        model=model or "gemini-unknown",
        messages=messages,
        tool_calls=tool_calls,
        file_diffs=file_diffs,
        total_input_tokens=total_input if has_tokens else None,
        total_output_tokens=total_output if has_tokens else None,
        tool_token_stats=analytics.get("tool_token_stats"),
        tool_call_count=analytics.get("tool_call_count"),
        unique_files_modified=analytics.get("unique_files_modified"),
        dominant_tool=analytics.get("dominant_tool"),
        is_sync=True,
    )


# ---------------------------------------------------------------------------
# Codex parsing
# ---------------------------------------------------------------------------


def _parse_codex_session(path: Path) -> HookSessionRecord | None:
    """Parse a Codex CLI JSONL file into a HookSessionRecord.

    Uses the existing codex_transcript parser for messages, tool calls, and file
    diffs, then reads the session_meta entry for session_id and cwd.
    """
    # Extract session_id and cwd from the session_meta entry.
    session_id: str | None = None
    cwd = ""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "session_meta":
                        payload = entry.get("payload") or {}
                        session_id = payload.get("id")
                        cwd = payload.get("cwd", "")
                        break
                except json.JSONDecodeError:
                    pass
    except OSError:
        return None

    if not session_id:
        session_id = _derive_session_id(path)

    transcript_data = codex_transcript.parse_transcript(str(path))
    if not transcript_data.messages and not transcript_data.tool_calls:
        return None

    timestamps = [msg["timestamp"] for msg in transcript_data.messages if msg.get("timestamp")]
    started_at = min(timestamps) if timestamps else None
    ended_at = max(timestamps) if timestamps else None
    duration_s = _timestamps_to_duration(started_at, ended_at)

    analytics = compute_session_analytics(transcript_data.tool_calls, transcript_data.file_diffs)

    return HookSessionRecord(
        session_id=session_id,
        cwd=cwd,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=duration_s,
        transcript_path=str(path),
        model=transcript_data.model or "codex-unknown",
        messages=transcript_data.messages,
        tool_calls=transcript_data.tool_calls,
        file_diffs=transcript_data.file_diffs,
        total_input_tokens=None,
        total_output_tokens=None,
        tool_token_stats=analytics.get("tool_token_stats"),
        tool_call_count=analytics.get("tool_call_count"),
        unique_files_modified=analytics.get("unique_files_modified"),
        dominant_tool=analytics.get("dominant_tool"),
        is_sync=True,
    )


# ---------------------------------------------------------------------------
# Failed-upload reprocessing
# ---------------------------------------------------------------------------


def _deserialize_record(data: dict) -> AnyRecord | None:
    """Reconstruct a quarantined record from its JSON dict.

    Discriminates on ``record_type`` (ProxyRecord), ``command`` (SessionRecord),
    or falls back to HookSessionRecord.
    """
    try:
        if data.get("record_type") == "proxy":
            return ProxyRecord(
                session_id=data["session_id"],
                timestamp=data["timestamp"],
                request_body=data["request_body"],
                response_body=data.get("response_body"),
                is_streaming=data["is_streaming"],
                duration_ms=data["duration_ms"],
                model=data.get("model"),
                messages=data.get("messages", []),
                tool_calls=[ToolCall(**tc) for tc in data.get("tool_calls", [])],
                file_diffs=[FileDiff(**fd) for fd in data.get("file_diffs", [])],
                provider=data.get("provider"),
                response_cost=data.get("response_cost"),
                total_input_tokens=data.get("total_input_tokens"),
                total_output_tokens=data.get("total_output_tokens"),
            )
        if "command" in data:
            return SessionRecord(
                session_id=data["session_id"],
                command=data["command"],
                started_at=data["started_at"],
                ended_at=data["ended_at"],
                duration_s=data["duration_s"],
                exit_code=data.get("exit_code"),
                pty_output=data.get("pty_output", ""),
                file_events=[FileEvent(**fe) for fe in data.get("file_events", [])],
                diffs=[FileDiff(**d) for d in data.get("diffs", [])],
            )
        # Default: HookSessionRecord
        return HookSessionRecord(
            session_id=data["session_id"],
            cwd=data.get("cwd", ""),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            duration_s=data.get("duration_s", 0.0),
            transcript_path=data.get("transcript_path"),
            model=data.get("model"),
            messages=data.get("messages", []),
            tool_calls=[ToolCall(**tc) for tc in data.get("tool_calls", [])],
            file_diffs=[FileDiff(**fd) for fd in data.get("file_diffs", [])],
            total_input_tokens=data.get("total_input_tokens"),
            total_output_tokens=data.get("total_output_tokens"),
            tool_token_stats=data.get("tool_token_stats"),
            tool_call_count=data.get("tool_call_count"),
            unique_files_modified=data.get("unique_files_modified"),
            dominant_tool=data.get("dominant_tool"),
            compression_savings=data.get("compression_savings"),
            is_sync=data.get("is_sync", False),
        )
    except Exception:
        return None


async def _reprocess_failed_uploads() -> tuple[int, int]:
    """Re-upload every file in ~/.reclaimllm/failed_uploads/.

    Files that upload successfully are deleted. Files that fail again are
    left in place (``_quarantine`` will overwrite them on the next retry).

    Returns:
        (uploaded, failed) counts.
    """
    if not _FAILED_UPLOADS_DIR.exists():
        return 0, 0

    paths = sorted(_FAILED_UPLOADS_DIR.glob("*.json"))
    if not paths:
        return 0, 0

    uploaded = 0
    failed = 0
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            print(
                f"    ! {path.name}: could not parse JSON — skipping",
                file=sys.stderr,
            )
            failed += 1
            continue

        record = _deserialize_record(data)
        if record is None:
            print(
                f"    ! {path.name}: unknown record type — skipping",
                file=sys.stderr,
            )
            failed += 1
            continue

        # Remember file size before upload so we can detect if quarantine re-wrote it.
        mtime_before = path.stat().st_mtime
        await upload_single(record, max_retries=_FAILED_UPLOAD_MAX_RETRIES)

        # If the file's mtime changed, upload failed and quarantine re-wrote it; leave it.
        try:
            mtime_after = path.stat().st_mtime
            if mtime_after != mtime_before:
                failed += 1
                print(f"    ✗ {path.name}")
            else:
                path.unlink(missing_ok=True)
                uploaded += 1
                print(f"    ✓ {path.name}")
        except FileNotFoundError:
            # File was removed somehow; treat as success.
            uploaded += 1
            print(f"    ✓ {path.name}")

    return uploaded, failed


# ---------------------------------------------------------------------------
# Discovery + upload orchestration
# ---------------------------------------------------------------------------


def _discover_sessions(providers: list[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    if "claude" in providers:
        result["claude"] = _iter_claude_sessions()
    if "gemini" in providers:
        result["gemini"] = _iter_gemini_sessions()
    if "codex" in providers:
        result["codex"] = _iter_codex_sessions()
    return result


def _parse_session(provider: str, path: Path) -> HookSessionRecord | None:
    try:
        if provider == "claude":
            return _parse_claude_session(path)
        if provider == "gemini":
            return _parse_gemini_session(path)
        if provider == "codex":
            return _parse_codex_session(path)
    except Exception:
        pass
    return None


async def _upload_all(
    by_provider: dict[str, list[Path]],
    already_synced: set[str],
) -> int:
    """Parse and upload all un-synced sessions. Returns count of uploaded records."""
    uploaded = 0
    for provider, paths in by_provider.items():
        new_paths = [p for p in paths if str(p) not in already_synced]
        if not new_paths:
            continue
        print(f"\n  {provider.capitalize()} ({len(new_paths)} new sessions):")
        for path in new_paths:
            record = _parse_session(provider, path)
            if record is None:
                # Empty or unreadable session — mark as synced to skip next time.
                already_synced.add(str(path))
                continue
            await upload_single(record)
            already_synced.add(str(path))
            uploaded += 1
            print(f"    ✓ {path.name}")
    return uploaded


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def prompt_and_run_sync(
    providers: list[str],
    *,
    force_yes: bool = False,
    resync: bool = False,
    failed: bool = False,
) -> None:
    """Discover and upload historical sessions, prompting the user first.

    Args:
        providers: Which providers to scan ("claude", "gemini", "codex").
        force_yes: Skip the confirmation prompt (for rclm-sync --yes).
        resync: Ignore the synced-sessions index and re-upload everything,
                including sessions already uploaded in a previous run.
                The index is updated after a successful resync.
        failed: Reprocess quarantined records from ~/.reclaimllm/failed_uploads/.

    Silently returns when stdin is not a TTY and force_yes is False (CI/piped).
    """
    if not force_yes and not sys.stdin.isatty():
        return

    # --- Failed-upload reprocessing (independent of provider sync) ---
    if failed:
        failed_paths = (
            sorted(_FAILED_UPLOADS_DIR.glob("*.json")) if _FAILED_UPLOADS_DIR.exists() else []
        )
        if not failed_paths:
            print("No failed uploads to reprocess.")
        else:
            print(f"\nFound {len(failed_paths)} failed upload(s) to reprocess.")
            if not force_yes:
                try:
                    answer = input("Reprocess them now? [y/N] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                if answer != "y":
                    return
            print("Reprocessing failed uploads...")

            async def _run_failed() -> tuple[int, int]:
                return await _reprocess_failed_uploads()

            ok, bad = asyncio.run(_run_failed())
            print(f"\nDone. Reprocessed {ok} succeeded, {bad} still failing.")
        return

    # --- Normal historical provider sync ---
    by_provider = _discover_sessions(providers)
    already_synced: set[str] = set() if resync else _load_synced_index()

    new_count = sum(
        1 for paths in by_provider.values() for p in paths if str(p) not in already_synced
    )
    if new_count == 0:
        if force_yes:
            print("No new sessions to sync.")
        return

    provider_summary = ", ".join(
        f"{len([p for p in paths if str(p) not in already_synced])} {name}"
        for name, paths in by_provider.items()
        if any(str(p) not in already_synced for p in paths)
    )
    if resync:
        print(f"\nResync: uploading all {new_count} session(s) ({provider_summary}).")
    else:
        print(f"\nFound {new_count} existing session(s) to sync ({provider_summary}).")

    if not force_yes:
        try:
            answer = input("Upload them to ReclaimLLM now? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if answer != "y":
            return

    print("Syncing historical sessions...")

    async def _run() -> int:
        return await _upload_all(by_provider, already_synced)

    uploaded = asyncio.run(_run())
    _save_synced_index(already_synced)
    print(f"\nDone. Synced {uploaded} session(s).")


def sync_main() -> None:
    """Entry point for the rclm-sync CLI command.

    Usage:
        rclm-sync                    # all providers, interactive prompt
        rclm-sync --claude           # Claude only
        rclm-sync --gemini --codex   # Gemini + Codex
        rclm-sync --yes              # skip confirmation prompt
        rclm-sync --resync           # re-upload everything, ignoring prior sync index
        rclm-sync --failed           # reprocess quarantined failed uploads
        rclm-sync --failed --yes     # reprocess without confirmation prompt
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync existing Claude/Gemini/Codex sessions to ReclaimLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s                   # scan all providers, ask before uploading
  %(prog)s --claude          # Claude Code only
  %(prog)s --gemini --codex  # Gemini + Codex
  %(prog)s --yes             # upload without confirmation prompt
  %(prog)s --resync          # re-upload all sessions, ignoring prior sync index
  %(prog)s --resync --yes    # resync without confirmation prompt
  %(prog)s --failed          # reprocess quarantined failed uploads
  %(prog)s --failed --yes    # reprocess failed uploads without confirmation""",
    )
    parser.add_argument("--claude", action="store_true", help="Sync Claude Code sessions")
    parser.add_argument("--gemini", action="store_true", help="Sync Gemini CLI sessions")
    parser.add_argument("--codex", action="store_true", help="Sync Codex CLI sessions")
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Upload without confirmation prompt",
    )
    parser.add_argument(
        "--resync",
        action="store_true",
        help="Ignore the synced-sessions index and re-upload all discovered sessions",
    )
    parser.add_argument(
        "--failed",
        action="store_true",
        help="Reprocess quarantined records from ~/.reclaimllm/failed_uploads/",
    )
    args = parser.parse_args()

    providers = [p for p in ("claude", "gemini", "codex") if getattr(args, p)]
    if not providers:
        providers = ["claude", "gemini", "codex"]

    from rclm import _config

    cfg = _config.load()
    if not cfg.get("api_key"):
        print(
            "No API key configured. Run rclm-hooks-install first.",
            file=sys.stderr,
        )
        sys.exit(1)

    prompt_and_run_sync(providers, force_yes=args.yes, resync=args.resync, failed=args.failed)


if __name__ == "__main__":
    sync_main()
