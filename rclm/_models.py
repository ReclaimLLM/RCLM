"""Shared dataclasses used by rclm-proxy and rclm-claude-hooks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProxyRecord:
    session_id: str  # uuid4
    timestamp: str  # ISO-8601, request start
    request_body: dict | str  # {"messages": [...], "model": "...", ...}
    response_body: dict | list | str | None  # parsed response or {"error": "..."}
    is_streaming: bool
    duration_ms: float
    model: str | None  # fully-qualified e.g. "anthropic/claude-sonnet-4-5"
    # Synthesised from request_body/response_body at build time for blob consistency
    messages: list[dict] = field(default_factory=list)  # [{role, content, timestamp}]
    tool_calls: list[ToolCall] = field(default_factory=list)  # always empty for proxy
    file_diffs: list[FileDiff] = field(default_factory=list)  # always empty for proxy
    provider: str | None = None  # inferred from model prefix
    response_cost: float | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    record_type: str = "proxy"


@dataclass
class FileEvent:
    path: str
    event_type: str  # "created" | "modified" | "deleted"
    timestamp: str


@dataclass
class FileDiff:
    path: str
    before: str | None  # None if file didn't exist before
    after: str | None  # None if file deleted
    unified_diff: str  # output of difflib.unified_diff


@dataclass
class SessionRecord:
    session_id: str
    command: list[str]  # argv
    started_at: str
    ended_at: str
    duration_s: float
    exit_code: int | None
    pty_output: str  # ANSI-stripped terminal transcript
    file_events: list[FileEvent] = field(default_factory=list)
    diffs: list[FileDiff] = field(default_factory=list)


@dataclass
class ToolCall:
    tool_use_id: str
    tool_name: str
    tool_input: dict
    tool_result: str | dict | list | None
    timestamp: str  # ISO-8601
    input_token_estimate: int | None = None
    output_token_estimate: int | None = None


@dataclass
class HookSessionRecord:
    session_id: str
    cwd: str
    started_at: str
    ended_at: str
    duration_s: float
    transcript_path: str | None
    model: str | None
    messages: list[dict] = field(default_factory=list)  # [{role, content, timestamp}]
    tool_calls: list[ToolCall] = field(default_factory=list)
    file_diffs: list[FileDiff] = field(default_factory=list)
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    tool_token_stats: dict | None = (
        None  # {"Bash": {"count": N, "input_tokens": N, "output_tokens": N}, ...}
    )
    tool_call_count: int | None = None
    unique_files_modified: int | None = None
    dominant_tool: str | None = None
    compression_savings: dict | None = (
        None  # {"total_original_chars": N, "total_compressed_chars": N, "savings_pct": float}
    )
    is_sync: bool = False  # True for historical sync uploads; server skips if session exists
