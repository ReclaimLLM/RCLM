"""Small provider-name adapter for shared tool-result mechanisms.

Provider handlers keep ownership of their wire contracts.  This module only
maps the explicit tool names and input keys we have observed to the semantic
families used by replay and the provider-neutral transform core.
"""

from __future__ import annotations

from typing import Literal

ToolFamily = Literal["shell", "read", "search", "edit", "listing"]

_SHELL_NAMES = frozenset(
    {"bash", "exec", "exec_command", "shell", "run_command", "run_shell_command"}
)
_READ_NAMES = frozenset({"read", "read_file", "read_text_file", "view_file"})
_SEARCH_NAMES = frozenset({"grep", "grep_search", "search_files"})
_EDIT_NAMES = frozenset(
    {
        "edit",
        "replace",
        "replace_file_content",
        "multi_replace_file_content",
        "write_file",
        "write_to_file",
    }
)
_LISTING_NAMES = frozenset({"find_by_name", "list_dir", "list_directory"})


def normalized_tool_name(tool_name: str) -> str:
    """Return the provider-local leaf name in lower case."""
    return tool_name.rsplit("__", 1)[-1].lower()


def tool_family(tool_name: str) -> ToolFamily | None:
    """Map an explicitly supported provider tool name to one shared family."""
    name = normalized_tool_name(tool_name)
    if name in _SHELL_NAMES:
        return "shell"
    if name in _READ_NAMES:
        return "read"
    if name in _SEARCH_NAMES:
        return "search"
    if name in _EDIT_NAMES:
        return "edit"
    if name in _LISTING_NAMES:
        return "listing"
    return None


def command_text(tool_input: object) -> str | None:
    """Return the command from Claude/Codex or Antigravity input shapes."""
    if not isinstance(tool_input, dict):
        return None
    for key in ("command", "cmd", "CommandLine"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def file_path(tool_input: object) -> str | None:
    """Return a file target from the supported native provider shapes."""
    if not isinstance(tool_input, dict):
        return None
    for key in ("file_path", "AbsolutePath", "TargetFile"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def semantic_target(tool_name: str, tool_input: object) -> str | None:
    """Build a conservative key for comparing successive related results."""
    if not isinstance(tool_input, dict):
        return None
    family = tool_family(tool_name)
    if family in {"read", "edit"}:
        return file_path(tool_input)
    if family == "shell":
        command = command_text(tool_input)
        cwd = tool_input.get("Cwd") or tool_input.get("cwd") or ""
        return f"{cwd}\x00{command}" if command else None
    if family == "search":
        query = tool_input.get("Query") or tool_input.get("pattern")
        path = tool_input.get("SearchPath") or tool_input.get("path") or ""
        return f"{path}\x00{query}" if isinstance(query, str) and query else None
    if family == "listing":
        path = (
            tool_input.get("SearchDirectory")
            or tool_input.get("DirectoryPath")
            or tool_input.get("path")
        )
        pattern = tool_input.get("Pattern") or ""
        return f"{path}\x00{pattern}" if isinstance(path, str) and path else None
    return None


def supports_result_compaction(tool_name: str) -> bool:
    """Whether the shared transform has an explicit adapter for this name."""
    normalized = normalized_tool_name(tool_name)
    return tool_family(tool_name) in {"shell", "search", "edit", "listing"} or normalized in {
        "view_file",
        "browser_snapshot",
        "browser_accessibility_snapshot",
        "execute_sql",
    }


__all__ = [
    "ToolFamily",
    "command_text",
    "file_path",
    "normalized_tool_name",
    "semantic_target",
    "supports_result_compaction",
    "tool_family",
]
