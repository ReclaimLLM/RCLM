"""Filters for git command output."""

from __future__ import annotations

import re


def filter_git(subcommand: str, output: str) -> str | None:
    """Filter git command output. Returns compressed string or None if no filter applies."""
    fn = _GIT_FILTERS.get(subcommand)
    if fn is None:
        return None
    return fn(output)  # type: ignore[operator]


def _filter_status(output: str) -> str:
    """Compact git status to summary counts + file list."""
    if not output.strip():
        return "clean — nothing to commit"

    counts: dict[str, list[str]] = {
        "modified": [],
        "added": [],
        "deleted": [],
        "renamed": [],
        "untracked": [],
        "staged": [],
    }

    for line in output.splitlines():
        line = line.rstrip()
        if not line or len(line) < 3:
            continue
        # Short-format: XY filename
        # Long-format: detect common prefixes
        if line.startswith("??"):
            counts["untracked"].append(line[3:].strip())
        elif line.startswith("A ") or line.startswith(" A"):
            counts["added"].append(line[2:].strip())
        elif line.startswith("M ") or line.startswith(" M") or line.startswith("MM"):
            counts["modified"].append(line[2:].strip())
        elif line.startswith("D ") or line.startswith(" D"):
            counts["deleted"].append(line[2:].strip())
        elif line.startswith("R"):
            counts["renamed"].append(line[2:].strip())
        # Long-format fallback
        elif "modified:" in line:
            m = re.search(r"modified:\s+(.+)", line)
            if m:
                counts["modified"].append(m.group(1).strip())
        elif "new file:" in line:
            m = re.search(r"new file:\s+(.+)", line)
            if m:
                counts["added"].append(m.group(1).strip())
        elif "deleted:" in line:
            m = re.search(r"deleted:\s+(.+)", line)
            if m:
                counts["deleted"].append(m.group(1).strip())
        elif "renamed:" in line:
            m = re.search(r"renamed:\s+(.+)", line)
            if m:
                counts["renamed"].append(m.group(1).strip())
        elif line.startswith("\t"):
            # Untracked files in long format are indented
            counts["untracked"].append(line.strip())

    parts = []
    for label, files in counts.items():
        if files:
            # Deduplicate
            unique = list(dict.fromkeys(files))
            parts.append(f"{len(unique)} {label}: {', '.join(unique)}")

    return "\n".join(parts) if parts else output


def _filter_diff(output: str) -> str:
    """Summarize git diff: stat line + truncated hunks."""
    if not output.strip():
        return "no changes"

    lines = output.splitlines()
    result: list[str] = []
    hunk_lines = 0
    max_hunk_lines = 20

    for line in lines:
        # Always keep diff headers and stat lines
        if (
            line.startswith("diff --git")
            or line.startswith("---")
            or line.startswith("+++")
            or line.startswith("@@")
        ):
            result.append(line)
            hunk_lines = 0
        elif line.startswith("+") or line.startswith("-"):
            hunk_lines += 1
            if hunk_lines <= max_hunk_lines:
                result.append(line)
            elif hunk_lines == max_hunk_lines + 1:
                result.append(f"  ... ({_count_remaining_hunk_lines(lines, line)} more lines)")
        # Context lines
        elif line.startswith(" "):
            if hunk_lines <= max_hunk_lines:
                result.append(line)

    return "\n".join(result)


def _count_remaining_hunk_lines(lines: list[str], current: str) -> int:
    """Count how many +/- lines remain after truncation point."""
    found = False
    count = 0
    for line in lines:
        if line is current:
            found = True
            continue
        if found:
            if line.startswith("@@") or line.startswith("diff --git"):
                break
            if line.startswith("+") or line.startswith("-"):
                count += 1
    return count


def _filter_log(output: str) -> str:
    """Condense git log to one-line-per-commit, max 20."""
    if not output.strip():
        return "no commits"

    lines = output.splitlines()
    commits: list[str] = []
    current_hash = ""
    current_msg = ""

    for line in lines:
        if line.startswith("commit "):
            if current_hash:
                commits.append(f"{current_hash[:7]} {current_msg.strip()}")
            current_hash = line[7:].strip()
            current_msg = ""
        elif line.startswith("    "):
            if not current_msg:
                current_msg = line.strip()

    if current_hash:
        commits.append(f"{current_hash[:7]} {current_msg.strip()}")

    # If output already looks one-line-per-commit, just truncate
    if not commits:
        commits = [line.strip() for line in lines if line.strip()]

    if len(commits) > 20:
        commits = commits[:20]
        commits.append(f"... ({len(lines)} total lines truncated)")

    return "\n".join(commits)


def _filter_action(output: str) -> str:
    """For git add/commit/push/pull — keep just the summary line."""
    if not output.strip():
        return "ok"

    lines = [line.strip() for line in output.splitlines() if line.strip()]

    # For commit: look for the summary line like "[branch hash] message"
    for line in lines:
        if re.match(r"\[.+\s+[0-9a-f]+\]", line):
            return line

    # For push/pull: look for the -> line or summary
    for line in lines:
        if "->" in line or "Already up to date" in line or "Everything up-to-date" in line:
            return line

    # Fallback: first meaningful line, capped
    first = lines[0] if lines else "ok"
    if len(first) > 200:
        first = first[:200] + "..."
    return first


_GIT_FILTERS = {
    "status": _filter_status,
    "diff": _filter_diff,
    "log": _filter_log,
    "add": _filter_action,
    "commit": _filter_action,
    "push": _filter_action,
    "pull": _filter_action,
    "fetch": _filter_action,
    "merge": _filter_action,
    "rebase": _filter_action,
    "checkout": _filter_action,
    "switch": _filter_action,
    "branch": _filter_action,
    "stash": _filter_action,
    "reset": _filter_action,
    "restore": _filter_action,
}
