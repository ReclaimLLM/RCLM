"""Filters for shell commands (ls, find, generic truncation)."""

from __future__ import annotations


def filter_shell(command: str, output: str) -> str | None:
    """Filter shell command output. Returns compressed string or None if no filter."""
    parts = command.strip().split()
    if not parts:
        return None

    base_cmd = parts[0]
    if base_cmd in ("ls", "find"):
        return _filter_listing(output)

    return None


def _filter_listing(output: str) -> str:
    """Compress directory listings: group by directory, show counts."""
    if not output.strip():
        return "(empty)"

    lines = [line.rstrip() for line in output.splitlines() if line.strip()]

    if len(lines) <= 30:
        return output  # Small enough, keep as-is

    # Group files by parent directory
    dirs: dict[str, list[str]] = {}
    plain_files: list[str] = []

    for line in lines:
        # Skip header/summary lines from ls -l
        if line.startswith("total ") or line.startswith("d") or line.startswith("-"):
            plain_files.append(line)
            continue

        parts = line.rsplit("/", 1)
        if len(parts) == 2:
            parent, name = parts
            dirs.setdefault(parent, []).append(name)
        else:
            plain_files.append(line)

    if not dirs:
        # No directory structure found, just truncate
        return _truncate_lines(lines, 30)

    result: list[str] = []
    for parent, files in sorted(dirs.items()):
        if len(files) <= 5:
            for f in files:
                result.append(f"{parent}/{f}")
        else:
            result.append(f"{parent}/ ({len(files)} files)")
            for f in files[:3]:
                result.append(f"  {f}")
            result.append(f"  ... +{len(files) - 3} more")

    if plain_files:
        result.extend(plain_files[:10])
        if len(plain_files) > 10:
            result.append(f"... +{len(plain_files) - 10} more files")

    return "\n".join(result)


def _truncate_lines(lines: list[str], max_lines: int) -> str:
    """Truncate a list of lines to max_lines with a count of omitted lines."""
    if len(lines) <= max_lines:
        return "\n".join(lines)
    kept = lines[:max_lines]
    kept.append(f"... ({len(lines) - max_lines} more lines)")
    return "\n".join(kept)


def truncate_output(output: str, max_lines: int = 50) -> str:
    """Generic output truncation. Used as fallback for unrecognized commands."""
    lines = output.splitlines()
    if len(lines) <= max_lines:
        return output
    return _truncate_lines(lines, max_lines)
