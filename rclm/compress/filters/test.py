"""Filters for test runner output (pytest, npm test, cargo test)."""

from __future__ import annotations

import re


def filter_test(command: str, output: str) -> str | None:
    """Filter test runner output. Returns compressed string or None if no filter."""
    cmd_lower = command.lower()
    if "pytest" in cmd_lower or "python -m pytest" in cmd_lower:
        return _filter_pytest(output)
    if (
        "npm test" in cmd_lower
        or "npm run test" in cmd_lower
        or "npx jest" in cmd_lower
        or "npx vitest" in cmd_lower
    ):
        return _filter_js_test(output)
    if "cargo test" in cmd_lower:
        return _filter_cargo_test(output)
    return None


def _filter_pytest(output: str) -> str:
    """Keep failures + summary, hide passing tests."""
    lines = output.splitlines()
    result: list[str] = []
    in_failure = False

    for line in lines:
        # Always keep FAILED/ERROR lines
        if "FAILED" in line or "ERROR" in line:
            result.append(line)
            in_failure = True
            continue

        # Keep failure detail blocks
        if line.startswith("____") or line.startswith("===="):
            if "FAILURES" in line or "ERRORS" in line or "short test summary" in line:
                in_failure = True
                result.append(line)
            elif in_failure:
                # End of failure block
                in_failure = False
                result.append(line)
            # Summary line at end
            if re.search(r"\d+ passed", line) or re.search(r"\d+ failed", line):
                result.append(line)
            continue

        if in_failure:
            result.append(line)
            continue

        # Keep the final summary line
        if re.search(r"=+ .*(passed|failed|error)", line):
            result.append(line)

    if not result:
        # All passed — find summary
        for line in lines:
            if re.search(r"\d+ passed", line):
                return line.strip()
        return f"all passed ({len(lines)} lines of output)"

    return "\n".join(result)


def _filter_js_test(output: str) -> str:
    """Keep failures + summary for Jest/Vitest output."""
    lines = output.splitlines()
    result: list[str] = []
    in_failure = False

    for line in lines:
        stripped = line.strip()

        # Jest/Vitest failure markers
        if (
            stripped.startswith("● ")
            or stripped.startswith("✕ ")
            or stripped.startswith("× ")
            or "FAIL " in line
        ):
            in_failure = True
            result.append(line)
            continue

        if in_failure:
            result.append(line)
            # Empty line after failure block often ends it
            if not stripped:
                in_failure = False
            continue

        # Summary lines
        if re.search(r"Tests?:\s+\d+", line) or re.search(r"Test Suites?:\s+\d+", line):
            result.append(line)

    if not result:
        # All passed
        for line in lines:
            if re.search(r"Tests?:\s+\d+", line):
                return line.strip()
        return f"all passed ({len(lines)} lines of output)"

    return "\n".join(result)


def _filter_cargo_test(output: str) -> str:
    """Keep failures + summary for cargo test output."""
    lines = output.splitlines()
    result: list[str] = []
    in_failure = False

    for line in lines:
        if "---- " in line and "stdout ----" in line:
            in_failure = True
            result.append(line)
            continue

        if line.startswith("failures:"):
            in_failure = True
            result.append(line)
            continue

        if in_failure:
            result.append(line)
            if line.startswith("test result:"):
                in_failure = False
            continue

        # Summary
        if line.startswith("test result:"):
            result.append(line)

    if not result:
        for line in lines:
            if line.startswith("test result:"):
                return line.strip()
        return f"all passed ({len(lines)} lines of output)"

    return "\n".join(result)
