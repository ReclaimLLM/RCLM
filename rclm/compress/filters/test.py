"""Conservative test-runner output filters.

Each parser returns ``None`` on ambiguity. Callers must pass through original
output in that case, especially for a non-zero exit status.
"""

from __future__ import annotations

import random
import re

DEFAULT_MAX_CHARS = 8_000

# Varied plain-language phrasings for the same fact, so repeated cap/summary
# messages within one session don't all read as the same canned string.
_CAP_MESSAGES = (
    "{dropped} more output lines cut off (limit: {max_chars:,} characters).",
    "{dropped} additional lines left out to stay under the {max_chars:,}-character limit.",
    "{dropped} more lines not shown (output capped at {max_chars:,} characters).",
)
_DETAILS_OMITTED_PHRASES = (
    "details left out to save space",
    "details skipped to keep this short",
    "details not shown here",
)

_PYTEST_SUMMARY = re.compile(r"=+ .*?(?:passed|failed|error|errors).*?=+$", re.IGNORECASE)
_PYTEST_COUNTS = re.compile(r"(?:(\d+) failed)?(?:.*?(\d+) passed)?", re.IGNORECASE)
_JS_SUMMARY = re.compile(r"^(?:Test Suites|Tests):\s+\d+", re.MULTILINE)
_GO_SUMMARY = re.compile(r"^(?:ok|FAIL)\s+\S+", re.MULTILINE)


def filter_test(
    command: str, output: str, *, exit_code: int = 0, max_chars: int = DEFAULT_MAX_CHARS
) -> str | None:
    """Filter a recognized runner only when its output is confidently understood."""
    cmd = command.lower()
    if "pytest" in cmd:
        return _filter_pytest(output, exit_code=exit_code, max_chars=max_chars)
    if any(marker in cmd for marker in ("npm test", "npm run test", "npx jest", "npx vitest")):
        return _filter_js(output, exit_code=exit_code, max_chars=max_chars)
    if re.search(r"(?:^|\s)go\s+test(?:\s|$)", cmd):
        return _filter_go(output, exit_code=exit_code, max_chars=max_chars)
    if "cargo test" in cmd:
        return _filter_cargo(output, exit_code=exit_code, max_chars=max_chars)
    return None


def _cap(lines: list[str], max_chars: int) -> str:
    kept: list[str] = []
    used = 0
    dropped = 0
    for line in lines:
        extra = len(line) + 1
        if used + extra > max_chars:
            dropped += 1
            continue
        kept.append(line)
        used += extra
    if dropped:
        message = random.choice(_CAP_MESSAGES).format(dropped=dropped, max_chars=max_chars)
        kept.append(f"[RCLM] {message}")
    return "\n".join(kept)


def _passed_line(count: int | str) -> str:
    phrase = random.choice(_DETAILS_OMITTED_PHRASES)
    return f"{count} passed ({phrase})"


def _filter_pytest(output: str, *, exit_code: int, max_chars: int) -> str | None:
    lines = output.splitlines()
    summaries = [line for line in lines if _PYTEST_SUMMARY.search(line)]
    if not summaries:
        return None
    summary = summaries[-1]
    has_failure = bool(re.search(r"\b(?:failed|error|errors)\b", summary, re.IGNORECASE))
    if exit_code and not has_failure:
        return None
    kept: list[str] = []
    in_details = False
    for line in lines:
        if "ERRORS" in line or "FAILURES" in line:
            in_details = True
        if in_details:
            kept.append(line)
            if line == summary:
                in_details = False
        elif line.startswith("ERROR collecting") or line.startswith("ImportError"):
            kept.append(line)
    if not has_failure:
        passed = re.search(r"(\d+) passed", summary)
        if not passed:
            return None
        kept = [summary, _passed_line(passed.group(1))]
    elif summary not in kept:
        kept.append(summary)
    return _cap(kept, max_chars)


def _filter_js(output: str, *, exit_code: int, max_chars: int) -> str | None:
    lines = output.splitlines()
    summary_lines = [line for line in lines if _JS_SUMMARY.search(line)]
    if not summary_lines:
        return None
    has_failure = any(
        re.search(r"\b[1-9]\d* failed\b|FAIL\s", line, re.IGNORECASE) for line in lines
    )
    if exit_code and not has_failure:
        return None
    kept: list[str] = []
    in_failure = False
    for line in lines:
        stripped = line.strip()
        if "FAIL " in line or stripped.startswith(("● ", "✕ ", "× ")):
            in_failure = True
        if in_failure:
            kept.append(line)
            if not stripped:
                in_failure = False
    kept.extend(summary_lines)
    if not has_failure:
        tests = next((line for line in summary_lines if line.strip().startswith("Tests:")), None)
        if tests is None:
            return None
        match = re.search(r"(\d+) passed", tests)
        if match:
            kept.append(_passed_line(match.group(1)))
    return _cap(kept, max_chars)


def _filter_go(output: str, *, exit_code: int, max_chars: int) -> str | None:
    lines = output.splitlines()
    summaries = [line for line in lines if _GO_SUMMARY.search(line)]
    has_failure = any(line.startswith("FAIL") or line.startswith("--- FAIL:") for line in lines)
    if not summaries or (exit_code and not has_failure):
        return None
    kept: list[str] = []
    include = False
    for line in lines:
        if line.startswith("--- FAIL:"):
            include = True
        if include:
            kept.append(line)
        if line.startswith("FAIL"):
            include = False
    kept.extend(summaries)
    if not has_failure:
        passed = sum(1 for line in lines if line.startswith("ok\t"))
        kept.append(_passed_line(passed))
    return _cap(kept, max_chars)


def _filter_cargo(output: str, *, exit_code: int, max_chars: int) -> str | None:
    lines = output.splitlines()
    summaries = [line for line in lines if line.startswith("test result:")]
    has_failure = any("FAILED" in line for line in summaries)
    if not summaries or (exit_code and not has_failure):
        return None
    if not has_failure:
        return _cap([summaries[-1]], max_chars)
    start = next((i for i, line in enumerate(lines) if line.startswith("failures:")), None)
    kept = lines[start:] if start is not None else []
    if summaries[-1] not in kept:
        kept.append(summaries[-1])
    return _cap(kept, max_chars)
