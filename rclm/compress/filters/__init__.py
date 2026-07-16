"""Command output filters for rclm-compress."""

from __future__ import annotations

from rclm.compress.filters.git import filter_git
from rclm.compress.filters.search import filter_search
from rclm.compress.filters.shell import filter_generic, filter_shell, strip_ansi
from rclm.compress.filters.test import filter_test

__all__ = [
    "filter_git",
    "filter_test",
    "filter_shell",
    "filter_search",
    "filter_generic",
    "strip_ansi",
]
