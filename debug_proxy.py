#!/usr/bin/env python3
"""Debug-mode launcher for rclm-proxy.

Runs LiteLLM in-process instead of as a subprocess so that:
  - debugpy breakpoints work in litellm_callback.py
  - source edits are picked up without reinstalling the package

Usage (VS Code: "Debug rclm-proxy (dev)" launch config):
    python debug_proxy.py [extra litellm args...]

Manual:
    cd ReclaimLLM-data-capture
    ANTHROPIC_API_KEY=... python debug_proxy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── 1. Put the source tree first so edits to litellm_callback.py are live ────
#       (takes precedence over any installed rclm wheel)
_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# ── 2. Locate the litellm_config.yaml written by 'rclm-proxy setup' ────
_config = Path.home() / ".reclaimllm" / "litellm_config.yaml"
if not _config.exists():
    print(
        f"error: config not found at {_config}\n"
        "Run 'rclm-proxy setup' first.",
        file=sys.stderr,
    )
    sys.exit(1)

# ── 3. Patch sys.argv before any litellm import consumes it ──────────────────
extra_args = sys.argv[1:]
sys.argv = ["litellm", "--config", str(_config), *extra_args]

# ── 4. Call the same entry point the installed 'litellm' command uses ─────────
#       importlib.metadata resolves the real function regardless of litellm's
#       internal module layout, which has shifted across major versions.
from importlib.metadata import entry_points  # noqa: E402

_eps = entry_points(group="console_scripts")
_litellm_eps = [ep for ep in _eps if ep.name == "litellm"]
if not _litellm_eps:
    print("error: 'litellm' console_script entry point not found.\n"
          "Install litellm: pip install 'rclm[proxy]'", file=sys.stderr)
    sys.exit(1)

_litellm_eps[0].load()()
