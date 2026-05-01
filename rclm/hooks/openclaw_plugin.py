"""OpenClaw plugin file generation and config patching."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

PLUGIN_ID = "reclaimllm"
OPENCLAW_DIR = Path.home() / ".openclaw"
OPENCLAW_CONFIG_PATH = OPENCLAW_DIR / "openclaw.json"
PLUGIN_DIR = OPENCLAW_DIR / "extensions" / PLUGIN_ID

_HOOKS = [
    "session_start",
    "session_end",
    "llm_input",
    "llm_output",
    "before_tool_call",
    "after_tool_call",
    "agent_end",
    "message_received",
    "message_sent",
]


def _plugin_index_ts(binary: str) -> str:
    hooks_array = ",\n".join(f'  "{hook}"' for hook in _HOOKS)
    return f"""import {{ spawn }} from "node:child_process";
import {{ definePluginEntry }} from "openclaw/plugin-sdk/plugin-entry";

const RCLM_BINARY = {json.dumps(binary)};
const HOOKS = [
{hooks_array}
];

async function forward(hookName: string, event: unknown): Promise<void> {{
  await new Promise<void>((resolve) => {{
    const child = spawn(RCLM_BINARY, [hookName], {{
      stdio: ["pipe", "ignore", "ignore"]
    }});

    child.on("error", () => resolve());
    child.on("close", () => resolve());
    child.stdin.on("error", () => resolve());
    child.stdin.end(JSON.stringify({{
      hook_name: hookName,
      received_at: new Date().toISOString(),
      event
    }}));
  }});
}}

export default definePluginEntry({{
  id: "reclaimllm",
  name: "ReclaimLLM Capture",
  register(api) {{
    for (const hookName of HOOKS) {{
      api.on(hookName as any, async (event: unknown) => {{
        await forward(hookName, event);
      }});
    }}
  }}
}});
"""


def _manifest_json() -> str:
    return (
        json.dumps(
            {
                "id": PLUGIN_ID,
                "name": "ReclaimLLM Capture",
                "version": "0.1.0",
                "description": "Capture OpenClaw sessions into ReclaimLLM.",
                "configSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
            },
            indent=2,
        )
        + "\n"
    )


def _package_json() -> str:
    return (
        json.dumps(
            {
                "name": "reclaimllm-openclaw-capture",
                "version": "0.1.0",
                "type": "module",
                "openclaw": {
                    "extensions": ["./index.ts"],
                },
            },
            indent=2,
        )
        + "\n"
    )


def _resolve_binary(name: str) -> str:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    return name


def install_plugin(use_global: bool = True) -> Path:
    if not use_global:
        raise ValueError("OpenClaw plugin install is only supported globally")

    binary = _resolve_binary("rclm-openclaw-hooks")
    PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
    (PLUGIN_DIR / "index.ts").write_text(_plugin_index_ts(binary), encoding="utf-8")
    (PLUGIN_DIR / "openclaw.plugin.json").write_text(_manifest_json(), encoding="utf-8")
    (PLUGIN_DIR / "package.json").write_text(_package_json(), encoding="utf-8")
    _patch_config_for_install(PLUGIN_DIR)
    return PLUGIN_DIR


def uninstall_plugin(use_global: bool = True) -> tuple[bool, bool]:
    if not use_global:
        path = Path(".openclaw") / "extensions" / PLUGIN_ID
        removed_files = _remove_plugin_dir(path)
        removed_config = _patch_config_for_uninstall(path)
        return removed_files, removed_config

    removed_files = _remove_plugin_dir(PLUGIN_DIR)
    removed_config = _patch_config_for_uninstall(PLUGIN_DIR)
    return removed_files, removed_config


def _remove_plugin_dir(path: Path) -> bool:
    if path.exists():
        shutil.rmtree(path)
        return True
    return False


def _load_config() -> dict | None:
    if not OPENCLAW_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(OPENCLAW_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(
            f"Warning: {OPENCLAW_CONFIG_PATH} is not strict JSON; "
            "OpenClaw plugin files were written but config was not modified.",
            file=sys.stderr,
        )
        return None


def _write_config(data: dict) -> None:
    OPENCLAW_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPENCLAW_CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _patch_config_for_install(plugin_dir: Path) -> bool:
    config = _load_config()
    if config is None:
        return False

    original = copy.deepcopy(config)
    plugins = config.setdefault("plugins", {})
    plugins["enabled"] = True
    load = plugins.setdefault("load", {})
    paths = load.setdefault("paths", [])
    plugin_path = str(plugin_dir)
    if plugin_path not in paths:
        paths.append(plugin_path)
    entries = plugins.setdefault("entries", {})
    entry = entries.setdefault(PLUGIN_ID, {})
    entry["enabled"] = True
    hooks = entry.setdefault("hooks", {})
    hooks["allowConversationAccess"] = True

    allow = plugins.get("allow")
    if isinstance(allow, list) and PLUGIN_ID not in allow:
        allow.append(PLUGIN_ID)

    if config != original:
        _write_config(config)
        return True
    return False


def _patch_config_for_uninstall(plugin_dir: Path) -> bool:
    config = _load_config()
    if config is None:
        return False

    original = copy.deepcopy(config)
    plugins = config.get("plugins")
    if isinstance(plugins, dict):
        entries = plugins.get("entries")
        if isinstance(entries, dict):
            entries.pop(PLUGIN_ID, None)
            if not entries:
                plugins.pop("entries", None)

        load = plugins.get("load")
        if isinstance(load, dict):
            paths = load.get("paths")
            if isinstance(paths, list):
                plugin_path = str(plugin_dir)
                load["paths"] = [path for path in paths if path != plugin_path]
                if not load["paths"]:
                    load.pop("paths", None)
            if not load:
                plugins.pop("load", None)

        allow = plugins.get("allow")
        if isinstance(allow, list):
            plugins["allow"] = [item for item in allow if item != PLUGIN_ID]
            if not plugins["allow"]:
                plugins.pop("allow", None)

        if not plugins:
            config.pop("plugins", None)

    if config != original:
        _write_config(config)
        return True
    return False
