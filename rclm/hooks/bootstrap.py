"""Single-request SessionStart bootstrap shared by hook providers."""

from __future__ import annotations

from datetime import date

from rclm import _config
from rclm.hooks.updater import installed_version
from rclm.mcp_server import ReclaimLLMClient, ReclaimLLMError

POLICY_SNAPSHOT_SCHEMA_VERSION = 1


async def fetch(cwd: str, *, include_context: bool) -> dict:
    try:
        result = await ReclaimLLMClient().hook_bootstrap(
            cwd=cwd or None, include_context=include_context
        )
    except ReclaimLLMError:
        return {}
    if "org_hook_policy" in result:
        policy = result.get("org_hook_policy")
        _config.patch(org_hook_policy=policy if isinstance(policy, dict) else None)
    return result


def context_text(result: dict) -> str | None:
    sessions = result.get("context_sessions")
    if not isinstance(sessions, list) or not sessions:
        return None
    lines = ["[rclm] Recent sessions in this project (via ReclaimLLM):"]
    for session in sessions:
        if not isinstance(session, dict):
            continue
        title = session.get("title") or "Untitled session"
        summary = session.get("session_summary") or ""
        lines.append(f"- {title}" + (f" — {summary}" if summary else ""))
    return "\n".join(lines) if len(lines) > 1 else None


def signal_nudge_text(result: dict, cfg: dict) -> str | None:
    signal = result.get("signal")
    if not isinstance(signal, dict) or not signal.get("signal_id"):
        return None
    today = str(result.get("date") or date.today().isoformat())
    signal_id = str(signal["signal_id"])
    if cfg.get("last_signal_nudge_date") == today and cfg.get("last_signal_nudge_id") == signal_id:
        return None
    meta = signal.get("pattern_meta")
    meta = meta if isinstance(meta, dict) else {}
    label = meta.get("label") or signal.get("pattern") or "Workflow signal"
    action = meta.get("fix") or meta.get("admin_action") or "Review this signal in ReclaimLLM."
    _config.patch(last_signal_nudge_date=today, last_signal_nudge_id=signal_id)
    return f"[rclm] Signal: {label}. {action}"


def policy_snapshot(provider: str) -> dict:
    snapshot = _config.effective_hook_policy(_config.load(), provider=provider).snapshot()
    snapshot.update(
        {
            "snapshot_schema_version": POLICY_SNAPSHOT_SCHEMA_VERSION,
            "capture_client_version": installed_version(),
            "capture_provider": provider,
        }
    )
    return snapshot


def policy_snapshot_from_events(events: list[dict], provider: str) -> dict:
    for event in events:
        if event.get("event_type") == "HookPolicySnapshot" and isinstance(
            event.get("policy"), dict
        ):
            return event["policy"]
    return policy_snapshot(provider)
