"""Tests for the cross-session outline served on re-reads of known files."""

import json
from io import StringIO

import pytest

from rclm import _config
from rclm.hooks import claude_handler as handler
from rclm.hooks import cross_session_outline, session_store
from rclm.hooks.cross_session_outline import MECHANISM, build_outline


def _python_module(functions: int = 40) -> str:
    body = "".join(
        f"    step_{n} = transform(event, {n})  # normalise field {n}\n" for n in range(10)
    )
    blocks = [f"def handler_{i}(event):\n{body}    return event\n" for i in range(functions)]
    return "import os\n\n\n" + "\n".join(blocks)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(cross_session_outline, "_STORE_ROOT", tmp_path / "project_reads")
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    target = repo / "service.py"
    target.write_text(_python_module())

    def configure(mode: str) -> None:
        config_path.write_text(json.dumps({"read_cache": True, "cross_session_outline": mode}))

    configure("enforce")
    return {"repo": repo, "target": target, "configure": configure}


def _read(monkeypatch, session_id: str, target, *, start: int = 1, lines: int = 200) -> dict | None:
    all_lines = target.read_text().splitlines(keepends=True)
    content = "".join(all_lines[start - 1 : start - 1 + lines])
    tool_input: dict = {"file_path": str(target)}
    if start != 1 or lines < len(all_lines):
        # rclm's read window uses a zero-based offset (see native_read_request).
        tool_input.update({"offset": start - 1, "limit": lines})
    payload = {
        "session_id": session_id,
        "tool_name": "Read",
        "tool_input": tool_input,
        "tool_response": {
            "type": "text",
            "file": {
                "filePath": str(target),
                "content": content,
                "numLines": content.count("\n"),
                "startLine": start,
                "totalLines": len(all_lines),
            },
        },
        "tool_use_id": f"tool-{session_id}-{start}",
        "cwd": str(target.parent),
    }
    stdout = StringIO()
    monkeypatch.setattr("sys.argv", ["rclm-claude-hooks", "PostToolUse"])
    monkeypatch.setattr("sys.stdin", StringIO(json.dumps(payload)))
    monkeypatch.setattr("sys.stdout", stdout)
    with pytest.raises(SystemExit):
        handler.main()
    return json.loads(stdout.getvalue()) if stdout.getvalue().strip() else None


def _outline_events(session_id: str) -> list[dict]:
    return [
        event
        for event in session_store.read_events(session_id)
        if event.get("event_type") == "MechanismSaving" and event.get("mechanism") == MECHANISM
    ]


def test_build_outline_lists_definitions_with_file_line_numbers():
    entries = build_outline(_python_module(5), "service.py")

    assert entries[0] == "L4: def handler_0(event):"
    assert len(entries) == 5


def test_build_outline_skips_unsupported_or_sparse_files():
    assert build_outline("a\nb\nc\n", "notes.txt") is None
    assert build_outline("def only_one():\n    pass\n", "tiny.py") is None


def test_first_session_read_is_untouched(env, monkeypatch):
    assert _read(monkeypatch, "sid-one", env["target"]) is None
    assert _outline_events("sid-one") == []


def test_later_session_gets_outline_and_net_savings(env, monkeypatch):
    _read(monkeypatch, "sid-one", env["target"])

    output = _read(monkeypatch, "sid-two", env["target"])

    updated = output["hookSpecificOutput"]["updatedToolOutput"]
    outline = updated["file"]["content"]
    assert outline.startswith("[RCLM] service.py (")
    assert "L4: def handler_0(event):" in outline
    served = _outline_events("sid-two")
    assert len(served) == 1 and served[0]["applied"] is True
    assert served[0]["tokens_saved_estimate"] > 0
    # The range cache must not believe the model saw the outlined lines.
    assert not session_store.read_read_cache_state("sid-two").get("files")

    # A follow-up ranged read passes through and is charged against the saving.
    assert _read(monkeypatch, "sid-two", env["target"], start=40, lines=30) is None
    follow_up = _outline_events("sid-two")[-1]
    assert follow_up["applied"] is True
    assert follow_up["tokens_saved_estimate"] < 0


def test_reread_in_same_session_is_not_outlined(env, monkeypatch):
    _read(monkeypatch, "sid-one", env["target"])
    _read(monkeypatch, "sid-two", env["target"], start=1, lines=10)

    # Already read in this session (even partially): the agent wants content.
    output = _read(monkeypatch, "sid-two", env["target"])
    assert output is None or "[RCLM] service.py (" not in json.dumps(output)


def test_changed_file_is_not_outlined(env, monkeypatch):
    _read(monkeypatch, "sid-one", env["target"])
    env["target"].write_text(_python_module() + "\n# edited\n")

    assert _read(monkeypatch, "sid-two", env["target"]) is None
    assert _outline_events("sid-two") == []


def test_observe_mode_measures_without_changing_output(env, monkeypatch):
    env["configure"]("observe")
    _read(monkeypatch, "sid-one", env["target"])

    assert _read(monkeypatch, "sid-two", env["target"]) is None
    served = _outline_events("sid-two")
    assert len(served) == 1 and served[0]["applied"] is False
    # In observe mode the range cache still tracks the delivered lines.
    assert session_store.read_read_cache_state("sid-two").get("files")


def test_outline_is_off_when_configured_off(env, monkeypatch):
    env["configure"]("off")
    _read(monkeypatch, "sid-one", env["target"])

    assert _read(monkeypatch, "sid-two", env["target"]) is None
    assert _outline_events("sid-two") == []


def test_outline_defaults_to_observe_even_when_org_forces_enforcement(env, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"read_cache": True, "org_hook_policy": {"force_compression_enforcement": True}})
    )

    policy = _config.effective_hook_policy(provider="claude")

    assert policy.enabled("cross_session_outline")
    assert policy.shadow_for("cross_session_outline")
    assert not policy.shadow_for("range_cache")
