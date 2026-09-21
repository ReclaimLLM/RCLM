import json

from rclm.hooks.task_ledger import LedgerLimits, TaskLedger, TaskLedgerStore, infer_ledger_patch


def test_empty_ledger_has_stable_typed_shape():
    assert TaskLedger().to_dict() == {
        "version": 1,
        "revision": 0,
        "goal": None,
        "constraints": [],
        "decisions": [],
        "files": [],
        "tests": [],
        "errors": [],
        "next_actions": [],
    }


def test_explicit_plan_fields_populate_goal_constraints_decisions_and_pending_actions(tmp_path):
    store = TaskLedgerStore(tmp_path / "ledgers")
    result = store.observe_tool(
        "session",
        "update_plan",
        {
            "goal": "Reduce repeated context",
            "constraints": ["Stay offline", "Stay offline", "Fail open"],
            "decisions": ["Use content hashes"],
            "plan": [
                {"step": "Inspect handlers", "status": "completed"},
                {"step": "Add artifact store", "status": "in_progress"},
                {"step": "Run replay", "status": "pending"},
            ],
        },
        {},
        tool_use_id="plan-1",
    )

    assert result.persisted
    assert result.ledger.goal == "Reduce repeated context"
    assert result.ledger.constraints == ("Stay offline", "Fail open")
    assert result.ledger.decisions == ("Use content hashes",)
    assert result.ledger.next_actions == ("Add artifact store", "Run replay")
    assert store.read("session") == result.ledger


def test_known_file_semantics_update_one_record_per_path(tmp_path):
    store = TaskLedgerStore(tmp_path / "ledgers")
    store.observe_tool(
        "session",
        "view_file",
        {"file_path": "src/app.py"},
        "content",
        tool_use_id="read-1",
        artifact_handle="rclm://artifact/sha256/" + "a" * 64,
        cwd="/repo",
    )
    result = store.observe_tool(
        "session",
        "replace_file_content",
        {"file_path": "src/app.py", "old_string": "a", "new_string": "b"},
        {"exit_code": 0},
        tool_use_id="edit-1",
        cwd="/repo",
    )

    assert len(result.ledger.files) == 1
    file_state = result.ledger.files[0]
    assert file_state.path == "/repo/src/app.py"
    assert file_state.operation == "modified"
    assert file_state.tool_name == "replace_file_content"
    assert file_state.tool_use_id == "edit-1"


def test_apply_patch_extracts_each_explicit_file_path(tmp_path):
    store = TaskLedgerStore(tmp_path / "ledgers")
    result = store.observe_tool(
        "session",
        "apply_patch",
        {"patch": "*** Begin Patch\n*** Update File: a.py\n*** Add File: src/b.py\n*** End Patch"},
        "Done!",
        cwd="/repo",
    )
    assert [item.path for item in result.ledger.files] == ["/repo/a.py", "/repo/src/b.py"]


def test_camel_case_provider_tool_names_are_canonicalized(tmp_path):
    store = TaskLedgerStore(tmp_path / "ledgers")
    store.observe_tool(
        "session",
        "NotebookEdit",
        {"notebook_path": "analysis.ipynb"},
        "updated",
    )
    result = store.observe_tool(
        "session",
        "TodoWrite",
        {"todos": [{"content": "Verify replay", "status": "pending"}]},
        {},
    )

    assert result.ledger.files[0].operation == "modified"
    assert result.ledger.next_actions == ("Verify replay",)


def test_recognized_test_command_records_status_summary_and_failure(tmp_path):
    store = TaskLedgerStore(tmp_path / "ledgers")
    passed = store.observe_tool(
        "session",
        "exec_command",
        {"cmd": "uv run pytest tests/test_store.py -q"},
        {"stdout": "......\n6 passed in 0.2s\n", "exit_code": 0},
        tool_use_id="test-1",
    )
    assert passed.ledger.tests[0].status == "passed"
    assert passed.ledger.tests[0].summary == "6 passed in 0.2s"

    failed = store.observe_tool(
        "session",
        "run_command",
        {"command": "npm test"},
        {"stderr": "AssertionError: expected 2", "exit_code": 1},
        tool_use_id="test-2",
        artifact_handle="rclm://artifact/sha256/" + "b" * 64,
    )
    assert failed.ledger.tests[-1].status == "failed"
    assert failed.ledger.errors[-1].message == "AssertionError: expected 2"
    assert failed.ledger.errors[-1].artifact_handle.endswith("b" * 64)


def test_explicit_structured_error_is_recorded_but_error_words_are_not_inferred(tmp_path):
    store = TaskLedgerStore(tmp_path / "ledgers")
    clean = store.observe_tool(
        "session",
        "custom_tool",
        {},
        "Documentation discusses error handling",
    )
    assert clean.ledger.errors == ()

    failed = store.observe_tool(
        "session",
        "custom_tool",
        {},
        {"isError": True, "content": [{"type": "text", "text": "permission denied"}]},
        tool_use_id="call-2",
    )
    assert failed.ledger.errors[0].source == "custom_tool"
    assert failed.ledger.errors[0].message == "permission denied"


def test_failed_write_does_not_claim_file_was_modified(tmp_path):
    store = TaskLedgerStore(tmp_path / "ledgers")
    result = store.observe_tool(
        "session",
        "write_file",
        {"file_path": "unchanged.py"},
        {"error": "permission denied"},
    )

    assert result.ledger.files == ()
    assert result.ledger.errors[0].message == "permission denied"


def test_state_is_bounded_and_newest_records_win(tmp_path):
    limits = LedgerLimits(max_files=2, max_decisions=2, max_item_chars=20)
    store = TaskLedgerStore(tmp_path / "ledgers", limits=limits)
    for index in range(3):
        store.observe_tool(
            "session",
            "write_file",
            {"file_path": f"file-{index}.py"},
            {"exit_code": 0},
        )
        store.observe_tool(
            "session",
            "update_plan",
            {"decisions": [f"decision-{index}"], "plan": []},
            {},
        )

    ledger = store.read("session")
    assert [item.path for item in ledger.files] == ["file-1.py", "file-2.py"]
    assert ledger.decisions == ("decision-1", "decision-2")


def test_zero_collection_limits_retain_no_entries(tmp_path):
    limits = LedgerLimits(max_files=0, max_decisions=0, max_next_actions=0)
    store = TaskLedgerStore(tmp_path / "ledgers", limits=limits)
    store.observe_tool(
        "session",
        "write_file",
        {"file_path": "file.py"},
        {"exit_code": 0},
    )
    result = store.observe_tool(
        "session",
        "update_plan",
        {"decisions": ["decision"], "plan": ["next"]},
        {},
    )

    assert result.ledger.files == ()
    assert result.ledger.decisions == ()
    assert result.ledger.next_actions == ()


def test_atomic_write_failure_fails_open_and_keeps_previous_state(tmp_path, monkeypatch):
    store = TaskLedgerStore(tmp_path / "ledgers")
    first = store.observe_tool(
        "session",
        "write_file",
        {"file_path": "kept.py"},
        {"exit_code": 0},
    )
    assert first.persisted

    monkeypatch.setattr(store, "_write", lambda *_args: (_ for _ in ()).throw(OSError()))
    failed = store.observe_tool(
        "session",
        "write_file",
        {"file_path": "lost.py"},
        {"exit_code": 0},
    )

    assert not failed.persisted
    assert [item.path for item in failed.ledger.files] == ["kept.py"]
    assert [item.path for item in store.read("session").files] == ["kept.py"]


def test_serialization_is_deterministic_for_identical_observations(tmp_path):
    first = TaskLedgerStore(tmp_path / "one")
    second = TaskLedgerStore(tmp_path / "two")
    observation = (
        "update_plan",
        {"goal": "Ship", "constraints": ["offline"], "plan": ["test"]},
        {},
    )
    first.observe_tool("session", *observation)
    second.observe_tool("session", *observation)

    first_json = first._path("session").read_text()
    second_json = second._path("session").read_text()
    assert first_json == second_json
    assert json.loads(first_json)["next_actions"] == ["test"]


def test_antigravity_paths_and_test_command_are_typed() -> None:
    file_patch = infer_ledger_patch(
        "replace_file_content",
        {"TargetFile": "/repo/app.py"},
        "edit complete",
    )
    test_patch = infer_ledger_patch(
        "run_command",
        {"CommandLine": "pytest -q"},
        "The command exited with code 0.\nOutput:\n10 passed",
    )

    assert file_patch.files[0].path == "/repo/app.py"
    assert file_patch.files[0].operation == "modified"
    assert test_patch.tests[0].status == "passed"
