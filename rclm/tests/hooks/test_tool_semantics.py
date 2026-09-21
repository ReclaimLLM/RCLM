from rclm.hooks.tool_semantics import command_text, semantic_target, tool_family


def test_maps_antigravity_tools_and_input_keys() -> None:
    assert tool_family("view_file") == "read"
    assert tool_family("run_command") == "shell"
    assert tool_family("replace_file_content") == "edit"
    assert command_text({"CommandLine": "pytest -q"}) == "pytest -q"
    assert semantic_target("view_file", {"AbsolutePath": "/repo/a.py"}) == "/repo/a.py"


def test_unknown_tool_is_not_inferred() -> None:
    assert tool_family("mystery_execute") is None
    assert semantic_target("mystery_execute", {"path": "/repo/a.py"}) is None
