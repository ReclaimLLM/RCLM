import json
import os
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

sys.path.append(os.getcwd())
from rclm.hooks import codex_handler

SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "additionalProperties": False,
    "definitions": {
        "BlockDecisionWire": {"enum": ["block"], "type": "string"},
        "HookEventNameWire": {
            "enum": [
                "PreToolUse",
                "PermissionRequest",
                "PostToolUse",
                "SessionStart",
                "UserPromptSubmit",
                "Stop",
            ],
            "type": "string",
        },
        "PostToolUseHookSpecificOutputWire": {
            "additionalProperties": False,
            "properties": {
                "additionalContext": {"default": None, "type": "string"},
                "hookEventName": {"$ref": "#/definitions/HookEventNameWire"},
                "updatedMCPToolOutput": {"default": None},
            },
            "required": ["hookEventName"],
            "type": "object",
        },
    },
    "properties": {
        "continue": {"default": True, "type": "boolean"},
        "decision": {"allOf": [{"$ref": "#/definitions/BlockDecisionWire"}], "default": None},
        "hookSpecificOutput": {
            "allOf": [{"$ref": "#/definitions/PostToolUseHookSpecificOutputWire"}],
            "default": None,
        },
        "reason": {"default": None, "type": "string"},
        "stopReason": {"default": None, "type": "string"},
        "suppressOutput": {"default": False, "type": "boolean"},
        "systemMessage": {"default": None, "type": "string"},
    },
    "title": "post-tool-use.command.output",
    "type": "object",
}


def test_codex_post_tool_use_schema():
    session_id = "schema-test-session"
    payload = {
        "session_id": session_id,
        "cwd": os.getcwd(),
        "tool_response": "My secret is password123",
        "turn_id": "turn-1",
    }

    mock_config = MagicMock()
    mock_config.get.side_effect = lambda k, d=None: True if k == "dlp" else d

    mock_session_store = MagicMock()

    mock_dlp = MagicMock()
    mock_dlp.maybe_redact_output.return_value = "My secret is [REDACTED:PASSWORD]"

    stdin_content = json.dumps(payload)

    with (
        patch("sys.argv", ["rclm-codex-hooks", "PostToolUse"]),
        patch("sys.stdin", StringIO(stdin_content)),
        patch("rclm._config.load", return_value=mock_config),
        patch("rclm.hooks.session_store.append_event", mock_session_store.append_event),
        patch("rclm.hooks.dlp.maybe_redact_output", mock_dlp.maybe_redact_output),
        patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        patch("sys.exit"),
    ):
        codex_handler.main()

        output = mock_stdout.getvalue().strip()
        print(f"Captured stdout: {output}")

        parsed_output = json.loads(output)

        try:
            import jsonschema

            jsonschema.validate(instance=parsed_output, schema=SCHEMA)
            print("✅ SUCCESS: The output payload strictly matches the Codex PostToolUse schema.")
        except ImportError:
            print("⚠️ jsonschema not installed, doing manual validation...")
            # Manual assertions mirroring schema
            assert "hookSpecificOutput" in parsed_output
            hso = parsed_output["hookSpecificOutput"]
            assert "hookEventName" in hso
            assert hso["hookEventName"] == "PostToolUse"
            # Ensure no unsupported keys
            for k in hso:
                assert k in ["hookEventName", "additionalContext", "updatedMCPToolOutput"], (
                    f"Invalid key: {k}"
                )
            print("✅ SUCCESS: Manual schema validation passed.")


if __name__ == "__main__":
    test_codex_post_tool_use_schema()
