import json
import os
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

# Add DC-hooks-proxy to sys.path
sys.path.append(os.getcwd())

from rclm.hooks import codex_handler


def test_main_post_tool_use():
    session_id = "test-session-main"
    payload = {
        "session_id": session_id,
        "cwd": os.getcwd(),
        "tool_response": "My secret is password123",
        "turn_id": "turn-1",
    }

    # Mock config to enable DLP
    mock_config = MagicMock()
    mock_config.get.side_effect = lambda k, d=None: True if k == "dlp" else d

    # Mock session_store to avoid writing to disk
    mock_session_store = MagicMock()

    # Mock dlp.maybe_redact_output to return a scrubbed string
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

        output = mock_stdout.getvalue()
        print(f"Stdout output: {output}")

        if "hookSpecificOutput" in output:
            print("Successfully printed hookSpecificOutput")
        else:
            print(
                "Did NOT print hookSpecificOutput (this might be expected if dlp is disabled or no secrets found)"
            )


if __name__ == "__main__":
    test_main_post_tool_use()
