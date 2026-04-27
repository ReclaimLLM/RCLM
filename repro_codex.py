import os
import sys
from unittest.mock import MagicMock, patch

# Add DC-hooks-proxy to sys.path
sys.path.append(os.getcwd())

from rclm.hooks import codex_handler


def test_post_tool_use_dlp_enabled():
    session_id = "test-session"
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

    with (
        patch("rclm._config.load", return_value=mock_config),
        patch("rclm.hooks.session_store.append_event", mock_session_store.append_event),
        patch("rclm.hooks.dlp.maybe_redact_output", mock_dlp.maybe_redact_output),
        patch("sys.stdout", new_callable=MagicMock) as mock_stdout,
    ):
        codex_handler._handle_post_tool_use(session_id, payload)

        # Verify event appended
        mock_session_store.append_event.assert_called_once()
        args, _ = mock_session_store.append_event.call_args
        assert args[0] == session_id
        assert args[1]["event_type"] == "PostToolUse"
        assert args[1]["tool_response"] == "My secret is password123"

        # Verify dlp called
        mock_dlp.maybe_redact_output.assert_called_once_with(
            "Bash", "My secret is password123", payload["cwd"]
        )

        # Verify output printed
        mock_stdout.write.assert_called()
        output = "".join(call.args[0] for call in mock_stdout.write.call_args_list)
        assert "hookSpecificOutput" in output
        assert "updatedResponse" in output
        assert "[REDACTED:PASSWORD]" in output


if __name__ == "__main__":
    try:
        test_post_tool_use_dlp_enabled()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
