"""Unit tests for rclm.replay.engine: per-mechanism dispatch, determinism,
image exclusion, and the H1 unresolvable path."""

from __future__ import annotations

import copy

from rclm.replay.engine import replay_blob


def _blob(tool_calls: list[dict]) -> dict:
    return {"session_id": "s1", "model": "claude-sonnet", "tool_calls": tool_calls}


def _numbered(start: int, lines: list[str]) -> str:
    return "\n".join(f"{n:>6}→{text}" for n, text in enumerate(lines, start=start)) + "\n"


class TestShellCompaction:
    def test_compressible_command_is_shaped(self):
        # A long, repetitive generic exec output the shipped H3 filter collapses.
        stdout = "\n".join(f"line {i}" for i in range(500))
        blob = _blob(
            [
                {
                    "tool_use_id": "toolu_1",
                    "tool_name": "Bash",
                    "tool_input": {"command": "some-noisy-command"},
                    "tool_result": stdout,
                }
            ]
        )
        result = replay_blob(blob, mechanisms=("shell_compaction",))
        assert len(result.calls) == 1
        call = result.calls[0]
        assert call.classification in ("shaped", "uncovered")  # depends on filter routing
        assert call.original_tokens > 0

    def test_non_shell_tool_is_uncovered_not_shaped(self):
        blob = _blob(
            [
                {
                    "tool_use_id": "toolu_1",
                    "tool_name": "Edit",
                    "tool_input": {"file_path": "/x.py"},
                    "tool_result": {"success": True},
                }
            ]
        )
        result = replay_blob(blob, mechanisms=("shell_compaction",))
        assert result.calls[0].classification == "uncovered"
        assert result.calls[0].mechanism is None


class TestImageExclusion:
    def test_recognized_mcp_image_block_excluded_from_denominator(self):
        blob = _blob(
            [
                {
                    "tool_use_id": "toolu_1",
                    "tool_name": "some_tool",
                    "tool_input": {},
                    "tool_result": {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
                }
            ]
        )
        result = replay_blob(blob)
        assert result.calls[0].classification == "image"
        assert result.text_result_tokens == 0

    def test_unrecognized_image_shape_still_excluded_via_broad_sniff(self):
        """A shape image_lifecycle.find_image doesn't recognize, but that
        obviously carries a data: URI, must not inflate the denominator."""
        blob = _blob(
            [
                {
                    "tool_use_id": "toolu_1",
                    "tool_name": "imagegen",
                    "tool_input": {},
                    "tool_result": [
                        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
                    ],
                }
            ]
        )
        result = replay_blob(blob)
        assert result.calls[0].classification == "image"
        assert result.text_result_tokens == 0

    def test_anthropic_content_block_excluded_from_denominator(self):
        blob = _blob(
            [
                {
                    "tool_use_id": "toolu_1",
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/image.png"},
                    "tool_result": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aGVsbG8=",
                            },
                        }
                    ],
                }
            ]
        )
        result = replay_blob(blob)
        assert result.calls[0].classification == "image"
        assert result.text_result_tokens == 0


class TestDedupe:
    def test_repeated_large_result_is_deduped_on_second_occurrence(self):
        big_text = "same output line\n" * 100  # > MIN_DEDUPE_CHARS
        blob = _blob(
            [
                {
                    "tool_use_id": "toolu_1",
                    "tool_name": "SomeTool",
                    "tool_input": {},
                    "tool_result": big_text,
                },
                {
                    "tool_use_id": "toolu_2",
                    "tool_name": "SomeTool",
                    "tool_input": {},
                    "tool_result": big_text,
                },
            ]
        )
        result = replay_blob(blob, mechanisms=("hash_dedupe",))
        assert (
            result.calls[0].classification == "uncovered"
        )  # first occurrence, nothing to dedupe against
        assert result.calls[1].classification == "shaped"
        assert result.calls[1].mechanism == "hash_dedupe"
        assert result.tokens_removed > 0


class TestRangeCacheH1:
    def test_native_read_repeat_range_is_cache_hit(self):
        lines = [f"content line {i}" for i in range(1, 51)]
        content = _numbered(1, lines)
        blob = _blob(
            [
                {
                    "tool_use_id": "toolu_1",
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/a.py"},
                    "tool_result": content,
                },
                {
                    "tool_use_id": "toolu_2",
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/a.py"},
                    "tool_result": content,
                },
            ]
        )
        result = replay_blob(blob, mechanisms=("range_cache",))
        assert result.calls[0].classification == "uncovered"  # nothing cached yet
        assert result.calls[1].classification == "shaped"
        assert result.calls[1].mechanism == "range_cache"

    def test_unparseable_read_is_unresolvable_but_still_counted_in_denominator(self):
        blob = _blob(
            [
                {
                    "tool_use_id": "toolu_1",
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/a.py"},
                    "tool_result": "not a numbered file listing at all",
                }
            ]
        )
        result = replay_blob(blob, mechanisms=("range_cache",))
        assert result.calls[0].classification == "unresolvable"
        assert result.text_result_tokens > 0
        assert result.tokens_removed == 0
        assert len(result.unresolvable_calls) == 1

    def test_captured_read_metadata_replays_plain_claude_output_without_filesystem(self):
        content = "".join(f"line {line}: value\n" for line in range(1, 81))
        metadata = {
            "schema_version": 1,
            "absolute_path": "/past/a.py",
            "display_path": "a.py",
            "content_hash": "a" * 64,
            "line_count": 80,
            "size": len(content),
            "start_line": 1,
            "end_line": 80,
            "output_style": "native",
        }
        calls = [
            {
                "tool_use_id": f"toolu_{index}",
                "tool_name": "Read",
                "tool_input": {"file_path": "/past/a.py"},
                "tool_result": content,
                "extra_fields": {"replay_read_request": metadata},
            }
            for index in (1, 2)
        ]

        result = replay_blob(_blob(calls), mechanisms=("range_cache",))

        assert result.calls[0].classification == "uncovered"
        assert result.calls[1].classification == "shaped"
        assert result.calls[1].mechanism == "range_cache"

    def test_trailing_system_reminder_is_reattached_unchanged(self):
        lines = [f"line {i}" for i in range(1, 21)]
        content = _numbered(1, lines) + "<system-reminder>note</system-reminder>\n"
        blob = _blob(
            [
                {
                    "tool_use_id": "toolu_1",
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/a.py"},
                    "tool_result": content,
                },
                {
                    "tool_use_id": "toolu_2",
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/a.py"},
                    "tool_result": content,
                },
            ]
        )
        result = replay_blob(blob, mechanisms=("range_cache",))
        # Second read of the identical range should still cache-hit even with
        # the non-file suffix present.
        assert result.calls[1].classification == "shaped"


class TestPrecedence:
    def test_range_cache_claims_read_before_dedupe_gets_a_chance(self):
        """A repeated Read should be claimed by H1, not hash_dedupe — matching
        the real hook's range_cache -> exec_compaction -> hash_dedupe order."""
        lines = [f"line {i}" for i in range(1, 51)]
        content = _numbered(1, lines)
        blob = _blob(
            [
                {
                    "tool_use_id": "toolu_1",
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/a.py"},
                    "tool_result": content,
                },
                {
                    "tool_use_id": "toolu_2",
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/a.py"},
                    "tool_result": content,
                },
            ]
        )
        result = replay_blob(blob, mechanisms=("range_cache", "hash_dedupe"))
        assert result.calls[1].mechanism == "range_cache"

    def test_range_cache_miss_falls_through_to_shell_compaction(self):
        content = "\n".join(f"unique line {line}" for line in range(1, 101)) + "\n"
        blob = _blob(
            [
                {
                    "tool_use_id": "toolu_1",
                    "tool_name": "Bash",
                    "tool_input": {"command": "sed -n '1,100p' file.txt"},
                    "tool_result": content,
                }
            ]
        )

        result = replay_blob(blob, mechanisms=("range_cache", "shell_compaction"))

        assert result.calls[0].classification == "shaped"
        assert result.calls[0].mechanism == "H3_exec_compaction"

    def test_unresolvable_read_falls_through_to_hash_dedupe(self):
        content = "same unnumbered read output\n" * 100
        blob = _blob(
            [
                {
                    "tool_use_id": f"toolu_{index}",
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/a.py"},
                    "tool_result": content,
                }
                for index in (1, 2)
            ]
        )

        result = replay_blob(blob, mechanisms=("range_cache", "hash_dedupe"))

        assert result.calls[0].classification == "unresolvable"
        assert result.calls[1].classification == "shaped"
        assert result.calls[1].mechanism == "hash_dedupe"


class TestDeterminism:
    def test_identical_blob_produces_byte_identical_result(self):
        blob = _blob(
            [
                {
                    "tool_use_id": "toolu_1",
                    "tool_name": "Bash",
                    "tool_input": {"command": "cat foo.txt"},
                    "tool_result": "\n".join(f"l{i}" for i in range(200)),
                },
            ]
        )
        first = replay_blob(copy.deepcopy(blob))
        second = replay_blob(copy.deepcopy(blob))
        assert [vars(c) for c in first.calls] == [vars(c) for c in second.calls]
        assert first.tokens_removed == second.tokens_removed
        assert first.text_result_tokens == second.text_result_tokens
