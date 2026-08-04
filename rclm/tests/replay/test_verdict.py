"""Tests for rclm.replay.verdict: thresholds, output assembly, honesty rules."""

from __future__ import annotations

import json

from rclm.replay.eligibility import EligibilityResult, build_funnel
from rclm.replay.engine import replay_blob
from rclm.replay.provenance import build_provenance
from rclm.replay.verdict import (
    build_insufficient_data_output,
    build_result_output,
    compute_verdict,
    infer_source,
)


class TestVerdictThresholds:
    def test_helps_requires_both_pct_and_coverage(self):
        assert compute_verdict(20.0, 30.0) == "helps"

    def test_high_pct_but_low_coverage_is_marginal_not_helps(self):
        assert compute_verdict(20.0, 15.0) == "marginal"

    def test_marginal_band(self):
        assert compute_verdict(7.0, 50.0) == "marginal"

    def test_low_pct_is_no_effect(self):
        assert compute_verdict(2.0, 50.0) == "no_effect"

    def test_low_coverage_is_no_effect_regardless_of_pct(self):
        assert compute_verdict(40.0, 5.0) == "no_effect"

    def test_boundary_helps_exact_thresholds(self):
        assert compute_verdict(15.0, 25.0) == "helps"

    def test_boundary_just_under_helps_pct_is_marginal(self):
        assert compute_verdict(14.99, 30.0) == "marginal"


class TestInferSource:
    def test_claude_from_tool_use_id_absence_falls_back_to_model(self):
        blob = {"model": "claude-sonnet-4-5", "tool_calls": [{"tool_use_id": "toolu_abc"}]}
        assert infer_source(blob) == "claude"

    def test_codex_from_tool_use_id_prefix(self):
        blob = {"tool_calls": [{"tool_use_id": "call_123"}]}
        assert infer_source(blob) == "codex"

    def test_unattributed_when_nothing_matches(self):
        blob = {"tool_calls": [{"tool_use_id": "toolu_xyz"}]}
        assert infer_source(blob) == "unattributed"


class TestProvenanceMinTurns:
    def test_default_min_turns_is_the_prd_floor(self):
        assert build_provenance(("shell_compaction",))["min_turns_applied"] == 10

    def test_min_turns_override_is_recorded(self):
        provenance = build_provenance(("shell_compaction",), min_turns=5)
        assert provenance["min_turns_applied"] == 5

    def test_default_min_tool_calls_is_the_prd_floor(self):
        assert build_provenance(("shell_compaction",))["min_tool_calls_applied"] == 10

    def test_min_tool_calls_override_is_recorded(self):
        provenance = build_provenance(("shell_compaction",), min_tool_calls=1)
        assert provenance["min_tool_calls_applied"] == 1


class TestOutputAssembly:
    def _blob(self):
        return {
            "model": "claude-sonnet-4-5",
            "tool_calls": [
                {
                    "tool_use_id": "toolu_1",
                    "tool_name": "Bash",
                    "tool_input": {"command": "cat foo.txt"},
                    "tool_result": "\n".join(f"line {i}" for i in range(500)),
                }
            ],
        }

    def test_result_output_shape_matches_prd_section_9(self):
        blob = self._blob()
        result = replay_blob(blob, mechanisms=("shell_compaction",))
        funnel = build_funnel(1, {})
        provenance = build_provenance(("shell_compaction",))
        output = build_result_output([(blob, result)], funnel, provenance)

        for key in (
            "verdict",
            "claim_being_verified",
            "eligibility",
            "reduction",
            "coverage",
            "by_source",
            "concentration",
            "unresolvable",
            "reconciliation",
            "cost",
            "provenance",
            "cannot_tell_you",
        ):
            assert key in output, f"missing PRD §9 field: {key}"

        assert output["reduction"]["unit"] == "text tool-result tokens"
        assert output["claim_being_verified"].startswith("replayed tool-result token reduction")

    def test_output_is_json_serializable(self):
        blob = self._blob()
        result = replay_blob(blob)
        funnel = build_funnel(1, {})
        provenance = build_provenance(("shell_compaction",))
        output = build_result_output([(blob, result)], funnel, provenance)
        json.dumps(output)  # must not raise

    def test_insufficient_data_never_carries_a_number(self):
        funnel = build_funnel(1, {"turn_count": 1})
        eligibility = EligibilityResult(False, "turn_count", 3)
        provenance = build_provenance(("shell_compaction",))
        output = build_insufficient_data_output(funnel, eligibility, provenance)
        assert output["verdict"] == "insufficient_data"
        assert "reduction" not in output
        assert output["insufficient_data"] == {"constraint": "turn_count", "actual_value": 3}


class TestHonestyRules:
    def test_savings_field_never_appears_anywhere_in_output(self):
        """PRD §10: field is `reduction`, never `savings`."""
        blob = {
            "model": "claude-sonnet-4-5",
            "tool_calls": [
                {
                    "tool_use_id": "toolu_1",
                    "tool_name": "Bash",
                    "tool_input": {"command": "cat foo.txt"},
                    "tool_result": "\n".join(f"line {i}" for i in range(500)),
                }
            ],
        }
        result = replay_blob(blob)
        funnel = build_funnel(1, {})
        provenance = build_provenance(("shell_compaction",))
        output = build_result_output([(blob, result)], funnel, provenance)
        serialized = json.dumps(output)
        assert '"savings"' not in serialized

    def test_cost_never_available(self):
        blob = {"model": "x", "tool_calls": []}
        result = replay_blob(blob)
        funnel = build_funnel(1, {})
        provenance = build_provenance(("shell_compaction",))
        output = build_result_output([(blob, result)], funnel, provenance)
        assert output["cost"]["available"] is False

    def test_cannot_tell_you_always_present(self):
        funnel = build_funnel(1, {"turn_count": 1})
        eligibility = EligibilityResult(False, "turn_count", 3)
        provenance = build_provenance(("shell_compaction",))
        output = build_insufficient_data_output(funnel, eligibility, provenance)
        assert output["cannot_tell_you"]
