"""Reject observed r10 protocol conflicts before any model gate."""

import json
from pathlib import Path

import pytest

from skill3d.evolution.fitset_revision_v11 import (
    FitSetSlotReviser, _safe_static_feedback, _static_validate,
    load_active_parent_slots,
)
from skill3d.synthesis.execution_contract import ExecutionContract, review_method_contract
from skill3d.synthesis.prompt_builder import PromptBuilder

LIBRARY = Path(__file__).resolve().parents[2] / "skill_library"


@pytest.mark.parametrize(("text", "code"), [
    ("Generate the final answer without extraneous text, markdown code blocks, or trailing whitespace.", "execution_contract_conflict"),
    ("`selected_object_count` must be explicitly set to the final integer answer.", "telemetry_as_api"),
    ("`selected_object_count` must be > 0 before finalizing any answer.", "telemetry_as_api"),
    ("If no clear visual separation is visible, abstain rather than guessing.", "execution_contract_conflict"),
    ("Use the `connectivity_graph` output to determine the number of free-space cells.", "return_schema_mismatch"),
])
def test_known_r10_contract_conflicts_have_line_and_rule(text, code):
    issues = review_method_contract("# Method\n" + text)
    assert any(issue.code == code and issue.line == 2 and issue.rule for issue in issues)


@pytest.mark.parametrize("text", [
    "JSON is the internal tool representation.",
    "Do not use metric tools without scale; inspect_frames is optional.",
    "Never write selected_object_count or require nonzero telemetry.",
    "Do not abstain; give a visual estimate.",
    "The connectivity_graph does not return the number of free-space cells.",
    "Output one Python code block with no explanation outside the block.",
])
def test_legal_negative_and_conditional_advice_is_not_rejected(text):
    assert review_method_contract(text) == []


def test_same_contract_hash_is_rendered_online_and_offline(tmp_path):
    parent = load_active_parent_slots(LIBRARY)[0][("object_counting", "g0m0")]
    reviser = _reviser(tmp_path, None)
    contract = ExecutionContract(8000)
    online = PromptBuilder().render(
        question="Count", scene_summary="", scene_frame="", tool_docs="")
    offline = reviser._prompt(parent=parent, feedback={}, static_feedback=[])
    assert contract.sha256 in online and contract.sha256 in offline
    assert "8000" in offline and "no_change" in offline


def test_first_batch_reuses_parent_text_but_later_batch_carries_changed_working_method(tmp_path):
    parent = load_active_parent_slots(LIBRARY)[0][("object_counting", "g0m0")]
    obj = _reviser(tmp_path, None)
    first = obj._prompt(parent=parent, feedback={}, static_feedback=[],
                        batch_context={"working_skill_md": parent.skill_md})
    assert first.count(parent.skill_md) == 1
    assert "Same as Parent complete SKILL.md above." in first
    changed = parent.skill_md.rstrip() + "\n\nSynthetic changed rule.\n"
    later = obj._prompt(parent=parent, feedback={}, static_feedback=[],
                        batch_context={"working_skill_md": changed})
    assert "Working candidate complete SKILL.md:\n" + changed in later
    assert "Same as Parent complete SKILL.md above." not in later
    assert first.count("Return one JSON object with action=") == 1


def test_compact_feedback_preserves_all_process_values_and_signature_refs():
    from copy import deepcopy
    from skill3d.evolution.revision_view_v11 import compact_slot_feedback
    from skill3d.evolution.process_feedback import ProcessFacts

    process = ProcessFacts().model_dump(mode="json")
    modified = deepcopy(process)
    modified["execution"]["model_rounds"] = 3
    modified["error_chain"] = [{"round_index": 2, "event_type": "runtime_error"}]
    feedback = {
        "n_cases": 12, "mean_score": 0.25, "score_bands": {"full_score": 3, "no_answer": 9},
        "signatures": [
            {"n": 3, "signature_ref": "a" * 64, "process": process},
            {"n": 9, "signature_ref": "b" * 64, "process": modified}]}
    before = deepcopy(feedback)
    packed = compact_slot_feedback(feedback)

    def overlay(base, delta):
        merged = deepcopy(base)
        for key, child in delta.items():
            merged[key] = (overlay(merged[key], child)
                           if isinstance(child, dict) and isinstance(merged.get(key), dict)
                           else child)
        return merged

    for original, row in zip(feedback["signatures"], packed["signatures"]):
        assert overlay(packed["shared_process"], row["process"]) == original["process"]
        assert row["signature_ref"] == original["signature_ref"]
        assert row["n"] == original["n"]
    assert packed["mean_score"] == 0.25 and packed["n_cases"] == 12
    assert feedback == before


def _reviser(tmp_path, client):
    return FitSetSlotReviser(
        output_root=tmp_path, panel={}, panel_name="fixture", variants=[],
        accumulation_root=tmp_path, label_metadata={}, library_root=LIBRARY,
        client=client)


class NoChangeClient:
    calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return json.dumps({"action": "no_change", "modification_reason": "No supported method defect."})


def test_no_change_is_one_request_and_resumes_with_identity_check(tmp_path):
    client = NoChangeClient()
    reviser = _reviser(tmp_path, client)
    parent = load_active_parent_slots(LIBRARY)[0][("object_counting", "g0m0")]
    args = dict(slot=("object_counting", "g0m0"), parent=parent,
                feedback={"n_cases": 12}, forbidden_ids=set())
    assert reviser._revise_slot(**args) is None
    assert reviser._revise_slot(**args) is None
    assert client.calls == 1
    assert not list(tmp_path.rglob("SKILL.md"))
    with pytest.raises(Exception, match="identity changed"):
        reviser._revise_slot(**{**args, "feedback": {"n_cases": 13}})


def test_safe_retry_explains_length_and_identity_without_echoing_private_details():
    parent = load_active_parent_slots(LIBRARY)[0][("object_counting", "g0m0")]
    _, problems = _static_validate(
        parent=parent, candidate_text=parent.skill_md,
        forbidden_sample_ids=set(), method_context_max_chars=10)
    safe = _safe_static_feedback(
        problems + ["candidate matched leakage patterns: qa-secret reference_answer=937.777"],
        candidate=parent, max_chars=10)
    assert any(row["code"] == "candidate_identical" for row in safe)
    assert any(row.get("actual_chars") == len(parent.skill_md) for row in safe)
    assert "937.777" not in json.dumps(safe)
    assert "qa-secret" not in json.dumps(safe)
