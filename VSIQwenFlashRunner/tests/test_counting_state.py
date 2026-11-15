"""Recovery and image-reference regressions from the counting trajectory."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.online.counting_state import (
    count_tool_status, episode_feedback, validate_instance_evidence, verification_scope,
)
from skill3d.online.recovery import collect_validated
from skill3d.online.vsi_kernel import RestrictedNamespaceKernel
from skill3d.sandbox.ast_guard import FORBIDDEN_CALLS, ast_guard, model_execution_rules
from skill3d.schemas import ToolResult
from skill3d.synthesis.prompt_builder import build_image_messages
from skill3d.tools import REGISTRY


def test_complete_reads_survive_ast_rejection_and_namespace_reset():
    scene = SimpleNamespace(question_type="object_counting", frame="world",
                            scene_route="fallback_2d_only", question_tool_scope="fallback_2d_only",
                            summary="Synthetic scene", available_artifacts=set(), local_facts={})
    kernel = RestrictedNamespaceKernel(REGISTRY, scene, episode_id="recovery")
    rows = [{"obj_id": f"instance-{i}", "visible_frames": [i]} for i in range(12)]
    kernel.tool_results.append(ToolResult(
        tool="list_objects", source_tool="list_objects", result_id="inventory",
        request_digest="synthetic", value=json.dumps(rows)))
    for _ in range(2):
        cell = kernel.run_cell(
            'page = ReadNextObservation("inventory", fields=["obj_id","visible_frames"])\n'
            'YieldObservations([page["result_id"]], "Inspect instances")')
        assert cell.error is None
        kernel.reset_user_namespace()
    assert kernel.vsi_reader_progress[-1]["coverage_complete"]
    assert not ast_guard('getattr({}, "data")').ok  # No execution occurred.
    kernel.reset_user_namespace()
    text = episode_feedback(kernel, collect_validated(kernel))
    assert all(f"instance-{i}" in text for i in range(12))
    assert '"coverage_complete": true' in text and '"covered_ranges": [[0, 12]]' in text
    kernel.unresolved_result_ids = {"inventory"}
    # Online loop passes only non-quarantined observations to the same renderer.
    assert "instance-0" not in episode_feedback(kernel, [])


def test_ast_prompt_contains_every_enforced_forbidden_call():
    rules = model_execution_rules()
    assert all(f'"{name}"' in rules for name in FORBIDDEN_CALLS)
    assert "getattr" in rules and "no calls" in rules


def test_labels_stay_adjacent_when_derived_images_precede_originals():
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    extra = [("crop-1", np.ones((8, 8, 3), dtype=np.uint8))]
    labels = ["image_id=crop-1; source_frame=17", "image_id=orig-17", "image_id=orig-31"]
    parts = build_image_messages("task", frames, extra=extra, image_labels=labels)[0]["content"]
    assert [parts[i]["text"] for i in (1, 3, 5)] == labels
    assert all(parts[i]["type"] == "image_url" for i in (2, 4, 6))
    # Labeling changes no encoded bytes or order.
    plain = build_image_messages("task", frames, extra=extra)[0]["content"][1:]
    assert [parts[i]["image_url"] for i in (2, 4, 6)] == [p["image_url"] for p in plain]


def evidence(image="orig-17", **changes):
    return {"instance_id": "desk", "image_id": image, "region": [100, 100, 800, 900],
            "decision": "include", "reason": "Whole visible object", **changes}


def test_reference_checks_require_current_completed_image_and_valid_regions():
    ledger = SimpleNamespace(current_round=3, rounds=[
        SimpleNamespace(round_index=2, image_ids=["crop-old"], observed=True),
        SimpleNamespace(round_index=3, image_ids=["orig-17", "crop-new"], observed=True)])
    assert validate_instance_evidence([evidence()], ledger)[0]["image_id"] == "orig-17"
    for row in (evidence("crop-old"), evidence("crop-produced-only"),
                evidence(region=[0, 0, 1001, 1000]), evidence(region=[800, 0, 100, 100])):
        with pytest.raises(ValueError):
            validate_instance_evidence([row], ledger)
    ledger.rounds[-1].observed = False
    with pytest.raises(ValueError):
        validate_instance_evidence([evidence("crop-new")], ledger)


def test_merge_must_reference_included_instance():
    ledger = SimpleNamespace(current_round=1, rounds=[
        SimpleNamespace(round_index=1, image_ids=["orig-17"], observed=True)])
    merged = evidence(instance_id="repeat-view", decision="merge", same_as="desk")
    assert len(validate_instance_evidence([merged, evidence()], ledger)) == 2
    with pytest.raises(ValueError):
        validate_instance_evidence([merged], ledger)


def test_count_gate_is_reported_separately_from_visual_evidence():
    handle = SimpleNamespace(
        grounding_coverage={"requirements": [{"name": "table", "scan_complete": True,
                                              "status": "unavailable",
                                              "binding_failures": ["mask"],
                                              "usable_object_ids": ["obj-1"]}]},
        category_binding_complete=lambda _name: False)
    text = count_tool_status(handle)
    assert '"callable_now":false' in text and "binding_failures" in text
    assert "does not change this gate" in text


def test_admission_does_not_claim_unperformed_semantic_verification():
    verify = SimpleNamespace(passed=True, check_status={
        "answer_contract": "pass", "valid_json": "pass",
        "binding_integrity": "not_applicable", "derivation_replay": "not_performed"})
    kernel = SimpleNamespace(cell_vsi_interface_calls=[
        {"name": "CountingEvidence", "allowed": True}])
    scope = verification_scope(verify, kernel)
    assert scope["admission_passed"] and scope["instance_image_references"] == "pass"
    assert scope["instance_pixel_semantics"] == scope["instance_identity_and_coverage"] == "not_performed"
