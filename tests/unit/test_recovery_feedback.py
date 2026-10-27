"""恢复摘要保留决策关键事实，压缩不能改变证据与引用语义。"""

import json
from types import SimpleNamespace

import pytest

from skill3d.online.recovery import (
    ValidatedObservation, collect_validated, failed_call_feedback,
    observation_feedback, summarize_payload,
)
from skill3d.sandbox.ast_guard import ast_guard
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas import ToolResult
from skill3d.schemas.answer import CANONICAL_UNIT_BY_QUESTION_TYPE
from skill3d.synthesis.prompt_builder import round_action_instructions
from skill3d.tools import REGISTRY

from test_geometry_tools import _handle, _scene


def test_rank_summary_keeps_incomplete_status_after_large_nested_audit():
    payload = {
        "candidates": [{"category": "mouse", "instances": [
            {"obj_id": "mouse-1", "audit": {"details": "x" * 6000}}]}],
        "reference": {"obj_id": "bin-1", "category_name": "bin"},
        "ranking": [{"category": "table", "obj_id": "table-1", "distance_normalized": 0.3}],
        "status": "incomplete_candidates",
        "closest_category": None,
        "categories_without_detection": ["keyboard"],
        "categories_with_invalid_geometry": ["mouse"],
        "tied_categories": [],
        "requested_categories": ["table", "keyboard", "mouse"],
    }
    text = ValidatedObservation("rank-1", "relative_distance_rank", payload).summary()
    for fact in (
        '"status": "incomplete_candidates"', '"closest_category": null',
        '"categories_without_detection": ["keyboard"]',
        '"categories_with_invalid_geometry": ["mouse"]',
        '"obj_id": "table-1"', '"distance_normalized": 0.3',
        "[result_id=rank-1]",
    ):
        assert fact in text
    assert len(text) < 1600
    assert "Summary omissions" in text and "audit" in text
    assert "x" * 100 not in text
    assert payload["candidates"][0]["instances"][0]["audit"]["details"] == "x" * 6000


@pytest.mark.parametrize("payload", [
    [{"obj_id": f"object-{i}", "category_name": "chair"} for i in range(80)],
    {"detections": [{"category_name": "chair", "frame_id": i} for i in range(80)],
     "status": "fault", "service_healthy": False, "error": "detector offline"},
    {"unknown_field": "x" * 4000, "status": "ambiguous_grounding"},
])
def test_trimming_is_explicit_and_preserves_parseable_values(payload):
    text = summarize_payload(payload, max_chars=300)
    assert len(text) <= 300
    body, omission = text.split("\n", 1)
    parsed = json.loads(body)
    assert parsed
    assert "Summary omissions" in omission
    if isinstance(payload, dict):
        assert parsed["status"] == payload["status"]
    if "detections" in payload:
        assert parsed["service_healthy"] is False
        assert parsed["error"] == "detector offline"


def test_long_binding_audit_keeps_both_tv_candidates_within_budget():
    import copy
    import hashlib
    payload = [{
        "obj_id": f"tv-{i}", "category_name": "television",
        "track_id": f"qtrk-{i}", "grounding_status": "question_targeted_fill",
        "det_conf": .85, "det_conf_source": "groundingdino",
        "mask_support_ratio": .75, "duplicate_suspect": i == 1,
        "binding_flags": [{"code": "semantic_label_mask_conflict",
                           "audit_sha256": str(i) * 64, "endpoints": ["x" * 9000]}],
        "binding_provenance": [{"prompt": "y" * 8000}],
    } for i in range(2)]
    original = copy.deepcopy(payload)
    text = summarize_payload(payload, tool="list_objects", max_chars=1200)
    assert len(text) <= 1200
    data = json.loads(text)
    assert [o["obj_id"] for o in data["objects"]] == ["tv-0", "tv-1"]
    assert all(o["binding_flags"][0]["code"] == "semantic_label_mask_conflict"
               for o in data["objects"])
    assert data["audit_omitted"] is True and data["coverage_complete"] is True
    assert data["payload_sha256"] == hashlib.sha256(json.dumps(
        original, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
    assert payload == original


@pytest.mark.parametrize("budget", [128, 256, 300, 1200])
def test_list_summary_has_complete_prefix_and_truthful_omission_counts(budget):
    payload = [{"obj_id": f"o{i}", "category_name": "chair", "visible_frames": list(range(32))}
               for i in range(100)]
    text = summarize_payload(payload, tool="list_objects", max_chars=budget)
    assert len(text) <= budget
    data = json.loads(text)
    if "objects" in data:
        assert data["shown"] + data["omitted"] == data["total"] == 100
        assert data["coverage_complete"] is False
        assert [o["obj_id"] for o in data["objects"]] == [
            o["obj_id"] for o in payload[:data["shown"]]]
    else:
        assert data["omitted"]


def test_summary_rejects_budget_below_minimum():
    with pytest.raises(ValueError, match="128"):
        summarize_payload([], max_chars=127)


def test_sorted_summary_exposes_original_list_paths_for_derivation():
    payload = [
        {"obj_id": "chair-10", "category_name": "chair", "visible_frames": [3, 7]},
        {"obj_id": "chair-2", "category_name": "chair", "visible_frames": [1, 4]},
    ]
    summary = json.loads(summarize_payload(payload, tool="list_objects"))
    assert summary["source_item_paths"] == [[1], [0]]
    for row, path in zip(summary["objects"], summary["source_item_paths"]):
        assert payload[path[0]]["obj_id"] == row["obj_id"]
    feedback = observation_feedback([ValidatedObservation("r1", "list_objects", payload)])
    assert "summary view of raw list payload=" in feedback
    assert "raw data root is a list" in feedback


@pytest.mark.parametrize("error_type", ["NameError", "UnboundLocalError"])
def test_name_errors_are_deterministic_but_timeouts_keep_retry(error_type):
    from skill3d.online.recovery import deterministic_cell_failure
    cell = SimpleNamespace(error_code="execution", error_type=error_type)
    assert deterministic_cell_failure(cell, [])
    result = SimpleNamespace(error="ConnectionError: endpoint unavailable", payload=None)
    assert not deterministic_cell_failure(cell, [result])


def test_only_identical_observations_share_a_summary_and_all_ids_remain_real():
    a = ValidatedObservation("r1", "measure", {"distance_m": 2}, "v1", {"obj_id": "a"})
    b = ValidatedObservation("r2", "measure", {"distance_m": 2}, "v1", {"obj_id": "a"})
    changed = [
        ValidatedObservation("r3", "measure", {"distance_m": 3}, "v1", {"obj_id": "a"}),
        ValidatedObservation("r4", "measure", {"distance_m": 2}, "v2", {"obj_id": "a"}),
        ValidatedObservation("r5", "measure", {"distance_m": 2}, "v1", {"obj_id": "b"}),
    ]
    text = observation_feedback([a, b, *changed])
    assert text.count("payload=") == 4
    assert 'Identical observations also have result_ids=["r2"]' in text
    for obs in [a, *changed]:
        assert f"[result_id={obs.result_id}]" in text


def test_collect_parses_list_values_and_excludes_failed_or_revoked_results():
    good = ToolResult(result_id="list-id", tool="list_objects",
                      value='[{"obj_id": "chair-1"}]', evidence_version="original")
    results = [
        good,
        good.model_copy(update={"result_id": "failed", "status": "failed"}),
        good.model_copy(update={"result_id": "error", "error": "failed"}),
        good.model_copy(update={"result_id": "revoked", "invalidated_by": ["geometry_3d"]}),
    ]
    valid = collect_validated(SimpleNamespace(tool_results=results), evidence_version="new")
    assert len(valid) == 1 and valid[0].result_id == "list-id"
    assert valid[0].payload == [{"obj_id": "chair-1"}]
    assert valid[0].evidence_version == "original"


def test_failed_call_reports_actual_runtime_arguments():
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene(question_type="object_counting")))
    kernel.run_cell("object_centroid(None)")
    text = failed_call_feedback(kernel.cell_results)
    assert 'object_centroid({"obj_id": null})' in text
    assert kernel.cell_results[-1].result_id in text
    assert "error_code=domain_value" in text


@pytest.mark.parametrize("question_type,unit", CANONICAL_UNIT_BY_QUESTION_TYPE.items())
def test_ordinary_visual_submission_example_runs_with_explicit_payload(question_type, unit):
    text = round_action_instructions(question_type, finalizing=False)
    line = next(line for line in text.splitlines() if line.startswith("For an answer genuinely based entirely on original images"))
    code = line.split(": ", 1)[1]
    value = "B" if unit == "option" else 2
    code = f"answer = {value!r}\n" + code
    assert ast_guard(code, allowed_tools=set()).ok
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    cell = kernel.run_cell(code)
    assert cell.error is None
    payload = kernel.answer_slot.payload
    assert payload.value == value and payload.unit == unit
    assert payload.basis == "visual_estimate"
    assert payload.used_result_ids == [] and payload.derivation is None
    assert kernel.cell_results == []
