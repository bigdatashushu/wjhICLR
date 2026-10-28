from copy import deepcopy
import hashlib
import json

from skill3d.evolution.revision_view_v11 import (
    decision_case_view, split_round_components, reassemble_round_components,
)


def case_with_reads():
    read = {
        "result_id": "synthetic-result", "data": [
            {"frame_id": 19, "confidence": 0.9876543210987, "bbox_xyxy": [1, 2, 3, 4]}],
        "source_item_paths": ["detections.12"], "next_cursor": 13,
        "total_count": 71, "coverage_complete": False,
    }
    import json
    case = {
        "case_id": "synthetic-case", "trace_ref": "synthetic:trace",
        "question_text": "Synthetic question", "options": ["A", "B"],
        "reference_answer": "A", "model_answer": "B", "score": 0.0,
        "outcome": "failure", "selected_object_ids": [],
        "rounds": [
            {"index": 1, "program_source": "page = ReadObservation('synthetic-result')\nYieldObservations()\n",
             "observation_reads": [read]},
            {"index": 2, "program_source": "ReturnAnswer(answer='B')\n",
             "feedback": "Requested observation pages (raw data, original references):\n"
                         + json.dumps([read])},
        ],
    }
    return case, read


def test_literal_stdout_projection_preserves_numbers_errors_and_reassembles():
    case, _ = case_with_reads()
    objects = [{
        "obj_id": f"synthetic-object-{index}", "category_name": "chair",
        "centroid_world": [1.12345678901234, 0.0, -2.5],
        "det_conf": 0.987654321, "binding_flags": ["mask_missing"],
        "binding_validation": {"valid": False, "errors": ["mask_missing"],
                               "generation_sha256": "a" * 64},
        "binding_provenance": [{"synthetic": "repeated-metadata" * 500}],
    } for index in range(20)]
    text = repr(objects) + "\n"
    case["rounds"] = [{
        "index": 1, "program_source": "print(objects)\n",
        "execution_diagnostics": {
            "stdout": text, "error_type": "ValueError", "error_message": "synthetic error",
            "error_line": 2, "traceback_text": "synthetic traceback"},
    }]
    before = deepcopy(case)
    view = decision_case_view(case)
    diagnostics = view["rounds"][0]["execution_diagnostics"]
    stdout = diagnostics["stdout"]
    assert stdout["source_text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert stdout["data"][0]["centroid_world"] == objects[0]["centroid_world"]
    assert stdout["data"][0]["det_conf"] == objects[0]["det_conf"]
    assert stdout["data"][0]["binding_validation"] == {
        "valid": False, "errors": ["mask_missing"]}
    assert all(diagnostics[key] == value for key, value in before["rounds"][0][
        "execution_diagnostics"].items() if key != "stdout")
    row = {"case_index": 0, "complete_case": view}
    parts = split_round_components(row, lambda rows: len(json.dumps(rows)) < 4000)
    assert len(parts) > 1
    assert reassemble_round_components(parts) == [row]
    assert case == before


def test_nonobject_or_executable_stdout_stays_verbatim(tmp_path):
    case, _ = case_with_reads()
    marker = tmp_path / "must-not-exist"
    for text in (
            "distance=0.123456789\nstatus=False\n",
            "[1.123456789, 0, False, None]\n",
            "[{'obj_id': 'a'}, 1]\n",
            f"[__import__('pathlib').Path({str(marker)!r}).touch()]\n"):
        case["rounds"][0]["execution_diagnostics"] = {"stdout": text}
        assert decision_case_view(case)["rounds"][0]["execution_diagnostics"]["stdout"] == text
    assert not marker.exists()


def test_full_code_numbers_and_observation_delivery_chronology_survive_projection():
    case, read = case_with_reads()
    before = deepcopy(case)
    view = decision_case_view(case)
    assert case == before
    assert [row["program_source"] for row in view["rounds"]] == [
        row["program_source"] for row in case["rounds"]]
    assert view["rounds"][0]["observation_reads"][0]["data"] == read["data"]
    assert view["rounds"][1]["delivered_observation_pages"] == [
        {"recorded_read_ref": [0, 0]}]
    assert not view["rounds"][0].get("delivered_observation_pages")
    assert view["reference_answer"] == "A" and view["score"] == 0
    assert view["projection"]["raw_record_retained"]


def test_unmatched_delivered_page_is_kept_instead_of_discarded():
    import json
    case, read = case_with_reads()
    changed = deepcopy(read)
    changed["data"][0]["confidence"] = 0.123456789
    case["rounds"][1]["feedback"] = (
        "Requested observation pages (raw data, original references):\n" + json.dumps([changed]))
    view = decision_case_view(case)
    assert view["rounds"][1]["delivered_observation_pages"][0][
        "original_delivered_page"] == changed


def test_selected_and_dependency_candidates_omit_only_binding_provenance():
    case, _ = case_with_reads()
    case["selected_object_ids"] = ["synthetic-selected"]
    case["rounds"][0]["results"] = [{
        "tool": "synthetic-tool", "object_dependencies": ["synthetic-dependency"]}]
    case["candidate_objects"] = [{
        "obj_id": obj_id, "grounding_status": "bound",
        "binding_validation": {"valid": True}, "binding_flags": [],
        "centroid_world": [1.123456789, 2, 3], "visible_frames": [2, 19],
        "binding_provenance": [{"semantic_request_identity": {"synthetic": "metadata"}}],
    } for obj_id in ("synthetic-selected", "synthetic-dependency", "synthetic-unused")]
    before = deepcopy(case)
    view = decision_case_view(case)
    assert case == before
    assert view["selected_candidates"] == [
        {key: value for key, value in obj.items() if key != "binding_provenance"}
        for obj in before["candidate_objects"][:2]]
    assert "selected_candidates.binding_provenance" in view["projection"]["omitted"]
    assert view["projection"]["raw_record_retained"]
    view["selected_candidates"][0]["binding_validation"]["valid"] = False
    assert case == before


def test_initial_candidate_values_survive_without_asserting_model_observation():
    case, _ = case_with_reads()
    case["rounds"][0]["program_source"] += "\nobject_id = 'synthetic-object'\n"
    case["evidence_history"] = [{"candidates": [{
        "obj_id": "synthetic-object", "category_name": "chair",
        "centroid_world": [1.123456789, 2, 3], "visible_frames": [2, 19],
        "grounding_status": "unbound", "pointcloud_world": "/synthetic/storage.npy",
    }]}]
    view = decision_case_view(case)
    facts = view["initial_candidate_facts"]
    assert facts["objects"][0]["centroid_world"] == [1.123456789, 2, 3]
    assert facts["objects"][0]["visible_frames"] == [2, 19]
    assert "pointcloud_world" not in facts["objects"][0]
    assert "not asserted" in facts["delivery_status"]


def test_catalog_filters_by_recorded_ids_and_retains_unused_counts():
    case, _ = case_with_reads()
    names = ["program-object", "argument-object", "read-object", "feedback-object",
             "dependency-object", "unused-object", "program-object-extra"]
    objects = [{
        "obj_id": name, "category_name": "chair", "centroid_world": [1.123456789, 2, 3],
        "binding_validation": {
            "status": "suspect", "checks": {"mask": False}, "errors": ["mask_missing"],
            "generation_sha256": "a" * 64, "files": {"mask": {"path": "/synthetic/mask.npy"}}},
        "pointcloud_world": "/synthetic/cloud.npy", "mask_per_frame": "",
        "candidate_ids": ["synthetic-redundant-id"],
    } for name in names]
    case["evidence_history"] = [{"candidates": objects}]
    case["candidate_objects"] = objects
    case["rounds"][0].update(
        program_source="object_id = 'program-object'",
        results=[{"args": {"object_id": "argument-object"},
                  "object_dependencies": ["dependency-object"]}],
        observation_reads=[{"data": [{"obj_id": "read-object"}]}])
    case["rounds"][1]["feedback"] = 'Delivered object: {"obj_id": "feedback-object"}'
    before = deepcopy(case)
    view = decision_case_view(case)
    facts = view["initial_candidate_facts"]
    assert [obj["obj_id"] for obj in facts["objects"]] == names[:5]
    assert facts["catalog_count"] == 7 and facts["omitted_count"] == 2
    assert facts["categories"] == {"chair": 7}
    for obj in [*facts["objects"], *view["selected_candidates"]]:
        assert obj["centroid_world"] == [1.123456789, 2, 3]
        assert obj["binding_validation"] == {
            "status": "suspect", "checks": {"mask": False}, "errors": ["mask_missing"]}
        assert obj["artifact_reference_present"] == {
            "mask_per_frame": False, "pointcloud_world": True}
        assert not {"candidate_ids", "pointcloud_world", "mask_per_frame"} & set(obj)
    assert case == before


def test_inline_object_payload_consumption_retains_matching_catalog_candidates():
    case, _ = case_with_reads()
    case["evidence_history"] = [{"candidates": [
        {"obj_id": "inline-object"}, {"obj_id": "unused-object"}]}]
    case["rounds"][0].update(
        program_source="r = tools.list_objects(ctx)\nrows = r['data']",
        results=[{"payload": [{"obj_id": "inline-object"}]}])
    assert [obj["obj_id"] for obj in decision_case_view(case)[
        "initial_candidate_facts"]["objects"]] == ["inline-object"]


def test_inline_raw_detection_consumption_keeps_every_original_detection_value():
    case, _ = case_with_reads()
    payload = {"status": "ok", "n_detections": 1, "detections": [
        {"frame_id": 19, "confidence": 0.9876543210987}]}
    row = {"program_source": "r = tools.detect_objects(ctx)\nx = r.payload['detections']\n",
           "results": [{"source_tool": "detect_objects", "payload": payload}]}
    case["rounds"] = [row]
    assert decision_case_view(case)["rounds"][0]["results"][0]["payload"] == payload
    row["program_source"] = "r = tools.detect_objects(ctx)\nYieldObservations([r.result_id])\n"
    result = decision_case_view(case)["rounds"][0]["results"][0]
    assert "detections" not in result["payload"]
    assert result["detection_array"]["count"] == 1
    assert "unread detection arrays" in decision_case_view(case)["projection"]["summarized"]


def test_known_input_call_budget_exhaustion_preserves_parent_and_all_canonical_cases(
        tmp_path, monkeypatch):
    import json
    from skill3d.evolution.fitset_revision_v11 import (
        FitSetSlotReviser, RevisionInputBudgetExhausted,
    )
    from test_fitset_stream_v11 import make_runner, full_evidence_factory

    def refuse(self, **kwargs):
        raise RevisionInputBudgetExhausted(
            needed=19, available=15, source_sha256="a" * 64)

    monkeypatch.setattr(FitSetSlotReviser, "_evidence_batches", refuse)
    runner = make_runner(
        tmp_path, n=12, learning_input_mode="full_cases_v1",
        revision_client_factory=full_evidence_factory)
    assert runner.run(resume=False)["status"] == "complete"
    campaign = next((runner.root / "workers").glob("*/campaigns/*"))
    refusal = json.loads((campaign / "input_budget_refusal.json").read_text())
    assert refusal["model_requests_initiated"] == 0 and not refusal["method_assessed"]
    assert json.loads((campaign / "result.json").read_text())["reason"] == (
        "revision_input_request_budget_exhausted")
    assert not list(campaign.glob("revision/slots/*/*/request_intent*"))
    assert len(json.loads((campaign / "experience_bundle.json").read_text())["cases"]) == 12
    stream = json.loads(next((runner.root / "workers").glob("*/buffer/stream.json")).read_text())
    assert len(stream["slots"]["object_counting/g0m0"]["case_refs"]) == 12
