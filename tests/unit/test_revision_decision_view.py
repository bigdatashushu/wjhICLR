from copy import deepcopy

from skill3d.evolution.revision_view_v11 import decision_case_view


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

def test_initial_candidate_values_survive_without_asserting_model_observation():
    case, _ = case_with_reads()
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
