"""Counting behavior presentation checked at real request and feedback boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.online import runner
from skill3d.online.recovery import ValidatedObservation, observation_feedback as shared_feedback
from skill3d.synthesis.counting_context import (
    COUNTING_CONTEXT_VERSION, finalization_pages_text, observation_feedback,
)


def test_fault_keeps_raw_error_and_id_separate_from_candidate_evidence():
    candidates = ValidatedObservation(
        result_id="inventory-id", source_tool="list_objects",
        payload=[{"obj_id": "a", "category_name": "target", "visible_frames": [1, 5]}])
    error = "Detector unavailable: " + "service-detail " * 200
    fault = ValidatedObservation(
        result_id="fault-id", source_tool="detect_objects",
        payload={"status": "fault", "service_healthy": False, "error": error,
                 "frame_ids": [5], "n_detections": 0})
    text = observation_feedback([candidates, fault], question_type="object_counting")
    evidence, diagnostics = text.split("Failed diagnostics", 1)
    assert "fault-id" not in evidence and "inventory-id" in evidence
    assert "source_item_paths" in evidence
    assert "fault-id" in diagnostics and error in diagnostics
    assert "Still-valid observations" not in text
    assert fault.payload["status"] == "fault"


@pytest.mark.parametrize("task", (
    "object_abs_distance", "object_size_estimation", "room_size_estimation",
    "object_rel_distance", "object_rel_direction", "obj_appearance_order", "route_planning", ""))
def test_other_task_feedback_is_byte_identical(task):
    observations = [ValidatedObservation(
        result_id="same-id", source_tool="detect_objects",
        payload={"status": "fault", "error": "original error"})]
    for reader in (True, False):
        assert observation_feedback(observations, question_type=task, include_reader=reader) == (
            shared_feedback(observations, question_type=task, include_reader=reader))


def test_final_page_preview_preserves_absolute_paths_after_sorting_and_omissions():
    import json

    rows = [{"obj_id": f"obj_{i:02d}", "category_name": "target",
             "visible_frames": [i], "track_id": "long-track-" + "x" * 80}
            for i in range(12)]
    rows.reverse()
    paths = [[i + 30] for i in range(12)]
    page = {"result_id": "root-id", "source_tool": "list_objects", "data": rows,
            "source_item_paths": paths, "data_path": [], "returned_range": [30, 42],
            "total_count": 42, "coverage_complete": True, "next_cursor": None,
            "fields_omitted": []}
    text = finalization_pages_text([page], question_type="object_counting")
    projected = json.loads(text.split("):\n", 1)[1])[0]
    preview = projected["bounded_data_preview"]
    assert preview["omitted"] > 0 and preview["coverage_complete"] is False
    assert projected["coverage_complete"] is True and projected["returned_range"] == [30, 42]
    for row, path in zip(preview["objects"], preview["source_item_paths"]):
        assert rows[path[0] - 30]["obj_id"] == row["obj_id"]
    assert projected["result_id"] == "root-id"
    assert page["data"] == rows and page["source_item_paths"] == paths


@pytest.mark.parametrize("finalizing", (False, True))
def test_counting_actual_request_delivers_specialization_and_allows_zero_tools(finalizing):
    class Client:
        last_generation = {}
        last_usage = {}

        def set_request_context(self, context):
            self.context = context

        def chat(self, messages, **kwargs):
            self.messages = messages
            if finalizing:
                return '{"action":"answer","value":2,"basis":"visual_estimate","used_result_ids":[],"derivation":null}'
            return """```python
def solve(ctx):
    # Two whole instances coexist in the delivered view.
    answer = 2
    return ReturnAnswer(AnswerPayload(value=answer, unit='count',
        basis='visual_estimate', used_result_ids=[], derivation=None))
```"""
    scene = SimpleNamespace(
        question_type="object_counting", question_tool_scope="fallback_2d_only",
        summary="scene=synthetic metric_gate=not_passed overall_quality=0.8 n_objects=0",
        available_artifacts={"frames"}, grounding_coverage={}, local_facts={},
        evidence_profile=None, metric_evidence_gate_result=None)
    ep = SimpleNamespace(
        qa_id="synthetic", scene_name="synthetic", frame_set=None, options=[],
        question_type="object_counting", question="Count the target whole objects.")
    client = Client()
    fault = ValidatedObservation(
        result_id="failed-detector", source_tool="detect_objects",
        payload={"status": "fault", "service_healthy": False, "error": "no endpoint"})
    result = runner._synthesize(
        ep, scene, scene, [], runner.OnlineRunConfig(mode="real"), client,
        pixels=[np.zeros((8, 8, 3), dtype=np.uint8)], trigger="finalize" if finalizing else "initial",
        finalization_observations=[fault] if finalizing else None)
    assert result.program is not None, result.note
    assert client.context["counting_context_version"] == COUNTING_CONTEXT_VERSION
    assert COUNTING_CONTEXT_VERSION in result.prompt
    assert "answer = 1" not in result.prompt and "raw_list_count_example" not in result.prompt
    assert "Zero tools is legal" in result.prompt
    if finalizing:
        assert "Failed diagnostics" in result.prompt and "no endpoint" in result.prompt
        assert "ReadNextObservation" not in result.prompt and "```python" not in result.prompt
        assert client.context["tool_execution_allowed"] is False
    else:
        assert "overall_quality=" not in result.prompt and "metric_gate=" not in result.prompt
        assert "n_objects=0" in result.prompt
