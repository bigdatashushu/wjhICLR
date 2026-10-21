"""Content progress, bounded raw observation pages, and exact request image identity."""

import base64
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.online.observation_reader import MAX_PAGE_CHARS, read_observation
from skill3d.online.progress import ProgressState, transient_results
from skill3d.online.recovery import ValidatedObservation, summarize_payload
from skill3d.sandbox.ast_guard import ast_guard
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas.tool import ToolResult
from skill3d.synthesis.prompt_builder import build_image_messages
from skill3d.tools import REGISTRY
from skill3d.tools.image_ledger import ImageLedger
from test_geometry_tools import _handle, _scene


def candidates(n=31):
    return [{"obj_id": f"chair-{i}", "category_name": "chair", "visible_frames": [28, 29, 30, 31],
             "track_id": None, "source_label": "detector", "mask_available": False,
             "binding_validation": None, "grounding_status": "box_only"}
            for i in range(n)]


def test_pages_are_stable_complete_and_keep_frame_fields():
    rows = candidates()
    a = ValidatedObservation("r1", "list_objects", rows)
    b = ValidatedObservation("r2", "list_objects", rows[::-1])
    cursor, seen = 0, []
    while cursor is not None:
        page = read_observation([a], "r1", offset=cursor)
        reordered = read_observation([b], "r2", offset=cursor)
        assert page["data"] == reordered["data"]
        assert page["payload_hash"] == reordered["payload_hash"]
        assert len(json.dumps(page, separators=(",", ":"))) <= MAX_PAGE_CHARS
        assert all(row["visible_frames"] == [28, 29, 30, 31] for row in page["data"])
        assert [rows[path[0]] for path in page["source_item_paths"]] == page["data"]
        seen.extend(row["obj_id"] for row in page["data"])
        cursor = page["next_cursor"]
    assert seen == [row["obj_id"] for row in rows]
    summary = json.loads(summarize_payload(rows, tool="list_objects", question_type="obj_appearance_order"))
    assert summary["coverage_complete"] is False and summary["next_cursor"] > 0
    assert summary["objects"][0]["visible_frames"] == {"inclusive_ranges": [[28, 31]]}
    assert summary["objects"][0]["earliest_recorded_frame"] == 28
    assert summary["objects"][0]["track_id"] is None


@pytest.mark.parametrize("kwargs", [
    {"offset": -1}, {"offset": True}, {"offset": 100}, {"limit": 0}, {"limit": 33},
    {"fields": ["ground_truth"]}, {"fields": "obj_id"}, {"fields": []},
])
def test_invalid_reader_arguments_fail_closed(kwargs):
    with pytest.raises(ValueError):
        read_observation([ValidatedObservation("r", "list_objects", candidates())], "r", **kwargs)


def test_page_budget_never_silently_cuts_a_field():
    obs = ValidatedObservation("r", "list_objects", [
        {"obj_id": "a", "binding_flags": [{"code": "x" * 5000}]}])
    with pytest.raises(ValueError, match="fewer fields"):
        read_observation([obs], "r")
    page = read_observation([obs], "r", fields=["obj_id"])
    assert page["data"] == [{"obj_id": "a"}]
    assert page["fields_omitted"] == ["binding_flags"]


def test_named_collection_pages_keep_raw_paths_and_reject_unknown_data_fields():
    detections = [{"frame_id": i, "category_name": "chair"} for i in range(40)]
    obs = ValidatedObservation("r", "detect_objects", {
        "detections": detections, "status": "ok", "ground_truth": "private"})
    page = read_observation([obs], "r", fields=["detections"], offset=32, limit=8)
    assert page["data"] == detections[32:]
    assert page["source_item_paths"] == [["detections", i] for i in range(32, 40)]
    assert page["next_cursor"] is None and not page["coverage_complete"]
    assert "private" not in json.dumps(read_observation([obs], "r", fields=["status"]))
    with pytest.raises(ValueError, match="allowed"):
        read_observation([obs], "r", fields=["ground_truth"])


def test_kernel_reader_is_reserved_bounded_and_revalidates_original_result():
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    kernel.tool_results.append(ToolResult(
        tool="list_objects", result_id="original", value=json.dumps(candidates())))
    code = (
        'page = ReadObservation("original", fields=["obj_id"], offset=24)\n'
        'page["data"][0]["obj_id"] = "tampered"\n'
        'YieldObservations(["original"], "inspect page")')
    assert ast_guard(code, allowed_tools=set(REGISTRY.names())).ok
    cell = kernel.run_cell(code)
    assert cell.terminated == "yield" and not cell.error
    assert kernel.observation_reads[0]["data"][0]["obj_id"] == "chair-24"
    assert not kernel.cell_calls
    assert kernel.authorization_receipts[-1]["allowed"]
    kernel.reset_user_namespace()
    kernel.tool_results[0] = kernel.tool_results[0].model_copy(update={"invalidated_by": ["geometry_3d"]})
    cell = kernel.run_cell('ReadObservation("original")')
    assert cell.error and "invalidated" in cell.error
    kernel.reset_user_namespace()
    kernel.set_tools_enabled(False)
    cell = kernel.run_cell('ReadObservation("original")')
    assert cell.error and "ordinary-phase" in cell.error
    assert not ast_guard("ReadObservation = 1", allowed_tools=set()).ok


def test_progress_ignores_receipt_churn_but_honors_new_values_context_and_pages():
    state = ProgressState()
    obs = ValidatedObservation("r1", "inspect_frames", {
        "images": [{"image_id": "first", "content_sha256": "pixels", "produced_round": 1}]})
    def update(observations, **kw):
        return state.update(observations, None, context=kw.pop("context", "g1"),
                            eligible=True, **kw)
    assert update([obs])["made_progress"]
    dup = ValidatedObservation("r2", "inspect_frames", {
        "images": [{"image_id": "second", "content_sha256": "pixels", "produced_round": 2}]})
    assert update([obs, dup])["consecutive_no_progress"] == 1
    assert not update([obs, dup], transient=True)["finalize_required"]
    assert update([obs, dup])["finalize_required"]
    assert update([obs, dup], context="g0")["candidate_or_capability_change"]
    assert update([obs, dup], context="g0", pages=[{"data": ["next row"]}])["made_progress"]
    fresh = ValidatedObservation("r3", "inspect_frames", {"images": [{"content_sha256": "different"}]})
    assert update([obs, dup, fresh], context="g0")["made_progress"]
    # Same measurement on another object is not a duplicate.
    assert update([ValidatedObservation("v1", "centroid", {"x": 1}, args={"obj_id": "a"})])["made_progress"]
    assert update([ValidatedObservation("v2", "centroid", {"x": 1}, args={"obj_id": "b"})])["made_progress"]
    assert transient_results([SimpleNamespace(
        source_tool="detect_objects", payload={"status": "ok", "error": "frame request failed"})])


def ledger():
    value = ImageLedger(max_images=32, max_derived_images=8)
    value.set_frames(list(range(32)), [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(32)])
    value.ensure_frame_records()
    return value


def test_pack_covers_tail_sources_and_exact_jpeg_request_identity():
    value = ledger()
    crop = value.produce(value.frames[31][:4, :4], kind="crop", produced_by="inspect_frames",
                         source_frame_id=31, box_xyxy=[0, 0, 4, 4])
    value.plan_round(round_index=1, trigger="initial", derived_image_ids=[],
                     original_frame_ids=value.frame_ids)
    value.mark_round_sent(1)
    with pytest.raises(ValueError, match="immutable"):
        value.plan_round(round_index=1, trigger="observation", derived_image_ids=[crop.image_id],
                         original_frame_ids=value.frame_ids)
    plan = value.plan_round(round_index=2, trigger="observation",
                            derived_image_ids=[crop.image_id, crop.image_id],
                            original_frame_ids=value.frame_ids)
    assert plan.n_images == 32 and len(plan.derived_image_ids) == 1
    assert "frame-31" in plan.original_image_ids and "frame-0" in plan.original_image_ids
    frames = [value.pixels_of(i) for i in plan.original_image_ids]
    messages = build_image_messages("frozen", frames, max_images=32,
                                    extra=[(crop.image_id, value.pixels_of(crop.image_id))])
    value.bind_request(2, messages)
    urls = [part["image_url"]["url"] for part in messages[0]["content"]
            if part["type"] == "image_url"]
    assert plan.jpeg_sha256 == [hashlib.sha256(base64.b64decode(url.split(",")[1])).hexdigest()
                               for url in urls]
    assert plan.request_sha256
    value.mark_round_sent(2)
    value.mark_delivered(2, plan.image_ids)
    value.mark_round_observed(2)
    assert crop.sent_in_requests == crop.response_completed_in_requests == ["image-request-2"]
    assert crop.present_in_current_request
    assert not value.get(plan.omitted_originals[0]).present_in_current_request


def test_unsent_replan_replaces_manifest_instead_of_merging():
    value = ledger()
    value.plan_round(round_index=1, trigger="initial", derived_image_ids=[], original_frame_ids=[0, 1])
    plan = value.plan_round(round_index=1, trigger="initial", derived_image_ids=[], original_frame_ids=[30, 31])
    assert plan.image_ids == ["frame-30", "frame-31"] and len(value.rounds) == 1
