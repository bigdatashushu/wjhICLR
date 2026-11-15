"""Counting-only presentation; shared authorization, ledger and M11 are unchanged."""

from __future__ import annotations

import json
import re

from skill3d.online.recovery import (
    observation_feedback as shared_observation_feedback, summarize_payload,
)

COUNTING_CONTEXT_VERSION = "vsi-counting-context/1.0"

COUNTING_RULES = (
    "Enumerate whole instances from images independently of inventory/count values; keep "
    "representative frame/location anchors. Check same-frame coexistence, cross-frame identity, "
    "parts versus wholes, occluded regions and uncertain exclusions. Revisit doubts that could "
    "change the total; seams or legs alone do not establish separate objects. Zero tools is legal "
    "when delivered views suffice. Use targeted tools for concrete doubts. Scan completion, "
    "candidate count and replay acceptance do not certify real-instance coverage. A detector "
    "fault resolves no omission or conflict with an old count. Give the best supported integer."
)

COUNTING_PROGRAM = """You are a spatial reasoning coding agent. Count from the question, delivered images and valid observations.

## Execution and evidence
{{ execution_contract }}
Counting presentation: {{ counting_context_version }}.
- Output one fenced Python block, no outside prose. Use top-level code or def solve(ctx); the host calls solve. Define variables; use None/True/False. Comments and observation reasons must be English.
- Imports: numpy/scipy/math/statistics only; no eval/exec/__import__/open, other imports, writes or network. Do not overwrite show/ReturnAnswer/YieldObservations/AnswerPayload/ReadObservation/ToolData/ReadNextObservation/MeasurementAnswer/tools/scene/frames.
- End with return ReturnAnswer(payload) or return YieldObservations([actual_result_id,...], "reason"). ReturnAnswer takes one argument, provisional until geometry/evidence/replay checks pass. Budgets and recovery are host-controlled; Skill grants no tools or bypass.
- Variables reset each round. Rebuild from current valid IDs. Current successful calls without a control action may yield feedback only; print/prose never submits. Caught failures reject submissions.
- Use only listed tools/unit modes and actual usable bindings. The complete method stays fixed; later feedback may withdraw capabilities/invalidate results. Outer status=ok is call success only: inspect inner status, errors, missing items and degradation. Unhealthy, failed, invalidated or unresolved evidence supports no answer; empty/omitted/None is not absence or zero.
- M5 certifies neither referent identity nor complete coverage. Geometry needs explicit verified obj_id values. Never repeat unchanged invalid calls; changed comments/basis repairs nothing. Never invent observations/IDs/geometry/scale. Selection telemetry is read-only, not an answer or a required nonzero selection.

## Answer and observation interfaces
AnswerPayload(value=answer, unit='count', basis=..., used_result_ids=..., derivation=...) is available without import.
- visual_estimate: zero-tool image-only segment, used_result_ids=[], derivation=None.
- mixed: visual plus valid tool support; list actual IDs used when one replay operation cannot establish the total.
- tool_derived: nonempty valid IDs plus {"op":...,"input_result_ids":[...],"parameters":{...}}; one registered operation must establish the value. Paths start at raw tool data, omit "data."; parameters are fields, not expressions. M11 replays ToolResults, not instance completeness.
Task derivation help: {{ derivation_contract_json }}
- Short-form ReturnAnswer(value) is mixed and checks historical successful evidence. Only a genuine zero-tool visual segment with complete payload can discard history; current results are always checked.
- Tool envelope: result_id, source_tool, status, data, schema_version, evidence_version. Read unchanged raw data via r["data"] or ToolData(r); use host result_id, not obj_id/inner IDs. Edited/ID-only envelopes are rejected.
- ReadObservation(result_id, fields=None, offset=0, limit=8); ReadNextObservation(result_or_id, fields=None, limit=8) reads the next uncovered gap without rerunning tools. Reader data is ALWAYS a list; dict/scalar original root is ToolData(page)[0]. Use returned cursor, never offset += limit. Tail coverage_complete is local; vsi_read_progress.coverage_complete tracks cumulative selected-field coverage. Keep fields fixed; omission inherits selection, changes start independent coverage.
- YieldObservations([page["result_id"]], "reason") delivers the page next request; [] returns successful still-valid episode results. Use real IDs. Crop pixels must arrive in a completed subsequent request before use.

## Question
{{ question }}
Question type: {{ question_type }}. Submission unit: count.
Task meaning: {{ task_definition }}
Submit one finite non-negative integer, never a list/dict/string. Preserve category, scope and exclusions.
{{ task_rules }}

## Current evidence
{{ route_header }}
{{ scene_summary }}
Technical scan, usable binding and real-instance coverage are separate; neither geometry nor metric scale establishes completeness.

## Authorized tools
{{ tool_docs }}
Signatures, dependencies and allowed modes apply at execution; local binding/call preconditions still apply.
{% if skill_entries %}
## Reference method (complete, unchanged)
{% for e in skill_entries %}{{ e }}{% endfor %}{% endif %}"""

COUNTING_DESCRIPTIONS = {
    "list_objects": (
        "data=list[dict]: obj_id,category_name,visible_frames,track_id,source_label,"
        "binding_provenance,det_conf/det_conf_source,mask_available,mask_support_ratio,"
        "binding_flags,grounding_status,duplicate_suspect. Exact normalized category, not substring; "
        "empty filter lists all. obj_id encodes no category. Confidence is detector confidence, "
        "mask_support_ratio is frame support; mask_available is false without source, otherwise "
        "unknown until validated. 2D candidates may lack geometry; semantic_label_mask_conflict "
        "blocks geometry. Overlap/unique match certifies no identity or merging. Inspect relevant "
        "candidates and images, select explicit IDs. List length is not physical count."),
    "count_objects": (
        "data=dict: count,n_records,n_distinct_tracks,n_geometric_merges,n_without_track,"
        "duplicate_suspect,diagnostic_warnings,instance_consolidation(method,min_overlap,error). "
        "Exact normalized category filter. Group nonempty track_id, untracked records separately; "
        "merge group clouds by bidirectional NN overlap, keep cloudless groups once. Question-driven "
        "M5 instances are already consolidated. Requires complete technical scan and usable bindings, "
        "neither proves all real instances were detected. Cross-check images and identities before "
        "using count; do not sum detections or copy inventory length."),
    "detect_objects": (
        "Detect nonempty categories on specified frozen slots. data=dict: status(ok/fault/empty),"
        "detections(category_name,frame_id,bbox_xyxy pixels,confidence,bbox_norm1000),n_detections,"
        "n_new_objects,new_object_ids,service_healthy,frame_ids,categories,error,"
        "detector_endpoint_hash,note. fault is service failure; healthy empty is not absence; "
        "status=ok may include frame errors. Adds 2D boxes/visibility, no centroid/cloud/track. "
        "Detections have no obj_id; new IDs may be empty on repeats. Resolve with list_objects, "
        "not row order. Target unresolved views; no unchanged repeat after full scan or fault "
        "without a changed service/binding premise. New boxes do not resolve cross-frame identity."),
    "inspect_frames": (
        "Crop/read frozen slots 0..N-1, <=32/call, no resampling. Optional one pixel box/frame "
        "[x0,y0,x1,y1], top-left origin; finite sorted/clamped/rounded endpoints, >=8 pixels/side; "
        "outside/small boxes raise. data=dict: images(image_id,frame_id,source_frame_index,"
        "box_xyxy,source_hw,sent_hw,scale,content_sha256,transform_version),n_images,frame_ids,note. "
        "Longest side <=640px. Target an uncertain region and YieldObservations; pixels must "
        "arrive next request, image_id alone is not observation."),
    "object_visible_frames": (
        "data=list[int]: mask-verified frozen slots 0..N-1, not physical video indices. Independent "
        "of 3D validity; box-only/missing masks unknown. Empty visibility proves no absence. "
        "Use original images when masks are unavailable."),
    "object_centroid": (
        "data=dict: centroid_normalized(world 3D),centroid_metric(m or None),category_name,"
        "track_id,world_up_used(unit up),handedness_used(right/left). Invalid world conventions "
        "make the latter two None, not centroid. Metric depends on current scale; withdrawal "
        "invalidates old metric results, authorized normalized calls may remain usable. "
        "Grounded positions assist identity, not completeness."),
    "reproject": (
        "Project world 3D point via pose/intrinsics to data=[u,v] pixels. frame_idx is frozen "
        "slot; behind/on-camera-plane points raise. Use only valid grounded geometry."),
}


def compact_counting_scene_summary(text: str) -> str:
    """Remove only named, irrelevant quality/scale telemetry from public scene prose."""
    noise = (
        "warp_inlier|cloud_overlap|head_relative_p95|cloud_neighbor_tail|overall_quality|"
        "metric_scale_status|metric_gate|frame_set_hash"
    )
    return re.sub(rf"(?<!\S)(?:{noise})=\S+\s*", "", text)


def observation_feedback(observations, *, question_type: str = "", include_reader: bool = True) -> str:
    """Route counting presentation locally; every other task keeps shared exact text."""
    if question_type != "object_counting":
        return shared_observation_feedback(
            observations, question_type=question_type, include_reader=include_reader)
    valid, faults = [], []
    for obs in observations:
        payload = obs.payload
        bad = isinstance(payload, dict) and (
            payload.get("status") in {"fault", "failed", "error"}
            or payload.get("service_healthy") is False)
        (faults if bad else valid).append(obs)
    lines = ["Observed candidates/data (non-invalidated calls; identity and coverage unverified):"]
    # Reuse the shared raw-list projection and exact source paths, without its generic guidance.
    def append_groups(items):
        groups = {}
        for obs in items:
            key = json.dumps([obs.source_tool, obs.args, obs.evidence_version, obs.payload,
                              obs.degraded_evidence], sort_keys=True, default=str)
            groups.setdefault(key, []).append(obs)
        for group in groups.values():
            lines.append(group[0].summary(question_type=question_type))
            aliases = list(dict.fromkeys(o.result_id for o in group[1:]
                                        if o.result_id != group[0].result_id))
            if aliases:
                lines.append("  Identical source result_ids=" + json.dumps(aliases))
    append_groups(valid)
    if not valid:
        lines.append("No usable prior data; delivered images remain available for visual reasoning.")
    if faults:
        lines.append("Failed diagnostics (not count evidence; resolves no omission or old-count conflict):")
        append_groups(faults)
        for obs in faults:
            # The bounded preview can omit long scalars; preserve original error semantics in full.
            lines.append(f"  result_id={obs.result_id} raw_error="
                         + json.dumps(obs.payload.get("error"), ensure_ascii=False, default=str))
    lines.append("Paths start at raw data. List summaries use source_item_paths[i], not objects[i]; "
                 "visibility ranges are summary transforms. Missing detections are not absence. "
                 "Partial frame errors leave those views unresolved; old count is not completeness.")
    if include_reader:
        lines.append("Read existing result IDs, keep selected fields fixed and follow returned cursors; "
                     "vsi_read_progress tracks cumulative coverage. Yield pages; do not rerun inventory "
                     "for omitted rows or restart completed field coverage.")
    return "\n".join(lines)


def finalization_pages_text(pages, *, question_type: str = "") -> str:
    """Bound counting page previews while preserving source paths and cursor facts."""
    if question_type != "object_counting":
        return ("\nRecent requested valid pages (at most 3):\n"
                + json.dumps(list(pages), ensure_ascii=False, sort_keys=True))
    compact = []
    for page in pages:
        data = page.get("data", [])
        if page.get("source_tool") == "list_objects" and isinstance(data, list):
            preview = json.loads(summarize_payload(
                data, tool="list_objects", question_type=question_type, max_chars=1200))
            # Shared summary sorting may reorder rows; map page-local paths to source-root paths.
            original_paths = page.get("source_item_paths", [])
            preview["source_item_paths"] = [
                original_paths[path[0]] for path in preview.get("source_item_paths", [])
                if path and path[0] < len(original_paths)]
            preview.pop("payload_hash", None)
            preview.pop("payload_sha256", None)
        else:
            preview = summarize_payload(data, tool=page.get("source_tool", ""), max_chars=1200)
        compact.append({
            **{key: page[key] for key in (
                "result_id", "source_tool", "evidence_version", "data_path", "returned_range",
                "total_count", "coverage_complete", "next_cursor", "fields_omitted",
                "unknown_fields_omitted_count") if key in page},
            "bounded_data_preview": preview,
        })
    return (
        "\nRecent requested valid pages (bounded previews; coverage refers to original page, "
        "not preview completeness; source_item_paths address raw source data):\n"
        + json.dumps(compact, ensure_ascii=False, sort_keys=True))
