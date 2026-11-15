"""Presentation-only compaction; authorization and source receipts stay unchanged."""

from __future__ import annotations

from collections import Counter
import copy
import json

from skill3d.schemas.answer import derivation_contract
from skill3d.tools.docs_v11 import TOOL_DOCS_VERSION as SOURCE_TOOL_DOCS_VERSION

COMPACTION_VERSION = "vsi-prompt-compaction/1.0"
TOOL_DOCS_VERSION = "vsi-tool-docs-compact/1.0"

# Neutral interface descriptions. Registry-rendered signatures, dependencies and
# dynamic unit modes are copied verbatim, after the registry's own authorization.
DESCRIPTIONS = {
    "list_objects": (
        "data=list[dict]: obj_id, category_name, visible_frames, track_id, source_label, "
        "binding_provenance, det_conf/det_conf_source, mask_available, mask_support_ratio, "
        "binding_flags, grounding_status, duplicate_suspect. Empty filter lists all; "
        "complete-category normalization, never substring matching. obj_id encodes no category. "
        "det_conf is detector confidence; mask_support_ratio is frame support. "
        "mask_available is false without a source, otherwise unknown until validated. "
        "2D-only candidates may lack geometry. semantic_label_mask_conflict blocks object geometry; "
        "other overlap diagnostics certify neither identity nor merging. Inspect all candidates "
        "and actual images; choose explicit IDs, never first-row or detection-order matches. "
        "Even a unique category match is unverified. Inventory length is not physical-instance coverage."),
    "count_objects": (
        "data=dict: count, n_records, n_distinct_tracks, n_geometric_merges, n_without_track, "
        "duplicate_suspect, diagnostic_warnings, instance_consolidation(method,min_overlap,error). "
        "Same category filter as list_objects. Group nonempty track_id; untracked records each "
        "form a group. Merge group point clouds and consolidate by bidirectional NN overlap; "
        "retain groups without clouds once. Question-driven M5 instances are already consolidated. "
        "Requires complete technical category scan and usable bindings; this does not prove "
        "every real instance was detected."),
    "object_centroid": (
        "data=dict: centroid_normalized(world 3D), centroid_metric(m or None), category_name, "
        "track_id, world_up_used(unit up), handedness_used(right/left). Missing/invalid world "
        "conventions make the last two None, not the centroid. Metric coordinates depend on "
        "current scale; withdrawal invalidates old metric results. A new authorized normalized "
        "call may remain usable."),
    "object_surface_distance": (
        "data=dict: distance, unit, definition, distance_normalized, distance_metric(None in "
        "normalized mode), a/b(obj_id/category_name), n_nn_samples, n_valid_points, quantile_q, "
        "voxel_size, point_contamination_suspect, degradation_flags. Low quantile of bidirectional "
        "nearest-neighbor distances between distinct object point sets: a nearest-surface proxy. "
        "No centroid/camera distance or silent unit conversion; nonfinite requested distance raises."),
    "relative_distance_rank": (
        "reference=object ID or unique category; candidate_categories=ALL current option categories, "
        "at least two, no duplicates/synonyms. Each category uses its nearest instance by low-quantile "
        "bidirectional point-set surface distance. data=dict: reference, ranking(ascending category, "
        "obj_id,distance_normalized,n_instances_in_category), status(ok/incomplete_candidates/"
        "uncertain_geometry/ambiguous_grounding), closest_category(only status=ok), "
        "per_candidate(None if indeterminate), candidates(category status/n_instances/instances "
        "with obj_id,track_id,distance_normalized,n_valid_points,degradation_flags,audit), "
        "requested_categories, categories_without_detection, categories_with_invalid_geometry, "
        "tied_categories, margin_normalized, contract_version, definition, quantile_q, degradation_flags. "
        "Missing measurements, object/track reuse across roles or tied minima are indeterminate; "
        "never minimize incomplete per_candidate. Close unequal margins are audit only. No metric scale needed."),
    "object_3d_extent": (
        "data=dict: extent_normalized=[dx,dy,dz], extent_metric=[dx,dy,dz] in m, n_valid_points, "
        "degradation_flags; point-cloud branch also returns extent_longest_metric. Bbox fallback "
        "is marked bbox_fallback_no_pointcloud. Missing valid scale/measurable geometry raises."),
    "plane_fit_room_size": (
        "Fit floor/wall planes. data=dict: room_diagonal_normalized, room_area_m2, plane_inlier_ratio"
        "(ground/all points), floor_fit_ratio(ground/near-floor candidates), n_near_floor, "
        "fit_quality=sqrt(floor_fit_ratio*ground_planarity) in [0,1], fit_quality_formula_version, "
        "degradation_flags, measurement_source, prepared_receipt_sha256. prepared_verified_cache "
        "means identity-verified paired preparation. plane_fit_low_quality is an uncalibrated "
        "audit flag, not automatic rejection. Missing finite area raises."),
    "relative_direction_of": (
        "Direction of target_id centroid from observer_id centroid facing facing_at_id centroid. "
        "IDs or resolvable categories accepted. easy=left/right; medium=left/right/back "
        "(absolute angle >=135 degrees is back); hard=front-left/front-right/back-left/back-right. "
        "data=dict: direction, difficulty, world_up_used, handedness_used. Missing/invalid world "
        "conventions or degenerate horizontal vectors raise."),
    "object_visible_frames": (
        "data=list[int] of mask-verified frozen frame slots (0..N-1), not physical video indices; "
        "independent of 3D validity. Box-only/missing masks are unknown. Empty registered visibility "
        "proves neither video-wide absence nor true first appearance."),
    "connectivity_graph": (
        "Ground-plane occupancy-grid free-region reachability. data=dict: nodes, edges, "
        "traversable_matrix, start_node, grid diagnostics. Geometric approximation; "
        "scene_span_normalized is cloud span, not scale. No floor/free-cell mask or room area. "
        "Every object node requires a usable binding/finite centroid and is a result dependency; "
        "invalid nodes raise instead of being omitted. Missing cloud/world_up raises."),
    "reproject": (
        "Project world-space 3D point via camera pose/intrinsics to data=[u,v] pixels. "
        "frame_idx is a frozen slot; points behind/on the camera plane raise."),
    "inspect_frames": (
        "Read/crop frozen slots 0..N-1, <=32 per call; no video resampling. Optional boxes: one "
        "top-left-origin [x0,y0,x1,y1] pixel box/frame; finite endpoints sorted/clamped/rounded, "
        ">=8 pixels/side. Outside/too-small boxes raise. data=dict: images(image_id,frame_id,"
        "source_frame_index,box_xyxy,source_hw,sent_hw,scale,content_sha256,transform_version), "
        "n_images, frame_ids, note. Longest side resized to <=640px. YieldObservations must "
        "deliver pixels in a completed next request; an image_id alone is not observation."),
    "detect_objects": (
        "Detect nonempty categories on frozen frame slots. data=dict: status(ok/fault/empty), "
        "detections(category_name,frame_id,bbox_xyxy pixels,confidence,bbox_norm1000), n_detections, "
        "n_new_objects, new_object_ids, service_healthy, frame_ids, categories, error, "
        "detector_endpoint_hash, note. fault=service failure; healthy empty is not absence. "
        "Partial frame errors can coexist with status=ok. Adds 2D-only box/visibility candidates, "
        "no 3D centroid/cloud/track. Detections have no obj_id; new_object_ids covers only new IDs "
        "and may be empty on repeats. Resolve IDs with list_objects, never list order. Partial "
        "results certify no full coverage. After a successful full requested scan, do not repeat "
        "unchanged frames/categories; inspect candidates/views or obtain materially new evidence."),
}


def compact_tool_docs(registry, *args, **kwargs) -> str:
    """Shorten descriptions only; unknown/new interfaces retain their full docs."""
    return compact_rendered_tool_docs(registry.docs(*args, **kwargs))


def compact_rendered_tool_docs(full: str) -> str:
    """Also support offline projection of an already-authorized frozen directory."""
    lines = [f"Tool data interfaces [{TOOL_DOCS_VERSION}]; signatures and dependencies "
             f"from {SOURCE_TOOL_DOCS_VERSION}. Return fields below describe envelope data."]
    for line in full.splitlines()[1:]:
        if not line.startswith("- ") or "): " not in line:
            lines.append(line)
            continue
        prefix, body = line.split("): ", 1)
        name = prefix[2:].split("(", 1)[0]
        suffix_at = body.rfind(" [requires_artifacts: ")
        if name not in DESCRIPTIONS or suffix_at < 0:
            lines.append(line)
        else:
            lines.append(prefix + "): " + DESCRIPTIONS[name] + body[suffix_at:])
    return "\n".join(lines)


def _failure_summary(failures: list) -> list[dict]:
    # Preserve every distinct semantic field. Only per-candidate opaque identity
    # is omitted; complete identities and raw responses remain in source receipts.
    counts = Counter()
    for failure in failures:
        value = ({k: v for k, v in failure.items() if not (
            k == "candidate_id" and isinstance(v, str) and len(v) == 64
            and all(char in "0123456789abcdef" for char in v))}
                 if isinstance(failure, dict) else {"error": failure})
        counts[json.dumps(value, ensure_ascii=False, sort_keys=True)] += 1
    return [{**json.loads(key), "count": count} for key, count in sorted(counts.items())]


def compact_m5_coverage(coverage: dict) -> dict:
    """Copy public scan facts and aggregate failure lists without changing M5."""
    public = copy.deepcopy({k: v for k, v in coverage.items() if k not in {
        "phase_receipts", "noun_extraction", "stages", "request_identity", "elapsed_s"}})
    requirements = []
    for source in coverage.get("requirements", []):
        row = copy.deepcopy({k: v for k, v in source.items() if k not in {
            "detector_errors", "vlm_errors", "binding_failures", "blocking_failures"}})
        binding = source.get("binding_failures", [])
        blocking = source.get("blocking_failures", [])
        if "binding_failures" in source:
            row["binding_failure_summary"] = _failure_summary(binding)
            row["binding_failure_count"] = len(binding)
        if "blocking_failures" in source:
            row["blocking_failure_count"] = len(blocking)
            if blocking == binding and "binding_failures" in source:
                row["blocking_failures_same_as_binding"] = True
            else:
                row["blocking_failure_summary"] = _failure_summary(blocking)
        requirements.append(row)
    public["requirements"] = requirements
    public["presentation_version"] = COMPACTION_VERSION
    return public


_TASK_OPS = {
    "object_counting": ("field", "count"),
    "object_abs_distance": ("field", "convert"),
    "object_size_estimation": ("field", "convert"),
    "room_size_estimation": ("field", "convert"),
    "object_rel_distance": ("field", "argmin", "option_map"),
    "object_rel_direction": ("field", "option_map"),
    "obj_appearance_order": ("field", "sort", "option_map"),
    "route_planning": ("field", "option_map"),
}


def task_derivation_contract(question_type: str) -> dict:
    """Task-relevant parameter help; the shared registered operation set is intact."""
    contract = derivation_contract()
    result = {k: v for k, v in contract.items() if k not in {
        "allowed_parameters", "raw_list_count_example", "direction_option_example",
        "list_length_is_not_physical_instance_coverage"}}
    result["registered_operations"] = sorted(contract["allowed_parameters"])
    result["task_parameter_help"] = {
        op: contract["allowed_parameters"][op]
        for op in _TASK_OPS.get(question_type, contract["allowed_parameters"])}
    if question_type == "object_counting":
        result["raw_list_count_example"] = contract["raw_list_count_example"]
        result["list_length_is_not_physical_instance_coverage"] = True
    if question_type == "object_rel_direction":
        result["direction_option_example"] = contract["direction_option_example"]
    return result
