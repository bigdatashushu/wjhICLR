"""Bounded reads from valid episode results; no tool execution or label access."""

from __future__ import annotations

import copy
import hashlib
import json
import re

OBSERVATION_READER_VERSION = "observation-reader/1.0"
MAX_PAGE_CHARS = 4000
MAX_PAGE_ITEMS = 32
MAX_READS_PER_ROUND = 3
IDENTITY_FIELDS = ("obj_id", "category_name", "category")
OBJECT_FIELDS = (
    *IDENTITY_FIELDS, "visible_frames", "track_id", "question_roles", "source_label",
    "mask_available", "mask_source_present", "mask_support_ratio", "binding_validation", "grounding_status",
    "binding_flags", "duplicate_suspect", "det_conf", "det_conf_source",
)
_PUBLIC_FIELDS = {
    "inspect_frames": {"images", "frame_ids", "n_images", "note"},
    "detect_objects": {"status", "service_healthy", "error", "detections", "n_detections",
                       "n_new_objects", "new_object_ids", "frame_ids", "categories", "note"},
    "count_objects": {"count", "n_records", "n_distinct_tracks", "n_geometric_merges",
                      "n_without_track", "duplicate_suspect", "diagnostic_warnings", "instance_consolidation"},
    "object_centroid": {"centroid_normalized", "centroid_metric", "category_name", "track_id",
                        "world_up_used", "handedness_used"},
    "object_3d_extent": {"extent_normalized", "extent_metric", "extent_longest_metric",
                         "n_valid_points", "degradation_flags"},
    "plane_fit_room_size": {"room_diagonal_normalized", "room_area_m2", "plane_inlier_ratio",
                           "floor_fit_ratio", "n_near_floor", "fit_quality",
                           "fit_quality_formula_version", "degradation_flags",
                           "measurement_source", "prepared_receipt_sha256"},
    "object_surface_distance": {"distance", "unit", "definition", "distance_normalized",
                               "distance_metric", "a", "b", "n_nn_samples", "n_valid_points",
                               "quantile_q", "voxel_size", "point_contamination_suspect",
                               "degradation_flags"},
    "relative_distance_rank": {"reference", "ranking", "status", "closest_category",
                              "per_candidate", "candidates", "requested_categories",
                              "categories_without_detection", "categories_with_invalid_geometry",
                              "tied_categories", "margin_normalized", "contract_version",
                              "definition", "quantile_q", "degradation_flags"},
    "relative_direction_of": {"direction", "difficulty", "world_up_used", "handedness_used"},
    "connectivity_graph": {"nodes", "edges", "traversable_matrix", "node_components",
                           "start_node", "grid", "version", "notes",
                           "scene_span_normalized", "scene_span_unit", "scene_span_definition"},
}


def canonical_payload(payload, *, tool=""):
    """Stable enumeration identity; never reorder coordinates, rankings or frame arrays."""
    if tool == "list_objects" and isinstance(payload, list):
        def key(row):
            identity = str(row.get("obj_id", "")) if isinstance(row, dict) else ""
            parts = tuple((0, int(p)) if p.isdigit() else (1, p)
                          for p in re.split(r"(\d+)", identity))
            return parts, _json(row)
        return sorted(payload, key=key)
    return payload


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


def payload_hash(payload, *, tool=""):
    return hashlib.sha256(_json(canonical_payload(payload, tool=tool)).encode()).hexdigest()


def frame_ranges(values):
    frames = sorted(set(map(int, values)))
    ranges = []
    for frame in frames:
        if ranges and frame == ranges[-1][1] + 1:
            ranges[-1][1] = frame
        else:
            ranges.append([frame, frame])
    return ranges


def object_summary_row(obj, *, question_type=""):
    """All critical columns, including unknowns, before adding further candidates."""
    priority = (("visible_frames", "track_id", "question_roles")
                if question_type == "obj_appearance_order"
                else ("question_roles", "track_id", "visible_frames"))
    keys = (*IDENTITY_FIELDS, *priority, "source_label", "mask_available", "mask_source_present",
            "mask_support_ratio", "binding_validation", "grounding_status",
            "binding_flags", "duplicate_suspect")
    row = {key: obj[key] for key in keys if key in obj}
    row.setdefault("track_id", None)
    row.setdefault("mask_available", None)
    if "visible_frames" in row:
        row["visible_frames"] = {"inclusive_ranges": frame_ranges(row["visible_frames"])}
        if question_type == "obj_appearance_order":
            row["earliest_recorded_frame"] = min(obj["visible_frames"], default=None)
    if "binding_flags" in row:
        row["binding_flags"] = [
            {k: flag[k] for k in ("code",) if k in flag}
            for flag in row["binding_flags"] if isinstance(flag, dict)]
    return row


def read_observation(observations, result_id, fields=None, offset=0, limit=8):
    if not isinstance(result_id, str) or not result_id:
        raise ValueError("result_id must be an existing nonempty result ID")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    if type(limit) is not int or not 1 <= limit <= MAX_PAGE_ITEMS:
        raise ValueError(f"limit must be 1..{MAX_PAGE_ITEMS}")
    obs = next((o for o in observations if o.result_id == result_id), None)
    if obs is None:
        raise ValueError("unknown, failed, unresolved or invalidated result_id")
    original = obs.payload
    payload = canonical_payload(original, tool=obs.source_tool)
    rows = payload if isinstance(payload, list) else [payload]
    present = {k for row in rows if isinstance(row, dict) for k in row}
    allowed = (set(OBJECT_FIELDS) if obs.source_tool == "list_objects"
               else _PUBLIC_FIELDS.get(obs.source_tool, set()))
    if fields is not None and (
            not isinstance(fields, list) or not fields or len(fields) > 32
            or any(not isinstance(k, str) or k not in allowed for k in fields)):
        raise ValueError("fields must be a nonempty list of allowed top-level public data fields")
    selected = set(fields) if fields is not None else allowed
    omitted = sorted((present & allowed) - selected)
    data_path = []
    # A named list column can be paged directly without copying the whole dict.
    collection = (fields[0] if fields is not None and len(fields) == 1
                  and isinstance(payload, dict) and isinstance(payload.get(fields[0]), list) else None)
    if collection is not None:
        rows = payload[collection]
        data_path = [collection]
    if offset > len(rows):
        raise ValueError("offset exceeds total_count")
    original_rows = (original[collection] if collection is not None else original
                     if isinstance(original, list) else [original])
    # Preserve raw source paths even when the view sorts object records.
    locations = {}
    for index, row in enumerate(original_rows):
        locations.setdefault(_json(row), []).append(index)
    source_indices = [locations[_json(row)].pop(0) for row in rows]
    page = {
        "result_id": obs.result_id, "source_tool": obs.source_tool,
        "schema_version": OBSERVATION_READER_VERSION,
        "evidence_version": obs.evidence_version, "total_count": len(rows),
        "returned_range": [offset, offset], "next_cursor": offset if offset < len(rows) else None,
        "coverage_complete": not rows, "fields_omitted": omitted,
        "unknown_fields_omitted_count": len(present - allowed),
        "payload_hash": payload_hash(payload, tool=obs.source_tool), "data": [],
        "data_path": data_path, "source_item_paths": [],
    }
    for position, row in enumerate(rows[offset:offset + limit], offset):
        filtered = (row if collection is not None else
                    {key: row[key] for key in sorted(selected) if key in row}
                    if isinstance(row, dict) else row)
        page["data"].append(copy.deepcopy(filtered))
        page["source_item_paths"].append(
            data_path + [source_indices[position]] if isinstance(original, list) or collection
            else [])
        end = offset + len(page["data"])
        page.update(returned_range=[offset, end], next_cursor=end if end < len(rows) else None,
                    coverage_complete=offset == 0 and end == len(rows))
        if len(_json(page)) > MAX_PAGE_CHARS:
            page["data"].pop()
            page["source_item_paths"].pop()
            end -= 1
            page.update(returned_range=[offset, end], next_cursor=end,
                        coverage_complete=False)
            if end == offset:
                raise ValueError("single item exceeds page budget; request fewer fields")
            break
    return page


def reader_contract() -> dict:
    return {
        "name": "ReadObservation", "schema_version": OBSERVATION_READER_VERSION,
        "arguments": "result_id: str, fields: list[str] | None=None, offset: int=0, limit: int=8",
        "max_page_items": MAX_PAGE_ITEMS, "max_page_chars": MAX_PAGE_CHARS,
        "max_reads_per_round": MAX_READS_PER_ROUND,
        "authorization": "ordinary phase, current episode, successful non-invalidated and resolved result only",
        "return": "result_id/source_tool/schema_version/evidence_version/total_count/returned_range/"
                  "next_cursor/coverage_complete/fields_omitted/payload_hash/data/data_path/source_item_paths",
        "semantics": "Reads raw result data without executing detection or reconstruction. "
                     "Pages are copied to the next request after YieldObservations. "
                     "Use the original result_id and raw data paths for derivation. "
                     "source_item_paths maps each returned item to its original raw-data path. "
                     "fields=[one_list_field] pages that collection directly. "
                     "print is diagnostic and does not request observations.",
    }
