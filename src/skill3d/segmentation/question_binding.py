"""Question-driven M5: one text parser, exhaustive detection, verified saved bindings."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from skill3d.infra.m5_identity import (
    M5_IDENTITY_VERSION, array_identity, content_sha256,
    inventory_generation_identity, source_identity,
)
from skill3d.schemas.reconstruction import ObjectRecord
from skill3d.synthesis.request_context import request_phase
from .binding_validation import binding_is_usable, validate_binding
from . import mask_semantic_review as msr
from . import open_vocab_detector as ovd

M5_BINDING_VERSION = "m5-question-binding-v3-grouped-review"
PARSER_VERSION = "m5-question-objects-v3-qwen38-json-mode"
CACHE_VERSION = "m5-question-cache-v3-grouped-review"
MAX_OBJECTS = 16
PARAMETERS = {
    "parser_version": PARSER_VERSION, "max_objects": MAX_OBJECTS,
    "temperature": 0.0, "max_tokens": 2048, "format_repairs": 1,
    "nms_iou": 0.7, "instance_mask_iou": 0.8, "centroid_depth_tolerance": 0.15,
    "seed_mask_inside_fraction": 0.5, "max_points": 20000,
    "box_threshold": ovd.DEFAULT_BOX_THRESHOLD,
    "detector_max_attempts": 3, "detector_timeout_s": ovd.DEFAULT_TIMEOUT_S,
    "detector_total_timeout_s": ovd.DEFAULT_TOTAL_TIMEOUT_S,
    "segmentation_mode": "sam2_image_per_frame",
    "semantic_visual_version": msr.VISUAL_REVIEW_VERSION,
    "semantic_relation_version": msr.RELATION_REVIEW_VERSION,
    "semantic_min_pixels": 65536, "semantic_max_pixels": 1048576,
    "semantic_relation_batch_size": 32,
    "frame_scan": "all_readable_in_order",
    "detector_scan": "multi_category_per_frame_v1",
    "semantic_selection": "complete_link_geometry_two_views_v1",
    "semantic_views_per_group": 2,
    "semantic_all_views_tasks": ["obj_appearance_order"],
}
ROLES = {
    "object_counting": {"target"},
    "object_size_estimation": {"target"},
    "object_abs_distance": {"target_a", "target_b"},
    "object_rel_direction": {"observer_anchor", "facing_anchor", "target"},
    "object_rel_distance": {"reference", "candidate"},
    "route_planning": {"start_anchor", "facing_anchor", "waypoint", "destination"},
    "obj_appearance_order": {"target"},
    "room_size_estimation": set(),
}
PARSER_PROMPT = """Extract only the physical objects or route landmarks that must be
located to solve the supplied question. Read question, question_type and options only.
Return exactly this JSON object:
{"objects":[{"category":"cup","role":"target","source_text":"cup"}]}.
No prose, markdown, extra keys or inferred objects. category is a common English
singular category name, lower case, with spaces. source_text is a nonempty exact
substring copied verbatim from question or one option. Exclude directions, actions,
units, video, the robot and the camera operator. Resolve explicit references using
their original words; do not invent entities. At most 16 entries. Deduplicate
category+role while retaining different roles for the same category.
Allowed roles by question_type:
object_counting, object_size_estimation, obj_appearance_order: target
object_abs_distance: target_a, target_b
object_rel_direction: observer_anchor, facing_anchor, target
object_rel_distance: reference, candidate (extract every physical option)
route_planning: start_anchor, facing_anchor, waypoint, destination
room_size_estimation: empty objects.
"""


def question_input(question, question_type, options):
    return {"question": str(question), "question_type": str(question_type),
            "options": list(options or [])}


def binding_parameters(*, semantic_min_pixels=None, semantic_max_pixels=None,
                       semantic_relation_batch_size=None, semantic_review_config=None):
    result = dict(PARAMETERS)
    if semantic_review_config:
        from .intern_decision_client import review_config
        from .intern_decision_protocol import PROTOCOL_SHA256, RELATION_IDENTITY
        result["experimental_semantic_review"] = review_config(semantic_review_config)
        if semantic_review_config.get("mode") == "binary":
            from .intern_binary_review import PROTOCOL_SHA256 as binary_sha, admission_identity
            result["experimental_visual_protocol_sha256"] = binary_sha
            result["binary_admission_identity"] = admission_identity(
                semantic_review_config.get("admission_policy", "frozen-quality"))
        else:
            result["experimental_visual_protocol_sha256"] = PROTOCOL_SHA256
        result["experimental_relation_rules_sha256"] = content_sha256(RELATION_IDENTITY)
    overrides = {
        "semantic_min_pixels": semantic_min_pixels,
        "semantic_max_pixels": semantic_max_pixels,
        "semantic_relation_batch_size": semantic_relation_batch_size,
    }
    result.update({key: int(value) for key, value in overrides.items()
                   if value is not None})
    if result["semantic_min_pixels"] > result["semantic_max_pixels"]:
        raise ValueError("semantic_min_pixels must not exceed semantic_max_pixels")
    if result["semantic_relation_batch_size"] <= 0:
        raise ValueError("semantic_relation_batch_size must be positive")
    return result


def validate_parse(raw, inputs):
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {"objects"}:
        raise ValueError("expected exactly the objects key")
    rows = data["objects"]
    if not isinstance(rows, list) or len(rows) > MAX_OBJECTS:
        raise ValueError("objects must be a list of at most 16 entries")
    task = inputs["question_type"]
    allowed = ROLES.get(task)
    if allowed is None:
        raise ValueError(f"unsupported question_type: {task}")
    texts = [inputs["question"], *inputs["options"]]
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"category", "role", "source_text"}:
            raise ValueError("invalid object fields")
        cat, role, source = (row[key] for key in ("category", "role", "source_text"))
        if not all(isinstance(v, str) and v for v in (cat, role, source)):
            raise ValueError("object fields must be nonempty strings")
        if not re.fullmatch(r"[a-z]+(?:[ -][a-z]+)*", cat) or len(cat) > 80:
            raise ValueError("category must be a common lower-case English name")
        if cat in {"robot", "video", "image", "left", "right", "meter", "centimeter",
                   "direction", "turn", "forward", "backward", "camera operator"}:
            raise ValueError("non-target entity")
        if role not in allowed or not any(source in str(text) for text in texts):
            raise ValueError("invalid role or source_text without verbatim evidence")
        key = (cat, role)
        if key not in seen:
            result.append(dict(row))
            seen.add(key)
    required = allowed - ({"waypoint"} if task == "route_planning" else set())
    if {r["role"] for r in result} != allowed and not required.issubset(
            {r["role"] for r in result}):
        raise ValueError("missing required question roles")
    if task == "object_rel_distance" and any(
            not any(row["role"] == "candidate" and row["source_text"] in str(option)
                    for row in result) for option in inputs["options"]):
        raise ValueError("a physical answer option was not parsed as a candidate")
    return result


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                   indent=2, allow_nan=False))
    temporary.replace(path)


def _save_envelope(path, key, payload):
    body = {"version": CACHE_VERSION, "identity_sha256": key, "payload": payload}
    _write(path, {**body, "sha256": content_sha256(body)})


def _load_envelope(path, key):
    try:
        data = json.loads(path.read_text())
        digest = data.pop("sha256")
        if (data["version"] != CACHE_VERSION or data["identity_sha256"] != key
                or content_sha256(data) != digest):
            return None
        return data["payload"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def parse_question_objects(inputs, client, *, seed, identity, out_dir=None):
    parse_identity = {
        "input": inputs, "model": identity["identity"]["vlm"],
        "prompt": PARSER_PROMPT, "parameters": {k: PARAMETERS[k] for k in
            ("parser_version", "max_objects", "temperature", "max_tokens", "format_repairs")},
        "seed": seed, "environment": identity["identity"]["environment"],
        "sources": identity["identity"]["sources"],
    }
    key = content_sha256(parse_identity)
    cache = Path(out_dir) / "object_parses" / f"{key}.json" if out_dir else None
    if cache and identity["cache_eligible"]:
        saved = _load_envelope(cache, key)
        if saved and saved.get("status") == "parsed":
            try:
                rows = validate_parse(json.dumps({"objects": saved["objects"]}), inputs)
                return rows, {**saved, "cache_hit": True}
            except (ValueError, KeyError, TypeError):
                pass
    receipt = {"version": PARSER_VERSION, "identity_sha256": key,
               "identity": parse_identity, "status": "unavailable",
               "objects": [], "requests": [], "cache_hit": False}
    messages = [{"role": "system", "content": PARSER_PROMPT},
                {"role": "user", "content": json.dumps(inputs, ensure_ascii=False)}]
    if client is None or seed is None:
        receipt["error"] = "frozen text model or request seed unavailable"
    else:
        actual_client = getattr(client, "client", client)
        token_limit = min(PARAMETERS["max_tokens"],
                          int(getattr(actual_client, "max_output_tokens", 2048)))
        for attempt in range(2):
            row = {"attempt": attempt + 1, "seed": int(seed), "temperature": 0.0,
                   "max_tokens": token_limit, "messages": messages.copy(),
                   "request_sha256": content_sha256({
                       "messages": messages, "seed": int(seed), "temperature": 0.0,
                       "max_tokens": token_limit,
                       "model": parse_identity["model"]})}
            started = time.perf_counter()
            try:
                with request_phase("m5_object_parse"):
                    raw = client.chat(messages, max_tokens=token_limit,
                                      seed=int(seed), temperature=0.0,
                                      response_format={"type": "json_object"},
                                      enable_thinking=False)
                row["raw_response"] = raw
                receipt["objects"] = validate_parse(raw, inputs)
                receipt["status"] = "parsed"
            except (ValueError, TypeError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                messages = messages + [
                    {"role": "assistant", "content": row.get("raw_response", "")},
                    {"role": "user", "content":
                        "Repair the JSON once. Use only the original question and options. "
                        "Do not add unsupported objects. Validation error: " + str(exc)}]
            except Exception as exc:  # transport failure is not a format repair
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["service_failure"] = True
            row["elapsed_s"] = time.perf_counter() - started
            row["usage"] = dict(getattr(client, "last_usage", {}) or {})
            receipt["requests"].append(row)
            if receipt["status"] == "parsed" or row.get("service_failure"):
                break
        if receipt["status"] != "parsed":
            receipt["error"] = receipt["requests"][-1].get("error", "parse unavailable")
    if cache:
        _save_envelope(cache, key, receipt)
    return receipt["objects"], receipt


def _box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area = ((a[2] - a[0]) * (a[3] - a[1])
            + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / area if area > 0 else 0.0


def _nms(detections, category, frame_slot, request_sha):
    rows = [{
        "candidate_id": content_sha256({
            "category": category, "frame_slot": frame_slot,
            "request_sha256": request_sha, "raw_index": index}),
        "raw_index": index, "category": category, "frame_slot": frame_slot,
        "source_label": det.label, "bbox": list(det.bbox_xyxy),
        "score": float(det.confidence), "request_sha256": request_sha,
        "suppressed_by": None,
    } for index, det in enumerate(detections)]
    kept = []
    for row in sorted(rows, key=lambda r: (-r["score"], r["raw_index"])):
        duplicate = next((old for old in kept if _box_iou(row["bbox"], old["bbox"])
                          >= PARAMETERS["nms_iou"]), None)
        if duplicate:
            row["suppressed_by"] = duplicate["candidate_id"]
        else:
            kept.append(row)
    return rows, kept


def _mask_iou(left, right):
    common = sorted(set(left) & set(right))
    numerator, denominator = 0, 0
    for slot in common:
        a, b = left[slot], right[slot]
        if a.shape == b.shape and a.any() and b.any():
            numerator += int(np.logical_and(a, b).sum())
            denominator += int(np.logical_or(a, b).sum())
    return numerator / denominator if denominator else 0.0


def _robust_bbox(obj):
    try:
        points = np.load(obj.pointcloud_world, allow_pickle=False, mmap_mode="r")
        if points.ndim == 2 and points.shape[1] == 3 and len(points):
            return np.quantile(points, 0.05, axis=0), np.quantile(points, 0.95, axis=0)
    except (OSError, ValueError, TypeError):
        pass
    box = np.asarray(obj.bbox, dtype=float)
    return box[:3], box[3:]


def _bbox_gap(left, right):
    a0, a1 = _robust_bbox(left)
    b0, b1 = _robust_bbox(right)
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(a0, b0) - np.minimum(a1, b1))))


def _merge_instances(objects, masks, depth, frame_slots=None):
    """Conservatively group single-frame observations into object instances."""
    from .sam2_tracker import _merge_mask_groups
    valid = np.asarray(depth)
    valid = valid[np.isfinite(valid) & (valid > 0)]
    tolerance = PARAMETERS["centroid_depth_tolerance"] * (
        float(np.median(valid)) if valid.size else 0.0)
    slots = list(frame_slots or [
        next(iter(mask), -1) for mask in masks])

    def pair_compatible(left, right):
        if objects[left].category_name != objects[right].category_name:
            return False
        distance = np.linalg.norm(
            np.asarray(objects[left].centroid_world)
            - np.asarray(objects[right].centroid_world))
        if not tolerance or distance >= tolerance:
            return False
        if slots[left] == slots[right]:
            return _mask_iou(masks[left], masks[right]) >= PARAMETERS["instance_mask_iou"]
        return _bbox_gap(objects[left], objects[right]) < tolerance * 0.5

    groups = []
    for index in sorted(range(len(objects)), key=lambda i: (-objects[i].det_conf, i)):
        # Complete-link prevents a chain of individually close observations from
        # bridging two neighboring instances.
        hit = next((group for group in groups
                    if all(pair_compatible(member, index) for member in group)), None)
        if hit is None:
            groups.append([index])
        else:
            hit.append(index)
    merged = _merge_mask_groups(masks, groups)
    suspect = set()
    for i, group in enumerate(groups):
        for j in range(i):
            other = groups[j]
            if (objects[group[0]].category_name == objects[other[0]].category_name
                    and np.linalg.norm(np.asarray(objects[group[0]].centroid_world)
                                       - objects[other[0]].centroid_world) < tolerance):
                suspect.update([i, j])
    return groups, merged, suspect


def _review_groups(observations, depth, c2w, intrinsics):
    """Group provisional geometry for review scheduling, without admitting it."""
    geometry, masks, slots = [], [], []
    intrinsics = np.asarray(intrinsics)
    for observation in observations:
        candidate = observation["candidate"]
        slot = int(candidate["frame_slot"])
        mask = observation["mapped"][slot]
        ys, xs = np.nonzero(mask & np.isfinite(depth[slot]) & (depth[slot] > 0))
        # Bounded deterministic sampling is used only for provisional grouping.
        stride = max(1, int(np.ceil(len(xs) / 4096)))
        ys, xs = ys[::stride], xs[::stride]
        z = depth[slot, ys, xs]
        k = intrinsics[slot] if intrinsics.ndim == 3 else intrinsics
        camera = np.stack(((xs - k[0, 2]) * z / k[0, 0],
                           (ys - k[1, 2]) * z / k[1, 1], z), axis=1)
        points = camera @ c2w[slot, :3, :3].T + c2w[slot, :3, 3]
        if not len(points) or not np.isfinite(points).all():
            raise ValueError("provisional review geometry is not finite")
        geometry.append(ObjectRecord(
            obj_id=candidate["candidate_id"], category_name=candidate["category"],
            centroid_world=points.mean(axis=0).tolist(),
            bbox=np.concatenate(np.quantile(points, [0.05, 0.95], axis=0)).tolist(),
            det_conf=candidate["score"],
        ))
        masks.append(observation["mapped"])
        slots.append(slot)
    groups, _, _ = _merge_instances(geometry, masks, depth, frame_slots=slots)
    return groups


def _select_review_observations(observations, *, depth, c2w, intrinsics, task):
    """Keep at most two deterministic views per group; temporal tasks keep all."""
    if task in PARAMETERS["semantic_all_views_tasks"]:
        groups = [[index] for index in range(len(observations))]
    else:
        try:
            groups = _review_groups(observations, depth, c2w, intrinsics)
        except (ValueError, IndexError, TypeError, FloatingPointError):
            # Uncertain geometry cannot safely collapse review candidates.
            groups = [[index] for index in range(len(observations))]
    selected_indices = []
    for group_index, members in enumerate(groups):
        ranked = sorted(members, key=lambda index: (
            -float(observations[index]["candidate"]["score"]),
            -int(observations[index]["raw_mask"].sum()),
            int(observations[index]["candidate"]["frame_slot"]),
            observations[index]["candidate"]["candidate_id"],
        ))
        views = ranked[:1]
        if len(ranked) > 1:
            first_slot = observations[ranked[0]]["candidate"]["frame_slot"]
            views.append(min(ranked[1:], key=lambda index: (
                -abs(observations[index]["candidate"]["frame_slot"] - first_slot),
                ranked.index(index),
            )))
        views = views[: int(PARAMETERS["semantic_views_per_group"])]
        for index in members:
            row = observations[index]["row"]
            row["provisional_review_group"] = group_index
            row["review_selected"] = index in views
            if index not in views:
                row["status"] = "unreviewed_repeated_observation"
        selected_indices.extend(views)
    return [observations[index] for index in sorted(selected_indices)], groups


def refresh_coverage(grounding, objects):
    """Report category scan/candidate facts without certifying question referents."""
    body = dict(grounding)
    requirements = []
    for prior in body.get("requirements", []):
        row = dict(prior)
        matched = [o for o in objects if o.category_name == row["name"]]
        usable = [o for o in matched if binding_is_usable(o)]
        blocking = [failure for failure in row.get("binding_failures", [])
                    if failure.get("blocking", True)]
        row["matched_object_ids"] = [o.obj_id for o in matched]
        row["usable_object_ids"] = [o.obj_id for o in usable]
        row["blocking_failures"] = blocking
        row["status"] = ("candidates_available" if usable and row.get("scan_complete")
                         and not blocking else "unavailable")
        requirements.append(row)
    unresolved = [r["name"] for r in requirements if r["status"] != "candidates_available"]
    body.update(
        requirements=requirements, unresolved_names=unresolved,
        candidate_scan_complete=bool(requirements) and not unresolved,
        status="candidates_available" if requirements and not unresolved
        else "unavailable", n_new=sum(binding_is_usable(o) for o in objects))
    body.pop("all_present", None)
    body.pop("miss", None)
    return body


def validate_prepared_binding(binding, *, frames, depth, c2w, intrinsics, point_conf,
                              depth_conf, point_map, inputs, scene_identity,
                              seed=None, model_weights_sha256="", parameters=None,
                              source_compatibility=None):
    """Frozen preparation may be consumed without requiring a live detector."""
    stats = binding.stats
    generation = stats.get("generation") or {}
    identity = generation.get("identity") or {}
    expected_geometry = {name: array_identity(value) for name, value in (
        ("depth", depth), ("c2w", c2w), ("intrinsics", intrinsics),
        ("point_conf", point_conf), ("depth_conf", depth_conf), ("point_map", point_map))}
    if (stats.get("version") != M5_BINDING_VERSION
            or identity.get("version") != M5_IDENTITY_VERSION
            or generation.get("sha256") != content_sha256(identity)
            or identity.get("question") != inputs
            or identity.get("frames") != [array_identity(frame) for frame in frames]
            or identity.get("geometry") != expected_geometry
            or identity.get("reconstruction_identity") != scene_identity):
        raise ValueError("prepared M5 binding identity mismatch or legacy version")
    if identity.get("parameters") != (parameters or PARAMETERS):
        raise ValueError("prepared M5 parameters or implementation changed")
    current_sources = source_identity(Path(__file__).resolve().parents[1])
    if source_compatibility is not None:
        from .frozen_source_compatibility import reviewed_source_compatibility
        reviewed_source_compatibility(
            identity.get("sources"), current_sources, source_compatibility)
    elif identity.get("sources") != current_sources:
        raise ValueError("prepared M5 parameters or implementation changed")
    model = identity.get("vlm") or {}
    if seed is not None and model.get("seed", seed) != seed:
        raise ValueError("prepared M5 request seed changed")
    if model_weights_sha256 and model.get("weights_sha256") != model_weights_sha256:
        raise ValueError("prepared M5 model weights changed")
    for obj in binding.objects:
        if obj.grounding_status != "question_driven":
            raise ValueError("legacy objects cannot enter question-driven M5")
        if obj.binding_validation and obj.binding_validation.status == "validated":
            verdict = validate_binding(
                obj, depth_maps=depth, n_frames=len(frames),
                generation_sha256=generation["sha256"], check_receipt=True)
            if verdict.status != "validated":
                raise ValueError(f"prepared M5 binding changed: {obj.obj_id}")


def bind_question_objects(
    frames, predictor=None, question="", *, depth_maps, c2w_list,
    intrinsics=None, out_dir=None, scene_name="scene", point_conf=None,
    depth_conf=None, grid_transform=None, vlm_client=None, seed=None,
    frame_set_hash="", question_type="", options=None, qa_id="",
    scene_identity="", frame_provenance=None, model_weights_sha256="",
    point_map=None, geometry_usable=True,
    semantic_min_pixels=None, semantic_max_pixels=None,
    semantic_relation_batch_size=None, semantic_review_config=None,
):
    from . import sam2_tracker as st

    started = time.perf_counter()
    stage_timings = {}

    def progress(stage, **detail):
        if out_dir is not None:
            _write(Path(out_dir) / "binding_progress.json", {
                "qa_id": str(qa_id), "stage": stage,
                "elapsed_s": time.perf_counter() - started,
                "stage_timings": dict(stage_timings), **detail,
            })

    progress("identity")
    inputs = question_input(question, question_type, options)
    parameters = binding_parameters(
        semantic_min_pixels=semantic_min_pixels,
        semantic_max_pixels=semantic_max_pixels,
        semantic_relation_batch_size=semantic_relation_batch_size,
        semantic_review_config=semantic_review_config)
    generation = inventory_generation_identity(
        scene_name=scene_name, frame_set_hash=frame_set_hash, frames=frames,
        depth_maps=depth_maps, c2w_list=c2w_list, intrinsics=intrinsics,
        point_conf=point_conf, depth_conf=depth_conf, grid_transform=grid_transform,
        frame_provenance=frame_provenance, predictor=predictor, vlm_client=vlm_client,
        seed=seed, parameters=parameters, model_weights_sha256=model_weights_sha256,
        question=question, question_type=question_type, options=options,
        reconstruction_identity=scene_identity, point_map=point_map)
    stage_timings["identity_s"] = time.perf_counter() - started
    from .intern_decision_client import make_review_client, compare_fixed_categories
    from .intern_decision_protocol import RELATION_VERSION
    visual_client = make_review_client(semantic_review_config) or vlm_client
    if semantic_review_config:
        # No Qwen cache can serve an experimental protocol, including failed service calls.
        generation["cache_eligible"] = False
        generation["missing"] = [*generation["missing"], "experimental review disk reuse disabled"]
    key = generation["sha256"]
    root = Path(out_dir) / "question_bindings" / key if out_dir else None
    cache = root / "binding.json" if root else None
    if cache and generation["cache_eligible"] and geometry_usable:
        saved = _load_envelope(cache, key)
        if saved:
            try:
                objects = []
                for row in saved["objects"]:
                    row = dict(row)
                    for field in ("mask_per_frame", "pointcloud_world", "pointconf_world"):
                        if row.get(field):
                            if Path(row[field]).is_absolute():
                                raise ValueError("cache object references must be relative")
                            path = (root / row[field]).resolve()
                            if not path.is_relative_to(root.resolve()):
                                raise ValueError("cache reference escapes binding directory")
                            row[field] = str(path)
                    obj = ObjectRecord.model_validate(row)
                    if not binding_is_usable(obj, strict=True):
                        raise ValueError("cached binding is unverified or changed")
                    objects.append(obj)
                stats = saved["stats"]
                if (stats["version"] != M5_BINDING_VERSION
                        or stats["generation"] != generation):
                    raise ValueError("cache generation mismatch")
                stats = {**stats, "cache_hit": True, "actual_model_requests": 0,
                         "actual_detector_requests": 0, "actual_sam2_tracks": 0,
                         "elapsed_s": time.perf_counter() - started}
                stats["grounding"] = refresh_coverage(stats["grounding"], objects)
                return objects, stats, ["M5 复用同题、同冻结输入的已验证绑定"]
            except (KeyError, ValueError, TypeError, OSError):
                pass
    stats = {"version": M5_BINDING_VERSION, "generation": generation,
             "cache_hit": False, "detector_attempts": [], "detection_receipts": [],
             "candidate_receipts": [], "n_propagations": 0,
             "input_degraded": len(frames) != 32, "materialized": False,
             "stage_timings": stage_timings}
    if not geometry_usable:
        stats.update(status="m4_unavailable", elapsed_s=time.perf_counter() - started)
        return [], stats, ["M4 整场质量门未通过 → 冻结图像二维求解"]
    if question_type == "room_size_estimation":
        stats.update(materialized=True, status="objects_not_required")
        progress("completed", status=stats["status"])
        return [], stats, ["M5 房间面积题无需对象解析"]
    progress("object_parse")
    stage_started = time.perf_counter()
    parsed, receipt = parse_question_objects(inputs, vlm_client, seed=seed,
                                            identity=generation, out_dir=out_dir)
    stage_timings["object_parse_s"] = time.perf_counter() - stage_started
    stats["object_parse"] = receipt
    stats["actual_model_requests"] = (0 if receipt["cache_hit"] else len(receipt["requests"]))
    if receipt["status"] != "parsed":
        stats.update(status="parse_unavailable", parse_error=receipt.get("error", ""),
                     elapsed_s=time.perf_counter() - started)
        progress("completed", status="parse_unavailable", error=stats["parse_error"])
        return [], stats, ["M5 对象解析不可用 → 使用冻结图像二维求解"]
    categories = list(dict.fromkeys(row["category"] for row in parsed))
    progress("detector", categories=categories)
    readable = [i for i, frame in enumerate(frames)
                if frame is not None and np.asarray(frame).ndim == 3
                and np.asarray(frame).size]
    stats["input_degraded"] |= len(readable) != len(frames)
    if (not readable or len(readable) != len(frames)
            or depth_maps is None or c2w_list is None or intrinsics is None
            or len(depth_maps) != len(frames) or len(c2w_list) != len(frames)):
        stats["binding_error"] = "frozen image/geometry alignment unavailable"
        # Do not compact or renumber the frozen frame sequence.
    requirements = [{
            "name": category, "roles": [r for r in parsed if r["category"] == category],
            "scanned_slots": [], "scan_errors": [], "scan_complete": False,
            "binding_failures": [],
        } for category in categories]
    selected = []
    detector_requests = 0
    stage_started = time.perf_counter()
    for slot in readable:
        before = time.perf_counter()
        with ovd.capture_detector_attempts() as attempts:
            try:
                ovd.detect.last_error = ""
                detections = ovd.detect(frames[slot], ovd.prompt_from_nouns(categories))
                error = getattr(ovd.detect, "last_error", "")
            except Exception as exc:
                detections, error = [], f"{type(exc).__name__}: {exc}"
        request_sha = (attempts[0]["request_sha256"] if attempts else
                       content_sha256({"frame": array_identity(frames[slot]),
                                       "categories": categories}))
        category_request_shas = {category: request_sha for category in categories}
        assigned, unassigned = ovd.assign_categories(detections, categories)
        detector_requests += 1
        if unassigned:
            # Some services emit composite phrases. Rescan that frame with single
            # categories rather than guessing which category a box belongs to.
            assigned = {}
            for category in categories:
                with ovd.capture_detector_attempts() as fallback_attempts:
                    try:
                        ovd.detect.last_error = ""
                        assigned[category] = ovd.detect(
                            frames[slot], ovd.prompt_from_nouns([category]))
                        fallback_error = getattr(ovd.detect, "last_error", "")
                    except Exception as exc:
                        assigned[category] = []
                        fallback_error = f"{type(exc).__name__}: {exc}"
                detector_requests += 1
                attempts.extend(fallback_attempts)
                category_request_shas[category] = (
                    fallback_attempts[0]["request_sha256"] if fallback_attempts else
                    content_sha256({"frame": array_identity(frames[slot]),
                                    "categories": [category]}))
                error = error or fallback_error
        stats["detector_attempts"].extend(attempts)
        for requirement in requirements:
            category = requirement["name"]
            rows, kept = _nms(
                assigned[category], category, slot, category_request_shas[category])
            stats["detection_receipts"].append({
                "category": category, "frame_slot": slot, "candidates": rows,
                "transport_attempts": [], "error": error,
                "elapsed_s": time.perf_counter() - before,
                "request_sha256": category_request_shas[category],
                "unassigned_labels": unassigned,
            })
            selected.extend(kept)
            requirement["scanned_slots"].append(slot)
            if error:
                requirement["scan_errors"].append({"frame_slot": slot, "error": error})
        if (slot + 1) % 8 == 0:
            progress("detector", n_scanned_frames=slot + 1,
                     n_detector_requests=detector_requests, n_candidates=len(selected))
    for requirement in requirements:
        requirement["scan_complete"] = bool(readable) and not requirement["scan_errors"]
    stage_timings["detector_s"] = time.perf_counter() - stage_started
    stats["detector_fault"] = any(row["scan_errors"] for row in requirements)
    stats["actual_detector_requests"] = detector_requests
    work = root / uuid4().hex if root else None
    req_by_name = {row["name"]: row for row in requirements}
    rows_by_id = {
        candidate["candidate_id"]: {
            **candidate, "status": "unavailable",
            "track_id": candidate["candidate_id"],
        } for candidate in selected}
    observations = []
    progress("sam2", n_candidates=len(selected))
    stage_started = time.perf_counter()
    stats.update(n_propagations=0, n_sam2_image_frames=0, n_sam2_masks=0)
    if selected and not stats.get("binding_error"):
        try:
            image_predictor = st.build_image_predictor(predictor)
        except Exception as exc:
            stats["binding_error"] = f"{type(exc).__name__}: {exc}"
            image_predictor = None
    else:
        image_predictor = None
    for slot in sorted({int(row["frame_slot"]) for row in selected}):
        batch = [row for row in selected if int(row["frame_slot"]) == slot]
        before = time.perf_counter()
        try:
            if stats.get("binding_error"):
                raise ValueError(stats["binding_error"])
            segmented = st.segment_frame_boxes(
                frames[slot], [row["bbox"] for row in batch], image_predictor)
            if len(segmented) != len(batch):
                raise ValueError("SAM2 image result count mismatch")
            stats["n_sam2_image_frames"] += 1
            stats["n_sam2_masks"] += len(segmented)
        except Exception as exc:
            segmented = None
            error = f"{type(exc).__name__}: {exc}"
            for candidate in batch:
                row = rows_by_id[candidate["candidate_id"]]
                row.setdefault("error", error)
                req_by_name[candidate["category"]]["binding_failures"].append({
                    "candidate_id": candidate["candidate_id"],
                    "error": row["error"], "blocking": True})
        if segmented is not None:
            for candidate, seed_mask in zip(batch, segmented):
                row = rows_by_id[candidate["candidate_id"]]
                try:
                    row["sam2_requested"] = True
                    row["sam2_completed"] = True
                    raw_mask = np.asarray(seed_mask, dtype=bool)
                    if not raw_mask.any():
                        raise ValueError("empty mask")
                    h, w = raw_mask.shape
                    x0, y0 = max(0, int(candidate["bbox"][0])), max(
                        0, int(candidate["bbox"][1]))
                    x1, y1 = min(w, int(np.ceil(candidate["bbox"][2]))), min(
                        h, int(np.ceil(candidate["bbox"][3])))
                    inside = raw_mask[y0:y1, x0:x1].sum() / raw_mask.sum()
                    row["seed_mask_inside_fraction"] = float(inside)
                    if inside < PARAMETERS["seed_mask_inside_fraction"]:
                        raise ValueError("mask extends predominantly outside detector box")
                    mapped = {
                        slot: st._resize_mask_to(
                            raw_mask, np.asarray(depth_maps).shape[1:])}
                    if not np.any(
                            mapped[slot] & np.isfinite(depth_maps[slot])
                            & (depth_maps[slot] > 0)):
                        raise ValueError("mask lacks valid depth")
                    observations.append({
                        "candidate": candidate, "row": row,
                        "raw_mask": raw_mask, "mapped": mapped,
                        "req": req_by_name[candidate["category"]],
                    })
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    req_by_name[candidate["category"]]["binding_failures"].append({
                        "candidate_id": candidate["candidate_id"],
                        "error": row["error"], "blocking": True})
        elapsed = time.perf_counter() - before
        for candidate in batch:
            rows_by_id[candidate["candidate_id"]]["sam2_frame_elapsed_s"] = elapsed

    relation_input = []
    stage_timings["sam2_s"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    all_observations = observations
    observations, review_groups = _select_review_observations(
        all_observations, depth=np.asarray(depth_maps),
        c2w=np.asarray(c2w_list), intrinsics=intrinsics, task=question_type)
    stats["review_selection"] = {
        "n_observations": len(all_observations), "n_groups": len(review_groups),
        "n_selected": len(observations),
        "selected_candidate_ids": [o["candidate"]["candidate_id"] for o in observations],
        "unreviewed_masks_admitted": False,
    }
    stage_timings["review_grouping_s"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    progress("semantic_review", n_observations=len(all_observations),
             n_groups=len(review_groups), n_selected=len(observations), n_reviewed=0)
    binary_mode = bool(getattr(visual_client, "target_binary", False))
    review_kwargs = dict(
        seed=seed, min_pixels=parameters["semantic_min_pixels"],
        max_pixels=parameters["semantic_max_pixels"],
        out_dir=(work / "semantic_reviews") if work else None)
    if hasattr(visual_client, "observe_candidates"):
        visual_results = visual_client.observe_candidates(observations, frames, **review_kwargs)
    else:
        visual_results = (
            msr.observe_candidate(
                observation["candidate"],
                frames[observation["candidate"]["frame_slot"]],
                observation["raw_mask"], visual_client, **review_kwargs)
            for observation in observations)
    for review_index, (observation, visual) in enumerate(
            zip(observations, visual_results, strict=True), start=1):
        candidate, row = observation["candidate"], observation["row"]
        row["semantic_visual"] = visual
        progress("semantic_review", n_observations=len(all_observations),
                 n_groups=len(review_groups), n_selected=len(observations),
                 n_reviewed=review_index)
        if visual["status"] == "observed" and not binary_mode:
            relation_input.append({
                "candidate_id": candidate["candidate_id"],
                "category": candidate["category"],
                "observation": visual["observation"],
            })
        elif visual["status"] != "observed":
            row["error"] = visual.get("error", "semantic visual review unavailable")
            observation["req"]["binding_failures"].append({
                "candidate_id": candidate["candidate_id"],
                "error": row["error"], "blocking": True})
    if binary_mode:
        from .intern_binary_review import binary_relation
        relations = {cid: binary_relation(row.get("semantic_visual") or {})
                     for cid, row in rows_by_id.items()}
        relation_stats = {"actual_requests": 0, "n_batches": 0}
    elif (semantic_review_config or {}).get("relation") == "frozen-rules":
        relations, relation_stats = compare_fixed_categories(
            relation_input, out_dir=(work / "semantic_reviews") if work else None)
    else:
        relations, relation_stats = msr.compare_categories(
            relation_input, vlm_client, seed=seed,
            batch_size=parameters["semantic_relation_batch_size"],
            out_dir=(work / "semantic_reviews") if work else None)
    stats["semantic_review"] = {
        "visual_version": parameters.get("experimental_visual_protocol_sha256", msr.VISUAL_REVIEW_VERSION),
        "experimental_config": semantic_review_config or {},
        "relation_version": ("m5-target-binary-v1" if binary_mode else RELATION_VERSION if (semantic_review_config or {}).get("relation")
                             == "frozen-rules" else msr.RELATION_REVIEW_VERSION),
        "n_visual_requests": sum(bool(
            row.get("semantic_visual", {}).get("request_sent"))
            for row in rows_by_id.values()),
        "n_relation_requests": relation_stats["actual_requests"],
        "n_relation_batches": relation_stats["n_batches"],
        "admission_policy": (semantic_review_config or {}).get("admission_policy", "frozen-quality")
                            if binary_mode else "visual-quality-and-relation",
        "quality_assessed": not (binary_mode and (semantic_review_config or {}).get(
            "admission_policy") == "category-only"),
    }
    stats["actual_model_requests"] += (
        stats["semantic_review"]["n_visual_requests"]
        + stats["semantic_review"]["n_relation_requests"])
    stage_timings["semantic_review_s"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()

    candidates, masks, accepted, accepted_slots = [], [], [], []
    for observation in observations:
        candidate, row = observation["candidate"], observation["row"]
        visual = row.get("semantic_visual") or {}
        if visual.get("status") != "observed":
            continue
        relation = relations.get(candidate["candidate_id"], {
            "status": "unavailable", "relation": "uncertain"})
        row["semantic_relation"] = relation
        quality = visual["observation"]
        category_only = binary_mode and (semantic_review_config or {}).get("admission_policy") == "category-only"
        if category_only:
            from .intern_binary_review import admission_identity
            evidence_ok = (visual.get("admission") == admission_identity("category-only")
                           and quality == {} and not visual.get("frozen_quality"))
        else:
            evidence_ok = ((binary_mode or quality.get("observed_object") not in {None, "other", "unknown"})
                          and quality.get("visibility") in {"clear", "partial"}
                          and quality.get("mask_purity") == "clean"
                          and quality.get("mask_completeness") in {"whole", "major_part"})
        semantic_ok = (
            relation.get("status") == "reviewed"
            and relation.get("relation") == "match"
            and evidence_ok)
        if not semantic_ok:
            if relation.get("status") == "reviewed" and relation.get(
                    "relation") == "mismatch":
                row["status"] = "semantic_rejected"
                observation["req"].setdefault("semantic_rejections", []).append(
                    candidate["candidate_id"])
                continue
            row["status"] = "semantic_unavailable"
            row["error"] = (
                relation.get("error")
                or f"semantic relation={relation.get('relation')}; "
                   f"visibility={quality.get('visibility', 'not_assessed')}; "
                   f"purity={quality.get('mask_purity', 'not_assessed')}; "
                   f"completeness={quality.get('mask_completeness', 'not_assessed')}")
            observation["req"]["binding_failures"].append({
                "candidate_id": candidate["candidate_id"],
                "error": row["error"], "blocking": True})
            continue
        before = time.perf_counter()
        try:
            bound = st.bind_masks_to_world(
                [observation["mapped"]], depth_maps, c2w_list, intrinsics,
                class_hints=[candidate["category"]], confidences=[candidate["score"]],
                det_conf_sources=["groundingdino"], max_points=PARAMETERS["max_points"],
                point_conf=point_conf, grounding_status="question_driven",
                track_prefix=f"qtrk:{key}:{candidate['candidate_id']}:",
                out_dir=work / "candidates" / candidate["candidate_id"] if work else None)
            provenance = {
                **candidate,
                "semantic_visual_request_sha256": visual["request_sha256"],
                "semantic_observation": quality,
                "semantic_provider": visual.get("provider", "bailian"),
                "semantic_binary_review": visual.get("binary_review", {}),
                "semantic_binary_review_sha256": visual.get("binary_review_sha256", ""),
                "semantic_frozen_quality": visual.get("frozen_quality", {}),
                "semantic_admission": visual.get("admission", {}),
                "semantic_binary_response": visual.get("raw_response", {}) if category_only else {},
                "semantic_binary_response_sha256": content_sha256(visual.get("raw_response", {}))
                                                  if category_only else "",
                "semantic_request_identity": visual.get("request_identity", {}),
                "semantic_observation_sha256": visual.get("observation_sha256", ""),
                "semantic_model_identity_sha256": visual.get("model_identity_sha256", ""),
                "semantic_relation": relation["relation"],
                "semantic_relation_request_sha256": relation["request_sha256"],
            }
            obj = bound[0].model_copy(update={
                "candidate_ids": [candidate["candidate_id"]],
                "source_label": candidate["source_label"],
                "question_roles": observation["req"]["roles"],
                "binding_provenance": [provenance]})
            verdict = validate_binding(
                obj, depth_maps=depth_maps, n_frames=len(frames), generation_sha256=key)
            if verdict.status != "validated":
                raise ValueError("candidate binding: " + ", ".join(verdict.errors))
            candidates.append(obj)
            masks.append(observation["mapped"])
            accepted.append(candidate)
            accepted_slots.append(candidate["frame_slot"])
            row.update(status="candidate_usable", visible_frames=obj.visible_frames)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            observation["req"]["binding_failures"].append({
                "candidate_id": candidate["candidate_id"],
                "error": row["error"], "blocking": True})
        row["binding_elapsed_s"] = time.perf_counter() - before
    stats["candidate_receipts"] = [
        rows_by_id[row["candidate_id"]] for row in selected]
    for row in stats["candidate_receipts"]:
        row["elapsed_s"] = (
            float(row.get("sam2_frame_elapsed_s", 0.0))
            + float((row.get("semantic_visual") or {}).get("elapsed_s", 0.0))
            + float(row.get("binding_elapsed_s", 0.0)))
    groups, merged, suspect = _merge_instances(
        candidates, masks, depth_maps, frame_slots=accepted_slots)
    objects = st.bind_masks_to_world(
        merged, depth_maps, c2w_list, intrinsics,
        class_hints=[candidates[g[0]].category_name for g in groups],
        confidences=[max(candidates[i].det_conf for i in g) for g in groups],
        det_conf_sources=["groundingdino"] * len(groups),
        max_points=PARAMETERS["max_points"], point_conf=point_conf,
        out_dir=work / "objects" if work else None,
        grounding_status="question_driven", track_prefix=f"qtrk:{key}:")
    for index, (obj, group) in enumerate(zip(objects, groups)):
        objects[index] = obj.model_copy(update={
            "candidate_ids": [accepted[i]["candidate_id"] for i in group],
            "source_label": candidates[group[0]].source_label,
            "question_roles": candidates[group[0]].question_roles,
            "binding_provenance": [
                candidates[i].binding_provenance[0] for i in group],
            "duplicate_suspect": index in suspect})
    audited, audits = st.audit_semantic_mask_conflicts(
        [(str(qa_id), objects)], scene_identity=str(scene_identity or scene_name))
    objects = audited[str(qa_id)]
    for index, obj in enumerate(objects):
        verdict = validate_binding(
            obj, depth_maps=depth_maps, n_frames=len(frames), generation_sha256=key)
        objects[index] = obj.model_copy(update={"binding_validation": verdict})
    stats["semantic_conflict_audits"] = audits
    stats["grounding"] = refresh_coverage({
        "version": M5_BINDING_VERSION, "attempted": True, "requirements": requirements,
        "required_names": categories, "readable_slots": readable,
        "question_type": question_type, "candidate_names": [
            r["category"] for r in parsed if r["role"] == "candidate"],
        "request_identity": key,
    }, objects)
    failures = sum(len(r["binding_failures"]) for r in requirements)
    rejected = sum(len(r.get("semantic_rejections", [])) for r in requirements)
    stats["track_consensus"] = {
        "version": M5_BINDING_VERSION, "scope": "question_binding",
        "n_propagations": stats["n_propagations"], "n_candidates": len(selected),
        "n_failed_candidates": failures, "n_semantic_rejected": rejected,
        "n_dedup_absorbed": len(candidates) - len(objects),
        # Normal duplicate observations are not track fragmentation.
        "track_fragmentation_ratio": failures / len(selected) if selected else None,
        "duplicate_suspect_ratio": len(suspect) / len(objects) if objects else None,
        "n_objects": len(objects), "n_suspect": len(suspect),
    }
    stats.update(materialized=True, status=stats["grounding"]["status"],
                 elapsed_s=time.perf_counter() - started,
                 track_stable_ratio=st.track_stable_ratio(objects, len(frames)))
    from skill3d.reconstruction_gate.confidence_map import track_ious_from_masks
    stats["track_ious"] = track_ious_from_masks(merged)
    dynamic, dynamic_notes = st.dynamic_mask_from_rigidity(
        frames, depth_maps, c2w_list, intrinsics)
    stats["g7_dynamic_ratio"] = float(dynamic.mean()) if dynamic.size else None
    stats["evidence_notes"] = dynamic_notes
    stats["actual_sam2_tracks"] = 0
    stats["actual_sam2_image_frames"] = stats["n_sam2_image_frames"]
    stats["actual_sam2_masks"] = stats["n_sam2_masks"]
    stage_timings["binding_finalize_s"] = time.perf_counter() - stage_started
    stats["elapsed_s"] = time.perf_counter() - started
    progress("completed", status=stats["status"], n_objects=len(objects),
             n_detector_requests=stats["actual_detector_requests"],
             n_visual_requests=stats["semantic_review"]["n_visual_requests"])
    if cache and work:
        rows = [obj.model_dump(mode="json") for obj in objects]
        for row in rows:
            for field in ("mask_per_frame", "pointcloud_world", "pointconf_world"):
                if row.get(field):
                    row[field] = str(Path(row[field]).relative_to(root))
        _save_envelope(cache, key, {"objects": rows, "stats": stats})
    return objects, stats, [
        f"M5 本题 {len(categories)} 类×{len(readable)} 帧全扫描；"
        f"{len(selected)} 候选→{sum(binding_is_usable(o) for o in objects)} 可用实例；"
        f"未解决={stats['grounding']['unresolved_names']}"]
