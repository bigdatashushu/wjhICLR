"""Strict, pickle-free public inputs for the action execution process."""
from __future__ import annotations

import base64
from dataclasses import fields
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

from action_contract import (DecisionSnapshot, EpisodeBinding, ExecutionEnvelope,
                             plain, public_hash, public_observation)
from protocol import ContractError, sha256

VERSION = "embodied-sandbox/1"
ARRAY_DTYPES = frozenset({"<f4", "|b1"})


def pack_array(array):
    if array is None:
        return None
    value = np.ascontiguousarray(array)
    if value.dtype.str not in ARRAY_DTYPES or value.nbytes > 4_000_000:
        raise ContractError("invalid_evidence_array")
    raw = value.tobytes()
    return {"dtype": value.dtype.str, "shape": list(value.shape),
            "sha256": sha256(raw), "base64": base64.b64encode(raw).decode("ascii")}


def unpack_array(value, *, shape, dtype):
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {"dtype", "shape", "sha256", "base64"}
            or value["dtype"] != dtype or value["shape"] != list(shape)
            or any(type(v) is not int for v in value["shape"])):
        raise ContractError("evidence_array_shape")
    try:
        raw = base64.b64decode(value["base64"], validate=True)
    except (ValueError, TypeError):
        raise ContractError("invalid_evidence_bytes") from None
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(raw) != expected or sha256(raw) != value["sha256"]:
        raise ContractError("evidence_array_hash")
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def pack_prepared(snapshot, prepared=None):
    """Forward no backend objects, paths, metadata or arbitrary candidate fields."""
    geometries, candidates = {}, []
    if prepared is not None:
        frames = prepared.frames
        if (len(frames) != 1 or np.asarray(frames[0]).dtype != np.uint8
                or sha256(np.ascontiguousarray(frames[0]).tobytes()) != snapshot.observation.pixel_sha256):
            raise ContractError("prepared_rgb_identity")
        views = getattr(prepared, "geometries", None)
        if views is None:
            views = {i: {"depth": d, "valid_mask": prepared.geometry_masks[i],
                         "points": prepared.points.get(i), "intrinsics": prepared.intrinsics.get(i)}
                     for i, d in prepared.depths.items()}
        if set(views) - {0}:
            raise ContractError("cross_view_evidence_denied")
        for index, geometry in views.items():
            if geometry.get("depth") is None or geometry.get("valid_mask") is None:
                raise ContractError("incomplete_evidence")
            geometries[str(index)] = {
                "depth": pack_array(np.asarray(geometry["depth"], dtype=np.float32)),
                "valid_mask": pack_array(np.asarray(geometry["valid_mask"], dtype=bool)),
                "points": pack_array(None if geometry.get("points") is None else
                                     np.asarray(geometry["points"], dtype=np.float32)),
            }
        for row in prepared.candidates:
            candidates.append({
                "candidate_id": row["candidate_id"], "view_index": row["view_index"],
                "category": row["category"], "box_xyxy": list(row["box_xyxy"]),
                "category_confirmed": row.get("category_confirmed") is True,
                "mask_quality_assessed": False, "geometry_usable": False,
                "review_status": str(row.get("review_status", "unreviewed"))[:200],
            })
    value = {"geometries": geometries, "candidates": candidates,
             "pixel_sha256": snapshot.observation.pixel_sha256,
             "unit": "per_view_relative", "metric_world": False}
    return {**value, "sha256": public_hash(value)}


def unpack_prepared(snapshot, value):
    if (type(value) is not dict or set(value) != {
            "geometries", "candidates", "pixel_sha256", "unit", "metric_world", "sha256"}
            or value["sha256"] != public_hash({k: v for k, v in value.items() if k != "sha256"})
            or value["pixel_sha256"] != snapshot.observation.pixel_sha256
            or value["unit"] != "per_view_relative" or value["metric_world"] is not False
            or type(value["geometries"]) is not dict or set(value["geometries"]) - {"0"}):
        raise ContractError("prepared_identity")
    width, height = snapshot.observation.rgb_size
    geometries = {}
    for index, row in value["geometries"].items():
        if type(row) is not dict or set(row) != {"depth", "valid_mask", "points"}:
            raise ContractError("unexpected_geometry_fields")
        geometry = {
            "depth": unpack_array(row["depth"], shape=(height, width), dtype="<f4"),
            "valid_mask": unpack_array(row["valid_mask"], shape=(height, width), dtype="|b1"),
            "points": unpack_array(row["points"], shape=(height, width, 3), dtype="<f4")}
        if geometry["depth"] is None or geometry["valid_mask"] is None:
            raise ContractError("incomplete_evidence")
        geometries[int(index)] = geometry
    candidates = value["candidates"]
    if geometries and "relative_depth" not in snapshot.capabilities:
        raise ContractError("undeclared_relative_depth")
    if any(row.get("points") is not None for row in geometries.values()) and (
            "relative_points" not in snapshot.capabilities):
        raise ContractError("undeclared_relative_points")
    if candidates and "candidate_category" not in snapshot.capabilities:
        raise ContractError("undeclared_candidates")
    if not isinstance(candidates, list) or len(candidates) > 128:
        raise ContractError("candidate_cap")
    ids = set()
    for row in candidates:
        if (type(row) is not dict or set(row) != {
                "candidate_id", "view_index", "category", "box_xyxy", "category_confirmed",
                "mask_quality_assessed", "geometry_usable", "review_status"}
                or type(row["view_index"]) is not int or row["view_index"] != 0
                or not isinstance(row["candidate_id"], str) or not 1 <= len(row["candidate_id"]) <= 200
                or row["candidate_id"] in ids or not isinstance(row["category"], str)
                or not 1 <= len(row["category"]) <= 200
                or type(row["category_confirmed"]) is not bool
                or row["mask_quality_assessed"] is not False or row["geometry_usable"] is not False
                or not isinstance(row["review_status"], str) or len(row["review_status"]) > 200
                or not isinstance(row["box_xyxy"], list) or len(row["box_xyxy"]) != 4
                or any(type(x) not in (int, float) or not np.isfinite(x)
                       for x in row["box_xyxy"])):
            raise ContractError("invalid_public_candidate")
        x1, y1, x2, y2 = row["box_xyxy"]
        if not (0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height):
            raise ContractError("candidate_coordinate_mismatch")
        ids.add(row["candidate_id"])
    with Image.open(io.BytesIO(snapshot.observation.png)) as image:
        rgb = np.array(image.convert("RGB"))
    return SimpleNamespace(frames=[rgb], geometries=geometries, candidates=candidates,
                           provenance={"public_evidence_sha256": value["sha256"]})


def pack_snapshot(snapshot, binding):
    value = {"binding": {f.name: getattr(binding, f.name) for f in fields(EpisodeBinding)},
            "snapshot": snapshot.public_dict(),
            "png_base64": base64.b64encode(snapshot.observation.png).decode("ascii")}
    if snapshot.observation.planner_png:
        value["planner_png_base64"] = base64.b64encode(snapshot.observation.planner_png).decode("ascii")
    return value


def unpack_snapshot(value):
    expected = {"binding", "snapshot", "png_base64"}
    if type(value) is not dict or set(value) not in (expected, expected | {"planner_png_base64"}):
        raise ContractError("unexpected_sandbox_context")
    try:
        binding = EpisodeBinding(**value["binding"])
        wire = value["snapshot"]
        obs = wire["observation"]
        png = base64.b64decode(value["png_base64"], validate=True)
        names = [row["action_name"] for row in obs["action_table"]]
        observation = public_observation(
            binding, png=png, instruction=obs["instruction"],
            environment_step=obs["environment_step"], mutation_revision=obs["mutation_revision"],
            observation_revision=obs["observation_revision"], action_table=names,
            public_feedback=obs["public_feedback"], public_memory=obs["public_memory"],
            manipulation_input=obs.get("manipulation_input"),
            planner_png=base64.b64decode(value["planner_png_base64"], validate=True)
            if "planner_png_base64" in value else None)
        result = DecisionSnapshot(
            observation, wire["decision_id"], wire["method_sha256"], wire["budget_sha256"],
            wire["evidence_revision"], wire["evidence_manifest_sha256"], tuple(wire["capabilities"]),
            wire["input_policy"], wire["execution_policy"])
    except (TypeError, KeyError, ValueError):
        raise ContractError("invalid_sandbox_snapshot") from None
    if result.public_dict() != wire:
        raise ContractError("sandbox_snapshot_identity")
    return binding, result


def envelope_dict(envelope):
    return {f.name: getattr(envelope, f.name) for f in fields(ExecutionEnvelope)}
