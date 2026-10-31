"""Instance-local tools for monocular predictions on original image grids."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np

from skill3d.schemas import ToolSpec
from skill3d.schemas.authorization import AuthorizationDecision, not_applicable_gate
from skill3d.tools.contract import ConfidenceGateError, DomainValueError
from skill3d.tools.registry import ToolRegistry

VERSION = "small-image-tools/1"
TOOL_NAMES = frozenset({
    "view_candidates", "view_depth_summary", "point_depth",
    "pair_point_distance", "candidate_depth_summary",
})
PUBLIC_FIELDS = {
    "view_candidates": {"candidates", "count", "coverage", "coordinate_convention"},
    "view_depth_summary": {"views", "unit", "coordinate_convention"},
    "point_depth": {"view_index", "pixel_xy", "depth", "unit", "n_valid",
                    "definition", "coordinate_convention"},
    "pair_point_distance": {"view_index", "a_xy", "b_xy", "distance", "unit",
                            "definition", "coordinate_convention"},
    "candidate_depth_summary": {"candidate_id", "view_index", "median_depth",
                               "n_valid", "unit", "definition"},
}
COORDINATES = "original image pixels: x right, y down; views remain independent"


def _digest(value):
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, np.ndarray):
            digest.update(str((item.dtype.str, item.shape)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                add(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                add(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode())
    add(value)
    return digest.hexdigest()


class SmallImageHandle:
    """Host-owned data; kernel exposes only its existing frozen context view."""
    frame = "per_view_camera"
    scene_route = "small_image"
    question_tool_scope = "small_image"
    metric_gate_passed = None
    metric_evidence_gate_result = None
    allowed_metric_tasks = ()
    local_facts = {}
    question_type = "small_image_option"

    def __init__(self, prepared, problem):
        self.frames = tuple(np.array(frame, copy=True) for frame in prepared.frames)
        self.candidates = copy.deepcopy(prepared.candidates)
        views = getattr(prepared, "geometries", None)
        if views is None:
            views = {index: {
                "depth": depth, "valid_mask": prepared.geometry_masks[index],
                "points": prepared.points.get(index),
                "intrinsics": prepared.intrinsics.get(index),
            } for index, depth in prepared.depths.items()}
        self.geometries = {index: {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray)
            else copy.deepcopy(value)
            for key, value in geometry.items()
        } for index, geometry in views.items()}
        self.input_sha256 = problem.input_sha256
        self.summary = f"{problem.benchmark}; {len(self.frames)} ordered images; {COORDINATES}"
        self.evidence_profile = SimpleNamespace(profile_version=VERSION)
        self.available_artifacts = frozenset(
            {"frames", "objects"} | ({"depth"} if self.geometries else set()))
        self._consumed_object_ids = []
        self._resolved_object_ids = []
        self._consumed_dependency_refs = []
        self._digest = _digest({
            "input": self.input_sha256, "frames": self.frames,
            "candidates": self.candidates, "geometry": self.geometries,
            "provenance": prepared.provenance, "tools": VERSION,
        })

    def state_digest(self):
        return self._digest

    def geometry(self, view_index):
        if type(view_index) is not int or not 0 <= view_index < len(self.frames):
            raise DomainValueError("small_image", "view_index is outside original image order")
        geometry = self.geometries.get(view_index)
        if geometry is None:
            raise ConfidenceGateError("small_image", "predicted depth unavailable for this view")
        depth = np.asarray(geometry["depth"])
        valid = np.asarray(geometry["valid_mask"], dtype=bool)
        if depth.shape != self.frames[view_index].shape[:2] or valid.shape != depth.shape:
            raise ConfidenceGateError("small_image", "geometry must match original image grid")
        return geometry, depth, valid & np.isfinite(depth) & (depth > 0)

    def sample(self, view_index, xy):
        geometry, depth, valid = self.geometry(view_index)
        if (not isinstance(xy, (tuple, list)) or len(xy) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not np.isfinite(v) for v in xy)):
            raise DomainValueError("small_image", "pixel must be a finite [x,y] pair")
        x, y = map(float, xy)
        h, w = depth.shape
        if not 0 <= x < w or not 0 <= y < h:
            raise DomainValueError("small_image", "pixel lies outside original image")
        ix, iy = int(round(x)), int(round(y))
        ix, iy = min(ix, w - 1), min(iy, h - 1)
        ys, xs = slice(max(0, iy - 2), min(h, iy + 3)), slice(max(0, ix - 2), min(w, ix + 3))
        selected = valid[ys, xs]
        if np.count_nonzero(selected) < 3:
            raise ConfidenceGateError("small_image", "insufficient finite local depth samples")
        z = float(np.median(depth[ys, xs][selected]))
        points = geometry.get("points")
        point = None
        if points is not None:
            points = np.asarray(points)
            if points.shape != (h, w, 3):
                raise ConfidenceGateError("small_image", "point map grid mismatch")
            patch = points[ys, xs]
            selected = selected & np.all(np.isfinite(patch), axis=-1)
            if np.count_nonzero(selected) >= 3:
                point = np.median(patch[selected], axis=0)
        return z, point, int(np.count_nonzero(selected))


def build_registry(prepared, problem):
    scene = SmallImageHandle(prepared, problem)
    def authorize(name, spec, **context):
        missing = sorted(set(spec.requires_artifacts) -
                         set(context.get("available_artifacts", ())))
        reasons = []
        if name not in TOOL_NAMES:
            reasons.append("task_denied")
        if context.get("scope") != "small_image":
            reasons.append("scope_denied")
        if not context.get("tools_enabled", True):
            reasons.append("tools_disabled")
        args = context.get("args") or {}
        if "depth" in spec.requires_artifacts and "view_index" in args:
            index = args["view_index"]
            if type(index) is not int or index not in scene.geometries:
                missing.append("predicted_depth_for_view")
        if name == "candidate_depth_summary":
            # This minimal pipeline has no independent mask quality receipt.
            # Mutable candidate flags cannot manufacture that evidence.
            missing.append("independently_validated_candidate_geometry")
        if missing:
            reasons.append("confidence_gate")
        dependencies = sorted(set(spec.requires_artifacts) |
                              set(context.get("consumed_dependency_refs", ())))
        return AuthorizationDecision(
            allowed=not (missing or reasons), reason_codes=reasons,
            missing_refs=missing, dependency_refs=dependencies,
            metric_gate_result=not_applicable_gate())

    registry = ToolRegistry(
        authorizer=authorize, model_tools=TOOL_NAMES,
        public_fields=PUBLIC_FIELDS, decision_version=VERSION)

    def register(name, description, artifacts):
        return registry.register(ToolSpec(
            name=name, description=description, args_schema_ref=VERSION,
            returns_schema_ref=VERSION, cost_estimate_ms=1, source_default="real",
            requires_artifacts=artifacts, requires_evidence=[]))

    @register("view_candidates",
              "Inspect detected regions and category review. Category confirmation does not certify mask quality, physical counts or cross-view identity.",
              ["frames", "objects"])
    def view_candidates(handle, view_index=-1, category=""):
        if type(view_index) is not int or not -1 <= view_index < len(handle.frames):
            raise DomainValueError("view_candidates", "invalid view_index")
        rows = []
        for c in handle.candidates:
            if (view_index >= 0 and c["view_index"] != view_index) or (
                    category and c["category"] != category):
                continue
            rows.append({key: c[key] for key in (
                "candidate_id", "view_index", "category", "box_xyxy",
                "category_confirmed", "mask_quality_assessed", "geometry_usable",
                "review_status") if key in c})
        return {"candidates": rows, "count": len(rows), "coverage": "detections only; not exhaustive",
                "coordinate_convention": COORDINATES}

    @register("view_depth_summary",
              "Per-view predicted depth availability and quantiles; no cross-view registration or metric scale.",
              ["frames"])
    def view_depth_summary(handle):
        rows = []
        for index in range(len(handle.frames)):
            row = {"view_index": index, "available": index in handle.geometries}
            if row["available"]:
                _, depth, valid = handle.geometry(index)
                values = depth[valid]
                row.update(n_valid=int(values.size),
                           quantiles=np.quantile(values, [.1, .5, .9]).tolist()
                           if values.size else [])
            rows.append(row)
        return {"views": rows, "unit": "per_view_relative", "coordinate_convention": COORDINATES}

    @register("point_depth",
              "Predicted depth at original pixel (x,y), robust median of finite 5x5 neighborhood. Compare only within one view.",
              ["frames", "depth"])
    def point_depth(handle, view_index, x, y):
        depth, _, count = handle.sample(view_index, [x, y])
        handle._consumed_dependency_refs.append(f"view:{view_index}:predicted_depth")
        return {"view_index": view_index, "pixel_xy": [x, y], "depth": depth,
                "n_valid": count, "unit": "per_view_relative",
                "definition": "local median of predicted camera-depth, radius 2 pixels",
                "coordinate_convention": COORDINATES}

    @register("pair_point_distance",
              "Predicted Euclidean distance between two original pixels in the SAME camera view, with independently predicted scale.",
              ["frames", "depth"])
    def pair_point_distance(handle, view_index, a_xy, b_xy):
        _, a, _ = handle.sample(view_index, a_xy)
        _, b, _ = handle.sample(view_index, b_xy)
        if a is None or b is None:
            raise ConfidenceGateError("pair_point_distance", "finite point map unavailable")
        handle._consumed_dependency_refs.append(f"view:{view_index}:predicted_points")
        return {"view_index": view_index, "a_xy": list(a_xy), "b_xy": list(b_xy),
                "distance": float(np.linalg.norm(a - b)), "unit": "per_view_relative",
                "definition": "Euclidean distance of local median camera points",
                "coordinate_convention": COORDINATES}

    @register("candidate_depth_summary",
              "Diagnostic mask depth only after independent mask/geometry validation. Intern category-only review cannot authorize this measurement.",
              ["frames", "objects", "depth"])
    def candidate_depth_summary(handle, candidate_id):
        raise ConfidenceGateError(
            "candidate_depth_summary", "independent mask quality evidence is unavailable")

    return registry, scene
