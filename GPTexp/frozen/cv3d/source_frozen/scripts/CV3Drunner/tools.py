"""Instance-local box-relative tool surface with unchanged numerical sampling."""
import copy

import numpy as np

from skill3d.schemas import ToolSpec
from skill3d.schemas.authorization import AuthorizationDecision, not_applicable_gate
from skill3d.small_image.tools import SmallImageHandle, COORDINATES
from skill3d.tools.contract import ConfidenceGateError, DomainValueError
from skill3d.tools.registry import ToolRegistry

from .grounding import AnnotationCatalog, digest

VERSION = "cv3d-box-tools/1"
MEASUREMENTS = frozenset({"point_depth_in_box", "pair_distance_in_boxes"})
TOOL_NAMES = frozenset({"view_annotations", "view_depth_summary", *MEASUREMENTS})
PUBLIC_FIELDS = {
    "view_annotations": {"catalog", "coordinate_convention"},
    "view_depth_summary": {"views", "unit", "coordinate_convention"},
    "point_depth_in_box": {
        "view_index", "box_id", "color", "uv", "pixel_xy", "depth", "n_valid",
        "unit", "definition", "coordinate_convention", "surface_validated"},
    "pair_distance_in_boxes": {
        "view_index", "a_box_id", "b_box_id", "a_color", "b_color",
        "a_uv", "b_uv", "a_xy", "b_xy", "a_n_valid", "b_n_valid", "distance",
        "unit", "definition", "coordinate_convention", "surface_validated"},
}


class BoxHandle(SmallImageHandle):
    def __init__(self, prepared, problem):
        if problem.benchmark != "cv3d" or len(prepared.frames) != 1:
            raise ValueError("CV3Drunner only accepts one-image CV3D")
        super().__init__(prepared, problem)
        self.catalog = AnnotationCatalog(self.frames[0], problem.question, problem.task_type)
        self.task = problem.task_type
        self._digest = digest({
            "parent": super().state_digest(), "catalog": self.catalog.identity, "tools": VERSION})
        self.evidence_profile.profile_version = VERSION

    def box_pixel(self, view_index, box_id, uv):
        if type(view_index) is not int or view_index != 0:
            raise DomainValueError("box_pixel", "CV3D has only view_index=0")
        if (not self.catalog.measurements_available or not isinstance(box_id, str)
                or box_id not in self.catalog.boxes
                or self.catalog.boxes[box_id].color not in self.catalog.required_colors):
            raise ConfidenceGateError("box_pixel", "unique question-bound rectangle required")
        if (not isinstance(uv, (tuple, list)) or len(uv) != 2
                or any(type(v) not in (int, float) or not np.isfinite(v)
                       or not 0 <= v <= 1 for v in uv)):
            raise DomainValueError("box_pixel", "uv must be two finite numbers in [0,1]")
        box = self.catalog.boxes[box_id]
        x0, y0, x1, y1 = box.safe_xyxy
        x, y = x0 + uv[0] * (x1 - x0), y0 + uv[1] * (y1 - y0)
        return box, [float(x), float(y)]


def allowed_tools(registry, scene):
    return set(registry.names_for_scope(
        scene.question_tool_scope, scene.available_artifacts,
        evidence_profile=scene.evidence_profile, question_type=scene.question_type,
        local_facts=scene.local_facts))


def build_registry(prepared, problem):
    scene = BoxHandle(prepared, problem)

    def authorize(name, spec, **context):
        missing = sorted(set(spec.requires_artifacts) -
                         set(context.get("available_artifacts", ())))
        reasons = []
        if name not in TOOL_NAMES or (
                name == "pair_distance_in_boxes" and scene.task != "Distance"):
            reasons.append("task_denied")
        if context.get("scope") != "small_image":
            reasons.append("scope_denied")
        if not context.get("tools_enabled", True):
            reasons.append("tools_disabled")
        if name in MEASUREMENTS:
            if not scene.catalog.measurements_available:
                missing.append("unique_question_annotation_boxes")
            if 0 not in scene.geometries:
                missing.append("predicted_depth_for_view")
            if (name == "pair_distance_in_boxes" and (
                    0 not in scene.geometries or scene.geometries[0].get("points") is None)):
                missing.append("predicted_points_for_view")
        if missing:
            reasons.append("confidence_gate")
        return AuthorizationDecision(
            allowed=not (missing or reasons), reason_codes=reasons, missing_refs=missing,
            dependency_refs=sorted(set(spec.requires_artifacts) | (
                {f"annotations:{scene.catalog.identity}"} if name in MEASUREMENTS else set())
                | set(context.get("consumed_dependency_refs", ()))),
            metric_gate_result=not_applicable_gate())

    registry = ToolRegistry(
        authorizer=authorize, model_tools=TOOL_NAMES,
        public_fields=PUBLIC_FIELDS, decision_version=VERSION)

    def register(name, description, artifacts):
        return registry.register(ToolSpec(
            name=name, description=description, args_schema_ref=VERSION,
            returns_schema_ref=VERSION, cost_estimate_ms=1, source_default="real",
            requires_artifacts=artifacts, requires_evidence=[]))

    @register("view_annotations", "Inspect host-extracted boxes and question color bindings.", ["frames"])
    def view_annotations(handle):
        return {"catalog": copy.deepcopy(handle.catalog.public()), "coordinate_convention": COORDINATES}

    @register("view_depth_summary", "Inspect per-view predicted depth availability and 10/50/90 percentiles.", ["frames"])
    def view_depth_summary(handle):
        row = {"view_index": 0, "available": 0 in handle.geometries}
        if row["available"]:
            _, depth, valid = handle.geometry(0)
            values = depth[valid]
            row.update(n_valid=int(values.size),
                       quantiles=np.quantile(values, [.1, .5, .9]).tolist() if values.size else [])
        return {"views": [row], "unit": "per_view_relative", "coordinate_convention": COORDINATES}

    @register("point_depth_in_box",
              "Sample a visible object surface using box_id and normalized u,v in its stroke-safe interior; returns pixel_xy, depth and n_valid.", ["frames", "depth"])
    def point_depth_in_box(handle, view_index, box_id, u, v):
        box, xy = handle.box_pixel(view_index, box_id, [u, v])
        depth, _, count = handle.sample(view_index, xy)
        handle._consumed_dependency_refs.append("view:0:predicted_depth")
        return {
            "view_index": view_index, "box_id": box_id, "color": box.color,
            "uv": [u, v], "pixel_xy": xy, "depth": depth, "n_valid": count,
            "unit": "per_view_relative", "surface_validated": False,
            "definition": "local median of predicted camera-depth, radius 2 pixels",
            "coordinate_convention": COORDINATES}

    @register("pair_distance_in_boxes",
              "Euclidean distance between local median camera points in two different question boxes; a_uv,b_uv are normalized pairs in their safe interiors.", ["frames", "depth"])
    def pair_distance_in_boxes(handle, view_index, a_box_id, a_uv, b_box_id, b_uv):
        if a_box_id == b_box_id:
            raise DomainValueError("pair_distance_in_boxes", "two different question boxes required")
        a_box, a_xy = handle.box_pixel(view_index, a_box_id, a_uv)
        b_box, b_xy = handle.box_pixel(view_index, b_box_id, b_uv)
        _, a, na = handle.sample(view_index, a_xy)
        _, b, nb = handle.sample(view_index, b_xy)
        if a is None or b is None:
            raise ConfidenceGateError("pair_distance_in_boxes", "finite point map unavailable")
        handle._consumed_dependency_refs.append("view:0:predicted_points")
        return {
            "view_index": view_index, "a_box_id": a_box_id, "b_box_id": b_box_id,
            "a_color": a_box.color, "b_color": b_box.color,
            "a_uv": list(a_uv), "b_uv": list(b_uv), "a_xy": a_xy, "b_xy": b_xy,
            "a_n_valid": na, "b_n_valid": nb,
            "distance": float(np.linalg.norm(a - b)), "unit": "per_view_relative",
            "surface_validated": False,
            "definition": "Euclidean distance of local median camera points",
            "coordinate_convention": COORDINATES}

    return registry, scene
