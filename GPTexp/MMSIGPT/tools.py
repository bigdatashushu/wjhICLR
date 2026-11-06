"""MMSI task authorization over the shared small-image tools."""
from __future__ import annotations

import hashlib
import json
from types import MethodType

TOOL_VERSION = "mmsigpt-tools/2"


def _sample(handle, view_index, xy):
    """Count depth samples independently of optional finite camera points."""
    import numpy as np
    from skill3d.tools.contract import ConfidenceGateError, DomainValueError

    geometry, depth, valid = handle.geometry(view_index)
    if (not isinstance(xy, (tuple, list)) or len(xy) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not np.isfinite(value) for value in xy)):
        raise DomainValueError("small_image", "pixel must be a finite [x,y] pair")
    x, y = map(float, xy)
    h, w = depth.shape
    if not 0 <= x < w or not 0 <= y < h:
        raise DomainValueError("small_image", "pixel lies outside original image")
    ix, iy = min(int(round(x)), w - 1), min(int(round(y)), h - 1)
    ys = slice(max(0, iy - 2), min(h, iy + 3))
    xs = slice(max(0, ix - 2), min(w, ix + 3))
    selected = valid[ys, xs]
    count = int(np.count_nonzero(selected))
    if count < 3:
        raise ConfidenceGateError("small_image", "insufficient finite local depth samples")
    z = float(np.median(depth[ys, xs][selected]))
    points = geometry.get("points")
    point = None
    if points is not None:
        points = np.asarray(points)
        if points.shape == (h, w, 3):
            patch = points[ys, xs]
            point_selected = selected & np.all(np.isfinite(patch), axis=-1)
            if np.count_nonzero(point_selected) >= 3:
                point = np.median(patch[point_selected], axis=0)
    return z, point, count


def _point_views(scene):
    import numpy as np
    from skill3d.tools.contract import ToolContractError

    views = set()
    for index in scene.geometries:
        try:
            geometry, depth, valid = scene.geometry(index)
            points = geometry.get("points")
            if points is None:
                continue
            points = np.asarray(points)
            if (points.shape == (*depth.shape, 3)
                    and np.count_nonzero(valid & np.all(np.isfinite(points), axis=-1)) >= 3):
                views.add(index)
        except (ValueError, TypeError, ToolContractError):
            continue
    return frozenset(views)


def registry_for(allowed):
    """Preserve the frozen task allowlist and underlying geometry checks."""
    from skill3d.schemas.authorization import AuthorizationDecision, not_applicable_gate
    from skill3d.small_image.tools import build_registry

    allowed = frozenset(allowed)

    def make(prepared, problem):
        registry, scene = build_registry(prepared, problem)
        scene.sample = MethodType(_sample, scene)
        point_views = _point_views(scene)
        scene.evidence_profile.profile_version = TOOL_VERSION
        scene._digest = hashlib.sha256(json.dumps({
            "shared_state_sha256": scene.state_digest(),
            "mmsi_tools_version": TOOL_VERSION,
        }, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
        original = registry._authorizer

        def authorize(name, spec, **context):
            if name not in allowed:
                return AuthorizationDecision(
                    allowed=False,
                    reason_codes=["task_denied"],
                    missing_refs=["task_visible_tool"],
                    dependency_refs=[],
                    metric_gate_result=not_applicable_gate(),
                )
            decision = original(name, spec, **context)
            if name == "pair_point_distance" and decision.allowed:
                args = context.get("args") or {}
                missing = not point_views
                if "view_index" in args:
                    missing = type(args["view_index"]) is not int or args["view_index"] not in point_views
                if missing:
                    return AuthorizationDecision(
                        allowed=False,
                        reason_codes=["confidence_gate"],
                        missing_refs=["finite_predicted_points_for_view"],
                        dependency_refs=list(decision.dependency_refs),
                        metric_gate_result=decision.metric_gate_result,
                    )
            return decision

        registry._authorizer = authorize
        registry._model_tools = allowed
        registry.decision_version = TOOL_VERSION
        return registry, scene

    return make
