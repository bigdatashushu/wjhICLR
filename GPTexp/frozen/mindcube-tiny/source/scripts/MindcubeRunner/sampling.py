"""Instance-local pixel stability checks; stable sampling does not verify identity."""
from __future__ import annotations

from skill3d.tools.contract import ConfidenceGateError

VERSION = "mindcube-pixel-stability/1"
JITTER_PIXELS = 3
MAX_RELATIVE_DEPTH_RANGE = 0.10

EDGE_REASON = (
    "Unreliable due to edge sampling: nearby pixels vary sharply in depth, "
    "suggesting an object boundary or mixed surfaces. "
    "Choose a visible interior surface and sample again."
)
MISSING_REASON = (
    "The local neighbourhood has insufficient valid depth. "
    "Choose a visible interior surface with valid depth and sample again."
)


def stability(handle, view_index, xy):
    """Compare 5x5 depth medians at a 3x3 grid of nearby original pixels."""
    x, y = map(float, xy)
    h, w = handle.frames[view_index].shape[:2]
    depths, center, failed = [], None, 0
    for dy in (-JITTER_PIXELS, 0, JITTER_PIXELS):
        for dx in (-JITTER_PIXELS, 0, JITTER_PIXELS):
            point = [x + dx, y + dy]
            if not (0 <= point[0] < w and 0 <= point[1] < h):
                continue
            try:
                depth, _, _ = handle.sample(view_index, point)
            except ConfidenceGateError:
                failed += 1
                continue
            depths.append(depth)
            if dx == 0 and dy == 0:
                center = depth
    relative_range = ((max(depths) - min(depths)) / max(abs(center), 1e-9)
                      if depths and center is not None else None)
    reason = ("insufficient_local_depth" if failed or relative_range is None else
              "edge_sampling" if relative_range > MAX_RELATIVE_DEPTH_RANGE else None)
    return {
        "version": VERSION, "reliable": reason is None, "reason_code": reason,
        "jitter_pixels": JITTER_PIXELS, "probe_count": len(depths),
        "failed_probes": failed, "relative_depth_range": relative_range,
        "max_relative_depth_range": MAX_RELATIVE_DEPTH_RANGE,
    }


def unreliable(original, checks, *, endpoints=()):
    """Return quality feedback without publishing rejected depth or distance."""
    reason = ("edge_sampling" if any(check["reason_code"] == "edge_sampling"
                                    for check in checks.values())
              else "insufficient_local_depth")
    result = {key: value for key, value in original.items()
              if key not in {"depth", "distance"}}
    result.update(
        status="unreliable", reason_code=reason,
        reason=EDGE_REASON if reason == "edge_sampling" else MISSING_REASON,
        sample_stability=checks,
    )
    if endpoints:
        result["unreliable_endpoints"] = list(endpoints)
    return result


def install_sampling_checks(registry):
    """Wrap only this adapter's tool instances; internal probes cost no extra calls."""
    point_tool = registry.get("point_depth")
    distance_tool = registry.get("pair_point_distance")
    original_point, original_distance = point_tool.fn, distance_tool.fn

    def point_depth(handle, view_index, x, y):
        original = original_point(handle, view_index=view_index, x=x, y=y)
        check = stability(handle, view_index, [x, y])
        if not check["reliable"]:
            return unreliable(original, {"point": check})
        return {**original, "status": "ok", "sample_stability": {"point": check}}

    def pair_point_distance(handle, view_index, a_xy, b_xy):
        original = original_distance(handle, view_index=view_index, a_xy=a_xy, b_xy=b_xy)
        checks = {
            "a": stability(handle, view_index, a_xy),
            "b": stability(handle, view_index, b_xy),
        }
        rejected = [endpoint for endpoint, check in checks.items() if not check["reliable"]]
        if rejected:
            return unreliable(original, checks, endpoints=rejected)
        return {**original, "status": "ok", "sample_stability": checks}

    point_tool.fn, distance_tool.fn = point_depth, pair_point_distance
    quality_fields = {"status", "reason_code", "reason", "sample_stability", "unreliable_endpoints"}
    for name in ("point_depth", "pair_point_distance"):
        registry.public_fields[name] = registry.public_fields[name] | quality_fields
