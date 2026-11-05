"""Original-pixel depth samples and center-point distances; no aggregation."""
from __future__ import annotations

import numpy as np

from skill3d.tools.contract import ConfidenceGateError
from .recovery import input_error

VERSION = "omni-raw-pixel-sampling/2"
EDGE_NOTE = ("At object edges, samples may hit other surfaces and differ greatly; "
             "identify which returned values belong to the target.")


def sample(handle, view_index, xy, field="pixel_xy"):
    geometry, depth, valid = handle.geometry(view_index)
    if (not isinstance(xy, (tuple, list)) or len(xy) != 2
            or any(type(v) not in (int, float) or not np.isfinite(v) for v in xy)):
        raise input_error("omni_sampling", field, "finite [x,y]", xy)
    x, y = xy
    h, w = depth.shape
    if not 0 <= x < w or not 0 <= y < h:
        raise input_error("omni_sampling", field, f"original pixels: 0<=x<{w}, 0<=y<{h}", xy)
    ix, iy = min(round(x), w-1), min(round(y), h-1)
    x0, y0, x1, y1 = max(0, ix-2), max(0, iy-2), min(w, ix+3), min(h, iy+3)
    mask = valid[y0:y1, x0:x1]
    if not mask.any():
        raise ConfidenceGateError("omni_sampling", "no finite local depth samples")
    values = [[float(depth[py, px]) if valid[py, px] else None
               for px in range(x0, x1)] for py in range(y0, y1)]
    center = float(depth[iy, ix]) if valid[iy, ix] else None
    return geometry, [ix, iy], center, {
        "origin_xy": [x0, y0], "values": values}, int(mask.sum())


def point_depth(handle, view_index, x, y):
    _, pixel, center, samples, count = sample(handle, view_index, [x, y])
    handle._consumed_dependency_refs.append(f"view:{view_index}:predicted_depth")
    return {"view_index": view_index, "pixel_xy": pixel, "depth": center,
            "depth_samples": samples, "n_valid": count, "unit": "per_view_relative"}


def pair_point_distance(handle, view_index, a_xy, b_xy):
    geometry, a_pixel, a_z, a_samples, _ = sample(handle, view_index, a_xy, "a_xy")
    _, b_pixel, b_z, b_samples, _ = sample(handle, view_index, b_xy, "b_xy")
    points = geometry.get("points")
    if points is None or np.shape(points) != (*handle.frames[view_index].shape[:2], 3):
        raise ConfidenceGateError("pair_point_distance", "point map unavailable or grid mismatch")
    a, b = np.asarray(points)[a_pixel[1], a_pixel[0]], np.asarray(points)[b_pixel[1], b_pixel[0]]
    distance = None
    if a_z is not None and b_z is not None and np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
        measured = float(np.linalg.norm(a-b))
        if np.isfinite(measured):
            distance = measured
    handle._consumed_dependency_refs.append(f"view:{view_index}:predicted_points")
    return {"view_index": view_index, "a_xy": a_pixel, "b_xy": b_pixel,
            "distance": distance, "a_depth_samples": a_samples, "b_depth_samples": b_samples,
            "unit": "per_view_relative"}


def depth_value(data, sample_xy=None):
    """Select an exact delivered value, never a neighborhood statistic."""
    if sample_xy is None:
        value = data["depth"]
    else:
        if (not isinstance(sample_xy, list) or len(sample_xy) != 2
                or any(type(v) is not int for v in sample_xy)):
            raise ValueError("calculation_sample_xy_invalid")
        samples = data["depth_samples"]
        x, y = sample_xy[0]-samples["origin_xy"][0], sample_xy[1]-samples["origin_xy"][1]
        values = samples["values"]
        if not 0 <= y < len(values) or not 0 <= x < len(values[y]):
            raise ValueError("calculation_sample_not_delivered")
        value = values[y][x]
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise ValueError("calculation_sample_invalid")
    return value
