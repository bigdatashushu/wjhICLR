"""Host-owned transforms for delivered images and original-pixel sampling."""
from __future__ import annotations

import hashlib
import math

from .recovery import input_error

VERSION = "omni-delivered-image-coordinates/3"
SURFACE_ALIASES = {"face_left": "left", "left_edge": "left",
                   "face_right": "right", "right_edge": "right"}
EXTENT_ROLES = ("top", "base", "left", "right", "front", "back", "rim_a", "rim_b")
LEGACY_EXTENT_ROLES = frozenset({"top", "base", "left", "right"})


def surface_name(value):
    return SURFACE_ALIASES.get(value, value) if isinstance(value, str) else value


def extent_requirement(record):
    """Only canonical legacy endpoints imply a role, never component/face names."""
    role = record.get("extent_role")
    if role is None:
        surface = surface_name(record.get("observed_surface", record.get("surface")))
        role = surface if isinstance(surface, str) and surface in LEGACY_EXTENT_ROLES else None
    return role, record.get("extent_basis")


def original_descriptor(rgb, width, height):
    return {"image_id": "rgb-original", "image_sha256": hashlib.sha256(rgb).hexdigest(),
            "width": width, "height": height, "view_index": 0,
            "source_box_xyxy": [0, 0, width, height], "offset_xy": [0, 0],
            "content_box_xyxy": [0, 0, width, height], "coordinate_version": VERSION}


def resolve(xy, image_id, space, images, field="pixel_xy"):
    if (not isinstance(xy, (list, tuple)) or len(xy) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy)):
        raise input_error("omni_coordinates", field, "finite [x,y]", xy)
    if space not in ("original_pixel", "image_pixel", "normalized"):
        raise input_error("omni_coordinates", "coordinate_space",
                          ["original_pixel", "image_pixel", "normalized"], space)
    if space == "original_pixel" and image_id is None:
        return list(xy), None
    if not isinstance(image_id, str) or image_id not in images:
        raise input_error("omni_coordinates", "image_id", "a currently delivered image ID", image_id)
    image = images[image_id]
    x, y = xy
    if space == "normalized":
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise input_error("omni_coordinates", field, "normalized x,y in [0,1]", xy)
        x, y = x * (image["width"]-1), y * (image["height"]-1)
    if space == "original_pixel" and image_id != "rgb-original":
        raise input_error("omni_coordinates", "coordinate_space",
                          "image_pixel or normalized for crops", space)
    x0, y0, x1, y1 = image["content_box_xyxy"]
    if not x0 <= x < x1 or not y0 <= y < y1:
        raise input_error("omni_coordinates", field,
                          f"delivered content: {x0}<=x<{x1}, {y0}<=y<{y1}", xy)
    source = image["source_box_xyxy"]
    mapped = [source[0]+x-x0, source[1]+y-y0]
    return mapped, {"version": VERSION, "image_id": image_id,
                    "image_sha256": image["image_sha256"], "coordinate_space": space,
                    "requested_xy": list(xy), "original_xy": mapped}


def intent(value, count):
    if value is None:
        return None
    required = {"target", "surface"}
    allowed = required | {"extent_role", "extent_basis"}
    if (not isinstance(value, list) or len(value) != count
            or any(not isinstance(r, dict) or not required <= set(r) or set(r) - allowed
                   or any(not isinstance(v, str) or not v.strip() or len(v) > 80
                          for v in r.values())
                   or "extent_role" in r and r["extent_role"] not in EXTENT_ROLES
                   for r in value)):
        raise input_error("omni_sampling", "targets",
                          f"{count} target/surface records in point or a/b order; optional "
                          "extent_role (top/base/left/right/front/back/rim_a/rim_b) and extent_basis; "
                          "all names are nonempty strings of at most 80 characters")
    return [{**r, "surface": surface_name(r["surface"])} for r in value]
