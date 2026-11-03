"""Render cached relative camera Z on the exact original image grid."""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from .data import digest_bytes

VERSION = "mindcube-depth-preview/1"
PALETTE = ((253, 231, 37), (94, 201, 98), (33, 145, 140),
           (59, 82, 139), (68, 1, 84))
QUANTILES = (0.02, 0.98)
DEPTH_LEGEND = (
    "Depth previews share their original RGB pixel grid: no resize, crop, rotation or border; "
    "origin top-left, x right and y down. "
    "Per-view 2%-98% camera-Z display ranges clip outliers to endpoint colors. "
    "Near to far: yellow -> green -> teal -> blue -> purple. "
    "Black means invalid depth; no_contrast uses constant teal with no depth contrast; "
    "an unavailable all-black map supplies no depth evidence. "
    "Each view is normalized independently; colors cannot be compared across views."
)


def depth_preview(geometry, *, view_index, width, height):
    """Return deterministic PNG bytes, alignment/scale text and a public receipt."""
    if not geometry or not geometry.get("relative_only"):
        raise ValueError("depth_preview_relative_geometry_required")
    depth = np.asarray(geometry["depth"], dtype=np.float64)
    mask = np.asarray(geometry["valid_mask"], dtype=bool)
    if depth.shape != (height, width) or mask.shape != depth.shape:
        raise ValueError("depth_preview_grid_changed")
    valid = mask & np.isfinite(depth) & (depth > 0)
    values = depth[valid]
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    metadata = {
        "schema": VERSION, "view_index": view_index, "width": width, "height": height,
        "units": "per_view_relative", "quantity": "camera_z",
        "valid_pixel_count": int(values.size), "total_pixel_count": width * height,
        "quantiles": list(QUANTILES), "display_min": None, "display_max": None,
        "valid_depth_min": None, "valid_depth_max": None,
        "palette_near_to_far": [list(color) for color in PALETTE],
        "invalid_color": [0, 0, 0],
    }
    caption = (
        f"Depth Image {view_index + 1}: view_index={view_index}, PNG {width}x{height}; "
        f"aligned with Original Image {view_index + 1}; "
        f"x=0..{width - 1}, y=0..{height - 1}; "
        f"valid={values.size}/{width * height}. "
    )
    if not values.size:
        metadata["status"] = "unavailable"
        caption += "status=unavailable; no valid finite positive depth."
    else:
        low, high = map(float, np.quantile(values, QUANTILES))
        metadata.update(display_min=low, display_max=high,
                        valid_depth_min=float(values.min()), valid_depth_max=float(values.max()))
        if high > low:
            normalized = np.clip((values - low) / (high - low), 0, 1)
            anchors = np.linspace(0, 1, len(PALETTE))
            colors = np.asarray(PALETTE)
            pixels[valid] = np.rint(np.column_stack([
                np.interp(normalized, anchors, colors[:, channel]) for channel in range(3)
            ])).astype(np.uint8)
            metadata["status"] = "available"
            caption += (
                f"status=available; display Z 2%-98%=[{low:.8g}, {high:.8g}]."
            )
        else:
            pixels[valid] = PALETTE[2]
            metadata["status"] = "no_contrast"
            caption += (
                f"status=no_contrast; display Z 2%-98%=[{low:.8g}, {high:.8g}]; "
                f"valid Z=[{values.min():.8g}, {values.max():.8g}]."
            )
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG", optimize=False, compress_level=9)
    raw = stream.getvalue()
    metadata["image_sha256"] = digest_bytes(raw)
    return raw, caption, metadata
