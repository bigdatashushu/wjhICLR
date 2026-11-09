"""Deterministic depth display; numerical tool evidence remains unmodified."""
import hashlib
from io import BytesIO
import json

import numpy as np
from PIL import Image

VERSION = "cv3d-original-rgb-and-depth/1"


def render_depth_image(depth, valid_mask) -> tuple[bytes, dict]:
    depth = np.asarray(depth)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    if depth.ndim != 2 or valid.shape != depth.shape or not np.any(valid):
        raise ValueError("no valid predicted depth")
    lo, hi = map(float, np.quantile(depth[valid], [0.02, 0.98]))
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if hi > lo:
        gray[valid] = np.rint(
            np.clip((depth[valid] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    else:
        gray[valid] = 128
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    rgb[~valid] = [255, 0, 255]
    output = BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    png = output.getvalue()
    raw_identity = {
        "dtype": depth.dtype.str, "shape": list(depth.shape),
        "bytes_sha256": hashlib.sha256(depth.tobytes(order="C")).hexdigest(),
    }
    raw_hash = hashlib.sha256(json.dumps(
        raw_identity, ensure_ascii=False, sort_keys=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()
    return png, {
        "display": "linear_grayscale", "lower_quantile": 0.02, "upper_quantile": 0.98,
        "lower_depth": lo, "upper_depth": hi, "invalid_color": [255, 0, 255],
        "width": depth.shape[1], "height": depth.shape[0],
        "png_sha256": hashlib.sha256(png).hexdigest(), "raw_depth_sha256": raw_hash,
    }


def depth_image_description(width, height):
    return (
        f"Image 2: PREDICTED DEPTH MAP, {width} x {height}, aligned to Image 1 "
        "(same view_index=0, not a second camera view). Darker grayscale means nearer "
        "camera-depth; lighter means farther. Display clips the 2nd-98th percentiles; "
        "magenta is invalid. Tools read original numerical depth/3D points, not grayscale."
    )
