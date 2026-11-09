"""Synchronized depth evidence; display pixels never replace raw meter values."""
import base64
import io
import struct

from .sources import ContractError, digest

SHAPE = (500, 500)
DISPLAY_MAX_METERS = 10.0
IMAGE_SOURCE = "AI2-THOR same last_event.depth_frame; inverse linear grayscale 0-10 meters"


def grayscale(raw):
    import numpy as np
    valid = np.isfinite(raw) & (raw > 0)
    result = np.zeros(SHAPE, dtype=np.uint8)
    result[valid] = np.rint(255 * (1 - np.clip(raw[valid], 0, DISPLAY_MAX_METERS)
                                 / DISPLAY_MAX_METERS)).astype(np.uint8)
    return result, valid


def statistics(raw, valid):
    import numpy as np
    count = int(np.count_nonzero(valid))
    if not count:
        raise ContractError("no_valid_simulated_depth")
    return {"valid_pixels": count, "invalid_pixels": int(raw.size - count),
            "min_meters": float(raw[valid].min()), "max_meters": float(raw[valid].max())}


def make_depth(value, rgb_sha256, step):
    import numpy as np
    from PIL import Image
    if value is None:
        raise ContractError("simulated_depth_missing")
    raw = np.asarray(value, dtype="<f4")
    if raw.shape != SHAPE:
        raise ContractError("rgb_depth_shape_mismatch")
    raw = np.ascontiguousarray(raw)
    pixels, valid = grayscale(raw)
    stats = statistics(raw, valid)
    stream = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(stream, format="PNG")
    png, data = stream.getvalue(), raw.tobytes(order="C")
    return {
        "image": {"view": "head_depth", "source": IMAGE_SOURCE,
                  "base64": base64.b64encode(png).decode(), "sha256": digest(png)},
        "raw": {"encoding": "float32-little-endian", "shape": list(SHAPE), "unit": "meters",
                "base64": base64.b64encode(data).decode(), "sha256": digest(data)},
        "visualization": {"mapping": "inverse_linear_grayscale", "min_meters": 0.0,
                          "max_meters": DISPLAY_MAX_METERS, "invalid_gray": 0},
        "synchronization": {"environment_step": step, "rgb_sha256": rgb_sha256, "same_event": True},
        "stats": stats}


def decode_raw(value):
    import numpy as np
    try:
        raw = value["raw"]
        if (set(raw) != {"encoding", "shape", "unit", "base64", "sha256"}
                or raw["encoding"] != "float32-little-endian" or raw["shape"] != list(SHAPE)
                or raw["unit"] != "meters"):
            raise ContractError("invalid_raw_depth_metadata")
        data = base64.b64decode(raw["base64"], validate=True)
        if len(data) != SHAPE[0] * SHAPE[1] * 4 or digest(data) != raw["sha256"]:
            raise ContractError("invalid_raw_depth_payload")
        return np.frombuffer(data, dtype="<f4").reshape(SHAPE)
    except (KeyError, TypeError, ValueError):
        raise ContractError("invalid_raw_depth_payload") from None


def validate_depth(value, rgb_sha256, step):
    import numpy as np
    from PIL import Image
    try:
        if set(value) != {"image", "raw", "visualization", "synchronization", "stats"}:
            raise ContractError("invalid_depth_fields")
        if value["synchronization"] != {
                "environment_step": step, "rgb_sha256": rgb_sha256, "same_event": True}:
            raise ContractError("rgb_depth_synchronization_mismatch")
        if value["visualization"] != {
                "mapping": "inverse_linear_grayscale", "min_meters": 0.0,
                "max_meters": DISPLAY_MAX_METERS, "invalid_gray": 0}:
            raise ContractError("depth_visualization_changed")
        image = value["image"]
        if (set(image) != {"view", "source", "base64", "sha256"}
                or image["view"] != "head_depth" or image["source"] != IMAGE_SOURCE):
            raise ContractError("invalid_depth_image_metadata")
        png = base64.b64decode(image["base64"], validate=True)
        if (digest(png) != image["sha256"] or png[:8] != b"\x89PNG\r\n\x1a\n"
                or struct.unpack(">II", png[16:24]) != (500, 500)):
            raise ContractError("invalid_depth_image_payload")
        raw = decode_raw(value)
        pixels, valid = grayscale(raw)
        if value["stats"] != statistics(raw, valid):
            raise ContractError("invalid_depth_statistics")
        with Image.open(io.BytesIO(png)) as display:
            if display.mode != "L" or not np.array_equal(np.asarray(display), pixels):
                raise ContractError("depth_display_not_from_raw_values")
        return value
    except (KeyError, TypeError, ValueError, OSError, struct.error):
        raise ContractError("invalid_synchronized_depth") from None
