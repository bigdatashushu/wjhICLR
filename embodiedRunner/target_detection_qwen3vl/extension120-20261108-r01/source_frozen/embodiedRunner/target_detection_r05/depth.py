"""Synchronized depth evidence; display pixels never replace raw meter values."""
import base64
import io
import struct

from .sources import ContractError, digest

SHAPE = (500, 500)
DISPLAY_MAX_METERS = 10.0
SOURCES = {
    "navigation": ("head_depth", "AI2-THOR same last_event.depth_frame", "meters", "identity"),
    "alfred": ("head_depth", "AI2-THOR 2.1 same last_event.depth_frame", "millimeters", "divide_by_1000"),
    "habitat": ("head_depth", "Habitat same observation head_depth; RGB-aligned sensor",
                "meters", "identity"),
    "manipulation": ("front_depth", "CoppeliaSim same observation front_depth",
                     "normalized_0_1", "near_plus_depth_times_far_minus_near"),
}


def to_meters(value, environment, near=None, far=None):
    """Convert native units only; retain the camera's original pixel grid."""
    import numpy as np
    if environment not in SOURCES or value is None:
        raise ContractError("simulated_depth_missing_or_unknown_source")
    raw = np.asarray(value, dtype="<f4")
    if environment == "habitat" and raw.shape == SHAPE + (1,):
        raw = raw[..., 0]
    if raw.shape != SHAPE:
        raise ContractError("rgb_depth_shape_mismatch")
    if environment == "alfred":
        raw = raw / np.float32(1000)
    elif environment == "manipulation":
        if (type(near) not in (int, float) or type(far) not in (int, float)
                or not np.isfinite(near) or not np.isfinite(far) or not 0 < near < far):
            raise ContractError("invalid_depth_clip_planes")
        finite = raw[np.isfinite(raw)]
        if finite.size and (finite.min() < 0 or finite.max() > 1):
            raise ContractError("manipulation_depth_not_normalized")
        raw = np.float32(near) + raw * np.float32(far - near)
    return np.ascontiguousarray(raw, dtype="<f4")


def provenance(environment, near=None, far=None):
    view, source, unit, conversion = SOURCES[environment]
    return {"environment": environment, "native_source": source, "native_unit": unit,
            "conversion": conversion, "near_meters": near, "far_meters": far,
            "alignment": "same_camera_same_observation_no_resampling"}


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


def make_depth(value, rgb_sha256, step, environment, near=None, far=None):
    import numpy as np
    from PIL import Image
    raw = to_meters(value, environment, near, far)
    pixels, valid = grayscale(raw)
    stats = statistics(raw, valid)
    stream = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(stream, format="PNG")
    png, data = stream.getvalue(), raw.tobytes(order="C")
    return {
        "image": {"view": SOURCES[environment][0], "source": SOURCES[environment][1],
                  "base64": base64.b64encode(png).decode(), "sha256": digest(png)},
        "raw": {"encoding": "float32-little-endian", "shape": list(SHAPE), "unit": "meters",
                "base64": base64.b64encode(data).decode(), "sha256": digest(data)},
        "visualization": {"mapping": "inverse_linear_grayscale", "min_meters": 0.0,
                          "max_meters": DISPLAY_MAX_METERS, "invalid_gray": 0},
        "synchronization": {"environment_step": step, "rgb_sha256": rgb_sha256, "same_event": True},
        "stats": stats, "provenance": provenance(environment, near, far)}


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


def validate_depth(value, rgb_sha256, step, environment):
    import numpy as np
    from PIL import Image
    try:
        if set(value) != {"image", "raw", "visualization", "synchronization", "stats", "provenance"}:
            raise ContractError("invalid_depth_fields")
        if environment not in SOURCES:
            raise ContractError("invalid_depth_environment")
        native = value["provenance"]
        near, far = native["near_meters"], native["far_meters"]
        if environment == "manipulation":
            if (type(near) not in (int, float) or type(far) not in (int, float)
                    or not np.isfinite(near) or not np.isfinite(far) or not 0 < near < far):
                raise ContractError("invalid_depth_clip_planes")
        elif near is not None or far is not None:
            raise ContractError("unexpected_depth_clip_planes")
        if native != provenance(environment, near, far):
            raise ContractError("invalid_depth_provenance")
        if value["synchronization"] != {
                "environment_step": step, "rgb_sha256": rgb_sha256, "same_event": True}:
            raise ContractError("rgb_depth_synchronization_mismatch")
        if value["visualization"] != {
                "mapping": "inverse_linear_grayscale", "min_meters": 0.0,
                "max_meters": DISPLAY_MAX_METERS, "invalid_gray": 0}:
            raise ContractError("depth_visualization_changed")
        image = value["image"]
        if (set(image) != {"view", "source", "base64", "sha256"}
                or image["view"] != SOURCES[environment][0]
                or image["source"] != SOURCES[environment][1]):
            raise ContractError("invalid_depth_image_metadata")
        png = base64.b64decode(image["base64"], validate=True)
        if (digest(png) != image["sha256"] or png[:8] != b"\x89PNG\r\n\x1a\n"
                or struct.unpack(">II", png[16:24]) != (500, 500)):
            raise ContractError("invalid_depth_image_payload")
        raw = decode_raw(value)
        if environment == "manipulation":
            finite = raw[np.isfinite(raw)]
            tolerance = max(1e-6, far * 1e-6)
            if finite.size and (finite.min() < near - tolerance or finite.max() > far + tolerance):
                raise ContractError("manipulation_meter_depth_outside_clip_planes")
        pixels, valid = grayscale(raw)
        if value["stats"] != statistics(raw, valid):
            raise ContractError("invalid_depth_statistics")
        with Image.open(io.BytesIO(png)) as display:
            if display.mode != "L" or not np.array_equal(np.asarray(display), pixels):
                raise ContractError("depth_display_not_from_raw_values")
        return value
    except (KeyError, TypeError, ValueError, OSError, struct.error):
        raise ContractError("invalid_synchronized_depth") from None
