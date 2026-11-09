"""Read verified predicted depth only; render it on the unchanged original pixel grid."""
from __future__ import annotations

import base64
from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ...traces import digest, file_digest

RENDERER_VERSION = "relative-depth-p2-p98-gray-magenta/1"
FIELDS = {
    "problem", "frames", "candidates", "depths", "points", "intrinsics",
    "geometry_masks", "geometries", "categories", "provenance", "capabilities",
    "cache_key", "normalized_image_paths", "errors",
}


def read_depth(directory, problem, images, expected_artifact):
    """Verify full storage/content hashes without importing the Harness runtime.

    The tagged tree is hashed in the exact prepared-content/1 framing. Arrays are
    inspected one at a time. Only problem, depth, validity and provenance are
    decoded; candidate geometry and reference answers never become model input.
    """
    directory = Path(directory)
    metadata_path, arrays_path = directory / "metadata.json", directory / "arrays.npz"
    metadata = json.loads(metadata_path.read_bytes())
    if (set(metadata) != {"schema", "artifact_sha256", "input_sha256", "cache_key",
                         "fields", "arrays_sha256"}
            or metadata["schema"] != "small-image-prepared-storage/1"
            or metadata["input_sha256"] != problem["input_sha256"]
            or metadata["artifact_sha256"] != expected_artifact
            or file_digest(arrays_path) != metadata["arrays_sha256"]):
        raise ValueError("prepared_storage_identity_changed")
    referenced = set()
    with np.load(arrays_path, allow_pickle=False) as stored:
        def frame(node):
            if not isinstance(node, list) or len(node) != 2:
                raise ValueError("invalid_storage_node")
            kind, value = node
            if kind == "array":
                array = stored[value]
                if array.dtype.hasobject:
                    raise ValueError("object_array_forbidden")
                referenced.add(value)
                return ["ndarray", list(array.shape), np.lib.format.dtype_to_descr(array.dtype),
                        hashlib.sha256(array.tobytes(order="C")).hexdigest()]
            if kind == "mapping":
                rows = [[frame(key), frame(item)] for key, item in value]
                rows.sort(key=lambda row: json.dumps(
                    row[0], ensure_ascii=False, separators=(",", ":"), allow_nan=False))
                if len({json.dumps(row[0]) for row in rows}) != len(rows):
                    raise ValueError("duplicate_storage_key")
                return ["mapping", rows]
            if kind in ("tuple", "list"):
                return ["sequence", [frame(item) for item in value]]
            if kind == "bytes":
                raw = base64.b64decode(value, validate=True)
                return ["bytes", len(raw), hashlib.sha256(raw).hexdigest()]
            if kind == "path":
                return ["path", value]
            if kind == "float":
                return ["float", float.fromhex(value).hex()]
            if kind == "scalar":
                if value is None:
                    return ["null"]
                if type(value) is bool:
                    return ["bool", value]
                if type(value) is int:
                    return ["int", str(value)]
                if isinstance(value, str):
                    return ["str", value]
            raise ValueError("unsupported_storage_node")

        fields_frame = frame(metadata["fields"])
        root = ["mapping", [
            [["str", "fields"], fields_frame],
            [["str", "schema"], ["str", "small-image-prepared-content/1"]],
        ]]
        if digest(root) != expected_artifact or referenced != set(stored.files):
            raise ValueError("prepared_content_hash_changed")

        def decode(node):
            kind, value = node
            if kind == "mapping":
                return {decode(key): decode(item) for key, item in value}
            if kind in ("tuple", "list"):
                return [decode(item) for item in value]
            if kind == "scalar":
                return value
            if kind == "float":
                return float.fromhex(value)
            if kind == "path":
                return Path(value)
            if kind == "array":
                return stored[value]
            raise ValueError("unsupported_depth_field")

        tree = metadata["fields"]
        if tree[0] != "mapping":
            raise ValueError("invalid_prepared_fields")
        nodes = {decode(key): value for key, value in tree[1]}
        if set(nodes) != FIELDS:
            raise ValueError("prepared_field_schema_changed")
        public = decode(nodes["problem"])
        provenance = decode(nodes["provenance"])
        if (public != problem or decode(nodes["cache_key"]) != metadata["cache_key"]
                or provenance.get("labels_used") is not False
                or provenance.get("geometry_scope") != "per_view_relative_only"
                or provenance.get("metric_scale_verified") is not False
                or provenance.get("world_frame_verified") is not False):
            raise ValueError("depth_input_or_provenance_changed")
        depths, masks = decode(nodes["depths"]), decode(nodes["geometry_masks"])
        if set(depths) != set(range(len(images))) or set(masks) != set(depths):
            raise ValueError("requires_predicted_depth_for_every_original_view")
        rendered = [
            render(depths[i], masks[i], (image["width"], image["height"]))
            for i, image in enumerate(images)
        ]
    return rendered, {
        "input_sha256": problem["input_sha256"], "artifact_sha256": expected_artifact,
        "metadata_sha256": file_digest(metadata_path),
        "arrays_sha256": metadata["arrays_sha256"],
        "provenance_sha256": digest(provenance), "labels_used": False,
        "renderer_version": RENDERER_VERSION,
    }


def render(depth, mask, size):
    width, height = size
    if (depth.shape != (height, width) or mask.shape != depth.shape
            or depth.dtype.kind not in "fiu" or mask.dtype.kind not in "buif"
            or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all()):
        raise ValueError("depth_must_use_original_rgb_pixel_grid")
    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("no_valid_predicted_depth")
    low, high = np.percentile(depth[valid], [2, 98])
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = [255, 0, 255]
    gray = (np.rint(np.clip((depth[valid] - low) / (high - low), 0, 1) * 255)
            if high > low else np.full(valid.sum(), 128)).astype(np.uint8)
    image[valid] = gray[:, None]
    output = BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    raw = output.getvalue()
    return raw, {
        "width": width, "height": height, "mime_type": "image/png",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "valid_pixels": int(valid.sum()), "pixels": width * height,
        "p2": float(low), "p98": float(high), "relative_only": True,
    }
