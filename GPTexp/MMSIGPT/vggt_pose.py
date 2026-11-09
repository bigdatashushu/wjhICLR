"""Convert joint VGGT world-to-camera predictions into camera-relative evidence."""
from __future__ import annotations

import numpy as np

POSE_VERSION = "joint-vggt-opencv-camera-relative/1"
COORDINATES = "reference camera axes: x right, y down, z forward"


def same_evidence(left, right):
    """Allow numerical backend roundoff while retaining exact schema and provenance."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_evidence(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same_evidence(a, b) for a, b in zip(left, right))
    if type(left) in (int, float) and type(right) in (int, float):
        return bool(np.isclose(left, right, atol=1e-12, rtol=1e-12))
    return type(left) is type(right) and left == right


def relative_cameras(extrinsics):
    """Keep camera-center translation separate from the extrinsic translation."""
    matrices = np.asarray(extrinsics, dtype=np.float64)
    if (matrices.ndim != 3 or matrices.shape[1:] != (3, 4)
            or not 2 <= len(matrices) <= 6 or not np.isfinite(matrices).all()):
        raise ValueError("requires_two_to_six_finite_world_to_camera_matrices")
    rotations, translations = matrices[:, :, :3], matrices[:, :, 3]
    for rotation in rotations:
        if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-4, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1, atol=2e-4, rtol=0)):
            raise ValueError("invalid_vggt_rotation")
    centers = -np.einsum("nji,nj->ni", rotations, translations)
    baseline = float(max(np.linalg.norm(center - centers[0]) for center in centers[1:]))
    translation_valid = baseline > 1e-8
    rows = []
    # Both directions relative to image 1 cover camera-location questions
    # without asking the language model to invert an extrinsic matrix.
    for view in range(1, len(matrices)):
        for reference, camera in ((0, view), (view, 0)):
            rotation = rotations[reference] @ rotations[camera].T
            center = rotations[reference] @ (centers[camera] - centers[reference])
            if translation_valid:
                center = center / baseline
            rows.append({
                "reference_image": reference + 1, "camera_image": camera + 1,
                "camera_center_xyz": center.tolist() if translation_valid else None,
                "camera_axes_to_reference_rotation": rotation.tolist(),
                "camera_forward_in_reference": rotation[:, 2].tolist(),
            })
    return {
        "schema": POSE_VERSION, "coordinate_convention": COORDINATES,
        "input_image_count": len(matrices), "world_to_camera_extrinsics": matrices.tolist(),
        "raw_translation_normalizer": baseline,
        "translation_status": "normalized_prediction" if translation_valid else "degenerate_prediction",
        "translation_unit": "one shared arbitrary normalized unit; not meters",
        "translation_normalization": "maximum camera-center distance from camera 1 equals one",
        "world_north_or_room_axes_provided": False, "metric_scale_verified": False,
        "pose_accuracy_verified": False, "relative_cameras": rows,
    }


def pose_note(evidence):
    if evidence["schema"] != POSE_VERSION:
        raise ValueError("pose_schema_changed")
    expected = relative_cameras(evidence["world_to_camera_extrinsics"])
    if not same_evidence(evidence, expected):
        raise ValueError("relative_pose_conversion_changed")
    def vector(values):
        return "[" + ",".join(f"{value:.4f}" for value in values) + "]"
    lines = [
        "Joint VGGT camera estimates for the original RGB images in their supplied order.",
        "Image numbers identify cameras. Each reference camera uses x=right, y=down, "
        "z=forward. Centers share one normalized arbitrary scale, not meters. "
        "These predictions do not define room front, north, or object-facing directions.",
        "For each row, center is the other camera location in the reference camera's axes; "
        "rotation maps the other camera's vectors into those axes. "
        "Use RGBs and the question to interpret the requested reference frame.",
    ]
    for row in evidence["relative_cameras"]:
        center = ("unavailable (degenerate predicted baseline)" if row["camera_center_xyz"] is None
                  else vector(row["camera_center_xyz"]))
        rotation = "[" + ",".join(vector(r) for r in row["camera_axes_to_reference_rotation"]) + "]"
        lines.append(
            f"Reference image{row['reference_image']}, camera image{row['camera_image']}: "
            f"center_xyz={center}; rotation_rows={rotation}; "
            f"forward_xyz={vector(row['camera_forward_in_reference'])}."
        )
    return "\n".join(lines)
