"""Describe photo shooting locations without exposing numeric camera poses."""
from __future__ import annotations

import math

from .vggt_pose import POSE_VERSION, relative_cameras, same_evidence

RELATION_VERSION = "mmsi-vggt-shooting-location-relation/1"
AXIS_TOLERANCE = 1e-6


def describe_position(center):
    """Classify the x/z displacement; do not invent a physical ground normal."""
    if center is None:
        return {"available": False, "position": None, "dominant": None,
                "description": "cannot be reliably determined from the estimated baseline"}
    x, _, z = center
    if not all(math.isfinite(value) for value in center):
        raise ValueError("nonfinite_camera_location")
    horizontal = "right" if x > AXIS_TOLERANCE else "left" if x < -AXIS_TOLERANCE else None
    longitudinal = "front" if z > AXIS_TOLERANCE else "behind" if z < -AXIS_TOLERANCE else None
    side = f"to your {horizontal}" if horizontal else None
    ahead = ("in front of you" if longitudinal == "front" else
             "behind you" if longitudinal == "behind" else None)
    if side and ahead:
        if abs(abs(x) - abs(z)) <= AXIS_TOLERANCE:
            dominance = "balanced"
            description = f"{ahead} and {side}, with similar sideways and forward/backward displacement"
        else:
            dominance = "lateral" if abs(x) > abs(z) else "longitudinal"
            detail = (f"more {horizontal}ward than {'forward' if z > 0 else 'backward'}"
                      if dominance == "lateral" else
                      f"more {'forward' if z > 0 else 'backward'} than {horizontal}ward")
            description = f"{ahead} and {side}, with displacement {detail}"
    elif side or ahead:
        dominance = "lateral" if side else "longitudinal"
        description = side or ahead
    else:
        return {"available": False, "position": None, "dominant": None,
                "description": "cannot be reliably distinguished in left/right or front/behind"}
    return {"available": True,
            "position": "_".join(value for value in (longitudinal, horizontal) if value),
            "dominant": dominance, "description": description}


def location_relations(evidence):
    if evidence["schema"] != POSE_VERSION or evidence["input_image_count"] != 2:
        raise ValueError("requires_joint_two_camera_evidence")
    expected = relative_cameras(evidence["world_to_camera_extrinsics"])
    if not same_evidence(evidence, expected):
        raise ValueError("saved_relative_camera_evidence_changed")
    rows = sorted(evidence["relative_cameras"], key=lambda row: row["reference_image"])
    if [(row["reference_image"], row["camera_image"]) for row in rows] != [(1, 2), (2, 1)]:
        raise ValueError("bidirectional_camera_binding_changed")
    return [{
        "reference_photo": row["reference_image"],
        "target_photo": row["camera_image"],
        "reference_origin": "shooting_location_of_reference_photo",
        "reference_facing": "camera_viewing_direction_of_reference_photo",
        "rule": "x_z_components_in_reference_camera_axes_y_not_used",
        **describe_position(row["camera_center_xyz"]),
    } for row in rows]


def relation_note(evidence):
    lines = [
        "VGGT-estimated relationship between the shooting locations of the two photos, "
        "numbered in their supplied order.",
    ]
    for row in location_relations(evidence):
        ref, target = row["reference_photo"], row["target_photo"]
        anchor = (
            f"Reference Photo {ref}: stand where Photo {ref} was taken and face in the "
            f"same direction that its camera was looking. "
        )
        location = (f"The shooting location of Photo {target} is {row['description']}."
                    if row["available"] else
                    f"The position of Photo {target}'s shooting location {row['description']}.")
        lines.append(anchor + location)
    lines.append(
        "Each statement uses its own reference photo's facing. These are photographer "
        "locations, not the locations or facing directions of objects or regions in the photos."
    )
    return "\n".join(lines)
