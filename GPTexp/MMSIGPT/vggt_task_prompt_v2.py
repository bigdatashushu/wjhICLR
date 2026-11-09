"""Compact pose context with question-defined frames and conditional navigation."""
from __future__ import annotations

import re

from .vggt_pose import POSE_VERSION, relative_cameras, same_evidence
from .vggt_task_prompt import task_binding as original_binding

PROMPT_VERSION = "mmsi-vggt-task-frame/2"
NAVIGATION = re.compile(
    r"\bwhat (?:should|must) (?:i|you|we) do\b"
    r"|\bhow (?:should|can|do|to)\b.{0,80}\b(?:reach|get to|go to|move|walk)\b"
    r"|\b(?:which|what)\s+(?:action|turn|movement|route|path)\b",
    re.I,
)


def task_binding(problem, image_count):
    binding = original_binding(problem, image_count)
    question = problem["question"]
    if binding["task_kind"] == "camera_region":
        match = re.search(
            r"(?:in which direction|where)\s+is\s+(.+?)\s+(?:located\s+)?relative to\b",
            question, re.I,
        )
        if match:
            binding["target_entity"] = " ".join(match[1].split())
    binding.update(schema=PROMPT_VERSION, navigation_requested=bool(NAVIGATION.search(question)))
    return binding


def relevant_rows(evidence, binding):
    """Use the requested camera frame, or camera 1 as a common evidence frame."""
    reference = binding["reference_image"] or 1
    rows = [row for row in evidence["relative_cameras"] if row["reference_image"] == reference]
    if binding["task_kind"] == "camera_camera" and binding["target_camera_image"] is not None:
        rows = [row for row in rows if row["camera_image"] == binding["target_camera_image"]]
    # Sparse evidence may not contain the requested pair; keep available rows explicit.
    return rows or evidence["relative_cameras"]


def _resolved(value):
    return value is not None and not value.startswith(("Bind ", "Identify "))


def pose_note(problem, evidence):
    if evidence["schema"] != POSE_VERSION or not same_evidence(
            evidence, relative_cameras(evidence["world_to_camera_extrinsics"])):
        raise ValueError("relative_pose_conversion_changed")
    binding = task_binding(problem, evidence["input_image_count"])
    camera_target = binding["task_kind"] == "camera_camera"
    show_forward = binding["navigation_requested"] and not camera_target
    lines = [
        "Estimated camera poses for the supplied RGB images.",
        "Reference-camera axes: x=right, y=down, z=forward. Center coordinates use "
        "one shared arbitrary normalized scale, not meters.",
        "center = photographer position.",
    ]
    if not camera_target:
        lines.append("rotation maps camera vectors into the named reference camera's axes.")
    if show_forward:
        lines.append("forward = viewing direction.")
    lines.append("Interpret directions using the question's reference origin and facing.")
    if not camera_target:
        lines.append(
            "Locate objects/regions from RGB; camera pose fields do not give their "
            "positions. Camera axes do not automatically define body or room axes."
        )
    frame = []
    if binding["reference_image"] is not None:
        frame.extend((f"origin=camera{binding['reference_image']}",
                      f"facing=camera{binding['reference_image']} at the stated moment"))
    else:
        if _resolved(binding["reference_origin"]):
            frame.append("origin=" + binding["reference_origin"])
        if binding["semantic_axis_anchor"]:
            frame.append("axis anchor=" + binding["semantic_axis_anchor"])
        elif _resolved(binding["reference_forward"]):
            frame.append("facing=" + binding["reference_forward"])
    if binding["target_camera_image"] is not None:
        frame.append(f"target=camera{binding['target_camera_image']} position")
    elif _resolved(binding["target_entity"]):
        frame.append("target=" + binding["target_entity"])
    if frame:
        lines.append("Question frame: " + "; ".join(frame) + ".")
    if binding["navigation_requested"]:
        lines.append(
            "Compare the target bearing with the question's initial facing "
            "to select the requested action."
        )

    def vector(values):
        return "[" + ",".join(f"{value:.4f}" for value in values) + "]"

    for row in relevant_rows(evidence, binding):
        camera, reference = row["camera_image"], row["reference_image"]
        center = ("unavailable (degenerate predicted baseline)" if row["camera_center_xyz"] is None
                  else vector(row["camera_center_xyz"]))
        lines.append(f"camera{camera}_center_in_camera{reference}_axes = {center}")
        if not camera_target:
            rotation = "[" + ",".join(vector(r) for r in row["camera_axes_to_reference_rotation"]) + "]"
            lines.append(f"camera{camera}_vectors_to_camera{reference}_axes_rotation_rows = {rotation}")
        if show_forward:
            lines.append(
                f"camera{camera}_forward_in_camera{reference}_axes = "
                + vector(row["camera_forward_in_reference"])
            )
    return "\n".join(lines)
