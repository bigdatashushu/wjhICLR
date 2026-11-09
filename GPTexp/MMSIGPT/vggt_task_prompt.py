"""Question-bound camera evidence; routing never uses sample IDs or answers."""
from __future__ import annotations

import re

from .vggt_pose import POSE_VERSION, relative_cameras, same_evidence

PROMPT_VERSION = "mmsi-vggt-task-frame/1"
TASKS = {
    "Positional Relationship (Cam.–Cam.)": "camera_camera",
    "Positional Relationship (Cam.–Obj.)": "camera_object",
    "Positional Relationship (Cam.–Reg.)": "camera_region",
    "Positional Relationship (Obj.–Obj.)": "object_object",
    "Positional Relationship (Obj.–Reg.)": "object_region",
    "Positional Relationship (Reg.–Reg.)": "region_region",
}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
VIEW = (
    r"(?:the\s+)?(?:(first|second|third|fourth|fifth|sixth|last)\s+"
    r"(?:picture|image|frame)|(?:picture|image|frame|figure)\s*([1-6]))"
)
GUIDANCE = {
    "camera_camera": (
        "When the target is a camera position, use that camera's center in the "
        "reference camera's axes directly. Its viewing direction answers a different "
        "question. Keep the requested camera pair and direction of the transform explicit."
    ),
    "camera_object": (
        "Identify the queried physical object in RGB and match the same instance across "
        "views. An object absent from the reference image may still be located using "
        "shared landmarks. Use poses for view alignment, then express the object's "
        "direction in the question's reference camera frame. Another camera's center "
        "is not the object position."
    ),
    "camera_region": (
        "Locate the target region using shared RGB landmarks. The center of a camera "
        "that photographs the region is the photographer's position; its forward "
        "vector describes where the photographer looks. Neither vector alone gives "
        "the target region's bearing. Compare the target bearing with the question's "
        "initial facing, then map the turn and movement to the original choices."
    ),
    "object_object": (
        "Bind the reference object or seated person, the target object, and the "
        "question-defined facing in the physical scene. A phrase such as upper-left "
        "in a specified image identifies a facing cue; transfer that cue into the "
        "scene. Use poses to align views, then answer in the object's or person's "
        "body frame. Image-left and camera-forward do not define that body frame."
    ),
    "object_region": (
        "Bind the reference origin and the target from the question, including whether "
        "either is a region. Use the question's semantic front/back anchor to establish "
        "the scene axes before classifying the target. Match physical landmarks across "
        "RGB views and distinguish mirrors or reflections from physical entities. "
        "Camera positions serve view alignment and do not replace the queried entities."
    ),
    "region_region": (
        "Bind the origin region or landmark and the target region or landmark. Match "
        "the same physical instances across RGB views, keeping similarly named surfaces "
        "separate. Establish east/north or front/right from the question's semantic "
        "anchor and scene orientation. Image-right is not automatically east and "
        "image-up is not automatically north. Express the target direction from the "
        "question's origin after aligning observations."
    ),
}


def _text(question, pattern, group=1):
    match = re.search(pattern, question, re.I)
    return " ".join(match[group].split()) if match else None


def _view(question, prefixes, count):
    """Only identify views in explicit reference/target clauses; no first-view default."""
    views = set()
    for prefix in prefixes:
        for match in re.finditer(prefix + VIEW, question, re.I):
            ordinal, number = match.groups()
            ordinal = ordinal.lower() if ordinal else None
            value = count if ordinal == "last" else (
                ORDINALS.get(ordinal) if ordinal else int(number))
            if value is None or not 1 <= value <= count:
                return None
            views.add(value)
    return next(iter(views)) if len(views) == 1 else None


def task_binding(problem, image_count):
    kind = TASKS.get(problem["task_type"])
    if kind is None:
        raise ValueError("unsupported_mmsi_pose_task")
    if type(image_count) is not int or not 2 <= image_count <= 6:
        raise ValueError("requires_two_to_six_images")
    question = problem["question"]
    reference = None
    if kind.startswith("camera_"):
        reference = _view(question, (
            r"(?:i am|you are|we are)\s+taking\s+",
            r"from\s+(?:the\s+)?position\s+in\s+",
            r"at\s+(?:the\s+)?(?:moment|time)\s+of\s+",
            r"from\s+(?:the\s+)?perspective\s+of\s+",
        ), image_count)
    target_camera = _view(question, (
        r"camera\s+(?:.{0,80}?\s+)?(?:when\s+)?taking\s+",
    ), image_count) if kind == "camera_camera" else None
    origin = (f"Camera taking image {reference}" if reference else
              "Bind the reference origin explicitly from the question")
    forward = (f"Image {reference} camera's facing at the question's stated moment"
               if reference else "Bind the facing or axis definition from the question")
    target = "Identify the queried target entity from RGB and the question"
    if kind == "camera_camera":
        target = (f"Position of camera taking image {target_camera}" if target_camera else
                  "Bind the queried camera position from the question")
    elif kind == "camera_object":
        target = _text(question, r"in which direction is (.+?) relative to (?:you|me)\b") or target
    elif kind == "camera_region":
        target = _text(question, r"(?:reach|go to|walk to|get to)\s+(.+?)(?:\s+(?:shown|seen|observed)\s+in|[.?])") or target
    elif kind == "object_object":
        seat = _text(question, r"(?:sit|sitting|seated)\s+(?:on|in)\s+(.+?)\s+facing\b")
        origin = f"Person seated on {seat}" if seat else origin
        forward = _text(question, r"facing\s+(.+?)(?:,\s*(?:where|what|in which)|[.?])") or forward
        target = _text(question, r"where is (.+?) (?:located )?relative to\b") or target
    elif kind == "object_region":
        origin = _text(question, r"relative to\s+(.+?)[.?\n]") or origin
        target = _text(question, r"in which direction is (.+?) relative to\b") or target
    else:
        origin = _text(question, r"in which direction from (.+?) is\b") or origin
        target = _text(question, r"in which direction from .+? is (.+?) located\b") or target
    anchor = _text(
        question, r"([^.!?\n]+\bis\s+(?:on\s+)?(?:the\s+)?(?:east side|in front)\s+of\s+[^.!?\n]+)"
    )
    if anchor and kind in ("object_region", "region_region"):
        forward = "Scene axes defined by this question statement: " + anchor
    return {
        "schema": PROMPT_VERSION, "task_kind": kind, "reference_image": reference,
        "target_camera_image": target_camera, "reference_origin": origin,
        "reference_forward": forward, "target_entity": target,
        "semantic_axis_anchor": anchor, "binding_source": "public_task_type_and_question_only",
    }


def pose_note(problem, evidence):
    if evidence["schema"] != POSE_VERSION or not same_evidence(
            evidence, relative_cameras(evidence["world_to_camera_extrinsics"])):
        raise ValueError("relative_pose_conversion_changed")
    binding = task_binding(problem, evidence["input_image_count"])
    lines = [
        "Task-frame binding for the supplied original RGB images.",
        "Reference origin: " + binding["reference_origin"] + ".",
        "Reference facing/axes: " + binding["reference_forward"] + ".",
        "Target entity: " + binding["target_entity"] + ".",
        GUIDANCE[binding["task_kind"]],
        "Confirm these bindings from the full question. If a clause was not resolved, "
        "establish it from the question and RGB rather than assuming a camera or room axis.",
        "Joint VGGT camera predictions use x=right, y=down, z=forward for each reference "
        "camera. Camera centers describe photographer positions; camera forward vectors "
        "describe viewing directions. A rotation maps vectors from the named camera "
        "into the reference camera's axes.",
        "Centers share one normalized arbitrary scale, not meters. Pose accuracy has "
        "not been independently verified. Poses do not supply object/region locations "
        "or room north/front axes. Use shared physical RGB landmarks to check alignment; "
        "if poses disagree with visible landmarks, do not force the target bearing "
        "to follow the pose. Follow the original answer-format instruction.",
    ]
    rows = sorted(evidence["relative_cameras"], key=lambda row: (
        row["reference_image"] != binding["reference_image"],
        row["camera_image"] != binding["target_camera_image"],
        row["reference_image"], row["camera_image"]))
    vector = lambda values: "[" + ",".join(f"{v:.4f}" for v in values) + "]"
    for row in rows:
        camera, reference = row["camera_image"], row["reference_image"]
        center = ("unavailable (degenerate predicted baseline)" if row["camera_center_xyz"] is None
                  else vector(row["camera_center_xyz"]))
        rotation = "[" + ",".join(vector(r) for r in row["camera_axes_to_reference_rotation"]) + "]"
        lines.extend((
            f"camera{camera}_center_in_camera{reference}_axes (photographer position) = {center}",
            f"camera{camera}_forward_in_camera{reference}_axes (viewing direction) = "
            + vector(row["camera_forward_in_reference"]),
            f"camera{camera}_vectors_to_camera{reference}_axes_rotation_rows = {rotation}",
        ))
    return "\n".join(lines)
