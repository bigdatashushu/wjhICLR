"""Explicit public input boundary and native action spaces."""
import base64
import math
import re
import struct
from dataclasses import dataclass

from .sources import ContractError, LIMITS, canonical, digest

COORD_SOURCE = "official eb_man_utils.form_object_coord_for_input; anonymous object names"
PUBLIC_KEYS = {"instruction", "images", "actions", "step", "env_feedback",
               "last_action_success", "manipulation", "depth"}
RESULT_KEYS = {"done", "task_success", "task_progress", "invalid_actions", "reward",
               "distance", "episode_elapsed_seconds", "termination_source"}


@dataclass(frozen=True)
class Task:
    environment: str
    subset: str
    index: int

    def __post_init__(self):
        if (self.environment not in LIMITS or not re.fullmatch(r"[a-z_]+", self.subset)
                or type(self.index) is not int or self.index < 0):
            raise ContractError("invalid_task")

    @property
    def identity(self):
        return f"{self.environment}/{self.subset}:{self.index:03d}"

    @property
    def directory(self):
        return f"{self.environment}-{self.subset}-{self.index:03d}"

    def as_dict(self):
        return {"environment": self.environment, "subset": self.subset, "index": self.index}


def vector(value):
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ContractError("invalid_manipulation_action_shape")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ContractError("invalid_manipulation_action_number")
    if any(int(x) != x or not 0 <= x <= 100 for x in value[:3]):
        raise ContractError("invalid_manipulation_position")
    if any(int(x) != x or not 0 <= x <= 120 for x in value[3:6]):
        raise ContractError("invalid_manipulation_rotation")
    if value[6] not in (0, 1):
        raise ContractError("invalid_manipulation_gripper")
    return list(value)


def _validate_official_packet(packet, task):
    """Reject private/unknown channels instead of silently forwarding them."""
    if set(packet) != {"task", "public", "official", "input_sha256"}:
        raise ContractError("invalid_packet_fields")
    if packet["task"] != task.as_dict():
        raise ContractError("episode_identity_mismatch")
    public, result = packet["public"], packet["official"]
    if set(public) != PUBLIC_KEYS or not set(result) <= RESULT_KEYS:
        raise ContractError("private_or_unknown_packet_field")
    if not isinstance(public["instruction"], str) or not isinstance(public["env_feedback"], str):
        raise ContractError("invalid_public_text")
    if type(public["step"]) is not int or not 0 <= public["step"] <= LIMITS[task.environment]:
        raise ContractError("invalid_official_step")
    if type(public["last_action_success"]) is not bool or type(result["done"]) is not bool:
        raise ContractError("invalid_official_flags")
    if (not isinstance(public["actions"], list)
            or any(not isinstance(x, str) for x in public["actions"])):
        raise ContractError("invalid_action_space")
    if task.environment == "navigation" and len(public["actions"]) != 8:
        raise ContractError("navigation_requires_eight_actions")
    if task.environment in ("alfred", "habitat") and not public["actions"]:
        raise ContractError("empty_action_space")
    if len(public["images"]) != 1:
        raise ContractError("direct_requires_one_current_view")
    for image in public["images"]:
        if set(image) != {"view", "base64", "sha256", "source"}:
            raise ContractError("invalid_image_fields")
        png = base64.b64decode(image["base64"], validate=True)
        if (digest(png) != image["sha256"] or png[:8] != b"\x89PNG\r\n\x1a\n"
                or struct.unpack(">II", png[16:24]) != (500, 500)):
            raise ContractError("invalid_public_image")
    manip = public["manipulation"]
    if task.environment == "manipulation":
        if public["actions"] or not isinstance(manip, dict) or set(manip) != {
                "objects", "task_variation", "source"} or manip["source"] != COORD_SOURCE:
            raise ContractError("invalid_official_manipulation_input")
        for name, point in manip["objects"].items():
            if not re.fullmatch(r"object \d+", name) or len(point) != 3 or any(
                    type(x) is not int for x in point):
                raise ContractError("non_anonymous_coordinate_input")
    elif manip is not None:
        raise ContractError("unexpected_manipulation_input")
    from .depth import validate_depth
    validate_depth(public["depth"], public["images"][0]["sha256"], public["step"], task.environment)
    for key, value in result.items():
        if key not in ("done", "termination_source") and (
                type(value) not in (int, float) or not math.isfinite(value)):
            raise ContractError("invalid_official_metric")
    if result["task_success"] not in (0, 1) or type(result["invalid_actions"]) is not int:
        raise ContractError("invalid_official_score")
    expected = digest(canonical({"task": packet["task"], "instruction": public["instruction"]}))
    if packet["input_sha256"] != expected:
        raise ContractError("input_identity_mismatch")
    return packet


def validate_packet(packet, task):
    """Validate the original boundary plus same-observation raw RGB and anonymous anchors."""
    if not isinstance(packet, dict) or not isinstance(packet.get("public"), dict):
        raise ContractError("invalid_packet")
    public = packet["public"]
    if set(public) != PUBLIC_KEYS | {"raw_rgb", "anonymous_anchors"}:
        raise ContractError("private_or_unknown_packet_field")
    original = dict(packet, public={k: v for k, v in public.items() if k in PUBLIC_KEYS})
    _validate_official_packet(original, task)
    raw = public["raw_rgb"]
    if set(raw) != {"view", "base64", "sha256", "source"}:
        raise ContractError("invalid_raw_rgb_fields")
    png = base64.b64decode(raw["base64"], validate=True)
    if (digest(png) != raw["sha256"] or png[:8] != b"\x89PNG\r\n\x1a\n"
            or struct.unpack(">II", png[16:24]) != (500, 500)
            or raw["view"] != public["images"][0]["view"]
            or raw["source"] != "official same-observation unannotated RGB"):
        raise ContractError("invalid_raw_rgb")
    if task.environment != "manipulation" and raw["sha256"] != public["images"][0]["sha256"]:
        raise ContractError("raw_rgb_observation_mismatch")
    anchors = public["anonymous_anchors"]
    if not isinstance(anchors, list):
        raise ContractError("invalid_anonymous_anchors")
    if task.environment != "manipulation" and anchors:
        raise ContractError("unexpected_anonymous_anchors")
    names = set()
    for row in anchors:
        if (set(row) != {"object", "pixel"} or row["object"] in names
                or row["object"] not in public["manipulation"]["objects"]
                or not isinstance(row["pixel"], list) or len(row["pixel"]) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < 500
                       for v in row["pixel"])):
            raise ContractError("invalid_anonymous_anchor")
        names.add(row["object"])
    return packet
