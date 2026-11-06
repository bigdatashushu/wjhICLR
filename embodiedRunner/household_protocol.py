"""Household wire contract. Scoring fields never enter model observations."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import io

from protocol import ROOT, REVISION, ContractError, canonical, json_object, sha256

VERSION = "embodiedrunner-household/1"
SUBSETS = {
    "alfred": ("base", "common_sense", "complex_instruction", "spatial",
               "visual_appearance", "long_horizon"),
    "habitat": ("base", "common_sense", "complex_instruction", "spatial_relationship",
                "visual_appearance", "long_horizon"),
}


def source(name):
    manifest = json_object((ROOT / "household_sources.json").read_text())
    if manifest["revision"] != REVISION:
        raise ContractError("upstream_revision_mismatch")
    row = next((x for x in manifest["files"] if x["path"] == name), None)
    data = (ROOT / "upstream-household" / name).read_bytes()
    if row is None or sha256(data) != row["sha256"]:
        raise ContractError("official_source_changed")
    return data


def verify_environment(environment):
    prefix = "embodiedbench/envs/eb_" + ("alfred" if environment == "alfred" else "habitat") + "/"
    manifest = json_object((ROOT / "household_sources.json").read_text())
    names = [x["path"] for x in manifest["files"] if x["path"].startswith(prefix)]
    for name in names:
        source(name)
    return len(names)


@dataclass(frozen=True)
class Observation:
    png: bytes
    image_sha256: str
    step: int
    instruction: str
    actions: tuple
    input_sha256: str


def decode(packet):
    required = {"image_base64", "image_sha256", "step", "instruction", "actions",
                "input_sha256", "done", "task_success", "last_action_success",
                "env_feedback", "task_progress", "invalid_actions", "reward"}
    if set(packet) != required:
        raise ContractError("unexpected_household_observation_fields")
    if (type(packet["step"]) is not int or not 0 <= packet["step"] <= 30
            or type(packet["invalid_actions"]) is not int or not 0 <= packet["invalid_actions"] <= 10
            or any(type(packet[k]) is not bool for k in
                   ("done", "task_success", "last_action_success"))
            or not isinstance(packet["actions"], list) or not packet["actions"]
            or not all(isinstance(x, str) for x in packet["actions"])
            or not isinstance(packet["instruction"], str) or not packet["instruction"]):
        raise ContractError("invalid_household_transition")
    png = base64.b64decode(packet["image_base64"], validate=True)
    from PIL import Image
    with Image.open(io.BytesIO(png)) as image:
        if image.format != "PNG" or image.size != (500, 500):
            raise ContractError("unexpected_rgb_shape")
        image.verify()
    if sha256(png) != packet["image_sha256"]:
        raise ContractError("rgb_hash_mismatch")
    return Observation(png, packet["image_sha256"], packet["step"],
                       packet["instruction"], tuple(packet["actions"]), packet["input_sha256"])
