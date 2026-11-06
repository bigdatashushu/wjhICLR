"""Public seven-dimensional action contract; no simulator imports or labels."""
from __future__ import annotations

import math
from pathlib import Path

from protocol import ContractError, canonical, json_object, sha256

ENVIRONMENT = "eb_manipulation"
SUBSETS = ("base", "common_sense", "complex", "spatial", "visual")
TASKS = {
    "base": ("pick_cube_shape", "stack_cubes_color", "place_into_shape_sorter_color", "wipe_table_direction"),
    "common_sense": ("pick_cube_shape", "stack_cubes_color", "place_into_shape_sorter_color", "wipe_table_direction"),
    "complex": ("pick_cube_shape", "stack_cubes_color", "place_into_shape_sorter_color", "wipe_table_direction"),
    "spatial": ("pick_cube_relative", "stack_cubes_relative", "place_into_shape_sorter_relative", "wipe_table_relative"),
    "visual": ("pick_cube_shape", "stack_cubes_color", "place_into_shape_sorter_color"),
}
ACTION_SPEC = {
    "schema": "embodied-manipulation-action/1",
    "fields": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    "position_index_range": [0, 100], "rotation_index_range": [0, 120],
    "rotation_resolution_degrees": 3, "gripper_states": [0, 1],
    "scene_bounds_m": [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6],
    "voxel_size": 100, "frame": "world",
}
INPUT_SOURCE = "official_depth_instance_masks_and_camera_parameters"
ROOT = Path(__file__).resolve().parent


def vector(value):
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ContractError("invalid_manipulation_action")
    for i, x in enumerate(value[:6]):
        if type(x) is not int or not 0 <= x <= (100 if i < 3 else 120):
            raise ContractError("invalid_manipulation_action")
    grip = value[6]
    if type(grip) not in (int, float) or not math.isfinite(grip) or grip not in (0, 1):
        raise ContractError("invalid_manipulation_gripper")
    return tuple(value[:6]) + (int(grip),)


def public_input(value, subset):
    if (not isinstance(value, dict) or set(value) != {
            "action_spec", "task_variation", "objects", "source", "sha256"}
            or value["action_spec"] != ACTION_SPEC or value["source"] != INPUT_SOURCE
            or value["task_variation"] not in TASKS.get(subset, ())
            or not isinstance(value["objects"], dict) or len(value["objects"]) > 128
            or set(value["objects"]) != {f"object {i + 1}" for i in range(len(value["objects"]))}):
        raise ContractError("invalid_manipulation_public_input")
    for coord in value["objects"].values():
        if (not isinstance(coord, list) or len(coord) != 3
                or any(type(x) is not int or not -1000 <= x <= 1000 for x in coord)):
            raise ContractError("invalid_manipulation_object_coordinate")
    if value["sha256"] != sha256(canonical({k: v for k, v in value.items() if k != "sha256"})):
        raise ContractError("manipulation_input_hash_mismatch")
    return value


def source(name):
    manifest = json_object((ROOT / "manipulation_sources.json").read_bytes())
    row = next((r for r in manifest["files"] if r["path"] == name), None)
    if row is None:
        raise ContractError("unverified_manipulation_source")
    data = (ROOT / "upstream-manipulation" / name).read_bytes()
    if sha256(data) != row["sha256"] or len(data) != row["bytes"]:
        raise ContractError("manipulation_source_changed")
    return data


def verify_sources():
    manifest = json_object((ROOT / "manipulation_sources.json").read_bytes())
    for row in manifest["files"]:
        source(row["path"])
    return sha256(canonical(manifest))
