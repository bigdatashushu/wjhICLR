"""Navigation contracts shared by the Python 3.11 agent and Python 3.9 simulator."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

VERSION = "embodiedrunner-navigation/1"
REVISION = "9be4e980e9cd6bcb38373cd4aab7c32724bdd401"
ROOT = Path(__file__).resolve().parent
SUBSETS = ("base", "common_sense", "complex_instruction",
           "visual_appearance", "long_horizon")
ACTIONS = (
    "Move forward by 0.25", "Move backward by 0.25",
    "Move rightward by 0.25", "Move leftward by 0.25",
    "Rotate to the right by 90 degrees.", "Rotate to the left by 90 degrees.",
    "Tilt the camera upward by 30 degrees.", "Tilt the camera downward by 30 degrees.",
)


class ContractError(ValueError):
    pass


class OperationalError(RuntimeError):
    """A service/environment failure stops the run without automatic retry."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_object(text):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ContractError("duplicate_json_key")
            value[key] = item
        return value

    def invalid_constant(_value):
        raise ContractError("nonfinite_json")

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)
    except (ValueError, TypeError) as exc:
        raise ContractError("invalid_json") from None
    if not isinstance(value, dict):
        raise ContractError("json_object_required")
    return value


def action_id(value):
    if type(value) is not int or not 0 <= value < len(ACTIONS):
        raise ContractError("invalid_action_id")
    return value


def within_root(path, *, base=ROOT):
    candidate = Path(path)
    result = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if not result.is_relative_to(base.resolve()) or result == base.resolve():
        raise ContractError("output_must_be_inside_embodiedRunner")
    return result


def verified_sources(upstream, subsets):
    upstream = Path(upstream).resolve()
    manifest = json_object((ROOT / "official_sources.json").read_text(encoding="utf-8"))
    if manifest["revision"] != REVISION:
        raise ContractError("upstream_revision_mismatch")
    expected = {item["path"]: item for item in manifest["files"]}
    paths = ["embodiedbench/envs/eb_navigation/EBNavEnv.py",
             "embodiedbench/envs/eb_navigation/utils.py"]
    paths += [f"embodiedbench/envs/eb_navigation/datasets/{subset}.json"
              for subset in subsets]
    hashes = {}
    for name in paths:
        if name not in expected:
            raise ContractError("missing_official_source_identity")
        path = (upstream / name).resolve()
        if not path.is_relative_to(upstream):
            raise ContractError("source_path_escaped")
        data = path.read_bytes()
        record = expected[name]
        if (sha256(data) != record["sha256"]
                or hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data
                                ).hexdigest() != record["git_blob_sha"]):
            raise ContractError("official_source_changed")
        hashes[name] = record["sha256"]
    return hashes


@dataclass(frozen=True)
class Task:
    subset: str
    index: int
    input_sha256: str

    @property
    def task_id(self):
        return f"{self.subset}:{self.index:03d}"

    def public_dict(self):
        return {"task_id": self.task_id, "subset": self.subset,
                "index": self.index, "input_sha256": self.input_sha256}


def select_tasks(upstream, subsets, limit_per_subset):
    if (not subsets or len(set(subsets)) != len(subsets)
            or any(subset not in SUBSETS for subset in subsets)):
        raise ContractError("invalid_eval_sets")
    if type(limit_per_subset) is not int or not 1 <= limit_per_subset <= 60:
        raise ContractError("invalid_limit_per_subset")
    verified_sources(upstream, subsets)
    tasks = []
    for subset in subsets:
        source = Path(upstream) / f"embodiedbench/envs/eb_navigation/datasets/{subset}.json"
        rows = json_object(source.read_text(encoding="utf-8"))["tasks"]
        if not isinstance(rows, list) or len(rows) != 60:
            raise ContractError("navigation_subset_shape_changed")
        tasks.extend(Task(subset, index, sha256(canonical(row)))
                     for index, row in enumerate(rows[:limit_per_subset]))
    return tuple(tasks)


@dataclass(frozen=True)
class Observation:
    """Only current RGB and its public identity are visible to the planner."""
    png: bytes
    image_sha256: str
    step: int


@dataclass(frozen=True)
class Transition:
    observation: Observation
    instruction: str
    done: bool
    task_success: bool
    last_action_success: bool
    feedback: str


def decode_transition(packet, *, resolution):
    if set(packet) != {"image_base64", "image_sha256", "step", "instruction",
                       "done", "task_success", "last_action_success", "env_feedback"}:
        raise ContractError("unexpected_observation_fields")
    if (type(packet["step"]) is not int or not 0 <= packet["step"] <= 20
            or any(type(packet[key]) is not bool for key in
                   ("done", "task_success", "last_action_success"))
            or not isinstance(packet["instruction"], str) or not packet["instruction"]
            or not isinstance(packet["env_feedback"], str)):
        raise ContractError("invalid_transition")
    try:
        png = base64.b64decode(packet["image_base64"], validate=True)
        from PIL import Image
        with Image.open(io.BytesIO(png)) as image:
            if image.format != "PNG" or image.size != (resolution, resolution):
                raise ContractError("unexpected_rgb_shape")
            image.verify()
    except (ValueError, TypeError, OSError):
        raise ContractError("invalid_rgb") from None
    if sha256(png) != packet["image_sha256"]:
        raise ContractError("rgb_hash_mismatch")
    return Transition(
        Observation(png, packet["image_sha256"], packet["step"]),
        packet["instruction"], packet["done"], packet["task_success"],
        packet["last_action_success"], packet["env_feedback"],
    )


@dataclass(frozen=True)
class Plan:
    actions: tuple
    reasoning: dict


def parse_plan(text, *, max_actions):
    value = json_object(text)
    fields = {"visual_state_description", "reasoning_and_reflection",
              "language_plan", "executable_plan"}
    if set(value) != fields:
        raise ContractError("unexpected_plan_fields")
    if any(not isinstance(value[key], str) for key in fields - {"executable_plan"}):
        raise ContractError("invalid_plan_text")
    items = value["executable_plan"]
    if not isinstance(items, list) or not 1 <= len(items) <= max_actions:
        raise ContractError("invalid_plan_length")
    actions = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {"action_id", "action_name"}:
            raise ContractError("invalid_plan_action")
        number = action_id(item["action_id"])
        if item["action_name"] != ACTIONS[number]:
            raise ContractError("action_name_id_mismatch")
        actions.append(number)
    return Plan(tuple(actions), value)
