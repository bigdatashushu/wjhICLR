"""Explicit official-packet adapters. V1 workers and baseline entries are unchanged."""
from __future__ import annotations

import base64
import io
import math

from household_protocol import SUBSETS as HOUSEHOLD_SUBSETS
from mutation_protocol import action_table_hash, hash_string, nonnegative
from protocol import ACTIONS, SUBSETS, ContractError, canonical, sha256

PACKET_FIELDS = frozenset({
    "environment", "official_task_id", "subset", "input_sha256", "image_base64",
    "image_sha256", "step", "instruction", "actions", "env_feedback",
    "last_action_success", "official"})
OFFICIAL_FIELDS = frozenset({"done", "task_success", "task_progress", "invalid_actions"})


def validate_packet(packet):
    """Host/worker only: scorer fields remain in a separate private mapping."""
    manip = type(packet) is dict and packet.get("environment") == "eb_manipulation"
    extra = {"manipulation_input", "planner_image_base64", "planner_image_sha256"} if manip else set()
    if type(packet) is not dict or set(packet) != PACKET_FIELDS | extra:
        raise ContractError("unexpected_v2_packet_fields")
    environment = packet["environment"]
    if environment not in {"eb_navigation", "eb_alfred", "eb_habitat", "eb_manipulation"}:
        raise ContractError("invalid_v2_environment")
    nav = environment == "eb_navigation"
    nonnegative(packet["step"])
    if packet["step"] > (15 if manip else 20 if nav else 30):
        raise ContractError("invalid_v2_step")
    for key in ("official_task_id", "subset"):
        if type(packet[key]) is not str or not 1 <= len(packet[key]) <= 200:
            raise ContractError("invalid_v2_task_identity")
    for key in ("input_sha256", "image_sha256"):
        hash_string(packet[key])
    if (type(packet["instruction"]) is not str or not 1 <= len(packet["instruction"]) <= 32000
            or type(packet["env_feedback"]) is not str or len(packet["env_feedback"]) > 30000
            or type(packet["last_action_success"]) is not bool):
        raise ContractError("invalid_v2_feedback")
    action_table_hash(packet["actions"])
    if manip:
        from manipulation_protocol import public_input
        if packet["actions"]:
            raise ContractError("manipulation_has_no_action_ids")
        public_input(packet["manipulation_input"], packet["subset"])
    elif not packet["actions"]:
        raise ContractError("empty_official_action_table")
    official = packet["official"]
    if (type(official) is not dict or set(official) != OFFICIAL_FIELDS
            or any(type(official[key]) is not bool for key in ("done", "task_success"))
            or type(official["task_progress"]) not in {int, float}
            or not math.isfinite(official["task_progress"])
            or not 0 <= official["task_progress"] <= 1):
        raise ContractError("invalid_v2_official_status")
    nonnegative(official["invalid_actions"])
    if official["invalid_actions"] > (0 if nav or manip else 10):
        raise ContractError("invalid_v2_invalid_count")
    if type(packet["image_base64"]) is not str or len(packet["image_base64"]) > 11_000_000:
        raise ContractError("invalid_v2_rgb")
    try:
        from PIL import Image
        png = base64.b64decode(packet["image_base64"], validate=True)
        with Image.open(io.BytesIO(png)) as image:
            if image.format != "PNG" or image.size != (500, 500):
                raise ContractError("unexpected_rgb_shape")
            image.verify()
    except (ValueError, OSError, TypeError):
        raise ContractError("invalid_v2_rgb") from None
    if sha256(png) != packet["image_sha256"]:
        raise ContractError("v2_rgb_hash_mismatch")
    if manip:
        hash_string(packet["planner_image_sha256"])
        try:
            if type(packet["planner_image_base64"]) is not str or len(packet["planner_image_base64"]) > 11_000_000:
                raise ValueError
            planner = base64.b64decode(packet["planner_image_base64"], validate=True)
            with Image.open(io.BytesIO(planner)) as image:
                if image.format != "PNG" or image.size != (500, 500):
                    raise ValueError
                image.verify()
            if sha256(planner) != packet["planner_image_sha256"]:
                raise ValueError
        except (ValueError, OSError, TypeError):
            raise ContractError("invalid_v2_planner_rgb") from None
    return png


class NavigationAdapter:
    environment = "eb_navigation"
    validate_packet = staticmethod(validate_packet)

    def __init__(self, official):
        self.official, self.subset, self.index, self.input_sha256 = official, "", 0, ""

    def validate_open(self, parameters):
        if parameters["subset"] not in SUBSETS or parameters["index"] >= 60:
            raise ContractError("invalid_navigation_selection")

    def _normalize(self, raw):
        from protocol import decode_transition
        decode_transition(raw, resolution=500)
        return {
            "environment": self.environment, "official_task_id": f"{self.subset}:{self.index:03d}",
            "subset": self.subset, "input_sha256": self.input_sha256,
            "image_base64": raw["image_base64"], "image_sha256": raw["image_sha256"],
            "step": raw["step"], "instruction": raw["instruction"], "actions": list(ACTIONS),
            "env_feedback": raw["env_feedback"], "last_action_success": raw["last_action_success"],
            "official": {"done": raw["done"], "task_success": raw["task_success"],
                         "task_progress": 0.0, "invalid_actions": 0},
        }

    def open(self, parameters):
        self.subset, self.index, self.input_sha256 = (
            parameters["subset"], parameters["index"], parameters["source_sha256"])
        return self._normalize(self.official.open(self.subset, self.index, self.input_sha256))

    def step(self, action, first_in_plan):
        return self._normalize(self.official.step(action, first_in_plan))

    def close(self):
        self.official.close()


class HouseholdAdapter:
    validate_packet = staticmethod(validate_packet)

    def __init__(self, official, environment):
        if environment not in {"eb_alfred", "eb_habitat"}:
            raise ContractError("invalid_household_environment")
        self.official, self.environment = official, environment
        self.subset, self.index = official.subset, 0

    def validate_open(self, parameters):
        kind = self.environment.removeprefix("eb_")
        if parameters["subset"] not in HOUSEHOLD_SUBSETS[kind] or parameters["subset"] != self.subset:
            raise ContractError("invalid_household_selection")

    def _normalize(self, raw):
        from household_protocol import decode
        decode(raw)
        return {
            "environment": self.environment, "official_task_id": f"{self.subset}:{self.index:03d}",
            "subset": self.subset, "input_sha256": raw["input_sha256"],
            "image_base64": raw["image_base64"], "image_sha256": raw["image_sha256"],
            "step": raw["step"], "instruction": raw["instruction"], "actions": raw["actions"],
            "env_feedback": raw["env_feedback"], "last_action_success": raw["last_action_success"],
            "official": {"done": raw["done"], "task_success": raw["task_success"],
                         "task_progress": raw["task_progress"], "invalid_actions": raw["invalid_actions"]},
        }

    def open(self, parameters):
        self.index = parameters["index"]
        return self._normalize(self.official.open(self.index, parameters["source_sha256"]))

    def step(self, action, first_in_plan):
        return self._normalize(self.official.step(action))

    def invalid(self):
        return self._normalize(self.official.invalid())

    def close(self):
        self.official.close()


class FakeAdapter:
    """Deterministic offline endpoint with the same observation/mutation shape."""
    validate_packet = staticmethod(validate_packet)

    def __init__(self, environment, *, done_at=2, failed_steps=(), stationary_rgb=False):
        if environment not in {"eb_navigation", "eb_alfred", "eb_habitat", "eb_manipulation"}:
            raise ContractError("invalid_fake_environment")
        self.environment, self.done_at = environment, done_at
        self.failed_steps, self.stationary_rgb = set(failed_steps), stationary_rgb
        self.step_count, self.invalid_count, self.open_count = 0, 0, 0
        self.closed, self.subset, self.index, self.input_sha256 = False, "", 0, ""
        self.actions = list(ACTIONS) if environment == "eb_navigation" else [
            "Navigate to the chair", "Pick up the cup", "Put the cup on the table"]
        if environment == "eb_manipulation":
            self.actions = []
        self._last = None

    def validate_open(self, parameters):
        if self.environment == "eb_manipulation":
            from manipulation_protocol import SUBSETS as subsets
        else:
            subsets = SUBSETS if self.environment == "eb_navigation" else HOUSEHOLD_SUBSETS[self.environment[3:]]
        if parameters["subset"] not in subsets or parameters["index"] >= 60:
            raise ContractError("invalid_fake_selection")

    def _packet(self, *, feedback="", action_success=True):
        from PIL import Image
        stream = io.BytesIO()
        color = (90, 100, 120) if self.stationary_rgb else (90 + self.step_count, 100, 120)
        Image.new("RGB", (500, 500), color).save(stream, format="PNG")
        png = stream.getvalue()
        done = self.step_count >= self.done_at or self.invalid_count >= 10
        success = self.step_count >= self.done_at and action_success
        result = {
            "environment": self.environment, "official_task_id": f"{self.subset}:{self.index:03d}",
            "subset": self.subset, "input_sha256": self.input_sha256,
            "image_base64": base64.b64encode(png).decode("ascii"), "image_sha256": sha256(png),
            "step": self.step_count, "instruction": "Find the chair and inspect the cup.",
            "actions": list(self.actions), "env_feedback": feedback,
            "last_action_success": action_success,
            "official": {"done": done, "task_success": success,
                         "task_progress": min(1.0, self.step_count / self.done_at),
                         "invalid_actions": self.invalid_count},
        }
        if self.environment == "eb_manipulation":
            from manipulation_protocol import ACTION_SPEC, INPUT_SOURCE
            value = {"action_spec": ACTION_SPEC, "task_variation": "pick_cube_shape",
                     "objects": {"object 1": [40 + self.step_count, 50, 12]},
                     "source": INPUT_SOURCE}
            result["manipulation_input"] = {**value, "sha256": sha256(canonical(value))}
            result["planner_image_base64"] = result["image_base64"]
            result["planner_image_sha256"] = result["image_sha256"]
            result["instruction"] = "Pick up the red cube."
        self._last = result
        return result

    def open(self, parameters):
        self.open_count += 1
        self.subset, self.index, self.input_sha256 = (
            parameters["subset"], parameters["index"], parameters["source_sha256"])
        return self._packet()

    def step(self, action, first_in_plan):
        self.step_count += 1
        success = self.step_count not in self.failed_steps
        return self._packet(feedback="Action completed" if success else "Action failed",
                            action_success=success)

    def invalid(self):
        self.invalid_count += 1
        result = {**self._last, "official": {
            **self._last["official"], "invalid_actions": self.invalid_count,
            "done": self.invalid_count >= 10}}
        self._last = result
        return result

    def close(self):
        self.closed = True
