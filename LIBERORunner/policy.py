from __future__ import annotations

import hashlib
import json
import math

from .observation import build_messages, public_observation
from .storage import write_json


class InvalidAction(ValueError):
    pass


class ModelIdentityError(RuntimeError):
    pass


def parse_action(text: str, low, high) -> list[float]:
    """No clipping, unit conversion, hidden repair, or extra control calls."""
    clean = text.strip()
    if clean.startswith("```json\n") and clean.endswith("\n```"):
        clean = clean[8:-4].strip()
    elif clean.startswith("```\n") and clean.endswith("\n```"):
        clean = clean[4:-4].strip()
    try:
        body = json.loads(clean)
    except (TypeError, ValueError):
        raise InvalidAction("Expected one JSON action object.") from None
    if not isinstance(body, dict) or set(body) != {"action"}:
        raise InvalidAction("Expected exactly the action field.")
    action = body["action"]
    if not isinstance(action, list) or len(action) != 7:
        raise InvalidAction("Expected a 7-dimensional action.")
    for i, value in enumerate(action):
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not low[i] <= value <= high[i]):
            raise InvalidAction(f"Invalid action coordinate {i}.")
    return [float(v) for v in action]


def make_system_prompt(controller: dict, orientation: str) -> str:
    return (
        "You control a simulated Franka Panda robot in LIBERO. Complete the task "
        "using the current third-person RGB image, wrist RGB image, and robot "
        "proprioception. Return ONLY one JSON object: "
        '{"action":[dx,dy,dz,drx,dry,drz,gripper]}. '
        "The seven numbers are normalized OSC_POSE controller inputs. The first "
        "three specify translation deltas in the controller's world frame. The "
        "next three specify delta axis-angle rotation commands, not absolute "
        "Euler angles. Gripper +1 closes and -1 opens. Use small movements when "
        "approaching or grasping objects. Each response executes exactly ONE "
        "20 Hz control step; you will then receive a new observation. There is "
        "no stop or success command; the environment evaluates task completion. "
        "robot0_eef_pos is in world meters; robot0_eef_quat is [x,y,z,w]; "
        "robot0_gripper_qpos and robot0_joint_pos are joint positions. "
        f"Image orientation: {orientation}. 'raw' preserves the simulator array, "
        "'flip_vertical' flips image rows, and 'rotate_180' reverses rows and "
        "columns. Image screen directions are not world-coordinate directions. "
        "Controller bounds and scaling are supplied by the environment: "
        + json.dumps(controller, sort_keys=True, allow_nan=False)
    )


class GPTPolicy:
    def __init__(self, client, config, controller: dict, journal, image_dir):
        self.client = client
        self.config = config
        self.controller = controller
        self.journal = journal
        self.image_dir = image_dir
        self.prompt = make_system_prompt(controller, config.image_orientation)
        self.last_action = None

    def act(self, obs: dict, language: str, step: int) -> list[float]:
        from xcodeapi import RequestLimitError
        if (self.client.max_requests is not None
                and self.client.attempts >= self.client.max_requests):
            raise RequestLimitError("The cumulative request budget is exhausted.")
        state, images = public_observation(obs, self.config.image_orientation)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        identity = {}
        for name, raw in images.items():
            path = self.image_dir / f"{step:04d}-{name}.png"
            path.write_bytes(raw)
            identity[name] = {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}
        write_json(self.image_dir / f"{step:04d}-state.json", state)
        messages = build_messages(language, step, state, images, self.last_action, self.prompt)
        payload = self.client.prepare(
            messages, max_output_tokens=self.config.max_output_tokens,
            extra_body={"reasoning": {"effort": self.config.reasoning_effort}},
        )
        self.journal({"event": "request_started", "step": step,
                      "request_index": self.client.attempts + 1,
                      "payload_sha256": hashlib.sha256(json.dumps(
                          payload, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                      "images": identity})
        reply = self.client.send(payload)
        # Persist only fixed-format response metadata, never a native API body.
        self.journal({"event": "response_received", "step": step,
                      "metadata": reply.metadata(),
                      "model_matches": reply.returned_model == self.config.model,
                      "finish_reason": reply.finish_reason,
                      "response_text_sha256": hashlib.sha256(reply.text.encode()).hexdigest()})
        if reply.returned_model != self.config.model:
            raise ModelIdentityError("Returned model does not match the requested model.")
        if reply.finish_reason != "completed":
            raise InvalidAction("Responses API did not finish the action output.")
        action = parse_action(reply.text, self.controller["action_low"], self.controller["action_high"])
        self.last_action = action
        self.journal({"event": "action_parsed", "step": step, "action": action})
        return action
