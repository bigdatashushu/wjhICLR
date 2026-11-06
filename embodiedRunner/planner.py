"""A label-free Qwen planner with optional, frozen navigation Skill delivery."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from protocol import ACTIONS, ContractError, OperationalError, canonical, parse_plan, sha256

PROMPT_VERSION = "embodiedrunner-nav-prompt/1"


@dataclass(frozen=True)
class PlannerConfig:
    mode: str = "skill"
    seed: int = 137
    max_tokens: int = 2048
    max_requests_per_episode: int = 20
    max_actions_per_plan: int = 2
    history_steps: int = 20
    enable_thinking: bool = False

    def __post_init__(self):
        if self.mode not in {"direct", "skill"} or type(self.seed) is not int:
            raise ContractError("invalid_planner_mode_or_seed")
        for name, low, high in (
            ("max_tokens", 128, 8192), ("max_requests_per_episode", 1, 20),
            ("max_actions_per_plan", 1, 8), ("history_steps", 1, 20),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ContractError(f"invalid_{name}")
        if type(self.enable_thinking) is not bool:
            raise ContractError("invalid_thinking")


class NavigationPlanner:
    def __init__(self, client, config=PlannerConfig(), *, skill_path=None):
        self.client, self.config = client, config
        self.skill = ""
        if config.mode == "skill":
            if skill_path is None:
                raise ContractError("skill_path_required")
            self.skill = Path(skill_path).read_text(encoding="utf-8")
            if not self.skill.strip() or len(self.skill) > 24000:
                raise ContractError("invalid_skill_source")
        self.skill_sha256 = sha256(self.skill.encode("utf-8")) if self.skill else None
        self.total_requests = 0
        self.reset()

    def reset(self):
        self.history = []
        self.planner_steps = 0
        self.output_json_error = 0
        self.last_receipt = {}

    def messages(self, observation, instruction):
        remaining = 20 - observation.step
        limit = min(self.config.max_actions_per_plan, remaining)
        if limit < 1:
            raise ContractError("environment_step_budget_exhausted")
        action_catalog = "\n".join(f"{index}: {name}" for index, name in enumerate(ACTIONS))
        system = (
            "You control a robot in EmbodiedBench EB-Navigation. Navigate to the object "
            "specified by the instruction using only the current first-person RGB and "
            "the action feedback. The environment executes the complete plan before "
            "you receive another image. Do not assume camera poses, target coordinates, "
            "metric depth, unseen objects, or task success.\n"
            f"Available actions (copy action_name exactly):\n{action_catalog}\n"
            f"Return 1 to {limit} actions as one JSON object with exactly these fields: "
            "visual_state_description (string), reasoning_and_reflection (string), "
            "language_plan (string), executable_plan (array of objects containing "
            "action_id (integer) and action_name (string)). Do not wrap JSON in Markdown."
        )
        if self.skill:
            system += "\n\nReference navigation Skill:\n" + self.skill
        text = canonical({
            "instruction": instruction, "environment_step": observation.step,
            "remaining_environment_steps": remaining,
            "action_history": self.history[-self.config.history_steps:],
        }).decode("utf-8")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,"
                           + base64.b64encode(observation.png).decode("ascii")}},
            ]},
        ]

    def act(self, observation, user_instruction):
        if self.planner_steps >= self.config.max_requests_per_episode:
            raise ContractError("episode_request_budget_exhausted")
        messages = self.messages(observation, user_instruction)
        self.planner_steps += 1
        self.total_requests += 1
        self.last_receipt = {
            "prompt_version": PROMPT_VERSION,
            "messages_sha256": sha256(canonical(messages)),
            "image_sha256": observation.image_sha256,
            "skill_sha256": self.skill_sha256,
            "request_index": self.planner_steps,
            "environment_step": observation.step,
            "reference_answer_visible": False,
        }
        try:
            text = self.client.chat(
                messages, max_tokens=self.config.max_tokens, seed=self.config.seed,
                temperature=0.0, response_format={"type": "json_object"},
                enable_thinking=self.config.enable_thinking,
            )
        except Exception:
            self.last_receipt["status"] = "service_error"
            error = getattr(self.client, "last_service_error", {}) or {}
            if error.get("reason_code") in {
                    "missing_credentials", "http_error", "timeout",
                    "transport_error", "invalid_response"}:
                self.last_receipt["service_reason_code"] = error["reason_code"]
            if type(error.get("http_status")) is int:
                self.last_receipt["http_status"] = error["http_status"]
            raise OperationalError("model_service_error") from None
        # Persist only numeric usage and hashes, never raw messages/model reasoning.
        usage = getattr(self.client, "last_usage", {}) or {}
        self.last_receipt["usage"] = {
            key: value for key, value in usage.items()
            if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
            and type(value) is int and value >= 0
        }
        self.last_receipt["response_sha256"] = sha256(text.encode("utf-8"))
        identity = getattr(self.client, "last_response_identity", {}) or {}
        if isinstance(identity.get("model"), str):
            self.last_receipt["reported_model_sha256"] = sha256(identity["model"].encode("utf-8"))
        try:
            plan = parse_plan(text, max_actions=min(
                self.config.max_actions_per_plan, 20 - observation.step))
        except ContractError:
            self.output_json_error += 1
            self.last_receipt["status"] = "planner_error"
            raise
        self.last_receipt["status"] = "valid_plan"
        self.last_receipt["action_ids"] = list(plan.actions)
        return plan

    def update_info(self, *, action, feedback, last_action_success, step):
        # This whitelist deliberately has no evaluator success/distance/target fields.
        self.history.append({
            "action_id": action, "action_name": ACTIONS[action], "step": step,
            "env_feedback": feedback, "last_action_success": last_action_success,
        })
        self.history = self.history[-self.config.history_steps:]
