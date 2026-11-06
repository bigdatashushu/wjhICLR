"""Pinned VLMPlanner prompt, messages, feedback, and action parser for Qwen."""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import json

from household_protocol import source
from protocol import ContractError, OperationalError, canonical, sha256

DEVIATIONS = [
    "Qwen3.8-Flash replaces the configured upstream model",
    "JSON-object API mode replaces upstream SDK parsed JSON schema",
    "seed=137, temperature=0, max_tokens=4096, enable_thinking=false are explicit",
    "Service or simulator failures stop the run without automatic retry",
    "Non-integer action IDs count as malformed plans",
]


def literal(name, variable):
    for node in ast.parse(source(name)).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(x, ast.Name) and x.id == variable for x in node.targets):
            return ast.literal_eval(node.value)
    raise ContractError("official_literal_missing")


def adapter(environment, subset, actions):
    name = "embodiedbench/planner/vlm_planner.py"
    original = next(x for x in ast.parse(source(name)).body
                    if isinstance(x, ast.ClassDef) and x.name == "VLMPlanner")
    selected = {"__init__", "set_actions", "get_availabel_action_prompt", "process_prompt",
                "get_message", "reset", "update_info", "json_to_action"}
    methods = [x for x in original.body if isinstance(x, ast.FunctionDef) and x.name in selected]
    constructor = next(x for x in methods if x.name == "__init__")
    provider = [x for x in constructor.body if isinstance(x, ast.If)
                and ast.unparse(x.test) == "model_type == 'custom'"]
    if len(provider) != 1 or {x.name for x in methods} != selected:
        raise ContractError("official_constructor_shape_changed")
    constructor.body.remove(provider[0])
    module = ast.fix_missing_locations(ast.Module(body=[
        ast.ClassDef(name="VLMPlanner", bases=[], keywords=[], body=methods, decorator_list=[])],
        type_ignores=[]))
    variable = "alfred_system_prompt" if environment == "alfred" else "habitat_system_prompt"
    system = literal("embodiedbench/evaluator/config/system_prompts.py", variable)
    example_file = ("alfred_long_horizon_examples.json" if subset == "long_horizon" else "alfred_examples.json") \
        if environment == "alfred" else "habitat_examples.json"
    examples = json.loads(source("embodiedbench/evaluator/config/" + example_file))
    # The real callback accepts a path; our sentinel maps to the current PNG.
    namespace = {"json": json}
    exec(compile(module, "verified-official-household-planner", "exec"), namespace)
    result = namespace["VLMPlanner"](
        "qwen3.8-flash", "remote", list(actions), system, examples, n_shot=10,
        obs_key="head_rgb", chat_history=False, language_only=False, use_feedback=True,
        multistep=False, kwargs={})
    result.reset()
    result.prompt_hashes = {
        name: sha256(source(name)),
        "system_prompts": sha256(source("embodiedbench/evaluator/config/system_prompts.py")),
        "examples": sha256(source("embodiedbench/evaluator/config/" + example_file)),
    }
    result.namespace = namespace
    return result


class OfficialHouseholdPlanner:
    def __init__(self, client, environment, subset, actions):
        self.client = client
        self.adapter = adapter(environment, subset, actions)
        self.total_requests = 0
        self.last_receipt = {}

    def reset(self, actions):
        self.adapter.reset()
        self.adapter.set_actions(list(actions))

    def messages(self, obs):
        data_url = "data:image/png;base64," + base64.b64encode(obs.png).decode()
        self.adapter.namespace["local_image_to_data_url"] = lambda image_path: data_url
        prompt = self.adapter.process_prompt(obs.instruction,
                                            prev_act_feedback=self.adapter.episode_act_feedback)
        return self.adapter.get_message("current-observation.png", prompt)

    def act(self, obs):
        if self.adapter.planner_steps >= 40:
            raise ContractError("episode_request_budget_exhausted")
        messages = self.messages(obs)
        self.adapter.planner_steps += 1
        self.total_requests += 1
        self.last_receipt = {
            "request_index": self.adapter.planner_steps, "environment_step": obs.step,
            "messages_sha256": sha256(canonical(messages)), "image_sha256": obs.image_sha256,
            "action_space_sha256": sha256(canonical(obs.actions)),
            "input_sha256": obs.input_sha256, "input_image_count": 1, "message_count": 1,
            "reference_answer_visible": False, "model_tool_calls": 0,
            "skill_sha256": None,
        }
        try:
            text = self.client.chat(messages, max_tokens=4096, seed=137, temperature=0,
                                    response_format={"type": "json_object"}, enable_thinking=False)
        except Exception:
            self.last_receipt["status"] = "service_error"
            raise OperationalError("model_service_error") from None
        self.last_receipt["response_sha256"] = sha256(text.encode())
        self.last_receipt["usage"] = {
            k: v for k, v in (getattr(self.client, "last_usage", {}) or {}).items()
            if k in ("prompt_tokens", "completion_tokens", "total_tokens") and type(v) is int}
        identity = getattr(self.client, "last_response_identity", {}) or {}
        self.last_receipt["reported_model_sha256"] = sha256(str(identity.get("model", "")).encode())
        self.client.last_reasoning_content = None
        try:
            value = json.loads(text)
            items = value["executable_plan"]
            if not isinstance(items, list) or any(type(x["action_id"]) is not int for x in items):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            self.adapter.output_json_error += 1
            action = -1
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                action = self.adapter.json_to_action(text)
        self.last_receipt.update(status="empty_plan" if action == -2 else
                                 "invalid_plan" if action == -1 else "valid_plan",
                                 action_ids=action)
        return action

    def update(self, action, feedback):
        self.adapter.update_info({"action_id": action, "env_feedback": feedback})
