"""Qwen-only navigation using the pinned official prompt-building methods."""
from __future__ import annotations

import ast
import base64
from dataclasses import dataclass

from planner import NavigationPlanner
from protocol import (ACTIONS, ROOT, REVISION, ContractError, OperationalError,
                      Plan, action_id, canonical, json_object, sha256)

PROMPT_VERSION = "embodiedrunner-official-nav-bare/1"
DEVIATIONS = [
    "Qwen3.8-Flash replaces the configured upstream model",
    "JSON-object API mode replaces upstream SDK parsed JSON schema",
    "seed=137 and enable_thinking=false are explicit",
    "Malformed plans end the episode; no random-action fallback",
    "Service or simulator failures stop the run; no automatic retry",
]


def official_tree(relative_path, upstream=ROOT / "upstream"):
    manifest = json_object((ROOT / "official_sources.json").read_text())
    if manifest["revision"] != REVISION:
        raise ContractError("upstream_revision_mismatch")
    record = next((row for row in manifest["files"]
                   if row["path"] == relative_path), None)
    data = (upstream / relative_path).read_bytes()
    if record is None or sha256(data) != record["sha256"]:
        raise ContractError("official_source_changed")
    return ast.parse(data, filename=relative_path), record["sha256"]


def literal(relative_path, name):
    tree, digest = official_tree(relative_path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value), digest
    raise ContractError("official_prompt_literal_missing")


def _png_url(image_path):
    return "data:image/png;base64," + base64.b64encode(image_path).decode("ascii")


def prompt_adapter(n_shots):
    """Compile only verified prompt methods, without importing model providers."""
    tree, digest = official_tree("embodiedbench/planner/nav_planner.py")
    original = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "EBNavigationPlanner")
    selected = {"__init__", "set_actions", "get_availabel_action_prompt",
                "process_prompt", "get_message"}
    methods = [node for node in original.body
               if isinstance(node, ast.FunctionDef) and node.name in selected]
    if {node.name for node in methods} != selected:
        raise ContractError("official_prompt_methods_missing")
    constructor = next(node for node in methods if node.name == "__init__")
    # The final constructor statement creates RemoteModel/CustomModel. All
    # earlier initialization and all selected prompt methods remain verbatim.
    final = constructor.body[-1]
    if not (isinstance(final, ast.If) and ast.unparse(final.test) == "model_type == 'custom'"):
        raise ContractError("official_constructor_shape_changed")
    constructor.body = constructor.body[:-1]
    cls = ast.ClassDef(name=original.name, bases=[], keywords=[],
                       body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    template, template_hash = literal("embodiedbench/planner/planner_utils.py", "template")
    template_lang, _ = literal("embodiedbench/planner/planner_utils.py", "template_lang")
    system, system_hash = literal(
        "embodiedbench/evaluator/config/system_prompts.py", "eb_navigation_system_prompt")
    examples, examples_hash = literal(
        "embodiedbench/evaluator/config/eb_navigation_example.py", "examples")
    namespace = {"template": template, "template_lang": template_lang,
                 "MESSAGE_WINDOW_LEN": 5, "local_image_to_data_url": _png_url}
    exec(compile(module, "verified-official-navigation-prompt", "exec"), namespace)
    adapter = namespace[original.name](
        model_name="qwen3.8-flash", model_type="remote", actions=list(ACTIONS),
        system_prompt=system, examples=examples, n_shot=n_shots,
        chat_history=False, language_only=False, multiview=False, multistep=False,
        visual_icl=False, truncate=True, kwargs={},
    )
    return adapter, {
        "nav_planner": digest, "planner_utils": template_hash,
        "system_prompts": system_hash, "examples": examples_hash,
    }


@dataclass(frozen=True)
class BareConfig:
    mode: str = "direct"
    n_shots: int = 3
    seed: int = 137
    max_tokens: int = 4096
    max_requests_per_episode: int = 20
    enable_thinking: bool = False

    def __post_init__(self):
        if self.mode != "direct" or type(self.seed) is not int:
            raise ContractError("bare_model_configuration_required")
        if type(self.n_shots) is not int or self.n_shots not in (0, 3):
            raise ContractError("unsupported_official_example_count")
        if type(self.max_tokens) is not int or not 128 <= self.max_tokens <= 8192:
            raise ContractError("invalid_max_tokens")
        if type(self.max_requests_per_episode) is not int or not 1 <= self.max_requests_per_episode <= 20:
            raise ContractError("invalid_episode_budget")
        if self.enable_thinking is not False:
            raise ContractError("bare_pilot_requires_thinking_disabled")


class OfficialBarePlanner(NavigationPlanner):
    uses_official_prompt = True
    uses_official_messages = True
    protocol_deviations = DEVIATIONS

    def __init__(self, client, config=BareConfig()):
        self.adapter, self.official_prompt_hashes = prompt_adapter(config.n_shots)
        super().__init__(client, config)

    def reset(self):
        super().reset()
        self.feedback = []

    def messages(self, observation, instruction):
        prompt = self.adapter.process_prompt(instruction, prev_act_feedback=self.feedback)
        return self.adapter.get_message(observation.png, prompt)

    def act(self, observation, user_instruction):
        if self.planner_steps >= self.config.max_requests_per_episode:
            raise ContractError("episode_request_budget_exhausted")
        if observation.step >= 20:
            raise ContractError("environment_step_budget_exhausted")
        messages = self.messages(observation, user_instruction)
        self.planner_steps += 1
        self.total_requests += 1
        self.last_receipt = {
            "prompt_version": PROMPT_VERSION, "n_shots": self.config.n_shots,
            "messages_sha256": sha256(canonical(messages)),
            "image_sha256": observation.image_sha256, "skill_sha256": None,
            "request_index": self.planner_steps, "environment_step": observation.step,
            "reference_answer_visible": False, "model_tool_calls": 0,
            "input_image_count": 1, "message_count": 1,
        }
        try:
            text = self.client.chat(
                messages, max_tokens=self.config.max_tokens, seed=self.config.seed,
                temperature=0.0, response_format={"type": "json_object"},
                enable_thinking=False,
            )
        except Exception:
            self.last_receipt["status"] = "service_error"
            error = getattr(self.client, "last_service_error", {}) or {}
            if error.get("reason_code") in {
                "missing_credentials", "http_error", "timeout", "transport_error", "invalid_response"
            }:
                self.last_receipt["service_reason_code"] = error["reason_code"]
            if type(error.get("http_status")) is int:
                self.last_receipt["http_status"] = error["http_status"]
            raise OperationalError("model_service_error") from None
        usage = getattr(self.client, "last_usage", {}) or {}
        self.last_receipt["usage"] = {
            key: value for key, value in usage.items()
            if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
            and type(value) is int and value >= 0
        }
        self.last_receipt["response_sha256"] = sha256(text.encode())
        identity = getattr(self.client, "last_response_identity", {}) or {}
        if isinstance(identity.get("model"), str):
            self.last_receipt["reported_model_sha256"] = sha256(identity["model"].encode())
        self.client.last_reasoning_content = None
        try:
            value = json_object(text)
            items = value.get("executable_plan")
            if not isinstance(items, list) or not items:
                raise ContractError("invalid_plan_length")
            # Upstream executes action_id and ignores the accompanying names.
            # Do not add a 2-action cap: first/following prompts advise 1-2/5-6.
            executed = items[:20 - observation.step]
            actions = tuple(action_id(item["action_id"]) for item in executed)
        except (ContractError, KeyError, TypeError):
            self.output_json_error += 1
            self.last_receipt["status"] = "planner_error"
            raise ContractError("invalid_official_action_plan") from None
        self.last_receipt["status"] = "valid_plan"
        self.last_receipt["action_ids"] = list(actions)
        self.last_receipt["proposed_action_count"] = len(items)
        return Plan(actions, {})

    def update_info(self, *, action, feedback, last_action_success, step):
        self.feedback.append([action_id(action), feedback])
