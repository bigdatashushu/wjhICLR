"""One governed completion per decision/repair; public inputs and hash-only audit."""
from __future__ import annotations

import base64
import contextlib
import io
from action_contract import freeze, public_hash
from action_host import Proposal
from action_runtime.delivery import plan_delivery
from action_runtime.execution_contract import ActionExecutionContract
from budget import numeric_usage
from protocol import ContractError, OperationalError, canonical, json_object, sha256
from skill3d.synthesis.program_assembler import assemble_program_ex, SynthesisError

VERSION = "embodied-solver/1"
MODEL_CONFIG = freeze({
    "model": "qwen3.8-flash", "seed": 137, "temperature": 0,
    "max_tokens": 4096, "enable_thinking": False,
    "response_format": {"type": "json_object"},
    "timeout_s": 120, "automatic_retry": False,
})
TOOL_API = (
    "view_candidates(view_index=-1, category='') -> candidate regions, incomplete coverage; "
    "view_depth_summary() -> per-view relative quantiles and availability; "
    "point_depth(view_index,x,y) -> 5x5 finite local median, relative depth; "
    "pair_point_distance(view_index,a_xy,b_xy) -> SAME-view relative point distance; "
    "candidate_depth_summary(candidate_id) -> denied until independent mask validation. "
    "All successful tool replies have result_id and data. "
    "ReadObservation(result_id,fields=None,offset=0,limit=8) reads a current result. "
    "YieldObservations(result_ids,reason='') ends this segment."
)
PROGRAM_GUIDANCE = (
    "ActionPayload, SubmitActions, ReadObservation, YieldObservations and all listed tools are "
    "already injected globals. Do not import framework modules or redefine these names. "
    "The host invokes solve(ctx); do not add a main guard or call solve yourself. "
    "ctx exposes exactly the context_fields in the contract, with no nested observation/snapshot. "
    "ctx.action_table is a tuple of action NAME STRINGS; its zero-based index is the action ID. "
    "The serialized snapshot shows action records for planning, but runtime ctx.action_table contains strings. "
    "Read ctx.instruction, ctx.feedback, ctx.memory and ctx.capabilities directly. "
    "Submission shape example (choose the actual action from this observation):\n"
    "def solve(ctx):\n"
    "    SubmitActions(ActionPayload(ctx.decision_id, ctx.observation_revision, "
    "ctx.snapshot_sha256, ctx.action_table_sha256, [0], []))\n"
)


def program_guard_codes(source):
    """Finite reason taxonomy only; never retain generated identifiers/text."""
    from action_runtime.ast_guard import ast_guard
    from action_runtime.control_profile import EMBODIED_ACTIONS_PROFILE
    prefixes = (
        ("Module ", "forbidden_import"), ("Importing ", "forbidden_import"),
        ("Undefined function", "undefined_call"), ("Access outside public", "context_field"),
        ("Access to internal", "internal_name"), ("Interface is forbidden", "forbidden_interface"),
        ("Calling unsafe", "unsafe_call"), ("Access to unsafe", "unsafe_attribute"),
        ("Access to underscore", "internal_attribute"), ("Reassigning", "reserved_assignment"),
        ("Modifying", "context_mutation"), ("Generated programs", "class_definition"),
        ("Bare except", "bare_except"), ("Secondary regex", "regex_rejection"),
        ("Syntax error", "syntax_error"),
    )
    result = ast_guard(source, {
        "view_candidates", "view_depth_summary", "point_depth", "pair_point_distance",
        "candidate_depth_summary"}, profile=EMBODIED_ACTIONS_PROFILE)
    return sorted({next((code for prefix, code in prefixes if row.startswith(prefix)), "other")
                   for row in result.violations})


def make_qwen_client():
    """Construct only when live execution has separately been authorized."""
    from skill3d.synthesis.bailian_client import BailianClient
    return BailianClient(model=MODEL_CONFIG["model"], temperature=0,
                         max_output_tokens=4096, timeout_s=120)


class ActionSolver:
    def __init__(self, client, *, arm, environment, method=None, trajectory=None):
        if arm not in {"Direct", "B01", "B11", "Full"} or bool(method) != (arm in {"B11", "Full"}):
            raise ContractError("solver_arm_method_mismatch")
        if method is not None and method.environment != environment:
            raise ContractError("solver_method_environment")
        if (getattr(client, "model", MODEL_CONFIG["model"]) != MODEL_CONFIG["model"]
                or getattr(client, "timeout_s", 120) != 120):
            raise ContractError("solver_client_policy_mismatch")
        self.client, self.arm, self.environment, self.method = client, arm, environment, method
        self.contract = ActionExecutionContract(environment=environment)
        self.receipts, self.adapter, self.previous_observation = [], None, None
        self.prompt_hashes = {}
        self.direct_feedback = []
        self.previous_step, self.pending_action = 0, None
        self.trajectory = trajectory

    def _direct_messages(self, snapshot):
        obs = snapshot.observation
        if self.adapter is None:
            if self.environment == "eb_navigation":
                from official_bare import prompt_adapter
                self.adapter, self.prompt_hashes = prompt_adapter(3)
            elif self.environment == "eb_manipulation":
                from manipulation_planner import adapter
                self.adapter = adapter()
                self.prompt_hashes = self.adapter.prompt_hashes
            else:
                from household_planner import adapter
                self.adapter = adapter(
                    self.environment.removeprefix("eb_"), obs.subset, obs.action_table)
                self.prompt_hashes = self.adapter.prompt_hashes
        if self.previous_observation != obs.observation_revision:
            if obs.environment_step > self.previous_step and self.pending_action is not None:
                self.direct_feedback.append([self.pending_action, obs.public_feedback])
            self.direct_feedback = self.direct_feedback[-40:]
            self.previous_observation = obs.observation_revision
            self.previous_step = obs.environment_step
        if self.environment == "eb_navigation":
            self.adapter.set_actions(list(obs.action_table))
            prompt = self.adapter.process_prompt(obs.instruction, prev_act_feedback=self.direct_feedback)
            prompt += "\nCurrent official feedback: " + obs.public_feedback
            return self.adapter.get_message(obs.png, prompt)
        if self.environment == "eb_manipulation":
            data_url = "data:image/png;base64," + base64.b64encode(obs.planner_png).decode("ascii")
            self.adapter.namespace["local_image_to_data_url"] = lambda image_path: data_url
            objects = obs.manipulation_input["objects"]
            coordinates = {f"object {i+1}": list(objects[f"object {i+1}"]) for i in range(len(objects))}
            prompt, task_prompt = self.adapter.process_prompt(
                obs.instruction, str(coordinates), obs.manipulation_input["task_variation"],
                prev_act_feedback=self.direct_feedback)
            task_prompt += "\n\n" + self.adapter.output_template
            task_prompt += "\nCurrent official feedback: " + obs.public_feedback
            return self.adapter.get_message(["current-rgb"], prompt, task_prompt)
        self.adapter.set_actions(list(obs.action_table))
        data_url = "data:image/png;base64," + base64.b64encode(obs.png).decode("ascii")
        self.adapter.namespace["local_image_to_data_url"] = lambda image_path: data_url
        prompt = self.adapter.process_prompt(obs.instruction, prev_act_feedback=self.direct_feedback)
        prompt += "\nCurrent official feedback: " + obs.public_feedback
        return self.adapter.get_message("current-rgb", prompt)

    def messages(self, snapshot, feedback):
        if snapshot.observation.environment != self.environment or (
                snapshot.method_sha256 != (self.method.content_sha256 if self.method else "")):
            raise ContractError("solver_snapshot_identity")
        plan = plan_delivery([self.method] if self.method else [],
                             max_chars=self.contract.method_context_max_chars)
        if self.arm == "Direct":
            messages = self._direct_messages(snapshot)
            if feedback:
                # Same host repair rules; the original prompt/examples remain intact.
                messages.append({"role": "user", "content": canonical({"repair": feedback}).decode()})
        else:
            if self.method and not set(self.method.required_capabilities) <= set(snapshot.capabilities):
                raise ContractError("method_required_capability_missing")
            system = (
                "Control " + self.environment + " using one immutable current observation. "
                "Return one JSON object containing only program_source, a Python string defining solve(ctx). "
                "The host executes the first submitted action then provides a new observation. "
                "Use the provided public inputs and legal action table.\n"
                + self.contract.render() + "\nRuntime context and submission:\n" + PROGRAM_GUIDANCE
                + "\nTool API:\n" + TOOL_API)
            if self.environment == "eb_manipulation":
                system = system.replace(PROGRAM_GUIDANCE,
                    PROGRAM_GUIDANCE.replace(
                        "ctx.action_table is a tuple of action NAME STRINGS; its zero-based index is the action ID. "
                        "The serialized snapshot shows action records for planning, but runtime ctx.action_table contains strings. ",
                        "ctx.action_table is empty. Use ctx.action_spec and ctx.official_objects for seven-dimensional "
                        "world voxel gripper actions. ")
                    .replace("[0], []", "[[50, 50, 30, 120, 60, 90, 1]], []"))
            if plan.entries:
                system += "\nComplete frozen environment Skill:\n" + plan.entries[0].text
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": canonical({
                        "snapshot": snapshot.public_dict(), "repair": feedback}).decode()},
                    {"type": "image_url", "image_url": {"url":
                        "data:image/png;base64," + base64.b64encode(
                            snapshot.observation.planner_png or snapshot.observation.png).decode("ascii")}},
                ]},
            ]
        text_chars = sum(len(content) if isinstance(content := m["content"], str) else
                         sum(len(x.get("text", "")) for x in content) for m in messages)
        if text_chars > 96000:
            raise ContractError("prompt_context_cap")
        return messages, plan, text_chars

    def _parse(self, text, snapshot):
        try:
            value = json_object(text)
            if self.arm != "Direct":
                if (set(value) != {"program_source"} or type(value["program_source"]) is not str
                        or not value["program_source"].strip()):
                    raise ContractError("model_output_format_error")
                program, recovered = assemble_program_ex(
                    value["program_source"],
                    skill_semver_used=[f"{self.method.skill_id}@{self.method.version}"] if self.method else [],
                    intended_answer_slot="SubmitActions")
                if recovered or program.program_source.strip() != value["program_source"].strip():
                    raise ContractError("model_output_format_error")
                return Proposal(source=program.program_source)
            if self.environment == "eb_manipulation":
                from manipulation_planner import parse_plan
                actions = parse_plan(text)
                self.pending_action = list(actions[0]) if actions else None
                return Proposal(actions=actions)
            items = value["executable_plan"]
            if (type(items) is not list or len(items) > 30
                    or any(type(item) is not dict or type(item.get("action_id")) is not int
                           or not 0 <= item["action_id"] < len(snapshot.observation.action_table)
                           for item in items)):
                raise ContractError("invalid_action_plan")
            if self.environment != "eb_navigation":
                with contextlib.redirect_stdout(io.StringIO()):
                    parsed = self.adapter.json_to_action(text)
                if parsed == -2:
                    return Proposal(actions=())
                if parsed == -1 or tuple(parsed) != tuple(item["action_id"] for item in items):
                    raise ContractError("invalid_action_plan")
            # Validate the entire original proposal before applying its first action.
            actions = tuple(item["action_id"] for item in items)
            self.pending_action = actions[0] if actions else None
            return Proposal(actions=actions)
        except (KeyError, TypeError, ValueError, SynthesisError, ContractError):
            raise ContractError("model_output_format_error") from None

    def generate(self, snapshot, *, feedback, usage):
        messages, delivery, chars = self.messages(snapshot, feedback)
        receipt = {
            "schema": VERSION, "arm": self.arm, "environment": self.environment,
            "decision_id": snapshot.decision_id, "snapshot_sha256": snapshot.snapshot_sha256,
            "observation_revision": snapshot.observation.observation_revision,
            "messages_sha256": public_hash(messages), "rgb_sha256": snapshot.observation.rgb_sha256,
            "action_table_sha256": snapshot.observation.action_table_sha256,
            "method_sha256": snapshot.method_sha256,
            "evidence_manifest_sha256": snapshot.evidence_manifest_sha256,
            "model_config_sha256": public_hash(MODEL_CONFIG), "contract_sha256": self.contract.sha256,
            "official_prompt_hashes": self.prompt_hashes,
            "stage": "repair" if feedback else "synthesis", "text_chars": chars,
            "reference_answer_visible": False, "input_image_count": 1,
            "delivery": delivery.to_dict(), "status": "started",
        }
        self.receipts.append(receipt)
        if self.environment == "eb_manipulation":
            receipt["manipulation_input_sha256"] = snapshot.observation.manipulation_input["sha256"]
            receipt["planner_rgb_sha256"] = sha256(snapshot.observation.planner_png)
        if self.trajectory:
            self.trajectory.emit("model_request", receipt=receipt, messages=messages,
                                 model_config=MODEL_CONFIG)
        try:
            # Host already owns the single request permit; no extra governor here.
            self.client.last_usage = None
            text = self.client.chat(
                messages, max_tokens=4096, seed=137, temperature=0,
                response_format={"type": "json_object"}, enable_thinking=False)
            if self.trajectory:
                self.trajectory.emit(
                    "model_response", decision_id=snapshot.decision_id, text=text,
                    reasoning_content=getattr(self.client, "last_reasoning_content", None),
                    usage=numeric_usage(getattr(self.client, "last_usage", None)))
        except Exception:
            delivery.mark_request_failed()
            receipt.update(status="service_error", delivery=delivery.to_dict())
            error = getattr(self.client, "last_service_error", None)
            if isinstance(error, dict) and error.get("schema") == "bailian-service-error/1":
                receipt["service_error"] = {k: error[k] for k in (
                    "schema", "reason_code", "request_stage", "exception_type", "http_status",
                    "elapsed_seconds", "timeout_s") if k in error}
            if self.trajectory:
                self.trajectory.emit("model_service_error", decision_id=snapshot.decision_id,
                                     receipt=receipt)
            raise OperationalError("model_service_error") from None
        finally:
            self.client.last_reasoning_content = None
        delivery.mark_delivered()
        receipt["delivery"] = delivery.to_dict()
        try:
            usage["usage"] = numeric_usage(getattr(self.client, "last_usage", None))
        except ContractError:
            receipt["status"] = "usage_error"
            raise
        receipt["usage"] = usage["usage"]
        receipt["usage_missing"] = "total_tokens" not in usage["usage"]
        if not isinstance(text, str) or len(text.encode()) > 65536:
            receipt["status"] = "format_error"
            raise ContractError("model_output_format_error")
        receipt["response_sha256"] = sha256(text.encode())
        identity = getattr(self.client, "last_response_identity", {}) or {}
        if isinstance(identity, dict) and isinstance(identity.get("model"), str):
            receipt["reported_model_sha256"] = sha256(identity["model"].encode())
            expected_model = ("offline-fake" if getattr(self.client, "offline_fake", False)
                              else MODEL_CONFIG["model"])
            if identity["model"] != expected_model:
                receipt["status"] = "service_deviation"
                raise OperationalError("model_service_deviation")
        try:
            proposal = self._parse(text, snapshot)
        except ContractError:
            receipt["status"] = "format_error"
            raise
        receipt["status"] = "valid_program" if proposal.source is not None else "valid_plan"
        if proposal.source is not None:
            receipt["program_sha256"] = sha256(proposal.source.encode())
            receipt["program_guard_codes"] = program_guard_codes(proposal.source)
        if self.trajectory:
            self.trajectory.emit("model_proposal", receipt=receipt, program_source=proposal.source,
                                 actions=proposal.actions)
        return proposal
