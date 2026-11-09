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
from task_card import (VERSION as CARD_VERSION, MAX_CHARS as CARD_MAX_CHARS, build_card,
                       validate_goal, validate_transition, validate_perception, validate_progress)

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
TASK_CARD_GUIDANCE = (
    "Task-card decision protocol embodied-task-card/1. Return a complete JSON object with subgoal, "
    "transition, and exactly ONE of program_source, perception_request, catalog_request. "
    "subgoal has exactly id,goal,object,destination,complete_when. Keep id stable while working on "
    "the same goal. goal<=160 chars; other strings<=80. object/destination may be empty. "
    "For keep, id/object/destination/complete_when stay stable; goal wording may be paraphrased. "
    "destination names the CURRENT subgoal waypoint or placement target; the ultimate destination "
    "remains in task. Use revise to change the target or completion condition. "
    "complete_when is holding,released,navigation_ack,action_ack,or visual. "
    "transition has kind (keep,advance,revise),source (none,official_feedback,visual),"
    "observation_revision (current integer),detail (<=160 chars). "
    "Initialize with keep/none. An advance evaluates the PREVIOUS subgoal, then selects a new one. "
    "Do not call a grasp complete merely because gripper motion succeeded. "
    "A successful official find of the matching object can complete a find/locate visual subgoal. "
    "A named navigation ACK must match the current waypoint; MoveAhead or Rotate ACKs only confirm motion. "
    "Visual conclusions are model assessments, not official facts. "
    "The host synchronizes the accepted goal into immutable ctx.task_state before running your program "
    "and into the next task card. Read snapshot identity from ctx; never hard-code a card snapshot hash. "
    "Preserve the ultimate object and destination from task; empty detection never proves absence. "
    "perception_request={\"targets\":[{\"role\":\"object\",\"category\":\"star\"},"
    "{\"role\":\"destination\",\"category\":\"silver container\"}],\"depth\":false}. "
    "Use 0..4 distinct lowercase English noun phrases<=64 chars; role is object,destination,support. "
    "At least one target or depth=true; at most one perception request per decision. "
    "Request only the evidence required for this subgoal; otherwise directly generate a program. "
    "No perception has been run when evidence.status is not_requested. "
    "catalog_request={\"offset\":0} requests a complete next action page; this costs a Qwen turn. "
    "The card contains whole action records with original IDs; omitted entries remain accessible "
    "as full strings in ctx.action_table, and manipulation objects in ctx.official_objects. "
    "You may search/enumerate those immutable runtime tables in your program. "
    "view_candidates returns result['data']['candidates'] (a list), result['data']['count'], "
    "and result['data']['coverage']; each candidate has candidate_id,category,box_xyxy "
    "and category_confirmed. An empty candidates list is legal. Do not index result['data'] as a list. "
    "Always obey confirmed holding facts and destination control preconditions. "
    "Never mutate ctx.task_state or ctx.memory. Use the subgoal output instead of writing narrative "
    "memory_update. On repair keep the same subgoal unless current evidence warrants revision. "
    "Read remaining limits and loop.switch_strategy. When set, choose a different legal action or waypoint; "
    "the host blocks loop.cycle_next_action and repeated stationary actions. "
    "No automatic service/action retry."
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
    def __init__(self, client, *, arm, environment, method=None, trajectory=None, task_cards=False):
        if arm not in {"Direct", "B01", "B11", "Full"} or bool(method) != (arm in {"B11", "Full"}):
            raise ContractError("solver_arm_method_mismatch")
        if method is not None and method.environment != environment:
            raise ContractError("solver_method_environment")
        if (getattr(client, "model", MODEL_CONFIG["model"]) != MODEL_CONFIG["model"]
                or getattr(client, "timeout_s", 120) != 120):
            raise ContractError("solver_client_policy_mismatch")
        self.client, self.arm, self.environment, self.method = client, arm, environment, method
        self.task_cards = bool(task_cards) and arm != "Direct"
        self.contract = ActionExecutionContract(environment=environment, task_cards=self.task_cards)
        self.receipts, self.adapter, self.previous_observation = [], None, None
        self.prompt_hashes = {}
        self.direct_feedback = []
        self.previous_step, self.pending_action = 0, None
        self.trajectory = trajectory
        self.card_budget, self.card_evidence, self.catalog_offset = None, None, None
        self.last_card = None

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
            if self.task_cards:
                from task_context import render, examples
                system = render(self.environment, self.contract, snapshot.observation.observation_revision)
                if plan.entries:
                    system += "\nComplete frozen environment Skill:\n" + plan.entries[0].text
                public_examples = examples(self.environment, snapshot.observation.subset)
                self.prompt_hashes = {"selected_examples": public_hash(public_examples)}
                self.last_card, user_text = build_card(
                    snapshot, self.card_budget, repair=feedback, evidence=self.card_evidence,
                    catalog_offset=self.catalog_offset, fixed_chars=len(system),
                    examples=public_examples)
            else:
                user_text = canonical({"snapshot": snapshot.public_dict(), "repair": feedback}).decode()
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
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
        self.last_format_feedback = None
        field, value = "json", None
        try:
            value = json_object(text)
            if self.arm != "Direct":
                goal = transition = perception = catalog = progress = None
                if self.task_cards:
                    field = "output_fields"
                    outputs = set(value) & {"program_source", "perception_request", "catalog_request"}
                    if len(outputs) != 1 or set(value) - {"progress"} != {"subgoal", "transition"} | outputs:
                        raise ContractError("model_output_format_error")
                    if "progress" in value:
                        field = "progress"
                        progress = validate_progress(value["progress"])
                    field = "subgoal"
                    goal = validate_goal(value["subgoal"])
                    field = "transition"
                    transition = validate_transition(
                        value["transition"], snapshot.observation.observation_revision)
                    if "perception_request" in outputs:
                        field = "perception_request"
                        perception = validate_perception(value["perception_request"])
                        return Proposal(subgoal=goal, transition=transition, perception_request=perception,
                                        progress=progress)
                    if "catalog_request" in outputs:
                        field = "catalog_request"
                        catalog = value["catalog_request"]
                        if (type(catalog) is not dict or set(catalog) != {"offset"}
                                or type(catalog["offset"]) is not int
                                or not 0 <= catalog["offset"] < max(1, len(snapshot.observation.action_table))):
                            raise ContractError("model_output_format_error")
                        return Proposal(subgoal=goal, transition=transition,
                                        catalog_request=freeze(catalog), progress=progress)
                    value = {"program_source": value["program_source"]}
                field = "program_source"
                if (set(value) != {"program_source"} or type(value["program_source"]) is not str
                        or not value["program_source"].strip()):
                    raise ContractError("model_output_format_error")
                field = "program_assembly"
                program, recovered = assemble_program_ex(
                    value["program_source"],
                    skill_semver_used=[f"{self.method.skill_id}@{self.method.version}"] if self.method else [],
                    intended_answer_slot="SubmitActions")
                if recovered or program.program_source.strip() != value["program_source"].strip():
                    raise ContractError("model_output_format_error")
                return Proposal(source=program.program_source, subgoal=goal, transition=transition,
                                progress=progress)
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
            if self.task_cards:
                reason = "invalid_" + field
                hints = {
                    "json": "Return one valid JSON object.",
                    "output_fields": "Return subgoal, transition, optional progress, and exactly one output.",
                    "progress": "Use completed, remaining, next strings, each at most 240 characters.",
                    "subgoal": "Use id, goal, object, destination, complete_when; obey the stated lengths and enums.",
                    "transition": "Use kind, source, integer observation_revision, detail<=160. "
                        "Advance requires source official_feedback or visual; use keep/none for a pending goal.",
                    "perception_request": "Use {targets:[{role,category}],depth:false}. "
                        "Depth is a boolean at the request level, outside targets; "
                        "target rows contain only role and lowercase category.",
                    "catalog_request": "Use only an integer offset within the action table. "
                        "The full action catalog is already supplied; prefer a program.",
                    "program_source": "Return a nonempty Python program_source string defining solve(ctx).",
                    "program_assembly": "Define solve(ctx) using injected globals; do not add wrappers or imports.",
                }
                row = value.get("transition") if isinstance(value, dict) else None
                if field == "transition" and isinstance(row, dict):
                    if row.get("kind") == "advance" and row.get("source") == "none":
                        reason = "advance_source_none"
                    elif (type(row.get("observation_revision")) is not int or
                          row.get("observation_revision") != snapshot.observation.observation_revision):
                        reason = "observation_revision_type_or_value"
                self.last_format_feedback = {
                    "format_field": field, "format_reason": reason, "corrective_action": hints[field]}
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
        if self.task_cards:
            card_text = messages[1]["content"][0]["text"]
            receipt.update(
                context_protocol=CARD_VERSION, task_card_chars=len(card_text),
                task_card_max_chars=CARD_MAX_CHARS, task_card_sha256=sha256(card_text.encode()),
                task_state_sha256=public_hash(snapshot.task_state), task_card_complete=True,
                task_card_over_budget=self.last_card["packing"]["required_fields_over_budget"])
            receipt.update(prompt_text_target=CARD_MAX_CHARS,
                           prompt_text_over_target=chars > CARD_MAX_CHARS,
                           catalog_complete=self.last_card["catalog"]["omitted"] == 0,
                           history_records=len(self.last_card["history"]),
                           official_examples=len(self.last_card["official_examples"]))
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
        receipt["status"] = ("valid_perception_request" if proposal.perception_request is not None else
                             "valid_catalog_request" if proposal.catalog_request is not None else
                             "valid_program" if proposal.source is not None else "valid_plan")
        if proposal.source is not None:
            receipt["program_sha256"] = sha256(proposal.source.encode())
            receipt["program_guard_codes"] = program_guard_codes(proposal.source)
        if self.trajectory:
            self.trajectory.emit("model_proposal", receipt=receipt, program_source=proposal.source,
                                 actions=proposal.actions, subgoal=proposal.subgoal,
                                 transition=proposal.transition, perception_request=proposal.perception_request,
                                 catalog_request=proposal.catalog_request, progress=proposal.progress)
            self.trajectory.emit("progress_assessment", decision_id=snapshot.decision_id,
                                 progress=proposal.progress, source="model_assessment")
        return proposal
