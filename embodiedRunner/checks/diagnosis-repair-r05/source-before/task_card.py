"""Complete, bounded decision cards and public action-derived subgoal state."""
from __future__ import annotations

from collections.abc import Mapping
import re

from action_contract import freeze, plain, public_hash
from protocol import ContractError, canonical

VERSION = "embodied-task-card/2"
MAX_CHARS = 16000
MAX_HISTORY = 40
MAX_LOOP_PERIOD = 12
POLICY = {
    "schema": VERSION, "max_chars": MAX_CHARS, "length_unit": "unicode_codepoints",
    "packing": "whole_records_and_fields", "critical_overflow": "preserve_and_send",
    "history_records": MAX_HISTORY, "perception_requests_per_decision": 1,
    "max_perception_targets": 4,
    "state_sync": "accepted_goal_before_execution_and_next_request",
    "result_rebind": "task_state_only",
    "keep_identity": "id_object_destination_complete_when",
    "completion_policy": "matching_find_ack_and_current_waypoint",
    "navigation_loop_period_max": MAX_LOOP_PERIOD,
    "navigation_loop_min_matching_views": 3,
    "navigation_loop_guard": "block_predicted_repeat",
    "budget_scope": "all_request_text",
    "section_targets": {"skill_and_protocol": 5500, "actions": 5000,
                        "task_state_history": 4000, "examples_and_evidence": 1500},
    "catalog_policy": "all_original_ids_and_names",
    "feedback_policy": "each_executed_action_complete_public_feedback",
    "progress_policy": "same_completion_optional_model_assessment",
}
GOAL_KEYS = {"id", "goal", "object", "destination", "complete_when"}
COMPLETION_KINDS = {"holding", "released", "navigation_ack", "action_ack", "visual"}


def text(value, maximum, *, empty=False):
    if (type(value) is not str or len(value) > maximum
            or (not empty and not value.strip())):
        raise ContractError("task_card_response_format")
    return value


def validate_goal(value):
    if not isinstance(value, Mapping) or set(value) != GOAL_KEYS:
        raise ContractError("task_card_response_format")
    result = {key: text(value[key], 160 if key == "goal" else 80,
                        empty=key in {"object", "destination"}) for key in GOAL_KEYS}
    if result["complete_when"] not in COMPLETION_KINDS:
        raise ContractError("task_card_response_format")
    return freeze(result)


def validate_transition(value, revision):
    if not isinstance(value, Mapping) or set(value) != {
            "kind", "source", "observation_revision", "detail"}:
        raise ContractError("task_card_response_format")
    if (value["kind"] not in {"keep", "advance", "revise"}
            or value["source"] not in {"none", "official_feedback", "visual"}
            or type(value["observation_revision"]) is not int
            or value["observation_revision"] != revision):
        raise ContractError("task_card_response_format")
    text(value["detail"], 160, empty=value["kind"] == "keep")
    if value["kind"] == "advance" and value["source"] == "none":
        raise ContractError("task_card_response_format")
    return freeze(value)


def validate_perception(value):
    if not isinstance(value, Mapping) or set(value) != {"targets", "depth"}:
        raise ContractError("task_card_response_format")
    targets = value["targets"]
    if (type(value["depth"]) is not bool or not isinstance(targets, (list, tuple))
            or len(targets) > 4 or (not targets and not value["depth"])):
        raise ContractError("task_card_response_format")
    seen = set()
    for row in targets:
        if (not isinstance(row, Mapping) or set(row) != {"role", "category"}
                or row["role"] not in {"object", "destination", "support"}
                or type(row["category"]) is not str
                or not re.fullmatch(r"[a-z][a-z0-9 -]{0,63}", row["category"])
                or row["category"] in seen):
            raise ContractError("task_card_response_format")
        seen.add(row["category"])
    return freeze(value)


def validate_progress(value):
    if not isinstance(value, Mapping) or set(value) != {"completed", "remaining", "next"}:
        raise ContractError("task_card_response_format")
    return freeze({key: text(value[key], 240, empty=True) for key in value})


def normalize_name(name):
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def action_kind(name):
    lowered = name.lower()
    if lowered.startswith(("pick ", "pick_", "pick up ", "pickup")):
        return "pickup"
    if lowered.startswith(("put ", "put_", "place ", "place_", "drop")):
        return "release"
    if lowered.startswith(("find ", "navigate", "move to ")):
        return "navigate"
    return "other"


def action_object(name):
    return re.sub(r"^(pick up the |pick up |pickup |pick_|pick |find an? |find |"
                  r"navigate to the |navigate to |move to |place at the |place at |"
                  r"put down the |put down |drop the |drop )", "", name,
                  flags=re.I).split("(", 1)[0].strip(" .")


def target_name(name):
    return re.sub(r"^(?:the|an?)\s+", "", normalize_name(name))


def navigation_target(goal):
    """Use an explicit current waypoint; fall back to the structured target."""
    match = re.match(
        r"^(?:navigate|move|go|walk)\s+(?:to|towards?)\s+(.+?)"
        r"(?=\s+(?:to\s+(?:find|locate|identify|pick|search)\b|and\b)|[.;!?]|$)",
        goal["goal"], re.I)
    return match.group(1).strip() if match else goal["destination"] or goal["object"]


def compact_feedback(feedback):
    """Remove only recognized redundant boilerplate; never cut a sentence."""
    lines = []
    for line in feedback.splitlines():
        line = " ".join(line.split())
        if not line or line.startswith("Last official action succeeded:"):
            continue
        # EB-Manipulation repeats the full instruction and scorer value here.
        match = re.fullmatch(
            r"You are currently performing the task intended to .+? At this moment, "
            r"you have completed executing \d+ steps\. (Last action is (?:valid|invalid)\.) "
            r"The current reward obtained is [-\d.]+\.", line)
        line = match.group(1) if match else line
        if line not in lines:
            lines.append(line)
    return "\n".join(lines)


class SubgoalLedger:
    """Facts derive from public ACK/feedback; model intentions remain separate."""
    def __init__(self, environment):
        self.environment = environment
        self.goal = None
        self.facts = {"holding": {"status": "unknown", "object": "", "revision": -1},
                      "control_target": None, "last_navigation": None}
        self.history = []
        self.transitions = []
        self.last_feedback = ""
        self.last_success = None
        self.last_action = None
        self.loop = {"same_action_no_progress": 0, "switch_strategy": False,
                     "cycle_period": 0, "cycle_next_action": None}
        self.last_pixel = None
        self.navigation_trace = []
        self.progress = None

    def _navigation_action(self, name):
        return (action_kind(name) == "navigate" or
                self.environment == "eb_navigation" and
                normalize_name(name).startswith(("move ", "rotate ", "look ")))

    def _observe_loop(self, observation, row):
        name, pixel = row["action"], observation.pixel_sha256
        navigation = self._navigation_action(name)
        repeated = (self.last_action is not None and self.last_action["action"] == name
                    and self.last_pixel == pixel and (row["success"] is False or navigation))
        n = self.loop["same_action_no_progress"] + 1 if repeated else 0
        self.loop = {"same_action_no_progress": n, "switch_strategy": n >= 2,
                     "cycle_period": 0, "cycle_next_action": None}
        if not navigation:
            self.navigation_trace = [{"pixel": pixel, "action": None}]
            return
        self.navigation_trace.append({"pixel": pixel, "action": name})
        self.navigation_trace = self.navigation_trace[-(2 * MAX_LOOP_PERIOD + 1):]
        for period in range(2, MAX_LOOP_PERIOD + 1):
            trace = self.navigation_trace
            overlap = 0
            for back in range(1, min(period + 1, len(trace) - period) + 1):
                if trace[-back]["pixel"] != trace[-period - back]["pixel"]:
                    break
                if back > 1 and trace[1 - back]["action"] != trace[1 - period - back]["action"]:
                    break
                overlap += 1
            # Three matching views and both connecting actions suffice even if
            # the budget stops the second traversal before its cycle completes.
            if overlap >= 3 and len({r["pixel"] for r in trace[-overlap:]}) >= 2:
                self.loop.update(switch_strategy=True, cycle_period=period,
                                 cycle_next_action=trace[-period]["action"])
                break

    def observe(self, observation, *, action=None, action_name=None, succeeded=None,
                ack_id="", decision_id="", kind="open"):
        self.last_feedback = compact_feedback(observation.public_feedback)
        if kind != "step" or action is None:
            if self.last_pixel != observation.pixel_sha256:
                self.navigation_trace = [{"pixel": observation.pixel_sha256, "action": None}]
                self.loop = {"same_action_no_progress": 0, "switch_strategy": False,
                             "cycle_period": 0, "cycle_next_action": None}
                self.last_pixel = observation.pixel_sha256
            return
        name = (str(list(action)) if self.environment == "eb_manipulation"
                else action_name if action_name is not None else observation.action_table[action])
        operation = action_kind(name) if self.environment != "eb_manipulation" else "motion"
        revision = observation.observation_revision
        row = {"action": name, "success": succeeded, "revision": revision, "ack": ack_id,
               "decision_id": decision_id, "source": "official_action_and_feedback",
               "action_id": action if self.environment != "eb_manipulation" else None,
               "feedback": observation.public_feedback}
        self._observe_loop(observation, row)
        self.last_action, self.last_success, self.last_pixel = row, succeeded, observation.pixel_sha256
        self.history.append(row)
        self.history = self.history[-MAX_HISTORY:]
        feedback = observation.public_feedback
        holding = self.facts["holding"]
        match = re.search(r"(?:you are holding|currently holding)\s+([^\n.]+)", feedback, re.I)
        if match:
            obj = match.group(1).strip()
            holding = {"status": "empty" if obj.lower() in {"nothing", "no object"} else "holding",
                       "object": "" if obj.lower() in {"nothing", "no object"} else obj,
                       "revision": revision}
        elif "when holding something" in feedback or "Please place the object before picking" in feedback:
            holding = {**holding, "status": "holding", "revision": revision}
        elif "when not holding something" in feedback:
            holding = {"status": "empty", "object": "", "revision": revision}
        elif self.environment in {"eb_alfred", "eb_habitat"} and succeeded is True:
            if operation == "pickup":
                holding = {"status": "holding", "object": action_object(name), "revision": revision}
            elif operation == "release":
                holding = {"status": "empty", "object": "", "revision": revision}
        self.facts["holding"] = holding
        if self.environment == "eb_alfred":
            # The published connector clears its target BEFORE most interactions,
            # including failed pickup. This is control state, not hidden geometry.
            if not name.lower().startswith(("put down ", "open ")):
                self.facts["control_target"] = None
            if operation == "navigate":
                self.facts["control_target"] = {
                    "target": action_object(name), "confirmed": succeeded is True,
                    "revision": revision}
        if operation == "navigate":
            self.facts["last_navigation"] = {
                "target": action_object(name), "success": succeeded is True, "revision": revision}

    def accept_goal(self, goal, transition, revision):
        goal, transition = plain(validate_goal(goal)), plain(validate_transition(transition, revision))
        old = self.goal
        if old is not None:
            if transition["kind"] == "keep" and any(
                    goal[key] != old[key] for key in GOAL_KEYS - {"goal"}):
                raise ContractError("subgoal_keep_changed")
            if transition["kind"] == "advance":
                condition = old["complete_when"]
                if transition["source"] == "official_feedback":
                    facts = self.facts
                    okay = {
                        "holding": facts["holding"]["status"] == "holding" and
                            (not old["object"] or normalize_name(old["object"]) ==
                             normalize_name(facts["holding"]["object"])),
                        "released": facts["holding"]["status"] == "empty" and self.last_success is True
                            and self.last_action is not None
                            and action_kind(self.last_action["action"]) == "release",
                        "navigation_ack": self._matching_navigation(navigation_target(old)),
                        "action_ack": self.last_success is True,
                        "visual": bool(old["object"]) and bool(re.match(
                            r"^(?:find|locate|identify|search for)\b", old["goal"], re.I))
                            and self._matching_navigation(old["object"]),
                    }[condition]
                    if not okay:
                        raise ContractError("subgoal_completion_not_supported")
                elif ((condition == "holding" and self.facts["holding"]["status"] == "empty")
                      or (condition == "released" and self.facts["holding"]["status"] == "holding")):
                    raise ContractError("subgoal_completion_not_supported")
                # Visual assessment is explicitly a model belief, never a host
                # verified fact or an official success result.
        self.goal = goal
        self.transitions.append({"revision": revision, "previous_goal_sha256": public_hash(old),
                                 "goal_sha256": public_hash(goal),
                                 "kind": transition["kind"], "source": transition["source"],
                                 "detail": transition["detail"]})
        self.transitions = self.transitions[-MAX_HISTORY:]

    def _matching_navigation(self, target):
        navigation = self.facts["last_navigation"]
        return (bool(target) and navigation is not None and navigation["success"]
                and self.last_success is True and self.last_action is not None
                and action_kind(self.last_action["action"]) == "navigate"
                and navigation["revision"] == self.last_action["revision"]
                and target_name(navigation["target"]) == target_name(target))

    def repair_feedback(self, code):
        hints = {
            "subgoal_keep_changed": "Keep id, object, destination and complete_when stable; "
                "goal wording may change. Use revise for a changed target or condition.",
            "subgoal_completion_not_supported": "The previous completion condition lacks matching "
                "public evidence. Keep it while collecting evidence; a visual advance needs image support. "
                "A successful find can locate its matching object, but cannot prove holding.",
            "navigation_loop_action_blocked": "Choose a different legal action or waypoint from the "
                "current observation; this action would continue the observed navigation loop.",
        }
        feedback = {"code": code}
        if code in hints:
            feedback["corrective_action"] = hints[code]
        if code == "subgoal_completion_not_supported" and self.goal:
            feedback["expected_condition"] = self.goal["complete_when"]
            feedback["expected_navigation_target"] = navigation_target(self.goal)
        if code == "navigation_loop_action_blocked":
            feedback["loop"] = plain(self.loop)
        return feedback

    def check_action(self, action, observation):
        if self.environment == "eb_manipulation":
            return None
        name = observation.action_table[action]
        operation = action_kind(name)
        if self.loop["switch_strategy"] and (
                name == self.loop["cycle_next_action"] or
                self.loop["same_action_no_progress"] >= 2 and
                self.last_action is not None and name == self.last_action["action"]):
            return "navigation_loop_action_blocked"
        if operation == "pickup" and self.facts["holding"]["status"] == "holding":
            return "already_holding_before_pickup"
        if operation == "release" and self.facts["holding"]["status"] == "empty":
            return "not_holding_before_release"
        if (self.environment == "eb_alfred" and operation == "release"
                and name.lower().startswith("put down ")
                and self.goal and self.goal["destination"]
                and not self.facts["control_target"]):
            return "find_destination_before_put"
        if operation == "release" and self.goal and self.goal["destination"]:
            if name.lower().startswith(("place at ", "place_", "put on ", "put in ")):
                wanted = normalize_name(self.goal["destination"])
                if wanted and wanted not in normalize_name(name):
                    return "release_destination_mismatch"
            if self.environment == "eb_alfred" and name.lower().startswith("put down "):
                target = self.facts["control_target"]
                if target and normalize_name(self.goal["destination"]) != normalize_name(target["target"]):
                    return "release_destination_mismatch"
        return None

    def public_state(self):
        return {"goal": self.goal, "facts": self.facts, "last_action": self.last_action,
                "last_success": self.last_success, "feedback": self.last_feedback,
                "history": self.history, "loop": self.loop,
                "last_transition": self.transitions[-1] if self.transitions else None,
                "progress": self.progress}

    def accept_progress(self, value, revision):
        if value is not None:
            self.progress = {"source": "model_assessment", "revision": revision,
                             **plain(validate_progress(value))}


def remaining_budget(budget, decision_id=""):
    value = {key: max(0, budget.policy[key] - budget.counts[key])
             for key in ("steps", "qwen", "invalid", "reviews", "dino", "sam", "moge")}
    decision = budget.decisions.get(decision_id, {})
    value["decision_requests"] = max(0, budget.policy["synthesis_per_decision"] -
                                     decision.get("synthesis", 0))
    value["wall_seconds"] = max(0, int(budget.policy["wall_s"] -
                                    (budget.run.clock() - budget.started)))
    return value


def build_legacy_card(snapshot, budget=None, *, repair=None, evidence=None, catalog_offset=None):
    """Pack complete JSON records under the cap, without text slicing."""
    VERSION, MAX_CHARS = "embodied-task-card/1", 4000
    obs = snapshot.observation
    state = plain(snapshot.task_state) if snapshot.task_state else {}
    goal = state.get("goal")
    card = {
        "schema": VERSION,
        "identity": {"episode_id": obs.episode_id, "decision_id": snapshot.decision_id,
                     "observation_revision": obs.observation_revision,
                     "snapshot_sha256": snapshot.snapshot_sha256,
                     "action_table_sha256": obs.action_table_sha256},
        "task": obs.instruction, "environment": obs.environment,
        "subgoal": goal, "facts": state.get("facts", {}),
        "latest": {"action": state.get("last_action"), "success": state.get("last_success"),
                   "feedback": state.get("feedback", compact_feedback(obs.public_feedback))},
        "loop": state.get("loop", {}),
        "remaining": remaining_budget(budget, snapshot.decision_id) if budget else {},
        "evidence": evidence or {"status": "not_requested", "capabilities": list(snapshot.capabilities)},
        "repair": {k: plain(v) for k, v in (repair or {}).items()
                   if k not in {"observations", "yielded_result_ids"}}, "history": [],
        "catalog": {"total": len(obs.action_table), "entries": [], "omitted": len(obs.action_table),
                    "next_offset": None, "complete_runtime_table": "ctx.action_table"},
        "objects": {"total": len(obs.manipulation_input.get("objects", {})), "entries": {},
                    "omitted": len(obs.manipulation_input.get("objects", {})),
                    "complete_runtime_table": "ctx.official_objects"},
        "packing": {"limit": MAX_CHARS, "unit": "unicode_codepoints",
                    "omitted_history": len(state.get("history", [])), "complete": True,
                    "required_fields_over_budget": False},
    }
    def fits():
        return len(canonical(card).decode("utf-8")) <= MAX_CHARS
    # The target limits optional context, never delivery of required fields.
    # Flag genuine required-field overflow and deliver the complete card.
    card["packing"]["required_fields_over_budget"] = not fits()
    if repair and ("observations" in repair or "yielded_result_ids" in repair):
        card["repair"].update(observations=[], yielded_result_ids=[],
                              omitted_observations=len(repair.get("observations", [])),
                              omitted_result_ids=len(repair.get("yielded_result_ids", [])))
        card["packing"]["required_fields_over_budget"] = (
            card["packing"]["required_fields_over_budget"] or not fits())
        for name, omitted_key in (("yielded_result_ids", "omitted_result_ids"),
                                  ("observations", "omitted_observations")):
            for item in repair.get(name, []):
                card["repair"][name].append(plain(item))
                card["repair"][omitted_key] -= 1
                if not fits():
                    card["repair"][name].pop()
                    card["repair"][omitted_key] += 1
                    break
    history = state.get("history", [])
    for row in reversed(history):
        if row == state.get("last_action"):
            continue
        card["history"].insert(0, row)
        card["packing"]["omitted_history"] = len(history) - len(card["history"])
        if not fits():
            card["history"].pop(0)
            card["packing"]["omitted_history"] = len(history) - len(card["history"])
            break
    names = obs.action_table
    if catalog_offset is not None:
        if type(catalog_offset) is not int or not 0 <= catalog_offset < max(1, len(names)):
            raise ContractError("task_card_catalog_offset")
        indices = list(range(catalog_offset, len(names)))
    else:
        phrases = [obs.instruction]
        if goal:
            phrases.extend(goal[k] for k in ("goal", "object", "destination"))
        words = set(normalize_name(" ".join(phrases)).split())
        indices = sorted(range(len(names)), key=lambda i: (
            -len(words & set(normalize_name(names[i]).split())), i))
    # Reclaim whole optional history records when needed for the first action.
    for index in indices:
        row = {"id": index, "name": names[index]}
        card["catalog"]["entries"].append(row)
        card["catalog"]["omitted"] = len(names) - len(card["catalog"]["entries"])
        card["catalog"]["next_offset"] = index + 1 if index + 1 < len(names) else None
        while not fits() and card["history"]:
            card["history"].pop(0)
            card["packing"]["omitted_history"] = len(history) - len(card["history"])
        if not fits():
            card["catalog"]["entries"].pop()
            card["catalog"]["omitted"] = len(names) - len(card["catalog"]["entries"])
            card["catalog"]["next_offset"] = index
            break
        if len(card["catalog"]["entries"]) == 16:
            card["catalog"]["next_offset"] = (
                index + 1 if catalog_offset is not None else 0) if card["catalog"]["omitted"] else None
            break
    objects = obs.manipulation_input.get("objects", {})
    for name, xyz in objects.items():
        card["objects"]["entries"][name] = list(xyz)
        card["objects"]["omitted"] = len(objects) - len(card["objects"]["entries"])
        if not fits():
            del card["objects"]["entries"][name]
            card["objects"]["omitted"] = len(objects) - len(card["objects"]["entries"])
            break
    encoded = canonical(card).decode("utf-8")
    validate_legacy_card(encoded)
    return card, encoded


def validate_legacy_card(encoded):
    VERSION, MAX_CHARS = "embodied-task-card/1", 4000
    from protocol import json_object
    if type(encoded) is not str:
        raise ContractError("task_card_context_cap")
    card = json_object(encoded)
    if (card.get("schema") != VERSION or not card.get("packing", {}).get("complete")
            or card["packing"]["limit"] != MAX_CHARS
            or not {"task", "subgoal", "facts", "latest", "remaining", "catalog",
                    "objects", "evidence", "repair", "identity", "history", "loop"} <= set(card)):
        raise ContractError("task_card_incomplete")
    if len(encoded) > MAX_CHARS and card["packing"].get("required_fields_over_budget") is not True:
        raise ContractError("task_card_context_cap")
    return card


def build_card(snapshot, budget=None, *, repair=None, evidence=None, catalog_offset=None,
               fixed_chars=0, examples=()):
    """Preserve task, full catalog and forty feedback records; lend section space."""
    obs = snapshot.observation
    state = plain(snapshot.task_state) if snapshot.task_state else {}
    if type(fixed_chars) is not int or fixed_chars < 0:
        raise ContractError("task_card_context_cap")
    if catalog_offset is not None and (
            type(catalog_offset) is not int or not 0 <= catalog_offset < max(1, len(obs.action_table))):
        raise ContractError("task_card_catalog_offset")
    history = state.get("history", [])
    retained = history[-MAX_HISTORY:]
    # ACK and decision hashes remain in the immutable runtime state/trajectory.
    # The planner sees the original action, outcome and complete public feedback.
    def action_row(row):
        return {k: row[k] for k in ("revision", "action_id", "action", "success", "feedback")
                if k in row}
    card = {
        "schema": VERSION,
        "identity": {"episode_id": obs.episode_id, "decision_id": snapshot.decision_id,
                     "observation_revision": obs.observation_revision,
                     "snapshot_sha256": snapshot.snapshot_sha256,
                     "action_table_sha256": obs.action_table_sha256},
        "task": obs.instruction, "environment": obs.environment,
        "subgoal": state.get("goal"), "facts": state.get("facts", {}),
        "latest": {"action": action_row(state["last_action"]) if state.get("last_action") else None,
                   "success": state.get("last_success"),
                   "feedback": state.get("feedback", compact_feedback(obs.public_feedback))},
        "previous_progress": state.get("progress"),
        "loop": state.get("loop", {}),
        "remaining": remaining_budget(budget, snapshot.decision_id) if budget else {},
        "history": [action_row(row) for row in retained],
        "catalog": {"total": len(obs.action_table), "omitted": 0, "next_offset": None,
                    "format": "ID:original action name, one record per line",
                    "text": "\n".join(f"{i}:{name}" for i, name in enumerate(obs.action_table)),
                    "complete_runtime_table": "ctx.action_table"},
        "objects": {"total": len(obs.manipulation_input.get("objects", {})),
                    "entries": plain(obs.manipulation_input.get("objects", {})), "omitted": 0,
                    "complete_runtime_table": "ctx.official_objects"},
        "repair": {k: plain(v) for k, v in (repair or {}).items()
                   if k not in {"observations", "yielded_result_ids"}},
        "evidence": evidence or {"status": "not_requested", "capabilities": list(snapshot.capabilities)},
        "official_examples": [],
        "packing": {"limit": MAX_CHARS, "scope": "all_request_text",
                    "fixed_chars": fixed_chars, "unit": "unicode_codepoints",
                    "omitted_history": max(0, len(history) - MAX_HISTORY), "complete": True,
                    "required_fields_over_budget": False},
    }
    def fits():
        return fixed_chars + len(canonical(card).decode()) <= MAX_CHARS
    card["packing"]["required_fields_over_budget"] = not fits()
    # Whole evidence pages first, then deterministic public examples. Mandatory
    # information is never evicted to accommodate these optional sections.
    if repair and ("observations" in repair or "yielded_result_ids" in repair):
        card["repair"].update(observations=[], yielded_result_ids=[],
                              omitted_observations=len(repair.get("observations", [])),
                              omitted_result_ids=len(repair.get("yielded_result_ids", [])))
        card["packing"]["required_fields_over_budget"] |= not fits()
        for key, omitted in (("yielded_result_ids", "omitted_result_ids"),
                             ("observations", "omitted_observations")):
            for row in repair.get(key, []):
                card["repair"][key].append(plain(row))
                card["repair"][omitted] -= 1
                if not fits():
                    card["repair"][key].pop()
                    card["repair"][omitted] += 1
                    break
    for row in examples:
        card["official_examples"].append(plain(row))
        if not fits():
            card["official_examples"].pop()
            break
    wire = canonical(card).decode()
    validate_card(wire)
    return card, wire


def validate_card(encoded):
    from protocol import json_object
    if type(encoded) is not str:
        raise ContractError("task_card_context_cap")
    card = json_object(encoded)
    if card.get("schema") == "embodied-task-card/1":
        return validate_legacy_card(encoded)
    packing = card.get("packing", {})
    if (card.get("schema") != VERSION or packing.get("complete") is not True
            or packing.get("limit") != MAX_CHARS or packing.get("scope") != "all_request_text"
            or type(packing.get("fixed_chars")) is not int or packing["fixed_chars"] < 0
            or not {"task", "subgoal", "facts", "latest", "remaining", "catalog",
                    "objects", "evidence", "repair", "identity", "history", "loop",
                    "official_examples", "previous_progress"} <= set(card)
            or card["catalog"].get("omitted") != 0 or card["objects"].get("omitted") != 0
            or len(card["history"]) > MAX_HISTORY):
        raise ContractError("task_card_incomplete")
    if packing["fixed_chars"] + len(encoded) > MAX_CHARS and (
            packing.get("required_fields_over_budget") is not True):
        raise ContractError("task_card_context_cap")
    return card
