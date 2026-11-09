"""Compact perception context and one-response visual review plus native plan."""
import json

from target_detection_r05.planner import OfficialPlanner, TargetPlanner as ParentPlanner
from .perception import PlannerError

MAX_SHOTS = 3

TARGET_TOOL_PROMPT = """
Output exactly one JSON object. There are two exclusive branches: a perception
tool request, or the official full action plan with a visual review.

On the FIRST response, extract 1-4 targets from the public task instruction and
request perceive_targets in the same JSON:
{"targets":[{"id":"T1","role":"object","category":"star","source_phrase":"star",
 "quantity":null,"attributes":[],"relation":"","aliases":["star shaped block"]}],
 "tool_request":{"name":"perceive_targets","target_ids":["T1"]}}
Each target has exactly those eight fields. IDs are T1-T4; role is
object/destination/support. source_phrase is verbatim from the instruction.
category is short English category wording; separate color, quantity and relation.
Include task-relevant destinations/supports, at most 3 category aliases per target.
Targets stay fixed for this episode. Never extract categories from private metadata
or anonymous coordinates.

perceive_targets detects each requested category and supplies a candidate overlay
and at most 2 unannotated RGB crops. Later observations are prepared automatically
using the frozen targets, before your next request. No tool moves the environment.
Use original RGB/depth when candidates are absent, irrelevant or uncertain.
Only request extra tools when the supplied view is insufficient:
{"tool_request":{"name":"inspect_candidates","group_ids":["G003"]}}
{"tool_request":{"name":"view_candidates","page":1}}
{"tool_request":{"name":"perceive_targets","target_ids":["T1"],
                 "query_aliases":{"T1":"star shaped block"}}}
Only frozen categories/aliases may be queried. Page indices start at 0; 12 groups
per page. All proposals are retained. G IDs are geometric proposal groups, not
official action IDs, official object numbers, object counts or verified semantics.
Qwen supplies no calibrated confidence. Labels sit outside the RGB; follow boxes.
Anonymous coordinate anchors are possible associations only. For Manipulation,
compare with the original official numbered RGB and use official coordinates.
Old G IDs expire when the observation changes. Only the currently supplied crops
can be reviewed or selected; requesting new crops replaces the prior crop set.

With the current crops, review and plan in the SAME response:
{"visual_state_description":"Brief visible state.",
 "reasoning_and_reflection":"Brief decision rationale using state and action feedback.",
 "language_plan":"Brief next actions.",
 "perception_review":[{"group_id":"G001","verdict":"match","reason":"Visible evidence."}],
 "selected_targets":[{"target_id":"T1","group_id":"G001"}],
 "executable_plan":[...]}
Put the three non-empty description strings before executable_plan. Give a concise
decision rationale, not extended private reasoning. Review only supplied current
crops with match/not_match/uncertain. Select only a reviewed match for its queried
target. If no groups exist, use [] for both review fields. Rejected/uncertain crops
can support a search plan. No separate review-only round is required.
Keep the full official executable_plan and legal native actions. Never mix
tool_request with executable_plan. Every additional tool response consumes a
planning request from the same episode budget.
"""

ALFRED_HAND_PROMPT = """
ALFRED: You can only hold ONE object at a time; to combine objects use put-in
actions from the official action list. Before picking another object, place the
held object in/on the intended receptacle or put it down as the task requires.
For placing an object inside a movable container: pick the inner object, find the
container, put the object into the container, then pick the container and place it
at the destination. Do not pick the container while still holding the inner object.
Use successful action feedback to track what is held; a failed pickup or placement
does not change the held-object state. Address the last failure before repeating it.
In reasoning_and_reflection briefly state the held object (or empty/unknown), the
next action's precondition, and how the plan respects the one-object constraint.
"""


class TargetPlanner(ParentPlanner):
    def __init__(self, sources, task, model_name, n_shots, actions, response_format=None):
        super().__init__(sources, task, model_name, min(n_shots, MAX_SHOTS),
                         actions, response_format)

    def messages(self, public, perception):
        messages = OfficialPlanner.messages(self, public)
        if perception.targets is None:
            stage = ("CURRENT PHASE: EXTRACT TARGETS AND PREPARE. Return targets plus a "
                     "perceive_targets tool_request; no executable_plan yet.")
        elif not perception.detected:
            stage = ("CURRENT PHASE: PREPARE CURRENT VIEW. Request perceive_targets "
                     "with the frozen target IDs before planning.")
        else:
            stage = ("CURRENT PHASE: REVIEW AND PLAN. Return the current crop review, "
                     "the three description strings, and the full official executable_plan "
                     "in one JSON. Use an extra tool only if needed.")
            if self.task.environment == "manipulation":
                stage += ' Native format: {"executable_plan":[{"action":[x,y,z,rx,ry,rz,gripper]}]}.'
            else:
                stage += ' Native format: {"executable_plan":[{"action_id":0,"action_name":"..."}]}.'
        protocol = TARGET_TOOL_PROMPT + "\n" + stage
        if self.task.environment == "alfred":
            protocol += "\n" + ALFRED_HAND_PROMPT
        messages.insert(0, {"role": "system", "content": protocol})
        current = [m for m in messages if m["role"] == "user"][-1]["content"]
        if self.task.environment == "manipulation":
            current.extend([
                {"type": "text", "text": "Additional image: same-observation unannotated RGB "
                 "used for detection. The first RGB retains official coordinate/number annotations."},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + public["raw_rgb"]["base64"]}}])
        current.append({"type": "text", "text": json.dumps(perception.summary(), ensure_ascii=False)})
        for image in perception.context_images():
            current.extend([
                {"type": "text", "text": image["caption"]},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image["base64"]}}])
        return messages

    def decision(self, text, perception):
        try:
            value = json.loads(text)
            if not isinstance(value, dict):
                raise PlannerError("decision_requires_json_object")
            if perception.targets is None and (
                    "targets" not in value or not isinstance(value.get("tool_request"), dict)
                    or value["tool_request"].get("name") != "perceive_targets"):
                raise PlannerError("first_response_requires_targets_and_perceive")
            if "targets" in value:
                perception.set_targets(value["targets"])
            if "tool_request" in value:
                if not set(value) <= {"targets", "tool_request", "reasoning"}:
                    raise PlannerError("tool_and_action_branches_must_be_separate")
                perception.tool(value["tool_request"])
                return None
            holder = value if "executable_plan" in value else value.get("properties", {})
            for field in ("visual_state_description", "reasoning_and_reflection", "language_plan"):
                if not isinstance(holder.get(field), str) or not holder[field].strip():
                    raise PlannerError("action_requires_" + field)
            if not {"perception_review", "selected_targets"} <= set(value):
                raise PlannerError("action_requires_current_visual_review")
            perception.approve_action(value)
            return OfficialPlanner.parse(self, self.action_text(value))
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
            if isinstance(exc, PlannerError):
                raise
            raise PlannerError("invalid_model_target_or_tool_json") from None
