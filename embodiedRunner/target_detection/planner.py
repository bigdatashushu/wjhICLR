"""Pinned official prompts and plan semantics, independent of old action hosts."""
import ast
import contextlib
import io
import json
from dataclasses import dataclass

from .protocol import vector
from .sources import ContractError


@dataclass
class Plan:
    status: str
    actions: list
    proposed_count: int = 0


class OfficialPlanner:
    def __init__(self, sources, task, model_name, n_shots, actions, response_format=None):
        self.sources, self.task = sources, task
        self.response_format = response_format
        kind = task.environment
        namespace = {"json": json, "print": lambda *args, **kwargs: None,
                     "VOXEL_SIZE": 100, "ROTATION_RESOLUTION": 3,
                     "MESSAGE_WINDOW_LEN": 5}
        # Manipulation's small snapshot omits planner_utils/schema: use the
        # same-revision complete snapshot, verified by its own manifest.
        utility_kind = "navigation" if kind == "manipulation" else kind
        utility_path = "embodiedbench/planner/planner_utils.py"
        for name in ("template", "template_lang", "template_manip", "template_lang_manip"):
            namespace[name] = sources.literal(utility_kind, utility_path, name)
        cls = sources.planner_class(kind, namespace)
        system_name = {"navigation": "eb_navigation_system_prompt",
                       "alfred": "alfred_system_prompt", "habitat": "habitat_system_prompt",
                       "manipulation": "eb_manipulation_system_prompt"}[kind]
        system = sources.literal(kind, "embodiedbench/evaluator/config/system_prompts.py", system_name)
        common = dict(model_name=model_name, model_type="remote", system_prompt=system,
                      n_shot=n_shots, chat_history=False, language_only=False, kwargs={})
        config_dir = "embodiedbench/evaluator/config/"
        if kind == "navigation":
            examples = sources.literal(kind, config_dir + "eb_navigation_example.py", "examples")
            self.adapter = cls(actions=actions, examples=examples, multiview=False,
                               multistep=False, visual_icl=False, truncate=True, **common)
        elif kind in ("alfred", "habitat"):
            filename = ("alfred_long_horizon_examples.json" if task.subset == "long_horizon"
                        else "alfred_examples.json") if kind == "alfred" else "habitat_examples.json"
            examples = json.loads(sources.read(kind, config_dir + filename))
            self.adapter = cls(actions=actions, examples=examples, use_feedback=True,
                               multistep=False, **common)
        else:
            examples_namespace = {}
            exec(compile(sources.read(kind, config_dir + "eb_manipulation_example.py"),
                         "verified-manipulation-examples", "exec"), examples_namespace)
            # Match evaluator.load_demonstration for detection_box=1.
            baseline = examples_namespace["vlm_examples_baseline"]
            names = ("pick_cube_shape", "pick_cube_relative", "place_into_shape_sorter_color",
                     "place_into_shape_sorter_relative", "stack_cubes_color",
                     "stack_cubes_relative", "wipe_table_direction", "wipe_table_relative")
            examples = {name: baseline[name.split("_")[0]] for name in names}
            self.adapter = cls(examples=examples, multiview=False, multistep=False,
                               visual_icl=False, **common)
        self.adapter.reset()
        self.namespace = namespace
        schema_file = "generation_guide_manip.py" if kind == "manipulation" else "generation_guide.py"
        schema_name = "vlm_generation_guide_manip" if kind == "manipulation" else "vlm_generation_guide"
        self.schema = sources.literal(utility_kind,
                                      "embodiedbench/planner/planner_config/" + schema_file, schema_name)

    def messages(self, public):
        """The caller supplies ONLY validated public observation fields."""
        urls = {f"current-{i}.png": "data:image/png;base64," + image["base64"]
                for i, image in enumerate(public["images"])}
        self.namespace["local_image_to_data_url"] = lambda image_path: urls[image_path]
        a, kind = self.adapter, self.task.environment
        if kind == "manipulation":
            m = public["manipulation"]
            general, task_prompt = a.process_prompt(public["instruction"], str(m["objects"]),
                                                   m["task_variation"], a.episode_act_feedback)
            if (self.response_format == "json_object" or any(
                    x in a.model_name for x in ("claude", "InternVL", "Qwen2-VL", "Qwen2.5-VL"))):
                task_prompt += "\n\n" + self.namespace["template_manip"]
            messages = a.get_message(list(urls), general, task_prompt)
        else:
            prompt = a.process_prompt(public["instruction"], a.episode_act_feedback)
            if kind in ("alfred", "habitat") and any(
                    x in a.model_name for x in ("claude", "InternVL", "Qwen2-VL", "Qwen2.5-VL", "Qwen3-VL")):
                prompt += self.namespace["template"]
            messages = a.get_message("current-0.png", prompt)
        current = [message for message in messages if message["role"] == "user"][-1]
        content = current["content"]
        content.append({"type": "text", "text":
            "Image 1 is the current RGB view. Image 2 is its synchronized "
            "500x500 simulated depth view, with the same pixel coordinates. "
            "The depth visualization uses a fixed scale: white is near 0 meters, "
            "black is 10 meters or farther; invalid depth pixels are also black. "
            "Use RGB to recognize objects and depth to assess visible surfaces. "
            "Dark pixels alone do not prove free space. Depth is camera-axis distance "
            "in meters; it does not directly specify a world-frame action coordinate."})
        content.append({"type": "text", "text": "Image 2: synchronized current depth."})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + public["depth"]["image"]["base64"]}})
        return messages

    def parse(self, text):
        a, kind = self.adapter, self.task.environment
        a.planner_steps += 1
        try:
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError
            items = value.get("executable_plan", value.get("properties", {}).get("executable_plan"))
            if kind == "manipulation":
                if isinstance(items, str):
                    items = ast.literal_eval(items)
                if not isinstance(items, (list, tuple)):
                    raise ValueError
                actions = []
                for item in items:
                    if isinstance(item, dict):
                        item = item["action"]
                    # Upstream also accepts a singleton nested action dictionary.
                    if isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], dict):
                        item = item[0]["action"]
                    if isinstance(item, str):
                        item = ast.literal_eval(item)
                    actions.append(vector(item))
                return Plan("valid" if actions else "empty", actions, len(items))
            if not isinstance(items, list) or any(type(x["action_id"]) is not int for x in items):
                raise ValueError
            if kind in ("alfred", "habitat"):
                parsed = a.json_to_action(text)
                if parsed == -2:
                    return Plan("empty", [])
                if parsed == -1:
                    return Plan("invalid", [], len(items))
                return Plan("valid", parsed, len(items))
            actions = [x["action_id"] for x in items]
            if any(not 0 <= x < len(a.actions) for x in actions):
                raise ValueError
            return Plan("valid" if actions else "empty", actions, len(items))
        except (ValueError, TypeError, KeyError, SyntaxError, AttributeError, ContractError):
            a.output_json_error += 1
            return Plan("invalid", [])

    def feedback(self, action, public):
        key = "action" if self.task.environment == "manipulation" else "action_id"
        self.adapter.update_info({key: action, "env_feedback": public["env_feedback"]})


TARGET_TOOL_PROMPT = """
This trial adds question-directed perception before the official full action plan.
Output exactly one JSON object. The tool branch below temporarily supersedes the
official executable_plan format. Tool calls use JSON, not provider-native function calls.

On your FIRST response extract 1-4 targets from the public task instruction, then
request detect_targets in the SAME JSON. Do not extract categories from hidden task
metadata or anonymous object coordinates. Each target has exactly:
{"id":"T1","role":"object","category":"star","source_phrase":"star",
 "quantity":null,"attributes":[],"relation":"","aliases":["star shaped block"]}
role is object/destination/support. category is a short English object category.
Keep color, quantity and relation separate from category. source_phrase must be a
verbatim phrase from the instruction. Include destination/support when task-relevant.
aliases are optional alternative category wording, at most 3; no scene-wide catalog.
These root targets stay fixed for the episode. Later responses may omit targets.

Tool JSON examples:
{"targets":[...],"tool_request":{"name":"detect_targets","target_ids":["T1","T2"]}}
{"tool_request":{"name":"detect_targets","target_ids":["T1"],
                 "query_aliases":{"T1":"star shaped block"}}}
{"tool_request":{"name":"inspect_candidates","group_ids":["G001","G003"]}}
{"tool_request":{"name":"view_candidates","page":1}}
{"tool_request":{"name":"review_candidates",
 "perception_review":[{"group_id":"G001","verdict":"match","reason":"visible star-shaped object"}],
 "selected_targets":[{"target_id":"T1","group_id":"G001"}]}}
Only extracted targets/aliases may be queried. Every category is detected separately.
Page indices start at 0. All proposals are retained; pages display 12 groups at a time.
G numbers are proposal-group IDs, NOT official object numbers and NOT object counts.
Nearly coincident boxes are grouped geometrically. Empty/mismatched labels and high
confidence can still be false detections. Check the original RGB and local crops.
Labels are in a separate sidebar with colored leaders; follow the box itself.
Possible anonymous objects are current-image projections of official anonymous
coordinate anchors. They are possible associations, not category or surface truth.
Compare these anchors with the ORIGINAL official numbered image before choosing
Manipulation coordinates. Tool G numbers never replace official object numbers.

Before each action decision, request detect_targets on the current observation.
If any group is found, inspect at least one relevant group (max 2 per call), then
explicitly judge the returned UNANNOTATED crops. A crop request alone is not a match.
If boxes are absent or irrelevant, use original RGB/depth to navigate/search safely.
All tool rounds share the same observation and do not move the environment.
After an environment observation changes, old G IDs expire; request detection again.
Every tool round consumes one model request from the fixed episode budget.

After receiving crops, call review_candidates to record the visual judgment:
"perception_review":[{"group_id":"G001","verdict":"match","reason":"visual evidence..."}],
"selected_targets":[{"target_id":"T1","group_id":"G001"}].
verdict is match/not_match/uncertain and must refer to a group whose crop you have
inspected in the CURRENT observation. Select only an inspected match for that target.
Use [] for both when no groups are found. Rejected/uncertain crops may justify search.
Once review_candidates is accepted, output the ORIGINAL official JSON action plan.
The accepted review is retained for this observation; action JSON need not repeat it.
Keep the full executable_plan and official actions; never mix tool_request with actions.
Perform all geometric/action reasoning yourself with the original RGB/depth.
"""


class TargetPlanner(OfficialPlanner):
    def messages(self, public, perception):
        messages = super().messages(public)
        # The pinned evaluator uses user-message demonstrations. Put the trial's
        # response protocol in a real system message so examples cannot override it.
        if perception.targets is None:
            stage = ("CURRENT PHASE: EXTRACT TARGETS AND DETECT. Your next response MUST contain "
                     "'targets' and 'tool_request' with name 'detect_targets'. "
                     "Do not output executable_plan in this phase.")
        elif not perception.detected:
            stage = ("CURRENT PHASE: DETECT THE CURRENT VIEW. Your next response MUST contain "
                     "'tool_request' with name 'detect_targets' using the frozen target IDs. "
                     "Do not output executable_plan in this phase.")
        elif perception.groups and not perception.inspected:
            stage = ("CURRENT PHASE: INSPECT CROPS. Your next response MUST contain 'tool_request' "
                     "with name 'inspect_candidates' and 1-2 relevant CURRENT group IDs "
                     "(or view_candidates to see other pages). Do not output executable_plan yet.")
        elif perception.groups and perception.completed_review is None:
            stage = ("CURRENT PHASE: REVIEW CROPS. Your next JSON MUST be a tool_request with "
                     "name 'review_candidates', a non-empty 'perception_review' list and "
                     "'selected_targets' list. Judge only CURRENT inspected crops. "
                     "Each review has group_id, verdict (match/not_match/uncertain), and reason. "
                     "Each selection has target_id and group_id; select only a match. "
                     "Do not output executable_plan in this phase. You may instead inspect more candidates.")
        else:
            stage = ("CURRENT PHASE: FULL ACTION PLAN. The visual review has been recorded (or no "
                     "candidates were found). Return the ORIGINAL official JSON executable_plan. "
                     "You do not need to repeat the stored review fields.")
        messages.insert(0, {"role": "system", "content": TARGET_TOOL_PROMPT + "\n\n" + stage})
        current = [m for m in messages if m["role"] == "user"][-1]
        current["content"].append({"type": "text", "text": TARGET_TOOL_PROMPT})
        if self.task.environment == "manipulation":
            current["content"].extend([
                {"type": "text", "text": "Additional image: same-observation UNANNOTATED RGB used for detection "
                 "and crops. The first RGB image retains the official coordinate/number annotations."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                                   public["raw_rgb"]["base64"]}}])
        current["content"].append({"type": "text", "text": json.dumps({
            "current_observation_serial": perception.serial,
            "frozen_targets": perception.targets,
            "current_group_ids": [g["id"] for g in perception.groups]}, ensure_ascii=False)})
        return messages + perception.turns

    def decision(self, text, perception):
        from .perception import PlannerError
        try:
            value = json.loads(text)
            if not isinstance(value, dict):
                raise PlannerError("decision_requires_json_object")
            if "targets" in value:
                perception.set_targets(value["targets"])
            if perception.targets is None:
                raise PlannerError("first_response_requires_targets")
            if "tool_request" in value:
                if not set(value) <= {"targets", "tool_request", "reasoning"}:
                    raise PlannerError("tool_and_action_branches_must_be_separate")
                if "executable_plan" in value:
                    raise PlannerError("tool_and_action_branches_must_be_separate")
                result, images = perception.tool(value["tool_request"])
                content = [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
                for image in images:
                    content.extend([{"type": "text", "text": image["caption"]},
                                    {"type": "image_url", "image_url": {
                                        "url": "data:image/png;base64," + image["base64"]}}])
                perception.turns.extend([{"role": "assistant", "content": text},
                                         {"role": "user", "content": content}])
                return None
            perception.approve_action(value)
            return super().parse(text)
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
            if isinstance(exc, PlannerError):
                raise
            raise PlannerError("invalid_model_target_or_tool_json") from None
