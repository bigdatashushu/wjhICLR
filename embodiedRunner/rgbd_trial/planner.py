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
            return a.get_message(list(urls), general, task_prompt)
        prompt = a.process_prompt(public["instruction"], a.episode_act_feedback)
        if kind in ("alfred", "habitat") and any(
                x in a.model_name for x in ("claude", "InternVL", "Qwen2-VL", "Qwen2.5-VL", "Qwen3-VL")):
            prompt += self.namespace["template"]
        messages = a.get_message("current-0.png", prompt)
        if kind == "navigation":
            content = messages[0]["content"]
            content.insert(1, {"type": "text", "text":
                "Image 1 is the current RGB view. Image 2 is its synchronized "
                "500x500 simulated depth view, with the same pixel coordinates. "
                "The depth visualization uses a fixed scale: white is near 0 meters, "
                "black is 10 meters or farther; invalid depth pixels are also black. "
                "Use RGB to recognize objects and depth to assess visible surfaces "
                "and obstacles. Dark pixels alone do not prove free space."})
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
