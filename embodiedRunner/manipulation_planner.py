"""Pinned official manipulation prompts/messages with a deterministic strict parser."""
from __future__ import annotations

import ast
import json

from manipulation_protocol import source, vector
from protocol import ContractError, sha256


def adapter(n_shot=4):
    name = "embodiedbench/planner/manip_planner.py"
    original = next(x for x in ast.parse(source(name)).body
                    if isinstance(x, ast.ClassDef) and x.name == "ManipPlanner")
    selected = {"__init__", "process_prompt", "get_message", "reset", "update_info"}
    methods = [x for x in original.body if isinstance(x, ast.FunctionDef) and x.name in selected]
    constructor = next(x for x in methods if x.name == "__init__")
    provider = [x for x in constructor.body if isinstance(x, ast.If)
                and ast.unparse(x.test) == "model_type == 'custom'"]
    if len(provider) != 1 or {x.name for x in methods} != selected:
        raise ContractError("official_manipulation_constructor_changed")
    constructor.body.remove(provider[0])
    module = ast.fix_missing_locations(ast.Module(body=[
        ast.ClassDef(name="ManipPlanner", bases=[], keywords=[], body=methods, decorator_list=[])],
        type_ignores=[]))
    system_name = "embodiedbench/evaluator/config/system_prompts.py"
    system = next(ast.literal_eval(x.value) for x in ast.parse(source(system_name)).body
                  if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name)
                    and t.id == "eb_manipulation_system_prompt" for t in x.targets))
    example_name = "embodiedbench/evaluator/config/eb_manipulation_example.py"
    # This official module contains only example strings and dictionaries.
    examples_namespace = {}
    exec(compile(ast.parse(source(example_name)), "verified-manipulation-examples", "exec"),
         examples_namespace)
    examples = examples_namespace["vlm_examples_baseline"]
    from manipulation_protocol import TASKS
    examples = {task: examples[task.split("_")[0]] for tasks in TASKS.values() for task in tasks}
    namespace = {"VOXEL_SIZE": 100, "ROTATION_RESOLUTION": 3}
    exec(compile(module, "verified-official-manipulation-planner", "exec"), namespace)
    result = namespace["ManipPlanner"](
        "qwen3.8-flash", "remote", system, examples, n_shot=n_shot,
        chat_history=False, language_only=False, multiview=False, multistep=False,
        visual_icl=False, kwargs={})
    result.reset()
    result.namespace = namespace
    template_name = "embodiedbench/planner/planner_utils.py"
    result.output_template = next(ast.literal_eval(x.value) for x in ast.parse(source(template_name)).body
                                  if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name)
                                    and t.id == "template_manip" for t in x.targets))
    result.prompt_hashes = {n: sha256(source(n)) for n in (name, system_name, example_name, template_name)}
    return result


def parse_plan(text, max_actions=15):
    """Accept official list/string variants; never synthesize random actions."""
    try:
        value = json.loads(text)
        plan = value["executable_plan"] if "executable_plan" in value else value["properties"]["executable_plan"]
        if isinstance(plan, str):
            plan = ast.literal_eval(plan)
        if not isinstance(plan, (tuple, list)) or len(plan) > max_actions:
            raise ValueError
        actions = []
        for item in plan:
            if isinstance(item, dict):
                if set(item) != {"action"}:
                    raise ValueError
                item = item["action"]
            if isinstance(item, str):
                item = ast.literal_eval(item)
            actions.append(vector(item))
        return tuple(actions)
    except (KeyError, TypeError, ValueError, SyntaxError, RecursionError, ContractError):
        raise ContractError("invalid_action_plan") from None
