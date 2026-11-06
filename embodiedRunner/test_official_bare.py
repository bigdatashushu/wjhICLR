"""Protocol fidelity tests; no live model or simulator calls."""
import ast
import base64
import json

from official_bare import (BareConfig, OfficialBarePlanner, literal,
                           official_tree, _png_url)
from protocol import ACTIONS, canonical
from runner import evaluate
from test_navigation import (
    Client, Environment, Task, observation, offline_test_environment, plan_text,
)


def complete_official_planner():
    """Independent reference: retain the entire official constructor and class."""
    tree, _ = official_tree("embodiedbench/planner/nav_planner.py")
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "EBNavigationPlanner")
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    template, _ = literal("embodiedbench/planner/planner_utils.py", "template")
    template_lang, _ = literal("embodiedbench/planner/planner_utils.py", "template_lang")
    system, _ = literal("embodiedbench/evaluator/config/system_prompts.py", "eb_navigation_system_prompt")
    examples, _ = literal("embodiedbench/evaluator/config/eb_navigation_example.py", "examples")
    namespace = {
        "template": template, "template_lang": template_lang, "MESSAGE_WINDOW_LEN": 5,
        "RemoteModel": lambda *_args, **_kwargs: None,
        "local_image_to_data_url": _png_url,
    }
    exec(compile(module, "complete-official-reference", "exec"), namespace)
    return namespace[cls.name](
        model_name="qwen3.8-flash", model_type="remote", actions=list(ACTIONS),
        system_prompt=system, examples=examples, n_shot=3,
        chat_history=False, language_only=False, multiview=False, multistep=False,
        visual_icl=False, truncate=True, kwargs={},
    )


def test_current_and_following_messages_equal_complete_official_class():
    actor = OfficialBarePlanner(Client([]))
    reference = complete_official_planner()
    for step, feedback in [(0, []), (1, [[4, "Last action RotateRight executed successfully."]])]:
        actor.feedback = feedback
        obs = observation(step)
        prompt = reference.process_prompt("Navigate to the laptop.", feedback)
        expected = reference.get_message(obs.png, prompt)
        actual = actor.messages(obs, "Navigate to the laptop.")
        assert canonical(actual) == canonical(expected)
        assert len(actual) == 1 and actual[0]["role"] == "user"
        assert [item["type"] for item in actual[0]["content"]] == ["image_url", "text"]
        assert base64.b64decode(actual[0]["content"][0]["image_url"]["url"].split(",")[1]) == obs.png
        text = actual[0]["content"][1]["text"]
        assert text.count("## Task Execution Example") == 3
        assert "reference_answer" not in text and "Reference navigation Skill" not in text
        assert ("1-2 actions" if step == 0 else "5-6 actions") in text
        assert ("The action history:" in text) == (step > 0)


def test_no_skill_tools_added_and_official_id_only_execution():
    value = json.loads(plan_text([0] * 6))
    for item in value["executable_plan"]:
        item["action_name"] = "MoveAhead"
    actor = OfficialBarePlanner(Client([json.dumps(value)]))
    result = actor.act(observation(0), "Navigate to the laptop")
    assert result.actions == (0,) * 6
    assert actor.skill == "" and actor.skill_sha256 is None
    messages, parameters = actor.client.calls[0]
    assert parameters == {
        "max_tokens": 4096, "seed": 137, "temperature": 0.0,
        "response_format": {"type": "json_object"}, "enable_thinking": False,
    }
    assert all(message["role"] != "system" for message in messages)
    assert actor.last_receipt["model_tool_calls"] == 0
    assert "test reasoning" not in json.dumps(actor.last_receipt)


def test_malformed_plan_does_not_trigger_random_fallback_or_retry(tmp_path):
    actor = OfficialBarePlanner(Client(["not JSON"]))
    env = Environment([])
    summary = evaluate([Task("base", 0, "a" * 64)], actor, env, tmp_path, max_requests=60)
    assert summary["statuses"] == {"planner_error": 1}
    assert summary["model_requests"] == 1 and env.actions == []
    assert summary["uses_official_prompt"] and summary["uses_official_messages"]
    assert summary["automatic_retry"] is False


def test_zero_example_option_retains_official_prompt():
    actor = OfficialBarePlanner(Client([]), BareConfig(n_shots=0))
    text = actor.messages(observation(), "Navigate to the laptop")[0]["content"][1]["text"]
    assert "*** Strategy ***" in text
    assert "## Task Execution Example" not in text
