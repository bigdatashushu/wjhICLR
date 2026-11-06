"""Engineering fixtures only; these tests are not benchmark measurements."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge import SimulatorBridge, simulator_environment
from planner import NavigationPlanner, PlannerConfig
from protocol import (
    ACTIONS, ROOT, ContractError, Observation, OperationalError, Task,
    Transition, canonical, decode_transition, parse_plan, select_tasks,
    sha256, within_root,
)
from runner import evaluate
from simulator_worker import OfficialNavigation


@pytest.fixture(autouse=True)
def offline_test_environment(monkeypatch):
    import os
    import socket

    for name in tuple(os.environ):
        if any(part in name.upper() for part in
               ("TOKEN", "API_KEY", "PASSWORD", "SECRET", "CREDENTIAL", "ENDPOINT")) or name.upper().endswith(("_KEY", "_AUTH")):
            monkeypatch.delenv(name, raising=False)

    def denied_connection(*_args, **_kwargs):
        raise OSError("Network disabled in engineering tests")

    monkeypatch.setattr(socket.socket, "connect", denied_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", denied_connection)


def image_bytes():
    stream = io.BytesIO()
    Image.fromarray(np.arange(12, dtype=np.uint8).reshape(2, 2, 3)).save(stream, format="PNG")
    return stream.getvalue()


def observation(step=0):
    png = image_bytes()
    return Observation(png, sha256(png), step)


def plan_text(actions):
    return json.dumps({
        "visual_state_description": "test observation",
        "reasoning_and_reflection": "test reasoning",
        "language_plan": "test plan",
        "executable_plan": [{"action_id": action, "action_name":
                            ACTIONS[action] if type(action) is int and 0 <= action < 8 else "invalid"}
                            for action in actions],
    })


class Client:
    last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def __init__(self, outputs):
        self.outputs, self.calls = iter(outputs), []

    def chat(self, messages, **parameters):
        self.calls.append((messages, parameters))
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value


class Environment:
    def __init__(self, transitions):
        self.transitions, self.actions, self.opens = iter(transitions), [], []

    def open(self, task):
        self.opens.append(task.task_id)
        return Transition(observation(), "Navigate to a test object", False, False, True, "")

    def step(self, action, *, first_in_plan):
        self.actions.append(action)
        step, done, success, valid = next(self.transitions)
        return Transition(observation(step), "Navigate to a test object",
                          done, success, valid, "action executed" if valid else "collision")


def planner(outputs, **kwargs):
    return NavigationPlanner(Client(outputs), PlannerConfig(mode="direct", **kwargs))


@pytest.mark.parametrize("number", [True, -1, 8, 1.0, "1", None])
def test_invalid_actions_are_rejected(number):
    with pytest.raises(ContractError):
        parse_plan(plan_text([number]), max_actions=2)


def test_plan_rejects_duplicates_wrong_names_and_excess_actions():
    value = json.loads(plan_text([0]))
    value["executable_plan"][0]["action_name"] = ACTIONS[1]
    with pytest.raises(ContractError):
        parse_plan(json.dumps(value), max_actions=2)
    with pytest.raises(ContractError):
        parse_plan(plan_text([0, 1, 2]), max_actions=2)
    with pytest.raises(ContractError):
        parse_plan('{"executable_plan":[],"executable_plan":[]}', max_actions=2)
    with pytest.raises(ContractError):
        parse_plan(plan_text([]), max_actions=2)


def test_feedback_and_scores_do_not_enter_model_messages():
    actor = planner([plan_text([0])])
    actor.update_info(action=0, feedback="collision", last_action_success=False, step=1)
    actor.act(observation(1), "Navigate to the laptop")
    messages, parameters = actor.client.calls[0]
    text = json.loads(messages[1]["content"][0]["text"])
    assert set(text) == {"instruction", "environment_step", "remaining_environment_steps", "action_history"}
    assert set(text["action_history"][0]) == {
        "action_id", "action_name", "step", "env_feedback", "last_action_success"}
    assert "collision" in canonical(messages).decode()
    assert parameters == {
        "max_tokens": 2048, "seed": 137, "temperature": 0.0,
        "response_format": {"type": "json_object"}, "enable_thinking": False}
    assert "test reasoning" not in json.dumps(actor.last_receipt)
    assert "Navigate to the laptop" not in json.dumps(actor.last_receipt)


def test_collision_discards_pending_actions_and_replans(tmp_path):
    actor = planner([plan_text([0, 1]), plan_text([4])])
    env = Environment([(1, False, False, False), (2, True, True, True)])
    tasks = [Task("base", 0, "a" * 64)]
    summary = evaluate(tasks, actor, env, tmp_path, max_requests=3)
    assert env.actions == [0, 4]
    assert summary["successful_tasks"] == 1
    result = json.loads((tmp_path / "episodes/base-000.json").read_text())
    assert result["invalid_actions"] == 1
    assert len(actor.client.calls) == 2
    assert "collision" in canonical(actor.client.calls[1][0]).decode()
    assert "test reasoning" not in (tmp_path / "trace.jsonl").read_text()


def test_service_error_stops_without_retry_and_preserves_denominator(tmp_path):
    actor = planner([RuntimeError("private credential and provider error")])
    actor.client.last_service_error = {
        "reason_code": "http_error", "http_status": 429,
        "body": "private credential and provider error",
    }
    env = Environment([])
    tasks = [Task("base", index, "b" * 64) for index in range(3)]
    summary = evaluate(tasks, actor, env, tmp_path, max_requests=10)
    assert summary["status"] == "incomplete"
    assert summary["planned_tasks"] == 3
    assert summary["statuses"] == {"not_run": 2, "operational_error": 1}
    assert len(env.opens) == len(actor.client.calls) == 1
    assert actor.last_receipt["http_status"] == 429
    assert actor.last_receipt["service_reason_code"] == "http_error"
    assert "private credential" not in "".join(path.read_text() for path in tmp_path.rglob("*.json*"))
    assert len(list((tmp_path / "episodes").glob("*.json"))) == 3


def test_global_budget_stops_before_next_model_call(tmp_path):
    actor = planner([plan_text([0]), plan_text([1])])
    env = Environment([(1, False, False, True)])
    tasks = [Task("base", index, "c" * 64) for index in range(2)]
    summary = evaluate(tasks, actor, env, tmp_path, max_requests=1)
    assert summary["model_requests"] == 1
    assert summary["stop_reason"] == "run_request_budget_exhausted"
    assert summary["statuses"] == {"not_run": 1, "operational_error": 1}
    assert env.actions == [0]


def test_bad_model_plan_is_failure_with_no_random_action(tmp_path):
    actor = planner([plan_text([8]), plan_text([0])])
    env = Environment([(1, True, False, True)])
    tasks = [Task("base", index, "d" * 64) for index in range(2)]
    summary = evaluate(tasks, actor, env, tmp_path, max_requests=3)
    assert summary["status"] == "complete"
    assert summary["statuses"] == {"failure": 1, "planner_error": 1}
    assert env.actions == [0]
    assert summary["task_success_rate"] == 0


def test_environment_step_and_instruction_identity_are_checked(tmp_path):
    actor = planner([plan_text([0])])
    env = Environment([(2, False, False, True)])
    summary = evaluate([Task("base", 0, "e" * 64)], actor, env, tmp_path, max_requests=1)
    assert summary["stop_reason"] == "transition_identity_mismatch"


def test_episode_budget_and_skill_are_frozen(tmp_path):
    skill = tmp_path / "skill.md"
    skill.write_text("Test frozen Skill", encoding="utf-8")
    actor = NavigationPlanner(Client([plan_text([0])]),
                              PlannerConfig(mode="skill", max_requests_per_episode=1),
                              skill_path=skill)
    skill.write_text("Changed after load", encoding="utf-8")
    actor.act(observation(), "test goal")
    assert "Test frozen Skill" in actor.client.calls[0][0][0]["content"]
    with pytest.raises(ContractError, match="episode_request_budget"):
        actor.act(observation(1), "test goal")
    assert len(actor.client.calls) == 1


def test_rgb_packets_reject_privileged_fields_and_changed_pixels():
    png = image_bytes()
    packet = {
        "image_base64": base64.b64encode(png).decode(), "image_sha256": sha256(png),
        "step": 0, "instruction": "test", "done": False, "task_success": False,
        "last_action_success": True, "env_feedback": "",
    }
    assert decode_transition(packet, resolution=2).observation.png == png
    with pytest.raises(ContractError):
        decode_transition({**packet, "target_position": [1, 2, 3]}, resolution=2)
    with pytest.raises(ContractError):
        decode_transition({**packet, "image_sha256": "0" * 64}, resolution=2)


def test_real_navigation_input_selection_and_output_boundary():
    tasks = select_tasks(ROOT / "upstream", ["base", "visual_appearance"], 3)
    assert len(tasks) == 6
    assert all(set(task.public_dict()) == {"task_id", "subset", "index", "input_sha256"} for task in tasks)
    with pytest.raises(ContractError):
        within_root("../escaped-output")
    assert within_root("runs/test").is_relative_to(ROOT)


def test_simulator_environment_strips_inherited_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "private-test-key")
    monkeypatch.setenv("CUSTOM_PROXY_TOKEN", "private-test-token")
    monkeypatch.setenv("MAKERS_MODELS_KEY", "private-test-makers-key")
    monkeypatch.setenv("HARNESS3D_DETECTOR_ENDPOINT", "private-test-endpoint")
    monkeypatch.setenv("PYTHONPATH", "/other/source")
    environment = simulator_environment(tmp_path)
    assert all(name not in environment for name in
               ("DASHSCOPE_API_KEY", "CUSTOM_PROXY_TOKEN", "MAKERS_MODELS_KEY",
                "HARNESS3D_DETECTOR_ENDPOINT", "PYTHONPATH"))
    assert environment["TMPDIR"].startswith(str(tmp_path))
    assert "HOME" not in environment or environment["HOME"] == __import__("os").environ["HOME"]


def test_worker_uses_official_actions_and_hides_scorer(tmp_path):
    class FakeOfficialEnv:
        language_skill_set, _max_episode_steps, feedback_verbosity = ACTIONS, 20, 0

        def __init__(self, **_kwargs):
            self._current_step, self.calls = 0, []
            self.episode_language_instruction = "test goal"

        def reset(self):
            return {"head_rgb": np.zeros((2, 2, 3), dtype=np.uint8)}

        def step(self, action, reasoning, flag):
            self.calls.append((action, reasoning, flag))
            self._current_step += 1
            return self.reset(), 1, True, {
                "task_success": 1, "last_action_success": True,
                "env_feedback": "done action", "distance": 0.1,
                "target_position": [100, 200, 300],
            }

        def close(self):
            pass

    task = select_tasks(ROOT / "upstream", ["base"], 1)[0]
    session = OfficialNavigation(ROOT / "upstream", tmp_path, resolution=2, factory=FakeOfficialEnv)
    reset = session.open(task.subset, task.index, task.input_sha256)
    assert reset["step"] == 0
    result = session.step(0, True)
    assert "target_position" not in result and "distance" not in result
    assert session.env.calls == [(0, {}, 1)]
    with pytest.raises(ContractError):
        session.step(True, True)
    session.close()


def test_subprocess_transport_and_response_identity(tmp_path):
    script = tmp_path / "fixture_worker.py"
    script.write_text(
        "import json,sys\n"
        "for line in sys.stdin:\n"
        " c=json.loads(line)\n"
        " print(json.dumps({'id':c['id'],'version':'embodiedrunner-navigation/1',"
        "'ok':True,'data':{'echo':c['op']}}),flush=True)\n",
        encoding="utf-8",
    )
    bridge = SimulatorBridge([sys.executable, "-B", str(script)], work_dir=tmp_path, timeout_s=2)
    try:
        assert bridge.request("probe") == {"echo": "probe"}
    finally:
        bridge.close()
    assert bridge.process.poll() is not None


def test_real_official_nav_step_keeps_action_and_success_semantics(monkeypatch):
    """Use actual official step/mapper/scorer code with an in-memory controller."""
    import importlib.util
    import types

    ai2thor = types.ModuleType("ai2thor")
    ai2thor.controller = types.ModuleType("ai2thor.controller")
    thor_platform = types.ModuleType("ai2thor.platform")
    thor_platform.CloudRendering = object
    gym = types.ModuleType("gym")
    gym.Env = object
    utils = types.ModuleType("embodiedbench.envs.eb_navigation.utils")
    utils.draw_target_box = utils.draw_boxes = lambda *_args: None
    main_module = types.ModuleType("embodiedbench.main")
    main_module.logger = types.SimpleNamespace(info=lambda *_args: None)
    for name, module in {
        "ai2thor": ai2thor, "ai2thor.controller": ai2thor.controller,
        "ai2thor.platform": thor_platform, "gym": gym,
        "embodiedbench.envs.eb_navigation.utils": utils,
        "embodiedbench.main": main_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    source = ROOT / "upstream/embodiedbench/envs/eb_navigation/EBNavEnv.py"
    spec = importlib.util.spec_from_file_location("official_nav_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = object.__new__(module.EBNavigationEnv)
    metadata = {
        "agent": {"position": {"x": 0.0, "z": 0.0}},
        "lastActionSuccess": True, "lastAction": "", "errorMessage": "",
    }
    event = types.SimpleNamespace(metadata=metadata, frame=np.zeros((2, 2, 3), dtype=np.uint8))
    calls = []

    def controller_step(**parameters):
        calls.append(parameters)
        metadata["lastAction"] = parameters["action"]
        metadata["agent"]["position"]["z"] += parameters["moveMagnitude"]
        return event

    env.env = types.SimpleNamespace(last_event=event, step=controller_step)
    env._reset, env._current_step, env._max_episode_steps = True, 0, 20
    env._episode_start_time = __import__("time").time()
    env.episode_data = {"targetObjectIds": ["fixture-target"],
                        "target_position": {"x": 0.0, "z": 0.25}}
    env.episode_language_instruction = "fixture goal"
    env.language_skill_set, env.episode_log = ACTIONS, []
    env.feedback_verbosity, env.standing = 0, True
    env.save_episode_log_per_step = lambda _flag: None
    _obs, reward, done, info = env.step(0, {}, 1)
    assert calls == [{"action": "MoveAhead", "moveMagnitude": 0.25}]
    assert reward == 1 and done and info["task_success"] == 1
    assert info["env_step"] == 1 and info["last_action_success"] is True
    assert "position" not in info["env_feedback"]


def test_subprocess_rejects_wrong_response_id(tmp_path):
    script = tmp_path / "fixture_wrong_id.py"
    script.write_text(
        "import json,sys\n"
        "for line in sys.stdin:\n"
        " print(json.dumps({'id':999,'version':'embodiedrunner-navigation/1',"
        "'ok':True,'data':{}}),flush=True)\n", encoding="utf-8",
    )
    bridge = SimulatorBridge([sys.executable, "-B", str(script)], work_dir=tmp_path, timeout_s=2)
    try:
        with pytest.raises(OperationalError, match="response_identity_mismatch"):
            bridge.request("probe")
    finally:
        bridge.close()


def test_subprocess_timeout_is_bounded_and_process_is_closed(tmp_path):
    script = tmp_path / "fixture_timeout.py"
    script.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    bridge = SimulatorBridge([sys.executable, "-B", str(script)], work_dir=tmp_path, timeout_s=0.05)
    try:
        with pytest.raises(OperationalError, match="simulator_timeout"):
            bridge.request("probe")
    finally:
        bridge.close()
    assert bridge.process.poll() is not None
