"""Offline contracts for the isolated trial. No API, SSH or simulator calls."""
import base64
import copy
import io
import json
import socket

from PIL import Image
import pytest

from manipulation_minimal.client import ModelClient
from manipulation_minimal.config import load_config
from manipulation_minimal.planner import OfficialPlanner
from manipulation_minimal.protocol import COORD_SOURCE, Task
from manipulation_minimal.runner import episode
from manipulation_minimal.sources import ROOT, OperationalError, Sources, canonical, digest
from manipulation_minimal.transport import child_environment, command


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("live network forbidden")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def config():
    return load_config(ROOT / "manipulation_minimal/experiment-20261107-r01.json")[0]


@pytest.fixture
def sources():
    return Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")


TASK = Task("manipulation", "base", 0)
ACTIONS = [[50, 40, 35, 0, 0, 0, 1], [50, 40, 30, 0, 0, 0, 0],
           [50, 40, 40, 0, 0, 0, 0]]


def packet(step=0, valid=True, done=False):
    stream = io.BytesIO()
    Image.new("RGB", (500, 500), (20 + step, 60, 100)).save(stream, format="PNG")
    png = stream.getvalue()
    public = {
        "instruction": "Lift the blue cube.", "images": [{
            "view": "front_rgb", "base64": base64.b64encode(png).decode(),
            "sha256": digest(png), "source": "offline fixture"}],
        "actions": [], "step": step, "env_feedback": "" if not step else "Motion valid.",
        "last_action_success": valid, "manipulation": {
            "objects": {f"object {1 + step}": [50, 40, 30 + step]},
            "task_variation": "pick_cube_shape", "source": COORD_SOURCE}}
    return {"task": TASK.as_dict(), "public": public, "official": {
        "done": done, "task_success": 0, "invalid_actions": int(not valid), "reward": 0},
        "input_sha256": digest(canonical({"task": TASK.as_dict(), "instruction": public["instruction"]}))}


def output(actions):
    return json.dumps({"visual_state_description": "offline state",
                       "reasoning_and_reflection": "grasp unverified",
                       "language_plan": "next action",
                       "executable_plan": [{"action": x} for x in actions]})


class Environment:
    def __init__(self, fail_first=False):
        self.step, self.calls, self.closed = 0, [], False
        self.fail_first = fail_first

    def request(self, op, **parameters):
        self.calls.append((op, copy.deepcopy(parameters)))
        if op == "step":
            self.step += 1
        return packet(self.step, valid=not (self.fail_first and self.step == 1),
                      done=self.step == 15)

    def close(self):
        self.closed = True


class Client:
    def __init__(self, config, responses):
        self.builder = ModelClient(config["model"])
        self.responses, self.calls, self.payloads = iter(responses), 0, []

    def payload(self, messages, schema):
        return self.builder.payload(messages, schema)

    def complete(self, payload):
        self.calls += 1
        self.payloads.append(copy.deepcopy(payload))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return {"text": value, "message": {"content": value},
                "usage": {"total_tokens": 8}, "model": "offline-fixture",
                "response_sha256": digest(value.encode()), "finish_reason": "stop"}


def run(tmp_path, config, sources, responses, fail_first=False):
    env = Environment(fail_first)
    client = Client(config, responses)
    result = episode(TASK, config, sources, client, lambda *args: env, tmp_path / "episode")
    return result, env, client


def test_one_action_then_observe_even_after_invalid_motion(tmp_path, config, sources):
    result, env, client = run(tmp_path, config, sources,
                             [output(ACTIONS), output([])], fail_first=True)
    sent = [p["action"] for op, p in env.calls if op == "step"]
    assert sent == ACTIONS[:1]  # No stale tail may cross a failed motion.
    assert [op for op, _ in env.calls] == ["open", "step", "observe"]
    assert client.calls == 2 and result["status"] == "empty_plan" and env.closed


def test_fifteen_actions_require_fifteen_decisions(tmp_path, config, sources):
    result, env, client = run(tmp_path, config, sources, [output(ACTIONS)] * 15)
    assert env.step == 15 and client.calls == 15
    assert result["status"] == "official_done"
    assert len([op for op, _ in env.calls if op == "observe"]) == 14
    assert all(p["action"] == ACTIONS[0] for op, p in env.calls if op == "step")


def test_two_frames_are_adjacent_public_decisions(config, sources):
    planner = OfficialPlanner(sources, TASK, config["model"]["name"], 4, [], "json_object")
    first = packet()["public"]
    m0 = planner.messages(first)
    assert sum(x["type"] == "image_url" for x in m0[0]["content"]) == 1
    planner.feedback(ACTIONS[0], packet(1)["public"])
    second = packet(1)["public"]
    m1 = planner.messages(second)
    first["manipulation"]["objects"].clear()
    assert "object 1" in m1[0]["content"][1]["text"]
    assert str(ACTIONS[0]) in m1[0]["content"][1]["text"]
    planner.feedback(ACTIONS[1], packet(2)["public"])
    m2 = planner.messages(packet(2)["public"])
    urls = [x["image_url"]["url"] for x in m2[0]["content"] if x["type"] == "image_url"]
    assert urls == ["data:image/png;base64," + second["images"][0]["base64"],
                    "data:image/png;base64," + packet(2)["public"]["images"][0]["base64"]]
    assert "object 2" in m2[0]["content"][1]["text"]
    assert "NOT\npersistent identities" in m2[0]["content"][0]["text"]
    assert sum(x["type"] == "image_url" for x in m2[0]["content"]) == 2


def test_official_scores_do_not_enter_requests(tmp_path, config, sources):
    result, env, client = run(tmp_path, config, sources, [output(ACTIONS)] * 15)
    for payload in client.payloads:
        raw = canonical(payload).decode()
        assert '"task_success"' not in raw
        assert '"reward"' not in raw and '"termination_source"' not in raw
        assert "reference_answer" not in raw and "waypoint_sets" not in raw
        assert payload["model"] == "qwen3.8-flash" and payload["seed"] == 137
        assert payload["max_tokens"] == 4096 and payload["enable_thinking"] is False


def test_service_failure_is_not_retried(tmp_path, config, sources):
    result, env, client = run(tmp_path, config, sources,
                             [OperationalError("model_http_error", 524)])
    assert client.calls == 1 and env.step == 0 and env.closed
    assert result["status"] == "operational_error"


def test_model_tasks_sources_and_budgets_match_parent(config):
    parent = json.loads((ROOT / "official_direct/baseline-20261107-r03.json").read_text())
    assert config["model"] == parent["model"]
    assert config["execution"] == parent["execution"]
    assert config["n_shots"] == parent["n_shots"]
    assert config["max_requests"] == parent["max_requests"]
    assert config["episodes"] == [t for t in parent["episodes"] if t["environment"] == "manipulation"]
    for key in ("source_root", "manipulation_catalog", "manipulation_dataset",
                "environment_seed", "python", "gpu", "timeout_s"):
        assert config["runtime"][key] == parent["runtime"][key]


def test_worker_path_and_credential_boundary(config, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-secret")
    child = child_environment()
    assert "DASHSCOPE_API_KEY" not in child and "OPENAI_API_KEY" not in child
    argv = command(config, TASK, probe=True)
    assert "/manipulation_minimal/worker.py" in argv[-1]
    assert "LD_LIBRARY_PATH=" in argv[-1]
    assert "offline-secret" not in canonical(argv).decode()
