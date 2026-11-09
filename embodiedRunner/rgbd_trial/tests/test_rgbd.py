"""Offline checks for sensor alignment, depth provenance and unchanged action execution."""
import base64
import copy
import io
import json
import socket

import numpy as np
from PIL import Image
import pytest

from rgbd_trial.client import ModelClient
from rgbd_trial.config import load_config
from rgbd_trial.depth import decode_raw, make_depth, validate_depth
from rgbd_trial.planner import OfficialPlanner
from rgbd_trial.protocol import Task, validate_packet
from rgbd_trial.runner import episode
from rgbd_trial.sources import ROOT, ContractError, Sources, canonical, digest
from rgbd_trial.transport import child_environment, command


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("live network forbidden during offline checks")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def config():
    return load_config(ROOT / "rgbd_trial/experiment-20261107-r01.json")[0]


@pytest.fixture
def sources():
    return Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")


TASK = Task("navigation", "base", 0)


def packet(step=0, done=False, valid=True):
    stream = io.BytesIO()
    Image.new("RGB", (500, 500), (20 + step, 60, 90)).save(stream, format="PNG")
    png = stream.getvalue()
    public = {"instruction": "Move close to the chair.",
              "images": [{"view": "head_rgb", "base64": base64.b64encode(png).decode(),
                          "sha256": digest(png), "source": "offline fixture"}],
              "actions": [f"official action {i}" for i in range(8)], "step": step,
              "env_feedback": "" if not step else "Last motion executed.",
              "last_action_success": valid, "manipulation": None,
              "depth": make_depth(np.full((500, 500), 1 + step, dtype=np.float32), digest(png), step)}
    return {"task": TASK.as_dict(), "public": public, "official": {
        "done": done, "task_success": int(done), "invalid_actions": int(not valid), "reward": int(done)},
        "input_sha256": digest(canonical({"task": TASK.as_dict(), "instruction": public["instruction"]}))}


def test_depth_preserves_values_and_fixed_grayscale():
    raw = np.full((500, 500), 5, dtype="<f4")
    raw[0, :5] = [1, 5, 10, 20, np.nan]
    value = make_depth(raw, "rgb-fixture", 3)
    validate_depth(value, "rgb-fixture", 3)
    assert np.array_equal(decode_raw(value), raw, equal_nan=True)
    with Image.open(io.BytesIO(base64.b64decode(value["image"]["base64"]))) as im:
        assert im.size == (500, 500) and im.mode == "L"
        assert list(np.asarray(im)[0, :5]) == [230, 128, 0, 0, 0]
    assert value["stats"]["invalid_pixels"] == 1
    assert value["stats"]["max_meters"] == 20


@pytest.mark.parametrize("kind", ("step", "rgb", "raw", "display", "unit"))
def test_rejects_misaligned_or_changed_depth(kind):
    value = packet()["public"]["depth"]
    if kind == "step":
        value["synchronization"]["environment_step"] = 1
    elif kind == "rgb":
        value["synchronization"]["rgb_sha256"] = "wrong"
    elif kind == "raw":
        value["raw"]["base64"] = base64.b64encode(bytes(500 * 500 * 4)).decode()
    elif kind == "display":
        stream = io.BytesIO()
        Image.new("L", (500, 500), 0).save(stream, format="PNG")
        png = stream.getvalue()
        value["image"].update(base64=base64.b64encode(png).decode(), sha256=digest(png))
    elif kind == "unit":
        value["raw"]["unit"] = "millimeters"
    with pytest.raises(ContractError):
        validate_depth(value, packet()["public"]["images"][0]["sha256"], 0)


@pytest.mark.parametrize("bad", (None, np.ones((250, 250)), np.zeros((500, 500))))
def test_missing_wrong_size_or_invalid_depth_never_falls_back(bad):
    with pytest.raises(ContractError):
        make_depth(bad, "fixture", 0)


def test_packet_rejects_missing_depth_and_private_fields():
    good = packet()
    validate_packet(good, TASK)
    bad = copy.deepcopy(good)
    del bad["public"]["depth"]
    with pytest.raises(ContractError):
        validate_packet(bad, TASK)
    bad = copy.deepcopy(good)
    bad["public"]["depth"]["target_object_position"] = [1, 2, 3]
    with pytest.raises(ContractError):
        validate_packet(bad, TASK)


def test_model_receives_rgb_then_depth_with_no_raw_sensor_or_score(config, sources):
    planner = OfficialPlanner(sources, TASK, config["model"]["name"], 3,
                              packet()["public"]["actions"], "json_object")
    public = packet()["public"]
    messages = planner.messages(public)
    urls = [x["image_url"]["url"] for x in messages[0]["content"] if x["type"] == "image_url"]
    assert urls == ["data:image/png;base64," + public["images"][0]["base64"],
                    "data:image/png;base64," + public["depth"]["image"]["base64"]]
    serialized = canonical(messages).decode()
    assert public["depth"]["raw"]["base64"] not in serialized
    assert "task_success" not in serialized and "target_object_position" not in serialized
    payload = ModelClient(config["model"]).payload(messages, planner.schema)
    assert payload["model"] == "qwen3.8-flash" and payload["seed"] == 137
    assert payload["max_tokens"] == 4096 and payload["enable_thinking"] is False


def test_full_plan_execution_remains_unchanged(tmp_path, config, sources):
    class Env:
        step = 0
        closed = False
        def request(self, op, **parameters):
            if op == "step":
                self.step += 1
            return packet(self.step, done=self.step == 3)
        def close(self):
            self.closed = True
    class Client:
        calls = 0
        def payload(self, messages, schema):
            return ModelClient(config["model"]).payload(messages, schema)
        def complete(self, payload):
            self.calls += 1
            text = json.dumps({"executable_plan": [
                {"action_id": i, "action_name": "ignored upstream name"} for i in (0, 4, 0)]})
            return {"text": text, "message": {"content": text}, "model": "offline-fixture",
                    "usage": {}, "response_sha256": digest(text.encode()), "finish_reason": "stop"}
    env, client = Env(), Client()
    result = episode(TASK, config, sources, client, lambda *args: env, tmp_path / "episode")
    assert env.step == 3 and client.calls == 1 and env.closed
    assert result["status"] == "official_done" and result["metrics"]["num_steps"] == 3
    depths = sorted((tmp_path / "episode").glob("observation-*-depth-meters.npy"))
    assert len(depths) == 4
    for i, path in enumerate(depths):
        raw = np.load(path, allow_pickle=False)
        assert raw.shape == (500, 500) and np.all(raw == 1 + i)


def test_original_model_budget_and_read_only_source_bindings(config, monkeypatch):
    parent = json.loads((ROOT / "official_direct/baseline-20261107-r03.json").read_text())
    for key in ("model", "n_shots", "max_requests", "execution"):
        assert config[key] == parent[key]
    for key in ("source_root", "manipulation_catalog", "manipulation_dataset",
                "python", "gpu", "environment_seed", "timeout_s"):
        assert config["runtime"][key] == parent["runtime"][key]
    assert config["episodes"] == [Task("navigation", "base", i).as_dict() for i in range(5)]
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-secret")
    assert "DASHSCOPE_API_KEY" not in child_environment()
    assert "/rgbd_trial/worker.py" in command(config, TASK, probe=True)[-1]
