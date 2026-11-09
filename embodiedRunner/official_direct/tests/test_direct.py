"""Offline contract checks. No live model, SSH, GPU or simulator calls."""
import ast
import base64
import copy
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import types
import urllib.error

import pytest
from PIL import Image

from official_direct import VERSION
from official_direct.client import ModelClient
from official_direct.config import load_config
from official_direct.planner import OfficialPlanner
from official_direct.protocol import COORD_SOURCE, Task, validate_packet, vector
from official_direct.runner import episode, suite
from official_direct.sources import ContractError, LIMITS, OperationalError, ROOT, Sources, canonical, digest
from official_direct import transport, worker


@pytest.fixture(autouse=True)
def no_live_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("live network forbidden during offline validation")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def config():
    return load_config(ROOT / "official_direct/default.json")[0]


@pytest.fixture
def sources():
    return Sources()


def image():
    stream = io.BytesIO()
    Image.new("RGB", (500, 500), (53, 89, 111)).save(stream, format="PNG")
    data = stream.getvalue()
    return {"view": "head_rgb", "base64": base64.b64encode(data).decode(),
            "sha256": digest(data), "source": "offline fixture"}


def packet(task, step=0, success=False, done=False, action_success=True, invalid=0):
    public = {"instruction": "Move the object onto the table.",
              "images": [image()], "actions": [f"official action {i}" for i in range(8)],
              "step": step, "env_feedback": "" if not step else "Last action executed.",
              "last_action_success": action_success, "manipulation": None}
    if task.environment == "manipulation":
        public.update(actions=[], manipulation={"objects": {"object 1": [12, 23, 34]},
                      "task_variation": "pick_cube_shape", "source": COORD_SOURCE})
    return {"task": task.as_dict(), "public": public,
            "official": {"done": done, "task_success": int(success), "task_progress": 0.31415926,
                         "invalid_actions": invalid, "reward": 1 if success else -1 if not action_success else 0},
            "input_sha256": digest(canonical({"task": task.as_dict(), "instruction": public["instruction"]}))}


def output(actions, kind="navigation"):
    return json.dumps({"visual_state_description": "state", "reasoning_and_reflection": "reason",
                       "language_plan": "plan", "executable_plan": [
                           {"action": str(x)} if kind == "manipulation" else
                           {"action_id": x, "action_name": "ignored upstream name"} for x in actions]})


@pytest.mark.parametrize("kind", list(LIMITS))
@pytest.mark.parametrize("model", ["gpt-4o-mini", "Qwen3-VL-8B-Instruct", "Qwen2.5-VL-7B-Instruct"])
def test_full_official_class_message_parity(kind, model, config, sources):
    """Compare to FULL upstream act(), including model-specific template logic."""
    task = Task(kind, "base", 0)
    public = packet(task)["public"]
    planner = OfficialPlanner(sources, task, model, config["n_shots"][kind], public["actions"])
    a = planner.adapter
    action = [50, 40, 30, 60, 60, 60, 1] if kind == "manipulation" else 0
    text = output([action], kind)
    captured = []
    class Remote:
        def __init__(self, *args, **kwargs):
            pass
        def respond(self, messages):
            captured.append(copy.deepcopy(messages))
            return text
    namespace = dict(planner.namespace, RemoteModel=Remote, ast=ast,
                     logger=types.SimpleNamespace(debug=lambda *a: None))
    urls = {"current-0.png": "data:image/png;base64," + public["images"][0]["base64"]}
    namespace["local_image_to_data_url"] = lambda image_path: urls[image_path]
    name = {"navigation": "nav_planner.py", "alfred": "vlm_planner.py",
            "habitat": "vlm_planner.py", "manipulation": "manip_planner.py"}[kind]
    tree = ast.parse(sources.read(kind, "embodiedbench/planner/" + name))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == type(a).__name__)
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, "full-upstream-reference", "exec"), namespace)
    kwargs = dict(model_name=model, model_type="remote", system_prompt=a.system_prompt,
                  examples=a.examples, n_shot=a.n_shot, chat_history=False, language_only=False, kwargs={})
    if kind != "manipulation":
        kwargs["actions"] = public["actions"]
    reference = namespace[cls.name](**kwargs)
    reference.reset()
    for iteration in range(2):
        if kind == "manipulation":
            m = public["manipulation"]
            reference.act(["current-0.png"], public["instruction"], str(m["objects"]), m["task_variation"])
        else:
            reference.act("current-0.png", public["instruction"])
        assert planner.messages(public) == captured[-1]
        assert planner.parse(text).actions == [action]
        key = "action" if kind == "manipulation" else "action_id"
        info = {key: action, "env_feedback": "A wall prevented the action."}
        reference.update_info(info)
        public["env_feedback"] = info["env_feedback"]
        planner.feedback(action, public)


class FakeClient:
    def __init__(self, config, responses):
        self.builder = ModelClient(config["model"])
        self.responses, self.calls, self.payloads = iter(responses), 0, []
    def payload(self, messages, schema):
        return self.builder.payload(messages, schema)
    def complete(self, payload):
        self.calls += 1
        self.payloads.append(payload)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return {"text": value, "message": {"content": value}, "usage": {"total_tokens": 8},
                "model": "offline-fixture", "response_sha256": digest(value.encode()), "finish_reason": "stop"}


class FakeEnvironment:
    def __init__(self, task, *, fail_first=False, fail_transport=False, success_step=None):
        self.task, self.step, self.invalid = task, 0, 0
        self.fail_first, self.fail_transport, self.success_step = fail_first, fail_transport, success_step
        self.calls, self.closed = [], False
    def request(self, op, **parameters):
        self.calls.append((op, parameters))
        if op == "open" or op == "observe":
            return packet(self.task, self.step, invalid=self.invalid)
        if op == "invalid":
            self.invalid += 1
            return packet(self.task, self.step, invalid=self.invalid, done=self.invalid >= 10)
        if self.fail_transport:
            raise OperationalError("worker_timeout_outcome_unknown")
        self.step += 1
        failed = self.fail_first and self.step == 1
        self.invalid += int(failed)
        success = self.step == self.success_step
        return packet(self.task, self.step, action_success=not failed, invalid=self.invalid,
                      success=success, done=success or self.step >= LIMITS[self.task.environment])
    def close(self):
        self.closed = True


def run_episode(tmp_path, kind, config, sources, responses, **kwargs):
    task = Task(kind, "base", 0)
    env = FakeEnvironment(task, **kwargs)
    client = FakeClient(config, responses)
    result = episode(task, config, sources, client, lambda *args: env, tmp_path / "episode")
    return result, env, client


@pytest.mark.parametrize("kind,executed", [("navigation", 1), ("alfred", 1), ("habitat", 1), ("manipulation", 3)])
def test_full_plan_failure_semantics(tmp_path, kind, executed, config, sources):
    action = [50, 40, 30, 60, 60, 60, 1] if kind == "manipulation" else 0
    result, env, client = run_episode(
        tmp_path, kind, config, sources, [output([action] * 3, kind), output([], kind)], fail_first=True)
    assert env.step == executed
    assert client.calls == 2
    assert result["status"] == ("planner_error" if kind == "navigation" else "empty_plan")
    assert env.closed


@pytest.mark.parametrize("kind", list(LIMITS))
def test_all_official_step_limits_without_plan_cap(tmp_path, kind, config, sources):
    action = [50, 40, 30, 60, 60, 60, 1] if kind == "manipulation" else 0
    result, env, client = run_episode(tmp_path, kind, config, sources, [output([action] * 42, kind)])
    assert result["status"] == "official_done"
    assert env.step == LIMITS[kind]
    assert client.calls == 1
    assert len([c for c in env.calls if c[0] == "step"]) == LIMITS[kind]
    assert result["metrics"]["num_steps"] == LIMITS[kind]


@pytest.mark.parametrize("kind", ["alfred", "habitat"])
def test_ten_malformed_plans_accumulate_without_steps(tmp_path, kind, config, sources):
    result, env, client = run_episode(tmp_path, kind, config, sources, ["bad json"] * 10)
    assert result["status"] == "invalid_action_limit"
    assert env.step == 0 and env.invalid == 10 and client.calls == 10
    assert result["metrics"]["num_invalid_actions"] == 10
    assert result["metrics"]["planner_output_error"] == 10


@pytest.mark.parametrize("kind", ["alfred", "habitat"])
def test_legal_prefix_matches_official_parser(kind, config, sources):
    task = Task(kind, "base", 0)
    planner = OfficialPlanner(sources, task, "gpt-4o-mini", 10, packet(task)["public"]["actions"])
    assert planner.parse(output([1, 2, 100, 3])).actions == [1, 2]
    assert planner.parse(output([100, 1])).status == "invalid"
    assert planner.parse(output([True])).status == "invalid"


@pytest.mark.parametrize("kind", list(LIMITS))
def test_scores_hidden_coordinates_retained_and_success_stops_plan(tmp_path, kind, config, sources):
    action = [50, 40, 30, 60, 60, 60, 1] if kind == "manipulation" else 0
    result, env, client = run_episode(tmp_path, kind, config, sources,
                                      [output([action] * 5, kind)], success_step=1)
    assert env.step == 1 and result["official"]["task_success"] == 1
    model_input = canonical(client.payloads).decode()
    assert "0.31415926" not in model_input
    if kind == "manipulation":
        assert "'object 1': [12, 23, 34]" in model_input
    assert all("tools" not in p for p in client.payloads)


def test_operation_failure_stops_suite_and_keeps_fixed_denominator(tmp_path, config, sources):
    tasks = [Task(k, "base", 0) for k in LIMITS]
    client = FakeClient(config, [OperationalError("model_http_error", 524)])
    created = []
    def factory(task, *args):
        env = FakeEnvironment(task)
        created.append(env)
        return env
    summary = suite(tasks, config, sources, client, factory, tmp_path / "suite")
    assert client.calls == len(created) == 1 and created[0].closed
    assert summary["selected"] == 4 and summary["status"] == "stopped_on_operational_failure"
    rows = json.loads((tmp_path / "suite/results.json").read_text())
    assert [r["status"] for r in rows] == ["operational_error"] + ["not_started"] * 3
    assert rows[0]["error"]["http_status"] == 524
    trajectory = (tmp_path / "suite/navigation-base-000/trajectory.jsonl").read_text()
    assert "model_request" in trajectory and "failure" in trajectory
    assert all(r["success_rate_selected"] == 0 for r in summary["by_environment"].values())


def test_unknown_step_outcome_is_never_resent(tmp_path, config, sources):
    result, env, client = run_episode(tmp_path, "navigation", config, sources,
                                     [output([0, 1])], fail_transport=True)
    assert result["status"] == "operational_error"
    assert len([c for c in env.calls if c[0] == "step"]) == 1 and client.calls == 1
    events = [json.loads(x) for x in (tmp_path / "episode/trajectory.jsonl").read_text().splitlines()]
    failure = next(e for e in events if e["type"] == "failure")
    assert failure["pending_action_may_have_executed"] is True


def test_private_fields_and_modified_images_rejected():
    task = Task("navigation", "base", 0)
    data = packet(task)
    data["public"]["reference_answer"] = "hidden"
    with pytest.raises(ContractError):
        validate_packet(data, task)
    data = packet(task)
    data["public"]["images"][0]["sha256"] = "0" * 64
    with pytest.raises(ContractError):
        validate_packet(data, task)


def test_manipulation_legal_domain_and_plan_variants(config, sources):
    task = Task("manipulation", "base", 0)
    planner = OfficialPlanner(sources, task, "gpt-4o-mini", 4, [])
    high = [100, 100, 100, 120, 120, 120, 1.0]
    assert vector(high) == high
    for plan in ([high], [{"action": str(high)}], [[{"action": high}]], str([high])):
        assert planner.parse(json.dumps({"executable_plan": plan})).actions == [high]
    for bad in ([True] * 7, [1.5] * 7, [101, 0, 0, 0, 0, 0, 1], [0] * 6 + [0.5]):
        with pytest.raises(ContractError):
            vector(bad)


def test_http_failure_single_attempt_and_no_error_body(config, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-credential-do-not-persist")
    client = ModelClient(config["model"])
    calls = []
    def fail(request, **kwargs):
        calls.append(request)
        raise urllib.error.HTTPError("https://hidden", 524, "private service text", {}, None)
    client.opener.open = fail
    with pytest.raises(OperationalError) as caught:
        client.complete(client.payload([], {}))
    assert caught.value.code == "model_http_error" and caught.value.http_status == 524
    assert len(calls) == 1 and "private" not in str(caught.value)
    with pytest.raises(OperationalError):
        client.safe({"content": client.key})


def test_exact_schema_sampling_and_gpt5_token_parameter(config):
    client = ModelClient(config["model"])
    payload = client.payload([], {"type": "object"})
    assert payload["response_format"]["json_schema"]["name"] == "embodied_planning"
    assert payload["max_tokens"] == 4096 and payload["temperature"] == 0
    assert "seed" not in payload and "enable_thinking" not in payload
    client.config = dict(config["model"], name="gpt-5.6-terra")
    payload = client.payload([], {})
    assert payload["max_completion_tokens"] == 4096 and "temperature" not in payload


def test_worker_environment_and_ssh_arguments(config, sources, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-worker")
    monkeypatch.setenv("CUSTOM_ENDPOINT", "must-not-reach-worker")
    assert "OPENAI_API_KEY" not in transport.child_environment()
    assert "CUSTOM_ENDPOINT" not in transport.child_environment()
    args = transport.command(config, Task("navigation", "base", 0), probe=True)
    assert "BatchMode=yes" in args and "StrictHostKeyChecking=yes" in args
    assert "UpdateHostKeys=no" in args and "--probe" in args[-1]
    # Preserve process environment after the worker's deliberate clear().
    original = dict(os.environ)
    try:
        worker.runtime_environment(tmp_path, sources, Task("habitat", "base", 0), 7)
        assert "OPENAI_API_KEY" not in os.environ and "CUSTOM_ENDPOINT" not in os.environ
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
        assert os.environ["HOME"] == str(tmp_path / "home")
    finally:
        os.environ.clear()
        os.environ.update(original)
        sys.modules.pop("embodiedbench.main", None)


def test_check_cannot_launch_processes(config, sources, monkeypatch):
    from official_direct.__main__ import check
    def blocked(*args, **kwargs):
        raise AssertionError("offline check must not launch subprocesses")
    monkeypatch.setattr(subprocess, "Popen", blocked)
    report = check(config, [Task(k, "base", 0) for k in LIMITS], sources)
    assert report["model_requests"] == report["simulator_launches"] == report["network_requests"] == 0


def test_preserved_sources_and_server_python39_grammar():
    baseline = json.loads((ROOT / "official_direct/verification/preserved-before.json").read_text())
    for name, sha in baseline["files"].items():
        assert digest((ROOT / name).read_bytes()) == sha, name
    for path in (ROOT / "official_direct").glob("*.py"):
        ast.parse(path.read_text(), feature_version=(3, 9))


def test_source_changes_are_detected(tmp_path, sources):
    path = "embodiedbench/planner/nav_planner.py"
    tree = "upstream"
    destination = tmp_path / tree / path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(sources.read("navigation", path) + b"\n# changed\n")
    clone = copy.copy(sources)
    clone.root = tmp_path
    with pytest.raises(ContractError, match="official_source_changed"):
        clone.read("navigation", path)


def test_config_rejects_secrets_invalid_gpu_and_duplicate_selection(tmp_path, config):
    for mutate in (
        lambda c: c["model"].update(api_key="must-not-be-stored"),
        lambda c: c["runtime"]["gpu"].update(habitat=2),
        lambda c: c["episodes"].append(c["episodes"][0]),
    ):
        value = copy.deepcopy(config)
        mutate(value)
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps(value))
        with pytest.raises(ContractError):
            load_config(path)


def test_request_budget_has_distinct_status(tmp_path, config, sources):
    config["max_requests"]["navigation"] = 1
    result, env, client = run_episode(tmp_path, "navigation", config, sources, [output([0])])
    assert result["status"] == "request_budget" and env.step == client.calls == 1
    assert not result["official"]["done"] and result["official"]["task_success"] == 0


@pytest.mark.parametrize("kind", ["alfred", "habitat"])
def test_environment_and_planner_invalid_counts_share_one_budget(tmp_path, kind, config, sources):
    # One failed real step followed by 9 malformed plans hits cumulative 10.
    result, env, client = run_episode(tmp_path, kind, config, sources,
                                     [output([0, 1])] + ["bad"] * 9, fail_first=True)
    assert result["status"] == "invalid_action_limit" and env.invalid == 10 and env.step == 1
    assert result["metrics"]["num_invalid_actions"] == 10 and client.calls == 10


def test_worker_manipulation_boundary_uses_only_official_anonymous_derivation(tmp_path, monkeypatch):
    import numpy as np
    utils = types.ModuleType("embodiedbench.envs.eb_manipulation.eb_man_utils")
    observed_private_fields = []
    def coordinates(obs, task_class, camera_views):
        observed_private_fields.append(sorted(obs))
        return {"object 1": np.array([10, 20, 30])}, [], [], []
    utils.form_object_coord_for_input = coordinates
    utils.draw_xyz_coordinate = lambda path, resolution: path
    utils.draw_bounding_boxes = lambda paths, *args: paths
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    session = worker.Environment.__new__(worker.Environment)
    session.task, session.work = Task("manipulation", "base", 0), tmp_path
    session.env = types.SimpleNamespace(_current_step=1, episode_language_instruction="Pick up the cube.",
                                        current_task_variation="pick_cube_shape", task_class="pick")
    session.obs = {"front_rgb": np.zeros((500, 500, 3), dtype=np.uint8),
                   "object_informations": {"private simulator objects": {}},
                   "front_depth": np.zeros((500, 500)), "waypoint_sets": "never in public"}
    session.info = {"action_success": 0.0, "task_success": 0.0, "env_feedback": "Last action is invalid.",
                    "episode_elapsed_seconds": 3.5, "private_reference": "never in public"}
    session.reward, session.done = -1, False
    result = session.packet(annotate=True)
    assert result["public"]["manipulation"]["objects"] == {"object 1": [10, 20, 30]}
    assert result["public"]["images"][0]["source"].startswith("official front_rgb")
    assert not result["public"]["last_action_success"] and result["official"]["reward"] == -1
    assert "waypoint_sets" not in canonical(result).decode() and "private_reference" not in canonical(result).decode()
    assert observed_private_fields  # Derived privately, published through the official utility.


def test_real_transport_dummy_process_and_timeout_no_resend(tmp_path, config, sources, monkeypatch):
    monkeypatch.setattr(transport, "preflight", lambda *args: {"offline_fixture": True})
    capture = tmp_path / "received.json"
    program = (
        "import sys,json,pathlib; rows=[]\n"
        "for line in sys.stdin:\n"
        " rows.append(json.loads(line))\n"
        f"pathlib.Path({str(capture)!r}).write_text(json.dumps(rows))\n"
    )
    monkeypatch.setattr(transport, "command", lambda *args, **kwargs: [sys.executable, "-B", "-c", program])
    config["runtime"]["timeout_s"] = 0.05
    journal = types.SimpleNamespace(append=lambda event: None)
    session = transport.Session(config, Task("navigation", "base", 0), tmp_path, journal, sources, "dummy")
    with pytest.raises(OperationalError, match="worker_timeout_outcome_unknown"):
        session.request("step", action=0)
    with pytest.raises(OperationalError, match="worker_connection_closed"):
        session.request("step", action=0)
    session.close()
    assert len(json.loads(capture.read_text())) == 1


def test_real_transport_dummy_process_success(tmp_path, config, sources, monkeypatch):
    monkeypatch.setattr(transport, "preflight", lambda *args: {"offline_fixture": True})
    program = (
        "import sys,json\n"
        "for line in sys.stdin:\n"
        " c=json.loads(line)\n"
        f" print(json.dumps(dict(id=c['id'],version={VERSION!r},ok=True,data=dict(op=c['op']))),flush=True)\n"
        " if c['op']=='close': break\n"
    )
    monkeypatch.setattr(transport, "command", lambda *args, **kwargs: [sys.executable, "-B", "-c", program])
    session = transport.Session(config, Task("navigation", "base", 0), tmp_path,
                                 types.SimpleNamespace(append=lambda event: None), sources, "dummy")
    assert session.request("step", action=0) == {"op": "step"}
    session.close()
    assert session.process.returncode == 0


def test_manipulation_probe_checks_all_recorded_runtime_packages(monkeypatch):
    fake_sources = types.SimpleNamespace(verify_all=lambda: {}, hashes={})
    assets = {"dataset_sha256": "fixture", "expected_packages": {"numpy": "1.26.4", "cffi": "1.17.1"}}
    monkeypatch.setattr(worker, "selected_assets", lambda *args: assets)
    monkeypatch.setattr(worker.platform, "system", lambda: "Linux")
    monkeypatch.setattr(worker.importlib.util, "find_spec", lambda name: object())
    versions = dict(assets["expected_packages"])
    monkeypatch.setattr(worker, "package_version", lambda name: versions.get(name))
    task = Task("manipulation", "base", 0)
    assert worker.probe(fake_sources, task, None)["runtime_ready"]
    versions["cffi"] = "changed"
    assert not worker.probe(fake_sources, task, None)["runtime_ready"]


def test_parallel_suite_preserves_plans_and_selection(tmp_path, config, sources):
    import threading
    barrier = threading.Barrier(2)
    config["execution"] = {"max_episodes": 2, "max_api_requests": 32, "max_gpu_episodes": 1,
                           "per_environment": {k: 2 for k in LIMITS}}
    tasks = [Task("alfred", "base", 3), Task("alfred", "base", 4)]
    class Client:
        payload = FakeClient(config, []).payload
        def complete(self, payload):
            barrier.wait(timeout=5)
            return {"text": output([0, 1], "alfred"), "model": "offline"}
    envs = {}
    def factory(task, *args):
        envs[task.identity] = FakeEnvironment(task, success_step=2)
        return envs[task.identity]
    result = suite(tasks, config, sources, Client(), factory, tmp_path / "parallel")
    assert result["status"] == "finished" and result["peak_http_concurrency"] == 2
    assert result["by_environment"]["alfred"]["successes"] == 2
    assert all(env.step == 2 and env.closed for env in envs.values())
    saved = json.loads((tmp_path / "parallel/results.json").read_text())
    assert [row["task"] for row in saved] == [t.as_dict() for t in tasks]


def test_api_admission_closes_without_retry_after_failure():
    from official_direct.runner import RequestGate
    class Client:
        calls = 0
        def complete(self, payload):
            self.calls += 1
            raise OperationalError("model_http_error", 524)
    client = Client()
    gate = RequestGate(client, 32)
    with pytest.raises(OperationalError, match="model_http_error"):
        gate.complete({})
    with pytest.raises(OperationalError, match="suite_request_admission_closed"):
        gate.complete({})
    assert client.calls == gate.requests == 1


def test_resolved_official_tree_symlink_is_readonly_compatible(tmp_path, sources):
    clone = copy.copy(sources)
    clone.root = tmp_path
    (tmp_path / "upstream").symlink_to(ROOT / "upstream", target_is_directory=True)
    name = "embodiedbench/planner/nav_planner.py"
    assert clone.read("navigation", name) == sources.read("navigation", name)


def test_native_library_path_is_set_before_worker_python(config):
    args = transport.command(config, Task("manipulation", "base", 0), probe=True)
    import shlex
    remote = shlex.split(args[-1])
    assert remote[0] == "env"
    assert remote[1].startswith("LD_LIBRARY_PATH=")
    assert remote.index(config["runtime"]["python"]["manipulation"]) > 1
    assert all("API_KEY" not in arg for arg in remote)


def test_manipulation_json_mode_retains_official_output_guidance(config, sources):
    task = Task("manipulation", "base", 0)
    config["model"].update(name="qwen3.8-flash", response_format="json_object")
    public = packet(task)["public"]
    planner = OfficialPlanner(sources, task, config["model"]["name"], 4, [], "json_object")
    messages = planner.messages(public)
    text = messages[0]["content"][0]["text"]
    assert sources.literal("navigation", "embodiedbench/planner/planner_utils.py", "template_manip") in text
    payload = ModelClient(config["model"]).payload(messages, planner.schema)
    assert payload["response_format"] == {"type": "json_object"}
    with pytest.raises(ContractError, match="requires_json_instruction"):
        ModelClient(config["model"]).payload([{"role": "user", "content": "Output a plan."}], {})
