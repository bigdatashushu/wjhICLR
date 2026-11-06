"""Offline checks for official prompt parity and real-run control flow."""
from __future__ import annotations

import ast
import base64
import io
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from household_planner import OfficialHouseholdPlanner, adapter
from household_protocol import Observation, source
from household_run import evaluate
from protocol import ROOT, sha256


class Client:
    last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    last_response_identity = {"model": "qwen3.8-flash"}
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    def chat(self, messages, **kw):
        self.messages.append(messages)
        return next(self.outputs)


def plan(actions):
    return json.dumps({"executable_plan": [{"action_id": i} for i in actions]})


def observation(actions=("navigate to the table 1", "pick up the apple")):
    from PIL import Image
    stream = io.BytesIO()
    Image.new("RGB", (500, 500)).save(stream, "PNG")
    png = stream.getvalue()
    return Observation(png, sha256(png), 0, "Move an apple.", actions, "0" * 64)


class Tests(unittest.TestCase):
    def test_native_simulator_logs_do_not_enter_json_transport(self):
        from bridge import simulator_environment
        code = (
            "import os,sys; from household_worker import native_log; "
            "wire=os.fdopen(os.dup(1),'w',buffering=1); "
            "log=open(sys.argv[1],'a'); "
            "ctx=native_log(log); ctx.__enter__(); "
            "os.write(1,b'native stdout\\n'); os.write(2,b'native stderr\\n'); "
            "print('Python diagnostic'); wire.write('{\"ok\":true}\\n'); "
            "ctx.__exit__(None,None,None); log.close(); wire.close()"
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "checks") as directory:
            log = Path(directory) / "runtime.log"
            process = subprocess.run(
                [sys.executable, "-B", "-c", code, str(log)], cwd=ROOT,
                env=simulator_environment(ROOT), capture_output=True, check=True)
            self.assertEqual(json.loads(process.stdout), {"ok": True})
            self.assertEqual(process.stderr, b"")
            text = log.read_text()
            self.assertIn("native stdout", text)
            self.assertIn("native stderr", text)
            self.assertIn("Python diagnostic", text)

    def test_habitat_packet_keeps_only_current_head_rgb(self):
        import numpy as np
        from PIL import Image
        from household_worker import OfficialHousehold
        tree = ast.parse(source("embodiedbench/envs/eb_habitat/utils.py"))
        methods = [x for x in tree.body if isinstance(x, ast.FunctionDef)
                   and x.name in ("process_image_array", "observations_to_image")]
        module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
        namespace = {"np": np}
        exec(compile(module, "official-rgb-conversion", "exec"), namespace)
        utils = types.ModuleType("embodiedbench.envs.eb_habitat.utils")
        utils.observations_to_image = namespace["observations_to_image"]
        session = OfficialHousehold("habitat", "base", ROOT / "checks")
        session.env = types.SimpleNamespace(
            _current_step=0, episode_language_instruction="fixture",
            language_skill_set=["action"], _cur_invalid_actions=0)
        session.input_sha256 = "0" * 64
        head = np.full((500, 500, 3), (33, 77, 155), dtype=np.uint8)
        sensors = {"head_rgb": head,
                   "third_rgb": np.full((500, 500, 3), 255, dtype=np.uint8),
                   "head_depth": np.ones((500, 500, 1), dtype=np.float32)}
        with patch.dict("sys.modules", {utils.__name__: utils}):
            packet = session.packet(sensors)
        png = base64.b64decode(packet["image_base64"])
        actual = np.asarray(Image.open(io.BytesIO(png)))
        np.testing.assert_array_equal(actual, head)
        self.assertEqual(actual.shape, (500, 500, 3))

    def test_original_prompt_and_message_parity(self):
        tree = ast.parse(source("embodiedbench/planner/vlm_planner.py"))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef))
        module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        url = "data:image/png;base64,test"
        ns = {"RemoteModel": lambda *a, **kw: None, "json": json,
              "local_image_to_data_url": lambda image_path: url}
        exec(compile(module, "full-official-class", "exec"), ns)
        for environment in ("alfred", "habitat"):
            for subset in ("base", "long_horizon"):
                with self.subTest(environment=environment, subset=subset):
                    actual = adapter(environment, subset, ["action 0", "action 1"])
                    expected = ns["VLMPlanner"](
                        "qwen3.8-flash", "remote", ["action 0", "action 1"],
                        actual.system_prompt, actual.examples, n_shot=10,
                        chat_history=False, use_feedback=True, multistep=False)
                    actual.namespace["local_image_to_data_url"] = lambda image_path: url
                    for history in ([], [[1, "Last action is invalid."]]):
                        a = actual.process_prompt("Instruction.", history)
                        b = expected.process_prompt("Instruction.", history)
                        self.assertEqual(a, b)
                        self.assertEqual(actual.get_message("image.png", a), expected.get_message("image.png", b))

    def test_parser_empty_invalid_and_prefix(self):
        obs = observation()
        client = Client([plan([]), plan([9, 0]), plan([0, 9, 1]), "{", plan([True])])
        planner = OfficialHouseholdPlanner(client, "habitat", "base", obs.actions)
        self.assertEqual(planner.act(obs), -2)
        self.assertEqual(planner.act(obs), -1)
        self.assertEqual(planner.act(obs), [0])
        self.assertEqual(planner.act(obs), -1)
        self.assertEqual(planner.act(obs), -1)
        self.assertEqual(planner.adapter.output_json_error, 2)
        self.assertFalse(hasattr(obs, "task_success"))
        self.assertFalse(hasattr(obs, "task_progress"))

    def test_dynamic_actions_and_feedback(self):
        obs = observation()
        client = Client([plan([1])])
        planner = OfficialHouseholdPlanner(client, "alfred", "base", ["old action"])
        planner.reset(obs.actions)
        self.assertEqual(planner.act(obs), [1])
        planner.update(1, "Last action is invalid.")
        message = planner.messages(obs)[0]["content"][1]["text"]
        self.assertIn("pick up the apple", message)
        self.assertIn("env feedback: Last action is invalid.", message)
        self.assertNotIn("old action", message)

    def test_loop_replans_on_failed_action_and_keeps_denominator(self):
        obs = observation()
        class Bridge:
            def __init__(self):
                self.steps = []
                self.index = 0
            def request(self, op, **kw):
                if op == "open":
                    self.index = kw["index"]
                    self.step = self.invalid = 0
                elif op == "invalid":
                    self.invalid += 1
                elif op == "step":
                    self.steps.append((self.index, kw["action"]))
                    self.step += 1
                    if self.step == 1:
                        self.invalid += 1
                return {
                    "image_base64": base64.b64encode(obs.png).decode(), "image_sha256": obs.image_sha256,
                    "instruction": obs.instruction, "actions": list(obs.actions), "input_sha256": obs.input_sha256,
                    "step": self.step, "invalid_actions": self.invalid, "done": self.step >= 2,
                    "task_success": self.step >= 2, "task_progress": self.step / 2,
                    "last_action_success": self.step != 1, "env_feedback": "Last action is invalid." if self.step == 1 else "",
                    "reward": 0,
                }
        client = Client([plan([9]), plan([0, 1]), plan([1])] * 3)
        bridge = Bridge()
        with tempfile.TemporaryDirectory(dir=ROOT / "checks") as directory:
            result = evaluate("habitat", "base", Path(directory), client, bridge, "0" * 64)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["completed_tasks"], 3)
            self.assertEqual(result["successful_tasks"], 3)
            self.assertEqual(result["model_requests"], 9)
            self.assertEqual(bridge.steps, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
            self.assertEqual([x["num_invalid_actions"] for x in result["results"]], [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
