from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from xcodeapi import RequestLimitError, TransportError, XcodeClient

from LIBERORunner.config import Config, RunnerError
from LIBERORunner.environment import SOURCE_FILES, read_settling_steps, upstream_identity
from LIBERORunner.observation import build_messages, public_observation
from LIBERORunner.policy import GPTPolicy, InvalidAction, ModelIdentityError, parse_action
from LIBERORunner.rollout import rollout
from LIBERORunner.runner import prepare, run, summarize
from LIBERORunner.storage import Journal, digest, file_digest, write_json

CONTROLLER = {
    "name": "OSC_POSE", "control_freq": 20, "use_delta": True,
    "action_low": [-1.0] * 7, "action_high": [1.0] * 7,
    "input_min": -1.0, "input_max": 1.0,
    "output_min": [-0.05] * 3 + [-0.5] * 3,
    "output_max": [0.05] * 3 + [0.5] * 3,
}


def observation():
    return {
        "agentview_image": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "robot0_eye_in_hand_image": np.zeros((2, 3, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0., 0., 0., 1.]),
        "robot0_gripper_qpos": np.zeros(2),
        "robot0_joint_pos": np.zeros(7),
        "hidden_object_position": "DO_NOT_SEND_THIS_GROUND_TRUTH",
    }


class FakeClient:
    def __init__(self, *, model="gpt-6.1-terra", error=None, text=None, budget=10):
        self.config = Config()
        self.serializer = XcodeClient(self.config.api_profile(), max_requests=0)
        self.returned_model = model
        self.error = error
        self.text = text or '{"action":[0,0,0,0,0,0,-1]}'
        self.attempts = 0
        self.max_requests = budget
        self.payloads = []

    def prepare(self, *args, **kwargs):
        return self.serializer.prepare(*args, **kwargs)

    def send(self, payload):
        self.attempts += 1
        self.payloads.append(payload)
        if self.error:
            raise self.error
        return SimpleNamespace(
            returned_model=self.returned_model, finish_reason="completed",
            text=self.text, metadata=lambda: {"input_tokens": 1, "output_tokens": 1},
        )


class FakeEnv:
    def __init__(self, success_after=None, inconsistent_done=False):
        self.actions = []
        self.success_after = success_after
        self.inconsistent_done = inconsistent_done
        self.closed = False
        self.initial_state = None

    def reset(self):
        self.actions = []
        return observation()

    def set_init_state(self, state):
        self.initial_state = state
        return observation()

    def step(self, action):
        self.actions.append(action)
        success = self.check_success()
        return observation(), float(success), success or self.inconsistent_done, {}

    def check_success(self):
        return self.success_after is not None and len(self.actions) >= self.success_after

    def close(self):
        self.closed = True


class ConstantPolicy:
    def __init__(self, error=None):
        self.calls = 0
        self.error = error

    def act(self, obs, language, step):
        self.calls += 1
        if self.error:
            raise self.error
        return [0.1, 0., 0., 0., 0., 0., -1.]


class ContractTests(unittest.TestCase):
    def test_action_valid_and_fenced(self):
        text = '{"action":[0.1,0,0,0,0,0,-1]}'
        self.assertEqual(parse_action("```json\n" + text + "\n```", [-1] * 7, [1] * 7)[0], 0.1)

    def test_action_rejects_invalid_values_and_extra_fields(self):
        for text in (
            '{"action":[2,0,0,0,0,0,-1]}',
            '{"action":[true,0,0,0,0,0,-1]}',
            '{"action":[NaN,0,0,0,0,0,-1]}',
            '{"action":[0,0,0]}',
            '{"action":[0,0,0,0,0,0,-1],"success":true}',
            'I think {"action":[0,0,0,0,0,0,-1]}',
        ):
            with self.subTest(text=text), self.assertRaises(InvalidAction):
                parse_action(text, [-1] * 7, [1] * 7)

    def test_observation_excludes_scene_ground_truth(self):
        state, images = public_observation(observation(), "raw")
        messages = build_messages("task", 0, state, images, None, "system")
        encoded = json.dumps(messages)
        self.assertNotIn("DO_NOT_SEND", encoded)
        self.assertNotIn("hidden_object_position", encoded)
        self.assertEqual(len(images), 2)

    def test_image_orientation_is_explicit(self):
        from PIL import Image
        from io import BytesIO
        raw = observation()["agentview_image"]
        for orientation, expected in (("raw", raw), ("flip_vertical", raw[::-1]),
                                      ("rotate_180", raw[::-1, ::-1])):
            _, images = public_observation(observation(), orientation)
            actual = np.asarray(Image.open(BytesIO(images["agentview_image"])))
            np.testing.assert_array_equal(actual, expected)

    def test_rejects_nonfinite_proprioception(self):
        obs = observation()
        obs["robot0_eef_pos"][0] = np.nan
        with self.assertRaises(ValueError):
            public_observation(obs, "raw")

    def test_original_settling_loop_is_anchored(self):
        source = (
            "def evaluate_one_task_success():\n"
            "    for i in range(99):\n        unrelated()\n"
            "    for _ in range(5):\n        obs = env.step(dummy)\n"
        )
        self.assertEqual(read_settling_steps(source), 5)
        with self.assertRaises(RunnerError):
            read_settling_steps(source.replace("range(5)", "range(10)"))

    def test_rollout_waits_five_steps_and_stops_on_goal(self):
        events = []
        env, policy = FakeEnv(success_after=7), ConstantPolicy()
        result = rollout(env, policy, "fixed-state", "task", 10, 5, events.append)
        self.assertEqual(result, {"status": "success", "success": True, "control_steps": 2})
        self.assertEqual(env.actions[:5], [[0.] * 7] * 5)
        self.assertEqual(policy.calls, 2)
        self.assertEqual(sum(e["event"] == "action_ack" for e in events), 2)
        self.assertEqual(env.initial_state, "fixed-state")

    def test_horizon_counts_only_policy_steps(self):
        events = []
        env = FakeEnv()
        result = rollout(env, ConstantPolicy(), "state", "task", 3, 5, events.append)
        self.assertEqual(result["status"], "task_failure")
        self.assertEqual(len(env.actions), 8)
        self.assertEqual(result["control_steps"], 3)

    def test_invalid_action_never_reaches_environment(self):
        env = FakeEnv()
        result = rollout(env, ConstantPolicy(InvalidAction()), "state", "task", 3, 5, lambda e: None)
        self.assertEqual(result["status"], "invalid_action")
        self.assertEqual(len(env.actions), 5)
        self.assertEqual(result["control_steps"], 0)

    def test_api_error_is_not_retried(self):
        env, policy = FakeEnv(), ConstantPolicy(TransportError("fixture"))
        with self.assertRaises(TransportError):
            rollout(env, policy, "state", "task", 3, 5, lambda e: None)
        self.assertEqual(policy.calls, 1)
        self.assertEqual(len(env.actions), 5)

    def test_time_termination_does_not_count_as_success(self):
        with self.assertRaises(RuntimeError):
            rollout(FakeEnv(inconsistent_done=True), ConstantPolicy(), "state",
                    "task", 3, 5, lambda e: None)

    def test_policy_serializes_two_images_without_credentials(self):
        with tempfile.TemporaryDirectory(prefix="libero-test-") as temp:
            client, events = FakeClient(), []
            policy = GPTPolicy(client, Config(), CONTROLLER, events.append, Path(temp))
            action = policy.act(observation(), "task", 0)
            self.assertEqual(action[-1], -1)
            inputs = client.payloads[0]["input"][0]["content"]
            self.assertEqual(sum(p["type"] == "input_image" for p in inputs), 2)
            self.assertEqual(client.attempts, 1)
            self.assertNotIn("DO_NOT_SEND", json.dumps(client.payloads))

    def test_wrong_model_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory(prefix="libero-test-") as temp:
            policy = GPTPolicy(FakeClient(model="different-model"), Config(),
                               CONTROLLER, lambda e: None, Path(temp))
            with self.assertRaises(ModelIdentityError):
                policy.act(observation(), "task", 0)

    def test_budget_prevents_request_start(self):
        events = []
        with tempfile.TemporaryDirectory(prefix="libero-test-") as temp:
            client = FakeClient(budget=0)
            policy = GPTPolicy(client, Config(), CONTROLLER, events.append, Path(temp))
            with self.assertRaises(RequestLimitError):
                policy.act(observation(), "task", 0)
            self.assertEqual(events, [])
            self.assertEqual(client.attempts, 0)

    def test_config_rejects_invalid_budget_and_duplicate_tasks(self):
        for kwargs in ({"max_requests": 0}, {"max_requests": True},
                       {"task_ids": (0, 0)}, {"seed": -1},
                       {"suites": ("libero_90",), "task_order_index": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RunnerError):
                Config(**kwargs)


class ArchiveTests(unittest.TestCase):
    def archive(self, path, *, link=False):
        import io
        import tarfile
        prefix = "Lifelong-Robot-Learning-LIBERO-8f1084e"
        with tarfile.open(path, "w:gz") as stream:
            top = tarfile.TarInfo(prefix)
            top.type = tarfile.DIRTYPE
            stream.addfile(top)
            for name in SOURCE_FILES:
                item = tarfile.TarInfo(prefix + "/" + name)
                raw = b"official-source-fixture"
                item.size = len(raw)
                stream.addfile(item, io.BytesIO(raw))
            if link:
                item = tarfile.TarInfo(prefix + "/escape")
                item.type = tarfile.SYMTYPE
                item.linkname = "../../outside"
                stream.addfile(item)

    def test_archive_member_tampering_is_rejected(self):
        from LIBERORunner.scripts.stage_official import stage
        with tempfile.TemporaryDirectory() as directory:
            archive, root = Path(directory) / "source.tar.gz", Path(directory) / "official"
            self.archive(archive)
            with patch("LIBERORunner.scripts.stage_official.ARCHIVE_SHA256",
                       file_digest(archive)), patch("builtins.print"):
                stage(archive, root)
                self.assertEqual(upstream_identity(root)["source_type"], "verified_archive")
                (root / SOURCE_FILES[0]).write_text("modified")
                with self.assertRaises(RunnerError):
                    upstream_identity(root)

    def test_archive_link_rejected_before_extraction(self):
        from LIBERORunner.scripts.stage_official import stage
        with tempfile.TemporaryDirectory() as directory:
            archive, root = Path(directory) / "source.tar.gz", Path(directory) / "official"
            self.archive(archive, link=True)
            with patch("LIBERORunner.scripts.stage_official.ARCHIVE_SHA256", file_digest(archive)):
                with self.assertRaises(ValueError):
                    stage(archive, root)
            self.assertFalse(root.exists())


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="libero-runner-test-")
        self.directory = Path(self.temp.name) / "run"
        self.directory.mkdir()
        config = Config(suites=("libero_spatial",), episodes_per_task=2, max_steps=2)
        self.manifest = {
            "config": config.to_dict(), "runner_source": {},
            "upstream": {"commit": "fixture", "files": {}}, "dependencies": {},
            "evaluation_settings": {"max_steps": 2, "settling_steps": 5},
            "tasks": [{"suite": "libero_spatial", "task_id": 0, "language": "task", "files": {}}],
            "preflight": [{"controller": CONTROLLER}],
            "episodes": [
                {"episode_id": f"episode-{i}", "suite": "libero_spatial",
                 "task_id": 0, "initial_state_index": i} for i in range(2)
            ],
        }
        write_json(self.directory / "manifest.json", self.manifest)
        write_json(self.directory / "manifest.sha256.json", {"sha256": digest(self.manifest)})

    def tearDown(self):
        self.temp.cleanup()

    def result(self, index, status, ack=0):
        directory = self.directory / "episodes" / f"episode-{index}"
        journal = Journal(directory / "trace.jsonl")
        for i in range(ack):
            journal({"event": "action_ack", "step": i})
        write_json(directory / "result.json", {
            "status": status, "success": status == "success",
            "control_steps": ack, "manifest_sha256": digest(self.manifest),
        })

    def test_partial_summary_has_no_final_success_rate(self):
        self.result(0, "success", 1)
        summary = summarize(self.directory)
        self.assertFalse(summary["complete"])
        self.assertIsNone(summary["success_rate"])
        self.assertEqual(summary["per_suite"]["libero_spatial"]["success_rate_completed"], 1)
        self.assertFalse((self.directory / "final_collection.json").exists())

    def test_invalid_action_stays_in_completed_denominator(self):
        self.result(0, "success", 1)
        self.result(1, "invalid_action")
        summary = summarize(self.directory)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertTrue((self.directory / "final_collection.json").exists())

    def test_operational_error_prevents_final_collection(self):
        self.result(0, "success", 1)
        self.result(1, "operational_error")
        summary = summarize(self.directory)
        self.assertFalse(summary["complete"])
        self.assertIsNone(summary["success_rate"])

    def test_completed_collection_rejects_changed_results(self):
        self.result(0, "success", 1)
        self.result(1, "task_failure", 1)
        summarize(self.directory)
        self.result(1, "success", 1)
        with self.assertRaises(RunnerError):
            summarize(self.directory)

    def test_ack_mismatch_is_rejected(self):
        self.result(0, "success", 1)
        path = self.directory / "episodes/episode-0/trace.jsonl"
        path.write_text("")
        with self.assertRaises(RunnerError):
            summarize(self.directory)

    def test_manifest_tampering_is_rejected(self):
        self.manifest["config"]["model"] = "different"
        write_json(self.directory / "manifest.json", self.manifest)
        with self.assertRaises(RunnerError):
            summarize(self.directory)

    def test_live_loop_stops_after_one_service_failure(self):
        env = FakeEnv()
        runtime = SimpleNamespace(
            identity=self.manifest["upstream"], versions={},
            settings=self.manifest["evaluation_settings"],
            create_env=lambda task: env, initial_states=lambda suite, task_id: ["s0", "s1"],
        )
        client = FakeClient(error=TransportError("fixture"))
        with patch("LIBERORunner.runner.Runtime", return_value=runtime), \
             patch("LIBERORunner.runner.controller_spec", return_value=CONTROLLER), \
             patch("LIBERORunner.runner.XcodeClient", return_value=client), \
             patch("builtins.print"):
            summary = run(self.directory)
            self.assertFalse(summary["complete"])
            self.assertEqual(client.attempts, 1)
            self.assertTrue(env.closed)
            self.assertFalse((self.directory / "episodes/episode-1").exists())
            with self.assertRaises(RunnerError):
                run(self.directory, resume=True)
            self.assertEqual(client.attempts, 1)

    def test_prepare_probes_before_freezing_without_api(self):
        fresh = Path(self.temp.name) / "prepared"
        task = self.manifest["tasks"][0]
        runtime = SimpleNamespace(
            defaults={"episodes_per_task": 2, "max_steps": 2, "settling_steps": 5},
            settings={"episodes_per_task": 2, "max_steps": 2, "settling_steps": 5},
            identity=self.manifest["upstream"], versions={},
            tasks=lambda: [task],
            probe=lambda task: {"controller": CONTROLLER, "model_requests": 0},
        )
        with patch("LIBERORunner.runner.Runtime", return_value=runtime), \
             patch("LIBERORunner.runner.XcodeClient") as api:
            manifest = prepare(Config(suites=("libero_spatial",)), fresh)
            self.assertEqual(len(manifest["episodes"]), 2)
            self.assertEqual(manifest["model_requests"], 0)
            api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
