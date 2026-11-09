"""Offline unit, synchronization and public model-input checks; no SDK requests."""
import base64
import copy
import importlib
import io
import unittest

import numpy as np
from PIL import Image

from rgbd_three_envs.config import load_config
from rgbd_three_envs.depth import decode_raw, make_depth, to_meters, validate_depth
from rgbd_three_envs.planner import OfficialPlanner
from rgbd_three_envs.protocol import Task, validate_packet
from rgbd_three_envs.sources import ContractError, ROOT, Sources, canonical, digest
from rgbd_three_envs.transport import child_environment, command


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.tasks = load_config(ROOT / "rgbd_three_envs/experiment-20261107-r02.json")
        cls.sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
        cls.sources.verify_all()
        out = io.BytesIO()
        Image.fromarray(np.zeros((500, 500, 3), dtype=np.uint8)).save(out, format="PNG")
        cls.png = out.getvalue()

    def packet(self, task):
        kind = task.environment
        normalized = np.full((500, 500), 0.5, dtype=np.float32)
        if kind == "alfred":
            native, near, far = normalized * 4000, None, None
        elif kind == "habitat":
            native, near, far = normalized[..., None] * 4, None, None
        else:
            native, near, far = normalized, 0.01, 10.0
        depth = make_depth(native, digest(self.png), 0, kind, near, far)
        manip = None
        if kind == "manipulation":
            from rgbd_three_envs.protocol import COORD_SOURCE
            manip = {"objects": {"object 1": [10, 20, 30]},
                     "task_variation": "pick_cube_shape", "source": COORD_SOURCE}
        public = {"instruction": "Pick up the cube.", "images": [{
            "view": "front_rgb" if manip else "head_rgb", "base64": base64.b64encode(self.png).decode(),
            "sha256": digest(self.png), "source": "offline synthetic fixture"}],
            "actions": [] if manip else ["Find a cube", "Pick up the cube"],
            "step": 0, "env_feedback": "", "last_action_success": True,
            "manipulation": manip, "depth": depth}
        return {"task": task.as_dict(), "public": public,
                "official": {"done": False, "task_success": 0, "invalid_actions": 0, "reward": 0},
                "input_sha256": digest(canonical({"task": task.as_dict(), "instruction": public["instruction"]}))}

    def test_alfred_millimeters(self):
        native = np.full((500, 500), 2345, dtype=np.float32)
        np.testing.assert_allclose(to_meters(native, "alfred"), 2.345, atol=1e-6)

    def test_habitat_meters_and_channel(self):
        native = np.linspace(0, 10, 250000, dtype=np.float32).reshape(500, 500, 1)
        np.testing.assert_array_equal(to_meters(native, "habitat"), native[..., 0])

    def test_manipulation_clip_planes(self):
        native = np.tile(np.linspace(0, 1, 500, dtype=np.float32), (500, 1))
        result = to_meters(native, "manipulation", 0.1, 2.1)
        np.testing.assert_allclose(result[:, 0], 0.1)
        np.testing.assert_allclose(result[:, -1], 2.1)
        np.testing.assert_allclose(result[:, 250], 0.1 + 2 * native[:, 250])

    def test_invalid_native_units(self):
        for value, near, far in ((2.0, 0.1, 2), (0.5, 2, 1), (0.5, None, None)):
            with self.assertRaises(ContractError):
                to_meters(np.full((500, 500), value), "manipulation", near, far)

    def test_no_resize_or_camera_substitution(self):
        with self.assertRaises(ContractError):
            to_meters(np.ones((250, 250)), "habitat")
        for task in self.tasks:
            packet = self.packet(task)
            validate_packet(packet, task)
            packet["public"]["depth"]["image"]["view"] = "other_camera"
            with self.assertRaises(ContractError):
                validate_packet(packet, task)

    def test_same_observation_binding(self):
        for task in self.tasks:
            depth = self.packet(task)["public"]["depth"]
            validate_depth(depth, digest(self.png), 0, task.environment)
            for sha, step in ((digest(b"another rgb"), 0), (digest(self.png), 1)):
                with self.assertRaises(ContractError):
                    validate_depth(depth, sha, step, task.environment)

    def test_display_is_derived_from_raw(self):
        for task in self.tasks:
            packet = self.packet(task)
            raw = decode_raw(packet["public"]["depth"])
            self.assertEqual(raw.shape, (500, 500))
            edited = copy.deepcopy(packet["public"]["depth"])
            fake = io.BytesIO()
            Image.fromarray(np.zeros((500, 500), dtype=np.uint8)).save(fake, format="PNG")
            edited["image"]["base64"] = base64.b64encode(fake.getvalue()).decode()
            edited["image"]["sha256"] = digest(fake.getvalue())
            with self.assertRaises(ContractError):
                validate_depth(edited, digest(self.png), 0, task.environment)

    def test_cross_environment_provenance_rejected(self):
        for task in self.tasks:
            depth = self.packet(task)["public"]["depth"]
            depth["provenance"]["native_unit"] = "private_world_coordinates"
            with self.assertRaises(ContractError):
                validate_depth(depth, digest(self.png), 0, task.environment)

    def test_all_prompts_rgb_then_depth_and_official_prefix(self):
        prior = importlib.import_module("official_direct.planner").OfficialPlanner
        for task in self.tasks:
            public = self.packet(task)["public"]
            options = (self.sources, task, self.config["model"]["name"],
                       self.config["n_shots"][task.environment], public["actions"], "json_object")
            old, new = prior(*options), OfficialPlanner(*options)
            for feedback in (None, "The action executed.", "The action failed."):
                if feedback is not None:
                    action = [10, 20, 30, 0, 0, 0, 1] if task.environment == "manipulation" else 0
                    old.feedback(action, {"env_feedback": feedback})
                    new.feedback(action, {"env_feedback": feedback})
                messages = new.messages(public)
                previous = old.messages(public)
                # Only append new text and one image to the existing current user content.
                reduced = copy.deepcopy(messages)
                current = [m for m in reduced if m["role"] == "user"][-1]
                del current["content"][-3:]
                self.assertEqual(reduced, previous)
                images = [part["image_url"]["url"] for message in messages
                          for part in message["content"] if part["type"] == "image_url"]
                self.assertEqual(len(images), 2)
                self.assertEqual(images[0], "data:image/png;base64," + public["images"][0]["base64"])
                self.assertEqual(images[1], "data:image/png;base64," + public["depth"]["image"]["base64"])
                self.assertNotIn(public["depth"]["raw"]["base64"], canonical(messages).decode())

    def test_worker_path_and_credentials_boundary(self):
        for task in self.tasks:
            argv = command(self.config, task, probe=True)
            self.assertIn("/rgbd_three_envs/worker.py", argv[-1])
            self.assertNotIn("/rgbd_trial/worker.py", argv[-1])
        self.assertNotIn("DASHSCOPE_API_KEY", child_environment())
        self.assertNotIn("DASHSCOPE_BASE_URL", child_environment())

    def test_private_field_rejected(self):
        for task in self.tasks:
            packet = self.packet(task)
            packet["public"]["object_ground_truth"] = {}
            with self.assertRaises(ContractError):
                validate_packet(packet, task)


if __name__ == "__main__":
    unittest.main()
