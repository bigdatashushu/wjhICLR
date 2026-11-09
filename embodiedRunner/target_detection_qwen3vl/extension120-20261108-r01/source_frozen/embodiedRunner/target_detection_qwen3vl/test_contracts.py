"""Offline checks for normalized boxes and the actual planner-to-action integration."""
import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from target_detection_r05.depth import make_depth
from target_detection_r05.perception import PlannerError
from target_detection_r05.protocol import Task
from . import runner
from .config import load_config, validate_detection
from .detector import Detector, pixel_box
from .perception import Perception, crop_bounds, page_image, png_bytes
from .sources import ContractError, OperationalError, ROOT, Sources, canonical, digest

TARGETS = [
    {"id": "T1", "role": "object", "category": "star", "source_phrase": "star",
     "quantity": 1, "attributes": [], "relation": "", "aliases": ["star shaped block"]}]


def public(size=(640, 360)):
    width, height = size
    pixels = np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3)
    png = png_bytes(Image.fromarray(pixels))
    return {"instruction": "Move towards the star.", "step": 0,
            "raw_rgb": {"base64": base64.b64encode(png).decode(), "sha256": digest(png)},
            "anonymous_anchors": []}


class MemoryJournal:
    def __init__(self):
        self.events = []

    def append(self, value):
        self.events.append(value)


def output(box=None):
    return {"status": "ok", "output": {"detections": [
        {"label": "star", "bbox_2d": box or [100, 200, 900, 1000]}]}}


class Contracts(unittest.TestCase):
    def setUp(self):
        self.config, _ = load_config(Path(__file__).with_name("experiment.json"))
        self.temp = tempfile.TemporaryDirectory(prefix="qwen3vl-contracts-", dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)

    def test_separate_axes_and_full_image_edge(self):
        self.assertEqual(pixel_box([100, 200, 900, 1000], (640, 360)), [64, 72, 576, 360])
        self.assertEqual(pixel_box([100, 200, 900, 1000], (360, 640)), [36, 128, 324, 640])
        self.assertEqual(pixel_box([0, 0, 1000, 1000], (640, 360)), [0, 0, 640, 360])
        self.assertEqual(pixel_box([716, 748, 788, 815], (500, 500)), [358, 374, 394, 407.5])

    def test_invalid_boxes_stop_instead_of_becoming_crops(self):
        for box in ([0, 0, 0, 100], [300, 0, 200, 100], [0, 200, 100, 100],
                    [-1, 0, 10, 10], [0, 0, 1001, 10], [False, 0, 10, 10],
                    [float("nan"), 0, 10, 10], [0, 0, float("inf"), 10], [1, 2, 3],
                    ["1", 0, 10, 10]):
            with self.subTest(box=box), self.assertRaises(OperationalError):
                pixel_box(box, (640, 360))

    def test_decoder_uses_actual_png_not_assumed_metadata(self):
        image = public()["raw_rgb"]
        image.update(width=500, height=500)
        detector = Detector(self.config)
        with patch.object(detector, "request", return_value=output()) as request:
            result = detector.detect(image, "star")
        self.assertEqual(result["image_size"], [640, 360])
        self.assertEqual(result["detections"][0]["bbox"], [64, 72, 576, 360])
        self.assertNotIn("confidence", result["detections"][0])
        self.assertFalse(result["confidence_available"])
        payload = request.call_args.args[1]
        contents = payload["messages"][0]["content"]
        self.assertEqual(len([c for c in contents if c["type"] == "image_url"]), 1)
        self.assertEqual(contents[0]["image_url"]["url"], "data:image/png;base64," + image["base64"])
        self.assertIn("NORMALIZED 0-1000", contents[1]["text"])
        image["sha256"] = "0" * 64
        with patch.object(detector, "request") as request:
            with self.assertRaises(OperationalError):
                detector.detect(image, "star")
            request.assert_not_called()

    def test_non_square_overlay_and_crop_pixels(self):
        journal = MemoryJournal()
        p = Perception(self.config, self.temp.name, journal)
        q = public()
        p.bind(q, 0)
        p.set_targets(TARGETS)
        with patch.object(p.detector, "request", return_value=output([800, 800, 1000, 1000])):
            result, images = p.tool({"name": "detect_targets", "target_ids": ["T1"]})
        self.assertEqual(p.groups[0]["bbox"], [512, 288, 640, 360])
        placement = result["label_placements"][0]
        self.assertEqual(placement["source_image_rect"], [0, 0, 640, 360])
        self.assertGreaterEqual(placement["label_rect"][0], 640)
        canvas = Image.open(io.BytesIO(base64.b64decode(images[0]["base64"])))
        self.assertEqual(canvas.size, (950, 500))
        crop_result, crops = p.tool({"name": "inspect_candidates", "group_ids": ["G001"]})
        bounds = crop_result["crops"][0]["bounds"]
        self.assertEqual(bounds, [496, 272, 640, 360])
        raw = Image.open(io.BytesIO(base64.b64decode(q["raw_rgb"]["base64"])))
        crop = Image.open(io.BytesIO(base64.b64decode(crops[0]["base64"])))
        self.assertEqual(crop.tobytes(), raw.crop(bounds).tobytes())
        self.assertEqual(crop.size, (144, 88))
        self.assertEqual(crop_bounds([10.1, 20.9, 11.2, 22.3], (640, 360)), [2, 12, 20, 31])
        png, _ = page_image(base64.b64decode(q["raw_rgb"]["base64"]), [], 0)
        self.assertEqual(Image.open(io.BytesIO(png)).size, (950, 500))

    def test_current_frame_cache_and_expired_review(self):
        p = Perception(self.config, self.temp.name, MemoryJournal())
        p.bind(public(), 0)
        p.set_targets(TARGETS)
        with patch.object(p.detector, "request", side_effect=lambda *a: output()) as request:
            p.tool({"name": "detect_targets", "target_ids": ["T1"]})
            p.tool({"name": "detect_targets", "target_ids": ["T1"]})
            self.assertEqual(request.call_count, 1)
            p.tool({"name": "inspect_candidates", "group_ids": ["G001"]})
            p.tool({"name": "review_candidates", "perception_review": [
                {"group_id": "G001", "verdict": "match", "reason": "offline fixture"}],
                "selected_targets": [{"target_id": "T1", "group_id": "G001"}]})
            p.bind(public(), 1)
            self.assertIsNone(p.completed_review)
            p.tool({"name": "detect_targets", "target_ids": ["T1"]})
            self.assertEqual(request.call_count, 2)
            self.assertEqual(p.groups[0]["id"], "G002")
            with self.assertRaises(PlannerError):
                p.tool({"name": "inspect_candidates", "group_ids": ["G001"]})

    def test_invalid_alias_is_rejected_before_any_request(self):
        p = Perception(self.config, self.temp.name, MemoryJournal())
        p.bind(public(), 0)
        p.set_targets(TARGETS)
        with patch.object(p.detector, "request") as request:
            with self.assertRaises(PlannerError):
                p.tool({"name": "detect_targets", "target_ids": ["T1"],
                        "query_aliases": {"T1": "hidden catalog"}})
            request.assert_not_called()

    def test_detector_failure_is_not_retried(self):
        detector = Detector(self.config)
        with patch.object(detector, "request", side_effect=OperationalError("offline_failure")) as request:
            with self.assertRaises(OperationalError):
                detector.detect(public()["raw_rgb"], "star")
            self.assertEqual(request.call_count, 1)

    def test_real_planner_tool_loop_reaches_full_native_plan(self):
        task = Task("navigation", "base", 0)
        q = public((500, 500))
        q.update(images=[dict(q["raw_rgb"], view="head_rgb", source="official current RGB")],
                 actions=[str(i) for i in range(8)], manipulation=None,
                 last_action_success=True, env_feedback="")
        q["raw_rgb"].update(view="head_rgb", source="official same-observation unannotated RGB")
        operations = []
        class OfflineSession:
            step = 0
            def request(self, operation, **kw):
                operations.append((operation, kw))
                if operation == "step":
                    self.step += 1
                p = copy.deepcopy(q)
                p["step"] = self.step
                p["depth"] = make_depth(np.ones((500, 500), dtype=np.float32),
                                        p["images"][0]["sha256"], self.step, "navigation")
                return {"task": task.as_dict(), "public": p,
                        "official": {"done": self.step == 2, "task_success": int(self.step == 2),
                                     "reward": 0., "invalid_actions": 0},
                        "input_sha256": digest(canonical({
                            "task": task.as_dict(), "instruction": p["instruction"]}))}
            def close(self):
                operations.append(("close", {}))
        replies = iter([
            {"targets": TARGETS, "tool_request": {"name": "detect_targets", "target_ids": ["T1"]}},
            {"tool_request": {"name": "inspect_candidates", "group_ids": ["G001"]}},
            {"tool_request": {"name": "review_candidates", "perception_review": [
                {"group_id": "G001", "verdict": "match", "reason": "offline fixture"}],
                "selected_targets": [{"target_id": "T1", "group_id": "G001"}]}},
            {"executable_plan": [0, 1, 2]},
        ])
        stages = []
        def payload(messages, schema):
            stages.append(messages[0]["content"].split("CURRENT PHASE: ")[-1].split(".")[0])
            return {"messages": messages}
        client = SimpleNamespace(stopped=threading.Event(), payload=payload,
                                 complete=lambda value: {"text": json.dumps(next(replies))})
        config = copy.deepcopy(self.config)
        config["n_shots"]["navigation"] = 0
        config["max_requests"]["navigation"] = 4
        sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
        with patch.object(Detector, "request", autospec=True, side_effect=lambda *a: output()) as request:
            result = runner.episode(task, config, sources, client,
                                    lambda *a: OfflineSession(), Path(self.temp.name) / "episode")
        self.assertEqual(result["status"], "official_done", result["error"])
        self.assertEqual(result["perception"]["detector_requests"], 1)
        self.assertEqual(result["metrics"]["planner_steps"], 4)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(stages, ["EXTRACT TARGETS AND DETECT", "INSPECT CROPS", "REVIEW CROPS", "FULL ACTION PLAN"])
        self.assertEqual([op for op, _ in operations], ["open", "step", "step", "close"])
        self.assertEqual([kw["action"] for op, kw in operations if op == "step"], [0, 1])
        self.assertEqual([kw["first_in_plan"] for op, kw in operations if op == "step"], [True, False])

    def test_config_has_no_dino_threshold_or_credential_fields(self):
        self.assertNotIn("box_threshold", self.config["detection"])
        bad = copy.deepcopy(self.config["detection"])
        bad["service"]["base_url"] = "http://user:secret@10.10.0.198:28101/v1"
        with self.assertRaises(ContractError):
            validate_detection(bad)


if __name__ == "__main__":
    unittest.main()
