"""Offline input-equivalence, geometry, budget and full-plan execution checks."""
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

from official_direct.planner import OfficialPlanner
from target_detection_r05.client import ModelClient
from target_detection_r05.depth import make_depth
from target_detection_r05.protocol import Task
from . import runner
from .config import load_config, validate_detection
from .detector import Detector, pixel_box
from .perception import Perception, PlannerError, boxes_image, crop_bounds, png_bytes
from .planner import IMAGE_PROMPT, TargetPlanner, extraction_messages, extract_targets
from .sources import ContractError, OperationalError, ROOT, Sources, canonical, digest

TARGETS = [{"id": "T1", "role": "object", "category": "star", "source_phrase": "star",
            "quantity": 1, "attributes": [], "relation": "", "aliases": ["star shaped block"]}]
SHOTS = {"alfred": 10, "habitat": 10, "manipulation": 4, "navigation": 3}


def public(size=(640, 360)):
    width, height = size
    pixels = np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3)
    png = png_bytes(Image.fromarray(pixels))
    return {"instruction": "Move towards the star.", "step": 0,
            "raw_rgb": {"base64": base64.b64encode(png).decode(), "sha256": digest(png)},
            "anonymous_anchors": []}


def planner_public(kind="navigation"):
    q = public((500, 500))
    q.update(images=[dict(q["raw_rgb"], view="head_rgb", source="official current RGB")],
             actions=[str(i) for i in range(8)], manipulation=None,
             last_action_success=True, env_feedback="")
    q["raw_rgb"].update(view="head_rgb", source="official same-observation unannotated RGB")
    native_depth = np.ones((500, 500), dtype=np.float32)
    q["depth"] = make_depth(native_depth, q["images"][0]["sha256"], 0, kind, near=0.1, far=10)
    if kind == "manipulation":
        q["actions"] = []
        q["manipulation"] = {"objects": {"object 0": [50, 50, 50]}, "task_variation": "pick_cube_shape"}
        # Official numbered RGB remains distinct from the raw detection view.
        numbered = png_bytes(Image.new("RGB", (500, 500), "#FFFFFF"))
        q["images"][0].update(base64=base64.b64encode(numbered).decode(), sha256=digest(numbered))
        q["depth"] = make_depth(native_depth, digest(numbered), 0, kind, near=0.1, far=10)
    return q


class MemoryJournal:
    def __init__(self):
        self.events = []

    def append(self, value):
        self.events.append(value)


def output(box=None):
    return {"status": "ok", "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            "output": {"detections": [{"label": "star", "bbox_2d": box or [100, 200, 900, 1000]}]}}


def action_reply(actions):
    return {"executable_plan": actions}


class Contracts(unittest.TestCase):
    def setUp(self):
        self.config, _ = load_config(Path(__file__).with_name("development-minimal.json"))
        self.temp = tempfile.TemporaryDirectory(prefix="qwen3vl-minimal-", dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)
        self.sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")

    def perception(self, q=None):
        p = Perception(self.config, self.temp.name, MemoryJournal())
        p.bind(q or public(), 0)
        p.set_targets(TARGETS)
        return p

    def prepare(self, p, detection=None):
        with patch.object(p.detector, "request", return_value=copy.deepcopy(detection or output())):
            return p.prepare()

    def test_separate_axes_float_coordinates_and_full_image_edge(self):
        self.assertEqual(pixel_box([100, 200, 900, 1000], (640, 360)), [64, 72, 576, 360])
        self.assertEqual(pixel_box([100, 200, 900, 1000], (360, 640)), [36, 128, 324, 640])
        self.assertEqual(pixel_box([0, 0, 1000, 1000], (640, 360)), [0, 0, 640, 360])
        self.assertEqual(pixel_box([716, 748, 788, 815], (500, 500)), [358, 374, 394, 407.5])

    def test_invalid_boxes_stop(self):
        for box in ([0, 0, 0, 100], [300, 0, 200, 100], [-1, 0, 10, 10],
                    [0, 0, 1001, 10], [False, 0, 10, 10], [float("nan"), 0, 10, 10],
                    [0, 0, float("inf"), 10], [1, 2, 3], ["1", 0, 10, 10]):
            with self.subTest(box=box), self.assertRaises(OperationalError):
                pixel_box(box, (640, 360))

    def test_decoder_uses_actual_png_and_raw_rgb_only(self):
        image = public()["raw_rgb"]
        image.update(width=500, height=500)
        detector = Detector(self.config)
        with patch.object(detector, "request", return_value=output()) as request:
            result = detector.detect(image, "star")
        self.assertEqual(result["image_size"], [640, 360])
        self.assertEqual(result["detections"][0]["bbox"], [64, 72, 576, 360])
        self.assertNotIn("confidence", result["detections"][0])
        self.assertFalse(result["confidence_available"])
        contents = request.call_args.args[1]["messages"][0]["content"]
        self.assertEqual(len([c for c in contents if c["type"] == "image_url"]), 1)
        self.assertEqual(contents[0]["image_url"]["url"], "data:image/png;base64," + image["base64"])
        image["sha256"] = "0" * 64
        with patch.object(detector, "request") as request:
            with self.assertRaises(OperationalError):
                detector.detect(image, "star")
            request.assert_not_called()

    def test_all_candidates_visible_beyond_twelve_with_labels_outside_rgb(self):
        groups = [{"id": f"G{i:03d}", "bbox": [i * 15, 20, i * 15 + 10, 30],
                   "members": [{"query": "star"}]} for i in range(25)]
        png, placements = boxes_image(base64.b64decode(public()["raw_rgb"]["base64"]), groups)
        canvas = Image.open(io.BytesIO(png))
        self.assertEqual(len(placements), 25)
        self.assertEqual(canvas.height, 360)
        self.assertGreater(canvas.width, 640)
        for placement in placements:
            self.assertEqual(placement["category"], "star")
            self.assertGreaterEqual(placement["label_rect"][0], 640)
            self.assertLess(placement["label_rect"][3], canvas.height)
            self.assertEqual(placement["source_image_rect"], [0, 0, 640, 360])
        raw = Image.open(io.BytesIO(base64.b64decode(public()["raw_rgb"]["base64"])))
        self.assertEqual(canvas.crop((0, 100, 640, 360)).tobytes(), raw.crop((0, 100, 640, 360)).tobytes())

    def test_empty_detections_still_supply_same_view_rgb_without_crops(self):
        p = self.perception()
        _, images = self.prepare(p, {"status": "ok", "output": {"detections": []}})
        self.assertTrue(p.detected)
        self.assertEqual(p.groups, [])
        self.assertEqual(p.crop_count, 0)
        self.assertEqual(p.context_images(), images)
        actual = Image.open(io.BytesIO(base64.b64decode(images[0]["base64"])))
        raw = Image.open(io.BytesIO(base64.b64decode(p.public["raw_rgb"]["base64"])))
        self.assertEqual(actual.size, raw.size)
        self.assertEqual(actual.tobytes(), raw.tobytes())

    def test_current_frame_cache_and_new_observation_expire_overlay(self):
        p = self.perception()
        with patch.object(p.detector, "request", side_effect=lambda *a: output()) as request:
            p.prepare()
            p.prepare()
            self.assertEqual(request.call_count, 1)
            p.bind(public(), 1)
            self.assertIsNone(p.overlay)
            self.assertEqual(p.context_images(), [])
            self.assertEqual(p.targets, TARGETS)
            p.prepare()
            self.assertEqual(request.call_count, 2)
            self.assertEqual(p.pipeline_preparations, 2)
            self.assertEqual(p.crop_count, 0)

    def test_invalid_alias_is_rejected_before_any_request(self):
        p = self.perception()
        with patch.object(p.detector, "request") as request:
            with self.assertRaises(PlannerError):
                p.prepare({"name": "perceive_targets", "target_ids": ["T1"],
                           "query_aliases": {"T1": "hidden catalog"}})
            request.assert_not_called()

    def test_standalone_crop_geometry_retains_original_pixels(self):
        p = self.perception()
        self.prepare(p, output([800, 800, 1000, 1000]))
        result, images = p.tool({"name": "inspect_candidates", "group_ids": [p.groups[0]["id"]]})
        bounds = result["crops"][0]["bounds"]
        self.assertEqual(bounds, [496, 272, 640, 360])
        raw = Image.open(io.BytesIO(base64.b64decode(p.public["raw_rgb"]["base64"])))
        crop = Image.open(io.BytesIO(base64.b64decode(images[0]["base64"])))
        self.assertEqual(crop.tobytes(), raw.crop(bounds).tobytes())
        self.assertEqual(crop_bounds([10.1, 20.9, 11.2, 22.3], raw.size), [2, 12, 20, 31])
        self.assertEqual(len(p.context_images()), 1)

    def test_official_messages_and_feedback_exactly_preserved_in_four_environments(self):
        for kind, shots in SHOTS.items():
            with self.subTest(environment=kind):
                q = planner_public(kind)
                p = self.perception(q)
                self.prepare(p)
                task = Task(kind, "base", 0)
                args = (self.sources, task, "qwen3.8-flash", shots, q["actions"], "json_object")
                planner, official = TargetPlanner(*args), OfficialPlanner(*args)
                self.assertEqual(planner.adapter.n_shot, shots)
                for feedback in ("", "Previous action failed: target is too far away."):
                    if feedback:
                        q["env_feedback"] = feedback
                        action = [50, 50, 50, 0, 0, 0, 1] if kind == "manipulation" else 0
                        planner.feedback(action, q)
                        official.feedback(action, q)
                    expected = official.messages(q)
                    actual = planner.messages(q, p)
                    self.assertEqual(actual[0], {"role": "system", "content": IMAGE_PROMPT})
                    prefix = copy.deepcopy(actual[1:])
                    current = [m for m in prefix if m["role"] == "user"][-1]["content"]
                    additions = current[-2:]
                    del current[-2:]
                    self.assertEqual(prefix, expected)
                    self.assertEqual([i["type"] for i in additions], ["image_url", "image_url"])
                    self.assertEqual(additions[0]["image_url"]["url"],
                                     "data:image/png;base64," + q["depth"]["image"]["base64"])
                    self.assertEqual(additions[1]["image_url"]["url"],
                                     "data:image/png;base64," + p.overlay["base64"])
                self.assertEqual(IMAGE_PROMPT.count("Detection boxes may be incorrect."), 1)
                for obsolete in ("tool_request", "targets", "perception_review", "selected_targets",
                                 "CURRENT PHASE", "G IDs", "ONE object", "CORRECTION"):
                    self.assertNotIn(obsolete, IMAGE_PROMPT)
                self.assertLess(len(IMAGE_PROMPT), 250)

    def test_native_plans_require_no_custom_description_or_review_fields(self):
        for kind in SHOTS:
            with self.subTest(environment=kind):
                q = planner_public(kind)
                planner = TargetPlanner(self.sources, Task(kind, "base", 0), "qwen3.8-flash",
                                        SHOTS[kind], q["actions"], "json_object")
                actions = ([{"action": [50, 50, 50, 0, 0, 0, 1]},
                            {"action": [60, 50, 50, 0, 0, 0, 0]}] if kind == "manipulation"
                           else [{"action_id": 0}, {"action_id": 1}])
                plan = planner.decision(json.dumps(action_reply(actions)))
                self.assertEqual(plan.status, "valid")
                self.assertEqual(plan.proposed_count, 2)

    def test_explicit_integer_and_pair_actions_preserve_full_plan(self):
        q = planner_public()
        planner = TargetPlanner(self.sources, Task("navigation", "base", 0), "qwen3.8-flash", 3,
                                q["actions"], "json_object")
        plan = planner.decision(json.dumps(action_reply([0, [1, "Forward"]])))
        self.assertEqual(plan.actions, [0, 1])
        for reply in ('not JSON', '{"tool_request":{"name":"perceive_targets"}}',
                      '{"executable_plan":[{"action_id":99}]}',
                      '{"executable_plan":[true]}'):
            self.assertEqual(planner.decision(reply).status, "invalid")

    def test_preprocessing_is_public_text_only_and_targets_are_frozen(self):
        p = self.perception()
        messages = extraction_messages(p.public["instruction"])
        self.assertEqual(messages[-1], {"role": "user", "content": p.public["instruction"]})
        self.assertEqual(len(messages), 2)
        payload = ModelClient(self.config["model"]).payload(messages, None)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        for forbidden in ("image_url", "object 0", "anonymous_anchors", "task_success", "reference_answer"):
            self.assertNotIn(forbidden, json.dumps(messages))
        extract_targets(json.dumps({"targets": TARGETS}), p)
        bad = copy.deepcopy(TARGETS)
        bad[0]["category"] = "lamp"
        with self.assertRaises(PlannerError):
            extract_targets(json.dumps({"targets": bad}), p)
        bad[0]["source_phrase"] = "hidden private label"
        fresh = Perception(self.config, self.temp.name, MemoryJournal())
        fresh.bind(public(), 0)
        with self.assertRaises(PlannerError):
            extract_targets(json.dumps({"targets": bad}), fresh)
        with self.assertRaises(PlannerError):
            extract_targets(json.dumps({"targets": TARGETS, "executable_plan": []}), fresh)

    def offline_episode(self, replies, *, max_requests, detector_output=None, detector_error=None):
        task, q, operations = Task("navigation", "base", 0), planner_public(), []

        class OfflineSession:
            step = 0

            def request(self, operation, **kw):
                operations.append((operation, kw))
                if operation == "step":
                    self.step += 1
                current = copy.deepcopy(q)
                current["step"] = self.step
                current["depth"] = make_depth(np.ones((500, 500), dtype=np.float32),
                                              current["images"][0]["sha256"], self.step, "navigation")
                return {"task": task.as_dict(), "public": current,
                        "official": {"done": self.step == 2, "task_success": int(self.step == 2),
                                     "reward": 0., "invalid_actions": 0},
                        "input_sha256": digest(canonical({
                            "task": task.as_dict(), "instruction": current["instruction"]}))}

            def close(self):
                operations.append(("close", {}))

        replies, payloads = iter(replies), []
        model = ModelClient(self.config["model"])

        def payload(messages, schema):
            value = model.payload(messages, schema)
            payloads.append(value)
            return value

        def complete(value):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return {"text": json.dumps(reply),
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}

        client = SimpleNamespace(stopped=threading.Event(), payload=payload, complete=complete)
        config = copy.deepcopy(self.config)
        config["max_requests"]["navigation"] = max_requests
        folder = Path(self.temp.name) / "episode"
        def detect(*args):
            if detector_error:
                raise detector_error
            return copy.deepcopy(detector_output or output())
        with patch.object(Detector, "request", autospec=True, side_effect=detect) as request:
            result = runner.episode(task, config, self.sources, client, lambda *a: OfflineSession(), folder)
        trace = [json.loads(line) for line in (folder / "trajectory.jsonl").read_text().splitlines()]
        return result, operations, payloads, request.call_count, trace

    def test_two_requests_include_preprocessing_and_execute_entire_native_plan_with_acks(self):
        result, operations, payloads, count, trace = self.offline_episode(
            [{"targets": TARGETS}, action_reply([{"action_id": 0}, {"action_id": 1}])], max_requests=2)
        self.assertEqual(result["status"], "official_done")
        self.assertEqual(result["metrics"]["planner_steps"], 1)
        self.assertEqual(result["request_accounting"]["total_model_requests"], 2)
        self.assertEqual(result["request_accounting"]["target_extraction"], 1)
        self.assertEqual(result["request_accounting"]["planning"], 1)
        self.assertEqual(result["request_accounting"]["token_usage"]["target_extraction"]["total_tokens"], 110)
        self.assertEqual(count, 1)
        self.assertEqual(result["perception"]["crop_count"], 0)
        self.assertEqual(result["perception"]["token_usage"]["total_tokens"], 25)
        self.assertEqual([kw["action"] for op, kw in operations if op == "step"], [0, 1])
        self.assertEqual(len([e for e in trace if e["type"] == "action_ack"]), 2)
        self.assertEqual([e["stage"] for e in trace if e["type"] == "model_request"],
                         ["target_extraction", "planning"])
        self.assertEqual(len([c for c in payloads[1]["messages"][-1]["content"]
                              if c["type"] == "image_url"]), 3)

    def test_later_observation_detects_before_single_planning_request(self):
        result, operations, payloads, count, trace = self.offline_episode(
            [{"targets": TARGETS}, action_reply([{"action_id": 0}]),
             action_reply([{"action_id": 1}])], max_requests=3)
        self.assertEqual(result["status"], "official_done")
        self.assertEqual(result["request_accounting"]["target_extraction"], 1)
        self.assertEqual(result["request_accounting"]["planning"], 2)
        self.assertEqual(count, 2)
        self.assertEqual(result["perception"]["pipeline_preparations"], 2)
        self.assertFalse(any(op == "observe" for op, kw in operations))
        self.assertEqual(len(payloads), 3)
        ordered = [e["type"] for e in trace if e["type"] in ("perception_pipeline_prepared", "model_request")]
        self.assertEqual(ordered, ["model_request", "perception_pipeline_prepared", "model_request",
                                   "perception_pipeline_prepared", "model_request"])

    def test_budget_exhausted_after_extraction_sends_no_detector_or_action(self):
        result, operations, payloads, count, trace = self.offline_episode(
            [{"targets": TARGETS}], max_requests=1)
        self.assertEqual(result["status"], "request_budget")
        self.assertEqual(result["request_accounting"]["planning"], 0)
        self.assertEqual(count, 0)
        self.assertEqual(len(payloads), 1)
        self.assertFalse(any(op == "step" for op, kw in operations))

    def test_extraction_failure_stops_without_retry_or_detection(self):
        for response in ({}, OperationalError("model_http_error", 503)):
            with self.subTest(response=type(response).__name__):
                # Each scenario gets a separate episode directory.
                with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
                    old, self.temp = self.temp, SimpleNamespace(name=temp)
                    try:
                        result, operations, payloads, count, trace = self.offline_episode(
                            [response], max_requests=4)
                    finally:
                        self.temp = old
                self.assertIn(result["status"], ("planner_error", "operational_error"))
                self.assertEqual(len(payloads), 1)
                self.assertEqual(count, 0)
                self.assertFalse(any(op == "step" for op, kw in operations))

    def test_detector_failure_stops_without_retry_or_planning(self):
        result, operations, payloads, count, trace = self.offline_episode(
            [{"targets": TARGETS}], max_requests=4, detector_error=OperationalError("detector_failure"))
        self.assertEqual(result["status"], "operational_error")
        self.assertEqual(result["perception"]["detector_requests"], 1)
        self.assertEqual(count, 1)
        self.assertEqual(len(payloads), 1)
        self.assertFalse(any(op == "step" for op, kw in operations))

    def test_planning_service_failure_stops_without_retry_or_action(self):
        result, operations, payloads, count, trace = self.offline_episode(
            [{"targets": TARGETS}, OperationalError("model_timeout")], max_requests=4)
        self.assertEqual(result["status"], "operational_error")
        self.assertEqual(len(payloads), 2)
        self.assertEqual(count, 1)
        self.assertEqual(result["request_accounting"]["planning"], 1)
        self.assertFalse(any(op == "step" for op, kw in operations))

    def test_config_preserves_official_shots_and_historical_config_values(self):
        self.assertEqual(self.config["n_shots"], SHOTS)
        path = Path(__file__).with_name("experiment.json")
        config, _ = load_config(path)
        self.assertEqual(config["n_shots"], json.loads(path.read_text())["n_shots"])
        compact = json.loads(Path(__file__).with_name("development-compact.json").read_text())
        restored = dict(self.config, n_shots=compact["n_shots"])
        self.assertEqual(restored, compact)
        with self.assertRaises(ContractError):
            validate_detection(dict(self.config["detection"], box_threshold=0.15))

    def test_suite_totals_separate_extraction_planning_and_detector_cost(self):
        task = Task("navigation", "base", 0)
        fake_client = SimpleNamespace()
        fake_result = {
            "task": task.as_dict(), "status": "official_done",
            "official": {"done": True, "task_success": 1},
            "request_accounting": {"target_extraction": 1, "planning": 2,
                                   "token_usage": {"target_extraction": {"total_tokens": 110},
                                                  "planning": {"total_tokens": 220}}},
            "perception": {"pipeline_preparations": 2, "detector_requests": 2,
                           "token_usage": {"total_tokens": 50}}}
        def episode(*args):
            gate = args[3]
            gate.requests = 3
            return fake_result
        with patch.object(runner, "episode", side_effect=episode):
            folder = Path(self.temp.name) / "suite"
            summary = runner.suite([task], self.config, self.sources, fake_client, None, folder)
        self.assertEqual(summary["http_requests"], 3)
        self.assertEqual(summary["target_extraction_requests"], 1)
        self.assertEqual(summary["planning_requests"], 2)
        self.assertEqual(summary["detector_requests"], 2)
        self.assertEqual(summary["model_tool_calls"], 0)
        self.assertEqual(summary["token_usage"]["target_extraction"]["total_tokens"], 110)
        self.assertEqual(summary["token_usage"]["planning"]["total_tokens"], 220)
        self.assertEqual(summary["detector_token_usage"]["total_tokens"], 50)
        self.assertEqual(json.loads((folder / "config.json").read_text())["n_shots"], SHOTS)


if __name__ == "__main__":
    unittest.main()
