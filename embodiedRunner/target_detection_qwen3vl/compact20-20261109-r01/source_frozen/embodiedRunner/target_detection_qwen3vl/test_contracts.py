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
from .planner import ALFRED_HAND_PROMPT, TARGET_TOOL_PROMPT, TargetPlanner
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


def action_reply(actions, group_id="G001"):
    return {
        "visual_state_description": "The star is visible.",
        "reasoning_and_reflection": "The current crop supports the target; use legal actions.",
        "language_plan": "Move towards the star.",
        "perception_review": ([{"group_id": group_id, "verdict": "match",
                               "reason": "offline visible star fixture"}] if group_id else []),
        "selected_targets": ([{"target_id": "T1", "group_id": group_id}] if group_id else []),
        "executable_plan": actions,
    }


def planner_public(kind="navigation"):
    q = public((500, 500))
    q.update(images=[dict(q["raw_rgb"], view="head_rgb", source="official current RGB")],
             actions=[str(i) for i in range(8)], manipulation=None,
             last_action_success=True, env_feedback="")
    q["raw_rgb"].update(view="head_rgb", source="official same-observation unannotated RGB")
    q["depth"] = make_depth(np.ones((500, 500), dtype=np.float32),
                            q["images"][0]["sha256"], 0, kind, near=0.1, far=10)
    if kind == "manipulation":
        q["manipulation"] = {"objects": {"object 0": [50, 50, 50]}, "task_variation": "pick_cube_shape"}
    return q


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

    def offline_episode(self, replies, *, max_requests, detector_output=None):
        task = Task("navigation", "base", 0)
        q = planner_public()
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
        replies, payloads = iter(replies), []
        def payload(messages, schema):
            payloads.append(messages)
            return {"messages": messages}
        client = SimpleNamespace(stopped=threading.Event(), payload=payload,
                                 complete=lambda value: {"text": json.dumps(next(replies))})
        config = copy.deepcopy(self.config)
        config["n_shots"]["navigation"] = 0
        config["max_requests"]["navigation"] = max_requests
        sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
        with patch.object(Detector, "request", autospec=True,
                          side_effect=lambda *a: detector_output or output()) as request:
            result = runner.episode(task, config, sources, client,
                                    lambda *a: OfflineSession(), Path(self.temp.name) / "episode")
        return result, operations, payloads, request.call_count

    def test_real_planner_tool_loop_reaches_full_native_plan_in_two_requests(self):
        result, operations, payloads, detector_requests = self.offline_episode([
            {"targets": TARGETS, "tool_request": {"name": "perceive_targets", "target_ids": ["T1"]}},
            action_reply([0, 1, 2]),
        ], max_requests=2)
        self.assertEqual(result["status"], "official_done", result["error"])
        self.assertEqual(result["perception"]["detector_requests"], 1)
        self.assertEqual(result["metrics"]["planner_steps"], 2)
        self.assertEqual(detector_requests, 1)
        stages = [m[0]["content"].split("CURRENT PHASE: ")[-1].split(".")[0] for m in payloads]
        self.assertEqual(stages, ["EXTRACT TARGETS AND PREPARE", "REVIEW AND PLAN"])
        self.assertEqual([op for op, _ in operations], ["open", "step", "step", "close"])
        self.assertEqual([kw["action"] for op, kw in operations if op == "step"], [0, 1])
        self.assertEqual([kw["first_in_plan"] for op, kw in operations if op == "step"], [True, False])

    def test_later_observation_prepares_before_single_planning_request(self):
        result, operations, payloads, detector_requests = self.offline_episode([
            {"targets": TARGETS, "tool_request": {"name": "perceive_targets", "target_ids": ["T1"]}},
            action_reply([0]),
            action_reply([1], "G002"),
        ], max_requests=3)
        self.assertEqual(result["status"], "official_done", result["error"])
        self.assertEqual(result["metrics"]["planner_steps"], 3)
        self.assertEqual(detector_requests, 2)
        current = payloads[-1][-1]["content"]
        summary = next(json.loads(c["text"]) for c in current if c["type"] == "text"
                       and c["text"].startswith('{"current_observation_serial"'))
        self.assertEqual(summary["current_crop_group_ids"], ["G002"])
        self.assertEqual(summary["environment_step"], 1)
        self.assertNotIn("G001", json.dumps(summary))
        self.assertEqual([kw["action"] for op, kw in operations if op == "step"], [0, 1])

    def test_empty_detections_allow_review_and_search_in_same_response(self):
        result, _, _, requests = self.offline_episode([
            {"targets": TARGETS, "tool_request": {"name": "perceive_targets", "target_ids": ["T1"]}},
            action_reply([0, 1], None),
        ], max_requests=2, detector_output={"status": "ok", "output": {"detections": []}})
        self.assertEqual(result["status"], "official_done", result["error"])
        self.assertEqual(requests, 1)
        self.assertEqual(result["perception"]["crop_count"], 0)

    def test_missing_reasoning_stops_before_any_action(self):
        reply = action_reply([0, 1])
        del reply["reasoning_and_reflection"]
        result, operations, _, _ = self.offline_episode([
            {"targets": TARGETS, "tool_request": {"name": "perceive_targets", "target_ids": ["T1"]}},
            reply,
        ], max_requests=2)
        self.assertEqual(result["status"], "planner_error")
        self.assertEqual(result["error"]["code"], "action_requires_reasoning_and_reflection")
        self.assertNotIn("step", [op for op, _ in operations])

    def test_current_context_has_one_protocol_and_no_tool_transcript(self):
        q = planner_public()
        p = Perception(self.config, self.temp.name, MemoryJournal())
        p.bind(q, 0)
        p.set_targets(TARGETS)
        planner = TargetPlanner(Sources(), Task("navigation", "base", 0), "qwen3.8-flash", 0, q["actions"])
        with patch.object(Detector, "request", return_value=output()):
            p.prepare()
            # Repeated crops must replace, not append images or allow stale reviews.
            for _ in range(3):
                p.tool({"name": "inspect_candidates", "group_ids": ["G001"]})
        p.turns = [{"role": "assistant", "content": "old tool transcript sentinel"}] * 40
        messages = planner.messages(q, p)
        serialized = json.dumps(messages)
        self.assertNotIn("old tool transcript sentinel", serialized)
        texts = [m["content"] if isinstance(m["content"], str) else
                 "\n".join(c.get("text", "") for c in m["content"]) for m in messages]
        self.assertEqual(sum(t.count(TARGET_TOOL_PROMPT) for t in texts), 1)
        self.assertTrue(messages[0]["content"].startswith(TARGET_TOOL_PROMPT))
        self.assertEqual(len(p.context_images()), 2)
        self.assertEqual(sum(c["type"] == "image_url" for m in messages if isinstance(m["content"], list)
                             for c in m["content"]), 4)  # RGB, depth, overlay, crop
        self.assertTrue(any(e["type"] == "perception_pipeline_prepared" for e in p.journal.events))

    def test_extra_crop_replaces_reviewable_set_and_bind_expires_images(self):
        q = planner_public()
        p = Perception(self.config, self.temp.name, MemoryJournal())
        p.bind(q, 0)
        p.set_targets(TARGETS)
        detections = {"status": "ok", "output": {"detections": [
            {"label": "star", "bbox_2d": [10 + i * 200, 100, 110 + i * 200, 200]}
            for i in range(4)]}}
        with patch.object(Detector, "request", return_value=detections):
            p.prepare()
            self.assertEqual(p.inspected, {"G001", "G002"})
            self.assertEqual(len(p.groups), 4)
            p.tool({"name": "inspect_candidates", "group_ids": ["G003"]})
            self.assertEqual(p.inspected, {"G003"})
            self.assertEqual(len(p.context_images()), 2)
            with self.assertRaises(PlannerError):
                p.approve_action(action_reply([0], "G001"))
            p.approve_action(action_reply([0], "G003"))
            p.bind(q, 1)
            self.assertEqual(p.context_images(), [])
            self.assertEqual(p.summary()["candidates"], [])
            self.assertFalse(p.detected)

    def test_automatic_crops_cover_distinct_targets_and_bound_candidate_summary(self):
        q = planner_public()
        q["instruction"] = "Move the star into a container."
        p = Perception(self.config, self.temp.name, MemoryJournal())
        p.bind(q, 0)
        p.set_targets(TARGETS + [dict(TARGETS[0], id="T2", role="destination", category="container",
                                     source_phrase="container", aliases=[])])
        def detections(detector, mode, payload):
            query = payload["messages"][0]["content"][1]["text"]
            category = "container" if "container" in query else "star"
            start, count = (13, 1) if category == "container" else (0, 13)
            return {"status": "ok", "output": {"detections": [
                {"label": category, "bbox_2d": [i * 60, 50, i * 60 + 40, 100]}
                for i in range(start, start + count)]}}
        with patch.object(Detector, "request", autospec=True, side_effect=detections):
            p.prepare()
        self.assertEqual(p.inspected, {"G001", "G014"})
        self.assertEqual(len(p.groups), 14)
        self.assertEqual(len(p.summary()["candidates"]), 13)
        p.tool({"name": "view_candidates", "page": 1})
        self.assertEqual(p.summary()["candidate_page"], 1)
        self.assertEqual(len(p.summary()["candidates"]), 3)
        self.assertEqual(len(p.context_images()), 3)

    def test_all_environments_keep_native_actions_fields_and_effective_shots(self):
        sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
        for kind in ("navigation", "alfred", "habitat", "manipulation"):
            with self.subTest(environment=kind):
                q = planner_public(kind)
                p = Perception(self.config, self.temp.name, MemoryJournal())
                p.bind(q, 0)
                p.set_targets(TARGETS)
                with patch.object(Detector, "request", return_value=output()):
                    p.prepare()
                planner = TargetPlanner(sources, Task(kind, "base", 0), "qwen3.8-flash", 10, q["actions"],
                                        "json_object")
                self.assertEqual(planner.adapter.n_shot, 3)
                messages = planner.messages(q, p)
                self.assertEqual(ALFRED_HAND_PROMPT in messages[0]["content"], kind == "alfred")
                if kind == "manipulation":
                    actions, expected = [{"action": [50, 50, 50, 0, 0, 0, 1]}], [[50, 50, 50, 0, 0, 0, 1]]
                else:
                    actions, expected = [0, [1, "native fixture"]], [0, 1]
                reply = action_reply(actions)
                for field in ("visual_state_description", "language_plan"):
                    with self.assertRaises(PlannerError):
                        planner.decision(json.dumps(dict(reply, **{field: ""})), p)
                plan = planner.decision(json.dumps(reply), p)
                self.assertEqual(plan.status, "valid")
                self.assertEqual(plan.actions, expected)
        self.assertEqual(self.config["n_shots"], dict.fromkeys(self.config["n_shots"], 3))
        original = json.loads(Path(__file__).with_name("experiment.json").read_text())
        self.assertEqual(original["n_shots"]["alfred"], 10)
        original["n_shots"]["alfred"] = 0
        path = Path(self.temp.name) / "config.json"
        path.write_text(json.dumps(original))
        self.assertEqual(load_config(path)[0]["n_shots"]["alfred"], 0)

    def test_config_has_no_dino_threshold_or_credential_fields(self):
        self.assertNotIn("box_threshold", self.config["detection"])
        bad = copy.deepcopy(self.config["detection"])
        bad["service"]["base_url"] = "http://user:secret@10.10.0.198:28101/v1"
        with self.assertRaises(ContractError):
            validate_detection(bad)


if __name__ == "__main__":
    unittest.main()
