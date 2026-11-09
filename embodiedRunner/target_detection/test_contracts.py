"""Offline behavioral checks, including tool-loop simulator isolation and full-plan ACK."""
import base64
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from .perception import (Perception, PlannerError, crop_bounds, iou, page_image, png_bytes,
                         validate_targets)
from .protocol import Task
from .sources import ROOT, digest


TARGETS = [
    {"id": "T1", "role": "object", "category": "star", "source_phrase": "star",
     "quantity": 1, "attributes": [], "relation": "", "aliases": ["star shaped block"]},
    {"id": "T2", "role": "destination", "category": "container", "source_phrase": "container",
     "quantity": None, "attributes": [], "relation": "", "aliases": []}]


class Journal:
    def __init__(self):
        self.events = []
    def append(self, value):
        self.events.append(value)


class FakeDetector:
    def __init__(self):
        self.calls = []
    def detect(self, image, category, threshold):
        self.calls.append((image["sha256"], category, threshold))
        return {"detections": [{"id": 0, "label": category, "confidence": 0.27,
                                 "bbox": [40, 50, 60, 70]},
                                {"id": 1, "label": "", "confidence": 0.18,
                                 "bbox": [200, 200, 260, 260]}]}


def public():
    array = np.arange(500 * 500 * 3, dtype=np.uint8).reshape(500, 500, 3)
    png = png_bytes(Image.fromarray(array))
    return {"instruction": "Put the star into the container.", "step": 0,
            "raw_rgb": {"base64": base64.b64encode(png).decode(), "sha256": digest(png)},
            "anonymous_anchors": []}


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "target_detection/verification")
        self.addCleanup(self.tmp.cleanup)
        self.journal, self.detector = Journal(), FakeDetector()
        self.p = Perception({"detection": {"box_threshold": .15, "group_iou": .8,
                            "page_size": 12, "max_crop_groups": 2}},
                            self.tmp.name, self.journal, self.detector)
        self.p.bind(public(), 0)
        self.p.set_targets(TARGETS)

    def test_targets_are_public_and_frozen(self):
        bad = copy.deepcopy(TARGETS)
        bad[0]["source_phrase"] = "hidden task star"
        with self.assertRaises(PlannerError):
            validate_targets(bad, public()["instruction"])
        bad = copy.deepcopy(TARGETS)
        bad[0]["category"] = "moon"
        with self.assertRaises(PlannerError):
            self.p.set_targets(bad)
        bad = copy.deepcopy(TARGETS)
        bad[0]["aliases"] = None
        with self.assertRaises(PlannerError):
            validate_targets(bad, public()["instruction"])

    def test_overlap_chain_does_not_merge_neighbors(self):
        boxes = [[0, 0, 100, 100], [8, 0, 108, 100], [16, 0, 116, 100]]
        self.assertGreater(iou(boxes[0], boxes[1]), .8)
        self.assertGreater(iou(boxes[1], boxes[2]), .8)
        self.assertLess(iou(boxes[0], boxes[2]), .8)
        self.p.group([{"id": n, "bbox": b, "target_id": "T1", "query": "star",
                       "confidence": .2, "label": ""} for n, b in enumerate(boxes)])
        self.assertEqual([len(g["members"]) for g in self.p.groups], [2, 1])

    def test_separate_queries_lossless_cache_and_expiration(self):
        request = {"name": "detect_targets", "target_ids": ["T1", "T2"]}
        result, _ = self.p.tool(request)
        self.assertEqual([c[1] for c in self.detector.calls], ["star", "container"])
        self.assertEqual(result["group_count"], 2)
        self.assertEqual(result["proposal_count"], 4)
        self.assertEqual(len([m for g in self.p.groups for m in g["members"] if not m["label"]]), 2)
        ids = [g["id"] for g in self.p.groups]
        self.p.tool(request)
        self.assertEqual(len(self.detector.calls), 2)
        self.assertEqual(ids, [g["id"] for g in self.p.groups])
        self.p.bind(public(), 1)
        with self.assertRaises(PlannerError):
            self.p.tool({"name": "inspect_candidates", "group_ids": [ids[0]]})
        self.p.tool(request)
        self.assertTrue(set(ids).isdisjoint(g["id"] for g in self.p.groups))

    def test_alias_request_validated_before_network(self):
        with self.assertRaises(PlannerError):
            self.p.tool({"name": "detect_targets", "target_ids": ["T1", "T2"],
                         "query_aliases": {"T2": "scene full catalog"}})
        self.assertEqual(self.detector.calls, [])

    def test_crop_pixels_and_sidebar_labels(self):
        self.p.tool({"name": "detect_targets", "target_ids": ["T1"]})
        result, images = self.p.tool({"name": "inspect_candidates", "group_ids": ["G001"]})
        raw = Image.open(io.BytesIO(base64.b64decode(self.p.public["raw_rgb"]["base64"])))
        crop = Image.open(io.BytesIO(base64.b64decode(images[0]["base64"])))
        bounds = crop_bounds(self.p.groups[0]["bbox"])
        self.assertTrue(np.array_equal(np.asarray(crop), np.asarray(raw.crop(bounds))))
        png, placements = page_image(base64.b64decode(self.p.public["raw_rgb"]["base64"]),
                                     self.p.groups, 0)
        self.assertEqual(Image.open(io.BytesIO(png)).size, (810, 500))
        self.assertTrue(all(p["label_rect"][0] >= 500 for p in placements))
        with self.assertRaises(PlannerError):
            self.p.approve_action({"perception_review": [], "selected_targets": []})
        self.p.approve_action({"perception_review": [
            {"group_id": "G001", "verdict": "match", "reason": "visible star"}],
            "selected_targets": [{"target_id": "T1", "group_id": "G001"}]})

    def test_real_prompt_has_authoritative_phase_instruction(self):
        from .planner import TargetPlanner
        from .sources import Sources
        sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
        planner = TargetPlanner(sources, Task("navigation", "base", 0), "qwen3.8-flash", 0,
                                [str(i) for i in range(8)], "json_object")
        q = copy.deepcopy(self.p.public)
        q.update(images=[q["raw_rgb"]], depth={"image": q["raw_rgb"]},
                 env_feedback="", manipulation=None)
        self.p.targets = None
        messages = planner.messages(q, self.p)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("CURRENT PHASE: EXTRACT TARGETS AND DETECT", messages[0]["content"])
        self.p.set_targets(TARGETS)
        self.assertIn("CURRENT PHASE: DETECT THE CURRENT VIEW", planner.messages(q, self.p)[0]["content"])
        self.p.tool({"name": "detect_targets", "target_ids": ["T1"]})
        self.assertIn("CURRENT PHASE: INSPECT CROPS", planner.messages(q, self.p)[0]["content"])
        self.p.tool({"name": "inspect_candidates", "group_ids": ["G001"]})
        self.assertIn("CURRENT PHASE: REVIEW CROPS", planner.messages(q, self.p)[0]["content"])
        self.p.tool({"name": "review_candidates", "perception_review": [
            {"group_id": "G001", "verdict": "match", "reason": "visible star"}],
            "selected_targets": [{"target_id": "T1", "group_id": "G001"}]})
        self.assertIn("CURRENT PHASE: FULL ACTION PLAN", planner.messages(q, self.p)[0]["content"])
        self.p.approve_action({"executable_plan": []})
        self.p.bind(public(), 1)
        self.assertIsNone(self.p.completed_review)

    def test_tool_rounds_no_extra_simulator_observe_and_full_plan_ack(self):
        from . import runner
        from .planner import Plan
        from .depth import make_depth
        task = Task("navigation", "base", 0)
        p = public()
        p.update(images=[dict(p["raw_rgb"], view="head_rgb", source="official current RGB")],
                 actions=[str(i) for i in range(8)], manipulation=None,
                 last_action_success=True, env_feedback="")
        p["raw_rgb"].update(view="head_rgb", source="official same-observation unannotated RGB")
        class Session:
            def __init__(self):
                self.operations, self.step = [], 0
            def request(self, operation, **kw):
                self.operations.append((operation, kw))
                if operation == "step":
                    self.step += 1
                q = copy.deepcopy(p)
                q["step"] = self.step
                q["depth"] = make_depth(np.ones((500, 500), dtype=np.float32),
                                         q["images"][0]["sha256"], self.step, "navigation")
                from .sources import canonical
                return {"task": task.as_dict(), "public": q, "official": {
                    "done": self.step == 2, "task_success": int(self.step == 2),
                    "reward": 0., "invalid_actions": 0}, "input_sha256":
                    digest(canonical({"task": task.as_dict(), "instruction": q["instruction"]}))}
            def close(self):
                self.operations.append(("close", {}))
        session = Session()
        class Planner:
            def __init__(self, *args):
                self.adapter = SimpleNamespace(output_json_error=0)
                self.schema, self.decisions = {}, 0
            def messages(self, p, perception):
                return []
            def decision(self, text, perception):
                self.decisions += 1
                if self.decisions == 1:
                    perception.set_targets(TARGETS)
                    perception.tool({"name": "detect_targets", "target_ids": ["T1"]})
                    return None
                if self.decisions == 2:
                    perception.tool({"name": "inspect_candidates", "group_ids": ["G001"]})
                    return None
                if self.decisions == 3:
                    perception.tool({"name": "review_candidates", "perception_review": [
                        {"group_id": "G001", "verdict": "match", "reason": "visible star"}],
                        "selected_targets": [{"target_id": "T1", "group_id": "G001"}]})
                    return None
                return Plan("valid", [0, 1, 2], 3)
            def feedback(self, *args):
                pass
        client = SimpleNamespace(payload=lambda *a: {}, complete=lambda p: {"text": "{}"})
        config = {"model": {"name": "offline", "response_format": "json_object"},
                  "n_shots": {"navigation": 0}, "max_requests": {"navigation": 4},
                  "detection": self.p.config["detection"]}
        class OfflinePerception(Perception):
            def __init__(self, c, d, j):
                super().__init__(c, d, j, FakeDetector())
        with patch.object(runner, "TargetPlanner", Planner), patch.object(runner, "Perception", OfflinePerception):
            result = runner.episode(task, config, None, client, lambda *a: session, Path(self.tmp.name) / "episode")
        self.assertEqual(result["status"], "official_done", result["error"])
        self.assertEqual(result["metrics"]["planner_steps"], 4)
        self.assertEqual([op for op, _ in session.operations], ["open", "step", "step", "close"])
        self.assertEqual([kw["first_in_plan"] for op, kw in session.operations if op == "step"], [True, False])
        events = [json.loads(line) for line in (Path(self.tmp.name) / "episode/trajectory.jsonl").read_text().splitlines()]
        self.assertEqual(sum(e["type"] == "action_ack" for e in events), 2)
        self.assertEqual(sum(e["type"] == "perception_bind" for e in events), 1)


if __name__ == "__main__":
    unittest.main()
