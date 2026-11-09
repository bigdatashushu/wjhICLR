"""Offline request/decision replay and pixel/action audit; no live model or detector calls."""
import argparse
import base64
from collections import deque
import io
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from .config import load_config
from .perception import Perception
from .planner import TargetPlanner
from .protocol import Task, validate_packet
from .sources import ROOT, Sources, canonical, digest, file_digest, implementation_hashes
from .__main__ import check


def episode_audit(directory, config, sources):
    events = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
    result = json.loads((directory / "result.json").read_text())
    task = Task(**result["task"])
    observations = {e["serial"]: {k: e[k] for k in ("task", "public", "official", "input_sha256")}
                    for e in events if e["type"] == "observation"}
    for serial, packet in observations.items():
        validate_packet(packet, task)
        public = packet["public"]
        assert (directory / f"observation-{serial:03d}-0.png").read_bytes() == base64.b64decode(
            public["images"][0]["base64"])
        assert (directory / f"observation-{serial:03d}-raw.png").read_bytes() == base64.b64decode(
            public["raw_rgb"]["base64"])
        assert (directory / f"observation-{serial:03d}-depth.png").read_bytes() == base64.b64decode(
            public["depth"]["image"]["base64"])
        from .depth import decode_raw
        actual_depth = np.load(directory / f"observation-{serial:03d}-depth-meters.npy", allow_pickle=False)
        assert np.array_equal(actual_depth, decode_raw(public["depth"]), equal_nan=True)
    requests = [e for e in events if e["type"] == "model_request"]
    responses = [e for e in events if e["type"] == "model_response"]
    assert len(requests) == len(responses) <= config["max_requests"][task.environment]
    sent = [e for e in events if e["type"] == "action_sent"]
    ack = [e for e in events if e["type"] == "action_ack"]
    assert len(sent) == len(ack) == result["metrics"]["num_steps"]
    assert [e["expected_step"] for e in sent] == list(range(1, len(sent) + 1))
    assert [e["action"] for e in sent] == [e["action"] for e in ack]
    assert events[-1]["type"] == "episode_end"
    detector = deque(e for e in events if e["type"] == "detector_result")
    class ReplayDetector:
        def detect(self, image, category, threshold):
            event = detector.popleft()
            assert event["query"] == category
            payload = {"image": image["base64"], "text_prompt": category + ".", "box_threshold": threshold}
            assert event["report"]["request_sha256"] == digest(canonical(payload))
            return event["report"]
    class ReplayJournal:
        def __init__(self):
            self.events = []
        def append(self, value):
            self.events.append(value)
    journal = ReplayJournal()
    replayed_requests, replayed_plans = 0, 0
    with tempfile.TemporaryDirectory(dir=ROOT / "target_detection_v2/verification") as tmp:
        perception = Perception(config, tmp, journal, ReplayDetector())
        planner = TargetPlanner(sources, task, config["model"]["name"],
                                config["n_shots"][task.environment],
                                observations[0]["public"]["actions"], config["model"]["response_format"])
        ack_step = 0
        for event in events:
            if event["type"] == "model_request":
                public = observations[event["observation_serial"]]["public"]
                perception.bind(public, event["observation_serial"])
                payload = json.loads((directory / event["request_file"]).read_text())
                assert digest(canonical(payload)) == event["payload_sha256"]
                assert payload["messages"] == planner.messages(public, perception), "model_request_replay_mismatch"
                assert payload["model"] == config["model"]["name"]
                assert payload["temperature"] == config["model"]["temperature"]
                assert payload["max_tokens"] == config["model"]["max_tokens"]
                assert all(payload[k] == v for k, v in config["model"]["extra_body"].items())
                replayed_requests += 1
            elif event["type"] == "model_response":
                response = json.loads((directory / f'response-{event["request_index"]:03d}.json').read_text())
                assert response == event["response"]
                plan = planner.decision(response["text"], perception)
                assert (vars(plan) if plan else None) == event["plan"]
                replayed_plans += int(plan is not None)
            elif event["type"] == "action_ack":
                # The actual next observation supplies the official feedback.
                ack_step += 1
                packet = next(p for p in observations.values() if p["public"]["step"] == ack_step)
                planner.feedback(event["action"], packet["public"])
        assert not detector
        emitted = [e for e in journal.events if e["type"] == "perception_tool_result"]
        recorded = [e for e in events if e["type"] == "perception_tool_result"]
        assert emitted == recorded, "tool_result_replay_mismatch"
        for event in recorded:
            assert event["simulator_steps"] == event["simulator_observations"] == 0
            packet = observations[event["result"]["observation_serial"]]
            assert event["result"]["environment_step"] == packet["public"]["step"]
            assert event["result"]["raw_rgb_sha256"] == packet["public"]["raw_rgb"]["sha256"]
            for image in event["images"]:
                assert file_digest(directory / image["file"]) == image["sha256"]
                assert (directory / image["file"]).read_bytes() == (Path(tmp) / image["file"]).read_bytes()
            if event["result"]["tool"] == "inspect_candidates":
                raw = Image.open(io.BytesIO(base64.b64decode(packet["public"]["raw_rgb"]["base64"]))).convert("RGB")
                for crop, image in zip(event["result"]["crops"], event["images"]):
                    actual = Image.open(directory / image["file"]).convert("RGB")
                    assert np.array_equal(np.asarray(actual), np.asarray(raw.crop(crop["bounds"])))
            for placement in event["result"].get("label_placements", []):
                assert placement["label_rect"][0] >= 500
        plans = {e["request_index"]: e["executed_prefix"] for e in events if e["type"] == "plan_execution"}
        for event in sent:
            assert event["action"] == plans[event["request_index"]][event["index_in_plan"]]
        assert result["perception"]["tool_calls"] == len(recorded) == perception.tool_calls
        assert result["perception"]["detector_requests"] == perception.detector_calls
        assert result["perception"]["crop_count"] == perception.crop_count
    return {"task": task.as_dict(), "status": result["status"], "official_success": result["official"]["task_success"],
            "model_requests": len(requests), "tool_calls": len(recorded),
            "detector_requests": result["perception"]["detector_requests"],
            "crops": result["perception"]["crop_count"], "actions": len(sent), "acks": len(ack),
            "targets": result["perception"]["targets"], "request_replay_verified": replayed_requests,
            "action_plans_replayed": replayed_plans, "crop_pixels_verified": True,
            "tool_steps_zero": True, "candidate_ids_and_review_replayed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    out = ROOT / "target_detection_v2"
    config, tasks = load_config(out / "experiment-20261108-r02.json")
    frozen = json.loads((out / "verification/frozen.json").read_text())
    assert implementation_hashes() == frozen["implementation_hashes"]
    sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
    check(config, tasks, sources)
    assert sources.hashes == frozen["source_hashes"]
    directory = out / "runs" / args.run_id
    rows = [episode_audit(directory / t.directory, config, sources) for t in tasks]
    report = {"status": "verified", "selected": len(rows), "episodes": rows,
              "model_requests": sum(r["model_requests"] for r in rows),
              "tool_calls": sum(r["tool_calls"] for r in rows),
              "detector_requests": sum(r["detector_requests"] for r in rows),
              "actions": sum(r["actions"] for r in rows), "acks": sum(r["acks"] for r in rows),
              "live_requests_in_audit": 0, "source_identity_verified": True}
    (directory / "audit.json").write_bytes(canonical(report) + b"\n")
    print(canonical(report).decode())


if __name__ == "__main__":
    main()
