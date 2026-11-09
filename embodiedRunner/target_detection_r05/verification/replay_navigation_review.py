"""Reproduce the observed review-wrapper failure without any live requests."""
from collections import deque
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from target_detection_r05.config import load_config
from target_detection_r05.perception import Perception
from target_detection_r05.planner import TargetPlanner
from target_detection_r05.protocol import Task
from target_detection_r05.sources import Sources, canonical, digest, file_digest, implementation_hashes


class Journal:
    def __init__(self):
        self.events = []

    def append(self, value):
        self.events.append(value)


def main():
    original = ROOT / "target_detection_r04/runs/target-detection-20261108-r04/navigation-base-000"
    events = [json.loads(line) for line in (original / "trajectory.jsonl").read_text().splitlines()]
    observations = {e["serial"]: e["public"] for e in events if e["type"] == "observation"}
    detector_events = deque(e for e in events if e["type"] == "detector_result")

    class RecordedDetector:
        def detect(self, image, category, threshold):
            event = detector_events.popleft()
            assert event["query"] == category
            payload = {"image": image["base64"], "text_prompt": category + ".", "box_threshold": threshold}
            assert event["report"]["request_sha256"] == digest(canonical(payload))
            return event["report"]

    config, _ = load_config(ROOT / "target_detection_r05/experiment-20261108-r05.json")
    sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
    journal = Journal()
    task = Task("navigation", "base", 0)
    planner = TargetPlanner(sources, task, config["model"]["name"], config["n_shots"]["navigation"],
                            observations[0]["actions"], config["model"]["response_format"])
    decisions = []
    with tempfile.TemporaryDirectory(dir=ROOT / "target_detection_r05/verification") as tmp:
        perception = Perception(config, tmp, journal, RecordedDetector())
        ack_step = 0
        for event in events:
            if event["type"] == "model_request":
                perception.bind(observations[event["observation_serial"]], event["observation_serial"])
            elif event["type"] == "model_response":
                decision = planner.decision(event["response"]["text"], perception)
                if event["request_index"] != 7:
                    assert (vars(decision) if decision else None) == event["plan"]
                else:
                    assert decision is None
                    authored = json.loads(event["response"]["text"])
                    assert perception.completed_review == authored
                    assert "CURRENT PHASE: FULL ACTION PLAN" in planner.messages(
                        observations[2], perception)[0]["content"]
                decisions.append({"request_index": event["request_index"],
                                  "plan": vars(decision) if decision else None})
            elif event["type"] == "action_ack":
                ack_step += 1
                feedback = next(p for p in observations.values() if p["step"] == ack_step)
                planner.feedback(event["action"], feedback)
        assert not detector_events
        crop_results = [e for e in journal.events if e["type"] == "perception_tool_result"
                        and e["tool_request"]["name"] == "inspect_candidates"]
        assert crop_results and all(e["result"]["tool"] == "inspect_candidates" for e in crop_results)
        normalized = [e for e in journal.events if e["type"] == "explicit_review_wrapper_normalized"]
        assert len(normalized) == 1 and not normalized[0]["action_invented"]
    receipt = {
        "status": "verified",
        "scope": "saved navigation responses replayed through corrected parser; no new model answer",
        "source_run_id": "target-detection-20261108-r04",
        "source_trajectory_sha256": file_digest(original / "trajectory.jsonl"),
        "source_review_response_sha256": file_digest(original / "response-007.json"),
        "implementation_hashes": implementation_hashes(),
        "responses_replayed": len(decisions),
        "explicit_review_wrapper_accepted": True,
        "crop_tool_name_preserved": True,
        "new_action_invented": False,
        "live_model_requests": 0,
        "live_detector_requests": 0,
        "simulator_launches": 0,
        "decisions": decisions,
        "online_retest_completed": False,
    }
    out = ROOT / "target_detection_r05/verification/navigation-review-replay.json"
    out.write_bytes(canonical(receipt) + b"\n")
    print(canonical({k: v for k, v in receipt.items() if k != "implementation_hashes"}).decode())


if __name__ == "__main__":
    main()
