"""Read-only verification of complete public trajectories against sealed runs."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from action_audit import audit_run
from action_contract import public_hash
from action_trajectory import restore
from protocol import ContractError, json_object, sha256


def require(condition, reason):
    if not condition:
        raise ContractError(reason)


def audit(path):
    result = json_object(Path(path).read_bytes())
    checked = audit_run(result)
    require(checked["status"] == "passed", "run_audit_failed")
    directory = Path(path).with_name(Path(path).stem + "-episodes")
    index = json_object((directory / "trajectories.json").read_bytes())
    require(index["run_sha256"] == result["sha256"] and index["manifest"] == result["manifest"],
            "trajectory_run_identity")
    require(len(index["episodes"]) == len(result["attempts"]), "trajectory_denominator")
    rows = []
    for entry, attempt in zip(index["episodes"], result["attempts"]):
        require((entry["episode_id"], entry["arm"]) == (attempt["episode_id"], attempt["arm"]),
                "trajectory_episode_identity")
        location = (directory / entry["directory"]).resolve()
        require(location.is_relative_to(directory.resolve()), "trajectory_path")
        trajectory = json_object((location / "trajectory.json").read_bytes())
        require(trajectory == entry["trajectory"], "trajectory_index_changed")
        for asset in trajectory["assets"]:
            p = (location / asset["path"]).resolve()
            require(p.is_relative_to(location), "trajectory_asset_path")
            require(p.stat().st_size == asset["bytes"] and sha256(p.read_bytes()) == asset["sha256"],
                    "trajectory_asset_changed")
        journal = [json_object(line) for line in (location / "events.jsonl").read_bytes().splitlines()]
        require(journal == trajectory["events"], "trajectory_journal_changed")
        events = restore(trajectory["events"], location)
        require([e["index"] for e in events] == list(range(len(events))), "trajectory_event_order")
        require(events[-1]["kind"] == "episode_result" and events[-1]["report"] == attempt,
                "trajectory_final_report")
        counts = Counter(e["kind"] for e in events)
        requests = [e for e in events if e["kind"] == "model_request"]
        responses = [e for e in events if e["kind"] == "model_response"]
        require(len(requests) == len(attempt["model_requests"]), "trajectory_request_denominator")
        for event, receipt in zip(requests, attempt["model_requests"]):
            require(public_hash(event["messages"]) == receipt["messages_sha256"],
                    "trajectory_messages_hash")
            require(event["receipt"]["decision_id"] == receipt["decision_id"],
                    "trajectory_request_decision")
        for event, receipt in zip(responses, (r for r in attempt["model_requests"]
                                              if "response_sha256" in r)):
            require(sha256(event["text"].encode()) == receipt["response_sha256"],
                    "trajectory_response_hash")
        observations = [e for e in events if e["kind"] == "observation"]
        require(len(observations) == len(attempt["observations"]), "trajectory_observation_denominator")
        for event, receipt in zip(observations, attempt["observations"]):
            require(event["receipt"] == receipt and sha256(event["png"]) == receipt["rgb_sha256"],
                    "trajectory_observation_hash")
            if "planner_rgb_sha256" in receipt:
                require(sha256(event["planner_png"]) == receipt["planner_rgb_sha256"],
                        "trajectory_planner_observation_hash")
        if attempt["arm"] == "Direct":
            require(not any(counts[k] for k in ("perception_request", "perception_response",
                                               "prepared_evidence", "execution", "tool_response")),
                    "direct_trajectory_extra_evidence")
        if attempt["environment"] == "eb_manipulation":
            for event in observations:
                public = event["observation"]
                require(not {"depth", "mask", "waypoints", "object_informations", "state", "reward",
                             "task_success", "task_progress"} & public.keys(),
                        "manipulation_private_observation")
                require(public["manipulation_input"]["source"] ==
                        "official_depth_instance_masks_and_camera_parameters",
                        "manipulation_public_coordinate_source")
            for event, receipt in zip(requests, attempt["model_requests"]):
                image_urls = [part["image_url"]["url"] for message in event["messages"]
                              if isinstance(message["content"], list) for part in message["content"]
                              if part.get("type") == "image_url"]
                import base64
                require(len(image_urls) == 1 and sha256(base64.b64decode(
                    image_urls[0].split(",", 1)[1], validate=True)) == receipt["planner_rgb_sha256"],
                    "manipulation_planner_image_identity")
        rows.append({"episode_id": attempt["episode_id"], "arm": attempt["arm"],
                     "events": len(events), "assets": len(trajectory["assets"]),
                     "model_requests": len(requests), "model_responses": len(responses),
                     "environment_steps": attempt["environment_steps"],
                     "event_counts": dict(counts)})
    return {"schema": "embodied-complete-trajectory-audit/1", "status": "passed",
            "run_sha256": result["sha256"], "run_audit": checked, "episodes": rows,
            "read_only": True, "model_requests": 0, "simulator_launches": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    from protocol import canonical
    print(canonical(audit(args.run)).decode())
