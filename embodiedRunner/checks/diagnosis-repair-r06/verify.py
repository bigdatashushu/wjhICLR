"""Audit a completed independent run and compare actual initial observations."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "src")]
from action_identity import source_identity
from action_trajectory import restore
from audit_action_trajectory import audit
from protocol import canonical, sha256


def initial(path):
    raw = json.loads(path.read_bytes())
    event = restore(next(e for e in raw["events"] if e["kind"] == "observation"), path.parent)
    return np.asarray(Image.open(io.BytesIO(event["png"])).convert("RGB")), event["observation"]


def main(scope):
    stem = "task-card-" + scope + "-r06"
    path = ROOT / "runs" / (stem + ".json")
    run = json.loads(path.read_bytes())
    result = audit(path)
    assert result["status"] == run["audit"]["status"] == "passed"
    manifest = run["manifest"]
    deployed = json.loads((HERE / scope / "deployment.json").read_bytes())["source"]
    assert manifest["implementation_source"] == deployed == source_identity()
    assert manifest["automatic_retry"] is False and manifest["reference_answer_visible"] is False
    old = json.loads((ROOT / "runs/task-card-15-full-context-r04.json").read_bytes())["manifest"]
    unchanged = ["model_config_sha256", "budget_sha256_by_environment", "method_sha256_by_environment",
                 "method_library_sha256", "evidence_config_sha256", "input_policy", "execution_policy"]
    for k in unchanged:
        if k in {"budget_sha256_by_environment", "method_sha256_by_environment"}:
            assert all(value == old[k][env] for env, value in manifest[k].items())
            if scope == "full":
                assert manifest[k] == old[k]
        else:
            assert manifest[k] == old[k]
    inventory = json.loads((ROOT / "checks/diagnosis-repair-r05/embodied_diagnosis_inventory_20261106.json").read_bytes())
    comparisons = {(r["episode_id"], r["version"]): {
        "trajectory_path": r["trajectory_path"], "success": r["success"]} for r in inventory["records"]}
    for r in json.loads((ROOT / "checks/diagnosis-repair-r05/pilot-verification.json").read_bytes())["episodes"]:
        comparisons[(r["episode_id"], "r05_pilot")] = r
    if scope == "full":
        for r in json.loads((HERE / "pilot/verification.json").read_bytes())["episodes"]:
            comparisons[(r["episode_id"], "r06_pilot")] = r
    index = json.loads((ROOT / "runs" / (stem + "-episodes/trajectories.json")).read_bytes())
    lookup = {r["episode_id"]: ROOT / "runs" / (stem + "-episodes") / r["directory"] / "trajectory.json"
              for r in index["episodes"]}
    episodes = []
    for attempt in run["attempts"]:
        ep = attempt["episode_id"]
        trace_path = lookup[ep]
        trace = json.loads(trace_path.read_bytes())
        events = [restore(e, trace_path.parent) for e in trace["events"]]
        assert events[-1]["kind"] == "episode_result"
        assert trace["reference_answer_visible"] is False and trace["automatic_retry"] is False
        assert attempt["status"] == "finished"
        pixels, obs = initial(trace_path)
        pairs = {}
        for (episode_id, version), reference in comparisons.items():
            if episode_id != ep:
                continue
            reference_pixels, reference_obs = initial(Path(reference["trajectory_path"]))
            delta = np.abs(pixels.astype(np.int16) - reference_pixels.astype(np.int16))
            assert obs["instruction"] == reference_obs["instruction"]
            assert obs["action_table_sha256"] == reference_obs["action_table_sha256"]
            pairs[version] = {
                "success": reference["success"], "initial_pixel_equal": bool(np.array_equal(pixels, reference_pixels)),
                "mean_channel_abs_difference": float(delta.mean()), "max_channel_abs_difference": int(delta.max()),
                "instruction_equal": True, "initial_action_table_equal": True}
        requests, latest_request, decisions, tools, actions, observations = [], None, [], [], [], []
        for event in events:
            if event["kind"] == "model_request":
                latest_request = event
                assert event["receipt"]["reference_answer_visible"] is False
                assert event["receipt"]["model_config_sha256"] == manifest["model_config_sha256"]
                requests.append(event)
            elif event["kind"] == "model_response":
                card = json.loads(latest_request["messages"][1]["content"][0]["text"])
                decisions.append({"event": event["index"], "request_event": latest_request["index"],
                                  "decision_id": event["decision_id"], "output": json.loads(event["text"]),
                                  "input_subgoal": card["subgoal"], "facts": card["facts"],
                                  "latest": card["latest"], "repair": card["repair"],
                                  "request_options": card["request_options"],
                                  "remaining": card["remaining"], "loop": card["loop"],
                                  "response_sha256": sha256(event["text"].encode())})
            elif event["kind"] == "execution":
                tools.append({"event": event["index"], "program": event["program_source"],
                              "receipt": event["receipt"], "results": event["results"]})
            elif event["kind"] == "action_submission":
                actions.append({"event": event["index"], "id": event["action_id"], "name": event["action_name"]})
            elif event["kind"] == "observation":
                observations.append({"event": event["index"], "observation": event["observation"]})
        repair_counts = Counter(d["repair"].get("code", "none") for d in decisions)
        episodes.append({
            "episode_id": ep, "success": attempt["task_success"], "progress": attempt["task_progress"],
            "steps": attempt["environment_steps"], "qwen": len(requests), "status": attempt["status"],
            "terminal_reason": attempt["terminal_reason"], "trajectory_path": str(trace_path),
            "trajectory_sha256": sha256(trace_path.read_bytes()), "event_count": len(events),
            "initial_comparisons": pairs, "actions": actions, "observations": observations,
            "decisions": decisions, "execution_evidence": tools, "repair_codes": dict(repair_counts),
            "text_chars": [e["receipt"]["text_chars"] for e in requests],
            "errors": attempt.get("errors", [])})
    assert len(episodes) == (2 if scope == "pilot" else 15)
    chars = [n for r in episodes for n in r["text_chars"]]
    report = {"schema": "embodied-r06-completed-verification/1", "status": "passed", "scope": scope,
              "task_denominator": len(episodes), "successes": sum(r["success"] for r in episodes),
              "environment_successes": {e: sum(r["success"] for r in episodes if r["episode_id"].startswith(e + ":"))
                                        for e in ("eb_navigation", "eb_alfred", "eb_habitat")},
              "qwen_requests": len(chars), "actual_actions": sum(r["steps"] for r in episodes),
              "run_sha256": sha256(path.read_bytes()), "trajectory_audit": result,
              "runtime_audit": run["audit"], "unchanged_identity_fields": unchanged,
              "source_identity_matches_deployment": True, "automatic_retry": False,
              "reference_answer_visible": False, "operational_failure_attempts": 0,
              "request_text_max": max(chars), "request_text_mean": sum(chars) / len(chars),
              "request_text_over_16000": sum(n > 16000 for n in chars),
              "episodes": episodes,
              "interpretation_scope": "Development tasks. Navigation RGB pairing differences are listed; "
                                      "task ID equality does not establish strict pixel pairing."}
    with (HERE / scope / "verification.json").open("xb") as f:
        f.write(canonical(report) + b"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"episodes", "trajectory_audit", "runtime_audit"}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scope", choices=("pilot", "full"))
    main(parser.parse_args().scope)
