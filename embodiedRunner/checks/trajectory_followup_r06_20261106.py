"""Read frozen trajectories and export public evidence; never call a service."""
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "runs/task-card-full-r06-episodes"
OUTPUT = ROOT / "runs/trajectory-followup-r06-20261106-evidence.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def inspect_episode(path):
    raw_hash = digest(path)
    trace = load(path)
    events_path = path.with_name("events.jsonl")
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert lines == trace["events"], path
    for asset in trace["assets"]:
        asset_path = path.parent / asset["path"]
        assert asset_path.stat().st_size == asset["bytes"], asset_path
        assert digest(asset_path) == asset["sha256"], asset_path
    assert trace["reference_answer_visible"] is False
    assert trace["automatic_retry"] is False
    eid = trace["episode_id"]
    review = load(ROOT / "checks/diagnosis-repair-r06/full"
                  / (eid.replace(":", "-") + "-review.json"))
    review_by_event = {row["response_event"]: row for row in review["chain"]}
    counters = Counter()
    decisions, observations, actions, perception = [], [], [], []
    goals = set()
    card = request = previous_output = None
    for event in trace["events"]:
        kind = event["kind"]
        if kind == "observation":
            observations.append({"event": event["index"], **event["observation"]})
        elif kind == "model_request":
            request = event
            messages = event["messages"]
            card = json.loads(messages[1]["content"][0]["text"])
            counters["qwen_requests"] += 1
            counters["repair_requests"] += bool(card["repair"])
            counters["assistant_messages"] += sum(
                message["role"] == "assistant" for message in messages)
            counters["model_images"] += sum(
                item["type"] == "image_url" for message in messages
                if isinstance(message["content"], list) for item in message["content"])
        elif kind == "model_response":
            output = json.loads(event["text"])
            reviewed = review_by_event[event["index"]]
            assert reviewed["output"] == output
            assert reviewed["old_goal"] == card["subgoal"]
            assert reviewed["facts"] == card["facts"]
            assert reviewed["latest"] == card["latest"]
            assert reviewed["repair"] == card["repair"]
            assert reviewed["remaining"] == card["remaining"]
            counters["response_review_matches"] += 1
            same_program = bool(
                card["repair"] and previous_output
                and output.get("program_source") == previous_output.get("program_source"))
            counters["repair_same_program"] += same_program
            benign_metadata = card["repair"].get("code") == "subgoal_keep_changed"
            counters["metadata_repair_same_program"] += same_program and benign_metadata
            goals.add(json.dumps(output["subgoal"], sort_keys=True))
            decisions.append({
                "request_event": request["index"],
                "response_event": event["index"],
                "decision_id": event["decision_id"],
                "response_sha256": hashlib.sha256(event["text"].encode()).hexdigest(),
                "input": {key: card[key] for key in (
                    "identity", "task", "subgoal", "facts", "latest", "previous_progress",
                    "loop", "remaining", "repair", "evidence", "request_options", "history")},
                "output": output,
                "same_program_as_previous_response_in_repair": same_program,
            })
            previous_output = output
        elif kind == "action_submission":
            counters["actual_actions"] += 1
            actions.append({"event": event["index"], "name": event["action_name"],
                            "action_id": event["action_id"],
                            "decision_id": event["payload"]["decision_id"]})
        elif kind == "execution":
            counters["program_tools"] += event["receipt"]["tool_calls"]
            counters["program_readers"] += event["receipt"]["reader_calls"]
        elif kind == "perception_request":
            counters["perception_backend_requests"] += 1
        elif kind == "perception_response":
            row = {"event": event["index"], "backend": event["backend"],
                   "decision_id": event["decision_id"]}
            if event["backend"] == "reviewer":
                row["output"] = {
                    key: event["output"][key] for key in (
                        "candidate_id", "admission_reason", "binary_review",
                        "frozen_quality", "provider")}
            elif event["backend"] == "detector":
                row["output"] = event["output"]
            else:
                row["output_assets"] = event["output"]
            perception.append(row)
    assert len(review_by_event) == len(decisions)
    assert review["actions"] == [
        {"event": action["event"], "id": action["action_id"], "name": action["name"]}
        for action in actions]
    result = trace["events"][-1]["report"]
    assert result["reference_answer_visible"] is False
    assert result["automatic_retry"] is False
    assert result["environment_steps"] == len(actions)
    assert digest(path) == raw_hash
    return {
        "episode_id": eid, "trajectory_path": str(path), "trajectory_sha256": raw_hash,
        "events_path": str(events_path), "events_sha256": digest(events_path),
        "verified_assets": len(trace["assets"]), "event_count": len(trace["events"]),
        "instruction": observations[0]["instruction"],
        "task_success": result["task_success"], "task_progress": result["task_progress"],
        "terminal_reason": result["terminal_reason"],
        "official_invalid_actions": result["invalid_actions"],
        "report_errors": result["errors"], "unique_output_goals": len(goals),
        "counts": dict(counters), "decisions": decisions,
        "actions": actions, "observations": observations, "perception": perception,
    }


def main():
    assert not OUTPUT.exists(), "Use a new output path; preserve existing evidence."
    source_paths = [
        ROOT / "task_card.py", ROOT / "task_context.py", ROOT / "action_host.py",
        ROOT / "action_solver.py", ROOT / "evidence.py",
        ROOT / "methods/alfred.md", ROOT / "methods/habitat.md",
        ROOT / "methods/navigation.md",
        ROOT / "runs/diagnosis-repair-r06-final.md",
        ROOT / "runs/diagnosis-repair-r06-final-evidence.json",
    ]
    sources_before = {str(path): digest(path) for path in source_paths}
    episodes = [inspect_episode(path) for path in sorted(
        ARCHIVE.glob("trajectory-*/trajectory.json"))]
    assert len(episodes) == 15
    totals = Counter()
    for episode in episodes:
        totals.update(episode["counts"])
    assert totals["qwen_requests"] == totals["response_review_matches"] == 155
    assert totals["actual_actions"] == 131
    assert totals["program_tools"] == totals["program_readers"] == 0
    inventory = load(ROOT / "checks/diagnosis-repair-r05"
                     / "embodied_diagnosis_inventory_20261106.json")
    history = []
    for row in inventory["records"]:
        path = Path(row["trajectory_path"])
        assert digest(path) == row["trajectory_sha256"], path
        assert digest(Path(row["events_path"])) == row["events_sha256"]
        history.append({
            key: row[key] for key in (
                "version", "episode_id", "trajectory_path", "trajectory_sha256",
                "events_path", "events_sha256", "success", "terminal_reason")})
    assert len(history) == 60
    assert sources_before == {str(path): digest(path) for path in source_paths}
    document = {
        "schema": "embodied-r06-trajectory-followup/1",
        "date": "2026-11-06", "scope": "Offline raw evidence inspection; no live requests.",
        "derived_input_policy": "Public task cards only; image assets remain in frozen archives.",
        "reference_answers_in_online_inputs": False,
        "new_model_or_perception_requests": 0,
        "successes": sum(episode["task_success"] for episode in episodes),
        "totals": dict(totals), "sources_preserved": sources_before,
        "historical_trajectory_hashes_reverified": history,
        "episodes": episodes,
        "interpretation_limits": [
            "Repair means a model_request carrying nonempty repair, not every rejected proposal.",
            "Two unchanged-program metadata repairs were appropriate; do not call them failed action repair.",
            "Unchanged goals can be appropriate in repeated navigation; semantic mismatch requires a case review.",
            "No geometry consumption supports no depth attribution for this run.",
            "Matching initial RGB does not prove identical hidden state or macro-navigation endpoints.",
            "Current source locations describe code; frozen deployment identities are in the original run audits.",
        ],
    }
    with OUTPUT.open("x") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "output": str(OUTPUT), "episodes": len(episodes),
        "successes": document["successes"], "totals": dict(totals),
        "assets_verified": sum(episode["verified_assets"] for episode in episodes),
        "historical_traces_preserved": len(history),
        "source_files_preserved": len(sources_before),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
