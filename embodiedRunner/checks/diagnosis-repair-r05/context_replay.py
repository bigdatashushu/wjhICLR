"""Verify repaired messages over every original r04 public request snapshot."""
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "src")]
from action_contract import plain, public_hash
from action_solver import ActionSolver
from action_trajectory import restore
from budget import RunBudget
from methods import load_methods
from protocol import canonical
from replay_raw import reconstruct


def main():
    methods = load_methods()
    inventory = json.loads((HERE / "embodied_diagnosis_inventory_20261106.json").read_bytes())
    rows = []
    for record in inventory["records"]:
        if record["version"] != "r04":
            continue
        path = Path(record["trajectory_path"])
        events = [restore(e, path.parent) for e in json.loads(path.read_bytes())["events"]]
        env = record["episode_id"].split(":")[0]
        budget = RunBudget("context-diagnostic", [env]).episode(record["episode_id"], env)
        solver = ActionSolver(SimpleNamespace(), arm="Full", environment=env,
                              method=methods[env], task_cards=True)
        solver.card_budget = budget
        snapshots, png = {}, None
        for event in events:
            if event["kind"] == "observation":
                png = event["png"]
            elif event["kind"] in {"decision", "decision_context"}:
                snapshots[event["snapshot"]["snapshot_sha256"]] = (event["snapshot"], png)
            elif event["kind"] == "model_request":
                card = json.loads(event["messages"][1]["content"][0]["text"])
                wire, image = snapshots[event["receipt"]["snapshot_sha256"]]
                _, snapshot = reconstruct(wire, image)
                for name in ("steps", "qwen", "invalid", "reviews", "dino", "sam", "moge"):
                    budget.counts[name] = budget.policy[name] - card["remaining"][name]
                budget.decisions[snapshot.decision_id] = {
                    "synthesis": budget.policy["synthesis_per_decision"] -
                                 card["remaining"]["decision_requests"]}
                solver.card_evidence = card["evidence"]
                messages, delivery, chars = solver.messages(snapshot, card["repair"])
                current = json.loads(messages[1]["content"][0]["text"])
                assert current["task"] == card["task"]
                assert current["catalog"] == card["catalog"]
                assert current["history"] == card["history"]
                assert methods[env].skill_md in messages[0]["content"]
                assert current["request_options"]["preferred_output"] == "program_source"
                if current["evidence"]["status"] == "prepared" or card["remaining"]["decision_requests"] <= 1:
                    assert current["request_options"]["available"] == ["program_source"]
                if env == "eb_navigation":
                    assert "*** Strategy ***" in messages[0]["content"]
                if env == "eb_habitat":
                    assert all("action_id" in x["excerpt"] and "Output:" in x["excerpt"]
                               for x in current["official_examples"])
                rows.append({
                    "episode_id": record["episode_id"], "event": event["index"],
                    "decision_id": snapshot.decision_id, "text_chars": chars,
                    "history": len(current["history"]), "examples": len(current["official_examples"]),
                    "request_options": current["request_options"],
                    "over_target": chars > 16000,
                    "required_fields_over_budget": current["packing"]["required_fields_over_budget"],
                    "prompt_sha256": public_hash(messages)})
    assert len(rows) == 145
    report = {
        "schema": "embodied-r05-context-replay/1", "status": "passed",
        "request_snapshots": len(rows), "model_requests": 0, "simulator_launches": 0,
        "scope": "Delivery and request-state verification only; no new decisions or task outcomes.",
        "max_text_chars": max(x["text_chars"] for x in rows),
        "mean_text_chars": sum(x["text_chars"] for x in rows) / len(rows),
        "over_target": sum(x["over_target"] for x in rows),
        "example_counts": dict(Counter(x["examples"] for x in rows)),
        "rows": rows}
    with (HERE / "context-replay.json").open("xb") as f:
        f.write(canonical(report) + b"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
