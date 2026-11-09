"""Validate real rejected transitions, legal continuations, and rebuilt prompts."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "src"), str(ROOT / "checks/diagnosis-repair-r05")]
from action_contract import plain
from action_identity import source_identity
from action_solver import ActionSolver
from action_trajectory import restore
from budget import RunBudget
from methods import load_methods
from protocol import canonical, sha256
from replay_raw import hydrate, evaluate, reconstruct
from task_card import SubgoalLedger
from task_context import GUIDANCE

prior = json.loads((ROOT / "checks/diagnosis-repair-r05/deployment.json").read_bytes())["source"]
current = source_identity()
changed = [p for p, digest in current["files_sha256"].items() if prior["files_sha256"].get(p) != digest]
assert set(changed) == {"embodiedRunner/task_context.py", "embodiedRunner/action_identity.py",
                        "embodiedRunner/action_live_launch.py"}
assert "Complete the instructed task" in GUIDANCE and '"complete_when":"action_ack"' in GUIDANCE
methods = load_methods()
checks, contexts = [], []
read = json.loads((HERE / "r05-followup-read.json").read_bytes())
for record in read["episodes"]:
    env = record["episode_id"].split(":")[0]
    solver = ActionSolver(SimpleNamespace(), arm="Full", environment=env, method=methods[env], task_cards=True)
    for decision in record["decisions"]:
        card = decision["input_card"]
        if (record["episode_id"] not in {"eb_alfred:base:000", "eb_alfred:base:004"}
                or decision["response_event"] not in {30, 34, 38}):
            continue
        revision = card["identity"]["observation_revision"]
        snapshot = SimpleNamespace(observation=SimpleNamespace(observation_revision=revision))
        proposal = solver._parse(json.dumps(decision["output"]), snapshot)
        rejected = evaluate(hydrate(SubgoalLedger, card), proposal, revision)
        assert rejected == "subgoal_completion_not_supported"
        continuation = deepcopy(decision["output"])
        continuation["subgoal"] = deepcopy(card["subgoal"])
        continuation["transition"].update(kind="keep", source="none", detail="")
        legal = solver._parse(json.dumps(continuation), snapshot)
        ledger = hydrate(SubgoalLedger, card)
        assert evaluate(ledger, legal, revision) == "accepted"
        selected = 79 if record["episode_id"].endswith("000") else 17
        guard = ledger.check_action(selected, SimpleNamespace(
            observation_revision=revision, action_table=[
                line.split(":", 1)[1] for line in card["catalog"]["text"].splitlines()]))
        assert guard is None
        # Action-specific ACK is also a legal lifecycle; it certifies one action only.
        one_action = deepcopy(card)
        one_action["subgoal"]["goal"] = "Execute the selected official action once"
        one_action["subgoal"]["complete_when"] = "action_ack"
        one_action["subgoal"]["destination"] = ""
        assert evaluate(hydrate(SubgoalLedger, one_action), proposal, revision) == "accepted"
        checks.append({"episode_id": record["episode_id"], "response_event": decision["response_event"],
                       "decision_id": decision["decision_id"], "original": rejected,
                       "keep_continuation": "accepted", "next_action_id": selected,
                       "action_guard": guard, "one_action_ack": "accepted"})
assert len(checks) == 6
inventory = json.loads((ROOT / "checks/diagnosis-repair-r05/embodied_diagnosis_inventory_20261106.json").read_bytes())
paths = [(r["episode_id"], Path(r["trajectory_path"])) for r in inventory["records"] if r["version"] == "r04"]
paths += [(r["episode_id"], Path(r["trajectory_path"])) for r in read["episodes"]]
for ep, path in paths:
    env = ep.split(":")[0]
    budget = RunBudget("lifecycle-context", [env]).episode(ep, env)
    solver = ActionSolver(SimpleNamespace(), arm="Full", environment=env, method=methods[env], task_cards=True)
    solver.card_budget = budget
    snapshots, png = {}, None
    for event in [restore(e, path.parent) for e in json.loads(path.read_bytes())["events"]]:
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
                "synthesis": budget.policy["synthesis_per_decision"] - card["remaining"]["decision_requests"]}
            solver.card_evidence = card["evidence"]
            messages, _, chars = solver.messages(snapshot, card["repair"])
            new = json.loads(messages[1]["content"][0]["text"])
            assert methods[env].skill_md in messages[0]["content"]
            assert new["history"] == card["history"] and new["task"] == card["task"]
            assert new["catalog"] == card["catalog"]
            assert "Intermediate ACKs do not complete multi-step goals." in messages[0]["content"]
            contexts.append({"episode_id": ep, "event": event["index"], "text_chars": chars,
                             "over_target": chars > 16000, "examples": len(new["official_examples"]),
                             "prompt_sha256": sha256(canonical(messages))})
assert len(contexts) == 206
report = {"schema": "embodied-r06-offline/1", "status": "passed", "changed_source_files": changed,
          "scientific_variable": "Subgoal lifecycle guidance and generic output example only.",
          "engineering_changes": "Allowlist and selection config for independent two-task launch.",
          "model_requests": 0, "simulator_launches": 0, "real_failure_continuations": checks,
          "context_snapshots": len(contexts), "max_text_chars": max(r["text_chars"] for r in contexts),
          "mean_text_chars": sum(r["text_chars"] for r in contexts)/len(contexts),
          "over_target": sum(r["over_target"] for r in contexts), "contexts": contexts,
          "source_identity": current, "scope": "Protocol legality and delivery, not model or task improvement."}
with (HERE / "offline-final.json").open("xb") as f:
    f.write(canonical(report) + b"\n")
print(json.dumps({k: v for k, v in report.items() if k not in {"source_identity", "contexts"}}))
