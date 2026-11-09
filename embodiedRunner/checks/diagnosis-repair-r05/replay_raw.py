"""Replay original public requests against preserved and repaired control logic."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "src")]
from action_contract import DecisionSnapshot, EpisodeBinding, ExecutionEnvelope, plain, public_observation
from action_solver import ActionSolver
from action_trajectory import restore
from methods import load_methods
from protocol import ContractError, canonical, sha256
from sandbox_process import execute_isolated
from task_card import SubgoalLedger


def preserved_ledger():
    spec = importlib.util.spec_from_file_location("preserved_task_card", HERE / "source-before/task_card.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SubgoalLedger


def hydrate(cls, card):
    ledger = cls(card["environment"])
    ledger.goal = deepcopy(card["subgoal"])
    ledger.facts = deepcopy(card["facts"])
    ledger.last_action = deepcopy(card["latest"]["action"])
    ledger.last_success = card["latest"]["success"]
    ledger.loop = deepcopy(card["loop"])
    return ledger


def evaluate(ledger, proposal, revision):
    try:
        ledger.accept_goal(proposal.subgoal, proposal.transition, revision)
        return "accepted"
    except ContractError as exc:
        return str(exc)


def reconstruct(wire, png, run="raw-diagnostic"):
    o = wire["observation"]
    binding = EpisodeBinding(run, o["episode_id"], o["official_task_id"], o["environment"],
                             o["subset"], sha256(b"diagnostic-input"))
    obs = public_observation(
        binding, png=png, instruction=o["instruction"], environment_step=o["environment_step"],
        mutation_revision=o["mutation_revision"], observation_revision=o["observation_revision"],
        action_table=[r["action_name"] for r in o["action_table"]], public_feedback=o["public_feedback"],
        public_memory=o["public_memory"])
    snapshot = DecisionSnapshot(
        obs, wire["decision_id"], wire["method_sha256"], wire["budget_sha256"],
        wire["evidence_revision"], wire["evidence_manifest_sha256"], tuple(wire["capabilities"]),
        wire["input_policy"], wire["execution_policy"], task_state=wire["task_state"])
    return binding, snapshot


def main():
    old = preserved_ledger()
    methods = load_methods()
    inventory = json.loads((HERE / "embodied_diagnosis_inventory_20261106.json").read_bytes())
    rows, tool_rows = [], []
    for record in inventory["records"]:
        if record["version"] not in {"r02", "r04"}:
            continue
        path = Path(record["trajectory_path"])
        assert sha256(path.read_bytes()) == record["trajectory_sha256"]
        events = [restore(e, path.parent) for e in json.loads(path.read_bytes())["events"]]
        env = record["episode_id"].split(":")[0]
        solver = ActionSolver(SimpleNamespace(), arm="Full", environment=env,
                              method=methods[env], task_cards=True)
        snapshots, png, request = {}, None, None
        for e in events:
            if e["kind"] == "observation":
                png = e["png"]
            elif e["kind"] in {"decision", "decision_context"}:
                snapshots[e["snapshot"]["snapshot_sha256"]] = (e["snapshot"], png)
            elif e["kind"] == "model_request":
                request = e
            elif e["kind"] == "model_response" and record["version"] == "r04":
                card = json.loads(request["messages"][1]["content"][0]["text"])
                revision = card["identity"]["observation_revision"]
                wire, _ = snapshots[request["receipt"]["snapshot_sha256"]]
                obs = SimpleNamespace(observation_revision=revision, action_table=[
                    r["action_name"] for r in wire["observation"]["action_table"]])
                row = {"episode_id": record["episode_id"], "response_event": e["index"],
                       "request_event": request["index"], "decision_id": e["decision_id"],
                       "old_goal": card["subgoal"], "latest": card["latest"],
                       "response_sha256": sha256(e["text"].encode())}
                try:
                    p = solver._parse(e["text"], SimpleNamespace(observation=obs))
                    row.update(old=evaluate(hydrate(old, card), p, revision),
                               repaired=evaluate(hydrate(SubgoalLedger, card), p, revision),
                               proposed_goal=plain(p.subgoal), proposed_transition=plain(p.transition))
                    if p.source:
                        row["proposed_program"] = p.source
                        if (record["episode_id"] == "eb_alfred:base:001" and
                                e["index"] in {44, 48}):
                            new_ledger = hydrate(SubgoalLedger, card)
                            new_ledger.accept_goal(p.subgoal, p.transition, revision)
                            row["retained_pickup_guard"] = new_ledger.check_action(105, obs)
                            assert row["retained_pickup_guard"] == "already_holding_before_pickup"
                    if p.perception_request:
                        row["perception_request"] = plain(p.perception_request)
                except ContractError:
                    row.update(old="model_output_format_error", repaired="model_output_format_error",
                               correction=solver.last_format_feedback)
                rows.append(row)
            elif e["kind"] == "execution":
                receipt = e["receipt"]
                if not receipt["tool_calls"] or "view_candidates" not in e["program_source"]:
                    continue
                wire, image = snapshots[receipt["receipt"]["snapshot_sha256"]]
                if wire["evidence_revision"] != 0:
                    continue
                binding, snapshot = reconstruct(wire, image)
                assert snapshot.snapshot_sha256 == wire["snapshot_sha256"]
                envelope = ExecutionEnvelope(snapshot.observation.episode_id, "diagnostic-session",
                                             snapshot.observation.mutation_revision,
                                             snapshot.observation.observation_revision,
                                             snapshot.decision_id, snapshot.snapshot_sha256)
                result = execute_isolated(e["program_source"], snapshot, binding, envelope,
                                          prepared=None, max_actions=30, capture_details=True)
                assert result.error_code is not None, (record["episode_id"], e["index"])
                assert any("confidence_gate" in r["reason_codes"]
                           for r in result.receipt["tool_receipts"])
                assert result.submission is None and not result.results
                tool_rows.append({
                    "version": record["version"], "episode_id": record["episode_id"],
                    "event": e["index"], "decision_id": receipt["decision_id"],
                    "original_program": e["program_source"], "original_results": e["results"],
                    "repaired_error": result.error_code, "repaired_action_submitted": False})
    assert len(rows) == 145 and len(tool_rows) == 16
    changed = [r for r in rows if r["old"] != r["repaired"]]
    print(json.dumps({"changed": [(r["episode_id"], r["response_event"]) for r in changed]}))
    assert len(changed) == 8
    assert Counter(r["episode_id"] for r in changed) == {
        "eb_alfred:base:000": 3, "eb_alfred:base:001": 2, "eb_alfred:base:003": 3}
    assert all(r["old"] == "subgoal_completion_not_supported" and r["repaired"] == "accepted" for r in changed)
    for r in rows:
        if r["episode_id"] == "eb_alfred:base:002" and r["response_event"] in {37, 74, 78, 82}:
            assert r["repaired"] == "subgoal_completion_not_supported"
    formats = Counter(r["correction"]["format_reason"] for r in rows if "correction" in r)
    assert formats == {"advance_source_none": 9, "invalid_perception_request": 1}
    output = {
        "schema": "embodied-raw-counterfactual-replay/1", "status": "passed",
        "scope": "Original proposals against original public cards; no counterfactual task-success claim.",
        "model_requests": 0, "simulator_launches": 0, "request_response_pairs": len(rows),
        "old_transition_outcomes": dict(Counter(r["old"] for r in rows)),
        "repaired_transition_outcomes": dict(Counter(r["repaired"] for r in rows)),
        "changed_transitions": len(changed), "format_reasons": dict(formats),
        "unprepared_tool_programs_denied": len(tool_rows), "requests": rows, "tools": tool_rows}
    with (HERE / "raw-replay.json").open("xb") as f:
        f.write(canonical(output) + b"\n")
    print(json.dumps({k: v for k, v in output.items() if k not in {"requests", "tools"}}))


if __name__ == "__main__":
    main()
