"""Offline repair acceptance on fake hosts and immutable public archived events."""
from __future__ import annotations

import argparse
from pathlib import Path

from action_contract import DecisionSnapshot, EpisodeBinding, public_hash, public_observation
from action_identity import source_identity, protocol_identity
from action_run import FakeClient, fake_run
from action_trajectory import restore
from audit_action_trajectory import audit
from budget import RunBudget
from protocol import canonical, json_object, sha256
from task_card import MAX_CHARS, SubgoalLedger, build_card, validate_card

ROOT = Path(__file__).resolve().parent.parent


class YieldClient(FakeClient):
    def chat(self, messages, **kwargs):
        if self.arm == "Direct":
            return super().chat(messages, **kwargs)
        self.calls += 1
        self.last_usage = {"prompt_tokens": 8, "completion_tokens": 8, "total_tokens": 16}
        self.last_response_identity = {"model": "offline-fake"}
        card = json_object(messages[1]["content"][0]["text"])
        goal = {"id": "find_chair", "goal": "Find the chair", "object": "chair",
                "destination": "", "complete_when": "visual"}
        value = {"subgoal": goal, "transition": {
            "kind": "keep", "source": "none",
            "observation_revision": card["identity"]["observation_revision"], "detail": ""}}
        if card["evidence"]["status"] == "not_requested":
            value["perception_request"] = {"targets": [], "depth": True}
        elif card["repair"].get("code") != "yield_observations":
            goal["goal"] = "Locate the chair using the current evidence"
            value["program_source"] = (
                "def solve(ctx):\n"
                "    assert ctx.task_state['goal']['goal'] == 'Locate the chair using the current evidence'\n"
                "    r = view_candidates()\n"
                "    YieldObservations([r['result_id']], 'inspect')\n")
        else:
            assert card["subgoal"]["goal"] == "Locate the chair using the current evidence"
            result_id = card["repair"]["yielded_result_ids"][0]
            goal["goal"] = "Search the current view for the chair"
            value["program_source"] = (
                "def solve(ctx):\n"
                "    assert ctx.task_state['goal']['goal'] == 'Search the current view for the chair'\n"
                f"    page = ReadObservation('{result_id}', ['candidates'])\n"
                "    assert page['scope']['snapshot_sha256'] == ctx.snapshot_sha256\n"
                "    assert page['data']['candidates'] == []\n"
                "    SubmitActions(ActionPayload(ctx.decision_id, ctx.observation_revision,"
                " ctx.snapshot_sha256, ctx.action_table_sha256, [0], []))\n")
        return canonical(value).decode()


def write_new(path, value):
    assert path.parent.is_dir() and not path.exists()
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def archived_replay(paths):
    episodes, cards = [], []
    for path in paths:
        raw = path.read_bytes()
        trajectory = json_object(raw)
        assert [json_object(line) for line in (path.parent / "events.jsonl").read_bytes().splitlines()] == (
            trajectory["events"])
        for asset in trajectory["assets"]:
            location = (path.parent / asset["path"]).resolve()
            assert location.is_relative_to(path.parent.resolve())
            assert location.stat().st_size == asset["bytes"]
            assert sha256(location.read_bytes()) == asset["sha256"]
        events = restore(trajectory["events"], path.parent)
        # Stopped partial work has no episode_result and is deliberately excluded.
        result = next(e["report"] for e in reversed(events) if e["kind"] == "episode_result")
        assert result["status"] == "finished"
        environment, episode_id = result["environment"], trajectory["episode_id"]
        ledger = SubgoalLedger(environment)
        budget = RunBudget("repair-replay", [environment]).episode(episode_id, environment)
        observation = None
        pending = pending_name = None
        accepted, rejected, advances, loops, blocked = [], [], [], [], []
        corrected_missing_goals = []
        episode_cards = []
        latest_goal_error = None
        for event in events:
            kind = event["kind"]
            if kind == "observation":
                public = event["observation"]
                binding = EpisodeBinding(
                    "repair-replay", episode_id, public["official_task_id"], environment,
                    public["subset"], event["receipt"]["input_sha256"])
                observation = public_observation(
                    binding, png=event["png"], instruction=public["instruction"],
                    environment_step=public["environment_step"],
                    mutation_revision=public["mutation_revision"],
                    observation_revision=public["observation_revision"],
                    action_table=[r["action_name"] for r in public["action_table"]],
                    public_feedback=public["public_feedback"], public_memory=public["public_memory"],
                    manipulation_input=public.get("manipulation_input"), planner_png=event.get("planner_png"))
                assert observation.pixel_sha256 == public["pixel_sha256"]
                ledger.observe(
                    observation, action=pending, action_name=pending_name,
                    succeeded="Last official action succeeded: true" in public["public_feedback"],
                    kind=event["mutation_kind"])
                if event["mutation_kind"] == "step":
                    pending = pending_name = None
                    if ledger.loop["switch_strategy"]:
                        loops.append({"revision": observation.observation_revision,
                                      "environment_step": observation.environment_step,
                                      **ledger.loop})
                budget.counts["steps"] = observation.environment_step
            elif kind == "action_submission":
                pending, pending_name = event.get("action", event.get("action_id")), event.get("action_name")
                conflict = ledger.check_action(pending, observation)
                if conflict:
                    blocked.append({"event": event["index"], "action": pending_name, "code": conflict})
            elif kind == "model_request":
                original = json_object(event["messages"][1]["content"][0]["text"])
                receipt = event["receipt"]
                snapshot = DecisionSnapshot(
                    observation, receipt["decision_id"], receipt["method_sha256"],
                    budget.sha256, task_state=ledger.public_state())
                repair = original["repair"]
                if repair.get("code") in {"subgoal_keep_changed", "subgoal_completion_not_supported"}:
                    repair = latest_goal_error or {}
                card, wire = build_card(snapshot, budget, repair=repair, evidence=original["evidence"])
                assert validate_card(wire) == card and card["subgoal"] == ledger.goal
                assert len(wire) <= MAX_CHARS
                if original["subgoal"] is None and ledger.goal is not None:
                    corrected_missing_goals.append(event["index"])
                row = {"event": event["index"], "chars": len(wire),
                       "sha256": sha256(wire.encode()), "card": card}
                cards.append(row)
                episode_cards.append(row)
                budget.counts["qwen"] += 1
                budget.decisions.setdefault(snapshot.decision_id, {}).setdefault("synthesis", 0)
                budget.decisions[snapshot.decision_id]["synthesis"] += 1
            elif kind == "model_response":
                try:
                    value = json_object(event["text"])
                    if not {"subgoal", "transition"} <= set(value):
                        continue
                    previous = ledger.goal
                    ledger.accept_goal(value["subgoal"], value["transition"],
                                       observation.observation_revision)
                except ValueError as error:
                    rejected.append({"event": event["index"], "code": str(error)})
                    latest_goal_error = ledger.repair_feedback(str(error))
                else:
                    latest_goal_error = None
                    accepted.append(event["index"])
                    if previous and value["transition"]["kind"] == "advance":
                        advances.append({"event": event["index"], "previous_id": previous["id"],
                                         "next_id": ledger.goal["id"],
                                         "source": value["transition"]["source"]})
        episodes.append({
            "episode_id": episode_id, "source_path": str(path), "source_sha256": sha256(raw),
            "cards": episode_cards, "accepted_goal_events": accepted, "rejected_goal_events": rejected,
            "advances_supported_by_repaired_ledger": advances, "navigation_loop_signals": loops,
            "guards_on_archived_actions": blocked,
            "corrected_missing_subgoal_request_events": corrected_missing_goals})
    assert len(episodes) == 7 and cards
    by_id = {r["episode_id"]: r for r in episodes}
    assert by_id["eb_habitat:base:000"]["corrected_missing_subgoal_request_events"]
    assert by_id["eb_alfred:base:001"]["advances_supported_by_repaired_ledger"]
    assert by_id["eb_habitat:base:001"]["advances_supported_by_repaired_ledger"]
    nav = by_id["eb_navigation:base:002"]
    assert any(r["cycle_period"] == 6 for r in nav["navigation_loop_signals"])
    assert any(r["code"] == "navigation_loop_action_blocked" for r in nav["guards_on_archived_actions"])
    return {
        "scope": "Repaired ledger on frozen public observations, actions and proposals; no resimulation.",
        "episodes": episodes, "episode_count": len(episodes), "card_count": len(cards),
        "max_chars": max(r["chars"] for r in cards), "all_cards_complete": True,
        "normal_required_overflow_cards": 0, "model_requests": 0, "simulator_launches": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.parent == ROOT / "checks" and not output.exists()
    prefix = output.stem
    integrations = []
    for name, options in (
        ("process", {"process_worker": True}),
        ("yield", {"process_worker": True, "environments": ("eb_navigation",),
                   "client_factory": YieldClient}),
    ):
        path = output.with_name(prefix + "-" + name + ".json")
        result = fake_run(task_cards=True, trajectory_dir=path.with_name(path.stem + "-episodes"),
                          **options)
        assert result["status"] == result["audit"]["status"] == "passed", result["audit"]
        write_new(path, result)
        checked = audit(path)
        integrations.append({"name": name, "run_path": str(path), "run_sha256": result["sha256"],
                             "attempt_count": len(result["attempts"]), "audit": checked})
    replay = archived_replay(sorted(
        (ROOT / "runs/task-card-15-full-r01-episodes").glob("trajectory-*/trajectory.json")))
    result = {"schema": "embodied-task-card-repair-acceptance/1", "status": "passed",
              "implementation_source": source_identity(), "protocol": protocol_identity(),
              "offline_integrations": integrations, "archived_replay": replay,
              "live_model_requests": 0, "official_simulator_launches": 0,
              "reference_answer_visible": False, "live_quality_improvement_evaluated": False}
    result["sha256"] = public_hash(result)
    write_new(output, result)
    print(canonical({"status": result["status"], "integration_attempts": sum(
        r["attempt_count"] for r in integrations), "replay_episodes": replay["episode_count"],
        "replay_cards": replay["card_count"], "max_chars": replay["max_chars"],
        "live_model_requests": 0, "official_simulator_launches": 0}).decode())


if __name__ == "__main__":
    main()
