"""Read public archived events and build complete cards; never call a service."""
from __future__ import annotations

import argparse
from pathlib import Path

from action_contract import DecisionSnapshot, EpisodeBinding, public_observation, public_hash
from action_trajectory import restore
from budget import RunBudget
from protocol import ROOT, canonical, json_object, sha256
from task_card import MAX_CHARS, SubgoalLedger, build_card, validate_card


def replay(paths):
    episodes = []
    all_cards = []
    for path in paths:
        trajectory = json_object(Path(path).read_bytes())
        result = next(e["report"] for e in reversed(trajectory["events"])
                      if e["kind"] == "episode_result")
        if result["status"] != "finished":
            continue
        environment, episode_id = result["environment"], trajectory["episode_id"]
        ledger = SubgoalLedger(environment)
        budget = RunBudget("replay", [environment]).episode(episode_id, environment)
        observation = None
        pending = pending_name = None
        conflicts = []
        cards = []
        for event in trajectory["events"]:
            kind = event["kind"]
            if kind == "action_submission":
                pending = event.get("action", event.get("action_id"))
                pending_name = event.get("action_name")
                if pending is not None and observation is not None:
                    conflict = ledger.check_action(pending, observation)
                    if conflict:
                        conflicts.append({"event": event["index"], "code": conflict})
            elif kind == "observation":
                event = restore(event, Path(path).parent)
                public = event["observation"]
                binding = EpisodeBinding("replay", episode_id, public["official_task_id"],
                                         environment, public["subset"], event["receipt"]["input_sha256"])
                observation = public_observation(
                    binding, png=event["png"], instruction=public["instruction"],
                    environment_step=public["environment_step"], mutation_revision=public["mutation_revision"],
                    observation_revision=public["observation_revision"],
                    action_table=[row["action_name"] for row in public["action_table"]],
                    public_feedback=public["public_feedback"],
                    manipulation_input=public.get("manipulation_input"),
                    planner_png=event.get("planner_png"))
                ledger.observe(
                    observation, action=pending, action_name=pending_name,
                    succeeded="Last official action succeeded: true" in public["public_feedback"],
                    kind=event["mutation_kind"])
                if event["mutation_kind"] == "step":
                    pending = pending_name = None
                budget.counts["steps"] = observation.environment_step
            elif kind == "model_request" and observation is not None:
                old = event["receipt"]
                snapshot = DecisionSnapshot(
                    observation, old["decision_id"], old["method_sha256"], budget.sha256,
                    task_state=ledger.public_state())
                card, wire = build_card(snapshot, budget)
                assert validate_card(wire) == card and len(wire) <= MAX_CHARS
                sample = {"source_event": event["index"], "chars": len(wire),
                          "sha256": sha256(wire.encode()), "card": card}
                cards.append(sample)
                all_cards.append(sample)
                budget.counts["qwen"] += 1
        episodes.append({"episode_id": episode_id, "source_path": str(path),
                         "source_sha256": sha256(Path(path).read_bytes()),
                         "cards": cards, "recognized_action_conflicts": conflicts})
    assert all_cards
    result = {
        "schema": "embodied-task-card-replay/1", "status": "passed",
        "scope": "public archived observations/actions; rebuilt task context, no model decisions or outcomes",
        "episodes": episodes, "episode_count": len(episodes), "card_count": len(all_cards),
        "max_chars": max(card["chars"] for card in all_cards),
        "min_chars": min(card["chars"] for card in all_cards), "limit": MAX_CHARS,
        "all_cards_complete": True, "model_requests": 0, "simulator_launches": 0,
        "reference_answer_visible": False,
    }
    result["sha256"] = public_hash(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    folders = [args.archive / "comparison-episodes",
               args.archive / "batch-2/comparison-episodes",
               args.archive / "batch-3/comparison-episodes"]
    paths = [p for folder in folders for p in sorted(folder.glob("trajectory-*-Full/trajectory.json"))]
    paths.append(ROOT / "runs/manipulation-smoke-r02-episodes/trajectory-000-Full/trajectory.json")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or not output.parent.is_dir() or output.exists():
        parser.error("output must be a new workspace file")
    result = replay(paths)
    with output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(canonical({key: result[key] for key in
                     ("status", "episode_count", "card_count", "max_chars", "limit",
                      "all_cards_complete", "model_requests")}).decode())
