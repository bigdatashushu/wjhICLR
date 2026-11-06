"""Audit real household pilot traces and image identity without API calls."""
from __future__ import annotations

import argparse
import json
import os

from protocol import ROOT, sha256, within_root
from runner import write_json


def main():
    for key in tuple(os.environ):
        if any(x in key.upper() for x in
               ("TOKEN", "API_KEY", "PASSWORD", "SECRET", "CREDENTIAL", "ENDPOINT")):
            os.environ.pop(key)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run = within_root(args.run_dir)
    startup = json.loads((run / "startup.json").read_text())
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    results = [json.loads((run / f"episode-{i:03d}/result.json").read_text()) for i in range(3)]
    traces = [[json.loads(line) for line in (run / f"episode-{i:03d}/trace.jsonl").read_text().splitlines()]
              for i in range(3)]
    requests = [[row for row in trace if row["kind"] == "model_request"] for trace in traces]
    steps = [[row for row in trace if row["kind"] == "environment_step"] for trace in traces]
    checks = {
        "complete_real_three_episodes": summary["status"] == "complete"
            and summary["planned_tasks"] == summary["completed_tasks"] == len(results) == 3
            and all(r["status"] == "complete" for r in results),
        "source_identity_preserved": all(sha256((ROOT / name).read_bytes()) == digest
                                         for name, digest in startup["source_sha256"].items()),
        "model_client_identity_preserved": sha256(
            (ROOT.parent / "src/skill3d/synthesis/bailian_client.py").read_bytes()) == startup["existing_client_sha256"],
        "fixed_episode_indexes": config["indexes"] == [r["index"] for r in results] == [0, 1, 2],
        "summary_results_match": results == summary["results"],
        "success_count_and_rate_match": summary["successful_tasks"] == sum(r["task_success"] for r in results)
            and summary["success_rate"] == sum(r["task_success"] for r in results) / 3,
        "request_counts_match": summary["model_requests"] == sum(map(len, requests))
            and all(len(q) == r["model_requests"] for q, r in zip(requests, results)),
        "step_counts_match": summary["num_steps"] == sum(map(len, steps))
            and all(len(q) == r["num_steps"] for q, r in zip(steps, results)),
        "budgets_respected": summary["model_requests"] <= config["max_requests"] == 120
            and all(r["model_requests"] <= 40 and r["num_steps"] <= 30
                    and r["num_invalid_actions"] <= 10 for r in results),
        "official_configuration": config["n_shots"] == summary["n_shots"] == 10
            and config["resolution"] == 500 and config["max_environment_steps"] == 30
            and config["max_invalid_actions"] == 10 and config["chat_history"] is False
            and config["env_feedback"] is True and config["multistep"] is False
            and config["detection_box"] is False,
        "qwen_only_without_tools": config["model"] == "qwen3.8-flash"
            and config["temperature"] == 0 and config["enable_thinking"] is False
            and summary["geometry_tools_enabled"] is False
            and summary["skill_evolution_enabled"] is False
            and all(r["model_tool_calls"] == 0 and r["skill_sha256"] is None
                    for group in requests for r in group),
        "provider_reported_model_matches": all(
            r["reported_model_sha256"] == sha256(config["model"].encode())
            for group in requests for r in group),
        "references_hidden": summary["reference_answer_visible"] is False
            and all(r["reference_answer_visible"] is False for group in requests for r in group),
        "no_automatic_retry": summary["automatic_retry"] is False and startup["automatic_retry"] is False
            and all(r["status"] in ("valid_plan", "invalid_plan", "empty_plan")
                    for group in requests for r in group),
        "one_current_rgb_per_request": all(r["input_image_count"] == r["message_count"] == 1
                                           for group in requests for r in group),
        "sequential_steps": all([r["step"] for r in group] == list(range(1, len(group) + 1))
                                for group in steps),
        "sequential_requests": all([r["request_index"] for r in group] == list(range(1, len(group) + 1))
                                   for group in requests),
    }
    images_valid = True
    inputs_valid = True
    for i, (queries, actions, result) in enumerate(zip(requests, steps, results)):
        folder = run / f"episode-{i:03d}"
        inp = json.loads((folder / "input.json").read_text())
        images_valid &= sha256((folder / "step-000.png").read_bytes()) == inp["image_sha256"]
        for row in actions:
            images_valid &= sha256((folder / f'step-{row["step"]:03d}.png').read_bytes()) == row["image_sha256"]
        for row in queries:
            images_valid &= sha256((folder / f'step-{row["environment_step"]:03d}.png').read_bytes()) == row["image_sha256"]
            inputs_valid &= row["input_sha256"] == inp["input_sha256"] == result["input_sha256"]
            inputs_valid &= row["action_space_sha256"] == inp["action_space_sha256"]
        inputs_valid &= inp["reference_answer_visible"] is False
    checks.update(actual_image_hashes_match=bool(images_valid), episode_input_identity_preserved=bool(inputs_valid))
    receipt = {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
               "tasks": 3, "model_requests": summary["model_requests"],
               "environment_steps": summary["num_steps"], "audit_model_requests": 0}
    write_json(run / "audit.json", receipt)
    print(json.dumps(receipt))
    return 0 if receipt["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
