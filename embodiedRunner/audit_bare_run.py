"""Audit saved pilot identity, request counts and outcomes without model calls."""
import argparse
from collections import Counter
import os
import sys

sys.dont_write_bytecode = True
from protocol import ROOT, json_object, sha256, within_root
from runner import write_json

for name in tuple(os.environ):
    if any(part in name.upper() for part in
           ("TOKEN", "API_KEY", "PASSWORD", "SECRET", "CREDENTIAL", "ENDPOINT")):
        os.environ.pop(name)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", nargs="?", default="runs/qwen-bare-official-3shot-pilot")
args = parser.parse_args()
run = within_root(args.run_dir)
startup = json_object((run / "startup.json").read_text())
summary = json_object((run / "summary.json").read_text())
config = json_object((run / "config.json").read_text())
rows = [json_object((run / "episodes" / (task["task_id"].replace(":", "-") + ".json")).read_text())
        for task in startup["tasks"]]
trace = [json_object(line) for line in (run / "trace.jsonl").read_text().splitlines()]
calls = [row for row in trace if "request_index" in row]
call_keys = {(row["task_id"], row["request_index"]) for row in calls}
steps = [row for row in trace if row.get("event") == "environment_step"]
checks = {
    "frozen_sources_match": all(sha256((ROOT / name).read_bytes()) == digest
                                for name, digest in startup["source_sha256"].items()),
    "registered_tasks_preserved": all(
        all(row[key] == task[key] for key in task) for row, task in zip(rows, startup["tasks"])),
    "denominator_preserved": len(rows) == summary["planned_tasks"] == 3,
    "success_count_matches": sum(row["task_success"] for row in rows) == summary["successful_tasks"],
    "success_rate_matches": summary["task_success_rate"] == sum(row["task_success"] for row in rows) / len(rows),
    "status_counts_match": dict(Counter(row["status"] for row in rows)) == summary["statuses"],
    "request_counts_match": len(call_keys) == sum(row["planner_requests"] for row in rows) == summary["model_requests"],
    "action_counts_match": len(steps) == sum(row["environment_steps"] for row in rows),
    "budgets_respected": summary["model_requests"] <= config["max_requests"] <= 60
                         and all(row["planner_requests"] <= 20 and row["environment_steps"] <= 20 for row in rows),
    "bare_model_identity": config["model"] == "qwen3.8-flash" and config["mode"] == "direct"
                          and config["n_shots"] == 3 and config["enable_thinking"] is False,
    "official_input_receipts": all(row["input_image_count"] == 1 and row["message_count"] == 1 for row in calls),
    "no_skill_or_tools": startup["skill_sha256"] is None and not startup["geometry_tools_enabled"]
                         and not startup["skill_evolution_enabled"]
                         and all(row["model_tool_calls"] == 0 for row in calls),
    "reference_visibility_receipts": all(row["reference_answer_visible"] is False for row in trace + rows),
    "no_automatic_retry": summary["automatic_retry"] is False and startup["automatic_retry"] is False,
}
audit = {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
         "tasks": len(rows), "model_requests": summary["model_requests"],
         "environment_steps": len(steps), "audit_model_requests": 0}
write_json(run / "audit.json", audit)
print(__import__("json").dumps(audit, ensure_ascii=False))
raise SystemExit(0 if audit["status"] == "pass" else 2)
