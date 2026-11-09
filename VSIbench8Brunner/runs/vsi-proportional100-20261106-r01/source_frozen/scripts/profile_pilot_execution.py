#!/usr/bin/env python3
"""Read-only resource sampling and post-run timing analysis; never load labels."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


REMOTE_BASE = "/home/wangy/wangjh/VSIbench8Brunner"
REMOTE_SAMPLE = """
import fcntl,json,pathlib,time
base=pathlib.Path(%r)
root=base/"runs"/%r
launch=json.loads((base/(%r+".pid.json")).read_text())
proc=pathlib.Path("/proc")/str(launch["pid"])
alive=proc.exists()
if alive:
    fields=proc.joinpath("stat").read_text().rsplit(")",1)[1].split()
    if fields[19]!=launch["process_identity"]["start_ticks"]:
        raise RuntimeError("pilot process identity changed")
    alive=fields[0]!="Z"
status=json.loads((root/"live_status.json").read_text())
with (root/"request_budget.json").open() as budget_file:
    fcntl.flock(budget_file,fcntl.LOCK_SH)
    budget=json.load(budget_file)
cpu=[int(x) for x in pathlib.Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
disks={}
for line in pathlib.Path("/proc/diskstats").read_text().splitlines():
    fields=line.split()
    if fields[2] in ("sda","sdb","sdc"):
        disks[fields[2]]={"read_sectors":int(fields[5]),"write_sectors":int(fields[9]),
                         "busy_ms":int(fields[12])}
waits=[]
for task in status.get("active_tasks",[]):
    p=pathlib.Path("/proc")/str(task["pid"])
    if p.exists():
        waits.append({"kind":task["kind"],"key":task["key"],
                      "wait":p.joinpath("wchan").read_text().strip()})
print(json.dumps({"at":time.time(),"alive":alive,"status":status.get("status"),
    "completed":status.get("completed"),"active_qwen_requests":len(status.get("active_qwen_requests",[])),
    "pending_tasks":status.get("pending_tasks",{}),"cpu_count":status.get("cpu_count"),
    "load_average":status.get("load_average"),"memory_bytes":status.get("memory_bytes"),
    "cpu_ticks":cpu,"disks":disks,"worker_waits":waits,
    "preparation_requests_used":budget["used"],"preparation_requests_cap":budget["max_requests"],
    "reference_answers_read":False}))
"""


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def watch(args):
    root = args.run_dir
    code = REMOTE_SAMPLE % (REMOTE_BASE, root.name, args.launch_name)
    command = [
        "ssh", "-S", args.control_path, "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15", "wangy@10.10.0.34",
        "python3 -c " + shlex.quote(code),
    ]
    with (root / ".execution-profile.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root / "execution_resource_samples.jsonl").open("a", buffering=1) as output:
            deadline = time.monotonic() + 5400
            while time.monotonic() < deadline:
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                if result.returncode:
                    raise RuntimeError("read-only resource query failed")
                sample = json.loads(result.stdout)
                output.write(json.dumps(sample) + "\n")
                print(json.dumps({key: sample[key] for key in (
                    "at", "completed", "active_qwen_requests", "status")}), flush=True)
                if not sample["alive"]:
                    return
                time.sleep(30)
            raise RuntimeError("read-only resource sampling reached its time bound")


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values), "sum_s": sum(values), "mean_s": sum(values) / len(values),
        "p50_s": values[(len(values) - 1) // 2],
        "p95_s": values[int((len(values) - 1) * .95)], "max_s": values[-1],
    }


def analyze(args):
    root = args.run_dir
    status = json.loads((root / "live_status.json").read_text())
    settings = json.loads((root / "parallel_execution.json").read_text())
    events = [json.loads(line) for line in (root / "parallel_events.jsonl").read_text().splitlines()]
    stages = {
        stage: distribution([row["elapsed_s"] for row in events
                             if row["kind"] == stage and row["status"] == "finished"])
        for stage in ("prepare", "solve")
    }
    requests, intervals, latency = [], [], []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for path in sorted((root / "solver_requests").glob("*/*.json")):
        row = json.loads(path.read_text())
        requests.append({"qa_id": path.parent.name, "index": path.stem,
                         "status": row["status"], "automatic_retry": row["automatic_retry"]})
        if row["status"] != "returned":
            continue
        duration = row.get("network_elapsed_s", row["elapsed_s"])
        latency.append(duration)
        intervals.extend([(row["started_at"], 1), (row["started_at"] + duration, -1)])
        for key in usage:
            usage[key] += row.get("raw_response", {}).get("usage", {}).get(key, 0)
    active, peak = 0, 0
    for _, delta in sorted(intervals):
        active += delta
        peak = max(peak, active)
    program_errors = []
    for path in sorted((root / "program_executions").glob("*/*.json")):
        row = json.loads(path.read_text())
        cell = row.get("cell") or {}
        if cell.get("error_code") or cell.get("error_type") or row.get("status") == "failed":
            program_errors.append({
                "qa_id": path.parent.name, "execution_index": path.stem,
                "file": str(path.relative_to(root)),
                "full_trajectory": f"trajectories/{path.parent.name}.json",
                "error_code": cell.get("error_code"),
                "error_type": cell.get("error_type") or row.get("error_type"),
                "execution_status": row.get("status", "legacy_returned"),
            })
    write(root / "trajectory_error_index.json", {
        "schema": "pilot-program-error-index/1", "reference_answers_read": False,
        "scope": "Current program journals; archived attempts remain in each full trajectory.",
        "n_program_errors": len(program_errors), "records": program_errors,
        "host_error_files": [str(path.relative_to(root)) for path in sorted(
            (root / "host_errors").glob("*.json"))],
        "archived_host_error_files": [str(path.relative_to(root)) for path in sorted(
            (root / "attempt_history").glob("*/host_errors/*.json"))],
        "execution_resume_history": (
            "execution_resume_history.json" if (root / "execution_resume_history.json").exists() else None),
        "unreturned_solver_requests": [row for row in requests if row["status"] != "returned"],
    })
    samples_path = root / "execution_resource_samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()] if samples_path.exists() else []
    rates = []
    worker_waits = {}
    for sample in samples:
        for worker in sample["worker_waits"]:
            waits = worker_waits.setdefault(worker["kind"], {})
            waits[worker["wait"]] = waits.get(worker["wait"], 0) + 1
    for before, after in zip(samples, samples[1:]):
        seconds = after["at"] - before["at"]
        deltas = [y - x for x, y in zip(before["cpu_ticks"], after["cpu_ticks"])]
        total = sum(deltas)
        if seconds <= 0 or total <= 0 or min(deltas) < 0:
            continue
        rates.append({
            "at": after["at"], "interval_s": seconds,
            "cpu_busy_percent": 100 * (total - deltas[3] - deltas[4]) / total,
            "cpu_iowait_percent": 100 * deltas[4] / total,
            "disk_rates": {
                device: {
                    "read_MB_s": (values["read_sectors"] - before["disks"][device]["read_sectors"]) * 512 / seconds / 1e6,
                    "write_MB_s": (values["write_sectors"] - before["disks"][device]["write_sectors"]) * 512 / seconds / 1e6,
                    "busy_percent": (values["busy_ms"] - before["disks"][device]["busy_ms"]) / seconds / 10,
                } for device, values in after["disks"].items() if device in before["disks"]
            },
        })
    sampled_seconds = sum(row["interval_s"] for row in rates)
    resource_summary = {
        "sampled_interval_s": sampled_seconds,
        "cpu_busy_mean_percent": (
            sum(row["cpu_busy_percent"] * row["interval_s"] for row in rates) / sampled_seconds
            if sampled_seconds else None),
        "cpu_iowait_mean_percent": (
            sum(row["cpu_iowait_percent"] * row["interval_s"] for row in rates) / sampled_seconds
            if sampled_seconds else None),
        "minimum_available_memory_bytes": min(
            (row["memory_bytes"]["MemAvailable"] for row in samples), default=None),
        "worker_wait_observations_by_stage": worker_waits,
        "disk_summary": {},
        "scope": "Host-wide resource use during sampling; includes other host workloads.",
    }
    for device in sorted({key for row in rates for key in row["disk_rates"]}):
        device_rows = [row for row in rates if device in row["disk_rates"]]
        seconds = sum(row["interval_s"] for row in device_rows)
        resource_summary["disk_summary"][device] = {
            f"mean_{key}": sum(row["disk_rates"][device][key] * row["interval_s"]
                              for row in device_rows) / seconds
            for key in ("read_MB_s", "write_MB_s", "busy_percent")
        }
        resource_summary["disk_summary"][device]["peak_busy_percent"] = max(
            row["disk_rates"][device]["busy_percent"] for row in device_rows)
    history_path = root / "execution_resume_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {"epochs": []}
    preserved_qa_ids = set()
    for epoch in history["epochs"]:
        for qa, expected in epoch.get("preserved_outcome_sha256", {}).items():
            digest = hashlib.sha256()
            with (root / "outcomes" / f"{qa}.json").open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError("completed outcome changed across parallel continuation")
            preserved_qa_ids.add(qa)
    elapsed = status["elapsed_s"] + sum(
        epoch["final_status"]["elapsed_s"] for epoch in history["epochs"])
    finished_at = (root / "live_status.json").stat().st_mtime
    first_settings_at = min([settings["at"], *(
        epoch["settings"]["at"] for epoch in history["epochs"])])
    current_span = max(0., finished_at - settings["at"])
    summary = {
        "schema": "pilot-execution-performance/1", "status": status["status"],
        "completed": status["completed"], "parallel_wall_s": elapsed,
        "current_parallel_epoch_wall_s": status["elapsed_s"],
        "previous_parallel_epochs": history["epochs"],
        "preserved_outcome_qa_ids_verified": sorted(preserved_qa_ids),
        "parallel_wall_definition": "Sum of drained and current parallel epoch wall times; initial serial episodes and gaps between epochs are excluded.",
        "parallel_span_wall_s": max(0., finished_at - first_settings_at),
        "parallel_span_definition": "From first scheduler-settings publication to final status write; includes startup exports and the continuation gap, excludes earlier serial episodes.",
        "current_epoch_startup_and_export_s": max(0., current_span - status["elapsed_s"]),
        "parallel_stages": stages, "solver_request_latency": distribution(latency),
        "peak_qwen_concurrency_from_request_intervals": peak, "solver_token_usage": usage,
        "settings": settings, "solver_requests": requests,
        "program_error_count": len(program_errors),
        "program_error_index": "trajectory_error_index.json",
        "resource_sample_count": len(samples), "resource_interval_rates": rates,
        "resource_summary": resource_summary,
        "peak_sampled_qwen_concurrency": max((row["active_qwen_requests"] for row in samples), default=None),
        "observed_overlap_ratio": sum(row.get("sum_s", 0) for row in stages.values()) / elapsed if elapsed else None,
        "overlap_ratio_definition": "Sum of finished parallel task wall times divided by pipeline wall time; not a controlled speedup estimate.",
        "latency_scope": "Current solver journals; earlier archived failed requests are retained in full trajectories.",
        "resource_scope": "Read-only samples begin after the parallel launch; CPU busy excludes idle and iowait.",
        "reference_answers_read": False,
    }
    write(root / "execution_performance.json", summary)
    print(json.dumps({key: summary[key] for key in (
        "status", "completed", "parallel_wall_s", "peak_qwen_concurrency_from_request_intervals",
        "resource_sample_count")}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("watch", "analyze"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--launch-name", default="pilot40-parallel01")
    parser.add_argument("--control-path")
    args = parser.parse_args()
    if args.mode == "watch" and not args.control_path:
        parser.error("watch requires --control-path")
    (watch if args.mode == "watch" else analyze)(args)


if __name__ == "__main__":
    main()
