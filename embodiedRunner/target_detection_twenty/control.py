"""Run the existing R05 method on the exact historical twenty tasks."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from target_detection_r05.__main__ import check
from target_detection_r05.client import ModelClient
from target_detection_r05.config import load_config
from target_detection_r05.lifecycle import REMOTE_STATE, preserved_local, remote
from target_detection_r05.perception import Detector
from target_detection_r05.protocol import Task
from target_detection_r05.runner import suite
from target_detection_r05.sources import Sources, canonical, file_digest, implementation_hashes
from target_detection_r05.transport import Session, preflight

OUT = ROOT / "target_detection_twenty"
VERIFY = OUT / "verification"
CONFIG = OUT / "experiment.json"
RUN_ID = "target-detection-twenty-20261108-r01"
RUNS = ROOT / "target_detection_r05/runs"
BASELINE = ROOT / "rgbd_twenty/rgbd20-results.json"
ENV_ORDER = ("alfred", "habitat", "manipulation", "navigation")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def source_files():
    return {p.name: file_digest(p) for p in sorted(OUT.glob("*.py"))}


def old_local_files():
    rows = preserved_local()
    for path in sorted((ROOT / "target_detection_r05").rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT / "target_detection_r05")
        if len(relative.parts) >= 2 and relative.parts[0] == "runs" and relative.parts[1].startswith(RUN_ID):
            continue
        rows[str(path.relative_to(ROOT))] = file_digest(path)
    for name in ("runs/task-card-full-r06.json", "runs/task-card-15-full-context-r04.json",
                 "runs/trajectory-followup-r06-20261106-evidence.json"):
        path = ROOT / name
        if path.exists():
            rows[name] = file_digest(path)
    return rows


def records():
    path = VERIFY / "run-records.json"
    return load(path) if path.exists() else []


def attempted():
    rows = {}
    for record in records():
        directory = Path(record["directory"])
        for task in record["tasks"]:
            label = Task(**task)
            folder = directory / label.directory
            if (folder / "trajectory.jsonl").exists():
                assert label.identity not in rows, "task_was_started_twice"
                rows[label.identity] = folder
    return rows


def verify_identity():
    config, tasks = load_config(CONFIG)
    frozen = load(VERIFY / "frozen.json")
    original = load(ROOT / "target_detection_r05/verification/frozen.json")
    assert implementation_hashes() == frozen["implementation_hashes"] == original["implementation_hashes"]
    assert file_digest(CONFIG) == frozen["config_sha256"]
    assert source_files() == frozen["experiment_source_hashes"]
    assert old_local_files() == load(VERIFY / "preserved-local.json")
    sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
    check(config, tasks, sources)
    assert sources.hashes == frozen["source_hashes"]
    return config, tasks, sources


def prepare():
    assert not (VERIFY / "frozen.json").exists(), "experiment_already_prepared"
    baseline = load(BASELINE)
    assert baseline["status"] == "verified" and baseline["completed"] == baseline["selected"] == 20
    rows = list(baseline["pairs"].values())
    selection = sorted([row["task"] for row in rows],
                       key=lambda t: (ENV_ORDER.index(t["environment"]), t["index"]))
    expected = {(e, "base", i) for e in ENV_ORDER for i in range(5)}
    assert {(t["environment"], t["subset"], t["index"]) for t in selection} == expected
    config = load(ROOT / "target_detection_r05/experiment-20261108-r05.json")
    config["episodes"] = selection
    for name in ("config-rgbd_trial.json", "config-rgbd_three_envs.json"):
        prior = load(ROOT / "rgbd_twenty" / name)
        for field in ("model", "n_shots", "max_requests"):
            assert prior[field] == config[field], "historical_model_or_budget_mismatch"
        for field in ("python", "gpu", "source_root", "environment_seed",
                      "manipulation_catalog", "manipulation_dataset"):
            assert prior["runtime"][field] == config["runtime"][field]
    save(CONFIG, config)
    config, tasks = load_config(CONFIG)
    sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
    offline = check(config, tasks, sources)
    original = load(ROOT / "target_detection_r05/verification/frozen.json")
    assert implementation_hashes() == original["implementation_hashes"]
    assert sources.hashes == original["source_hashes"]
    assert not (RUNS / RUN_ID).exists(), "new_run_directory_already_exists"
    baseline_index = {}
    for row in rows:
        task = Task(**row["task"])
        result_path, trace_path = Path(row["rgbd_result"]), Path(row["rgbd_trace"])
        result = load(result_path)
        assert result["task"] == task.as_dict()
        assert result["metrics"]["task_success"] == row["rgbd_success"]
        baseline_index[task.identity] = {
            "task": task.as_dict(), "result": str(result_path), "trace": str(trace_path),
            "result_sha256": file_digest(result_path), "trace_sha256": file_digest(trace_path),
            "rgbd_success": row["rgbd_success"], "rgb_success": row["baseline_success"],
            "rgbd_status": row["rgbd_status"], "rgbd_requests": row["rgbd_requests"],
            "rgbd_steps": row["rgbd_steps"],
        }
    save(VERIFY / "historical-index.json", baseline_index)
    save(VERIFY / "offline-check.json", offline)
    save(VERIFY / "preserved-local.json", old_local_files())
    state = remote(config, REMOTE_STATE)
    assert not state["owned_trial_workers"], "existing_r05_worker_is_active"
    assert state["trial_implementation_hashes"] == implementation_hashes()
    assert state["disk_free_bytes"] > 4 * 1024**3
    save(VERIFY / "server-before.json", state)
    frozen = {
        "schema": "target-detection-twenty-frozen/1", "created_at": now(),
        "implementation_package": "target_detection_r05",
        "implementation_hashes": implementation_hashes(),
        "experiment_source_hashes": source_files(), "source_hashes": sources.hashes,
        "config_sha256": file_digest(CONFIG), "selected_tasks": selection,
        "baseline_index_sha256": file_digest(VERIFY / "historical-index.json"),
        "baseline_results_sha256": file_digest(BASELINE),
        "run_id": RUN_ID, "selected": 20, "reference_answer_visible": False,
        "automatic_retry": False, "model_requests": 0, "simulator_launches": 0,
        "maximum_model_requests": sum(config["max_requests"][t.environment] for t in tasks),
    }
    save(VERIFY / "frozen.json", frozen)
    print(canonical({"status": "prepared", "selected": 20, "run_id": RUN_ID,
                     "maximum_model_requests": frozen["maximum_model_requests"],
                     "implementation_package": "target_detection_r05",
                     "model_requests": 0, "simulator_launches": 0}).decode())


def probe():
    config, tasks, sources = verify_identity()
    reports = {}
    for task in tasks:
        reports[task.identity] = preflight(config, task, sources)
    detector = Detector(config).status()
    save(VERIFY / "preflight.json", {"status": "runtime_ready", "checked_at": now(),
         "reports": reports, "detector": detector, "model_requests": 0, "simulator_launches": 0})
    print(canonical({"status": "runtime_ready", "tasks": len(reports),
                     "model_requests": 0, "simulator_launches": 0}).decode())


def run(args):
    config, tasks, sources = verify_identity()
    ready = load(VERIFY / "preflight.json")
    assert ready["status"] == "runtime_ready"
    prior = attempted()
    if args.remaining:
        tasks = [t for t in tasks if t.identity not in prior]
    else:
        assert not prior, "use_remaining_to_start_only_unstarted_tasks"
    assert tasks, "no_unstarted_tasks"
    run_id = args.run_id or RUN_ID
    assert run_id.startswith(RUN_ID) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id)
    directory = RUNS / run_id
    assert not directory.exists(), "run_directory_already_exists"
    state = remote(config, REMOTE_STATE)
    assert not state["owned_trial_workers"] and state["trial_implementation_hashes"] == implementation_hashes()
    Detector(config).status()
    key_path = ROOT.parent / "qwenkey.md"
    assert not key_path.stat().st_mode & 0o077, "authorized_key_permissions_not_private"
    keys = set(re.findall(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_.+/-]{16,}",
                          key_path.read_text(encoding="utf-8")))
    assert len(keys) == 1, "authorized_key_not_unique"
    key = keys.pop()
    os.environ["DASHSCOPE_API_KEY"] = key
    os.environ["DASHSCOPE_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    record = {
        "run_id": run_id, "directory": str(directory), "started_at": now(),
        "coordinator_pid": os.getpid(), "tasks": [t.as_dict() for t in tasks],
        "implementation_hashes": implementation_hashes(), "config_sha256": file_digest(CONFIG),
        "automatic_retry": False, "remaining_only": args.remaining,
    }
    previous = records()
    previous.append(record)
    save(VERIFY / "run-records.json", previous)
    try:
        client = ModelClient(config["model"])
        client.assert_available()

        def factory(task, local_dir, journal):
            return Session(config, task, local_dir, journal, sources, run_id)

        result = suite(tasks, config, sources, client, factory, directory)
        record["finished_at"], record["summary"] = now(), result
        save(VERIFY / "run-records.json", previous)
        print(canonical({"run_id": run_id, **result}).decode())
        return 0 if result["status"] == "finished" else 2
    finally:
        os.environ.pop("DASHSCOPE_API_KEY", None)
        os.environ.pop("DASHSCOPE_BASE_URL", None)
        del key


def status():
    _, tasks, _ = verify_identity()
    folders = attempted()
    rows = []
    for task in tasks:
        folder = folders.get(task.identity)
        row = {"task": task.as_dict(), "status": "not_started"}
        if folder:
            events = [json.loads(s) for s in (folder / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
            row.update(last_event=events[-1]["type"],
                       model_requests=sum(e["type"] == "model_request" for e in events),
                       actions=sum(e["type"] == "action_sent" for e in events),
                       acks=sum(e["type"] == "action_ack" for e in events))
            result = folder / "result.json"
            if result.exists():
                outcome = load(result)
                row.update(status=outcome["status"], success=outcome["metrics"]["task_success"],
                           error=outcome["error"], perception=outcome["perception"])
            else:
                row["status"] = "running"
        rows.append(row)
    output = {"checked_at": now(), "selected": 20,
              "completed": sum(r["status"] not in ("running", "not_started") for r in rows),
              "successes": sum(r.get("success") == 1 and r["status"] == "official_done" for r in rows),
              "rows": rows}
    save(VERIFY / "progress.json", output)
    print(canonical(output).decode())


def after():
    config, _, _ = verify_identity()
    before = load(VERIFY / "server-before.json")
    state = remote(config, REMOTE_STATE)
    state["old_evidence_unchanged"] = state["preserved_old_evidence"] == before["preserved_old_evidence"]
    state["local_old_files_unchanged"] = old_local_files() == load(VERIFY / "preserved-local.json")
    state["implementation_unchanged"] = state["trial_implementation_hashes"] == implementation_hashes()
    assert state["old_evidence_unchanged"] and state["local_old_files_unchanged"] and state["implementation_unchanged"]
    assert not state["owned_trial_workers"], "owned_runtime_not_closed"
    state["detector"] = Detector(config).status()
    save(VERIFY / "server-after.json", state)
    print(canonical({k: state[k] for k in ("status", "checked_at", "old_evidence_unchanged",
                                          "local_old_files_unchanged", "owned_trial_workers")}).decode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "probe", "run", "status", "after"))
    parser.add_argument("--run-id")
    parser.add_argument("--remaining", action="store_true")
    args = parser.parse_args()
    return run(args) if args.phase == "run" else globals()[args.phase]()


if __name__ == "__main__":
    raise SystemExit(main())
