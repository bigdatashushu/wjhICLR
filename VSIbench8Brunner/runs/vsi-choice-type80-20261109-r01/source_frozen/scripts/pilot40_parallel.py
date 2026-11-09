#!/usr/bin/env python3
"""Schedule frozen pilot scenes and episodes in isolated Linux processes."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

import pilot40 as pilot
from skill3d.infra.vsi_preparation_budget import ENV


def install_shared_reviewer_lock(lock):
    from skill3d.segmentation.intern_binary_review import InternBinaryClient
    original = InternBinaryClient.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._binary_slot = lock

    InternBinaryClient.__init__ = initialize


def worker(kind, tasks, events, stop, cpu_slot, api_slot, review_lock,
           root, cfg, source_provider, overview, predictor, questions):
    # Fork happens before the parent starts queue feeder/monitor threads.
    provider = type(source_provider).__new__(type(source_provider))
    provider.__setstate__(source_provider.__getstate__())
    pilot.CPU_SLOT, pilot.API_SLOT = cpu_slot, api_slot
    pilot.CURRENT["root"] = root
    install_shared_reviewer_lock(review_lock)
    pilot.install_execution_recorder(root)
    while True:
        task = tasks.get()
        if task is None:
            return
        key, qa_ids = task if kind == "prepare" else (task, [task])
        if stop.is_set():
            events.put({"kind": kind, "key": key, "status": "cancelled", "qa_ids": qa_ids})
            continue
        pilot.CURRENT["qa_id"] = qa_ids[0]
        events.put({"kind": kind, "key": key, "status": "started", "pid": os.getpid(),
                    "qa_ids": qa_ids, "at": time.time()})
        started = time.time()
        try:
            with pilot.cpu_admission():
                if kind == "prepare":
                    pilot.prepare_scene(root, cfg, provider, predictor, questions, key, qa_ids)
                else:
                    pilot.solve_question(root, cfg, provider, overview, questions[key], key)
            events.put({"kind": kind, "key": key, "status": "finished",
                        "qa_ids": qa_ids, "elapsed_s": time.time() - started})
        except Exception as exc:
            stop.set()
            error = {
                "stage": kind, "error_type": type(exc).__name__,
                "stack": [{"file": frame.filename, "line": frame.lineno,
                           "function": frame.name}
                          for frame in traceback.extract_tb(exc.__traceback__)],
                "reference_answer_visible": False,
            }
            qa_id = pilot.CURRENT["qa_id"]
            pilot.write(root / "host_errors" / f"{qa_id}.json", error)
            pilot.export_question(root, qa_id, status="failed", error=error)
            events.put({"kind": kind, "key": key, "status": "failed", "qa_ids": qa_ids,
                        "error": error, "elapsed_s": time.time() - started})
        finally:
            provider.cache.clear()
            provider._publication_files.clear()
            provider._scene_checks.clear()
            provider._scene_transport_scopes.clear()
            predictor.receipts.clear()


def resource_sample():
    memory = {}
    for row in Path("/proc/meminfo").read_text().splitlines():
        fields = row.split()
        if fields[0] in ("MemAvailable:", "MemTotal:"):
            memory[fields[0].rstrip(":")] = int(fields[1]) * 1024
    return {"memory_bytes": memory, "load_average": list(os.getloadavg()),
            "host": os.uname().nodename, "cpu_count": os.cpu_count()}


def update_status(root, started, active, pending, errors, settings):
    completed = len(list((root / "outcomes").glob("*.json")))
    requests = []
    for path in (root / "solver_requests").glob("*/*.json"):
        row = json.loads(path.read_text())
        if row["status"] in ("started", "queued"):
            requests.append({"qa_id": path.parent.name, "index": path.stem,
                             "status": row["status"], "started_at": row.get("started_at")})
    pilot.write(root / "live_status.json", {
        "status": "draining_after_error" if errors else "running",
        "completed": completed, "total": settings["selected_questions"],
        "elapsed_s": time.time() - started,
        "active_tasks": list(active.values()), "pending_tasks": pending,
        "active_qwen_requests": requests, "settings": settings, **resource_sample(),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--published-inputs", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=40)
    parser.add_argument("--preparation-request-cap", type=int, default=3884)
    parser.add_argument("--prepare-workers", type=int, default=4)
    parser.add_argument("--solve-workers", type=int, default=16)
    parser.add_argument("--cpu-limit", type=int, default=8)
    parser.add_argument("--api-limit", type=int, default=32)
    parser.add_argument("--wait-for-pid-file", type=Path)
    args = parser.parse_args()
    if sys.platform != "linux" or min(args.prepare_workers, args.solve_workers,
                                      args.cpu_limit, args.api_limit) < 1:
        raise ValueError("parallel pilot requires Linux and positive worker limits")
    if args.wait_for_pid_file:
        expected = json.loads(args.wait_for_pid_file.read_text())["process_identity"]
        proc = Path("/proc") / str(expected["pid"])
        pause_file = args.run_dir / "concurrency_cutover_pause.json"
        print(json.dumps({"status": "waiting_for_clean_episode_boundary",
                          "process_identity": expected}), flush=True)
        while proc.exists() and proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
            actual_ticks = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
            if actual_ticks != expected["start_ticks"]:
                raise RuntimeError("previous pilot process identity changed")
            time.sleep(0.25)
        if (not pause_file.exists() or
                json.loads(pause_file.read_text())["process_identity"] != expected):
            raise RuntimeError("previous pilot did not stop at the approved clean boundary")
    root, sampling, cfg, provider, overview, predictor = pilot.initialize(args)
    budget = root / "request_budget.json"
    if not budget.exists():
        pilot.write(budget, {"max_requests": args.preparation_request_cap,
                            "used": 0, "requests": []})
    elif json.loads(budget.read_text())["max_requests"] != args.preparation_request_cap:
        raise ValueError("frozen preparation request cap changed")
    os.environ[ENV] = str(budget)
    questions = {str(row["qa_id"]): row for row in sampling["questions"]}
    scenes = OrderedDict()
    for qa_id in questions:
        scenes.setdefault(provider.entries[qa_id]["scene_id"], []).append(qa_id)
    if threading.active_count() != 1:
        raise RuntimeError("fork preflight requires a single parent thread")
    settings = {
        "schema": "pilot-execution-scheduling/1", "prepare_workers": args.prepare_workers,
        "solve_workers": args.solve_workers, "cpu_limit": args.cpu_limit,
        "api_limit": args.api_limit, "binary_review_concurrency": 1,
        "execution_isolation": "one episode per process",
        "solver_timeout_s": cfg.online_timeout_s, "automatic_retry": False,
        "selected_questions": sampling["n_questions"],
        "preparation_request_cap": args.preparation_request_cap,
        "reference_answer_visible": False, "scientific_configuration_changed": False,
    }
    if cfg.online_timeout_s != 900 or not cfg.online_enable_thinking:
        raise ValueError("parallel scheduling changed the approved solver configuration")
    pilot.write(root / "parallel_execution.json", {**settings, "at": time.time()})
    context = mp.get_context("fork")
    prep_tasks, solve_tasks, events = (context.Queue() for _ in range(3))
    stop = context.Event()
    cpu_slot = context.BoundedSemaphore(args.cpu_limit)
    api_slot = context.BoundedSemaphore(args.api_limit)
    review_lock = context.Lock()
    processes = []
    for kind, count, tasks in (("prepare", args.prepare_workers, prep_tasks),
                               ("solve", args.solve_workers, solve_tasks)):
        for index in range(count):
            process = context.Process(
                name=f"pilot-{kind}-{index}", target=worker,
                args=(kind, tasks, events, stop, cpu_slot, api_slot, review_lock,
                      root, cfg, provider, overview, predictor, questions))
            process.start()
            processes.append(process)
    # No queue put occurs before all children have forked.
    pending = {"prepare": 0, "solve": 0}
    admitted = set()

    def enqueue_solve(ids):
        for qa_id in ids:
            if qa_id not in admitted and not (root / "outcomes" / f"{qa_id}.json").exists():
                admitted.add(qa_id)
                solve_tasks.put(qa_id)
                pending["solve"] += 1

    for scene_id, ids in sorted(scenes.items()):
        if (root / "scene_audits" / f"{scene_id}.json").exists():
            for qa_id in ids:
                if (root / "outcomes" / f"{qa_id}.json").exists():
                    pilot.export_question(root, qa_id)
            enqueue_solve(ids)
        elif any(not (root / "outcomes" / f"{qa}.json").exists() for qa in ids):
            prep_tasks.put((scene_id, ids))
            pending["prepare"] += 1
    active, errors = {}, []
    started, last_update = time.time(), 0.
    journal = root / "parallel_events.jsonl"
    with journal.open("a", buffering=1) as log:
        while any(pending.values()):
            try:
                event = events.get(timeout=2)
            except queue.Empty:
                event = None
            if event:
                log.write(json.dumps({**event, "observed_at": time.time()}) + "\n")
                key = (event["kind"], event["key"])
                if event["status"] == "started":
                    active[key] = event
                else:
                    active.pop(key, None)
                    pending[event["kind"]] -= 1
                    if event["status"] == "failed":
                        errors.append(event)
                    elif event["status"] == "finished" and event["kind"] == "prepare":
                        if not stop.is_set():
                            enqueue_solve(event["qa_ids"])
                print(json.dumps(event), flush=True)
            exited = [p for p in processes if p.exitcode is not None]
            if exited:
                stop.set()
                pilot.write(root / "parallel_failure.json", {
                    "error_type": "WorkerExited",
                    "workers": [{"name": p.name, "exitcode": p.exitcode} for p in exited],
                    "active_tasks": list(active.values()), "automatic_retry": False,
                })
                # Do not terminate active workers or restart an interrupted episode.
                for tasks, count in ((prep_tasks, args.prepare_workers),
                                     (solve_tasks, args.solve_workers)):
                    for _ in range(count):
                        tasks.put(None)
                for process in processes:
                    process.join()
                raise RuntimeError("owned parallel worker exited; retained without retry")
            if time.time() - last_update >= 10:
                update_status(root, started, active, pending, errors, settings)
                last_update = time.time()
    for tasks, count in ((prep_tasks, args.prepare_workers), (solve_tasks, args.solve_workers)):
        for _ in range(count):
            tasks.put(None)
    for process in processes:
        process.join()
    completed = len(list((root / "outcomes").glob("*.json")))
    pilot.write(root / "live_status.json", {
        "status": "complete" if completed == sampling["n_questions"] and not errors else "stopped",
        "completed": completed, "total": sampling["n_questions"],
        "elapsed_s": time.time() - started,
        "errors": errors, "settings": settings, **resource_sample(),
    })
    if errors or completed != sampling["n_questions"]:
        raise RuntimeError("parallel pilot incomplete; retained without automatic retry")


if __name__ == "__main__":
    main()
