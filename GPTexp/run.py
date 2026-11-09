"""Unified frozen preparation and persistent paired scheduling."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .adapters import COUNTS, ROOT
from .MMSIGPT.packaging import preflight_identity, source_files as mmsi_source_files
from .api_pool import ApiPool
from .traces import digest, file_digest, read, write

PAIRED_REQUESTS_PER_QUESTION = {
    "blink-multiview": 5, "cv3d": 7, "mmsi": 5, "mindcube-tiny": 7, "blink-sr": 4}


def environment():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def code_identity():
    files = list(ROOT.glob("*.py"))
    files.extend(mmsi_source_files(ROOT / "MMSIGPT"))
    vendor = ROOT / "vendor/xcodeapi"
    files.extend(vendor.glob("*.py"))
    if (vendor / "profiles.json").exists():
        files.append(vendor / "profiles.json")
    return {str(p.relative_to(ROOT)): file_digest(p) for p in sorted(files)}


def prepare(directory, config, *, datasets=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        raise ValueError("manifest_already_prepared")
    selected = list(datasets or config.get("datasets") or COUNTS)
    if not selected or len(set(selected)) != len(selected) or any(name not in COUNTS for name in selected):
        raise ValueError("invalid_dataset_selection")
    counts = {name: COUNTS[name] for name in selected}
    questions = sum(counts.values())
    arms = questions * 2
    max_requests = sum(counts[name] * PAIRED_REQUESTS_PER_QUESTION[name] for name in counts)
    if config["max_requests"] != max_requests:
        raise ValueError("selected_dataset_request_budget_changed")
    original_receipts = read(ROOT / "frozen/sources.json")
    source_receipts = {name: original_receipts[name] for name in selected}
    for dataset, receipt in source_receipts.items():
        for name, expected in receipt["files"].items():
            if file_digest(ROOT / "frozen" / dataset / name) != expected:
                raise ValueError("frozen_source_changed")
    def preflight_one(dataset):
        path = directory / "preflight" / (dataset + ".json")
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            subprocess.run(
                [sys.executable, "-B", "-m", "GPTexp.worker", "--dataset", dataset,
                 "--preflight", "--output", str(path)],
                cwd=ROOT.parent, env=environment(), check=True, timeout=864000)
        receipt = read(path)
        if (receipt["identity"] != digest(receipt["specs"])
                or receipt["adapter_files"] != preflight_identity(ROOT, dataset)):
            raise ValueError("preflight_receipt_or_adapter_changed")
        return dataset, receipt

    with ThreadPoolExecutor(max_workers=min(4, len(selected))) as executor:
        specs = dict(executor.map(preflight_one, selected))
    groups = defaultdict(list)
    for dataset, receipt in specs.items():
        for spec in receipt["specs"]:
            public = spec.get("problem", spec.get("item"))
            groups[(dataset, public["sample_id"])].append(spec)
    if counts != dict(Counter(dataset for dataset, _ in groups)):
        raise ValueError("question_count_changed")
    pairs = []
    for (dataset, sample_id), arm_specs in groups.items():
        if sorted(a["arm"] for a in arm_specs) != ["direct", "harness"]:
            raise ValueError("invalid_pair")
        public = arm_specs[0].get("problem", arm_specs[0].get("item"))
        pair = digest({"dataset": dataset, "sample_id": sample_id,
                       "input_sha256": public["input_sha256"], "source": source_receipts[dataset]["archive_sha256"]})
        pairs.append({"dataset": dataset, "sample_id": sample_id, "pair_identity": pair,
                      "arms": arm_specs})
    # Interleave datasets without changing each dataset's frozen question order.
    queues = {dataset: [pair for pair in pairs if pair["dataset"] == dataset] for dataset in counts}
    ordered = []
    for index in range(max(counts.values())):
        ordered.extend(queue[index] for queue in queues.values() if index < len(queue))
    manifest = {"schema": "gptexp-full-paired/1", "questions": questions, "arms": arms,
                "max_requests": max_requests, "counts": counts, "config": config, "pairs": ordered,
                "source_receipts": source_receipts, "adapter_source": code_identity(),
                "preflight": {name: row["identity"] for name, row in specs.items()},
                "reference_answer_visible": False, "automatic_retry": False, "skill_evolution": False,
                "allowed_changes": ["API protocol/client/model", "reasoning effort high",
                                    "transport and scheduler waits", "endpoint pooling",
                                    "passive complete execution capture", "path and output placement"],
                "prepared_at": time.time()}
    if config.get("service_failure_policy"):
        manifest["service_failure_policy"] = config["service_failure_policy"]
        manifest["allowed_changes"].append("user-requested fresh full pass with transient API cooldown")
    manifest["run_identity"] = digest(manifest)
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "preparation.json", {
        "status": "ready", "questions": questions, "arms": arms, "model_requests": 0,
        "inputs_ready": True,
        "api_launch_ready": any(api["enabled"] and api["validated"] for api in config["apis"]),
        "run_identity": manifest["run_identity"], "reference_answers_read": False}, exclusive=True)
    print(f"All {questions} pairs prepared; zero preparation model requests.", flush=True)


def verify_manifest(manifest):
    if digest({k: v for k, v in manifest.items() if k != "run_identity"}) != manifest["run_identity"]:
        raise ValueError("manifest_changed")
    if manifest["adapter_source"] != code_identity():
        raise ValueError("adapter_source_changed")
    for dataset, receipt in manifest["source_receipts"].items():
        for name, expected in receipt["files"].items():
            if file_digest(ROOT / "frozen" / dataset / name) != expected:
                raise ValueError("frozen_source_changed")


def execute(directory, *, score_results=True):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    verify_manifest(manifest)
    config = manifest["config"]
    worker_module = config.get("worker_module", "GPTexp.worker")
    if (worker_module not in {"GPTexp.worker", "GPTexp.terra_worker"}
            or worker_module == "GPTexp.terra_worker" and config["model"] != "gpt-5.6-terra"):
        raise ValueError("invalid_worker_model_binding")
    apis = [api for api in config["apis"] if api["enabled"] and api["validated"]]
    if not apis:
        raise ValueError("no_validated_api")
    with (directory / "operator.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pool = ApiPool(directory / "pool", config)
        write(directory / "startup.json", {"pid": os.getpid(), "at": time.time(),
              "run_identity": manifest["run_identity"], "automatic_retry": False,
              "service_failure_policy": config.get("service_failure_policy"),
              "http_timeouts": {api["id"]: api["timeout_seconds"] for api in apis}})
        assignments = read(directory / "assignments.json") if (directory / "assignments.json").exists() else {}
        pending_pairs = []
        unresolved = []
        for pair in manifest["pairs"]:
            pair_dir = directory / "episodes" / pair["dataset"] / pair["pair_identity"]
            completed = [(pair_dir / arm["arm"] / "episode.json").exists() for arm in pair["arms"]]
            if all(completed):
                for arm in pair["arms"]:
                    episode = read(pair_dir / arm["arm"] / "episode.json")
                    if (episode["run_identity"] != manifest["run_identity"]
                            or episode["pair_identity"] != pair["pair_identity"]):
                        raise ValueError("episode_identity_changed")
                continue
            if any(completed) or any(pair_dir.rglob("*.json")):
                unresolved.append(pair["pair_identity"])
                continue
            pending_pairs.append(pair)

        def arm_job(pair, spec, api):
            public = spec.get("problem", spec.get("item"))
            job = {"spec": spec, "sample_id": public["sample_id"], "api_id": api["id"],
                   "config": config, "run_identity": manifest["run_identity"],
                   "pair_identity": pair["pair_identity"], "pool_root": str(directory / "pool")}
            job["operation_id"] = digest({"pair": pair["pair_identity"], "arm": spec["arm"],
                                          "api": api["id"], "run": manifest["run_identity"]})
            output = directory / "episodes" / pair["dataset"] / pair["pair_identity"] / spec["arm"]
            output.mkdir(parents=True, exist_ok=True)
            write(output / "dispatch.json", {"job_sha256": digest(job), "api_id": api["id"],
                  "operation_id": job["operation_id"], "at": time.time()}, exclusive=True)
            with (output / "worker.log").open("xb") as log:
                process = subprocess.Popen(
                    [sys.executable, "-B", "-m", worker_module, "--dataset", pair["dataset"],
                     "--output", str(output)], cwd=ROOT.parent, env=environment(),
                    stdin=subprocess.PIPE, stdout=log, stderr=log)
                try:
                    process.communicate(json.dumps(job, ensure_ascii=False).encode(),
                                        timeout=config["worker_timeout_seconds"])
                except subprocess.TimeoutExpired:
                    process.terminate()
                    write(output / "process_timeout.json", {
                        "pid": process.pid, "timeout_seconds": config["worker_timeout_seconds"],
                        "request_outcome": "unknown", "automatic_retry": False}, exclusive=True)
                    pool.pause(api, "worker_timeout")
                    return False
            receipt = output / "episode.json"
            if not receipt.exists():
                pool.pause(api, "worker_failure")
                return False
            episode = read(receipt)
            if (episode["run_identity"] != manifest["run_identity"]
                    or episode["pair_identity"] != pair["pair_identity"]
                    or episode["job_sha256"] != digest(job)):
                raise ValueError("worker_result_identity_changed")
            result = episode["result"] or {}
            if process.returncode or episode["failure"] or result.get("status") in {
                    "service_error", "host_error", "input_error", "precompute_error",
                    "prepared_identity_mismatch", "prepared_input_mismatch",
                    "prepared_contract_error", "request_exhausted", "budget_exhausted"}:
                errors = [read(path) for path in sorted((output / "requests").glob("*.response.json"))
                          if read(path).get("status") == "error"]
                if not (errors and not pool.paused(api)
                        and all(pool.transient(reason_code=row.get("reason_code"),
                                               http_status=row.get("http_status")) for row in errors)):
                    pool.pause(api, "operational_failure")
                return False
            return True

        pending = {}
        done = 0
        finished_jobs = 0
        cursor = 0
        next_progress = 0
        with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
            while cursor < len(pending_pairs) or pending:
                active = Counter(api_id for _, api_id in pending.values())
                while cursor < len(pending_pairs) and len(pending) + 2 <= config["max_workers"]:
                    available = [api for api in apis if not pool.paused(api) and not pool.cooling(api)]
                    if not available:
                        break
                    pair = pending_pairs[cursor]
                    previous = assignments.get(pair["pair_identity"])
                    api = next((a for a in available if a["id"] == previous), None) if previous else min(
                        available, key=lambda a: active[a["id"]] / a["max_inflight"])
                    if api is None:
                        break
                    assignments[pair["pair_identity"]] = api["id"]
                    write(directory / "assignments.json", assignments)
                    cursor += 1
                    for spec in pair["arms"]:
                        pending[executor.submit(arm_job, pair, spec, api)] = (pair, api["id"])
                        active[api["id"]] += 1
                if not pending:
                    if (cursor < len(pending_pairs)
                            and any(not pool.paused(api) and pool.cooling(api) for api in apis)):
                        time.sleep(1)
                        continue
                    break
                finished, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in finished:
                    pair, api_id = pending.pop(future)
                    try:
                        good = future.result()
                    except BaseException:
                        pool.pause(next(a for a in apis if a["id"] == api_id), "scheduler_worker_failure")
                        good = False
                    done += bool(good)
                    finished_jobs += 1
                if time.monotonic() >= next_progress:
                    write(directory / "progress.json", {
                        "at": time.time(), "new_completed_arms": done, "new_finished_jobs": finished_jobs,
                        "pending_arms": len(pending),
                        "dispatched_pairs": cursor, "unresolved_pairs": unresolved,
                        "remaining_undispatched_pairs": len(pending_pairs) - cursor,
                        "accounting": pool.summary()})
                    next_progress = time.monotonic() + 60
        episode_count = sum(1 for _ in (directory / "episodes").rglob("episode.json"))
        complete = episode_count == manifest["arms"] and not any(pool.paused(a) for a in apis)
        write(directory / "execution.json", {
            "status": "complete" if complete else "paused_with_retained_evidence",
            "episode_count": episode_count, "expected_arms": manifest["arms"], "accounting": pool.summary(),
            "at": time.time(), "reference_answers_read": False, "automatic_retry": False})
        print("Execution", "complete" if complete else "paused", episode_count, "arms", flush=True)
    env = environment()
    for name in tuple(env):
        if any(term in name.upper() for term in ("API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "MODELS_KEY")):
            env.pop(name)
    action = "GPTexp.score" if complete and score_results else "GPTexp.export"
    subprocess.run([sys.executable, "-B", "-m", action, "--run", str(directory)],
                   cwd=ROOT.parent, env=env, check=True, timeout=864000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "execute"])
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--datasets", nargs="+", choices=COUNTS)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.run, read(args.config), datasets=args.datasets)
    else:
        execute(args.run)


if __name__ == "__main__":
    main()
