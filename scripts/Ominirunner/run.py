"""Frozen zero-request preflight and fixed paired admission; never auto-resume."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    __package__ = "Ominirunner"

from .common import Config, IDS, VERSION, digest, encoded, read, sha, validate_source, write
from .budget import audit
from .data import load, pair_identity, select
from .protocol import messages
from .transport import AUTH_SHA, health
from .worker import SCHEMA


def preflight(root, cache_root, shared_root, gateway, auth_root):
    validate_source(read(root / "source_manifest.json"))
    if sha(Path(auth_root) / "transport_auth.py") != AUTH_SHA:
        raise ValueError("transport_auth_changed")
    sys.path.insert(0, str(auth_root))
    from transport_auth import load_transport_token
    service = health(gateway, load_transport_token)
    accounting = audit(Path(shared_root).parents[1])
    if accounting["upper_bound_with_mmsi_mindcube_and_omni_caps"] > 20000:
        raise ValueError("aggregate_reserved_budget_insufficient")
    write(root / "budget_before.json", accounting, exclusive=True)
    inputs = select(cache_root)
    pairs = []
    for item in inputs:
        rgb, frame, prepared = load(item)
        receipts = {}
        for arm in ("direct", "skill"):
            _, receipt = messages(item, rgb, frame, prepared if arm == "skill" else None,
                                  arm=arm, config=Config())
            receipts[arm] = receipt
        pairs.append({"sample_id": item["sample_id"], "pair_identity": pair_identity(item),
                      "initial_receipts": receipts})
    write(root / "inputs.json", inputs, exclusive=True)
    frozen = {
        "schema": VERSION, "source": read(root / "source_manifest.json"),
        "config": Config().public(), "campaign": root.name, "pairs": pairs,
        "cache_root": str(cache_root), "gateway": gateway, "auth_module_root": str(auth_root),
        "shared_budget_root": str(shared_root), "max_requests": Config().trial_request_cap,
        "max_episode_workers": 2, "automatic_retry": False, "service_identity": service,
        "inputs_sha256": sha(root / "inputs.json")}
    write(root / "frozen.json", frozen, exclusive=True)
    write(root / "preflight.json", {
        "status": "passed", "pairs": len(inputs), "episodes": 2*len(inputs), "model_requests": 0,
        "gpu_requests": 0, "reference_answer_visible": False, "real_labels_available": False,
        "run_identity": digest(frozen), "at": time.time(), "initial_pairs": pairs}, exclusive=True)
    print(f"preflight passed: {len(inputs)} pairs / {2*len(inputs)} episodes / 0 model requests",
          flush=True)


def run(root):
    frozen, pre = read(root / "frozen.json"), read(root / "preflight.json")
    if (frozen["config"] != Config().public()
            or frozen["schema"] != VERSION
            or frozen["max_requests"] != Config().trial_request_cap):
        raise ValueError("run_contract_changed")
    if pre["run_identity"] != digest(frozen) or sha(root / "inputs.json") != frozen["inputs_sha256"]:
        raise ValueError("preflight_changed")
    validate_source(frozen["source"])
    accounting = audit(Path(frozen["shared_budget_root"]).parents[1])
    if accounting["upper_bound_with_mmsi_mindcube_and_omni_caps"] > 20000:
        raise ValueError("aggregate_reserved_budget_insufficient")
    write(root / "budget_at_launch.json", accounting, exclusive=True)
    guard = (root / "guard.lock").open("a+b")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "STARTED.json").exists() or (root / "STOP.json").exists():
        raise ValueError("no_automatic_resume")
    # Authenticated health uses no inference and occurs just before admission.
    sys.path.insert(0, frozen["auth_module_root"])
    from transport_auth import load_transport_token
    if health(frozen["gateway"], load_transport_token) != frozen["service_identity"]:
        raise ValueError("service_identity_changed")
    write(root / "STARTED.json", {
        "pid": os.getpid(), "at": time.time(), "run_identity": digest(frozen),
        "max_requests": frozen["max_requests"], "thinking_both_arms": True}, exclusive=True)
    (root / "episodes").mkdir()
    budget = root / "requests"
    budget.mkdir()
    inputs = read(root / "inputs.json")
    if tuple(item["sample_id"] for item in inputs) != IDS:
        raise ValueError("selected_questions_changed")
    expected = 2*len(inputs)
    results = []

    def episode(item, arm):
        if (root / "STOP.json").exists():
            return None
        directory = root / "episodes" / (item["sample_id"] + "-" + arm)
        directory.mkdir()
        pair = next(p for p in frozen["pairs"] if p["sample_id"] == item["sample_id"])
        job = {
            "schema": SCHEMA, "item": item, "arm": arm, "config": frozen["config"],
            "source": frozen["source"], "initial_receipt": pair["initial_receipts"][arm],
            "run_identity": digest(frozen), "output_directory": str(directory),
            "budget_root": str(budget), "stop_file": str(root / "STOP.json"),
            **{k: frozen[k] for k in (
                "gateway", "auth_module_root", "shared_budget_root", "campaign")}}
        environment = dict(os.environ)
        for name in tuple(environment):
            if any(word in name.upper() for word in (
                    "API_KEY", "TOKEN", "ENDPOINT", "SECRET", "PASSWORD", "CREDENTIAL")):
                environment.pop(name)
        process = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).parent / "worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=environment)
        write(directory / "process.json", {
            "pid": process.pid, "job_sha256": digest(job), "reference_answer_visible": False,
            "automatic_retry": False}, exclusive=True)
        process.communicate(encoded(job))
        if process.returncode:
            write(root / "STOP.json", {"reason": "worker_exit", "returncode": process.returncode})
            return None
        row = read(directory / "result.json")
        if (row["receipt_sha256"] != digest({k: v for k, v in row.items() if k != "receipt_sha256"})
                or row["pair_identity"] != pair["pair_identity"]):
            raise ValueError("episode_receipt_changed")
        print(f'{item["sample_id"]} {arm}: {row["status"]}; requests={row["request_count"]}',
              flush=True)
        return row

    # Each official input's two arms start together. Stop admission on any service
    # deviation and drain already owned child processes without request retries.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for item in inputs:
            if (root / "STOP.json").exists():
                break
            futures = [pool.submit(episode, item, arm) for arm in ("direct", "skill")]
            for future in futures:
                row = future.result()
                if row:
                    results.append(row)
            write(root / "progress.json", {"episodes": len(results), "expected": expected,
                                          "at": time.time(), "status": "running"})
    write(root / "completed.json", {
        "status": "complete" if len(results) == expected and not (root / "STOP.json").exists()
        else "stopped", "episodes": len(results), "expected": expected,
        "run_identity": digest(frozen), "at": time.time(),
        "episode_receipt_sha256": [r["receipt_sha256"] for r in results]}, exclusive=True)
    print(read(root / "completed.json")["status"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--shared-budget-root", type=Path)
    parser.add_argument("--gateway", default="http://10.10.0.198:27441")
    parser.add_argument("--auth-module-root", type=Path)
    args = parser.parse_args()
    if args.action == "preflight":
        preflight(args.root, args.cache_root, args.shared_budget_root,
                  args.gateway, args.auth_module_root)
    else:
        run(args.root)


if __name__ == "__main__":
    main()
