"""Frozen concurrent rerun with verified historical direct-answer reuse."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
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

from .budget import audit
from .common import (Config, IDS, VERSION, concurrency, digest, encoded, read, sha,
                     validate_source, write)
from .data import load, pair_identity, select
from .protocol import messages
from .reuse import inspect, materialize, request_cap, validate
from .transport import AUTH_SHA, health
from .worker import SCHEMA


def transition(shared_root, target, receipt_path):
    """Change concurrency while holding state and every slot; preserve all rows."""
    from skill3d.synthesis.request_governor import RequestGovernorConfig
    concurrency(target)
    root = Path(shared_root)
    handles = []
    with (root / "state.lock").open("a+b") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        state = read(root / "state.json")
        old = state["policy"]
        concurrency(old["max_inflight"])
        if (old["max_requests"] != 20000 or old["requests_per_second"] != 0
                or old["max_inflight"] == target):
            raise ValueError("unexpected_shared_concurrency_policy")
        try:
            for index in range(max(target, old["max_inflight"])):
                handle = (root / f"slot-{index:03d}.lock").open("a+b")
                handles.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before_sha = digest(state["requests"])
            new = asdict(RequestGovernorConfig(20000, target))
            receipt = {"before_policy": old, "after_policy": new, "at": time.time(),
                       "request_count": len(state["requests"]),
                       "request_rows_sha256": before_sha, "active_slots": 0,
                       "historical_rows_preserved": True}
            state["policy"] = new
            state.setdefault("policy_transitions", []).append(receipt)
            write(root / "state.json", state)
            write(receipt_path, receipt, exclusive=True)
        finally:
            for handle in handles:
                handle.close()
    return receipt


def preflight(args):
    root = args.root
    workers = concurrency(args.concurrency)
    validate_source(read(root / "source_manifest.json"))
    if sha(args.auth_module_root / "transport_auth.py") != AUTH_SHA:
        raise ValueError("transport_auth_changed")
    sys.path.insert(0, str(args.auth_module_root))
    from transport_auth import load_transport_token
    service = health(args.gateway, load_transport_token)
    if service["max_inflight"] != workers:
        raise ValueError("gateway_concurrency_mismatch")
    inputs, pairs, reused = select(args.cache_root), [], {}
    for item in inputs:
        rgb, frame, cache = load(item)
        receipts = {}
        for arm in ("direct", "skill"):
            context, receipt = messages(item, rgb, frame, cache if arm == "skill" else None,
                                        arm=arm, config=Config())
            receipts[arm] = receipt
            if arm == "direct":
                for source in args.reuse_root:
                    entry = inspect(source, item, context)
                    if entry:
                        reused[item["sample_id"]] = entry
                        break
        pairs.append({"sample_id": item["sample_id"], "pair_identity": pair_identity(item),
                      "initial_receipts": receipts})
    if len(reused) != args.expected_reused_direct:
        raise ValueError("verified_direct_count_changed")
    cap = request_cap(reused)
    accounting = audit(args.shared_budget_root.parents[1])
    upper = (accounting["used"] + accounting["mmsi_remaining_cap"]
             + accounting["mindcube_remaining_cap"] + cap)
    if upper > accounting["global_cap"]:
        raise ValueError("aggregate_reserved_budget_insufficient")
    shared = read(args.shared_budget_root / "state.json")["policy"]
    if shared != {"max_requests": 20000, "max_inflight": workers, "requests_per_second": 0.0}:
        raise ValueError("shared_policy_concurrency_mismatch")
    write(root / "budget_before.json", accounting, exclusive=True)
    write(root / "inputs.json", inputs, exclusive=True)
    frozen = {"schema": VERSION, "source": read(root / "source_manifest.json"),
              "config": Config().public(), "campaign": root.name, "pairs": pairs,
              "cache_root": str(args.cache_root), "gateway": args.gateway,
              "auth_module_root": str(args.auth_module_root),
              "shared_budget_root": str(args.shared_budget_root), "max_requests": cap,
              "max_episode_workers": workers, "automatic_retry": False,
              "service_identity": service, "inputs_sha256": sha(root / "inputs.json"),
              "reused_direct": reused, "local_budget_policy": {
                  "max_requests": cap, "max_inflight": workers, "requests_per_second": 0.0},
              "shared_budget_policy": shared}
    write(root / "frozen.json", frozen, exclusive=True)
    write(root / "preflight.json", {"status": "passed", "pairs": len(inputs), "episodes": 40,
          "new_episodes": 40-len(reused), "reused_direct": len(reused),
          "model_requests": 0, "gpu_requests": 0, "max_requests": cap,
          "reference_answer_visible": False, "real_labels_available": False,
          "run_identity": digest(frozen), "at": time.time()}, exclusive=True)
    print(f"preflight passed: 20 pairs; {len(reused)} reused; "
          f"{40-len(reused)} new episodes; cap{cap}", flush=True)


def run(root):
    frozen, pre = read(root / "frozen.json"), read(root / "preflight.json")
    workers = concurrency(frozen["max_episode_workers"])
    reused = frozen["reused_direct"]
    if (pre["run_identity"] != digest(frozen) or frozen["config"] != Config().public()
            or frozen["max_requests"] != request_cap(reused) or len(reused) != pre["reused_direct"]
            or frozen["local_budget_policy"]["max_inflight"] != workers
            or frozen["shared_budget_policy"]["max_inflight"] != workers
            or frozen["service_identity"]["max_inflight"] != workers
            or sha(root / "inputs.json") != frozen["inputs_sha256"]):
        raise ValueError("rerun_contract_changed")
    validate_source(frozen["source"])
    accounting = audit(Path(frozen["shared_budget_root"]).parents[1])
    if (accounting["used"] + accounting["mmsi_remaining_cap"]
            + accounting["mindcube_remaining_cap"] + frozen["max_requests"] > 20000):
        raise ValueError("aggregate_reserved_budget_insufficient")
    guard = (root / "guard.lock").open("a+b")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "STARTED.json").exists() or (root / "STOP.json").exists():
        raise ValueError("no_automatic_resume")
    sys.path.insert(0, frozen["auth_module_root"])
    from transport_auth import load_transport_token
    if health(frozen["gateway"], load_transport_token) != frozen["service_identity"]:
        raise ValueError("service_identity_changed")
    inputs = read(root / "inputs.json")
    if tuple(item["sample_id"] for item in inputs) != IDS:
        raise ValueError("selected_questions_changed")
    for item in inputs:
        if item["sample_id"] in reused:
            rgb, frame, _ = load(item, prepared=False)
            context, _ = messages(item, rgb, frame, None, arm="direct", config=Config())
            validate(reused[item["sample_id"]], item, context)
    write(root / "budget_at_launch.json", accounting, exclusive=True)
    write(root / "STARTED.json", {"pid": os.getpid(), "at": time.time(),
          "run_identity": digest(frozen), "max_requests": frozen["max_requests"],
          "max_episode_workers": workers, "thinking_both_arms": True}, exclusive=True)
    (root / "episodes").mkdir()
    (root / "requests").mkdir()
    results = [materialize(entry, root) for entry in reused.values()]
    tasks = [(item, arm) for item in inputs for arm in ("skill", "direct")
             if not (arm == "direct" and item["sample_id"] in reused)]

    def episode(item, arm):
        if (root / "STOP.json").exists():
            return None
        directory = root / "episodes" / (item["sample_id"] + "-" + arm)
        directory.mkdir()
        pair = next(p for p in frozen["pairs"] if p["sample_id"] == item["sample_id"])
        job = {"schema": SCHEMA, "item": item, "arm": arm, "config": frozen["config"],
               "source": frozen["source"], "initial_receipt": pair["initial_receipts"][arm],
               "run_identity": digest(frozen), "output_directory": str(directory),
               "budget_root": str(root / "requests"), "stop_file": str(root / "STOP.json"),
               **{k: frozen[k] for k in ("gateway", "auth_module_root", "shared_budget_root",
                    "campaign", "local_budget_policy", "shared_budget_policy")}}
        environment = {k: v for k, v in os.environ.items() if not any(
            w in k.upper() for w in ("API_KEY", "TOKEN", "ENDPOINT", "SECRET", "PASSWORD", "CREDENTIAL"))}
        process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).parent / "worker.py")],
                                  stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, env=environment)
        write(directory / "process.json", {"pid": process.pid, "job_sha256": digest(job),
              "reference_answer_visible": False, "automatic_retry": False}, exclusive=True)
        process.communicate(encoded(job))
        if process.returncode:
            write(root / "STOP.json", {"reason": "worker_exit", "returncode": process.returncode})
            return None
        result = read(directory / "result.json")
        if (result["receipt_sha256"] != digest({
                k: v for k, v in result.items() if k != "receipt_sha256"})
                or result["pair_identity"] != pair["pair_identity"]
                or result["run_identity"] != digest(frozen)):
            raise ValueError("episode_receipt_changed")
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(episode, item, arm) for item, arm in tasks]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
                print(f'{result["sample_id"]} {result["arm"]}: {result["status"]}; '
                      f'requests={result["request_count"]}; completed={len(results)}/40', flush=True)
            write(root / "progress.json", {"episodes": len(results), "expected": 40,
                  "reused_direct": len(reused), "at": time.time(), "status": "running"})
    write(root / "completed.json", {"status": "complete" if len(results) == 40 and
          not (root / "STOP.json").exists() else "stopped", "episodes": len(results),
          "expected": 40, "reused_direct": len(reused), "run_identity": digest(frozen),
          "at": time.time(), "episode_receipt_sha256": [r["receipt_sha256"] for r in results]},
          exclusive=True)
    print(read(root / "completed.json")["status"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run", "transition"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--shared-budget-root", type=Path)
    parser.add_argument("--auth-module-root", type=Path)
    parser.add_argument("--gateway", default="http://10.10.0.198:27443")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--reuse-root", type=Path, action="append", default=[])
    parser.add_argument("--expected-reused-direct", type=int, choices=range(len(IDS)+1), default=20)
    args = parser.parse_args()
    if args.action == "preflight":
        preflight(args)
    elif args.action == "run":
        run(args.root)
    else:
        transition(args.shared_budget_root, args.concurrency,
                   args.root / f"shared_policy_to_{args.concurrency}.json")


if __name__ == "__main__":
    main()
