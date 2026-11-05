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

from .budget import UNLIMITED_POLICY, admission, audit
from .common import (Config, IDS, VERSION, concurrency, digest, encoded, read, sha,
                     selected_ids, validate_source, write)
from .data import load, pair_identity, select
from .protocol import messages
from .reuse import inspect, materialize, request_cap, validate
from .transport import AUTH_SHA, health
from .worker import SCHEMA
from .selection import validate as validate_selection
from . import replay


def reviewer_policy(args):
    if not getattr(args, "intern_review", False):
        return None
    from .jev import Policy
    return Policy(provider="intern", endpoint=args.review_endpoint,
                  service_identity_sha256=args.review_identity,
                  yes_threshold=.5, max_requests=args.max_review_requests).public()


def public_contract(args):
    selection = read(args.selection) if getattr(args, "selection", None) else None
    replay_root = getattr(args, "replay_root", None)
    if replay_root:
        if selection:
            raise ValueError("replay_cannot_select_new_questions")
        inputs, provenance = replay.inspect(replay_root, args.cache_root)
        return inputs, selected_ids(provenance), None, provenance
    ids = validate_selection(args.cache_root, selection) if selection else IDS
    if selection and (args.reuse_root or args.expected_reused_direct):
        raise ValueError("fresh_selection_requires_new_direct")
    return select(args.cache_root, ids), ids, selection, None


def prepare_pairs(args, inputs, reviewer):
    pairs, reused = [], {}
    for item in inputs:
        rgb, frame, cache = load(item)
        receipts = {}
        for arm in ("direct", "skill"):
            context, receipt = messages(
                item, rgb, frame, cache if arm == "skill" else None,
                arm=arm, config=Config(),
                measurement_reviewer=reviewer if arm == "skill" else None)
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
    return pairs, reused


def plan(args):
    """Concrete zero-inference contract, including a blocked admission."""
    validate_source(read(args.root / "source_manifest.json"))
    inputs, ids, _, provenance = public_contract(args)
    reviewer = reviewer_policy(args)
    pairs, reused = prepare_pairs(args, inputs, reviewer)
    cap = request_cap(reused, ids, reviewer)
    accounting = audit(args.shared_budget_root.parents[1])
    upper = (accounting["used"] + accounting["mmsi_remaining_cap"]
             + accounting["mindcube_remaining_cap"] + accounting["cv3d_remaining_policy_cap"] + cap)
    unlimited = getattr(args, "no_global_budget_limit", False)
    write(args.root / "plan.json", {
        "schema": "omni-host-replay-plan/1", "sample_ids": list(ids),
        "question_replay": provenance, "pairs": pairs, "reused_direct": reused,
        "config": Config().public(), "measurement_reviewer": reviewer,
        "solver_cap": len(ids) * Config().request_limit("skill") + len(ids) - len(reused),
        "reviewer_cap": cap - request_cap(reused, ids), "total_cap": cap,
        "budget": accounting, "admission_upper_bound": upper,
        "aggregate_budget_policy": UNLIMITED_POLICY if unlimited else None,
        "status": "budget_blocked" if not unlimited and upper > accounting["global_cap"]
                  else "ready_for_health_preflight",
        "model_requests": 0, "gpu_requests": 0, "reference_answer_visible": False,
        "automatic_retry": False}, exclusive=True)
    print(f"plan: {len(ids)} questions; reused direct={len(reused)}; "
          f"cap={cap}; aggregate upper={upper}", flush=True)


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
    inputs, ids, selection, provenance = public_contract(args)
    if workers > 2 * len(ids):
        raise ValueError("concurrency_exceeds_episodes")
    reviewer = reviewer_policy(args)
    if reviewer:
        import urllib.request
        from .intern_review import native_client
        from .jev import NoRedirect, Policy
        native_client(Policy.parse(reviewer), urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect())).identity()
    pairs, reused = prepare_pairs(args, inputs, reviewer)
    cap = request_cap(reused, ids, reviewer)
    budget_policy = UNLIMITED_POLICY if getattr(args, "no_global_budget_limit", False) else None
    accounting = admission(audit(args.shared_budget_root.parents[1]), cap, budget_policy)
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
    if selection:
        frozen.update(sample_ids=list(ids), selection=selection)
    if provenance:
        frozen.update(sample_ids=list(ids), question_replay=provenance)
    if reviewer:
        frozen["measurement_reviewer"] = reviewer
    if budget_policy:
        frozen["aggregate_budget_policy"] = budget_policy
    write(root / "frozen.json", frozen, exclusive=True)
    expected = 2 * len(inputs)
    write(root / "preflight.json", {"status": "passed", "pairs": len(inputs), "episodes": expected,
          "new_episodes": expected-len(reused), "reused_direct": len(reused),
          "model_requests": 0, "gpu_requests": 0, "max_requests": cap,
          "reference_answer_visible": False, "real_labels_available": False,
          "run_identity": digest(frozen), "at": time.time()}, exclusive=True)
    print(f"preflight passed: {len(inputs)} pairs; {len(reused)} reused; "
          f"{expected-len(reused)} new episodes; cap{cap}", flush=True)


def run(root):
    frozen, pre = read(root / "frozen.json"), read(root / "preflight.json")
    workers = concurrency(frozen["max_episode_workers"])
    reused = frozen["reused_direct"]
    ids = selected_ids(frozen)
    reviewer = frozen.get("measurement_reviewer")
    expected = 2 * len(ids)
    if "selection" in frozen:
        if validate_selection(frozen["cache_root"], frozen["selection"]) != ids or reused:
            raise ValueError("fresh_selection_changed")
    if (pre["run_identity"] != digest(frozen) or frozen["config"] != Config().public()
            or frozen["max_requests"] != request_cap(reused, ids, reviewer) or len(reused) != pre["reused_direct"]
            or pre["episodes"] != expected or pre["pairs"] != len(ids) or workers > expected
            or frozen["local_budget_policy"]["max_inflight"] != workers
            or frozen["shared_budget_policy"]["max_inflight"] != workers
            or frozen["service_identity"]["max_inflight"] != workers
            or sha(root / "inputs.json") != frozen["inputs_sha256"]):
        raise ValueError("rerun_contract_changed")
    validate_source(frozen["source"])
    if frozen.get("question_replay"):
        if replay.validate(frozen["question_replay"], frozen["cache_root"]) != read(root / "inputs.json"):
            raise ValueError("replayed_public_inputs_changed")
    accounting = admission(audit(Path(frozen["shared_budget_root"]).parents[1]),
                          frozen["max_requests"], frozen.get("aggregate_budget_policy"))
    guard = (root / "guard.lock").open("a+b")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "STARTED.json").exists() or (root / "STOP.json").exists():
        raise ValueError("no_automatic_resume")
    sys.path.insert(0, frozen["auth_module_root"])
    from transport_auth import load_transport_token
    if health(frozen["gateway"], load_transport_token) != frozen["service_identity"]:
        raise ValueError("service_identity_changed")
    inputs = read(root / "inputs.json")
    if tuple(item["sample_id"] for item in inputs) != ids:
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
        if reviewer and arm == "skill":
            job.update(measurement_reviewer=reviewer, batch_frozen=frozen)
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
            try:
                result = future.result()
            except BaseException as exc:
                write(root / "STOP.json", {"reason": "episode_host_failure",
                                          "error_type": type(exc).__name__})
                result = None
            if result:
                results.append(result)
                print(f'{result["sample_id"]} {result["arm"]}: {result["status"]}; '
                      f'requests={result["request_count"]}; completed={len(results)}/{expected}', flush=True)
            write(root / "progress.json", {"episodes": len(results), "expected": expected,
                  "reused_direct": len(reused), "at": time.time(), "status": "running"})
    write(root / "completed.json", {"status": "complete" if len(results) == expected and
          not (root / "STOP.json").exists() else "stopped", "episodes": len(results),
          "expected": expected, "reused_direct": len(reused), "run_identity": digest(frozen),
          "at": time.time(), "episode_receipt_sha256": [r["receipt_sha256"] for r in results]},
          exclusive=True)
    print(read(root / "completed.json")["status"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "preflight", "run", "transition"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--shared-budget-root", type=Path)
    parser.add_argument("--auth-module-root", type=Path)
    parser.add_argument("--gateway", default="http://10.10.0.198:27443")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--reuse-root", type=Path, action="append", default=[])
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--intern-review", action="store_true")
    parser.add_argument("--review-endpoint", default="http://127.0.0.1:27248")
    parser.add_argument("--review-identity", default="")
    parser.add_argument("--max-review-requests", type=int, default=8)
    parser.add_argument("--no-global-budget-limit", action="store_true")
    parser.add_argument("--expected-reused-direct", type=int, choices=range(len(IDS)+1), default=20)
    args = parser.parse_args()
    if args.action == "plan":
        plan(args)
    elif args.action == "preflight":
        preflight(args)
    elif args.action == "run":
        run(args.root)
    else:
        transition(args.shared_budget_root, args.concurrency,
                   args.root / f"shared_policy_to_{args.concurrency}.json")


if __name__ == "__main__":
    main()
