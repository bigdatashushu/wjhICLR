"""Isolated one-question Jev trial; existing campaigns remain frozen."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    __package__ = "Ominirunner"

from .audit import replay_contexts
from .budget import audit as budget_audit
from .common import Config, digest, encoded, read, sha, validate_source, write
from .data import load, select
from .jev import NoRedirect, Policy, credentials
from .protocol import messages
from .transport import AUTH_SHA, health
from .worker import SCHEMA

VERSION = "omni-isolated-jev-trial/1"


def validate_job(job):
    policy = Policy.parse(job["measurement_reviewer"])
    if (job["schema"] != SCHEMA or job["arm"] != "skill" or job["config"] != Config().public()
            or job["local_budget_policy"] != {
                "max_requests": Config().request_limit("skill") + policy.max_requests,
                "max_inflight": 1, "requests_per_second": 0.0}
            or job["shared_budget_policy"]["max_requests"] != 20000
            or job["shared_budget_policy"]["requests_per_second"] != 0.0
            or not {"scripts/Ominirunner/jev.py", "scripts/Ominirunner/jev_protocol.py",
                    "scripts/Ominirunner/jev_trial.py",
                    "scripts/Ominirunner/static_skill_host.txt"} <= set(job["source"])):
        raise ValueError("jev_trial_contract_changed")
    if policy.provider == "intern" and "scripts/Ominirunner/intern_review.py" not in job["source"]:
        raise ValueError("intern_source_pin_missing")
    validate_source(job["source"])
    rgb, frame, cache = load(job["item"])
    receipt = messages(job["item"], rgb, frame, cache, arm="skill", config=Config(),
                       measurement_reviewer=policy.public())[1]
    if receipt != job["initial_receipt"]:
        raise ValueError("jev_initial_input_changed")
    return rgb, frame, cache


def admission(job):
    accounting = budget_audit(Path(job["shared_budget_root"]).parents[1])
    if (accounting["all_active_policy_upper_bound"] + job["local_budget_policy"]["max_requests"]
            > accounting["global_cap"]):
        raise ValueError("jev_aggregate_reserved_budget_insufficient")
    return accounting


def preflight(args):
    directory = args.root.resolve()
    if not directory.parent.is_dir():
        raise ValueError("trial_parent_missing")
    directory.mkdir()
    if args.cache_root is None or args.auth_module_root is None or args.shared_budget_root is None:
        raise ValueError("jev_preflight_paths_required")
    if sha(args.auth_module_root / "transport_auth.py") != AUTH_SHA:
        raise ValueError("transport_auth_changed")
    sys.path.insert(0, str(args.auth_module_root))
    from transport_auth import load_transport_token
    service = health(args.gateway, load_transport_token)
    item = select(args.cache_root, (args.sample_id,))[0]
    rgb, frame, cache = load(item)
    policy = Policy(provider=args.provider, max_requests=args.max_jev_requests,
                    endpoint=args.review_endpoint or "",
                    service_identity_sha256=args.review_identity or "",
                    yes_threshold=args.yes_threshold if args.provider == "intern" else None)
    if policy.provider == "intern":
        import urllib.request
        from .intern_review import native_client
        native_client(policy, urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect())).identity()
    else:
        credentials(policy)  # Availability check only; no key enters the frozen job.
    repository = Path(__file__).resolve().parents[2]
    files = [*Path(__file__).parent.glob("*.py"), *Path(__file__).parent.glob("*.txt"),
             *(repository / "src").rglob("*.py")]
    source = {str(p.relative_to(repository)): sha(p) for p in sorted(files)}
    local = {"max_requests": Config().request_limit("skill") + policy.max_requests,
             "max_inflight": 1, "requests_per_second": 0.0}
    shared = read(args.shared_budget_root / "state.json")["policy"]
    job = {"schema": SCHEMA, "config": Config().public(), "source": source, "item": item,
           "arm": "skill", "campaign": directory.name, "measurement_reviewer": policy.public(),
           "gateway": args.gateway, "auth_module_root": str(args.auth_module_root.resolve()),
           "shared_budget_root": str(args.shared_budget_root.resolve()),
           "budget_root": str(directory / "requests"), "output_directory": str(directory / "episode"),
           "stop_file": str(directory / "STOP.json"), "local_budget_policy": local,
           "shared_budget_policy": shared,
           "initial_receipt": messages(item, rgb, frame, cache, arm="skill", config=Config(),
                                       measurement_reviewer=policy.public())[1]}
    job["run_identity"] = digest({"schema": VERSION, "job": job, "service": service})
    validate_job(job)
    accounting = admission(job)
    write(directory / "job.json", job, exclusive=True)
    write(directory / "preflight.json", {
        "schema": VERSION, "job_sha256": digest(job), "service_identity": service,
        "model_requests": 0, "gpu_requests": 0, "reference_answer_visible": False,
        "automatic_retry": False, "main_request_cap": Config().request_limit("skill"),
        "jev_request_cap": policy.max_requests, "total_request_cap": local["max_requests"],
        "budget_before": accounting}, exclusive=True)
    print("preflight passed: one question; zero inference requests")


def run(directory):
    job, pre = read(directory / "job.json"), read(directory / "preflight.json")
    if pre["schema"] != VERSION or pre["job_sha256"] != digest(job):
        raise ValueError("jev_preflight_changed")
    validate_job(job)
    admission(job)
    sys.path.insert(0, job["auth_module_root"])
    from transport_auth import load_transport_token
    if health(job["gateway"], load_transport_token) != pre["service_identity"]:
        raise ValueError("service_identity_changed")
    runtime_secret = credentials(Policy.parse(job["measurement_reviewer"]))
    # Exclusive creation makes duplicate starts impossible; no automatic resume.
    write(directory / "STARTED.json", {"run_identity": job["run_identity"],
                                      "automatic_retry": False}, exclusive=True)
    (directory / "episode").mkdir()
    (directory / "requests").mkdir()
    environment = {k: v for k, v in os.environ.items() if not any(
        word in k.upper() for word in ("API_KEY", "TOKEN", "ENDPOINT", "SECRET",
                                      "PASSWORD", "CREDENTIAL", "MAKERS_MODELS_KEY"))}
    process = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("worker.py"))],
                             input=encoded({**job, "_jev_secret": runtime_secret}), stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, env=environment, check=False)
    if process.returncode:
        write(directory / "STOP.json", {"reason": "worker_exit", "returncode": process.returncode})
        print("trial stopped; inspect sanitized host receipts")
    else:
        result = read(directory / "episode" / "result.json")
        print(f'trial {result["status"]}; solver={result["request_count"]}, '
              f'reviewer={result["jev_request_count"]}')


def audit(directory):
    job, pre = read(directory / "job.json"), read(directory / "preflight.json")
    if pre["job_sha256"] != digest(job):
        raise ValueError("jev_preflight_changed")
    rgb, frame, cache = validate_job(job)
    result = read(directory / "episode" / "result.json")
    if (result.get("receipt_sha256") != digest({k: v for k, v in result.items() if k != "receipt_sha256"})
            or result.get("measurement_reviewer") != job["measurement_reviewer"]
            or result.get("run_identity") != job["run_identity"]):
        raise ValueError("jev_episode_changed")
    for index, expected in replay_contexts(job["item"], result, rgb, frame, cache):
        if expected != read(directory / "episode" / f"{index:02d}.messages.json"):
            raise ValueError("jev_saved_context_changed")
        row = result["rounds"][index]
        execution = directory / "episode" / f"{index:02d}.execution.json"
        if (execution.exists() and read(execution) != row
                or row.get("cell") and not execution.is_file()):
            raise ValueError("jev_saved_execution_changed")
    local = read(directory / "requests" / "state.json")["requests"]
    shared = [r for r in read(Path(job["shared_budget_root"]) / "state.json")["requests"]
              if r.get("context", {}).get("campaign_id") == job["campaign"]]
    projection = lambda r: {k: r[k] for k in ("category", "context", "usage", "status")}
    if ([projection(r) for r in local] != [projection(r) for r in shared]
            or len(local) != result["total_request_count"]
            or sum(r["category"] == "gate" for r in local) != result["jev_request_count"]
            or len(local) > job["local_budget_policy"]["max_requests"]):
        raise ValueError("jev_request_accounting_changed")
    write(directory / "audit.json", {
        "schema": VERSION, "status": "passed", "episode_status": result["status"],
        "solver_requests": result["request_count"], "jev_requests": result["jev_request_count"],
        "context_replayed": True, "reference_answer_visible": False,
        "automatic_retry": False}, exclusive=True)
    print("audit passed; this is a trace audit, not a benchmark score")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--auth-module-root", type=Path)
    parser.add_argument("--shared-budget-root", type=Path)
    parser.add_argument("--sample-id", default="15")
    parser.add_argument("--gateway", default="http://10.10.0.198:27441")
    parser.add_argument("--provider", choices=("intern", "makers", "typesafe"), default="intern")
    parser.add_argument("--review-endpoint")
    parser.add_argument("--review-identity")
    parser.add_argument("--yes-threshold", type=float, default=.5,
                        help="Intern calibrated p(yes) must strictly exceed this value (default .5)")
    parser.add_argument("--max-jev-requests", type=int, default=8)
    args = parser.parse_args()
    if args.action == "preflight":
        preflight(args)
    else:
        (run if args.action == "run" else audit)(args.root.resolve())


if __name__ == "__main__":
    main()
