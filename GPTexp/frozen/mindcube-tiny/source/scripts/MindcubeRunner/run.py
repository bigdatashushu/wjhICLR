"""Default zero-request preflight; execution needs an explicit --execute."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import subprocess
import sys

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    __package__ = "MindcubeRunner"

from .budget import audit
from .common import Config, digest, read, sha, source_identity, validate_preflight, write
from .data import PRODUCTION_PINS, load, pair_identity, select
from .protocol import messages
from .score import summarize
from .worker import JOB_SCHEMA
from .traces import POLICY, mark_orphans, storage_check

FIELDS = {"cache_root", "output", "selection", "arms", "runner", "input_pins",
          "gateway", "auth_module_root", "budget_base", "max_requests", "concurrency",
          "budget_authorization_sha256"}


def preflight(settings):
    if set(settings) - FIELDS:
        raise ValueError("unknown_run_settings")
    if "gateway" in settings:
        from .transport import validate_gateway
        validate_gateway(settings["gateway"])
    config = Config(**settings.get("runner", {}))
    arms = settings.get("arms", ["skill"])
    if (not isinstance(arms, list) or not arms or len(set(arms)) != len(arms)
            or set(arms) - {"skill", "direct"}):
        raise ValueError("invalid_arms")
    concurrency = settings.get("concurrency", 1)
    if type(concurrency) is not int or concurrency not in (1, 2, 4, 8, 16):
        raise ValueError("unsupported_bounded_concurrency")
    if concurrency > 2:
        if (sha(Path(settings["budget_base"]) / "authorization.json")
                != settings.get("budget_authorization_sha256")):
            raise ValueError("campaign_authorization_required")
    items = select(settings["cache_root"], settings.get("selection"),
                   pins=settings.get("input_pins", PRODUCTION_PINS))
    cap = len(items) * sum(config.request_limit(arm) for arm in arms)
    if (type(settings.get("max_requests", cap)) is not int
            or settings.get("max_requests", cap) != cap):
        raise ValueError("budget_must_cover_frozen_upper_bound")
    initial = {}
    for item in items:
        cache = load(item)
        initial[item["sample_id"]] = {
            arm: messages(item, cache, arm=arm, config=config)[1] for arm in arms}
    frozen = {"schema": "mindcube-tinybench-preflight/1", "settings": settings,
              "config": config.public(), "arms": arms, "items": items, "max_requests": cap,
              "concurrency": concurrency,
              "trace_policy": POLICY,
              "source": source_identity(), "initial_receipts": initial,
              "pair_identities": {item["sample_id"]: pair_identity(item) for item in items},
              "model_requests": 0, "gpu_requests": 0, "reference_answer_visible": False}
    frozen["run_identity"] = digest(frozen)
    output = Path(settings["output"]).resolve()
    # Never put mutable output under adapter source or an input cache.
    repo = Path(__file__).resolve().parents[2]
    cache_root = Path(settings["cache_root"]).resolve()
    if output == repo or repo in output.parents or output == cache_root or cache_root in output.parents:
        raise ValueError("output_must_be_outside_repo_and_input_cache")
    if not output.parent.is_dir():
        raise ValueError("output_parent_missing")
    if output.exists():
        previous = read(output / "preflight.json")
        if previous != frozen:
            raise ValueError("preflight_identity_changed")
    else:
        output.mkdir()
        write(output / "preflight.json", frozen, exclusive=True)
    return frozen, output


def dispatch(job):
    directory = Path(job["output_directory"])
    if Path(job["stop_file"]).exists():
        return None
    environment = {key: os.environ[key] for key in
                   ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "VIRTUAL_ENV")
                   if key in os.environ}
    environment.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": "",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_DATASETS_OFFLINE": "1"})
    config = Config(**job["config"])
    try:
        process = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).parent / "worker.py")],
            input=json.dumps(job), text=True, encoding="utf-8", env=environment,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=(config.timeout_s + config.cell_timeout_s + 15)
            * config.request_limit(job["arm"]) + 30)
        failed = process.returncode != 0
    except subprocess.TimeoutExpired:
        failed = True
        write(directory / "worker_failure.json",
              {"error_type": "TimeoutExpired", "automatic_retry": False}, exclusive=True)
    if (directory / "result.json").exists():
        result = read(directory / "result.json")
    else:
        failed = True
        result = {
            "schema": "mindcube-worker-failure/1", "sample_id": job["item"]["sample_id"],
            "arm": job["arm"], "run_identity": job["run_identity"],
            "pair_identity": pair_identity(job["item"]), "status": "worker_error",
            "answer": None, "request_count": len(list(directory.glob("*.intent.json"))),
            "reference_answer_visible": False, "real_labels_available": False,
            "automatic_retry": False,
        }
        result["receipt_sha256"] = digest(result)
        write(directory / "result.json", result, exclusive=True)
    if failed:
        write(job["stop_file"], {"reason": "worker_failure"})
        mark_orphans(directory, "worker_failure")
    return result


def bounded_pairs(items, pair_slots, execute_pair, stopped):
    """Start at most pair_slots pairs; drain admitted work after a stop."""
    results, iterator = [], iter(enumerate(items))
    with ThreadPoolExecutor(max_workers=pair_slots) as executor:
        pending = set()
        while True:
            while not stopped() and len(pending) < pair_slots:
                try:
                    index, item = next(iterator)
                except StopIteration:
                    break
                pending.add(executor.submit(execute_pair, index, item))
            if not pending:
                break
            complete, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in complete:
                results.extend(row for row in future.result() if row is not None)
    return results


def execute(frozen, output):
    validate_preflight(frozen)
    if frozen.get("trace_policy") != POLICY:
        raise ValueError("frozen_trace_policy_changed")
    storage_check(output)
    settings = frozen["settings"]
    if not {"gateway", "auth_module_root", "budget_base"} <= set(settings):
        raise ValueError("execution_transport_and_budget_required")
    from .transport import auth_loader, health
    # Both checks occur before creating any inference intent.
    accounting = audit(settings["budget_base"])
    if accounting["remaining"] < frozen["max_requests"]:
        raise ValueError("aggregate_budget_insufficient_for_selection")
    identity = health(settings["gateway"], auth_loader(settings["auth_module_root"]))
    write(output / "execution_admission.json",
          {"run_identity": frozen["run_identity"], "service": identity, "budget": accounting},
          exclusive=True)  # prevents automatic run/resume/retry
    episodes = output / "episodes"
    episodes.mkdir()
    budget_root, stop_file = output / "requests", output / "STOP.json"
    def execute_pair(index, item):
        if stop_file.exists():
            return []
        sample_root = episodes / f"{index:04d}"
        sample_root.mkdir()
        jobs = []
        for arm in frozen["arms"]:
            if stop_file.exists():
                break
            directory = sample_root / arm
            directory.mkdir()
            # Labels never enter this subprocess job or its environment.
            job = {
                "schema": JOB_SCHEMA, "item": item, "arm": arm,
                "output_directory": str(directory), "stop_file": str(stop_file),
                "config": frozen["config"], "source": frozen["source"],
                "run_identity": frozen["run_identity"],
                "trace_policy": frozen["trace_policy"],
                "initial_receipt": frozen["initial_receipts"][item["sample_id"]][arm],
                "campaign": "mindcube-tiny-" + frozen["run_identity"][:16],
                "budget_root": str(budget_root), "max_requests": frozen["max_requests"],
                "concurrency": frozen["concurrency"],
                **{key: settings[key] for key in ("gateway", "auth_module_root", "budget_base")},
            }
            jobs.append(job)
        pair_results = []
        if frozen["concurrency"] == 1:
            for job in jobs:
                if stop_file.exists():
                    break
                pair_results.append(dispatch(job))
        else:
            with ThreadPoolExecutor(max_workers=len(frozen["arms"])) as executor:
                futures = [executor.submit(dispatch, job) for job in jobs]
                pair_results.extend(future.result() for future in futures)
        return pair_results
    pair_slots = max(1, frozen["concurrency"] // len(frozen["arms"]))
    results = bounded_pairs(frozen["items"], pair_slots, execute_pair, stop_file.exists)
    summary = summarize(frozen["items"], frozen["arms"], results,
                        run_identity=frozen["run_identity"])
    write(output / "summary.json", summary, exclusive=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    frozen, output = preflight(read(args.config))
    if args.execute:
        value = execute(frozen, output)
    else:
        value = {"status": "preflight_passed", "questions": len(frozen["items"]),
                 "arms": frozen["arms"], "maximum_requests_if_executed": frozen["max_requests"],
                 "model_requests": 0, "gpu_requests": 0, "output": str(output)}
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
