"""Frozen 93-question BLINK comparison using the validated 24-question solver.

Prepare a new run from an existing campaign, then execute and score it. Online
workers receive public problems only. No training, revision, or request replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
import datetime
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import sys
import time

SCHEMA = "harness3d-blink-paired93/1"
ARMS = ("fixed_skill", "direct")
SIZE = 93
CAP = SIZE * 5
SKILL_ARTIFACT_SHA = "6e48fbb3c3e987e4c2ea4772608367564eb1eaffbe2898f7ed9634ae28186774"


def now():
    return datetime.datetime.now().astimezone().isoformat()


def load_exp(source):
    sys.path.insert(0, str(Path(source) / "src"))
    from skill3d.evaluation import blink_experiment
    return blink_experiment


def progress(exp, root, value):
    temporary = root / f".progress-{os.getpid()}.tmp"
    temporary.write_text(json.dumps(exp.seal({"at": now(), **value}),
                                    sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, root / "progress.json")


def jobs_for(exp, root, profile, inspection):
    selected = [p for p in inspection["problems"] if p["source_split"] == "test"]
    if len(selected) != SIZE or len({p["sample_id"] for p in selected}) != SIZE:
        raise ValueError("paired_test_denominator_mismatch")
    skill = inspection["seed"]["text"]
    jobs = []
    rng = random.Random(137)
    for problem in selected:
        arms = list(ARMS)
        rng.shuffle(arms)
        cache = inspection["cache_identities"][problem["sample_id"]]
        for arm in arms:
            text = skill if arm == "fixed_skill" else ""
            identity = {
                "schema": SCHEMA, "arm": arm, "source_split": "test",
                "sample_id": problem["sample_id"], "input_sha256": problem["input_sha256"],
                "images": cache["images"], "artifact_sha256": cache["artifact_sha256"],
                "model": profile["model"], "solver": profile["solver"], "seed": 137,
                "skill_sha256": exp.sha_value(text),
                "allowed_tools": exp.TOOLS if arm == "fixed_skill" else [],
                "mode": "system" if arm == "fixed_skill" else "direct",
                "preflight_sha256": inspection["content_sha256"],
            }
            operation_id = exp.sha_value(identity)
            worker = {
                "problem": problem, "skill_text": text, "model": profile["model"],
                "solver": profile["solver"], "run_root": str(root),
                "operation_id": operation_id, "category": "solver",
                "mode": identity["mode"], "max_requests": CAP, "request_concurrency": 4,
            }
            # Direct solving never receives a geometry location or artifact.
            if arm == "fixed_skill":
                worker.update(prepared_root=profile["inputs"]["prepared_root"],
                              artifact_sha256=cache["artifact_sha256"])
            if any(k in worker for k in ("answer", "host_answers", "reference_answer")):
                raise ValueError("reference_in_worker")
            jobs.append({"identity": identity, "operation_id": operation_id, "worker": worker})
    return jobs


def prepare(root, template):
    if root.exists():
        raise ValueError("new_run_requires_new_directory")
    exp = load_exp(template / "code/source")
    old = exp.read(template / "manifest.json")
    if exp.source_files(template / "code/source") != old["source_files"]:
        raise ValueError("template_source_changed")
    profile = deepcopy(old["profile"])
    profile["runtime"].update(max_requests=CAP, request_concurrency=4)
    # Use the original profile validator without creating an API client request.
    temporary_profile = template / "code/source/configs/experiments/blink.json"
    exp.profile_at(temporary_profile)  # Validate the frozen protocol definition.
    inspection = exp.preflight(profile)
    prior = old["preflight"]
    for key in ("inputs", "problems", "cache_identities", "seed", "runtime_identity"):
        if inspection[key] != prior[key]:
            raise ValueError("template_preflight_identity_changed")
    if inspection["seed"]["content_sha256"] != SKILL_ARTIFACT_SHA:
        raise ValueError("fixed_trajectory1_skill_changed")
    if exp.file_sha(profile["inputs"]["host_answers"]) != old["reference_file_sha256"]:
        raise ValueError("host_reference_bytes_changed")
    for relative, digest in old["source_files"].items():
        destination = root / "code/source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template / "code/source" / relative, destination)
        if exp.file_sha(destination) != digest:
            raise ValueError("source_copy_changed")
    driver = root / "driver/paired.py"
    driver.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, driver)
    manifest = exp.write_once(root / "manifest.json", {
        "schema": SCHEMA, "created_at": now(), "profile": profile, "preflight": inspection,
        "source_files": old["source_files"], "driver_sha256": exp.file_sha(driver),
        "template_root": str(template), "template_sha256": old["content_sha256"],
        "reference_file_sha256": old["reference_file_sha256"],
        "size": SIZE, "arms": list(ARMS), "max_requests": CAP, "max_inflight": 4,
        "source_split": "test", "sample_role": "internal_test",
        "automatic_retry": False, "reference_answer_visible": False,
        "skill_evolution": False, "training_questions_used": False,
        "jobs": jobs_for(exp, root, profile, inspection),
    })
    governor = exp.RequestGovernor(root / "budget", exp.RequestGovernorConfig(CAP, 4))
    exp.write_once(root / "preparation.json", {
        "status": "ready", "at": now(), "manifest_sha256": manifest["content_sha256"],
        "model_requests": 0, "reference_answers_read": False,
        "reference_file_hashed_only": True, "budget": governor.summary(),
    })
    print(json.dumps({"status": "ready", "root": str(root), "questions_per_arm": SIZE,
                      "episodes": len(manifest["jobs"]), "max_requests": CAP}), flush=True)


def verify(exp, root, manifest, check_caches=False):
    if (manifest["schema"] != SCHEMA or manifest["size"] != SIZE
            or manifest["arms"] != list(ARMS) or manifest["max_requests"] != CAP):
        raise ValueError("run_policy_changed")
    if exp.file_sha(root / "driver/paired.py") != manifest["driver_sha256"]:
        raise ValueError("frozen_driver_changed")
    if exp.source_files(root / "code/source") != manifest["source_files"]:
        raise ValueError("frozen_source_changed")
    inputs = manifest["preflight"]
    profile = manifest["profile"]
    if exp.runtime_identity() != inputs["runtime_identity"]:
        raise ValueError("runtime_changed")
    for key, expected in inputs["inputs"].items():
        actual = (exp.file_identity(profile["inputs"][key]) if key == "host_answers"
                  else exp.file_sha(profile["inputs"][key]))
        if actual != expected:
            raise ValueError("input_file_changed")
    if manifest["jobs"] != jobs_for(exp, root, profile, inputs):
        raise ValueError("paired_jobs_changed")
    if check_caches:
        current = exp.preflight(profile)
        for key in ("inputs", "problems", "cache_identities", "seed", "runtime_identity"):
            if current[key] != inputs[key]:
                raise ValueError("live_preflight_changed")


def request_records(exp, root, operation_id):
    return {str(p.relative_to(root)): exp.file_sha(p)
            for p in sorted((root / "requests" / operation_id).glob("*.json"))}


def validate_receipt(exp, root, job, receipt):
    identity = job["identity"]
    if receipt["identity"] != identity or receipt["status"] != "complete":
        raise ValueError("invalid_completed_receipt")
    if receipt["request_records"] != request_records(exp, root, job["operation_id"]):
        raise ValueError("request_records_changed")
    row = receipt["result"]
    if any(row[k] != identity[k] for k in ("sample_id", "source_split", "input_sha256")):
        raise ValueError("result_public_identity_changed")
    records = receipt["request_records"]
    requests = sorted(p for p in records if p.endswith(".intent.json"))
    responses = sorted(p for p in records if p.endswith(".response.json"))
    limit = 1 if identity["arm"] == "direct" else 4
    if not 1 <= len(requests) <= limit or len(responses) != len(requests):
        raise ValueError("episode_request_count_mismatch")
    if row["request_count"] != len(requests):
        raise ValueError("reported_request_count_mismatch")
    if identity["arm"] == "direct" and (row["attempted_tool_calls"] or row["succeeded_tool_calls"]):
        raise ValueError("direct_used_tools")
    for relative in responses:
        response = exp.read(root / relative)
        if (response["status"] != "received"
                or response["response_identity"].get("model") != "qwen3.8-flash"
                or response["response_identity"].get("model_reported") is not True):
            raise ValueError("response_service_identity_mismatch")
    for relative in requests:
        intent = exp.read(root / relative)
        params = intent["parameters"]
        # Category extraction has its own frozen, small token limit.
        if params.get("seed") != 137 or params.get("temperature") != 0:
            raise ValueError("actual_request_seed_changed")
    for round_ in row["rounds"]:
        if round_.get("phase") in ("solve", "finalization"):
            if round_.get("delivered_skill_sha256") != identity["skill_sha256"]:
                raise ValueError("actual_delivered_skill_changed")
    return receipt


def score(exp, root, manifest, receipts, governor):
    verify(exp, root, manifest)
    if len(receipts) != SIZE * 2:
        raise ValueError("incomplete_paired_predictions")
    usage = governor.summary()
    if any(usage[k] for k in ("active_requests", "unknown_requests", "error_requests")):
        raise ValueError("requests_not_successfully_drained")
    if sum(r["result"]["request_count"] for r in receipts) != usage["reserved_requests"]:
        raise ValueError("budget_episode_count_mismatch")
    by_arm = {arm: [r for r in receipts if r["identity"]["arm"] == arm] for arm in ARMS}
    for arm, rows in by_arm.items():
        if len(rows) != SIZE or len({r["result"]["sample_id"] for r in rows}) != SIZE:
            raise ValueError("scoring_denominator_mismatch")
    # All workers have exited before this freeze or any host label lookup.
    frozen = exp.write_once(root / "prediction_freeze.json", {
        "at": now(), "manifest_sha256": manifest["content_sha256"],
        "receipts": {r["operation_id"]: r["content_sha256"] for r in receipts},
        "episodes": len(receipts), "reference_answers_read": False,
        "workers_exited": True, "budget": usage,
    })
    os.environ.pop(manifest["profile"]["model"]["api_key_env"], None)
    # Campaign.references only opens labels on the host, after prediction freeze.
    host = exp.Campaign(root, {
        k: v for k, v in exp.seal({
            "profile": manifest["profile"], "preflight": manifest["preflight"],
            "reference_file_sha256": manifest["reference_file_sha256"],
            "source_files": manifest["source_files"],
        }).items()
    })
    references = host.references("test")
    scores = {arm: exp.metrics(rows, references) for arm, rows in by_arm.items()}
    paired = Counter()
    comparison = []
    lookup = {arm: {r["result"]["sample_id"]: r for r in rows} for arm, rows in by_arm.items()}
    for problem in manifest["preflight"]["problems"]:
        if problem["source_split"] != "test":
            continue
        sample_id = problem["sample_id"]
        left, right = (lookup[arm][sample_id] for arm in ARMS)
        for key in ("input_sha256", "images", "model", "solver", "seed"):
            if left["identity"][key] != right["identity"][key]:
                raise ValueError("paired_input_identity_mismatch")
        correct = {arm: (lookup[arm][sample_id]["result"]["status"] == "answered"
                        and lookup[arm][sample_id]["result"]["answer"] == references[sample_id])
                   for arm in ARMS}
        outcome = ("both_correct" if all(correct.values()) else
                   "skill_only" if correct["fixed_skill"] else
                   "direct_only" if correct["direct"] else "both_wrong")
        paired[outcome] += 1
        comparison.append({
            "sample_id": sample_id, "reference_answer": references[sample_id],
            "input_sha256": problem["input_sha256"], "outcome": outcome,
            **{arm: {k: lookup[arm][sample_id]["result"][k] for k in
                     ("answer", "status", "request_count", "attempted_tool_calls",
                      "succeeded_tool_calls")} | {"correct": correct[arm]} for arm in ARMS},
        })
    costs = {}
    for arm, rows in by_arm.items():
        token_usage = Counter()
        durations = []
        for receipt in rows:
            for relative in receipt["request_records"]:
                if relative.endswith(".response.json"):
                    response = exp.read(root / relative)
                    token_usage.update({k: v for k, v in response.get("usage", {}).items()
                                        if k in ("prompt_tokens", "completion_tokens", "total_tokens")
                                        and isinstance(v, (int, float))})
                elif relative.endswith(".intent.json"):
                    intent = exp.read(root / relative)
                    response = exp.read(root / relative.replace(".intent.", ".response."))
                    durations.append(response["finished_at"] - intent["started_at"])
        costs[arm] = {
            "requests": sum(r["result"]["request_count"] for r in rows),
            "usage": dict(token_usage),
            "attempted_tool_calls": sum(r["result"]["attempted_tool_calls"] for r in rows),
            "succeeded_tool_calls": sum(r["result"]["succeeded_tool_calls"] for r in rows),
            "summed_request_seconds": round(sum(durations), 3),
            "statuses": dict(Counter(r["result"]["status"] for r in rows)),
        }
    table = exp.write_once(root / "comparison.json", {
        "manifest_sha256": manifest["content_sha256"],
        "prediction_freeze_sha256": frozen["content_sha256"], "rows": comparison,
        "label_access": "host_scoring_after_prediction_freeze",
    })
    completion = exp.write_once(root / "completion.json", {
        "status": "complete", "at": now(), "size": SIZE, "episodes": len(receipts),
        "manifest_sha256": manifest["content_sha256"],
        "prediction_freeze_sha256": frozen["content_sha256"],
        "comparison_sha256": table["content_sha256"], "scores": scores,
        "paired": dict(paired), "costs": costs, "request_usage": usage,
        "accuracy_difference_pp": (scores["fixed_skill"]["accuracy"] - scores["direct"]["accuracy"]) * 100,
        "source_split": "test", "sample_role": "internal_test", "skill_evolution": False,
        "reference_answer_visible_to_solver": False, "automatic_retry": False,
        "audit": {"paired_inputs": True, "original_image_identity": True,
                  "fixed_skill_delivery": True, "reported_model": "qwen3.8-flash",
                  "source_and_runtime_identity": True, "complete_denominators": True,
                  "workers_exited_before_label_access": True, "requests_drained": True},
    })
    progress(exp, root, {"phase": "complete", "completed": len(receipts), "scores": scores})
    print(json.dumps(completion, sort_keys=True), flush=True)


def run(root):
    exp = load_exp(root / "code/source")
    manifest = exp.read(root / "manifest.json")
    verify(exp, root, manifest, check_caches=True)
    governor = exp.RequestGovernor(root / "budget", exp.RequestGovernorConfig(CAP, 4))
    usage = governor.summary()
    if any(usage[k] for k in ("active_requests", "unknown_requests", "error_requests")):
        raise ValueError("unresolved_prior_requests_no_replay")
    receipts, pending_jobs = [], []
    for job in manifest["jobs"]:
        path = root / "results" / job["identity"]["arm"] / (job["operation_id"] + ".json")
        if path.exists():
            receipts.append(validate_receipt(exp, root, job, exp.read(path)))
        else:
            if (root / "intents" / (job["operation_id"] + ".json")).exists():
                raise ValueError("unfinished_episode_no_automatic_replay")
            pending_jobs.append(job)
    stopped = False
    iterator = iter(pending_jobs)
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        active = {}

        def submit():
            job = next(iterator, None)
            if job is None:
                return False
            exp.write_once(root / "intents" / (job["operation_id"] + ".json"), {
                "identity": job["identity"], "started_at": time.time(),
                "reference_answer_visible": False,
            })
            active[pool.submit(exp.solve_worker, job["worker"])] = job
            return True

        for _ in range(4):
            if not submit():
                break
        while active:
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    result = future.result()
                except BaseException as exc:
                    result = {"status": "stopped", "error_type": type(exc).__name__}
                saved = exp.write_once(
                    root / "results" / job["identity"]["arm"] / (job["operation_id"] + ".json"),
                    {"identity": job["identity"], "operation_id": job["operation_id"], **result,
                     "request_records": request_records(exp, root, job["operation_id"]),
                     "reference_answer_visible": False, "finished_at": now()})
                if saved["status"] == "complete":
                    try:
                        validate_receipt(exp, root, job, saved)
                        receipts.append(saved)
                    except BaseException as exc:
                        stopped = True
                        exp.write_once(root / "audit_errors" / (job["operation_id"] + ".json"),
                                       {"error_type": type(exc).__name__})
                else:
                    stopped = True
                print(json.dumps({"arm": job["identity"]["arm"],
                                  "sample_id": job["identity"]["sample_id"],
                                  "status": saved["status"], "completed": len(receipts)}), flush=True)
            live = governor.summary()
            stopped |= bool(live["error_requests"] or live["unknown_requests"])
            progress(exp, root, {"phase": "draining" if stopped else "solving",
                                "completed": len(receipts), "total": SIZE * 2,
                                "inflight_episodes": len(active), "budget": live,
                                "by_arm": dict(Counter(r["identity"]["arm"] for r in receipts))})
            if not stopped:
                while len(active) < 4 and submit():
                    pass
    if stopped:
        exp.write_once(root / "stop.json", {"at": now(), "status": "stopped",
                                          "completed": len(receipts), "budget": governor.summary(),
                                          "automatic_retry": False})
        raise RuntimeError("operational_or_audit_failure_no_retry")
    score(exp, root, manifest, receipts, governor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare-from", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.prepare_from:
        prepare(root, args.prepare_from.resolve())
    else:
        with (root / "owner.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(root)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # Avoid echoing credential-bearing exceptions or provider response bodies.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), flush=True)
        raise SystemExit(1)
