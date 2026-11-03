"""Bounded BLINK comparison on 24 training examples; all labels stay host-side."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
import datetime
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time

RECOVERY = Path("/home/wangy/wangjh/harness3d/four-datasets-20261031/blink-recovery-20261031-1ee33f28c07e")
FROZEN_SOURCE = RECOVERY/"code/source"
sys.path.insert(0, str(FROZEN_SOURCE/"src"))

from skill3d.evaluation import blink_experiment as exp
from skill3d.small_image.blink_protocol import sha_value, strict_json
from skill3d.synthesis.request_governor import RequestBudgetExhausted
from continuation import campaign_class
RevisedCampaign = campaign_class(exp)

SCHEMA = "harness3d-blink-revised-skill-pilot/1"
ARMS = ("initial_skill",)
SIZE = 24
REQUEST_CAP = SIZE*4


def now():
    return datetime.datetime.now().astimezone().isoformat()


def progress(root, value):
    target = root/"progress.json"
    temporary = root/f".progress-{os.getpid()}.tmp"
    temporary.write_text(json.dumps(exp.seal({"at": now(), **value}),
                                    ensure_ascii=False, sort_keys=True)+"\n")
    os.replace(temporary, target)


def plan(campaign, root, driver_sha256):
    if len(campaign.train) != 40 or len(campaign.tests) != 93:
        raise ValueError("unexpected_original_denominators")
    selected = random.Random(137).sample(sorted(campaign.train, key=lambda p: p.sample_id), SIZE)
    if any(p.source_split != "train" for p in selected):
        raise ValueError("pilot_must_use_training_only")
    start = campaign.governor.summary()
    if start["active_requests"] or start["unknown_requests"]:
        raise ValueError("existing_requests_not_drained")
    if start["policy"]["max_requests"] - start["reserved_requests"] < REQUEST_CAP:
        raise ValueError("insufficient_existing_budget")
    jobs = []
    rng = random.Random(137)
    for p in selected:
        arms = list(ARMS)
        rng.shuffle(arms)
        for arm in arms:
            skill = campaign.inputs["seed"]["text"] if arm == "initial_skill" else ""
            cache = campaign.inputs["cache_identities"][p.sample_id]
            identity = {
                "pilot": SCHEMA, "arm": arm, "source_split": "train",
                "role": "development_validation", "input_sha256": p.input_sha256,
                "seed": 137, "solver": campaign.profile["solver"],
                "model": campaign.profile["model"], "skill_sha256": sha_value(skill),
                "images": cache["images"], "artifact_sha256": cache["artifact_sha256"],
                "allowed_tools": [] if arm == "direct" else exp.TOOLS,
                "recovery_campaign_sha256": campaign.manifest["content_sha256"],
            }
            operation_id = sha_value(identity)
            worker = {
                "problem": p.public_dict(), "skill_text": skill,
                "model": campaign.profile["model"], "solver": campaign.profile["solver"],
                "run_root": str(root),
                "prepared_root": campaign.profile["inputs"]["prepared_root"],
                "artifact_sha256": cache["artifact_sha256"],
                "operation_id": operation_id, "category": "diagnosis",
                "mode": "direct" if arm == "direct" else "system",
                "max_requests": campaign.profile["runtime"]["max_requests"],
                "request_concurrency": campaign.profile["runtime"]["request_concurrency"],
            }
            if any(k in worker for k in ("reference_answer", "answer", "host_answers")):
                raise ValueError("labels_in_worker")
            jobs.append({"identity": identity, "operation_id": operation_id, "worker": worker})
    return {
        "schema": SCHEMA, "created_at": now(),
        "recovery_root": str(campaign.root), "frozen_source": str(campaign.root/"code/source"),
        "recovery_campaign_sha256": campaign.manifest["content_sha256"],
        "source_files": campaign.manifest["source_files"],
        "driver_sha256": driver_sha256, "runtime_identity": exp.runtime_identity(),
        "size": SIZE, "arms": list(ARMS), "jobs": jobs,
        "sample_role": "development_validation", "source_split": "train",
        "selection_seed": 137, "global_budget_root": str(campaign.root/"budget"),
        "budget_start": start, "max_additional_requests": REQUEST_CAP,
        "max_inflight": 4, "automatic_retry": False,
        "reference_answers_visible_to_solver": False, "test_questions_used": False,
        "skill_evolution_in_this_pilot": False,
    }


def audit_prediction(root, job, receipt):
    if (receipt["identity"] != job["identity"] or receipt["status"] != "complete"
            or receipt["result"]["sample_id"] != job["worker"]["problem"]["sample_id"]
            or receipt["result"]["source_split"] != "train"):
        raise ValueError("pilot_result_binding_changed")
    for relative, digest in receipt["request_records"].items():
        if exp.file_sha(root/relative) != digest:
            raise ValueError("pilot_request_record_changed")
    if not 1 <= receipt["result"]["request_count"] <= (1 if job["identity"]["arm"] == "direct" else 4):
        raise ValueError("pilot_episode_request_cap")
    for row in receipt["result"]["rounds"]:
        if row.get("request_parameters"):
            identity = row.get("response_identity", {})
            if identity.get("model") != "qwen3.8-flash" or identity.get("model_reported") is not True:
                raise ValueError("pilot_actual_model_changed")
            if row["request_parameters"]["seed"] != 137:
                raise ValueError("pilot_actual_seed_changed")
            if row["phase"] in ("solve", "finalization") and row.get(
                    "delivered_skill_sha256") != job["identity"]["skill_sha256"]:
                raise ValueError("pilot_skill_not_delivered")


def freeze_predictions(root, manifest):
    rows = []
    for job in manifest["jobs"]:
        path = root/"results"/job["identity"]["arm"]/(job["operation_id"]+".json")
        row = exp.read(path)
        audit_prediction(root, job, row)
        rows.append((job, row))
    if len(rows) != SIZE*len(ARMS):
        raise ValueError("incomplete_pilot_denominator")
    frozen = exp.write_once(root/"predictions_frozen.json", {
        "at": now(), "predictions": {
            str((root/"results"/job["identity"]["arm"]/(job["operation_id"]+".json")).relative_to(root)):
            exp.file_sha(root/"results"/job["identity"]["arm"]/(job["operation_id"]+".json"))
            for job, _ in rows},
        "questions": SIZE, "episodes": len(rows), "reference_answers_read": False,
    })
    return rows, frozen


def score(root, manifest, campaign):
    rows, frozen = freeze_predictions(root, manifest)
    if campaign.governor.summary()["active_requests"] or campaign.governor.summary()["unknown_requests"]:
        raise ValueError("cannot_score_with_active_or_unknown_requests")
    # This host-only call checks the original label-file identity and loads
    # training labels. It occurs after all 24 predictions are frozen and workers exit.
    references = campaign.references("train")
    groups = {arm: [row for job, row in rows if job["identity"]["arm"] == arm] for arm in ARMS}
    scores = {arm: exp.metrics(group, references) for arm, group in groups.items()}
    paired = {}
    by_arm = {arm: {row["result"]["sample_id"]: row for row in group}
              for arm, group in groups.items()}
    for baseline in ("direct", "no_skill"):
        if baseline not in by_arm:
            continue
        wins = losses = 0
        for sample_id, candidate in by_arm["initial_skill"].items():
            parent = by_arm[baseline][sample_id]
            a = (candidate["result"]["status"] == "answered"
                 and candidate["result"]["answer"] == references[sample_id])
            b = (parent["result"]["status"] == "answered"
                 and parent["result"]["answer"] == references[sample_id])
            wins += a and not b
            losses += b and not a
        paired[baseline] = {"wins": wins, "losses": losses,
                            "accuracy_delta": scores["initial_skill"]["accuracy"]-scores[baseline]["accuracy"]}
    usage = campaign.governor.summary()
    additional = usage["reserved_requests"]-manifest["budget_start"]["reserved_requests"]
    if not 0 <= additional <= REQUEST_CAP:
        raise ValueError("pilot_global_request_accounting_mismatch")
    result = exp.write_once(root/"completion.json", {
        "status": "complete", "at": now(), "pilot_sha256": manifest["content_sha256"],
        "size": SIZE, "episodes": SIZE*len(ARMS), "source_split": "train",
        "role": "development_validation", "scores": scores, "paired": paired,
        "prediction_freeze_sha256": frozen["content_sha256"],
        "request_usage": usage, "additional_requests": additional,
        "max_additional_requests": REQUEST_CAP,
        "test_questions_used": False, "skill_evolution_in_this_pilot": False,
    })
    return result


def run(root):
    manifest = exp.read(root/"manifest.json")
    recovery = Path(manifest["recovery_root"])
    campaign = RevisedCampaign(recovery, exp.read(recovery/"manifest.json"))
    if campaign.manifest["content_sha256"] != manifest["recovery_campaign_sha256"]:
        raise ValueError("pilot_recovery_identity_changed")
    if exp.runtime_identity() != manifest["runtime_identity"]:
        raise ValueError("pilot_runtime_changed")
    if (exp.file_sha(root/"pilot.py") != manifest["driver_sha256"]
            or exp.file_sha(root/"continuation.py") != manifest["continuation_driver_sha256"]):
        raise ValueError("pilot_driver_changed")
    with ExitStack() as stack:
        for path in (root/"operator.lock", recovery/"operator.lock",
                     RECOVERY/"operator.lock"):
            stream = stack.enter_context(path.open("rb"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign.verify()
        if (root/"completion.json").exists():
            return exp.read(root/"completion.json")
        if any((root/"intents").glob("*.json")):
            raise RuntimeError("pilot_automatic_replay_forbidden")
        start = campaign.governor.summary()
        if start != manifest["budget_start"]:
            raise ValueError("pilot_budget_changed_before_start")
        completed = {arm: 0 for arm in ARMS}
        progress(root, {"status": "running", "phase": "revised_skill_mechanism_validation",
                        "completed": completed, "episodes": SIZE*len(ARMS), "size": SIZE})
        pending_jobs = iter(manifest["jobs"])
        outstanding = {}
        pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
        try:
            def dispatch():
                job = next(pending_jobs, None)
                if job is None:
                    return False
                exp.write_once(root/"intents"/(job["operation_id"]+".json"),
                               {"identity": job["identity"], "started_at": time.time()})
                future = pool.submit(exp.solve_worker, job["worker"])
                outstanding[future] = job
                return True
            for _ in range(4):
                dispatch()
            while outstanding:
                done, _ = wait(outstanding, return_when=FIRST_COMPLETED)
                for future in done:
                    job = outstanding.pop(future)
                    try:
                        row = future.result()
                    except BaseException as exc:
                        exp.write_once(root/"results"/job["identity"]["arm"]/(job["operation_id"]+".json"),
                                       {"identity": job["identity"], "status": "stopped",
                                        "error_type": type(exc).__name__})
                        raise
                    records = {str(p.relative_to(root)): exp.file_sha(p)
                               for p in sorted((root/"requests"/job["operation_id"]).glob("*.json"))}
                    row = exp.write_once(root/"results"/job["identity"]["arm"]/(job["operation_id"]+".json"),
                                         {**row, "identity": job["identity"], "request_records": records})
                    if row["status"] != "complete":
                        raise RuntimeError("pilot_operational_failure_no_retry")
                    audit_prediction(root, job, row)
                    completed[job["identity"]["arm"]] += 1
                    progress(root, {"status": "running", "phase": "revised_skill_mechanism_validation",
                                    "completed": completed, "episodes": SIZE*len(ARMS), "size": SIZE,
                                    "reserved_requests": campaign.governor.summary()["reserved_requests"]})
                for _ in done:
                    dispatch()
        except BaseException as exc:
            for future in outstanding:
                future.cancel()
            progress(root, {"status": "draining", "error_type": type(exc).__name__,
                            "completed": completed, "automatic_retry": False})
            pool.shutdown(wait=True, cancel_futures=True)
            progress(root, {"status": "stopped", "error_type": type(exc).__name__,
                            "completed": completed, "automatic_retry": False,
                            "request_usage": campaign.governor.summary()})
            raise
        else:
            pool.shutdown(wait=True)
        os.environ.pop("DASHSCOPE_API_KEY", None)
        campaign.verify()
        result = score(root, manifest, campaign)
        progress(root, {"status": "complete", "completed": completed, "scores": result["scores"]})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args.root.resolve())
        print(json.dumps({"status": result["status"], "scores": result["scores"],
                          "additional_requests": result["additional_requests"]}))
        return 0
    except BaseException as exc:
        print(json.dumps({"status": "stopped", "error_type": type(exc).__name__,
                          "automatic_retry": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
