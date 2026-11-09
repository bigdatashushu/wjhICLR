#!/usr/bin/env python3
"""Restore the original cached descriptor parameters before a zero-call recovery."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import choice_cached100 as cached
import pilot40 as pilot
from skill3d.infra.m5_identity import source_identity
from skill3d.segmentation.question_binding import binding_parameters


ROOT = Path(__file__).resolve().parent
ORIGINAL_BASE_CONFIG = pilot.base_config


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def parameters(cfg):
    return binding_parameters(
        semantic_min_pixels=cfg.semantic_min_pixels,
        semantic_max_pixels=cfg.semantic_max_pixels,
        semantic_relation_batch_size=cfg.semantic_relation_batch_size,
        semantic_review_config=cfg.semantic_review_config,
    )


def original_description():
    manifest = read(ROOT / "cached_preparation_manifest.json")
    sampling = read(ROOT / "sampling_manifest.json")
    qa = str(sampling["questions"][0]["qa_id"])
    row = next(r for r in manifest["files"] if r["file"] == f"bindings/{qa}.json")
    path = ROOT / "harness" / row["file"]
    if cached.sha(path) != row["sha256"]:
        raise ValueError("sealed description parameter source changed")
    identity = read(path)["stats"]["generation"]["identity"]
    return identity["parameters"]["experimental_semantic_review"]["target_description"]


def base_config():
    cfg, snapshot = ORIGINAL_BASE_CONFIG()
    old = original_description()
    current = cfg.semantic_review_config["target_description"]
    before = parameters(cfg)
    cfg.semantic_review_config = copy.deepcopy(cfg.semantic_review_config)
    cfg.semantic_review_config["target_description"]["base_url"] = old["base_url"]
    after = parameters(cfg)
    allowed = copy.deepcopy(before)
    allowed["experimental_semantic_review"]["target_description"]["base_url"] = old["base_url"]
    if after != allowed or current["base_url"] == old["base_url"]:
        raise ValueError("recovery is not the exact original descriptor address restoration")
    receipt_path = ROOT / "recovery_compatibility_verified.json"
    if receipt_path.exists() and digest(after) != read(receipt_path)["preparation_parameters_sha256"]:
        raise ValueError("restored cached preparation parameters changed")
    return cfg, snapshot


def compatibility():
    cached.verify_sources(ROOT)
    cached.verify_cache(ROOT, hash_files=False)
    preflight = read(ROOT / "harness/preflight_verified.json")
    if preflight["questions_verified"] != 100 or preflight["model_requests"] != 0:
        raise ValueError("complete original paired preflight is required")
    if list((ROOT / "harness/solver_requests").glob("*/*.json")):
        raise ValueError("recovery cannot repeat an issued solver request")
    if list((ROOT / "harness/outcomes").glob("*.json")):
        raise ValueError("zero-call recovery cannot replace an outcome")
    cfg, snapshot = base_config()
    expected = parameters(cfg)
    sources = source_identity(Path(cached.__file__).resolve().parents[1] / "src/skill3d")
    questions = read(ROOT / "sampling_manifest.json")["questions"]
    checked = []
    for question in questions:
        qa = str(question["qa_id"])
        identity = read(ROOT / "harness/bindings" / f"{qa}.json")["stats"]["generation"]["identity"]
        if identity["parameters"] != expected or identity["sources"] != sources:
            raise ValueError("cached preparation differs beyond the original descriptor address")
        checked.append(qa)
        if len(checked) % 10 == 0:
            print(json.dumps({"phase": "recovery_compatibility", "verified": len(checked),
                              "model_requests": 0}), flush=True)
    receipt = {
        "schema": "vsi-choice-cached-zero-call-recovery/1",
        "questions_verified": len(checked), "qa_ids": checked,
        "preparation_parameters_sha256": digest(expected),
        "original_description_sha256": digest(original_description()),
        "preparation_source_identity_sha256": digest(sources),
        "source_manifest_sha256": cached.sha(ROOT / "source_manifest.json"),
        "cached_preparation_manifest_sha256": cached.sha(ROOT / "cached_preparation_manifest.json"),
        "sampling_sha256": cached.sha(ROOT / "sampling_manifest.json"),
        "paired_preflight_sha256": cached.sha(ROOT / "harness/preflight_verified.json"),
        "recovery_source_sha256": cached.sha(Path(__file__)),
        "skill_snapshot": snapshot,
        "correction": "restore_cached_original_descriptor_base_url_for_read_only_validation",
        "solver_policy_changed": False, "source_frozen_modified": False,
        "issued_model_requests_repeated": 0, "new_preparation_requests": 0,
        "model_requests": 0, "reference_answers_read": False,
        "reference_answer_visible": False, "automatic_retry": False,
    }
    path = ROOT / "recovery_compatibility_verified.json"
    if path.exists() and read(path) != receipt:
        raise ValueError("recovery compatibility receipt changed")
    if not path.exists():
        path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def archive_zero_call_attempt():
    arm = ROOT / "harness"
    launch = read(ROOT / "execute.pid.json")
    proc = Path("/proc") / str(launch["pid"])
    if proc.exists():
        fields = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()
        if fields[19] != launch["process_identity"]["start_ticks"] or fields[0] != "Z":
            raise ValueError("original execute process has not exited")
    state = read(arm / "live_status.json")
    errors = [read(path) for path in (arm / "host_errors").glob("*.json")]
    if (state["status"] != "stopped" or state["completed"] != 0 or len(errors) != 12
            or any(error["stage"] != "solve" or error["error_type"] != "ValueError"
                   or error["stack"][-1]["function"] != "validate_prepared_binding"
                   or error["stack"][-1]["line"] != 486 for error in errors)
            or list((arm / "solver_requests").glob("*/*.json"))
            or list((arm / "outcomes").glob("*.json"))):
        raise ValueError("attempt was not the exact zero-model-call binding gate")
    archive = ROOT / "recovery_r02/pre_model_attempt"
    if archive.parent.exists():
        raise ValueError("recovery archive already exists; duplicate admission disabled")
    archive.mkdir(parents=True)
    for name in ("solver_trace", "solver_requests", "host_errors", "trajectories",
                 "trajectory_index.json", "parallel_events.jsonl", "live_status.json",
                 "parallel_execution.json", "parallel_failure.json"):
        path = arm / name
        if path.exists():
            path.rename(archive / name)
    files = [
        {"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
         "sha256": cached.sha(path)}
        for path in sorted(archive.rglob("*")) if path.is_file()
    ]
    receipt = {
        "schema": "vsi-choice-zero-call-attempt-archive/1",
        "files": files, "errors_preserved": 12, "prior_completed": 0,
        "prior_model_requests": 0, "issued_model_requests_repeated": 0,
        "original_execute_receipt_sha256": cached.sha(ROOT / "execute.pid.json"),
        "compatibility_receipt_sha256": cached.sha(ROOT / "recovery_compatibility_verified.json"),
        "source_frozen_modified": False, "solver_policy_changed": False,
        "reference_answer_visible": False, "automatic_retry": False, "at": time.time(),
    }
    (ROOT / "recovery_r02/admission.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def main():
    phase = sys.argv.pop(1)
    if phase == "compatibility":
        compatibility()
        return
    if phase != "execute":
        raise ValueError("unknown recovery phase")
    receipt = read(ROOT / "recovery_compatibility_verified.json")
    if (receipt["questions_verified"] != 100
            or receipt["recovery_source_sha256"] != cached.sha(Path(__file__))
            or receipt["paired_preflight_sha256"] != cached.sha(ROOT / "harness/preflight_verified.json")
            or receipt["sampling_sha256"] != cached.sha(ROOT / "sampling_manifest.json")
            or list((ROOT / "harness/solver_requests").glob("*/*.json"))
            or list((ROOT / "harness/outcomes").glob("*.json"))):
        raise ValueError("zero-call recovery admission changed")
    archive_zero_call_attempt()
    pilot.base_config = base_config
    cached.install_guard()
    import pilot40_parallel
    pilot40_parallel.main()


if __name__ == "__main__":
    main()
