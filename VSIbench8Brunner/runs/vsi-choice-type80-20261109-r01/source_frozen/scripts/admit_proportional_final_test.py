#!/usr/bin/env python3
"""Admit the fixed panel's zero-call final-test episodes after the parent exits."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
ARM = ROOT / "harness"
sys.path[:0] = [str(ROOT / "source_frozen" / "scripts"),
               str(ROOT / "source_frozen" / "src")]

import pilot40 as pilot
import pilot40_parallel as scheduler


GATE_NOTE = (
    "split=final_test：硬约束 9 禁止默认进在线链"
    "（需 allow_final_test 显式盲评）"
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def alive(launch):
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != (
            launch["process_identity"]["boot_id"]):
        raise ValueError("owned parent boot identity changed")
    proc = Path("/proc") / str(launch["pid"])
    if not proc.exists():
        return False
    fields = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()
    if fields[19] != launch["process_identity"]["start_ticks"]:
        raise ValueError("owned parent process identity changed")
    return fields[0] != "Z"


def config_sha(cfg):
    value = pilot.json_value(asdict(cfg))
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def archive_zero_call_episode(qa):
    destination = ARM / "attempt_history" / f"final-test-gated-{qa}-20261106"
    if destination.exists():
        raise ValueError("admission episode was already archived")
    paths = [
        ARM / "outcomes" / f"{qa}.json",
        ARM / "trajectories" / f"{qa}.json",
        *sorted((ARM / "solver_trace" / qa).glob("*.jsonl")),
    ]
    records = [{"original_file": str(p.relative_to(ARM)),
                "sha256": sha(p), "bytes": p.stat().st_size} for p in paths]
    destination.mkdir(parents=True)
    for path, record in zip(paths, records):
        target = destination / path.relative_to(ARM)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        if sha(target) != record["sha256"] or target.stat().st_size != record["bytes"]:
            raise ValueError("archived pre-model episode bytes changed")
        record["file"] = str(target.relative_to(ARM))
    receipt = {
        "qa_id": qa, "reason": "final_test_admission_gate", "model_requests": 0,
        "reference_answer_visible": False, "files": records,
    }
    pilot.write(destination / "archive_receipt.json", receipt)
    return receipt


def main():
    parent_bytes = (ROOT / "harness.pid.json").read_bytes()
    parent = json.loads(parent_bytes)
    launch_path = ROOT / "harness-final-test-admission.pid.json"
    launch_deadline = time.monotonic() + 10
    while not launch_path.exists():
        if time.monotonic() > launch_deadline:
            raise ValueError("owned admission launch receipt is missing")
        time.sleep(0.1)
    launch = read(launch_path)
    if launch["identity_guard"]["sha256"] != sha(Path(__file__)):
        raise ValueError("admission script identity changed")
    deadline = time.monotonic() + 4 * 3600
    pilot.write(ROOT / "final_test_admission_status.json", {
        "status": "waiting_for_parent", "completed": 0, "total": 33,
        "automatic_retry": False, "reference_answer_visible": False,
    })
    while alive(parent):
        if (ROOT / "harness.pid.json").read_bytes() != parent_bytes:
            raise ValueError("owned parent launch receipt changed")
        if time.monotonic() > deadline:
            raise RuntimeError("parent wait exceeded the admission bound")
        time.sleep(10)
    state = read(ARM / "live_status.json")
    if state["status"] != "complete" or state["completed"] != 100 or state.get("errors"):
        raise ValueError("admission requires the original full panel to drain normally")
    sampling = read(ROOT / "sampling_manifest.json")
    if sha(ROOT / "sampling_manifest.json") != parent["sampling_sha256"]:
        raise ValueError("fixed panel changed")
    questions = {str(row["qa_id"]): row for row in sampling["questions"]}
    final_ids = [qa for qa, row in questions.items() if row["source_split"] == "final_test"]
    if len(questions) != 100 or len(final_ids) != 33:
        raise ValueError("admission scope differs from the fixed panel")
    if list((ARM / "host_errors").glob("*.json")):
        raise ValueError("parent has host failures")
    frozen = read(ROOT / "source_manifest.json")
    for entry in frozen["files"]:
        path = ROOT / "source_frozen" / entry["file"]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("frozen source changed")
    preserved = {}
    for qa, question in questions.items():
        result = read(ARM / "outcomes" / f"{qa}.json")
        requests = list((ARM / "solver_requests" / qa).glob("*.json"))
        if any(read(path)["status"] != "returned" for path in requests):
            raise ValueError("an issued parent request is incomplete")
        if qa not in final_ids:
            if not requests:
                raise ValueError("non-final episode has no native model request")
            preserved[qa] = sha(ARM / "outcomes" / f"{qa}.json")
            continue
        outcome = result["outcome"]
        if (requests or result["native_requests"] or outcome["legal_answer"]
                or outcome["agent_rounds"] != 0 or outcome["synthesis_source"] != "none"
                or outcome["notes"] != [GATE_NOTE]
                or result["source_split"] != question["source_split"]
                or result["reference_answer_visible"] is not False
                or list((ARM / "program_executions" / qa).glob("*.json"))):
            raise ValueError("admission can only unblock the exact zero-call split gate")

    original_cfg, _ = pilot.base_config()
    if original_cfg.allow_final_test:
        raise ValueError("original admission setting differs")
    before_cfg_sha = config_sha(original_cfg)
    admitted_cfg = copy.deepcopy(original_cfg)
    admitted_cfg.allow_final_test = True
    admitted_cfg_sha = config_sha(admitted_cfg)
    check_cfg = copy.deepcopy(admitted_cfg)
    check_cfg.allow_final_test = False
    if config_sha(check_cfg) != before_cfg_sha:
        raise ValueError("admission changed scientific configuration")

    # Preserve the original process receipt before publishing the owned continuation.
    original_launch = ROOT / "harness.initial.pid.json"
    if original_launch.exists():
        raise ValueError("original launch was already archived")
    original_launch.write_bytes(parent_bytes)
    pilot.write(ROOT / "harness.pid.json", launch)
    archives = [archive_zero_call_episode(qa) for qa in final_ids]
    admission = {
        "schema": "vsi-fixed-panel-final-test-admission/1",
        "reason": "complete_user_authorized_fixed_100_question_blinded_comparison",
        "qa_ids": final_ids, "questions": 33, "prior_model_requests": 0,
        "issued_model_requests_repeated": 0, "automatic_retry": False,
        "reference_answers_read": False, "reference_answer_visible": False,
        "sampling_sha256": parent["sampling_sha256"],
        "initial_launch": "harness.initial.pid.json",
        "initial_launch_sha256": sha(original_launch),
        "preserved_outcome_sha256": preserved, "archived_episodes": archives,
        "configuration_change": {"allow_final_test": {"before": False, "after": True}},
        "original_config_sha256": before_cfg_sha,
        "admitted_config_sha256": admitted_cfg_sha,
        "scientific_configuration_changed": False,
        "source_frozen_modified": False, "at": time.time(),
    }
    pilot.write(ROOT / "final_test_admission.json", admission)

    guard_path = ROOT / parent["identity_guard"]["file"]
    if sha(guard_path) != parent["identity_guard"]["sha256"]:
        raise ValueError("original response identity guard changed")
    spec = importlib.util.spec_from_file_location("owned_original_guard", guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.install()
    original_config = pilot.base_config

    def blinded_config():
        cfg, snapshot = original_config()
        before = config_sha(cfg)
        if cfg.allow_final_test or before != before_cfg_sha:
            raise ValueError("original admission setting differs")
        cfg = copy.deepcopy(cfg)
        cfg.allow_final_test = True
        enabled = config_sha(cfg)
        check = copy.deepcopy(cfg)
        check.allow_final_test = False
        if config_sha(check) != before or enabled != admitted_cfg_sha:
            raise ValueError("admission changed scientific configuration")
        admission.update(original_config_sha256=before, admitted_config_sha256=enabled,
                         scientific_configuration_changed=False)
        pilot.write(ROOT / "final_test_admission.json", admission)
        return cfg, snapshot

    pilot.base_config = blinded_config
    pilot.write(ROOT / "final_test_admission_status.json", {
        "status": "running", "completed": 0, "total": 33,
        "automatic_retry": False, "reference_answer_visible": False,
    })
    scheduler.main()
    if any(sha(ARM / "outcomes" / f"{qa}.json") != expected
           for qa, expected in preserved.items()):
        raise ValueError("an already issued episode outcome changed")
    for qa in final_ids:
        result = read(ARM / "outcomes" / f"{qa}.json")
        if not result["native_requests"]:
            raise ValueError("admitted episode did not reach the model")
    pilot.write(ROOT / "final_test_admission_status.json", {
        "status": "complete", "completed": 33, "total": 33,
        "automatic_retry": False, "reference_answer_visible": False,
        "issued_model_requests_repeated": 0, "at": time.time(),
    })
    print(json.dumps({"status": "final_test_admission_complete", "questions": 33}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        pilot.write(ROOT / "final_test_admission_failed.json", {
            "error_type": type(exc).__name__, "automatic_retry": False,
            "reference_answer_visible": False, "at": time.time(),
            "stack": [{"file": frame.filename, "line": frame.lineno,
                       "function": frame.name}
                      for frame in traceback.extract_tb(exc.__traceback__)],
        })
        raise SystemExit(1)
