#!/usr/bin/env python3
"""Continue only unissued fixed-panel work, with exact native evidence replay."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
ARM = ROOT / "harness"
REC = ROOT / "recovery_r02"
sys.path[:0] = [str(ROOT / "source_frozen/scripts"),
               str(ROOT / "source_frozen/src")]
import pilot40 as pilot
import pilot40_parallel as scheduler
from skill3d.infra.m5_identity import array_identity, content_sha256, file_identity
from skill3d.infra import vsi_preparation_budget as budget_module
from skill3d.segmentation.cached_presemantic import CachedPresemantic
from skill3d.segmentation.remote_sam2 import RemoteSAM2

GATE = ("split=final_test：硬约束 9 禁止默认进在线链"
        "（需 allow_final_test 显式盲评）")
PUBLISHED = Path("/data/wangy/wangjh/harness3d/experiment-inputs/"
                 "vsibench-m4-success-1000-seed137/published_inputs.json")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def config_sha(cfg):
    return content_sha256(pilot.json_value(asdict(cfg)))


def alive(launch):
    identity = launch["process_identity"]
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != identity["boot_id"]:
        raise ValueError("owned process boot identity changed")
    proc = Path("/proc") / str(identity["pid"])
    if not proc.exists():
        return False
    fields = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()
    if fields[19] != identity["start_ticks"]:
        raise ValueError("owned process start identity changed")
    return fields[0] != "Z"


def frozen_check():
    for row in read(ROOT / "source_manifest.json")["files"]:
        p = ROOT / "source_frozen" / row["file"]
        if p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
            raise ValueError("frozen source changed")
    parent = read(ROOT / "harness.pid.json")
    if alive(parent):
        raise ValueError("previous owned scheduler is active")
    old_helper = ROOT / "harness-final-test-admission.pid.json"
    if old_helper.exists() and alive(read(old_helper)):
        raise ValueError("previous admission helper is active")
    if sha(ROOT / "sampling_manifest.json") != parent["sampling_sha256"]:
        raise ValueError("fixed sampling changed")
    guard = parent["identity_guard"]
    if sha(ROOT / guard["file"]) != guard["sha256"]:
        raise ValueError("original model identity guard changed")
    return parent


def make_plan():
    if (REC / "plan.json").exists():
        raise ValueError("recovery preflight already exists")
    parent = frozen_check()
    REC.mkdir(exist_ok=True)
    args = argparse.Namespace(
        run_dir=ARM, sampling=ROOT / "sampling_manifest.json",
        expected_questions=100, published_inputs=PUBLISHED)
    root, sampling, cfg, provider, overview, predictor = pilot.initialize(args)
    questions = {str(q["qa_id"]): q for q in sampling["questions"]}
    initial = read(ARM / "live_status.json")
    if initial["status"] != "stopped" or initial["completed"] != 91:
        raise ValueError("original stopped panel shape changed")
    budget = read(ARM / "request_budget.json")
    if budget["used"] != 9710 or budget["max_requests"] != 9710:
        raise ValueError("original exhausted budget changed")
    if len(budget["requests"]) != budget["used"]:
        raise ValueError("original budget ledger incomplete")
    final_ids = sorted(q for q, x in questions.items()
                       if x["source_split"] == "final_test")
    if len(final_ids) != 33 or cfg.allow_final_test:
        raise ValueError("original blind admission shape changed")
    outcomes = {p.stem: p for p in (ARM / "outcomes").glob("*.json")}
    preserved, gated, native_files = {}, [], {}
    for qa, p in sorted(outcomes.items()):
        requests = sorted((ARM / "solver_requests" / qa).glob("*.json"))
        if requests:
            if qa in final_ids:
                raise ValueError("final-test episode already issued")
            for request in requests:
                row = read(request)
                if row["status"] != "returned" or row["reference_answer_visible"] is not False:
                    raise ValueError("issued model request incomplete")
                native_files[str(request.relative_to(ARM))] = file_identity(request)
            preserved[qa] = sha(p)
            native_files[str(p.relative_to(ARM))] = file_identity(p)
            print(json.dumps({"preflight_native_qa": qa}), flush=True)
        else:
            result = read(p)
            old = result["outcome"]
            if (qa not in final_ids or result["native_requests"]
                    or old["legal_answer"] or old["agent_rounds"] != 0
                    or old["synthesis_source"] != "none" or old["notes"] != [GATE]
                    or result["reference_answer_visible"] is not False
                    or list((ARM / "program_executions" / qa).glob("*.json"))):
                raise ValueError("zero-call outcome is not the exact split gate")
            gated.append(qa)
    if len(preserved) != 63 or len(gated) != 28:
        raise ValueError("preserved native or zero-call denominator changed")
    pending = sorted(set(questions) - set(preserved))
    missing = sorted(set(questions) - set(outcomes))
    failed = sorted(p.stem for p in (ARM / "host_errors").glob("*.json"))
    if set(failed) != {"3061", "3080", "3085", "3923"}:
        raise ValueError("unexpected operational failure")
    for qa in pending:
        if list((ARM / "solver_requests" / qa).glob("*.json")):
            raise ValueError("continuation would repeat an issued solve")
    prepare_ids = sorted(q for q in missing
                         if q in failed or not (ARM / "bindings" / (q + ".json")).exists())
    if len(prepare_ids) != 8 or len(missing) != 9:
        raise ValueError("remaining preparation shape changed")
    baseline_files = {}
    for qa in failed:
        for base in [ARM / "preparation" / qa,
                     ARM / "bindings" / (qa + ".json"),
                     ARM / "sam2_receipts" / (qa + ".json"),
                     ARM / "host_errors" / (qa + ".json"),
                     ARM / "trajectories" / (qa + ".json")]:
            paths = sorted(base.rglob("*")) if base.is_dir() else [base]
            for p in paths:
                if p.is_file():
                    baseline_files[str(p.relative_to(ARM))] = file_identity(p)
        error = read(ARM / "host_errors" / (qa + ".json"))
        if error.get("stage") != "prepare" or error["error_type"] != "RuntimeError":
            raise ValueError("failure is not a zero-solver preparation stop")
    previous_inputs = {q["qa_id"]: q for q in read(
        ARM / "preflight_verified.json")["questions"]}
    details, extra_cap = {}, 0
    for qa in prepare_ids:
        entry = provider.entries[qa]
        v = pilot.variant(questions[qa], entry)
        item = provider(v)
        transport = provider.prepare_published(v, item)
        if (item.episode.ground_truth != "" or item.episode.qa_id != qa
                or item.episode.question != questions[qa]["question"]
                or item.episode.split != questions[qa]["source_split"]
                or (item.episode.options or []) != (questions[qa].get("options") or [])):
            raise ValueError("remaining input is not the original label-free episode")
        prior = previous_inputs[qa]
        prepared = transport["prepared"]
        if (item.episode.frame_set.frame_set_hash != prior["frame_set_hash"]
                or item.episode.frame_set.pixel_sha256 != prior["pixel_sha256"]
                or content_sha256(entry["reference"]) != prior["source_reference_sha256"]
                or prepared.stats["generation"]["sha256"] != prior["binding_generation_sha256"]
                or content_sha256(transport["transport_receipt"]) != prior["transport_receipt_sha256"]):
            raise ValueError("immutable prepared input changed")
        stats = prepared.stats
        candidates = [{k: x[k] for k in ("candidate_id", "frame_slot", "bbox", "category")}
                      for x in stats.get("candidate_receipts", [])]
        selected = stats.get("review_selection", {}).get("selected_candidate_ids", [])
        visual_paths = list((ARM / "preparation" / qa).rglob("visual/*.json"))
        visual, issued = {}, []
        qa_extra = 0
        for p in visual_paths:
            saved = read(p)
            cid = saved["candidate_id"]
            if cid in visual or cid not in selected:
                raise ValueError("duplicate or unexpected native candidate")
            d = saved.get("target_description") or {}
            if d.get("request_sent"):
                if d.get("status") != "described":
                    raise ValueError("an issued description failed; retry forbidden")
                issued.append({"provider": "qwen3vl8b_description",
                               "request_sha256": d["request_sha256"]})
            elif d.get("status") != "unavailable" or d.get("error_type") != "RuntimeError":
                raise ValueError("unexpected zero-issued description failure")
            if saved.get("request_sent"):
                if saved.get("status") != "observed":
                    raise ValueError("an issued binary review failed; retry forbidden")
                issued.append({"provider": "intern_binary_review",
                               "request_sha256": saved["request_sha256"]})
            elif saved.get("error") not in ("RuntimeError", "DescriptionFailure"):
                raise ValueError("unexpected binary admission failure")
            visual[cid] = {"original_file": str(p.relative_to(ARM)),
                           "identity": file_identity(p),
                           "description_observed": d.get("status") == "described",
                           "binary_observed": saved.get("status") == "observed"}
        for cid in selected:
            saved = visual.get(cid, {})
            qa_extra += (0 if saved.get("binary_observed") else
                         1 if saved.get("description_observed") else 2)
        batches = []
        if qa in failed:
            receipts = read(ARM / "sam2_receipts" / (qa + ".json"))
            groups = [[c for c in candidates if int(c["frame_slot"]) == slot]
                      for slot in sorted({int(c["frame_slot"]) for c in candidates})]
            if len(groups) != len(receipts):
                raise ValueError("issued SAM2 batch coverage is incomplete")
            import numpy as np
            for group, receipt in zip(groups, receipts):
                if [[float(v) for v in c["bbox"]] for c in group] != receipt["boxes"]:
                    raise ValueError("original SAM2 batch boxes differ")
                if len(group) != len(receipt["masks"]):
                    raise ValueError("original SAM2 batch denominator differs")
                masks = []
                for c, expected in zip(group, receipt["masks"]):
                    paths = list((ARM / "preparation" / qa).rglob(
                        "raw_masks/" + c["candidate_id"] + ".npz"))
                    if len(paths) != 1:
                        raise ValueError("issued SAM2 mask was not preserved")
                    p = paths[0]
                    with np.load(p, allow_pickle=False) as arrays:
                        if array_identity(arrays["mask"]) != expected:
                            raise ValueError("native SAM2 mask bytes do not replay")
                    masks.append({"original_file": str(p.relative_to(ARM)),
                                  "identity": file_identity(p), "array_identity": expected})
                batches.append({"receipt": receipt, "masks": masks})
        elif visual_paths or list((ARM / "sam2_receipts").glob(qa + ".json")):
            raise ValueError("unprepared question has prior issued evidence")
        extra_cap += qa_extra
        details[qa] = {"scene_id": entry["scene_id"], "selected_candidates": len(selected),
                       "additional_description_binary_cap": qa_extra,
                       "native_visual": visual, "prior_issued": issued,
                       "sam2_replay_batches": batches,
                       "new_sam2_batch_cap": 0 if batches else len(
                           {c["frame_slot"] for c in candidates}),
                       "source_reference_sha256": prior["source_reference_sha256"],
                       "binding_generation_sha256": prior["binding_generation_sha256"]}
        print(json.dumps({"preflight_prepare_qa": qa, "extra_cap": qa_extra,
                          "sam2_replay_batches": len(batches)}), flush=True)
        provider.cache.clear()
        provider._publication_files.clear()
        provider._scene_checks.clear()
        provider._scene_transport_scopes.clear()
    for filename in ("request_budget.json", "runtime_identity.json",
                     "preflight_verified.json", "trajectory_index.json",
                     "parallel_execution.json", "live_status.json"):
        shutil.copyfile(ARM / filename, REC / ("original_" + filename))
    for filename in ("harness.pid.json", "harness-final-test-admission.pid.json",
                     "final_test_admission_failed.json"):
        if (ROOT / filename).exists():
            shutil.copyfile(ROOT / filename, REC / ("original_" + filename))
    check = copy.deepcopy(cfg)
    check.allow_final_test = True
    plan = {
        "schema": "vsi-fixed100-native-continuation-plan/2", "at": time.time(),
        "sampling_sha256": sha(ROOT / "sampling_manifest.json"),
        "source_manifest_sha256": sha(ROOT / "source_manifest.json"),
        "script_sha256": sha(Path(__file__)), "parent": parent,
        "original_budget_sha256": sha(REC / "original_request_budget.json"),
        "original_budget_max": 9710, "original_budget_used": 9710,
        "additional_preparation_request_cap": extra_cap,
        "continued_budget_max": 9710 + extra_cap,
        "preserved_outcome_sha256": preserved, "preserved_native_files": native_files,
        "gated_qa_ids": gated, "final_test_qa_ids": final_ids,
        "unissued_final_test_qa_ids": sorted(set(final_ids) - set(gated)),
        "missing_qa_ids": missing, "pending_qa_ids": pending,
        "failed_preparation_qa_ids": failed, "prepare_qa_ids": prepare_ids,
        "preparation": details, "failed_native_files": baseline_files,
        "original_config_sha256": config_sha(cfg), "admitted_config_sha256": config_sha(check),
        "model_requests": 0, "gpu_requests": 0, "reference_answers_read": False,
        "reference_answer_visible": False, "automatic_retry": False,
        "scientific_configuration_changed": False, "source_frozen_modified": False,
    }
    pilot.write(REC / "plan.json", plan)
    print(json.dumps({"status": "recovery_zero_request_preflight_complete",
                      "pending_solves": len(pending), "prepare_questions": len(details),
                      "additional_preparation_request_cap": extra_cap}), flush=True)


def archive_paths(paths, destination):
    records = []
    for path in paths:
        if not path.exists():
            continue
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        rows = [{"original_file": str(p.relative_to(ARM)), "identity": file_identity(p)}
                for p in files]
        target = destination / path.relative_to(ARM)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        for row in rows:
            p = destination / row["original_file"]
            if file_identity(p) != row["identity"]:
                raise ValueError("archived native bytes changed")
            row["file"] = str(p.relative_to(ARM))
        records.extend(rows)
    return records


def execute():
    plan = read(REC / "plan.json")
    if sha(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("continuation script changed after preflight")
    if (REC / "started.json").exists():
        raise ValueError("continuation already admitted; automatic rerun disabled")
    parent = frozen_check()
    if parent != plan["parent"] or sha(ARM / "request_budget.json") != plan["original_budget_sha256"]:
        raise ValueError("original stopped state changed after preflight")
    for name, identity in {**plan["preserved_native_files"],
                           **plan["failed_native_files"]}.items():
        if file_identity(ARM / name) != identity:
            raise ValueError("native evidence changed after rehearsal")
    if sha(ROOT / "sampling_manifest.json") != plan["sampling_sha256"]:
        raise ValueError("fixed panel changed")
    launch = read(ROOT / "harness-recovery-r02.pid.json")
    if launch["identity_guard"]["sha256"] != plan["script_sha256"]:
        raise ValueError("owned continuation launch identity differs")
    original = read(REC / "original_request_budget.json")
    prior_failed = set(plan["failed_preparation_qa_ids"])
    archives, failed_archives = [], []
    for qa in plan["gated_qa_ids"]:
        destination = ARM / "attempt_history" / ("r02-zero-call-gate-" + qa)
        if destination.exists():
            raise ValueError("gate archive already exists")
        files = archive_paths([ARM / "outcomes" / (qa + ".json"),
                               ARM / "trajectories" / (qa + ".json"),
                               ARM / "solver_trace" / qa], destination)
        record = {"qa_id": qa, "model_requests": 0, "reference_answer_visible": False,
                  "files": files}
        pilot.write(destination / "archive_receipt.json", record)
        archives.append(record)
    for qa in sorted(prior_failed):
        destination = ARM / "attempt_history" / ("r02-budget-stop-" + qa)
        if destination.exists():
            raise ValueError("preparation archive already exists")
        files = archive_paths([
            ARM / "preparation" / qa, ARM / "bindings" / (qa + ".json"),
            ARM / "sam2_receipts" / (qa + ".json"),
            ARM / "host_errors" / (qa + ".json"),
            ARM / "trajectories" / (qa + ".json")], destination)
        for row in files:
            if row["identity"] != plan["failed_native_files"][row["original_file"]]:
                raise ValueError("failed attempt differs from rehearsal")
        failed_archives.append({"qa_id": qa, "files": files})
        detail = plan["preparation"][qa]
        replay_root = REC / "review_replay" / qa
        (replay_root / "visual").mkdir(parents=True)
        entries = {}
        for cid, row in detail["native_visual"].items():
            source = destination / row["original_file"]
            target = replay_root / "visual" / (cid + ".json")
            shutil.copyfile(source, target)
            if file_identity(target) != row["identity"]:
                raise ValueError("replay copy changed")
            entries[cid] = {"path": "visual/" + cid + ".json", "identity": row["identity"]}
        pilot.write(replay_root / "manifest.json", {"files": entries})
        for batch in detail["sam2_replay_batches"]:
            for mask in batch["masks"]:
                mask["file"] = str((destination / mask["original_file"]).relative_to(ROOT))
    original["max_requests"] = plan["continued_budget_max"]
    original["continuation"] = {
        "schema": "fixed100-budget-continuation/2", "prior_max_requests": 9710,
        "prior_used": 9710, "additional_cap": plan["additional_preparation_request_cap"],
        "reason": "unissued_candidates_in_the_same_authorized_fixed100_panel",
        "plan_sha256": sha(REC / "plan.json"), "automatic_retry": False}
    pilot.write(ARM / "request_budget.json", original)
    pilot.write(REC / "sam2_new_admission.json", {"requests": []})
    started = {
        "schema": "vsi-fixed100-native-continuation/2", "at": time.time(),
        "plan_sha256": sha(REC / "plan.json"), "sampling_sha256": plan["sampling_sha256"],
        "initial_launch_sha256": sha(REC / "original_harness.pid.json"),
        "failed_admission_launch_sha256": sha(REC / "original_harness-final-test-admission.pid.json"),
        "preserved_outcome_sha256": plan["preserved_outcome_sha256"],
        "pending_qa_ids": plan["pending_qa_ids"], "archives": archives,
        "failed_preparation_archives": failed_archives,
        "preparation": plan["preparation"], "automatic_retry": False,
        "issued_model_requests_repeated": 0, "reference_answers_read": False,
        "reference_answer_visible": False, "source_frozen_modified": False,
        "scientific_configuration_changed": False,
    }
    pilot.write(REC / "started.json", started)
    initial_launch = ROOT / "harness.initial.pid.json"
    if initial_launch.exists():
        raise ValueError("initial launch archive already exists")
    initial_launch.write_bytes((ROOT / "harness.pid.json").read_bytes())
    pilot.write(ROOT / "harness.pid.json", launch)
    admission = {
        "schema": "vsi-fixed-panel-final-test-admission/2",
        "qa_ids": plan["final_test_qa_ids"], "questions": 33,
        "prior_model_requests": 0, "issued_model_requests_repeated": 0,
        "automatic_retry": False, "reference_answers_read": False,
        "reference_answer_visible": False, "source_frozen_modified": False,
        "scientific_configuration_changed": False,
        "sampling_sha256": plan["sampling_sha256"],
        "initial_launch": "harness.initial.pid.json",
        "initial_launch_sha256": sha(initial_launch),
        "preserved_outcome_sha256": plan["preserved_outcome_sha256"],
        "archived_episodes": archives,
        "unissued_qa_ids": plan["unissued_final_test_qa_ids"],
        "configuration_change": {"allow_final_test": {"before": False, "after": True}},
        "original_config_sha256": plan["original_config_sha256"],
        "admitted_config_sha256": plan["admitted_config_sha256"],
        "recovery_receipt": "recovery_r02/started.json",
    }
    pilot.write(ROOT / "final_test_admission.json", admission)
    original_config = pilot.base_config

    def blinded_config():
        cfg, snapshot = original_config()
        if cfg.allow_final_test or config_sha(cfg) != plan["original_config_sha256"]:
            raise ValueError("original configuration differs from rehearsal")
        cfg = copy.deepcopy(cfg)
        cfg.allow_final_test = True
        if config_sha(cfg) != plan["admitted_config_sha256"]:
            raise ValueError("blind admission changed another method setting")
        return cfg, snapshot

    pilot.base_config = blinded_config
    guard_path = ROOT / parent["identity_guard"]["file"]
    spec = importlib.util.spec_from_file_location("original_model_guard", guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.install()
    original_cached_init = CachedPresemantic.__init__

    def cached_init(self, *args, **kwargs):
        qa = pilot.CURRENT["qa_id"]
        if qa not in plan["prepare_qa_ids"]:
            raise ValueError("preparation outside the unissued scope")
        replay_root = REC / "review_replay" / qa
        if qa in prior_failed:
            os.environ["VSI8B_REVIEW_REPLAY_ROOT"] = str(replay_root)
        else:
            os.environ.pop("VSI8B_REVIEW_REPLAY_ROOT", None)
        original_cached_init(self, *args, **kwargs)

    CachedPresemantic.__init__ = cached_init
    original_sam = RemoteSAM2.segment_frame_boxes

    def segment(self, image, boxes):
        qa = pilot.CURRENT["qa_id"]
        detail = plan["preparation"][qa]
        frame_id = array_identity(image)
        normalized = [[float(v) for v in b] for b in boxes]
        if qa in prior_failed:
            matches = [b for b in detail["sam2_replay_batches"]
                       if b["receipt"]["frame_identity"] == frame_id
                       and b["receipt"]["boxes"] == normalized]
            if len(matches) != 1:
                raise ValueError("native SAM2 batch does not match exactly")
            batch = matches[0]
            import numpy as np
            masks = []
            for row in batch["masks"]:
                p = ROOT / row["file"]
                if file_identity(p) != row["identity"]:
                    raise ValueError("native SAM2 mask file changed")
                with np.load(p, allow_pickle=False) as arrays:
                    mask = arrays["mask"].copy()
                if array_identity(mask) != row["array_identity"]:
                    raise ValueError("native SAM2 mask array changed")
                masks.append(mask)
            receipt = copy.deepcopy(batch["receipt"])
            if receipt["identity_sha256"] != self.identity_sha256:
                raise ValueError("replayed SAM2 service identity changed")
            receipt["native_replay"] = {"model_requests": 0,
                                       "source_masks": batch["masks"]}
            self.receipts.append(receipt)
            return masks
        key = content_sha256({"frame_identity": frame_id, "boxes": normalized})
        with (REC / "sam2_new_admission.json").open("r+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            state = json.load(stream)
            rows = [r for r in state["requests"] if r["qa_id"] == qa]
            if any(r["request_sha256"] == key for r in rows) or len(rows) >= detail["new_sam2_batch_cap"]:
                raise ValueError("SAM2 call repeated or outside the frozen frame scope")
            state["requests"].append({"qa_id": qa, "request_sha256": key,
                                      "at": time.time(), "automatic_retry": False})
            stream.seek(0)
            stream.truncate()
            json.dump(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        return original_sam(self, image, boxes)

    RemoteSAM2.segment_frame_boxes = segment

    def reserve(provider, request):
        qa = pilot.CURRENT["qa_id"]
        if qa not in plan["prepare_qa_ids"] or provider not in (
                "qwen3vl8b_description", "intern_binary_review"):
            raise ValueError("model admission outside fixed remaining preparation")
        digest = content_sha256(request)
        prior = plan["preparation"][qa]["prior_issued"]
        if any(r["provider"] == provider and r["request_sha256"] == digest for r in prior):
            raise ValueError("replay attempted to repeat an already issued model call")
        with (ARM / "request_budget.json").open("r+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            state = json.load(stream)
            new = state["requests"][9710:]
            own = [r for r in new if r["qa_id"] == qa]
            if (state["used"] >= state["max_requests"]
                    or len(own) >= plan["preparation"][qa]["additional_description_binary_cap"]
                    or any(r["provider"] == provider and r["request_sha256"] == digest for r in own)):
                raise ValueError("continuation model call repeated or cap exceeded")
            state["used"] += 1
            state["requests"].append({
                "number": state["used"], "provider": provider,
                "request_sha256": digest, "reserved_at": time.time(),
                "qa_id": qa, "continuation": "r02", "automatic_retry": False})
            stream.seek(0)
            stream.truncate()
            json.dump(state, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())

    budget_module.reserve = reserve
    original_argv = sys.argv
    sys.argv = [str(Path(__file__)), "--sampling", str(ROOT / "sampling_manifest.json"),
                "--published-inputs", str(PUBLISHED), "--run-dir", str(ARM),
                "--expected-questions", "100", "--preparation-request-cap",
                str(plan["continued_budget_max"]), "--prepare-workers", "4",
                "--solve-workers", "16", "--cpu-limit", "8", "--api-limit", "32"]
    pilot.write(REC / "status.json", {"status": "running", "pending_solves": 37,
                                     "reference_answer_visible": False})
    try:
        scheduler.main()
    finally:
        sys.argv = original_argv
    for qa, expected in plan["preserved_outcome_sha256"].items():
        if sha(ARM / "outcomes" / (qa + ".json")) != expected:
            raise ValueError("previously issued outcome changed")
    for name, identity in plan["preserved_native_files"].items():
        if file_identity(ARM / name) != identity:
            raise ValueError("previously issued native response changed")
    for qa in plan["pending_qa_ids"]:
        result = read(ARM / "outcomes" / (qa + ".json"))
        if not result["native_requests"] or any(r["status"] != "returned"
                                               for r in result["native_requests"]):
            raise ValueError("pending solve lacks complete native responses")
    final_budget = read(ARM / "request_budget.json")
    if final_budget["requests"][:9710] != read(REC / "original_request_budget.json")["requests"]:
        raise ValueError("original preparation ledger changed")
    if len(final_budget["requests"][9710:]) != plan["additional_preparation_request_cap"]:
        raise ValueError("remaining fixed candidate preparation denominator differs")
    pilot.write(REC / "status.json", {
        "status": "complete", "pending_solves_completed": 37,
        "all100_completed": True, "preserved_native_questions": 63,
        "issued_model_requests_repeated": 0, "reference_answer_visible": False,
        "new_preparation_requests": final_budget["used"] - 9710, "at": time.time()})
    pilot.write(ROOT / "final_test_admission_status.json", {
        "status": "complete", "completed": 33, "total": 33,
        "automatic_retry": False, "reference_answer_visible": False,
        "issued_model_requests_repeated": 0, "at": time.time()})
    print(json.dumps({"status": "fixed100_native_continuation_complete"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "execute"), required=True)
    args = parser.parse_args()
    try:
        make_plan() if args.phase == "preflight" else execute()
    except Exception as exc:
        REC.mkdir(exist_ok=True)
        pilot.write(REC / (args.phase + "_failed.json"), {
            "error_type": type(exc).__name__, "automatic_retry": False,
            "reference_answer_visible": False, "at": time.time(),
            "stack": [{"file": f.filename, "line": f.lineno, "function": f.name}
                      for f in traceback.extract_tb(exc.__traceback__)]})
        print(json.dumps({"status": "recovery_stopped", "phase": args.phase,
                          "error_type": type(exc).__name__}), flush=True)
        raise SystemExit(1)
