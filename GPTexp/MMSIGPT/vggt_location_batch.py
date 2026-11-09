"""Frozen location-language batch; continuations dispatch only untouched questions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time
from types import FunctionType

from ..traces import digest, file_digest, read, write
from . import vggt_location_trial as old
from . import vggt_location_batch_gpu as gpu
from .prototype.direct import operator_lock
from .prototype.inputs import public_problem

ARM = old.ARM
request = old.request
base = old.base


def code_identity():
    return {**old.code_identity(), **{
        str((base.PACKAGE / name).relative_to(base.ROOT)): file_digest(base.PACKAGE / name)
        for name in ("vggt_location_batch.py", "vggt_location_batch_gpu.py",
                     "vggt_location_batch_worker.py", "vggt_location_batch_host.py")
    }}


def assert_pose(root, item, pose_path, pose_sha):
    pose = read(Path(root) / pose_path)
    config = read(Path(root) / "gpu/gpu-input.json")
    if (file_digest(Path(root) / pose_path) != pose_sha
            or pose["schema"] != "mmsi-joint-vggt-camera/1"
            or pose["sample_id"] != item["problem"]["sample_id"]
            or pose["input_sha256"] != item["problem"]["input_sha256"]
            or pose["original_image_sha256"] != [i["sha256"] for i in item["images"]]
            or pose["weight_sha256"] != base.WEIGHT_SHA
            or pose["joint_forward"] is not True or len(item["images"]) != 2
            or pose["environment"] != config["environment"]
            or pose["preprocess"] != "official_load_and_preprocess_images_crop_width518_patch14"
            or pose["model_requests"] != 0
            or any(pose[k] is not False for k in (
                "depth_head_executed", "point_head_executed",
                "track_head_executed", "reference_answer_visible"))):
        raise ValueError("pose_or_original_input_changed")
    return pose


def prepare(root):
    root = Path(root).resolve()
    selected = gpu.selection(root)
    receipt = read(root / "gpu-fetch.json")
    if receipt["joint_forwards"] != 240 or receipt["status"] != "verified":
        raise ValueError("gpu_not_complete")
    if (root / "manifest.json").exists():
        raise ValueError("already_prepared")
    (root / "poses").mkdir()
    items = []
    for original in selected["items"]:
        public_problem(original["problem"])
        sample = original["problem"]["sample_id"]
        source = "gpu/poses/" + sample + ".json"
        expected = receipt["files"]["poses/" + sample + ".json"]
        pose = assert_pose(root, original, source, expected)
        path = "poses/" + sample + ".json"
        shutil.copyfile(root / source, root / path)
        item = {
            **original, "pose_path": path, "pose_sha256": expected,
            "relation_rows": old.location_relations(pose["camera_evidence"]),
            "relation_text": old.relation_note(pose["camera_evidence"]),
        }
        item["payload_sha256"] = {ARM: digest(request(root, item, ARM))}
        items.append(item)
    manifest = {
        "schema": "mmsi-location-batch/1", "batch_root": str(root), "phase": 0,
        "selection_sha256": file_digest(root / "selection.json"),
        "selection_identity": selected["selection_identity"],
        "gpu_identity_files": {name: file_digest(root / name) for name in (
            "gpu/gpu-input.json", "gpu/gpu-started.json", "gpu/gpu-complete.json",
            "gpu/launch.json", "gpu-fetch.json", "gpu-transfer.json", "gpu-probe.json")},
        "items": items, "questions": 240, "max_requests": 240,
        "campaign_max_requests": 240, "max_inflight": 6,
        "api_ids": selected["api_ids"], "inference_by_api": selected["inference_by_api"],
        "code_identity": code_identity(), "environment": base.environment(),
        "prompt_version": old.PROMPT_VERSION, "parser_version": old.PARSER_VERSION,
        "relation_version": old.RELATION_VERSION, "sample_role": selected["sample_role"],
        "reference_answer_visible": False, "automatic_retry": False, "tools_enabled": False,
        "depth_images_sent": False, "numeric_pose_sent": False,
        "camera_orientation_relation_sent": False, "gpu_forwards": 240,
        "prior_phases": [], "prepared_at": time.time(),
    }
    save(root, manifest)
    paths = {**manifest["code_identity"]}
    for name, expected in paths.items():
        path = root / "source_frozen" / name
        if path.exists():
            if file_digest(path) != expected:
                raise ValueError("historical_snapshot_changed")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base.ROOT / name, path)
    write(root / "batch-source-snapshot.json", {
        "run_identity": manifest["run_identity"], "code_identity": paths,
    }, exclusive=True)
    preview = ["# 本轮逐题拍摄位置关系提示", ""]
    for item in items:
        preview.extend([f"## {item['problem']['sample_id']} · {item['problem']['task_type']}",
                        "", item["relation_text"], ""])
    (root / "提示预览.md").write_text("\n".join(preview), encoding="utf-8")
    print('{"status":"batch_prepared","questions":240,"model_requests":0}', flush=True)


def save(directory, manifest):
    manifest["run_identity"] = digest(manifest)
    write(Path(directory) / "manifest.json", manifest, exclusive=True)
    write(Path(directory) / "preflight.json", {
        "run_identity": manifest["run_identity"], "model_requests": 0,
        "questions": manifest["questions"], "reference_answers_read": False,
        "original_images_verified": True, "relation_text_verified": True,
    }, exclusive=True)
    validate(directory, verify_requests=True)


def remaining_items(items, phase_records):
    expected = {i["problem"]["sample_id"]: i for i in items}
    dispatched = set()
    for records in phase_records:
        for p in records:
            sample = p["sample_id"]
            if sample not in expected or p["input_sha256"] != expected[sample]["problem"]["input_sha256"]:
                raise ValueError("phase_question_binding_changed")
            if p["status"] == "not_dispatched":
                continue
            if sample in dispatched:
                raise ValueError("duplicate_question_attempt")
            dispatched.add(sample)
    if len(dispatched) > 240:
        raise ValueError("campaign_request_budget_exceeded")
    return [item for item in items if item["problem"]["sample_id"] not in dispatched]


def validate(directory, *, verify_requests=False):
    directory = Path(directory).resolve()
    manifest = read(directory / "manifest.json")
    root = Path(manifest["batch_root"])
    selected = gpu.selection(root)
    if (manifest["schema"] != "mmsi-location-batch/1"
            or digest({k: v for k, v in manifest.items() if k != "run_identity"})
            != manifest["run_identity"] or manifest["code_identity"] != code_identity()
            or manifest["environment"] != base.environment()
            or manifest["selection_sha256"] != file_digest(root / "selection.json")
            or manifest["selection_identity"] != selected["selection_identity"]
            or any(file_digest(root / name) != expected
                   for name, expected in manifest["gpu_identity_files"].items())
            or manifest["campaign_max_requests"] != 240
            or manifest["questions"] != len(manifest["items"])
            or manifest["max_requests"] != manifest["questions"]
            or not 0 < manifest["questions"] <= 240
            or manifest["max_inflight"] != len(manifest["api_ids"])
            or manifest["inference_by_api"] != {api: base.settings(api) for api in manifest["api_ids"]}
            or manifest["relation_version"] != old.RELATION_VERSION
            or manifest["prompt_version"] != old.PROMPT_VERSION
            or manifest["parser_version"] != old.PARSER_VERSION
            or manifest["gpu_forwards"] != 240
            or any(manifest[k] is not False for k in (
                "reference_answer_visible", "automatic_retry", "tools_enabled",
                "depth_images_sent", "numeric_pose_sent", "camera_orientation_relation_sent"))):
        raise ValueError("batch_configuration_changed")
    if manifest["phase"] == 0:
        if (directory != root or manifest["prior_phases"] or manifest["questions"] != 240
                or manifest["api_ids"] != selected["api_ids"]):
            raise ValueError("root_batch_changed")
        expected_items = selected["items"]
    else:
        first = validate(root)
        phases = manifest["prior_phases"]
        if len(phases) != manifest["phase"] or phases[0]["directory"] != str(root):
            raise ValueError("continuation_chain_changed")
        records = []
        for index, prior in enumerate(phases):
            path = Path(prior["directory"])
            if any(file_digest(path / name) != prior[name + "_sha256"]
                   for name in ("manifest.json", "frozen.json")):
                raise ValueError("prior_phase_changed")
            previous = validate(path)
            if previous["phase"] != index:
                raise ValueError("prior_phase_order_changed")
            records.append(validate_frozen(path, previous)["predictions"])
        expected_items = remaining_items(first["items"], records)
        active = {i["api_id"] for i in expected_items}
        if manifest["api_ids"] != [api for api in selected["api_ids"] if api in active]:
            raise ValueError("active_slot_binding_changed")
    if [i["problem"]["sample_id"] for i in manifest["items"]] != [
            i["problem"]["sample_id"] for i in expected_items]:
        raise ValueError("question_selection_changed")
    for item, original in zip(manifest["items"], expected_items):
        if (any(item[k] != original[k] for k in (
                "index", "problem", "images", "api_id", "arm_order",
                "previously_selected_or_dispatched")) or item["arm_order"] != [ARM]):
            raise ValueError("original_question_or_api_changed")
        pose = assert_pose(root, item, "gpu/" + item["pose_path"], item["pose_sha256"])
        if (file_digest(directory / item["pose_path"]) != item["pose_sha256"]
                or item["relation_text"] != old.relation_note(pose["camera_evidence"])
                or item["relation_rows"] != old.location_relations(pose["camera_evidence"])):
            raise ValueError("relation_text_changed")
        if verify_requests and digest(request(directory, item, ARM)) != item["payload_sha256"][ARM]:
            raise ValueError("prepared_payload_changed")
    preflight = read(directory / "preflight.json")
    if (preflight["run_identity"] != manifest["run_identity"]
            or preflight["model_requests"] != 0 or preflight["reference_answers_read"] is not False):
        raise ValueError("zero_request_preflight_changed")
    return manifest


_namespace = {**vars(old.shared), "validate": validate, "request": request, "inference": base.settings}
execute = FunctionType(old.shared.execute.__code__, _namespace, "execute")
execute.__kwdefaults__ = {"client_factory": base.BoundedClient}
collect = old.collect
validate_frozen = old.validate_frozen


def freeze(directory):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory)
        records, files = collect(directory, manifest)
        if (directory / "frozen.json").exists():
            return validate_frozen(directory, manifest)
        value = {
            "schema": "mmsi-location-batch-phase-freeze/1",
            "run_identity": manifest["run_identity"], "predictions": records, "files": files,
            "reference_answer_visible": False, "automatic_retry": False, "frozen_at": time.time(),
        }
        value["freeze_identity"] = digest(value)
        write(directory / "frozen.json", value, exclusive=True)
        return value


def prepare_remaining(root, directories):
    root = Path(root).resolve()
    first = validate(root)
    priors, records = [], []
    for index, directory in enumerate(directories):
        directory = Path(directory).resolve()
        manifest = validate(directory)
        if manifest["phase"] != index or manifest["batch_root"] != str(root):
            raise ValueError("continuation_phase_order_changed")
        frozen = validate_frozen(directory, manifest)
        priors.append({"directory": str(directory), **{
            name + "_sha256": file_digest(directory / name) for name in ("manifest.json", "frozen.json")
        }})
        records.append(frozen["predictions"])
    pending = remaining_items(first["items"], records)
    if not pending:
        raise ValueError("no_untouched_questions")
    destination = root / f"continuation-{len(priors):02d}"
    destination.mkdir()
    (destination / "poses").mkdir()
    for item in pending:
        shutil.copyfile(root / item["pose_path"], destination / item["pose_path"])
    active = {i["api_id"] for i in pending}
    apis = [api for api in first["api_ids"] if api in active]
    manifest = {
        **{k: v for k, v in first.items() if k != "run_identity"},
        "phase": len(priors), "prior_phases": priors, "items": pending,
        "questions": len(pending), "max_requests": len(pending), "max_inflight": len(apis),
        "api_ids": apis, "inference_by_api": {api: base.settings(api) for api in apis},
        "prepared_at": time.time(),
    }
    save(destination, manifest)
    return destination


def finalize(root, directories):
    root = Path(root).resolve()
    with operator_lock(root):
        first = validate(root)
        phases, records, files = [], [], {}
        for index, path in enumerate(directories):
            path = Path(path).resolve()
            manifest = validate(path)
            if manifest["phase"] != index or manifest["batch_root"] != str(root):
                raise ValueError("final_phase_order_changed")
            frozen = validate_frozen(path, manifest)
            phases.append({"directory": str(path), "run_identity": manifest["run_identity"],
                           "freeze_identity": frozen["freeze_identity"]})
            records.append(frozen["predictions"])
            for name in ("manifest.json", "preflight.json", "execution.started.json",
                         "execution.json", "frozen.json"):
                files[str((path / name).relative_to(root))] = file_digest(path / name)
        if remaining_items(first["items"], records):
            raise ValueError("campaign_has_untouched_questions")
        completed = [p for rows in records for p in rows if p["status"] != "not_dispatched"]
        if len(completed) != 240 or any(p["status"] == "outcome_unknown" for p in completed):
            raise ValueError("campaign_not_terminal")
        completed.sort(key=lambda p: p["index"])
        value = {
            "schema": "mmsi-location-batch-campaign-freeze/1",
            "run_identity": first["run_identity"], "predictions": completed,
            "phases": phases, "files": files, "total_unique_attempts": 240,
            "reference_answer_visible": False, "automatic_retry": False, "frozen_at": time.time(),
        }
        value["freeze_identity"] = digest(value)
        write(root / "campaign-frozen.json", value, exclusive=True)
        return value


def validate_campaign(root):
    root = Path(root).resolve()
    manifest = validate(root)
    value = read(root / "campaign-frozen.json")
    if (digest({k: v for k, v in value.items() if k != "freeze_identity"})
            != value["freeze_identity"] or value["run_identity"] != manifest["run_identity"]
            or value["total_unique_attempts"] != 240 or value["reference_answer_visible"] is not False
            or any(file_digest(root / name) != h for name, h in value["files"].items())):
        raise ValueError("campaign_freeze_changed")
    records = []
    for phase in value["phases"]:
        path = Path(phase["directory"])
        previous = validate(path)
        frozen = validate_frozen(path, previous)
        if (previous["run_identity"] != phase["run_identity"]
                or frozen["freeze_identity"] != phase["freeze_identity"]):
            raise ValueError("campaign_phase_identity_changed")
        records.append(frozen["predictions"])
    if remaining_items(manifest["items"], records):
        raise ValueError("campaign_incomplete")
    actual = sorted([p for rows in records for p in rows if p["status"] != "not_dispatched"],
                    key=lambda p: p["index"])
    if value["predictions"] != actual:
        raise ValueError("campaign_answers_changed")
    return manifest, value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "validate", "execute"))
    parser.add_argument("--directory", default=str(gpu.DESTINATION))
    args = parser.parse_args()
    try:
        if args.phase == "prepare":
            prepare(args.directory)
        elif args.phase == "validate":
            value = validate(args.directory, verify_requests=True)
            print(json.dumps({"status": "validated", "questions": value["questions"]}), flush=True)
        else:
            value = execute(args.directory, json.loads(sys.stdin.buffer.read().decode()))
            freeze(args.directory)
            print(json.dumps(value), flush=True)
    except Exception:
        print('{"status":"failed","reason_code":"location_batch_operation_failed"}', flush=True)
        raise SystemExit(2)
