"""Independent 30-pair official RGB versus compact task-bound camera context."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from types import FunctionType

from ..traces import digest, file_digest, read, write
from . import vggt_pose_trial as base
from .depth_comparison import runner as paired
from .prototype.direct import operator_lock
from .prototype.inputs import public_problem
from .vggt_task_prompt import TASKS
from .vggt_task_prompt_v2 import PROMPT_VERSION, pose_note, task_binding
from .vggt_pose30_control import DESTINATION, SOURCE, SOURCE_SHA

SCHEMA = "mmsi-pose30-paired/1"
ITEM_FIELDS = ("index", "problem", "images", "api_id", "arm_order")


def code_identity():
    identity = base.code_identity()
    for name in (
        "vggt_task_prompt.py", "vggt_task_prompt_v2.py",
        "vggt_pose30_control.py", "vggt_pose30_worker.py", "vggt_pose30_gpu.py",
        "vggt_pose30_trial.py", "vggt_pose30_host.py", "vggt_pose30_selfcheck.py",
        "depth_comparison/runner.py", "prototype/evaluation.py",
    ):
        path = base.PACKAGE / name
        identity[str(path.relative_to(base.ROOT))] = file_digest(path)
    return identity


def selection(directory, *, verify_source=True):
    value = read(Path(directory) / "selection.json")
    items, apis = value["items"], value["api_ids"]
    if (value["schema"] != "mmsi-vggt30-public-selection/1"
            or digest({k: v for k, v in value.items() if k != "selection_identity"})
            != value["selection_identity"] or value["questions"] != 30
            or value["per_type"] != 5 or value["max_requests"] != 60 or value["seed"] != 137
            or len(items) != 30 or not 1 <= len(apis) <= 6 or len(set(apis)) != len(apis)
            or value["reference_answer_visible"] is not False
            or value["selection_reads_labels"] is not False):
        raise ValueError("public_selection_changed")
    if verify_source and (value["source_manifest_sha256"] != SOURCE_SHA
                          or file_digest(SOURCE) != SOURCE_SHA):
        raise ValueError("public_source_changed")
    ids = [item["problem"]["sample_id"] for item in items]
    tasks = Counter(item["problem"]["task_type"] for item in items)
    if (len(set(ids)) != 30 or set(tasks) != set(TASKS) or set(tasks.values()) != {5}
            or set(ids) & set(value["excluded_sample_ids"])
            or Counter(item["arm_order"][0] for item in items) != {"direct": 15, "pose": 15}):
        raise ValueError("six_strata_or_pair_order_changed")
    for index, item in enumerate(items):
        rank, task = divmod(index, 6)
        if (public_problem(item["problem"]) != item["problem"] or item["index"] != index
                or len(item["images"]) != 2 or len(item["problem"]["image_paths"]) != 2
                or item["api_id"] != apis[(rank + task) % len(apis)]
                or item["arm_order"] != (
                    ["direct", "pose"] if (rank + task) % 2 == 0 else ["pose", "direct"])):
            raise ValueError("selection_binding_changed")
    return value


def request(directory, item, arm):
    public_problem(item["problem"])
    return base.request(item, arm)


def gpu_evidence(directory, selected):
    directory = Path(directory)
    config = read(directory / "gpu/gpu-input.json")
    complete = read(directory / "gpu/gpu-complete.json")
    if (config["schema"] != "mmsi-pose30-gpu-input/1"
            or config["selection_identity"] != selected["selection_identity"]
            or config["weight_sha256"] != base.WEIGHT_SHA
            or complete["status"] != "complete" or complete["joint_forwards"] != 30
            or complete["config_sha256"] != file_digest(directory / "gpu/gpu-input.json")
            or any(config[k] is not False for k in (
                "reference_answer_visible", "depth_enabled", "point_enabled", "track_enabled"))
            or complete["reference_answer_visible"] is not False
            or complete["depth_enabled"] is not False
            or config["model_requests"] != 0 or complete["model_requests"] != 0):
        raise ValueError("gpu_receipt_changed")
    if len(config["items"]) != 30 or set(complete["files"]) != {
            "poses/" + item["problem"]["sample_id"] + ".json" for item in selected["items"]}:
        raise ValueError("gpu_coverage_changed")
    for name, expected in config["worker_identity"].items():
        if file_digest(directory / "gpu/code" / name) != expected:
            raise ValueError("worker_identity_changed")
    poses = []
    for item, gpu_item in zip(selected["items"], config["items"]):
        problem = item["problem"]
        if (gpu_item["sample_id"] != problem["sample_id"]
                or gpu_item["input_sha256"] != problem["input_sha256"]):
            raise ValueError("gpu_input_binding_changed")
        for index, (receipt, gpu_receipt) in enumerate(zip(item["images"], gpu_item["images"])):
            suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[
                receipt["mime_type"]]
            if ({k: v for k, v in gpu_receipt.items() if k != "path"} != receipt
                    or gpu_receipt["path"] != f"images/{problem['sample_id']}-v{index+1}{suffix}"
                    or file_digest(directory / "gpu" / gpu_receipt["path"]) != receipt["sha256"]):
                raise ValueError("gpu_rgb_binding_changed")
        if len(gpu_item["images"]) != 2:
            raise ValueError("gpu_pair_count_changed")
        path = "poses/" + problem["sample_id"] + ".json"
        pose = read(directory / "gpu" / path)
        if (file_digest(directory / "gpu" / path) != complete["files"][path]
                or pose["schema"] != "mmsi-joint-vggt-camera/1"
                or pose["sample_id"] != problem["sample_id"]
                or pose["input_sha256"] != problem["input_sha256"]
                or pose["original_image_sha256"] != [r["sha256"] for r in item["images"]]
                or pose["weight_sha256"] != base.WEIGHT_SHA or pose["joint_forward"] is not True
                or pose["environment"] != config["environment"]
                or pose["model_requests"] != 0
                or pose["camera_evidence"]["input_image_count"] != 2
                or any(pose[k] is not False for k in (
                    "reference_answer_visible", "depth_head_executed",
                    "point_head_executed", "track_head_executed"))):
            raise ValueError("pose_provenance_changed")
        poses.append(pose)
    return config, poses


def prepare(directory):
    directory = Path(directory)
    selected = selection(directory)
    config, poses = gpu_evidence(directory, selected)
    items = []
    for source, pose in zip(selected["items"], poses):
        item = {k: source[k] for k in ITEM_FIELDS}
        item.update(
            pose_path="gpu/poses/" + item["problem"]["sample_id"] + ".json",
            task_binding=task_binding(item["problem"], 2),
            pose_text=pose_note(item["problem"], pose["camera_evidence"]),
        )
        item["pose_sha256"] = file_digest(directory / item["pose_path"])
        item["payload_sha256"] = {arm: digest(request(directory, item, arm))
                                  for arm in ("direct", "pose")}
        items.append(item)
    manifest = {
        "schema": SCHEMA, "selection": selected, "items": items,
        "questions": 30, "max_requests": 60, "api_ids": selected["api_ids"],
        "max_inflight": len(selected["api_ids"]),
        "inference_by_api": {api: base.settings(api) for api in selected["api_ids"]},
        "code_identity": code_identity(), "environment": base.environment(),
        "selection_sha256": file_digest(directory / "selection.json"),
        "gpu_input_sha256": file_digest(directory / "gpu/gpu-input.json"),
        "gpu_complete_sha256": file_digest(directory / "gpu/gpu-complete.json"),
        "vggt_source_identity": config["vggt_source_identity"],
        "gpu_environment": config["environment"], "weight_sha256": base.WEIGHT_SHA,
        "prompt_version": base.PROMPT_VERSION, "pose_prompt_version": PROMPT_VERSION,
        "parser_version": base.PARSER_VERSION, "reference_answer_visible": False,
        "automatic_retry": False, "tools_enabled": False, "depth_images_sent": False,
        "sample_role": selected["sample_role"], "prepared_at": time.time(),
    }
    manifest["run_identity"] = digest(manifest)
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "preflight.json", {
        "run_identity": manifest["run_identity"], "model_requests": 0,
        "reference_answers_read": False, "questions": 30, "max_requests": 60,
        "original_rgb_and_pose_hashes_verified": True, "depth_images_sent": False,
    }, exclusive=True)
    validate(directory, verify_requests=True)
    preview(directory, manifest)
    return manifest


def validate(directory, *, verify_requests=False):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    preflight = read(directory / "preflight.json")
    selected = selection(directory)
    config, poses = gpu_evidence(directory, selected)
    if (manifest["schema"] != SCHEMA
            or digest({k: v for k, v in manifest.items() if k != "run_identity"})
            != manifest["run_identity"] or manifest["code_identity"] != code_identity()
            or manifest["environment"] != base.environment() or manifest["selection"] != selected
            or manifest["questions"] != 30 or manifest["max_requests"] != 60
            or len(manifest["items"]) != 30 or manifest["api_ids"] != selected["api_ids"]
            or manifest["max_inflight"] != len(selected["api_ids"])
            or manifest["inference_by_api"] != {
                api: base.settings(api) for api in selected["api_ids"]}
            or manifest["prompt_version"] != base.PROMPT_VERSION
            or manifest["pose_prompt_version"] != PROMPT_VERSION
            or manifest["parser_version"] != base.PARSER_VERSION
            or manifest["weight_sha256"] != base.WEIGHT_SHA
            or manifest["vggt_source_identity"] != config["vggt_source_identity"]
            or manifest["gpu_environment"] != config["environment"]
            or preflight["run_identity"] != manifest["run_identity"]
            or preflight["model_requests"] != 0 or preflight["reference_answers_read"] is not False
            or any(manifest[k] is not False for k in (
                "reference_answer_visible", "automatic_retry", "tools_enabled", "depth_images_sent"))):
        raise ValueError("run_configuration_changed")
    for name, key in (
        ("selection.json", "selection_sha256"), ("gpu/gpu-input.json", "gpu_input_sha256"),
        ("gpu/gpu-complete.json", "gpu_complete_sha256"),
    ):
        if file_digest(directory / name) != manifest[key]:
            raise ValueError("prepared_file_changed")
    for item, source, pose in zip(manifest["items"], selected["items"], poses):
        if (any(item[k] != source[k] for k in ITEM_FIELDS)
                or item["pose_path"] != "gpu/poses/" + item["problem"]["sample_id"] + ".json"
                or item["pose_sha256"] != file_digest(directory / item["pose_path"])
                or item["task_binding"] != task_binding(item["problem"], 2)
                or item["pose_text"] != pose_note(item["problem"], pose["camera_evidence"])):
            raise ValueError("prepared_pair_changed")
        if verify_requests:
            for arm in item["arm_order"]:
                if digest(request(directory, item, arm)) != item["payload_sha256"][arm]:
                    raise ValueError("prepared_request_changed")
    return manifest


def preview(directory, manifest):
    rows = [{
        "sample_id": item["problem"]["sample_id"], "task_type": item["problem"]["task_type"],
        "question": item["problem"]["question"], "task_binding": item["task_binding"],
        "pose_text": item["pose_text"], "api_id": item["api_id"], "arm_order": item["arm_order"],
    } for item in manifest["items"]]
    write(Path(directory) / "prompt-preview.json", {
        "run_identity": manifest["run_identity"], "reference_answer_visible": False,
        "model_requests": 0, "rows": rows,
    }, exclusive=True)


# The shared loop already supports variable budgets and arm names. Keep its
# module globals intact; bind only this new experiment's validation and payload.
_namespace = {
    **vars(paired), "validate": validate, "request": request, "inference": base.settings,
}
execute = FunctionType(paired.execute.__code__, _namespace, "execute")
execute.__kwdefaults__ = {"client_factory": base.BoundedClient}
paths = paired.paths


def collect(directory, manifest):
    # Use the generic trace audit with a private in-memory file-receipt adapter.
    # Persistent manifests and records retain their camera-pose names.
    adapter = {**manifest, "items": [
        {**item, "depth_images": [{"path": item["pose_path"], "sha256": item["pose_sha256"]}]}
        for item in manifest["items"]]}
    namespace = {**vars(paired), "inference": base.settings}
    audit = FunctionType(paired.collect.__code__, namespace, "collect")
    records, files = audit(directory, adapter)
    for name in ("selection.json", "gpu/gpu-input.json", "gpu/gpu-complete.json",
                 "gpu/gpu-started.json", "gpu/launch.json"):
        files[name] = file_digest(Path(directory) / name)
    execution = read(Path(directory) / "execution.json")
    if any(execution[k] is not False for k in ("reference_answer_visible", "automatic_retry")):
        raise ValueError("execution_policy_changed")
    return records, files


def freeze(directory):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory)
        records, files = collect(directory, manifest)
        if (directory / "frozen.json").exists():
            return validate_frozen(directory, manifest)
        result = {
            "schema": "mmsi-pose30-freeze/1", "run_identity": manifest["run_identity"],
            "predictions": records, "files": files, "reference_answer_visible": False,
            "automatic_retry": False, "frozen_at": time.time(),
        }
        result["freeze_identity"] = digest(result)
        write(directory / "frozen.json", result, exclusive=True)
        return result


def validate_frozen(directory, manifest):
    frozen = read(Path(directory) / "frozen.json")
    records, files = collect(directory, manifest)
    if (frozen["schema"] != "mmsi-pose30-freeze/1"
            or digest({k: v for k, v in frozen.items() if k != "freeze_identity"})
            != frozen["freeze_identity"] or frozen["run_identity"] != manifest["run_identity"]
            or frozen["predictions"] != records or frozen["files"] != files):
        raise ValueError("frozen_predictions_changed")
    return frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "validate", "execute", "freeze"))
    parser.add_argument("--directory", default=str(DESTINATION))
    args = parser.parse_args()
    if args.phase == "execute":
        keys = json.loads(sys.stdin.buffer.read())
        result = execute(args.directory, keys)
    elif args.phase == "validate":
        value = validate(args.directory, verify_requests=True)
        result = {"status": "valid", "run_identity": value["run_identity"], "questions": 30}
    else:
        value = globals()[args.phase](args.directory)
        result = {k: value[k] for k in ("run_identity",)}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print('{"status":"failed","reason_code":"pose30_trial_failed"}', flush=True)
        raise SystemExit(2)
