"""Independent task-bound pose trial, reusing immutable joint VGGT predictions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace

from ..traces import digest, file_digest, read, write
from . import vggt_pose_trial as base
from .prototype.inputs import public_problem
from .vggt_task_prompt import PROMPT_VERSION, pose_note, task_binding

SCHEMA = "mmsi-task-bound-camera-pair/1"
DESTINATION = base.ROOT.parent / "MMSI位姿题目适配-20261108"
ITEM_FIELDS = ("index", "source_index", "problem", "images", "api_id", "arm_order")


def code_identity():
    result = base.code_identity()
    for name in ("vggt_task_prompt.py", "vggt_task_trial.py"):
        path = base.PACKAGE / name
        result[str(path.relative_to(base.ROOT))] = file_digest(path)
    return result


def request(item, arm):
    # Direct never accesses pose bytes, task bindings or adapted prompt text.
    public_problem(item["problem"])
    return base.request(item, arm)


def parent_evidence(directory):
    directory = Path(directory).resolve()
    manifest = base.validate(directory, requests=True)
    frozen = base.validate_frozen(directory, manifest)
    return manifest, {
        "directory": str(directory), "run_identity": manifest["run_identity"],
        "freeze_identity": frozen["freeze_identity"],
        "manifest_sha256": file_digest(directory / "manifest.json"),
        "frozen_sha256": file_digest(directory / "frozen.json"),
    }


def validate(directory, *, requests=False):
    directory = Path(directory)
    manifest, preflight = read(directory / "manifest.json"), read(directory / "preflight.json")
    if (manifest["schema"] != SCHEMA
            or digest({k: v for k, v in manifest.items() if k != "run_identity"}) != manifest["run_identity"]
            or manifest["code_identity"] != code_identity()
            or manifest["environment"] != base.environment()
            or manifest["pose_prompt_version"] != PROMPT_VERSION
            or manifest["prompt_version"] != base.PROMPT_VERSION
            or manifest["parser_version"] != base.PARSER_VERSION
            or manifest["api_ids"] != list(base.APIS) or manifest["questions"] != 6
            or len(manifest["items"]) != 6 or manifest["max_requests"] != 12
            or manifest["max_inflight"] != 6
            or manifest["inference_by_api"] != {api: base.settings(api) for api in base.APIS}
            or manifest["weight_sha256"] != base.WEIGHT_SHA
            or any(manifest[k] is not False for k in (
                "reference_answer_visible", "automatic_retry", "tools_enabled", "depth_images_sent"))
            or preflight["run_identity"] != manifest["run_identity"]
            or preflight["model_requests"] != 0 or preflight["reference_answers_read"] is not False):
        raise ValueError("task_bound_configuration_changed")
    for name, expected in (
        ("selection.json", manifest["selection_sha256"]),
        ("gpu/gpu-input.json", manifest["gpu_input_sha256"]),
        ("gpu/gpu-complete.json", manifest["gpu_complete_sha256"]),
    ):
        if file_digest(directory / name) != expected:
            raise ValueError("prepared_evidence_changed")
    selected = read(directory / "selection.json")
    if selected["parent_evidence"] != manifest["parent_evidence"]:
        raise ValueError("parent_binding_changed")
    complete = read(directory / "gpu/gpu-complete.json")
    config = read(directory / "gpu/gpu-input.json")
    if (complete["status"] != "complete" or complete["joint_forwards"] != 6
            or complete["config_sha256"] != manifest["gpu_input_sha256"]
            or complete["reference_answer_visible"] is not False or complete["depth_enabled"] is not False
            or config["weight_sha256"] != base.WEIGHT_SHA
            or config["vggt_source_identity"] != manifest["vggt_source_identity"]):
        raise ValueError("joint_pose_receipt_changed")
    for index, item in enumerate(manifest["items"]):
        problem = public_problem(item["problem"])
        original = selected["items"][index]
        if (any(item[k] != original[k] for k in ITEM_FIELDS)
                or problem["sample_id"] != base.SAMPLES[index] or item["api_id"] != base.APIS[index]
                or item["index"] != index or item["arm_order"] != (
                    ["direct", "pose"] if index % 2 == 0 else ["pose", "direct"])
                or item["pose_path"] != "gpu/poses/" + problem["sample_id"] + ".json"
                or item["task_binding"] != task_binding(problem, len(item["images"]))):
            raise ValueError("task_bound_pair_changed")
        path = directory / item["pose_path"]
        pose = read(path)
        if (file_digest(path) != item["pose_sha256"]
                or item["pose_sha256"] != complete["files"][item["pose_path"][4:]]
                or pose["sample_id"] != problem["sample_id"]
                or pose["input_sha256"] != problem["input_sha256"]
                or pose["original_image_sha256"] != [r["sha256"] for r in item["images"]]
                or pose["weight_sha256"] != base.WEIGHT_SHA or pose["joint_forward"] is not True
                or any(pose[k] is not False for k in (
                    "depth_head_executed", "point_head_executed", "track_head_executed",
                    "reference_answer_visible"))
                or pose["camera_evidence"]["input_image_count"] != len(item["images"])
                or item["pose_text"] != pose_note(problem, pose["camera_evidence"])
                or item["payload_sha256"]["direct"] != original["parent_direct_payload_sha256"]):
            raise ValueError("task_bound_pose_changed")
        if requests:
            for arm in item["arm_order"]:
                if digest(request(item, arm)) != item["payload_sha256"][arm]:
                    raise ValueError("task_bound_request_changed")
    return manifest


def prepare(directory, parent=base.DESTINATION):
    directory, parent = Path(directory).resolve(), Path(parent).resolve()
    original, provenance = parent_evidence(parent)
    if not directory.parent.is_dir() or directory.exists():
        raise ValueError("requires_new_directory")
    directory.mkdir()
    (directory / "gpu/poses").mkdir(parents=True)
    for name in ("gpu/gpu-input.json", "gpu/gpu-complete.json"):
        shutil.copyfile(parent / name, directory / name)
    items, selection = [], []
    for source in original["items"]:
        item = {k: source[k] for k in ITEM_FIELDS}
        item["problem"] = public_problem(item["problem"])
        selected = {**item, "parent_direct_payload_sha256": source["payload_sha256"]["direct"]}
        selection.append(selected)
        item["pose_path"], item["pose_sha256"] = source["pose_path"], source["pose_sha256"]
        shutil.copyfile(parent / item["pose_path"], directory / item["pose_path"])
        evidence = read(directory / item["pose_path"])["camera_evidence"]
        item["task_binding"] = task_binding(item["problem"], len(item["images"]))
        item["pose_text"] = pose_note(item["problem"], evidence)
        item["payload_sha256"] = {arm: digest(request(item, arm)) for arm in ("direct", "pose")}
        if item["payload_sha256"]["direct"] != selected["parent_direct_payload_sha256"]:
            raise ValueError("parent_direct_request_changed")
        items.append(item)
    write(directory / "selection.json", {
        "schema": "mmsi-task-bound-selection/1", "items": selection,
        "parent_evidence": provenance, "reference_answer_visible": False,
        "sample_role": original["sample_role"],
        "selection_reason": "reuse_same_six_development_pairs_with_task_bound_pose_prompt",
    }, exclusive=True)
    manifest = {
        "schema": SCHEMA, "items": items, "max_requests": 12, "max_inflight": 6,
        "api_ids": list(base.APIS), "questions": 6,
        "inference_by_api": {api: base.settings(api) for api in base.APIS},
        "code_identity": code_identity(), "environment": base.environment(),
        "parent_evidence": provenance,
        "selection_sha256": file_digest(directory / "selection.json"),
        "gpu_complete_sha256": file_digest(directory / "gpu/gpu-complete.json"),
        "gpu_input_sha256": file_digest(directory / "gpu/gpu-input.json"),
        "vggt_source_identity": original["vggt_source_identity"], "weight_sha256": base.WEIGHT_SHA,
        "prompt_version": base.PROMPT_VERSION, "parser_version": base.PARSER_VERSION,
        "pose_prompt_version": PROMPT_VERSION,
        "reference_answer_visible": False, "automatic_retry": False, "tools_enabled": False,
        "depth_images_sent": False, "sample_role": original["sample_role"],
        "user_instruction": "按这个方案适配一下吧",
        "prepared_at": time.time(),
    }
    manifest["run_identity"] = digest(manifest)
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "preflight.json", {
        "run_identity": manifest["run_identity"], "model_requests": 0,
        "reference_answers_read": False, "questions": 6, "max_requests": 12,
        "original_rgb_and_pose_hashes_verified": True, "parent_direct_payloads_equal": True,
        "gpu_forwards_this_run": 0, "depth_images_sent": False,
    }, exclusive=True)
    validate(directory, requests=True)
    preview(directory)
    return manifest


def make_engine():
    """Bind the existing dispatch/freeze loop without mutating its module globals."""
    namespace = {**vars(base), "validate": validate, "request": request}
    names = ("execute", "collect", "freeze", "validate_frozen", "score")
    for name in names:
        original = getattr(base, name)
        bound = FunctionType(original.__code__, namespace, name, original.__defaults__,
                             original.__closure__)
        bound.__kwdefaults__ = original.__kwdefaults__
        namespace[name] = bound
    return SimpleNamespace(**{name: namespace[name] for name in names})


_engine = make_engine()
execute, freeze, validate_frozen, score = (
    _engine.execute, _engine.freeze, _engine.validate_frozen, _engine.score)


def preview(directory):
    """Export exactly the added public text, without duplicating image data URLs."""
    manifest = validate(directory, requests=True)
    lines = [
        "# MMSI 六类题目的 VGGT 位姿提示适配",
        "",
        f"运行身份：`{manifest['run_identity']}`",
        f"提示版本：`{PROMPT_VERSION}`。状态：零请求准备，尚未派发模型请求。",
        "原图字节、顺序、原题和选项均沿用父实验；Direct payload 与父实验逐字节对应。",
        "两臂保持原单字母输出协议。本预览展示实际待发送文字，不包含参考答案。",
    ]
    for item in manifest["items"]:
        lines.extend((
            "", "## " + item["problem"]["sample_id"] + " · " + item["problem"]["task_type"],
            "", "原始题目：", "", item["problem"]["question"],
            "", "位姿臂新增文字：", "", "```text", item["pose_text"], "```",
        ))
    (Path(directory) / "提示预览.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def controlled_execute(directory):
    from .depth_service_probe import all_keys
    keys = all_keys()
    result = subprocess.run(
        [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.vggt_task_trial",
         "execute", "--directory", str(directory)],
        input=json.dumps(keys).encode(), env=base.clean_environment(), cwd=base.ROOT)
    if result.returncode:
        raise RuntimeError("online_worker_failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=(
        "prepare", "validate", "preview", "controlled-execute", "execute", "freeze", "score"))
    parser.add_argument("--directory", default=str(DESTINATION))
    parser.add_argument("--parent", default=str(base.DESTINATION))
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare(args.directory, args.parent)
        print(json.dumps({"status": "prepared", "model_requests": 0,
                          "run_identity": result["run_identity"]}))
    elif args.phase in ("validate", "preview"):
        result = preview(args.directory) if args.phase == "preview" else validate(args.directory, requests=True)
        print(json.dumps({"status": "validated", "run_identity": result["run_identity"]}))
    elif args.phase == "controlled-execute":
        controlled_execute(args.directory)
    elif args.phase == "execute":
        print(json.dumps(execute(args.directory, json.loads(sys.stdin.read()))))
    elif args.phase == "freeze":
        result = freeze(args.directory)
        print(json.dumps({"status": "frozen", "freeze_identity": result["freeze_identity"],
                          "statuses": dict(Counter(p["status"] for p in result["predictions"]))}))
    elif args.phase == "score":
        print(json.dumps(score(args.directory)["overall"]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print('{"status":"failed","reason_code":"task_bound_pose_trial_failed"}', flush=True)
        raise SystemExit(2)
