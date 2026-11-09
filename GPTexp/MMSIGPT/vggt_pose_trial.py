"""Independent six-pair RGB versus RGB plus joint camera-pose diagnostic."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time

from xcodeapi.config import Profile

from ..traces import credential_free, digest, file_digest, read, write
from .depth_comparison.pool import BASE_URL, MODEL, Client, inference
from .prototype.client import ServiceError
from .prototype.direct import operator_lock
from .prototype.inputs import load_images, public_problem
from .prototype.protocol import messages, payload, parse_answer, PROMPT_VERSION, PARSER_VERSION
from .vggt_pose import pose_note

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
DESTINATION = ROOT.parent / "MMSI联合相机位姿-20261108"
SOURCE = ROOT.parent / "MMSI深度图对比-20261107/run-recovered"
OLD_SCORE = SOURCE.parent / "run-all-api/score.json"
OLD_SCORE_ID = "d25cc28e3c6174b08014082edfc52368f49ca58ce8b4366ed93501edb83e4f6f"
SAMPLES = ("MMSI_926", "MMSI_275", "MMSI_346", "MMSI_921", "MMSI_874", "MMSI_479")
APIS = tuple(f"pool-{i:02d}" for i in range(1, 7))
REMOTE = "/home/zhouky/wangjh/harness3d/mmsi-joint-pose-20261108"
REMOTE_SOURCE = "/home/zhouky/wangjh/harness3d/precompute-20261022-r04/source_frozen/third_party/vggt"
WEIGHT = "/home/zhouky/wangjh/skill3d-lab/pilot-20261021-01/inputs/VGGT-1B/model.safetensors"
WEIGHT_SHA = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
REMOTE_PYTHON = "/home/zhouky/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"
SSH_OPTIONS = ["-F", "/Users/bytedance/wangjh/.trae/skills/harness3d-cluster-ops/assets/ssh_config",
               "-o", "ControlPath=/Users/bytedance/.cache/harness3d-ssh/mmsi-pose-20261108-3090",
               "-o", "BatchMode=yes"]
SSH = ["ssh", *SSH_OPTIONS, "-T", "h3d-3090"]


def clean_environment():
    # Only startup paths and local numerical settings are needed in online workers.
    allowed = {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "PYTHONPATH"}
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return env


def settings(api):
    return {**inference(api), "timeout_seconds": 180}


class BoundedClient(Client):
    def __init__(self, api, key):
        super().__init__(api, key)
        self.inference = settings(api)
        self.profile = Profile("responses", BASE_URL, MODEL, "XCODE_GPT_API_KEY",
                               timeout_seconds=180)


def code_identity():
    files = [PACKAGE / name for name in (
        "vggt_pose.py", "vggt_pose_worker.py", "vggt_pose_trial.py", "__init__.py",
        "depth_comparison/pool.py", "prototype/client.py", "prototype/direct.py",
        "prototype/inputs.py", "prototype/protocol.py")]
    files += [ROOT / "GPTexp" / name for name in ("traces.py", "response_policy.py")]
    files += list((ROOT / "GPTexp/vendor/xcodeapi").glob("*.py"))
    return {str(p.relative_to(ROOT)): file_digest(p) for p in sorted(files)}


def environment():
    import numpy
    import PIL
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "pillow": PIL.__version__}


def selection():
    manifest = read(SOURCE / "manifest.json")
    if (digest({k: v for k, v in manifest.items() if k != "run_identity"})
            != manifest["run_identity"] or manifest["reference_answer_visible"] is not False):
        raise ValueError("public_source_changed")
    originals = {i["problem"]["sample_id"]: i for i in manifest["items"]}
    selected = []
    for index, (sample, api) in enumerate(zip(SAMPLES, APIS)):
        original = originals[sample]
        problem = public_problem(original["problem"])
        load_images(problem, original["images"])
        if len(problem["image_paths"]) != 2:
            raise ValueError("requires_two_original_images")
        selected.append({
            "index": index, "source_index": original["index"], "problem": problem,
            "images": original["images"], "api_id": api,
            "arm_order": ["direct", "pose"] if index % 2 == 0 else ["pose", "direct"],
        })
    if len({i["problem"]["task_type"] for i in selected}) != 6:
        raise ValueError("requires_six_types")
    return manifest, selected


def ssh_json(code):
    result = subprocess.run(SSH + [shlex.quote(REMOTE_PYTHON) + " -B -c " + shlex.quote(code)],
                            capture_output=True, timeout=60)
    if result.returncode:
        raise ValueError("remote_probe_failed")
    return json.loads(result.stdout)


def prepare_gpu(directory):
    directory = Path(directory)
    original, selected = selection()
    if not directory.parent.is_dir() or directory.exists():
        raise ValueError("requires_new_directory")
    probe = ssh_json(
        "import hashlib,json,pathlib,shutil; "
        f"s=pathlib.Path({REMOTE_SOURCE!r}); p=pathlib.Path({REMOTE!r}); "
        "print(json.dumps({'source':{str(f.relative_to(s)):hashlib.sha256(f.read_bytes()).hexdigest() "
        "for f in sorted((s/'vggt').rglob('*.py'))},'exists':p.exists(),"
        "'free_bytes':shutil.disk_usage(p.parent).free}))")
    if probe["exists"] or not probe["source"] or probe["free_bytes"] < 15 * 1024**3:
        raise ValueError("remote_task_or_disk_unavailable")
    directory.mkdir()
    gpu = directory / "gpu"
    for name in ("images", "poses", "code"):
        (gpu / name).mkdir(parents=True, exist_ok=False)
    for name in ("vggt_pose.py", "vggt_pose_worker.py"):
        shutil.copyfile(PACKAGE / name, gpu / "code" / name)
    items = []
    for item in selected:
        sample = item["problem"]["sample_id"]
        receipts = []
        for index, (receipt, raw) in enumerate(load_images(item["problem"], item["images"])):
            suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[receipt["mime_type"]]
            path = "images/" + sample + f"-v{index + 1}" + suffix
            (gpu / path).write_bytes(raw)
            receipts.append({**receipt, "path": path})
        items.append({"sample_id": sample, "input_sha256": item["problem"]["input_sha256"],
                      "images": receipts})
    config = {
        "schema": "mmsi-joint-camera-gpu-input/1", "vggt_source": REMOTE_SOURCE,
        "vggt_source_identity": probe["source"], "weight_path": WEIGHT, "weight_sha256": WEIGHT_SHA,
        "worker_identity": {name: file_digest(gpu / "code" / name)
                            for name in ("vggt_pose.py", "vggt_pose_worker.py")},
        "items": items, "reference_answer_visible": False, "depth_enabled": False,
        "point_enabled": False, "track_enabled": False, "model_requests": 0,
    }
    write(gpu / "gpu-input.json", config, exclusive=True)
    write(directory / "selection.json", {
        "source_manifest_sha256": file_digest(SOURCE / "manifest.json"),
        "source_run_identity": original["run_identity"], "items": selected,
        "reference_answer_visible": False, "sample_role": "previously_seen_development_diagnostic",
        "selection_reason": "one_per_type_including_prior_926_and_874_regressions",
        "max_new_model_requests": 12, "gpu_input_sha256": file_digest(gpu / "gpu-input.json"),
    }, exclusive=True)
    # Parent was independently verified by the probe before creating this one task.
    result = subprocess.run(SSH + ["mkdir -m 700 " + shlex.quote(REMOTE)], capture_output=True)
    if result.returncode:
        raise ValueError("remote_directory_creation_failed")
    for target in ("images", "poses", "code"):
        result = subprocess.run(SSH + ["mkdir " + shlex.quote(REMOTE + "/" + target)],
                                capture_output=True)
        if result.returncode:
            raise ValueError("remote_subdirectory_creation_failed")
    files = [gpu / "gpu-input.json", *sorted((gpu / "images").iterdir()),
             *sorted((gpu / "code").iterdir())]
    for file in files:
        relative = str(file.relative_to(gpu))
        result = subprocess.run(["scp", *SSH_OPTIONS, str(file), "h3d-3090:" + REMOTE + "/" + relative],
                                capture_output=True, timeout=60)
        if result.returncode:
            raise ValueError("original_input_transfer_failed")
    print(json.dumps({"status": "gpu_prepared", "questions": 6, "model_requests": 0}), flush=True)


def request(item, arm):
    content = messages(item["problem"], load_images(item["problem"], item["images"]))
    if arm == "pose":
        # Keep original images and the entire official question/options/instruction byte-identical.
        content[0]["content"].insert(-1, {"type": "text", "text": item["pose_text"]})
    elif arm != "direct":
        raise ValueError("invalid_arm")
    return payload(content, settings(item["api_id"]))


def prepare_online(directory):
    directory = Path(directory)
    selected = read(directory / "selection.json")
    original, items = selection()
    if (selected["items"] != items or selected["source_run_identity"] != original["run_identity"]
            or selected["source_manifest_sha256"] != file_digest(SOURCE / "manifest.json")):
        raise ValueError("selection_changed")
    config = read(directory / "gpu/gpu-input.json")
    complete = read(directory / "gpu/gpu-complete.json")
    if (complete["status"] != "complete" or complete["joint_forwards"] != 6
            or complete["config_sha256"] != selected["gpu_input_sha256"]
            or complete["config_sha256"] != file_digest(directory / "gpu/gpu-input.json")
            or complete["reference_answer_visible"] is not False or complete["depth_enabled"] is not False):
        raise ValueError("gpu_run_incomplete")
    for item in items:
        sample = item["problem"]["sample_id"]
        path = "poses/" + sample + ".json"
        pose = read(directory / "gpu" / path)
        if (file_digest(directory / "gpu" / path) != complete["files"][path]
                or pose["sample_id"] != sample or pose["input_sha256"] != item["problem"]["input_sha256"]
                or pose["original_image_sha256"] != [r["sha256"] for r in item["images"]]
                or pose["weight_sha256"] != WEIGHT_SHA or pose["joint_forward"] is not True
                or any(pose[k] is not False for k in ("depth_head_executed", "point_head_executed",
                                                     "track_head_executed", "reference_answer_visible"))):
            raise ValueError("pose_binding_changed")
        item["pose_path"] = "gpu/" + path
        item["pose_sha256"] = file_digest(directory / item["pose_path"])
        item["pose_text"] = pose_note(pose["camera_evidence"])
        item["payload_sha256"] = {arm: digest(request(item, arm)) for arm in ("direct", "pose")}
    manifest = {
        "schema": "mmsi-joint-camera-pair/1", "items": items, "max_requests": 12,
        "max_inflight": 6, "api_ids": list(APIS), "questions": 6,
        "inference_by_api": {api: settings(api) for api in APIS},
        "code_identity": code_identity(), "environment": environment(),
        "source_manifest_sha256": selected["source_manifest_sha256"],
        "selection_sha256": file_digest(directory / "selection.json"),
        "gpu_complete_sha256": file_digest(directory / "gpu/gpu-complete.json"),
        "gpu_input_sha256": file_digest(directory / "gpu/gpu-input.json"),
        "vggt_source_identity": config["vggt_source_identity"], "weight_sha256": WEIGHT_SHA,
        "prompt_version": PROMPT_VERSION, "parser_version": PARSER_VERSION,
        "reference_answer_visible": False, "automatic_retry": False, "tools_enabled": False,
        "depth_images_sent": False, "sample_role": selected["sample_role"],
        "user_instruction": "去掉深度图吧，我们使用 VGGT 给图片联合估计相机位姿然后发给模型试试行不行",
        "prepared_at": time.time(),
    }
    manifest["run_identity"] = digest(manifest)
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "preflight.json", {
        "run_identity": manifest["run_identity"], "model_requests": 0,
        "reference_answers_read": False, "original_rgb_and_pose_hashes_verified": True,
        "questions": 6, "max_requests": 12, "depth_images_sent": False,
    }, exclusive=True)
    validate(directory, requests=True)
    return manifest


def validate(directory, *, requests=False):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    preflight = read(directory / "preflight.json")
    if (manifest["schema"] != "mmsi-joint-camera-pair/1"
            or digest({k: v for k, v in manifest.items() if k != "run_identity"}) != manifest["run_identity"]
            or manifest["code_identity"] != code_identity() or manifest["environment"] != environment()
            or manifest["api_ids"] != list(APIS) or manifest["questions"] != 6
            or manifest["max_requests"] != 12 or manifest["max_inflight"] != 6
            or manifest["inference_by_api"] != {api: settings(api) for api in APIS}
            or manifest["reference_answer_visible"] is not False or manifest["automatic_retry"] is not False
            or manifest["tools_enabled"] is not False or manifest["depth_images_sent"] is not False
            or manifest["weight_sha256"] != WEIGHT_SHA
            or preflight["run_identity"] != manifest["run_identity"] or preflight["model_requests"] != 0
            or preflight["reference_answers_read"] is not False):
        raise ValueError("frozen_configuration_changed")
    for name, expected in (
        ("selection.json", manifest["selection_sha256"]),
        ("gpu/gpu-input.json", manifest["gpu_input_sha256"]),
        ("gpu/gpu-complete.json", manifest["gpu_complete_sha256"]),
    ):
        if file_digest(directory / name) != expected:
            raise ValueError("prepared_evidence_changed")
    selected = read(directory / "selection.json")
    for index, item in enumerate(manifest["items"]):
        if (any(item[k] != selected["items"][index][k] for k in selected["items"][index])
                or item["problem"]["sample_id"] != SAMPLES[index] or item["api_id"] != APIS[index]
                or item["index"] != index or item["arm_order"] != (
                    ["direct", "pose"] if index % 2 == 0 else ["pose", "direct"])):
            raise ValueError("pair_binding_changed")
        pose = read(directory / item["pose_path"])
        if (file_digest(directory / item["pose_path"]) != item["pose_sha256"]
                or item["pose_text"] != pose_note(pose["camera_evidence"])
                or pose["original_image_sha256"] != [r["sha256"] for r in item["images"]]):
            raise ValueError("pose_text_or_bytes_changed")
        if requests:
            for arm in item["arm_order"]:
                if digest(request(item, arm)) != item["payload_sha256"][arm]:
                    raise ValueError("payload_changed")
    return manifest


def paths(directory, item, arm):
    stem = Path(directory) / "traces" / f"{item['index']:04d}-{arm}"
    return Path(str(stem) + ".dispatch.json"), Path(str(stem) + ".result.json")


def execute(directory, keys, *, client_factory=BoundedClient):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory, requests=True)
        if any((directory / name).exists() for name in ("execution.started.json", "traces", "frozen.json")):
            raise ValueError("no_restart_or_automatic_retry")
        if len(keys) != 11 or len(set(keys.values())) != 11 or set(APIS) - keys.keys():
            raise ValueError("requires_all_original_distinct_credentials")
        def public(value):
            for key in keys.values():
                credential_free(value, key)
        public(manifest)
        clients = {api: client_factory(api, keys[api]) for api in APIS}
        stop, guard = threading.Event(), threading.Lock()
        state = {"dispatched": 0, "inflight": 0, "peak_inflight": 0}
        write(directory / "execution.started.json", {
            "run_identity": manifest["run_identity"], "max_requests": 12, "max_inflight": 6,
            "reference_answer_visible": False, "automatic_retry": False,
            "credential_storage": "process_memory_only", "at": time.time(),
        }, exclusive=True)
        def slot(item):
            for arm in item["arm_order"]:
                if stop.is_set():
                    return
                body = request(item, arm)
                public(body)
                if digest(body) != item["payload_sha256"][arm]:
                    stop.set()
                    raise ValueError("request_changed_before_dispatch")
                dispatch_path, result_path = paths(directory, item, arm)
                dispatch = {
                    "index": item["index"], "sample_id": item["problem"]["sample_id"],
                    "input_sha256": item["problem"]["input_sha256"], "api_id": item["api_id"],
                    "arm": arm, "payload_sha256": item["payload_sha256"][arm],
                    "requested": settings(item["api_id"]), "run_identity": manifest["run_identity"],
                    "reference_answer_visible": False, "automatic_retry": False, "at": time.time(),
                }
                with guard:
                    if stop.is_set():
                        return
                    if state["dispatched"] >= 12:
                        raise ValueError("request_budget_exceeded")
                    write(dispatch_path, dispatch, exclusive=True)
                    state["dispatched"] += 1
                    state["inflight"] += 1
                    state["peak_inflight"] = max(state["inflight"], state["peak_inflight"])
                print(json.dumps({"sample_id": dispatch["sample_id"], "arm": arm,
                                  "status": "dispatched"}), flush=True)
                try:
                    response = clients[item["api_id"]].call(body)
                    public(response)
                    if response["tool_call_count"] != 0:
                        raise ServiceError("unexpected_tool_call")
                    parsed = parse_answer(response["text"], len(item["problem"]["options"]))
                    record = {**parsed, "status": "answered" if parsed["answer"] else "invalid_answer",
                              "response": response}
                except ServiceError as error:
                    stop.set()
                    record = {"status": "service_error", "answer": None, "parse_status": "not_parsed",
                              "error_code": error.code, "http_status": error.http_status}
                except Exception:
                    stop.set()
                    record = {"status": "service_error", "answer": None, "parse_status": "not_parsed",
                              "error_code": "client_outcome_unknown", "http_status": None}
                record.update({
                    **{k: dispatch[k] for k in ("index", "sample_id", "input_sha256", "api_id",
                                              "arm", "run_identity", "reference_answer_visible",
                                              "automatic_retry")},
                    "dispatch_sha256": file_digest(dispatch_path), "at": time.time(),
                })
                public(record)
                write(result_path, record, exclusive=True)
                with guard:
                    state["inflight"] -= 1
                print(json.dumps({k: record[k] for k in ("sample_id", "arm", "status", "answer")}), flush=True)
        status = "interrupted"
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(slot, manifest["items"]))
            status = "stopped_service_error" if stop.is_set() else "complete"
        finally:
            stop.set()
            write(directory / "execution.json", {
                **state, "status": status, "run_identity": manifest["run_identity"],
                "reference_answer_visible": False, "automatic_retry": False, "at": time.time(),
            }, exclusive=True)
        return read(directory / "execution.json")


def collect(directory, manifest):
    directory = Path(directory)
    execution = read(directory / "execution.json")
    started = read(directory / "execution.started.json")
    if (execution["run_identity"] != manifest["run_identity"] or execution["inflight"] != 0
            or execution["peak_inflight"] > 6 or started["max_requests"] != 12
            or started["run_identity"] != manifest["run_identity"]
            or any(value[k] is not False for value in (execution, started)
                   for k in ("reference_answer_visible", "automatic_retry"))):
        raise ValueError("execution_changed")
    files = {name: file_digest(directory / name) for name in (
        "manifest.json", "preflight.json", "selection.json", "gpu/gpu-input.json",
        "gpu/gpu-complete.json", "execution.started.json", "execution.json")}
    predictions, known, dispatched = [], set(), 0
    for item in manifest["items"]:
        files[item["pose_path"]] = file_digest(directory / item["pose_path"])
        sent = []
        for arm in item["arm_order"]:
            dispatch_path, result_path = paths(directory, item, arm)
            known.update((dispatch_path.name, result_path.name))
            record = {
                "index": item["index"], "sample_id": item["problem"]["sample_id"],
                "task_type": item["problem"]["task_type"], "api_id": item["api_id"], "arm": arm,
                "input_sha256": item["problem"]["input_sha256"], "status": "not_dispatched",
                "answer": None,
            }
            if result_path.exists() and not dispatch_path.exists():
                raise ValueError("result_without_dispatch")
            if dispatch_path.exists():
                dispatched += 1
                sent.append(arm)
                dispatch = read(dispatch_path)
                if (any(dispatch[k] != record[k] for k in ("index", "sample_id", "api_id",
                                                          "arm", "input_sha256"))
                        or dispatch["run_identity"] != manifest["run_identity"]
                        or dispatch["payload_sha256"] != item["payload_sha256"][arm]
                        or dispatch["requested"] != settings(item["api_id"])
                        or dispatch["reference_answer_visible"] is not False
                        or dispatch["automatic_retry"] is not False):
                    raise ValueError("dispatch_changed")
                files[str(dispatch_path.relative_to(directory))] = file_digest(dispatch_path)
                record["status"] = "outcome_unknown"
            if result_path.exists():
                result = read(result_path)
                if (any(result[k] != record[k] for k in ("index", "sample_id", "api_id", "arm",
                                                        "input_sha256"))
                        or result["run_identity"] != manifest["run_identity"]
                        or result["dispatch_sha256"] != file_digest(dispatch_path)
                        or result["reference_answer_visible"] is not False
                        or result["automatic_retry"] is not False):
                    raise ValueError("result_changed")
                if result["status"] in ("answered", "invalid_answer"):
                    parsed = parse_answer(result["response"]["text"], len(item["problem"]["options"]))
                    if (parsed["answer"] != result["answer"] or parsed["parse_status"] != result["parse_status"]
                            or result["status"] != ("answered" if parsed["answer"] else "invalid_answer")
                            or result["response"]["tool_call_count"] != 0):
                        raise ValueError("answer_parse_changed")
                elif result["status"] != "service_error" or result["answer"] is not None:
                    raise ValueError("result_status_changed")
                record.update(status=result["status"], answer=result["answer"])
                files[str(result_path.relative_to(directory))] = file_digest(result_path)
            predictions.append(record)
        if sent != item["arm_order"][:len(sent)]:
            raise ValueError("pair_order_changed")
    if dispatched != execution["dispatched"] or dispatched > 12:
        raise ValueError("budget_changed")
    if (directory / "traces").exists() and any(
            p.name not in known for p in (directory / "traces").iterdir()):
        raise ValueError("unexpected_attempt")
    if execution["status"] == "complete" and (
            dispatched != 12 or any(p["status"] not in ("answered", "invalid_answer") for p in predictions)):
        raise ValueError("missing_prediction")
    return predictions, files


def freeze(directory):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory)
        predictions, files = collect(directory, manifest)
        if (directory / "frozen.json").exists():
            return validate_frozen(directory, manifest)
        frozen = {
            "schema": "mmsi-joint-camera-freeze/1", "run_identity": manifest["run_identity"],
            "predictions": predictions, "files": files,
            "reference_answer_visible": False, "automatic_retry": False, "frozen_at": time.time(),
        }
        frozen["freeze_identity"] = digest(frozen)
        write(directory / "frozen.json", frozen, exclusive=True)
        return frozen


def validate_frozen(directory, manifest):
    frozen = read(Path(directory) / "frozen.json")
    predictions, files = collect(directory, manifest)
    if (digest({k: v for k, v in frozen.items() if k != "freeze_identity"}) != frozen["freeze_identity"]
            or frozen["run_identity"] != manifest["run_identity"]
            or frozen["predictions"] != predictions or frozen["files"] != files):
        raise ValueError("frozen_predictions_changed")
    return frozen


def summarize(pairs):
    count = len(pairs)
    direct = sum(p["direct"]["correct"] for p in pairs)
    pose = sum(p["pose"]["correct"] for p in pairs)
    return {
        "questions": count, "direct_correct": direct, "pose_correct": pose,
        "direct_accuracy": direct / count if count else None,
        "pose_accuracy": pose / count if count else None,
        "delta_percentage_points": (pose - direct) * 100 / count if count else None,
        "gains": sum(not p["direct"]["correct"] and p["pose"]["correct"] for p in pairs),
        "regressions": sum(p["direct"]["correct"] and not p["pose"]["correct"] for p in pairs),
        "answers_changed": sum(p["direct"]["answer"] != p["pose"]["answer"] for p in pairs),
    }


def score(directory):
    # Separate host process: all predictions are immutable before references are read.
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory)
        frozen = validate_frozen(directory, manifest)
        if read(directory / "execution.json")["status"] != "complete":
            raise ValueError("incomplete_no_reference_access")
        previous = read(OLD_SCORE)
        if (digest({k: v for k, v in previous.items() if k != "score_identity"}) != OLD_SCORE_ID
                or previous["score_identity"] != OLD_SCORE_ID):
            raise ValueError("previous_host_reference_changed")
        references = {p["sample_id"]: p for p in previous["pairs"]}
        predictions = {(p["sample_id"], p["arm"]): p for p in frozen["predictions"]}
        pairs = []
        for item in manifest["items"]:
            sample = item["problem"]["sample_id"]
            old = references[sample]
            if old["input_sha256"] != item["problem"]["input_sha256"]:
                raise ValueError("reference_problem_changed")
            answer = old["reference_answer"]
            pair = {"sample_id": sample, "task_type": item["problem"]["task_type"],
                    "reference_answer": answer, "api_id": item["api_id"],
                    "old_direct_answer": old["direct"]["answer"],
                    "old_depth_answer": old["depth"]["answer"]}
            for arm in ("direct", "pose"):
                p = predictions[sample, arm]
                pair[arm] = {**p, "correct": p["status"] == "answered" and p["answer"] == answer}
            pairs.append(pair)
        result = {
            "schema": "mmsi-joint-camera-score/1", "run_identity": manifest["run_identity"],
            "freeze_identity": frozen["freeze_identity"], "reference_source_score_identity": OLD_SCORE_ID,
            "reference_access": "separate_host_process_after_all_12_predictions_frozen",
            "sample_role": "previously_seen_development_diagnostic",
            "overall": summarize(pairs), "pairs": pairs,
            "by_task": {p["task_type"]: summarize([p]) for p in pairs}, "scored_at": time.time(),
        }
        result["score_identity"] = digest(result)
        write(directory / "score.json", result, exclusive=True)
        return result


def controlled_execute(directory):
    from .depth_service_probe import all_keys
    keys = all_keys()
    result = subprocess.run(
        [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.vggt_pose_trial",
         "execute", "--directory", str(directory)],
        input=json.dumps(keys).encode(), env=clean_environment(), cwd=ROOT)
    if result.returncode:
        raise RuntimeError("online_worker_failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare-gpu", "prepare-online", "controlled-execute",
                                         "execute", "freeze", "score"))
    parser.add_argument("--directory", default=str(DESTINATION))
    args = parser.parse_args()
    if args.phase == "prepare-gpu":
        prepare_gpu(args.directory)
    elif args.phase == "prepare-online":
        result = prepare_online(args.directory)
        print(json.dumps({"status": "prepared", "max_requests": 12,
                          "run_identity": result["run_identity"]}))
    elif args.phase == "controlled-execute":
        controlled_execute(args.directory)
    elif args.phase == "execute":
        keys = json.loads(sys.stdin.read())
        print(json.dumps(execute(args.directory, keys)))
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
        print('{"status":"failed","reason_code":"pose_trial_failed"}', flush=True)
        raise SystemExit(2)
