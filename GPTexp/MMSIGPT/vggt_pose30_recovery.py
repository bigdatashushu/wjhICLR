"""One 503 replacement; retain five predictions, finish 30 distinct paired cases."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from io import BytesIO
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from types import FunctionType, SimpleNamespace

from ..traces import digest, file_digest, read, write
from . import vggt_pose30_trial as parent, vggt_pose30_gpu as gpu, vggt_pose30_host as host
from .depth_comparison import runner as shared
from .depth_comparison.pool import qualify
from .prototype.direct import operator_lock
from .prototype.inputs import extract_public, load_images, public_problem
from .vggt_pose30_control import SOURCE, SOURCE_SHA, transfer
from .vggt_task_prompt_v2 import pose_note, task_binding

DESTINATION = parent.DESTINATION / "continuation"
REMOTE = "/home/cvailab/wangjh/harness3d/mmsi-pose30-replacement-20261108-v2"
SESSION = "mmsi-pose30-replacement-20261108-v2"
NEW_API = "pool-07"
TERMINAL = {"answered", "invalid_answer"}


def code_identity():
    paths = [parent.base.PACKAGE / name for name in (
        "vggt_pose30_recovery.py", "vggt_pose30_replacement_worker.py",
        "vggt_pose30_recovery_selfcheck.py")]
    return {**parent.code_identity(),
            **{str(p.relative_to(parent.base.ROOT)): file_digest(p) for p in paths}}


def parent_evidence(directory):
    manifest = parent.validate(directory)
    frozen = parent.validate_frozen(directory, manifest)
    execution = read(Path(directory) / "execution.json")
    return manifest, frozen, execution


def failed_item(manifest, frozen, execution):
    failed = [p for p in frozen["predictions"] if p["status"] == "service_error"]
    terminal = [p for p in frozen["predictions"] if p["status"] in TERMINAL]
    if (execution["status"] != "stopped_service_error" or execution["dispatched"] != 6
            or len(failed) != 1 or len(terminal) != 5):
        raise ValueError("unexpected_parent_failure")
    item = next(i for i in manifest["items"] if i["problem"]["sample_id"] == failed[0]["sample_id"])
    _, path = parent.paths(parent.DESTINATION, item, failed[0]["arm"])
    result = read(path)
    if result["error_code"] != "http_error" or result["http_status"] != 503:
        raise ValueError("requires_observed_503")
    return item


def capture(directory=DESTINATION):
    directory = Path(directory)
    manifest, frozen, execution = parent_evidence(parent.DESTINATION)
    failed = failed_item(manifest, frozen, execution)
    if directory.exists() or not directory.parent.is_dir() or file_digest(SOURCE) != SOURCE_SHA:
        raise ValueError("requires_new_continuation")
    excluded = set(manifest["selection"]["excluded_sample_ids"])
    excluded.update(i["problem"]["sample_id"] for i in manifest["items"])
    rows = [r for r in extract_public(read(SOURCE))
            if r["problem"]["task_type"] == failed["problem"]["task_type"]
            and r["problem"]["sample_id"] not in excluded]
    row = sorted(rows, key=lambda r: digest({
        "seed": 137, "sample_id": r["problem"]["sample_id"],
        "input_sha256": r["problem"]["input_sha256"]}))[0]
    from .depth_service_probe import all_keys
    qualification = qualify({NEW_API: all_keys()[NEW_API]})
    if qualification != [{"api_id": NEW_API, "http_status": 200, "terra_listed": True}]:
        raise ValueError("replacement_api_not_qualified")
    receipt = transfer([row])
    directory.mkdir()
    value = {
        "schema": "mmsi-pose30-operational-replacement/1",
        "parent_directory": str(parent.DESTINATION),
        "parent_run_identity": manifest["run_identity"],
        "parent_freeze_identity": frozen["freeze_identity"],
        "failed_sample_id": failed["problem"]["sample_id"], "failed_api_id": failed["api_id"],
        "failed_index": failed["index"], "replacement": row, "new_api_id": NEW_API,
        "replacement_rule": "next_seed137_public_rank_same_type_excluding_all_prior_selected",
        "replacement_reads_labels": False, "reference_answer_visible": False,
        "historical_requests": 6, "planned_new_requests": 55,
        "cumulative_request_limit": 61, "final_questions": 30, "per_type": 5,
        "automatic_retry": False, "qualification": qualification, "transfer": receipt,
        "captured_at": time.time(),
    }
    value["replacement_identity"] = digest(value)
    write(directory / "replacement.json", value, exclusive=True)
    print(json.dumps({"status": "replacement_selected",
                      "failed_sample_id": value["failed_sample_id"],
                      "new_sample_id": row["problem"]["sample_id"],
                      "new_requests": 55, "cumulative_requests": 61}), flush=True)
    return value


def prepare_gpu(directory=DESTINATION):
    directory = Path(directory)
    replacement = read(directory / "replacement.json")
    row = replacement["replacement"]
    original = read(parent.DESTINATION / "gpu/gpu-input.json")
    target = directory / "gpu"
    for name in ("code", "images", "poses"):
        (target / name).mkdir(parents=True, exist_ok=False)
    names = ("vggt_pose.py", "vggt_pose_worker.py", "vggt_pose30_replacement_worker.py")
    for name in names:
        shutil.copyfile(parent.base.PACKAGE / name, target / "code" / name)
    receipts = []
    for index, (receipt, raw) in enumerate(load_images(row["problem"], row["images"])):
        suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[receipt["mime_type"]]
        path = f"images/{row['problem']['sample_id']}-v{index+1}{suffix}"
        (target / path).write_bytes(raw)
        receipts.append({**receipt, "path": path})
    config = {
        **{k: original[k] for k in ("vggt_source", "vggt_source_identity", "weight_path",
                                   "weight_sha256", "environment")},
        "schema": "mmsi-pose30-replacement-gpu/1",
        "worker_identity": {name: file_digest(target / "code" / name) for name in names},
        "items": [{"sample_id": row["problem"]["sample_id"],
                   "input_sha256": row["problem"]["input_sha256"], "images": receipts}],
        "reference_answer_visible": False, "depth_enabled": False,
        "point_enabled": False, "track_enabled": False, "model_requests": 0,
    }
    write(target / "gpu-input.json", config, exclusive=True)
    # Exact parent location was inspected during initial GPU deployment.
    probe = json.loads(gpu.remote(
        "import pathlib,json; "
        f"p=pathlib.Path({REMOTE!r}); print(json.dumps({{'exists':p.exists(),"
        "'parent_exists':p.parent.is_dir()}))"))
    if probe["exists"] or not probe["parent_exists"]:
        raise ValueError("replacement_remote_unavailable")
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for file in sorted(target.rglob("*")):
            if file.is_file():
                archive.add(file, arcname=str(file.relative_to(target)), recursive=False)
    actual = json.loads(gpu.remote(
        "import pathlib,tarfile,sys,io,json,hashlib; "
        f"root=pathlib.Path({REMOTE!r}); root.mkdir(mode=0o700); "
        "\nwith tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read())) as archive:"
        "\n for m in archive.getmembers():"
        "\n  p=pathlib.PurePosixPath(m.name); assert m.isfile() and not p.is_absolute() and '..' not in p.parts"
        "\n  dest=root/m.name; dest.parent.mkdir(parents=True,exist_ok=True)"
        "\n  with dest.open('xb') as f: f.write(archive.extractfile(m).read())"
        "\n(root/'poses').mkdir()"
        "\nprint(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() "
        "for p in sorted(root.rglob('*')) if p.is_file()}))", data=buffer.getvalue()))
    expected = {str(p.relative_to(target)): file_digest(p) for p in target.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("replacement_transfer_changed")
    write(directory / "gpu-transfer.json", {"files": actual, "model_requests": 0}, exclusive=True)
    command = (
        "env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 "
        "OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 "
        f"PYTHONPATH={shlex.quote(REMOTE+'/code')} {shlex.quote(gpu.PYTHON)} -B "
        f"{shlex.quote(REMOTE+'/code/vggt_pose30_replacement_worker.py')} {shlex.quote(REMOTE)} "
        f"> {shlex.quote(REMOTE+'/stdout.log')} 2> {shlex.quote(REMOTE+'/stderr.log')}"
    )
    launch = json.loads(gpu.remote(
        "import subprocess,pathlib,json,time; "
        f"root=pathlib.Path({REMOTE!r}); assert not (root/'gpu-started.json').exists(); "
        "used,total=map(int,subprocess.check_output(['nvidia-smi','--id=3',"
        "'--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).split(', ')); "
        "assert total-used>=14000; "
        f"subprocess.run(['tmux','new-session','-d','-s',{SESSION!r},{command!r}],check=True); "
        f"value={{'session':{SESSION!r},'command_sha256':{digest(command)!r},"
        f"'config_sha256':{file_digest(target/'gpu-input.json')!r},'at':time.time(),"
        "'physical_gpu':3,'model_requests':0,'reference_answer_visible':False}; "
        "\nwith (root/'launch.json').open('x') as f: json.dump(value,f)"
        "\nprint(json.dumps(value))"))
    write(target / "launch.json", launch, exclusive=True)
    print('{"status":"replacement_gpu_started","model_requests":0}', flush=True)


def fetch_gpu(directory=DESTINATION):
    directory = Path(directory)
    raw = gpu.remote(
        "import pathlib,tarfile,io,sys; "
        f"root=pathlib.Path({REMOTE!r}); assert (root/'gpu-complete.json').is_file(); "
        "buffer=io.BytesIO(); "
        "\nwith tarfile.open(fileobj=buffer,mode='w:gz') as archive:"
        "\n for p in [root/'gpu-started.json',root/'gpu-complete.json',*sorted((root/'poses').glob('*.json'))]:"
        "\n  archive.add(p,arcname=str(p.relative_to(root)),recursive=False)"
        "\nsys.stdout.buffer.write(buffer.getvalue())")
    with tarfile.open(fileobj=BytesIO(raw)) as archive:
        for member in archive.getmembers():
            if not member.isfile() or member.name not in {
                    "gpu-started.json", "gpu-complete.json",
                    "poses/" + read(directory / "replacement.json")["replacement"]["problem"]["sample_id"] + ".json"}:
                raise ValueError("unexpected_replacement_gpu_member")
            path = directory / "gpu" / member.name
            with path.open("xb") as stream:
                stream.write(archive.extractfile(member).read())
    replacement_pose(directory)
    print('{"status":"replacement_gpu_verified","joint_forwards":1}', flush=True)


def replacement_pose(directory):
    directory = Path(directory)
    row = read(directory / "replacement.json")["replacement"]
    config, complete = [read(directory / "gpu" / name)
                        for name in ("gpu-input.json", "gpu-complete.json")]
    path = "poses/" + row["problem"]["sample_id"] + ".json"
    pose = read(directory / "gpu" / path)
    original_config = read(parent.DESTINATION / "gpu/gpu-input.json")
    if any(config[k] != original_config[k] for k in (
            "vggt_source_identity", "weight_sha256", "environment")):
        raise ValueError("replacement_model_identity_changed")
    for name, expected in config["worker_identity"].items():
        if file_digest(directory / "gpu/code" / name) != expected:
            raise ValueError("replacement_worker_identity_changed")
    gpu_item = config["items"][0]
    if (gpu_item["sample_id"] != row["problem"]["sample_id"]
            or gpu_item["input_sha256"] != row["problem"]["input_sha256"]
            or len(gpu_item["images"]) != 2):
        raise ValueError("replacement_gpu_input_changed")
    for actual, receipt in zip(gpu_item["images"], row["images"]):
        if ({k: v for k, v in actual.items() if k != "path"} != receipt
                or file_digest(directory / "gpu" / actual["path"]) != receipt["sha256"]):
            raise ValueError("replacement_gpu_rgb_changed")
    if (complete["status"] != "complete" or complete["joint_forwards"] != 1
            or complete["config_sha256"] != file_digest(directory / "gpu/gpu-input.json")
            or complete["files"] != {path: file_digest(directory / "gpu" / path)}
            or pose["input_sha256"] != row["problem"]["input_sha256"]
            or pose["sample_id"] != row["problem"]["sample_id"]
            or pose["original_image_sha256"] != [r["sha256"] for r in row["images"]]
            or pose["weight_sha256"] != parent.base.WEIGHT_SHA
            or pose["environment"] != config["environment"] or pose["joint_forward"] is not True
            or any(pose[k] is not False for k in (
                "reference_answer_visible", "depth_head_executed", "point_head_executed", "track_head_executed"))):
        raise ValueError("replacement_pose_changed")
    pose_note(row["problem"], pose["camera_evidence"])
    return pose


def plan(manifest, frozen, replacement, pose):
    failed_sample = replacement["failed_sample_id"]
    old_predictions = {(p["sample_id"], p["arm"]): p for p in frozen["predictions"]}
    items, pending, reused = [], [], []
    for original in manifest["items"]:
        item = copy.deepcopy(original)
        sample = item["problem"]["sample_id"]
        if sample == failed_sample:
            item.update(**replacement["replacement"])
            sample = item["problem"]["sample_id"]
            item["api_id"] = NEW_API
            item["task_binding"] = task_binding(item["problem"], 2)
            item["pose_text"] = pose_note(item["problem"], pose["camera_evidence"])
        elif item["api_id"] == replacement["failed_api_id"]:
            item["api_id"] = NEW_API
        item["pose_path"] = "poses/" + sample + ".json"
        item["payload_sha256"] = {arm: digest(parent.request(None, item, arm))
                                  for arm in ("direct", "pose")}
        remaining = []
        for arm in item["arm_order"]:
            prior = old_predictions.get((sample, arm))
            if prior is not None and prior["status"] in TERMINAL:
                if prior["api_id"] != item["api_id"]:
                    raise ValueError("reused_pair_api_changed")
                reused.append({**prior, "source": "parent", "source_run_identity": manifest["run_identity"]})
            else:
                if prior is not None and prior["status"] != "not_dispatched":
                    raise ValueError("operational_retry_forbidden")
                remaining.append(arm)
        items.append(item)
        if remaining:
            pending.append({**item, "arm_order": remaining})
    if (len(items) != 30 or len(reused) != 5
            or sum(len(i["arm_order"]) for i in pending) != 55
            or len({i["problem"]["sample_id"] for i in items}) != 30
            or set(Counter(i["problem"]["task_type"] for i in items).values()) != {5}):
        raise ValueError("replacement_plan_changed")
    return items, pending, reused


def prepare(directory=DESTINATION):
    directory = Path(directory)
    original, frozen, _ = parent_evidence(parent.DESTINATION)
    replacement = read(directory / "replacement.json")
    pose = replacement_pose(directory)
    items, pending, reused = plan(original, frozen, replacement, pose)
    (directory / "poses").mkdir()
    for item in items:
        sample = item["problem"]["sample_id"]
        source = (directory / "gpu/poses" / (sample + ".json") if sample == pose["sample_id"]
                  else parent.DESTINATION / "gpu/poses" / (sample + ".json"))
        shutil.copyfile(source, directory / item["pose_path"])
        item["pose_sha256"] = file_digest(directory / item["pose_path"])
    # Plan pending copies inherited the old pose hash for the replacement.
    by_index = {i["index"]: i for i in items}
    pending = [{**by_index[i["index"]], "arm_order": i["arm_order"]} for i in pending]
    apis = list(dict.fromkeys(i["api_id"] for i in items))
    result = {
        "schema": "mmsi-pose30-continuation/1", "items": items, "pending_items": pending,
        "reused_predictions": reused, "parent_directory": str(parent.DESTINATION),
        "parent_run_identity": original["run_identity"], "parent_freeze_identity": frozen["freeze_identity"],
        "replacement": replacement, "questions": 30, "max_requests": 55, "max_inflight": 6,
        "api_ids": apis, "inference_by_api": {api: parent.base.settings(api) for api in apis},
        "historical_requests": 6, "cumulative_request_limit": 61,
        "gpu_files": {name: file_digest(directory / name) for name in (
            "gpu/gpu-input.json", "gpu/gpu-complete.json", "gpu/gpu-started.json", "gpu/launch.json")},
        "code_identity": code_identity(), "environment": parent.base.environment(),
        "sample_role": original["sample_role"], "reference_answer_visible": False,
        "automatic_retry": False, "tools_enabled": False, "depth_images_sent": False,
        "prepared_at": time.time(),
    }
    result["run_identity"] = digest(result)
    write(directory / "manifest.json", result, exclusive=True)
    write(directory / "preflight.json", {
        "run_identity": result["run_identity"], "model_requests": 0, "reference_answers_read": False,
        "retained_predictions": 5, "new_requests": 55, "failed_requests_repeated": 0,
    }, exclusive=True)
    validate(directory, verify_requests=True)
    print(json.dumps({"status": "prepared", "questions": 30, "new_requests": 55,
                      "retained_predictions": 5, "run_identity": result["run_identity"]}), flush=True)
    return result


def validate(directory, *, verify_requests=False):
    directory = Path(directory)
    result = read(directory / "manifest.json")
    original, frozen, execution = parent_evidence(result["parent_directory"])
    replacement = read(directory / "replacement.json")
    failed_item(original, frozen, execution)
    pose = replacement_pose(directory)
    if any(file_digest(directory / name) != expected for name, expected in result["gpu_files"].items()):
        raise ValueError("replacement_gpu_files_changed")
    items, pending, reused = plan(original, frozen, replacement, pose)
    for item in items:
        item["pose_sha256"] = file_digest(directory / item["pose_path"])
        pose_value = read(directory / item["pose_path"])
        if item["pose_text"] != pose_note(item["problem"], pose_value["camera_evidence"]):
            raise ValueError("continuation_pose_text_changed")
    by_index = {i["index"]: i for i in items}
    pending = [{**by_index[i["index"]], "arm_order": i["arm_order"]} for i in pending]
    if (result["schema"] != "mmsi-pose30-continuation/1"
            or digest({k: v for k, v in result.items() if k != "run_identity"}) != result["run_identity"]
            or digest({k: v for k, v in replacement.items() if k != "replacement_identity"})
            != replacement["replacement_identity"] or result["replacement"] != replacement
            or result["parent_run_identity"] != original["run_identity"]
            or result["parent_freeze_identity"] != frozen["freeze_identity"]
            or result["items"] != items or result["pending_items"] != pending
            or result["reused_predictions"] != reused or result["questions"] != 30
            or result["max_requests"] != 55 or result["cumulative_request_limit"] != 61
            or result["historical_requests"] != 6 or result["max_inflight"] != 6
            or result["code_identity"] != code_identity()
            or result["environment"] != parent.base.environment()
            or result["api_ids"] != list(dict.fromkeys(i["api_id"] for i in items))
            or result["inference_by_api"] != {api: parent.base.settings(api) for api in result["api_ids"]}
            or any(result[k] is not False for k in (
                "reference_answer_visible", "automatic_retry", "tools_enabled", "depth_images_sent"))):
        raise ValueError("continuation_configuration_changed")
    preflight = read(directory / "preflight.json")
    if (preflight["run_identity"] != result["run_identity"] or preflight["model_requests"] != 0
            or preflight["reference_answers_read"] is not False):
        raise ValueError("continuation_preflight_changed")
    if verify_requests:
        for item in result["items"]:
            for arm in item["arm_order"]:
                if digest(parent.request(directory, item, arm)) != item["payload_sha256"][arm]:
                    raise ValueError("continuation_payload_changed")
    return result


def execution_manifest(directory, *, verify_requests=False):
    result = validate(directory, verify_requests=verify_requests)
    return {**result, "items": result["pending_items"]}


_namespace = {**vars(shared), "validate": execution_manifest, "request": parent.request,
              "inference": parent.base.settings}
execute = FunctionType(shared.execute.__code__, _namespace, "execute")
execute.__kwdefaults__ = {"client_factory": parent.base.BoundedClient}


def collect(directory, manifest):
    adapter = {**manifest, "items": [
        {**i, "depth_images": [{"path": i["pose_path"], "sha256": i["pose_sha256"]}]}
        for i in manifest["pending_items"]]}
    namespace = {**vars(shared), "inference": parent.base.settings}
    audit = FunctionType(shared.collect.__code__, namespace, "collect")
    predictions, files = audit(directory, adapter)
    for name in ("replacement.json", "gpu/gpu-input.json", "gpu/gpu-complete.json", "gpu/launch.json"):
        files[name] = file_digest(Path(directory) / name)
    combined = [{**p, "source": "continuation"} for p in predictions] + manifest["reused_predictions"]
    if len(combined) != 60 or len({(p["sample_id"], p["arm"]) for p in combined}) != 60:
        raise ValueError("merged_prediction_count_changed")
    return sorted(combined, key=lambda p: (p["index"], p["arm"])), files


def freeze(directory):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory)
        predictions, files = collect(directory, manifest)
        result = {
            "schema": "mmsi-pose30-continuation-freeze/1", "run_identity": manifest["run_identity"],
            "parent_freeze_identity": manifest["parent_freeze_identity"], "predictions": predictions,
            "files": files, "reference_answer_visible": False, "automatic_retry": False,
            "frozen_at": time.time(),
        }
        result["freeze_identity"] = digest(result)
        write(directory / "frozen.json", result, exclusive=True)
        return result


def validate_frozen(directory, manifest):
    frozen = read(Path(directory) / "frozen.json")
    predictions, files = collect(directory, manifest)
    if (digest({k: v for k, v in frozen.items() if k != "freeze_identity"}) != frozen["freeze_identity"]
            or frozen["run_identity"] != manifest["run_identity"] or frozen["predictions"] != predictions
            or frozen["files"] != files):
        raise ValueError("continuation_freeze_changed")
    return frozen


def paths(directory, item, arm):
    reused = read(Path(directory) / "manifest.json")["reused_predictions"]
    source = parent.DESTINATION if any(
        p["sample_id"] == item["problem"]["sample_id"] and p["arm"] == arm for p in reused) else directory
    return shared.paths(source, item, arm)


def controlled_execute(directory):
    from .depth_service_probe import all_keys
    manifest = validate(directory, verify_requests=True)
    available = all_keys()
    keys = {api: available[api] for api in manifest["api_ids"]}
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.vggt_pose30_recovery",
         "execute", "--directory", str(directory)], input=json.dumps(keys).encode(),
        env=parent.base.clean_environment(), cwd=parent.base.ROOT)
    if completed.returncode:
        raise ValueError("continuation_worker_failed")


def host_engine():
    namespace = {**vars(host), "trial": SimpleNamespace(
        validate=validate, validate_frozen=validate_frozen, paths=paths)}
    names = ("score", "score_internal", "report")
    for name in names:
        source = getattr(host, name)
        bound = FunctionType(source.__code__, namespace, name, source.__defaults__)
        bound.__kwdefaults__ = source.__kwdefaults__
        namespace[name] = bound
    return SimpleNamespace(**{name: namespace[name] for name in names})


def controlled_score(directory):
    manifest = validate(directory)
    validate_frozen(directory, manifest)
    if read(Path(directory) / "execution.json")["status"] != "complete":
        raise ValueError("incomplete_no_reference_access")
    code = "from pathlib import Path; import sys; sys.stdout.buffer.write(Path(" + repr(host.REFERENCE) + ").read_bytes())"
    response = subprocess.run(host.SSH + [shlex.quote(host.REMOTE_PYTHON) + " -B -c " + shlex.quote(code)],
                              capture_output=True, timeout=60)
    import hashlib
    import tempfile
    import os
    if response.returncode or hashlib.sha256(response.stdout).hexdigest() != host.REFERENCE_SHA:
        raise ValueError("reference_transfer_failed")
    with tempfile.TemporaryDirectory(prefix="mmsi-pose30-continuation-score-") as temporary:
        path = Path(temporary) / "references.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(response.stdout)
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.vggt_pose30_recovery", "score-internal",
             "--directory", str(directory), "--references", str(path),
             "--reference-sha256", host.REFERENCE_SHA],
            env=parent.base.clean_environment(), cwd=parent.base.ROOT)
        if completed.returncode:
            raise ValueError("continuation_scoring_failed")


def report(directory):
    value = host_engine().report(directory)
    manifest = validate(directory)
    value.update(
        cumulative_model_requests=61, failed_service_requests=1,
        retained_parent_predictions=5, new_model_requests=55, cumulative_gpu_joint_forwards=31,
        operational_replacement={
            "failed_sample_id": manifest["replacement"]["failed_sample_id"],
            "replacement_sample_id": manifest["replacement"]["replacement"]["problem"]["sample_id"],
            "reason": "HTTP_503_before_complete_pair", "selection_reads_labels": False,
        },
    )
    directory = Path(directory)
    write(directory / "结果与轨迹.json", value)
    text = (directory / "实验结果.md").read_text()
    explanation = (
        "首轮pool-04对" + value["operational_replacement"]["failed_sample_id"]
        + "返回HTTP 503，触发停止。保留首轮6次记录和5个合法预测；"
        "按原seed=137排序换入同题型、此前未选择的"
        + value["operational_replacement"]["replacement_sample_id"]
        + "，pool-04尚未派发的题目转到pool-07，两臂共享同槽。"
        "后续55次请求补齐最终30个配对，累计61次模型调用，其中1次503；"
        "没有重复任何已派发的(sample_id, arm)。原30题与替换题共做31次GPU联合位姿计算。"
        "替换依据是服务失败，选择过程未读取参考答案。\n\n"
    )
    text = text.replace("抽样 seed=137，", explanation + "抽样 seed=137，")
    (directory / "实验结果.md").write_text(text, encoding="utf-8")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=(
        "capture", "prepare-gpu", "fetch-gpu", "prepare", "controlled-execute", "execute",
        "controlled-score", "score-internal", "report"))
    parser.add_argument("--directory", default=str(DESTINATION))
    parser.add_argument("--references")
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    try:
        if args.phase == "execute":
            result = execute(args.directory, json.loads(sys.stdin.buffer.read()))
            freeze(args.directory)
            print(json.dumps(result), flush=True)
        elif args.phase == "score-internal":
            value = host_engine().score_internal(args.directory, args.references, args.reference_sha256)
            print(json.dumps({"status": "scored", "overall": value["overall"]}), flush=True)
        else:
            globals()[args.phase.replace("-", "_")](args.directory)
    except Exception:
        print('{"status":"failed","reason_code":"pose30_continuation_failed"}', flush=True)
        raise SystemExit(2)
