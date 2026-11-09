"""Owned 240-camera precompute; keep old runs and GPU workers unchanged."""
from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
from types import FunctionType

from ..traces import digest, file_digest, read, write
from . import vggt_pose30_gpu as old
from .prototype.inputs import load_images

REMOTE = "/home/cvailab/wangjh/harness3d/mmsi-location240-20261109-r01"
SESSION = "mmsi-location240-20261109-r01"
DESTINATION = old.base.ROOT.parent / "MMSI六类四十题拍摄位置关系-20261109-r01"
SSH = old.SSH[:-1] + [
    "-o", "ControlPath=/Users/bytedance/.cache/harness3d-ssh/mmsi-pose-20261108-4090",
    old.SSH[-1],
]
remote = FunctionType(old.remote.__code__, {**vars(old), "SSH": SSH}, "remote")
remote.__kwdefaults__ = old.remote.__kwdefaults__
status = FunctionType(old.status.__code__, {
    **vars(old), "remote": remote, "REMOTE": REMOTE, "SESSION": SESSION,
}, "status")


def selection(directory):
    selected = read(Path(directory) / "selection.json")
    if (digest({k: v for k, v in selected.items() if k != "selection_identity"})
            != selected["selection_identity"] or selected["questions"] != 240
            or selected["max_requests"] != 240 or selected["max_inflight"] != 6
            or selected["reference_answer_visible"] is not False
            or selected["selection_reads_labels"] is not False):
        raise ValueError("public_selection_changed")
    items = selected["items"]
    if (len(items) != 240 or len({i["problem"]["sample_id"] for i in items}) != 240
            or [i["index"] for i in items] != list(range(240))
            or len(Counter(i["problem"]["task_type"] for i in items)) != 6
            or set(Counter(i["problem"]["task_type"] for i in items).values()) != {40}
            or Counter(i["api_id"] for i in items) != {api: 40 for api in selected["api_ids"]}
            or any(i["arm_order"] != ["relation"] for i in items)):
        raise ValueError("stratified_question_identity_changed")
    if any(file_digest(old.base.ROOT / p) != h
           for p, h in selected["solver_code_identity"].items()):
        raise ValueError("old_solver_changed")
    return selected


def prepare(directory):
    directory = Path(directory)
    selected = selection(directory)
    value = read(directory / "gpu-probe.json")
    if (value["weight_sha256"] != old.base.WEIGHT_SHA
            or value["source"] != read(old.base.DESTINATION / "gpu/gpu-input.json")[
                "vggt_source_identity"]):
        raise ValueError("gpu_source_binding_changed")
    gpu = directory / "gpu"
    if gpu.exists():
        raise ValueError("already_prepared")
    for name in ("images", "poses", "code"):
        (gpu / name).mkdir(parents=True)
    names = ("vggt_pose.py", "vggt_pose_worker.py", "vggt_location_batch_worker.py")
    for name in names:
        shutil.copyfile(old.base.PACKAGE / name, gpu / "code" / name)
    items = []
    for item in selected["items"]:
        sample = item["problem"]["sample_id"]
        receipts = []
        for index, (receipt, raw) in enumerate(load_images(item["problem"], item["images"])):
            suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[
                receipt["mime_type"]]
            path = f"images/{sample}-v{index+1}{suffix}"
            with (gpu / path).open("xb") as stream:
                stream.write(raw)
            receipts.append({**receipt, "path": path})
        items.append({"sample_id": sample, "input_sha256": item["problem"]["input_sha256"],
                      "images": receipts})
    config = {
        "schema": "mmsi-location-batch-gpu-input/1", "questions": 240,
        "selection_identity": selected["selection_identity"],
        "vggt_source": old.SOURCE, "vggt_source_identity": value["source"],
        "weight_path": old.WEIGHT, "weight_sha256": old.base.WEIGHT_SHA,
        "environment": value["environment"], "hostname": "cvailab", "physical_gpu": 3,
        "worker_identity": {name: file_digest(gpu / "code" / name) for name in names},
        "items": items, "reference_answer_visible": False, "depth_enabled": False,
        "point_enabled": False, "track_enabled": False, "model_requests": 0,
    }
    write(gpu / "gpu-input.json", config, exclusive=True)
    transfer(directory)


def transfer(directory):
    directory = Path(directory)
    gpu = directory / "gpu"
    config = read(gpu / "gpu-input.json")
    if config["selection_identity"] != selection(directory)["selection_identity"]:
        raise ValueError("transfer_selection_changed")
    if (directory / "gpu-transfer.json").exists():
        raise ValueError("already_transferred")
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for file in sorted(gpu.rglob("*")):
            if file.is_file():
                archive.add(file, arcname=str(file.relative_to(gpu)), recursive=False)
    code = (
        "import pathlib,tarfile,sys,io,json,hashlib; "
        f"target=pathlib.Path({REMOTE!r}); assert target.parent.is_dir(); "
        "assert not target.exists() or (target.is_dir() and not any(target.iterdir())); "
        "target.mkdir(mode=0o700,exist_ok=True); "
        "\nwith tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read())) as archive:"
        "\n for m in archive.getmembers():"
        "\n  p=pathlib.PurePosixPath(m.name);"
        "\n  assert m.isfile() and not p.is_absolute() and '..' not in p.parts"
        "\n  dest=target/m.name; dest.parent.mkdir(parents=True,exist_ok=True)"
        "\n  with dest.open('xb') as f: f.write(archive.extractfile(m).read())"
        "\n(target/'poses').mkdir()"
        "\nprint(json.dumps({'files':{str(p.relative_to(target)):"
        "hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.rglob('*')) if p.is_file()}}))"
    )
    actual = json.loads(remote(code, data=buffer.getvalue(), timeout=600))["files"]
    expected = {str(p.relative_to(gpu)): file_digest(p)
                for p in sorted(gpu.rglob("*")) if p.is_file()}
    if actual != expected:
        raise ValueError("gpu_transfer_changed")
    write(directory / "gpu-transfer.json", {
        "status": "verified", "remote": REMOTE, "files": actual,
        "reference_answer_visible": False, "model_requests": 0,
    }, exclusive=True)
    print('{"status":"gpu_prepared","questions":240,"model_requests":0}', flush=True)


def launch(directory):
    directory = Path(directory)
    transfer = read(directory / "gpu-transfer.json")
    config_sha = file_digest(directory / "gpu/gpu-input.json")
    command = (
        "env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 "
        "OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 "
        f"PYTHONPATH={shlex.quote(REMOTE+'/code')} "
        f"{shlex.quote(old.PYTHON)} -B {shlex.quote(REMOTE+'/code/vggt_location_batch_worker.py')} "
        f"{shlex.quote(REMOTE)} > {shlex.quote(REMOTE+'/stdout.log')} "
        f"2> {shlex.quote(REMOTE+'/stderr.log')}"
    )
    code = (
        "import pathlib,hashlib,json,subprocess,time; "
        f"root=pathlib.Path({REMOTE!r}); expected={transfer['files']!r}; "
        "assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in expected.items()); "
        "assert not (root/'gpu-started.json').exists() and not (root/'launch.json').exists(); "
        "gpu=subprocess.check_output(['nvidia-smi','--id=3',"
        "'--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip(); "
        "used,total=map(int,gpu.split(', ')); assert total-used>=14000; "
        f"subprocess.run(['tmux','new-session','-d','-s',{SESSION!r},{command!r}],check=True); "
        f"value={{'session':{SESSION!r},'command_sha256':{digest(command)!r},"
        f"'config_sha256':{config_sha!r},'at':time.time(),'physical_gpu':3,"
        "'hostname':'cvailab','free_mib_at_start':total-used,"
        "'reference_answer_visible':False,'model_requests':0}; "
        "\nwith (root/'launch.json').open('x') as f: json.dump(value,f)"
        "\nprint(json.dumps(value))"
    )
    value = json.loads(remote(code, timeout=30))
    write(directory / "gpu/launch.json", value, exclusive=True)
    print(json.dumps({"status": "gpu_started", **value}), flush=True)


def fetch(directory):
    directory = Path(directory)
    complete = json.loads(remote(
        "import pathlib,sys; "
        f"sys.stdout.buffer.write(pathlib.Path({REMOTE+'/gpu-complete.json'!r}).read_bytes())",
        timeout=30))
    selected = selection(directory)
    expected_names = {"poses/" + i["problem"]["sample_id"] + ".json" for i in selected["items"]}
    if (complete["status"] != "complete" or complete["joint_forwards"] != 240
            or set(complete["files"]) != expected_names
            or complete["reference_answer_visible"] is not False
            or complete["model_requests"] != 0
            or complete["config_sha256"] != file_digest(directory / "gpu/gpu-input.json")):
        raise ValueError("gpu_completion_changed")
    names = ["gpu-started.json", "gpu-complete.json", "launch.json", *sorted(complete["files"])]
    result = subprocess.run(SSH + [
        "tar -czf - -C " + shlex.quote(REMOTE) + " " + " ".join(shlex.quote(n) for n in names)],
        capture_output=True, timeout=60)
    if result.returncode:
        raise ValueError("gpu_evidence_transfer_failed")
    raw_files = {}
    with tarfile.open(fileobj=BytesIO(result.stdout)) as archive:
        for member in archive.getmembers():
            if member.name not in names or not member.isfile() or member.name in raw_files:
                raise ValueError("unexpected_gpu_archive")
            raw_files[member.name] = archive.extractfile(member).read()
    if set(raw_files) != set(names):
        raise ValueError("missing_gpu_evidence")
    for name, expected in complete["files"].items():
        if hashlib.sha256(raw_files[name]).hexdigest() != expected:
            raise ValueError("gpu_pose_hash_changed")
    for name, raw in raw_files.items():
        path = directory / "gpu" / name
        if path.exists():
            if name != "launch.json" or read(path) != json.loads(raw):
                raise ValueError("existing_gpu_evidence")
        else:
            with path.open("xb") as stream:
                stream.write(raw)
    write(directory / "gpu-fetch.json", {
        "status": "verified", "joint_forwards": 240, "files": complete["files"],
        "reference_answer_visible": False, "model_requests": 0,
    }, exclusive=True)
    print('{"status":"gpu_evidence_verified","joint_forwards":240,"model_requests":0}', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "transfer", "launch", "status", "fetch"))
    parser.add_argument("--directory", default=str(DESTINATION))
    args = parser.parse_args()
    try:
        globals()[args.phase](args.directory)
    except Exception:
        print('{"status":"failed","reason_code":"location_batch_gpu_operation_failed"}', flush=True)
        raise SystemExit(2)
