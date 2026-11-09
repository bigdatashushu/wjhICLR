"""Separate 4090 camera precompute, verified transfer, bounded task ownership."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time

from ..traces import digest, file_digest, read, write
from . import vggt_pose_trial as base
from .prototype.inputs import load_images
from .vggt_pose30_control import DESTINATION
from .vggt_pose30_trial import selection

REMOTE = "/home/cvailab/wangjh/harness3d/mmsi-pose30-20261108-v2"
PYTHON = "/home/cvailab/anaconda3/envs/skill3d-exp/bin/python"
SOURCE = "/home/cvailab/wangjh/harness3d/precompute-20261022-r04/source_frozen/third_party/vggt"
WEIGHT = "/home/cvailab/.cache/huggingface/geothinker/VGGT-1B/model.safetensors"
SESSION = "mmsi-pose30-20261108-v2"
SSH = [
    "ssh", "-F", str(base.ROOT.parent / ".trae/skills/harness3d-cluster-ops/assets/ssh_config"),
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-T", "h3d-4090",
]


def remote(code, *, timeout=120, data=None):
    result = subprocess.run(SSH + [shlex.quote(PYTHON) + " -B -c " + shlex.quote(code)],
                            input=data, capture_output=True, timeout=timeout)
    if result.returncode:
        raise ValueError("gpu_remote_operation_failed")
    return result.stdout


def probe(directory):
    code = (
        "import pathlib,hashlib,json,shutil,socket,subprocess,sys; "
        f"source=pathlib.Path({SOURCE!r}); weight=pathlib.Path({WEIGHT!r}); "
        f"target=pathlib.Path({REMOTE!r}); assert socket.gethostname()=='cvailab'; "
        "def_placeholder=None; "
        "\ndef sha(p):"
        "\n h=hashlib.sha256()"
        "\n with p.open('rb') as f:"
        "\n  for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)"
        "\n return h.hexdigest()"
        "\nimport torch,torchvision,numpy,PIL"
        "\ngpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total',"
        "'--format=csv,noheader,nounits'],text=True)"
        "\napps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',"
        "'--format=csv,noheader,nounits'],text=True)"
        "\nuuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid',"
        "'--format=csv,noheader'],text=True)"
        "\nuuid=dict(line.split(', ',1) for line in uuids.strip().splitlines())['3']"
        "\npids=[line.split(', ')[1] for line in apps.splitlines() if line.startswith(uuid+', ')]"
        "\nowners=subprocess.check_output(['ps','-o','pid,user,comm','-p',','.join(pids)],"
        "text=True) if pids else ''"
        "\nout={'hostname':socket.gethostname(),'target_exists':target.exists(),"
        "'parent_exists':target.parent.is_dir(),'free_bytes':shutil.disk_usage(target.parent).free,"
        "'source':{str(p.relative_to(source)):sha(p) for p in sorted((source/'vggt').rglob('*.py'))},"
        "'weight_sha256':sha(weight),'gpus':gpu,'gpu3_owners':owners,"
        "'environment':{'python':sys.version.split()[0],'torch':torch.__version__,"
        "'torchvision':torchvision.__version__,'numpy':numpy.__version__,'pillow':PIL.__version__,"
        "'gpu':'NVIDIA GeForce RTX 4090','precision':'bfloat16_aggregator_float32_camera'}}"
        "\nprint(json.dumps(out))"
    )
    value = json.loads(remote(code))
    expected = read(base.DESTINATION / "gpu/gpu-input.json")["vggt_source_identity"]
    gpu3 = [line for line in value["gpus"].splitlines() if line.startswith("3, ")][0].split(", ")
    if (value["source"] != expected or value["weight_sha256"] != base.WEIGHT_SHA
            or value["target_exists"] or not value["parent_exists"]
            or value["free_bytes"] < 15 * 1024**3 or int(gpu3[3]) - int(gpu3[2]) < 14000):
        raise ValueError("gpu_assets_or_resources_unavailable")
    value.update(at=time.time(), reference_answer_visible=False, model_requests=0,
                 authorized_physical_gpu=3)
    write(Path(directory) / "gpu-probe.json", value, exclusive=True)
    return value


def prepare(directory):
    directory = Path(directory)
    selected = selection(directory)
    value = (read(directory / "gpu-probe.json") if (directory / "gpu-probe.json").exists()
             else probe(directory))
    gpu = directory / "gpu"
    if (gpu / "gpu-input.json").exists() or (directory / "gpu-transfer.json").exists():
        raise ValueError("already_prepared")
    for name in ("images", "poses", "code"):
        (gpu / name).mkdir(parents=True, exist_ok=True)
    names = ("vggt_pose.py", "vggt_pose_worker.py", "vggt_pose30_worker.py")
    for name in names:
        path = gpu / "code" / name
        if path.exists():
            if file_digest(path) != file_digest(base.PACKAGE / name):
                raise ValueError("existing_worker_changed")
        else:
            shutil.copyfile(base.PACKAGE / name, path)
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
        "schema": "mmsi-pose30-gpu-input/1", "selection_identity": selected["selection_identity"],
        "vggt_source": SOURCE, "vggt_source_identity": value["source"],
        "weight_path": WEIGHT, "weight_sha256": base.WEIGHT_SHA,
        "environment": value["environment"], "hostname": "cvailab", "physical_gpu": 3,
        "worker_identity": {name: file_digest(gpu / "code" / name) for name in names},
        "items": items, "reference_answer_visible": False, "depth_enabled": False,
        "point_enabled": False, "track_enabled": False, "model_requests": 0,
    }
    write(gpu / "gpu-input.json", config, exclusive=True)
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for file in sorted(gpu.rglob("*")):
            if file.is_file():
                archive.add(file, arcname=str(file.relative_to(gpu)), recursive=False)
    code = (
        "import pathlib,tarfile,sys,io,json,hashlib; "
        f"target=pathlib.Path({REMOTE!r}); assert target.parent.is_dir(); "
        "target.mkdir(mode=0o700); "
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
    actual = json.loads(remote(code, data=buffer.getvalue()))["files"]
    expected = {str(p.relative_to(gpu)): file_digest(p) for p in sorted(gpu.rglob("*")) if p.is_file()}
    if actual != expected:
        raise ValueError("gpu_transfer_changed")
    write(directory / "gpu-transfer.json", {
        "status": "verified", "remote": REMOTE, "files": actual,
        "reference_answer_visible": False, "model_requests": 0,
    }, exclusive=True)
    print('{"status":"gpu_prepared","questions":30,"model_requests":0}', flush=True)


def launch(directory):
    directory = Path(directory)
    config_sha = file_digest(directory / "gpu/gpu-input.json")
    command = (
        "env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 "
        "OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 "
        f"PYTHONPATH={shlex.quote(REMOTE+'/code')} "
        f"{shlex.quote(PYTHON)} -B {shlex.quote(REMOTE+'/code/vggt_pose30_worker.py')} "
        f"{shlex.quote(REMOTE)} > {shlex.quote(REMOTE+'/stdout.log')} "
        f"2> {shlex.quote(REMOTE+'/stderr.log')}"
    )
    code = (
        "import pathlib,hashlib,json,subprocess,time; "
        f"root=pathlib.Path({REMOTE!r}); "
        f"assert hashlib.sha256((root/'gpu-input.json').read_bytes()).hexdigest()=={config_sha!r}; "
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


def status(directory):
    code = (
        "import pathlib,json,subprocess; "
        f"root=pathlib.Path({REMOTE!r}); "
        f"alive=subprocess.run(['tmux','has-session','-t',{SESSION!r}],"
        "capture_output=True).returncode==0; "
        "lines=(root/'stdout.log').read_text().splitlines() if (root/'stdout.log').exists() else []; "
        "print(json.dumps({'alive':alive,'poses':len(list((root/'poses').glob('*.json'))),"
        "'complete':(root/'gpu-complete.json').exists(),'recent':lines[-4:]}))"
    )
    value = json.loads(remote(code, timeout=30))
    print(json.dumps(value), flush=True)
    return value


def fetch(directory):
    directory = Path(directory)
    complete = json.loads(remote(
        "import pathlib,sys; "
        f"sys.stdout.buffer.write(pathlib.Path({REMOTE+'/gpu-complete.json'!r}).read_bytes())",
        timeout=30))
    if (complete["status"] != "complete" or complete["joint_forwards"] != 30
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
    import hashlib
    for name, expected in complete["files"].items():
        if hashlib.sha256(raw_files[name]).hexdigest() != expected:
            raise ValueError("gpu_pose_hash_changed")
    for name, raw in raw_files.items():
        path = directory / "gpu" / name
        if path.exists():
            # Local launch is serialized differently; compare its JSON identity.
            if name != "launch.json" or read(path) != json.loads(raw):
                raise ValueError("existing_gpu_evidence")
        else:
            with path.open("xb") as stream:
                stream.write(raw)
    write(directory / "gpu-fetch.json", {
        "status": "verified", "joint_forwards": 30, "files": complete["files"],
        "reference_answer_visible": False, "model_requests": 0,
    }, exclusive=True)
    print('{"status":"gpu_evidence_verified","joint_forwards":30,"model_requests":0}', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "launch", "status", "fetch"))
    parser.add_argument("--directory", default=str(DESTINATION))
    args = parser.parse_args()
    try:
        globals()[args.phase](args.directory)
    except Exception:
        print('{"status":"failed","reason_code":"pose30_gpu_operation_failed"}', flush=True)
        raise SystemExit(2)
