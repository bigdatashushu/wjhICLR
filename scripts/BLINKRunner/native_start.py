"""Launch two isolated SR preparation shards on checked physical GPUs 3 and 4."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    os.umask(0o077)
    for name in list(os.environ):
        if any(t in name.upper() for t in (
                "API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "PASSWORD", "SECRET")):
            os.environ.pop(name, None)
    sys.path.insert(0, str(root / "source/scripts/BLINKRunner"))
    import sr_trial as trial
    import yaml
    profile = trial.cohort(root)
    selected = trial.selected_ids(root)
    if any((root / name).exists() for name in ("native_startup.json", "native_supervise.py")):
        raise ValueError("native_launch_already_exists")
    config_path = root / "source/scripts/BLINKRunner/sr_precompute.yaml"
    config = yaml.safe_load(config_path.read_text())["precompute"]
    for name in ("sam2", "moge2"):
        if trial.sr.file_sha(config[name]["checkpoint"]) != config[name]["identity"]["checkpoint_sha256"]:
            raise ValueError("native_checkpoint_changed")
    dino = json.load(urllib.request.urlopen(config["detector"]["endpoint"] + "/health", timeout=15))
    review = json.load(urllib.request.urlopen(config["review"]["endpoint"] + "/health", timeout=15))
    if (not dino.get("status")
            or review.get("identity_sha256") != config["review"]["expected_identity_sha256"]):
        raise ValueError("native_service_identity_changed")
    gpu_rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    memory = {int(a): int(b) for a, b in (row.split(",") for row in gpu_rows.splitlines())}
    if any(memory[gpu] > 16 for gpu in (3, 4)):
        raise ValueError("native_physical_gpu_already_occupied")
    supervisor = '''
import datetime,hashlib,json,os,pathlib,subprocess,sys
root=pathlib.Path(RUN_ROOT)
def emit(name,value):
 with (root/name).open("x") as f: json.dump(value,f)
children=[]
for shard in (0,1):
 env=dict(os.environ)
 for key in list(env):
  if any(t in key.upper() for t in ("API_KEY","TOKEN","ENDPOINT","BASE_URL","PASSWORD","SECRET")):
   env.pop(key,None)
 env.update(CUDA_VISIBLE_DEVICES=str(3+shard),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
 command=[sys.executable,"-B",str(root/"source/scripts/BLINKRunner/sr_trial.py"),
          "precompute","--root",str(root),"--config",
          str(root/"source/scripts/BLINKRunner/sr_precompute.yaml"),"--shard",str(shard)]
 log=(root/("native-"+str(shard)+".log")).open("xb")
 child=subprocess.Popen(command,stdout=log,stderr=log,env=env)
 proc=pathlib.Path("/proc")/str(child.pid)
 children.append((shard,child,log))
 emit("native-start-"+str(shard)+".json",{
  "pid":child.pid,"physical_gpu":3+shard,
  "start_ticks":(proc/"stat").read_text().split(") ",1)[1].split()[19],
  "command_sha256":hashlib.sha256((proc/"cmdline").read_bytes()).hexdigest(),
  "at":datetime.datetime.now().astimezone().isoformat(),"automatic_retry":False})
codes=[]
for shard,child,log in children:
 code=child.wait(); log.close(); codes.append(code)
 emit("native-exit-"+str(shard)+".json",{
  "returncode":code,"at":datetime.datetime.now().astimezone().isoformat(),
  "automatic_retry":False})
emit("native-exit.json",{
 "returncodes":codes,"at":datetime.datetime.now().astimezone().isoformat(),
 "automatic_retry":False})
'''.replace("RUN_ROOT", repr(str(root)))
    (root / "native_supervise.py").write_text(supervisor)
    subprocess.run([
        "tmux", "-L", root.name + "-native", "new-session", "-d", "-s", "native",
        "-c", str(root), shlex.join([sys.executable, "-B", str(root / "native_supervise.py")]),
    ], check=True, capture_output=True, timeout=15)
    receipt = trial.sr.write_once(root / "native_startup.json", {
        "at": trial.now(), "questions": len(selected), "physical_gpus": [3, 4],
        "config_sha256": trial.sr.file_sha(config_path),
        "cohort_sha256": profile["content_sha256"],
        "candidate_policy": config["candidate_selection"],
        "review_identity_sha256": review["identity_sha256"],
        "checkpoint_hashes_verified": True, "model_requests": 0,
        "supervisor_sha256": hashlib.sha256(supervisor.encode()).hexdigest(),
        "launcher_sha256": trial.sr.file_sha(Path(__file__)),
        "reference_answers_read": False, "automatic_retry": False,
    })
    print(json.dumps({"status": "launched", "questions": len(selected),
                      "receipt_sha256": receipt["content_sha256"]}))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1)
