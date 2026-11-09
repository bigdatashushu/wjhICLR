"""Freeze, deploy and zero-model preflight for the independent acceptance trial."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from .__main__ import check
from .config import load_config
from .perception import Detector
from .sources import ROOT, Sources, canonical, file_digest, implementation_hashes
from .transport import child_environment, preflight

OUT = ROOT / "target_detection_trial"
CONFIG = OUT / "experiment-20261108-r01.json"


def remote(config, code, data=b"", timeout=180):
    # Script and stdin contain only public configuration/evidence.
    args = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "UpdateHostKeys=no", "-o", "ConnectTimeout=10",
            config["runtime"]["ssh_target"], shlex.join(["/usr/bin/python3", "-B", "-c", code])]
    result = subprocess.run(args, input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            env=child_environment(), timeout=timeout)
    if result.returncode:
        raise RuntimeError("remote_lifecycle_failed")
    return json.loads(result.stdout)


def preserved_local():
    rows = {}
    for directory in ("official_direct", "rgbd_trial", "rgbd_three_envs", "rgbd_twenty",
                      "groundingdino_probe", "manipulation_minimal"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.name != ".DS_Store":
                rows[str(path.relative_to(ROOT))] = file_digest(path)
    return rows


REMOTE_STATE = r'''
import datetime,hashlib,json,os,shutil,subprocess
from pathlib import Path
old=Path("/home/cvailab/wangjh/embodied-rgbd-twenty-20261107-r01/embodiedRunner")
preserved={}
for kind in ("rgbd_trial","rgbd_three_envs"):
    folder=old/kind
    for p in sorted(folder.rglob("*")):
        relative=p.relative_to(folder)
        if p.is_file() and "workers" not in relative.parts and "__pycache__" not in relative.parts:
            preserved[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
processes=[]
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        if proc.stat().st_uid!=os.getuid():
            continue
        raw=(proc/"cmdline").read_bytes()
        if b"target_detection_trial/worker.py" not in raw:
            continue
        stat=(proc/"stat").read_text().split(") ",1)[1].split()
        processes.append({"pid":int(proc.name),"start_ticks":int(stat[19]),
                          "command_sha256":hashlib.sha256(raw).hexdigest()})
    except (FileNotFoundError,PermissionError,ProcessLookupError):
        pass
print(json.dumps({"status":"ok","checked_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "node":"cvailab@10.10.0.198","disk_free_bytes":shutil.disk_usage("/home/cvailab/wangjh").free,
 "gpus":subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.free,utilization.gpu",
 "--format=csv,noheader,nounits"],timeout=15).decode().splitlines(),
 "owned_trial_workers":processes,"preserved_old_evidence":preserved}))
'''


def freeze():
    assert not (OUT / "verification/frozen.json").exists(), "already_frozen"
    config, tasks = load_config(CONFIG)
    sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
    offline = check(config, tasks, sources)
    for path in OUT.glob("*.py"):
        ast.parse(path.read_text(), feature_version=(3, 9))
    # Verify the reused packages against the prior frozen experiment.
    old = json.loads((ROOT / "rgbd_twenty/frozen.json").read_text())
    for package, pin in old["packages"].items():
        assert {p.name: file_digest(p) for p in (ROOT / package).glob("*.py")} == pin["implementation_hashes"]
    preserved = preserved_local()
    (OUT / "verification/preserved-local.json").write_bytes(canonical(preserved) + b"\n")
    (OUT / "verification/offline-check.json").write_bytes(canonical(offline) + b"\n")
    frozen = {"schema": "target-detection-frozen/1", "config_sha256": file_digest(CONFIG),
              "implementation_hashes": implementation_hashes(), "source_hashes": sources.hashes,
              "selected_tasks": [t.as_dict() for t in tasks], "preserved_local_files": len(preserved),
              "model_requests": 0, "simulator_launches": 0, "automatic_retry": False,
              "created_at": datetime.now(timezone.utc).isoformat()}
    (OUT / "verification/frozen.json").write_bytes(canonical(frozen) + b"\n")
    with tarfile.open(OUT / "verification/deployment.tar.gz", "w:gz") as tar:
        for p in sorted(OUT.glob("*.py")) + [CONFIG, OUT / "verification/frozen.json"]:
            tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    print(canonical({"status": "frozen", "preserved_files": len(preserved),
                     "source_files": len(frozen["implementation_hashes"])}).decode())


def deploy():
    config, _ = load_config(CONFIG)
    assert json.loads((OUT / "verification/frozen.json").read_text())["implementation_hashes"] == implementation_hashes()
    state = remote(config, REMOTE_STATE)
    assert not state["owned_trial_workers"] and state["disk_free_bytes"] >= 2 * 1024**3
    (OUT / "verification/server-before.json").write_bytes(canonical(state) + b"\n")
    code = r'''
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(ROOT_VALUE)
assert root.parent.parent.is_dir()
assert not root.parent.exists(), "deployment_exists"
root.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode="r:gz") as tar:
    for m in tar.getmembers():
        p=(root/m.name).resolve()
        assert root in p.parents and m.isfile() and not m.issym() and not m.islnk()
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(tar.extractfile(m).read())
folder=root/"target_detection_trial"
(folder/"runs").mkdir()
frozen=json.loads((folder/"verification/frozen.json").read_text())
actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob("*.py")}
assert actual==frozen["implementation_hashes"]
print(json.dumps({"status":"deployed","root":str(root),"implementation_hashes":actual}))
'''.replace("ROOT_VALUE", repr(config["runtime"]["code_root"]))
    result = remote(config, code, (OUT / "verification/deployment.tar.gz").read_bytes())
    (OUT / "verification/deployment.json").write_bytes(canonical(result) + b"\n")
    print(canonical({"status": "deployed", "root": result["root"]}).decode())


def probe():
    config, tasks = load_config(CONFIG)
    sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
    check(config, tasks, sources)
    reports = {t.identity: preflight(config, t, sources) for t in tasks}
    result = {"status": "runtime_ready", "reports": reports, "detector": Detector(config).status(),
              "model_requests": 0, "simulator_launches": 0}
    (OUT / "verification/preflight.json").write_bytes(canonical(result) + b"\n")
    print(canonical({"status": "runtime_ready", "environments": len(reports),
                     "model_requests": 0, "simulator_launches": 0}).decode())


def after():
    config, _ = load_config(CONFIG)
    before = json.loads((OUT / "verification/server-before.json").read_text())
    state = remote(config, REMOTE_STATE)
    state["old_evidence_unchanged"] = state["preserved_old_evidence"] == before["preserved_old_evidence"]
    state["local_old_files_unchanged"] = preserved_local() == json.loads(
        (OUT / "verification/preserved-local.json").read_text())
    assert state["old_evidence_unchanged"] and state["local_old_files_unchanged"]
    assert not state["owned_trial_workers"], "owned_worker_not_closed"
    state["implementation_unchanged"] = implementation_hashes() == json.loads(
        (OUT / "verification/frozen.json").read_text())["implementation_hashes"]
    assert state["implementation_unchanged"]
    state["detector"] = Detector(config).status()
    (OUT / "verification/server-after.json").write_bytes(canonical(state) + b"\n")
    print(canonical({k: state[k] for k in ("status", "checked_at", "old_evidence_unchanged",
                                          "local_old_files_unchanged", "owned_trial_workers")}).decode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "deploy", "probe", "after"))
    phase = parser.parse_args().phase
    globals()[phase]()
