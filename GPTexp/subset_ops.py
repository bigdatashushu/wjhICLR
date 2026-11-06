"""Stage the isolated subset and mirror passive status; secrets travel via stdin."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from .api_pool import resolve_key
from .deploy import PYTHON, ROOT, SERVER_ROOT, SSH
from .terra_ops import ADAPTER_HASHES, remote
from .traces import file_digest, read, write

CONTINUATION = False
RUN = "gpt56-terra-high-type30-20261105-r01"
SOURCE = SERVER_ROOT + "/runs/gpt56-terra-high-full1998-20261105-r02"
OLD_PACKAGE = SERVER_ROOT + "/terra-runtime-20261105-r02/GPTexp"
PARENT = SERVER_ROOT + "/type30-runtime-20261105-r01"
PACKAGE = PARENT + "/GPTexp"


def stage():
    subprocess.run(SSH + ["ls", "-ld", SERVER_ROOT, SERVER_ROOT + "/runs", OLD_PACKAGE],
                   check=True, timeout=45)
    archive = ROOT / "deployment/type30-public-code.tar.gz"
    if {n: file_digest(ROOT / n) for n in ADAPTER_HASHES} != ADAPTER_HASHES:
        raise ValueError("frozen_local_adapter_changed")
    with tarfile.open(archive, "w:gz") as tar:
        for path in ROOT.glob("*.py"):
            tar.add(path, arcname=path.name)
        tar.add(ROOT / "vendor", arcname="vendor")
    script = """import hashlib,json,pathlib,sys,tarfile
old=pathlib.Path(%r); parent=pathlib.Path(%r); target=parent/"GPTexp"
assert old.is_dir() and not parent.exists()
paths=[p for p in list(old.glob("*.py"))+list((old/"vendor/xcodeapi").glob("*")) if p.is_file()]
before={str(p.relative_to(old)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
parent.mkdir(); target.mkdir()
with tarfile.open(fileobj=sys.stdin.buffer,mode="r|gz") as archive:
    for member in archive:
        assert (target/member.name).resolve().is_relative_to(target.resolve())
        assert member.isfile() or member.isdir()
        archive.extract(member,target)
for name in ("frozen","inputs"): (target/name).symlink_to(old/name,target_is_directory=True)
expected=%r
assert {n:hashlib.sha256((target/n).read_bytes()).hexdigest() for n in expected}==expected
assert before=={n:hashlib.sha256((old/n).read_bytes()).hexdigest() for n in before}
print(json.dumps({"package":str(target),"original_runtime_unchanged":True,
                  "frozen_adapter_bytes_preserved":True,"credentials_copied":False}))
""" % (OLD_PACKAGE, PARENT, ADAPTER_HASHES)
    with archive.open("rb") as stream:
        reply = subprocess.run(SSH + [PYTHON, "-c", shlex.quote(script)], stdin=stream,
                               stdout=subprocess.PIPE, check=True, timeout=60)
    receipt = {**json.loads(reply.stdout), "at": time.time(), "archive_sha256": file_digest(archive)}
    write(ROOT / "deployment/type30-deployment.json", receipt)
    return receipt


def prepare():
    script = """import json,os,pathlib,sys
v=json.load(sys.stdin)
for k in tuple(os.environ):
    if any(t in k.upper() for t in ("API_KEY","TOKEN","ENDPOINT","BASE_URL","MODELS_KEY")): os.environ.pop(k)
sys.path.insert(0,v["parent"])
if v["continuation"]:
    from GPTexp.continuation import prepare
else:
    from GPTexp.subset import prepare
run=pathlib.Path(v["run"]); assert run.parent.is_dir() and not run.exists()
print(json.dumps(prepare(run,v["source"])))
"""
    receipt = remote(script, {
        "parent": PARENT, "run": SERVER_ROOT + "/runs/" + RUN, "source": SOURCE,
        "continuation": CONTINUATION}, timeout=600)
    write(ROOT / "deployment/type30-preparation.json", receipt)
    return receipt


def launch():
    credentials = {a["api_key_env"]: resolve_key(a) for a in read(ROOT / "config.json")["apis"]}
    script = """import hashlib,json,os,pathlib,subprocess,sys,time
v=json.load(sys.stdin); run=pathlib.Path(v["run"])
assert (run/"readiness_audit.json").exists() and not (run/"launcher_receipt.json").exists()
env=dict(os.environ)
for k in tuple(env):
    if any(t in k.upper() for t in ("API_KEY","TOKEN","ENDPOINT","BASE_URL","MODELS_KEY")): env.pop(k)
env.update(PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
monitor_env=dict(env); env.update(v["credentials"])
command=[v["python"],"-B","-m","GPTexp.subset","execute","--run",str(run)]
with (run/"execute.log").open("xb") as log:
    owner=subprocess.Popen(command,cwd=v["parent"],env=env,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=log,start_new_session=True)
fields=pathlib.Path("/proc/"+str(owner.pid)+"/stat").read_text().rsplit(")",1)[1].split()
ticks=int(fields[19])
command_monitor=[v["python"],"-B","-m","GPTexp.connection_monitor","--run",str(run),
                 "--owner-pid",str(owner.pid),"--owner-start-ticks",str(ticks)]
with (run/"connection_monitor.log").open("xb") as log:
    monitor=subprocess.Popen(command_monitor,cwd=v["parent"],env=monitor_env,
                             stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
receipt={"pid":owner.pid,"start_ticks":ticks,"monitor_pid":monitor.pid,"at":time.time(),
         "run":str(run),"package_parent":v["parent"],"model":"gpt-5.6-terra","reasoning_effort":"high",
         "command_sha256":hashlib.sha256(json.dumps(command).encode()).hexdigest(),
         "credentials_persisted":False,"monitor_has_credentials":False}
(run/"launcher_receipt.json").write_text(json.dumps(receipt)+"\\n")
print(json.dumps(receipt))
"""
    receipt = remote(script, {"parent": PARENT, "run": SERVER_ROOT + "/runs/" + RUN,
                              "python": PYTHON, "credentials": credentials})
    write(ROOT / "deployment/type30-launch.json", receipt)
    target = ROOT / "runs" / RUN
    target.mkdir(exist_ok=True)
    write(target / "launcher_receipt.json", receipt)
    return receipt


def status():
    script = """import json,pathlib,sys,time
v=json.load(sys.stdin); run=pathlib.Path(v["run"]); result={"queried_at":time.time(),"json":{},"text":{}}
for name in v["static_json"]+["connection_status.json","progress.json","execution.json","merged_export_receipt.json"]:
    p=run/name
    if p.exists(): result["json"][name]=json.loads(p.read_text())
for name in v["static_text"]+["connection_status.md"]:
    p=run/name
    if p.exists(): result["text"][name]=p.read_text()
print(json.dumps(result))
"""
    target = ROOT / "runs" / RUN
    receipt = remote(script, {
        "run": SERVER_ROOT + "/runs/" + RUN,
        "static_json": [n for n in ("manifest.json", "preparation.json", "readiness_audit.json",
                                     "launcher_receipt.json") if not (target / n).exists()],
        "static_text": [n for n in ("selection_report.md",) if not (target / n).exists()]})
    target.mkdir(exist_ok=True)
    for name, value in receipt["json"].items():
        write(target / name, value)
    for name, value in receipt["text"].items():
        (target / name).write_text(value, encoding="utf-8")
    write(target / "mirror_status.json", {
        "status": "connected", "at": receipt["queried_at"], "model_requests": 0})
    row = receipt["json"].get("connection_status.json", {})
    return {k: row.get(k) for k in ("at", "stage", "owner_alive", "validated_apis", "completed_arms",
                                    "expected_arms", "reused_arms", "new_completed_arms", "requests")}


def watch():
    for name in tuple(os.environ):
        if any(t in name.upper() for t in ("API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "MODELS_KEY")):
            os.environ.pop(name)
    target = ROOT / "runs" / RUN
    write(target / "local_monitor.json", {
        "pid": os.getpid(), "at": time.time(), "interval_seconds": 5, "model_requests": 0})
    while True:
        try:
            row = status()
            print(json.dumps(row), flush=True)
            if row.get("owner_alive") is False:
                break
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            write(target / "mirror_status.json", {
                "status": "ssh_unavailable", "at": time.time(), "error_type": type(error).__name__,
                "model_requests": 0})
        time.sleep(5)


def main():
    global RUN, SOURCE, OLD_PACKAGE, PARENT, PACKAGE, CONTINUATION
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["stage", "prepare", "launch", "status", "watch"])
    parser.add_argument("--continuation", action="store_true")
    args = parser.parse_args()
    if args.continuation:
        CONTINUATION = True
        SOURCE = SERVER_ROOT + "/runs/" + RUN
        OLD_PACKAGE = PACKAGE
        RUN = "gpt56-terra-high-type30-20261105-r02"
        PARENT = SERVER_ROOT + "/type30-runtime-20261105-r02"
        PACKAGE = PARENT + "/GPTexp"
    result = globals()[args.action]()
    if result is not None:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
