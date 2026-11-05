"""Isolated Terra launch, passive server monitor and local status mirroring."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from .api_pool import resolve_key
from .connection_monitor import report
from .deploy import PYTHON, ROOT, SERVER_ROOT, SSH
from .traces import file_digest, read, write

RUN = "gpt56-terra-high-full1998-20261105-r01"
PARENT = SERVER_ROOT + "/terra-runtime-20261105-r01"
PACKAGE = PARENT + "/GPTexp"
ADAPTER_HASHES = {
    "adapters.py": "b5d648493f40df99a34e672c9599b6163733187cb4b217047de7c0a25833fe76",
    "worker.py": "9661e9a03f793a31de887634f7fc886da1c98f1eece3621a66ee674fad6d1421"}


def remote(script, payload=None, timeout=45):
    reply = subprocess.run(
        SSH + ["-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10",
               "-o", "ServerAliveCountMax=2", PYTHON, "-c", shlex.quote(script)],
        input=json.dumps(payload or {}).encode(), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=timeout)
    if reply.returncode:
        raise RuntimeError("remote_command_failed")
    return json.loads(reply.stdout)


def terra_config():
    config = deepcopy(read(ROOT / "config.json"))
    config["model"] = "gpt-5.6-terra"
    config["worker_module"] = "GPTexp.terra_worker"
    config["service_failure_policy"] = {
        "mode": "cooldown_continue", "cooldown_seconds": 60,
        "transient_http_statuses": [408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
        "transient_reason_codes": ["transport_outcome_unknown", "invalid_response_json"],
        "automatic_request_retry": False,
        "authorization_basis": "2026-11-05: user requested fresh full attempts then gpt-5.6-terra"}
    for api in config["apis"]:
        api.update(model=config["model"], credentials_file="", validated=False, enabled=True)
    return config


def stage():
    subprocess.run(SSH + ["ls", "-ld", SERVER_ROOT, SERVER_ROOT + "/runs"],
                   check=True, timeout=45)
    config_path = ROOT / "deployment/terra.server.config.json"
    write(config_path, terra_config())
    archive = ROOT / "deployment/terra-isolated-public-code.tar.gz"
    assert {n: file_digest(ROOT / n) for n in ADAPTER_HASHES} == ADAPTER_HASHES
    with tarfile.open(archive, "w:gz") as tar:
        for path in ROOT.glob("*.py"):
            tar.add(path, arcname=path.name)
        tar.add(ROOT / "vendor", arcname="vendor")
        tar.add(config_path, arcname="config.json")
    script = """import hashlib,json,pathlib,sys,tarfile
base=pathlib.Path(%r); parent=pathlib.Path(%r); target=pathlib.Path(%r)
assert base.is_dir() and not parent.exists(), "isolated_runtime_already_exists"
before={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in list(base.glob("*.py"))+list((base/"vendor/xcodeapi").glob("*"))
        if p.is_file()}
parent.mkdir(); target.mkdir()
with tarfile.open(fileobj=sys.stdin.buffer,mode="r|gz") as tar:
    for member in tar:
        assert (target/member.name).resolve().is_relative_to(target.resolve())
        assert member.isfile() or member.isdir()
        tar.extract(member,target)
for name in ("frozen","inputs"): (target/name).symlink_to(base/name,target_is_directory=True)
expected=%r
assert {n:hashlib.sha256((target/n).read_bytes()).hexdigest() for n in expected}==expected
after={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in before}
assert before==after
print(json.dumps({"package":str(target),"original_runtime_unchanged":True,
                  "frozen_adapters_verified":True,"credentials_copied":False}))
""" % (SERVER_ROOT, PARENT, PACKAGE, ADAPTER_HASHES)
    with archive.open("rb") as stream:
        reply = subprocess.run(SSH + [PYTHON, "-c", shlex.quote(script)], stdin=stream,
                               stdout=subprocess.PIPE, check=True, timeout=60)
    receipt = {**json.loads(reply.stdout), "at": time.time(), "archive_sha256": file_digest(archive)}
    write(ROOT / "deployment/terra-deployment.json", receipt)
    return receipt


def launch(*, reuse_validation_run=None):
    local_config = read(ROOT / "config.json")
    credentials = {api["api_key_env"]: resolve_key(api) for api in local_config["apis"]}
    script = """import hashlib,json,os,pathlib,subprocess,sys,time
v=json.load(sys.stdin); base=pathlib.Path(v["base"]); run=base/"runs"/v["run"]
assert pathlib.Path(v["cwd"]).is_dir()
run.mkdir(exist_ok=False)
env=dict(os.environ)
for key in tuple(env):
    if any(t in key.upper() for t in ("API_KEY","TOKEN","ENDPOINT","BASE_URL","MODELS_KEY")):
        env.pop(key)
env.update(PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
monitor_env=dict(env); env.update(v["credentials"])
command=[v["python"],"-B","-m","GPTexp.relaunch" if v.get("validation_source") else "GPTexp.launch","--run",str(run),
         "--config",v["cwd"]+"/GPTexp/config.json",
         "--preflight-source",str(base/"runs"/"gpt61-high-full2098-20261105-r01")]
if v.get("validation_source"):
    command+=["--validation-source",str(base/"runs"/v["validation_source"])]
with (run/"bootstrap.log").open("xb") as log:
    owner=subprocess.Popen(command,cwd=v["cwd"],env=env,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=log,start_new_session=True)
fields=pathlib.Path("/proc/"+str(owner.pid)+"/stat").read_text().rsplit(")",1)[1].split()
ticks=int(fields[19])
monitor_command=[v["python"],"-B","-m","GPTexp.connection_monitor","--run",str(run),
                 "--owner-pid",str(owner.pid),"--owner-start-ticks",str(ticks)]
with (run/"connection_monitor.log").open("xb") as log:
    monitor=subprocess.Popen(monitor_command,cwd=v["cwd"],env=monitor_env,
                             stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
receipt={"pid":owner.pid,"start_ticks":ticks,"monitor_pid":monitor.pid,
         "run":str(run),"at":time.time(),"package_parent":v["cwd"],
         "model":"gpt-5.6-terra","reasoning_effort":"high",
         "command_sha256":hashlib.sha256(json.dumps(command).encode()).hexdigest(),
         "credentials_persisted":False,"monitor_has_credentials":False}
(run/"launcher_receipt.json").write_text(json.dumps(receipt)+"\\n")
print(json.dumps(receipt))
"""
    receipt = remote(script, {"base": SERVER_ROOT, "run": RUN, "python": PYTHON,
                              "cwd": PARENT, "credentials": credentials,
                              "validation_source": reuse_validation_run})
    write(ROOT / "deployment/terra-launch.json", receipt)
    (ROOT / "runs" / RUN).mkdir(exist_ok=False)
    write(ROOT / "runs" / RUN / "launcher_receipt.json", receipt)
    return receipt


def status():
    script = """import json,pathlib,sys,time
v=json.load(sys.stdin); run=pathlib.Path(v["run"])
file=run/"connection_status.json"
result={"available":file.exists(),"queried_at":time.time()}
if file.exists(): result["status"]=json.loads(file.read_text())
failure=run/"bootstrap_failure.json"
if failure.exists(): result["bootstrap_failure"]=json.loads(failure.read_text())
print(json.dumps(result))
"""
    receipt = remote(script, {"run": SERVER_ROOT + "/runs/" + RUN})
    target = ROOT / "runs" / RUN
    status_data = receipt.get("status")
    if status_data:
        write(target / "connection_status.json", status_data)
        write(target / "mirror_status.json", {
            "status": "connected", "at": time.time(),
            "server_snapshot_at": status_data["at"], "automatic_retry": False,
            "model_requests": 0})
        (target / "connection_status.md").write_text(report(status_data), encoding="utf-8")
    return receipt


def watch():
    target = ROOT / "runs" / RUN
    for name in tuple(os.environ):
        if any(t in name.upper() for t in ("API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "MODELS_KEY")):
            os.environ.pop(name)
    write(target / "local_monitor.json", {
        "pid": os.getpid(), "started_at": time.time(), "interval_seconds": 5,
        "model_requests": 0, "server_run": SERVER_ROOT + "/runs/" + RUN})
    while True:
        try:
            receipt = status()
            row = receipt.get("status", {})
            print(json.dumps({
                "at": time.time(), "stage": row.get("stage"),
                "validated_apis": row.get("validated_apis"),
                "tcp": row.get("https_tcp_states"), "requests": row.get("requests"),
                "completed_arms": row.get("completed_arms"),
                "server_status_age_seconds": round(time.time() - row.get("at", time.time()), 1)}),
                flush=True)
            if row.get("monitor_status") == "finished":
                break
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            write(target / "mirror_status.json", {
                "status": "ssh_unavailable", "at": time.time(),
                "error_type": type(error).__name__, "model_requests": 0,
                "next_action": "continue_passive_status_reads"})
            print(json.dumps({"at": time.time(), "status": "ssh_unavailable",
                              "error_type": type(error).__name__}), flush=True)
        time.sleep(5)


def main():
    global RUN, PARENT, PACKAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["stage", "launch", "status", "watch"])
    parser.add_argument("--recovery", action="store_true")
    args = parser.parse_args()
    validation_run = None
    if args.recovery:
        validation_run = RUN
        RUN = "gpt56-terra-high-full1998-20261105-r02"
        PARENT = SERVER_ROOT + "/terra-runtime-20261105-r02"
        PACKAGE = PARENT + "/GPTexp"
    result = (launch(reuse_validation_run=validation_run) if args.action == "launch"
              else {"stage": stage, "status": status, "watch": watch}[args.action]())
    if result is not None:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
