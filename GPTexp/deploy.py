"""Public-code deployment; launch credentials travel through SSH stdin only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

from .api_pool import resolve_key
from .traces import file_digest, read, write

ROOT = Path(__file__).resolve().parent
SERVER_ROOT = "/home/wangy/wangjh/harness3d/GPTexp"
PYTHON = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"
SSH = ["ssh", "-F", "/Users/bytedance/wangjh/.trae/skills/harness3d-cluster-ops/assets/ssh_config",
       "-T", "-o", "BatchMode=yes", "-o", "ControlMaster=no", "h3d-titan"]


def inputs():
    source = Path("/Users/bytedance/wangjh/outputs/BLINK_SR_remaining80_20261105_r01")
    target = ROOT / "inputs/sr80"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("precompute-0.json", "precompute-1.json", "split.json"):
        shutil.copyfile(source / name, target / name)
    if not (target / "prepared").exists():
        with tarfile.open(source / "prepared_evidence.tar.gz") as tar:
            for member in tar.getmembers():
                if (not member.name.startswith("prepared/") or member.issym() or member.islnk()
                        or not (target / member.name).resolve().is_relative_to(target.resolve())):
                    raise ValueError("unexpected_sr80_cache_archive_member")
            tar.extractall(target)
    vendor = ROOT / "vendor/xcodeapi"
    vendor.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "client.py", "config.py", "content.py", "errors.py",
                 "profiles.json", "transport.py"):
        shutil.copyfile(ROOT.parent / "xcodeapi" / name, vendor / name)
    write(ROOT / "inputs/deployment_receipt.json", {
        "sr80_archive_sha256": file_digest(source / "prepared_evidence.tar.gz"),
        "vendor": {p.name: file_digest(p) for p in vendor.iterdir()},
        "sr80_prepared_questions": sum(1 for _ in (target / "prepared").glob("*/metadata.json")),
        "credentials_copied": False, "model_requests": 0})


def deploy(*, code_only=False):
    if not code_only:
        inputs()
    directory = ROOT / "deployment"
    directory.mkdir(exist_ok=True)
    config = read(ROOT / "config.json")
    for api in config["apis"]:
        api["credentials_file"] = ""
    write(directory / "server.config.json", config)
    receipts = read(ROOT / "frozen/sources.json")
    sr_extra = ROOT / "frozen/blink-sr/source/BLINK/download_data.py"
    if sr_extra.exists():
        receipts["blink-sr"]["files"]["source/BLINK/download_data.py"] = file_digest(sr_extra)
        write(ROOT / "frozen/sources.json", receipts)
    archive = directory / ("public-code.tar.gz" if code_only else "public-runtime.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for name in (("vendor",) if code_only else ("frozen", "inputs", "vendor")):
            tar.add(ROOT / name, arcname=name)
        if code_only:
            tar.add(ROOT / "frozen/sources.json", arcname="frozen/sources.json")
            if sr_extra.exists():
                tar.add(sr_extra, arcname="frozen/blink-sr/source/BLINK/download_data.py")
        for path in ROOT.glob("*.py"):
            tar.add(path, arcname=path.name)
        tar.add(directory / "server.config.json", arcname="config.json")
    subprocess.run(SSH + ["mkdir", "-p", SERVER_ROOT], check=True, timeout=36000)
    with archive.open("rb") as stream:
        subprocess.run(SSH + ["tar", "-xzf", "-", "-C", SERVER_ROOT],
                       stdin=stream, check=True, timeout=36000)
    write(directory / "deployment.json", {"server_root": SERVER_ROOT, "python": PYTHON,
          "archive_sha256": file_digest(archive), "credentials_copied": False})
    print("Deployed public runtime to", SERVER_ROOT, flush=True)


def remote_action(action, run):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run):
        raise ValueError("invalid_run_directory_name")
    # Long-running owners detach after a successful launch, without losing stdio.
    code = """import json,os,pathlib,subprocess,sys
v=json.load(sys.stdin)
root=pathlib.Path(v["root"])
run=root/"runs"/v["run"]
run.mkdir(parents=True,exist_ok=True)
env=dict(os.environ)
env.update(v.get("credentials",{}))
log=(run/(v["action"]+".log")).open("ab")
if v["action"]=="bootstrap":
    if any((run/name).exists() for name in ("bootstrap.json","manifest.json","startup.json")):
        raise ValueError("run_already_launched")
    command=[v["python"],"-B","-m","GPTexp.launch","--run",str(run)]
else:
    command=[v["python"],"-B","-m","GPTexp.run",
             "prepare" if v["action"]=="prepare-four" else v["action"],"--run",str(run)]
if v["action"]=="prepare-four":
    command+=["--datasets","blink-multiview","cv3d","mmsi","mindcube-tiny"]
p=subprocess.Popen(command,cwd=root.parent,env=env,stdout=log,stderr=log,
                   stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({"pid":p.pid,"action":v["action"],"run":str(run)}))
"""
    config = read(ROOT / "config.json")
    credentials = {}
    if action in ("execute", "bootstrap"):
        if action == "execute" and not any(api["enabled"] and api["validated"] for api in config["apis"]):
            raise ValueError("formal_execution_requires_validated_configuration")
        credentials = {api["api_key_env"]: resolve_key(api)
                       for api in config["apis"]
                       if api["enabled"] and (action == "bootstrap" or api["validated"])}
    payload = {"root": SERVER_ROOT, "python": PYTHON, "action": action,
               "run": run, "credentials": credentials}
    import shlex
    reply = subprocess.run(
        SSH + [PYTHON, "-c", shlex.quote(code)],
        input=json.dumps(payload).encode(), stdout=subprocess.PIPE, check=True, timeout=36000)
    record = json.loads(reply.stdout)
    write(ROOT / "deployment" / (action + "-launch.json"), record)
    print(json.dumps(record))


def collect(run):
    """Collect public run artifacts and request ledger; locks stay remote."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run):
        raise ValueError("invalid_run_directory_name")
    directory = ROOT / "deployment"
    directory.mkdir(exist_ok=True)
    archive = directory / (run + "-collected.tar.gz")
    with archive.open("wb") as stream:
        subprocess.run(SSH + [
            "tar", "-czf", "-", "--exclude=*.lock",
            "--exclude=host_scoring_references.json", "-C", SERVER_ROOT + "/runs", run],
            stdout=stream, check=True, timeout=864000)
    target = ROOT / "runs"
    target.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            path = (target / member.name).resolve()
            if (not member.name.startswith(run + "/") and member.name != run
                    or member.issym() or member.islnk()
                    or not (member.isfile() or member.isdir())
                    or not path.is_relative_to((target / run).resolve())):
                raise ValueError("unexpected_collected_archive_member")
        tar.extractall(target)
    write(target / run / "collection.json", {
        "server_root": SERVER_ROOT + "/runs/" + run, "at": time.time(),
        "archive_sha256": file_digest(archive), "credentials_copied": False,
        "host_reference_files_copied": False})
    print("Collected public results to", target / run, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=[
        "deploy", "sync-code", "prepare", "prepare-four", "bootstrap", "execute", "collect"])
    parser.add_argument("--run", default="gpt61-high-full1998-20261105-r01")
    args = parser.parse_args()
    if args.action in ("deploy", "sync-code"):
        deploy(code_only=args.action == "sync-code")
    elif args.action == "collect":
        collect(args.run)
    else:
        remote_action(args.action, args.run)


if __name__ == "__main__":
    main()
