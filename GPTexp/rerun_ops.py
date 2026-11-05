"""Deploy only the isolated R02 package; credential values travel in SSH stdin."""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from .api_pool import resolve_key
from .deploy import PYTHON, ROOT, SERVER_ROOT, SSH
from .traces import file_digest, read, write

OLD = "gpt61-high-full1998-20261105-r01"
NEW = "gpt61-high-full1998-20261105-r02"
PACKAGE_PARENT = SERVER_ROOT + "/retry-runtime-20261105-r02"
PACKAGE = PACKAGE_PARENT + "/GPTexp"
SNAPSHOT = "partial-scoring-20261105-r02"


def remote(code, payload=None, timeout=60):
    result = subprocess.run(SSH + [PYTHON, "-c", shlex.quote(code)],
                            input=json.dumps(payload or {}).encode(), stdout=subprocess.PIPE,
                            check=True, timeout=timeout)
    return json.loads(result.stdout)


def stage():
    subprocess.run(SSH + ["ls", "-ld", SERVER_ROOT, SERVER_ROOT + "/runs"],
                   check=True, timeout=45)
    config = deepcopy(read(ROOT / "config.json"))
    config["service_failure_policy"] = {
        "mode": "cooldown_continue", "cooldown_seconds": 60,
        "transient_http_statuses": [408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
        "transient_reason_codes": ["transport_outcome_unknown", "invalid_response_json"],
        "automatic_request_retry": False,
        "authorization_basis": "2026-11-05: user requested a fresh attempt over the full dataset"}
    for api in config["apis"]:
        api["credentials_file"] = ""
    archive = ROOT / "deployment/r02-isolated-public-code.tar.gz"
    config_path = ROOT / "deployment/r02.server.config.json"
    write(config_path, config)
    with tarfile.open(archive, "w:gz") as tar:
        for path in ROOT.glob("*.py"):
            tar.add(path, arcname=path.name)
        tar.add(ROOT / "vendor", arcname="vendor")
        tar.add(config_path, arcname="config.json")
    code = """import json,pathlib,tarfile,sys,hashlib
base=pathlib.Path(%r); parent=pathlib.Path(%r); target=pathlib.Path(%r)
assert base.is_dir() and not parent.exists(), "isolated_runtime_already_exists"
manifest=json.loads((base/"runs"/%r/"manifest.json").read_text())
before={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in manifest["adapter_source"]}
assert before==manifest["adapter_source"], "active_r01_source_changed"
parent.mkdir(); target.mkdir()
with tarfile.open(fileobj=sys.stdin.buffer,mode="r|gz") as tar:
    for member in tar:
        out=target/member.name
        assert out.resolve().is_relative_to(target.resolve()) and (member.isfile() or member.isdir())
        tar.extract(member,target)
for name in ("frozen","inputs"): (target/name).symlink_to(base/name,target_is_directory=True)
after={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in before}
assert before==after
print(json.dumps({"package":str(target),"active_r01_source_unchanged":True,"credentials_copied":False}))
""" % (SERVER_ROOT, PACKAGE_PARENT, PACKAGE, OLD)
    with archive.open("rb") as stream:
        result = subprocess.run(SSH + [PYTHON, "-c", shlex.quote(code)], stdin=stream,
                                stdout=subprocess.PIPE, check=True, timeout=60)
    receipt = {**json.loads(result.stdout), "at": time.time(), "archive_sha256": file_digest(archive)}
    write(ROOT / "deployment/r02-isolated-deployment.json", receipt)
    return receipt


def score_old():
    code = """import json,os,pathlib,subprocess,sys
v=json.load(sys.stdin); env=dict(os.environ)
for key in tuple(env):
    if any(t in key.upper() for t in ("API_KEY","TOKEN","ENDPOINT","BASE_URL","MODELS_KEY")): env.pop(key)
env.update(PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
root=pathlib.Path(v["root"]); source=root/"runs"/v["old"]; target=source/v["snapshot"]
for action in ("freeze","score"):
    command=[v["python"],"-B","-m","GPTexp.partial_score",action,"--snapshot",str(target)]
    if action=="freeze": command+=["--source",str(source)]
    result=subprocess.run(command,cwd=v["cwd"],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=600)
    if result.returncode:
        print(json.dumps({"status":"error","action":action,"error_type":"scoring_process_failed",
                         "trace":result.stderr.decode()[-3000:]})); sys.exit(0)
summary=json.loads((target/"partial_scores.json").read_text())
print(json.dumps({"status":"scored","snapshot":str(target),"summary":summary}))
"""
    receipt = remote(code, {"root": SERVER_ROOT, "old": OLD, "snapshot": SNAPSHOT,
                           "python": PYTHON, "cwd": PACKAGE_PARENT}, timeout=600)
    write(ROOT / "runs" / OLD / "partial_scoring_remote_receipt.json", receipt)
    return receipt


def launch():
    config = read(ROOT / "config.json")
    credentials = {api["api_key_env"]: resolve_key(api) for api in config["apis"]}
    code = """import json,os,pathlib,subprocess,sys,time,hashlib
v=json.load(sys.stdin); root=pathlib.Path(v["root"]); run=root/"runs"/v["new"]
run.mkdir(exist_ok=False); env=dict(os.environ); env.update(v["credentials"])
env.update(PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
command=[v["python"],"-B","-m","GPTexp.relaunch","--run",str(run),
         "--preflight-source",str(root/"runs"/"gpt61-high-full2098-20261105-r01"),
         "--validation-source",str(root/"runs"/v["old"])]
with (run/"bootstrap.log").open("xb") as log:
    p=subprocess.Popen(command,cwd=v["cwd"],env=env,stdout=log,stderr=log,
                       stdin=subprocess.DEVNULL,start_new_session=True)
receipt={"pid":p.pid,"run":str(run),"at":time.time(),"package_parent":v["cwd"],
         "command_sha256":hashlib.sha256(json.dumps(command).encode()).hexdigest(),
         "credentials_persisted":False}
(run/"launcher_receipt.json").write_text(json.dumps(receipt)+"\\n")
print(json.dumps(receipt))
"""
    receipt = remote(code, {"root": SERVER_ROOT, "new": NEW, "old": OLD, "python": PYTHON,
                           "cwd": PACKAGE_PARENT, "credentials": credentials})
    write(ROOT / "deployment/r02-launch.json", receipt)
    return receipt


def status():
    code = """import json,pathlib,sys,time,collections,os
v=json.load(sys.stdin); root=pathlib.Path(v["root"]); result={"at":time.time(),"runs":{}}
for name in (v["old"],v["new"]):
    r=root/"runs"/name
    if not r.exists(): continue
    row={"episodes":sum(1 for _ in (r/"episodes").rglob("episode.json"))}
    for filename in ("bootstrap.json","bootstrap_failure.json","startup.json","progress.json","execution.json",
                     "readiness_audit.json","launcher_receipt.json"):
        p=r/filename
        if p.exists(): row[filename]=json.loads(p.read_text())
    p=r/"pool/state.json"
    if p.exists():
        state=json.loads(p.read_text()); requests=state["requests"]
        row["requests"]=dict(collections.Counter(x["status"] for x in requests))
        row["api_requests"]={api:dict(collections.Counter(x["status"] for x in requests if x["api"]==api))
                             for api in sorted({x["api"] for x in requests})}
        row["cooldowns"]=state.get("cooldowns",{}); row["service_failures"]=len(state.get("service_failures",[]))
    row["pauses"]=[p.name for p in (r/"pool").glob("paused-*.json")]
    pid=row.get("startup.json",row.get("launcher_receipt.json",{})).get("pid")
    row["pid"]=pid; row["pid_alive"]=bool(pid and pathlib.Path("/proc/"+str(pid)).exists())
    result["runs"][name]=row
print(json.dumps(result))
"""
    receipt = remote(code, {"root": SERVER_ROOT, "old": OLD, "new": NEW})
    write(ROOT / "deployment/r02-live-status.json", receipt)
    return receipt


def collect_snapshot():
    target = ROOT / "runs" / OLD / SNAPSHOT
    if target.exists():
        raise ValueError("local_scoring_snapshot_already_collected")
    archive = ROOT / "deployment/r01-partial-scored-snapshot.tar.gz"
    with archive.open("wb") as stream:
        subprocess.run(SSH + ["tar", "-czf", "-", "-C", SERVER_ROOT + "/runs/" + OLD, SNAPSHOT],
                       stdout=stream, check=True, timeout=600)
    parent = target.parent
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if (not (parent / member.name).resolve().is_relative_to(target.resolve())
                    or not (member.isfile() or member.isdir())):
                raise ValueError("scored_snapshot_archive_member_invalid")
        tar.extractall(parent)
    receipt = read(target / "snapshot.json")
    for entry in receipt["index"]:
        for name, expected in entry["files"].items():
            if file_digest(target / name) != expected:
                raise ValueError("scoring_collection_bytes_changed")
    write(target / "collection.json", {"at": time.time(), "archive_sha256": file_digest(archive),
                                      "all_frozen_record_hashes_verified": True})
    return {"snapshot": str(target), "completed_arms": receipt["completed_arms"]}


def collect_scores():
    """Collect small scored outputs; full frozen execution traces remain on Titan."""
    names = ["report.md", "partial_scores.json", "scored_records.json", "snapshot.json"]
    source = SERVER_ROOT + "/runs/" + OLD + "/" + SNAPSHOT
    hashes = remote("""import json,pathlib,hashlib,sys
v=json.load(sys.stdin); r=pathlib.Path(v["source"])
print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in v["names"]}))
""", {"source": source, "names": names})
    archive = ROOT / "deployment/r01-partial-score-summaries.tar.gz"
    with archive.open("wb") as stream:
        subprocess.run(SSH + ["tar", "-czf", "-", "-C", source, *names],
                       stdout=stream, check=True, timeout=60)
    target = ROOT / "runs" / OLD / "partial-score-summary-20261105-r02"
    target.mkdir(exist_ok=False)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.name not in names or not member.isfile():
                raise ValueError("score_summary_archive_member_invalid")
        tar.extractall(target)
    if any(file_digest(target / name) != expected for name, expected in hashes.items()):
        raise ValueError("score_summary_collection_changed")
    write(target / "collection.json", {
        "source": source, "at": time.time(), "file_sha256": hashes,
        "all_collected_summary_hashes_verified": True,
        "full_trajectories_location": source + "/scored_full_trajectories.jsonl"})
    return {"directory": str(target), "verified_files": len(hashes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["stage", "score-old", "launch", "status",
                                        "collect-snapshot", "collect-scores"])
    args = parser.parse_args()
    action = {"stage": stage, "score-old": score_old, "launch": launch, "status": status,
              "collect-snapshot": collect_snapshot, "collect-scores": collect_scores}[args.action]
    print(json.dumps(action(), ensure_ascii=False))


if __name__ == "__main__":
    main()
