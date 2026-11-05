"""Stop the misbound Terra launch, preserve evidence, and verify eight offline arms."""
from __future__ import annotations

import argparse
import json

from .deploy import PYTHON, ROOT, SERVER_ROOT
from .terra_ops import remote
from .traces import write

FIRST = "gpt56-terra-high-full1998-20261105-r01"
SECOND = "gpt56-terra-high-full1998-20261105-r02"
PARENT = SERVER_ROOT + "/terra-runtime-20261105-r02"


def stop_first():
    script = """import hashlib,json,os,pathlib,signal,sys,time
v=json.load(sys.stdin); run=pathlib.Path(v["run"])
receipt=json.loads((run/"launcher_receipt.json").read_text())
completed={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in (run/"episodes").rglob("episode.json")}
pids=[]
for p in pathlib.Path("/proc").iterdir():
    if not p.name.isdigit(): continue
    try:
        args=(p/"cmdline").read_bytes().decode().split("\\0")
        if not any(m in args for m in ("GPTexp.launch","GPTexp.worker")): continue
        if not any(str(run) in a for a in args): continue
        pid=int(p.name)
        if pid==receipt["pid"]:
            fields=(p/"stat").read_text().rsplit(")",1)[1].split()
            assert int(fields[19])==receipt["start_ticks"], "owner_pid_reused"
        pids.append(pid)
    except (OSError,UnicodeError): continue
state=json.loads((run/"pool/state.json").read_text())
stopped={"at":time.time(),"reason_code":"solver_expected_model_still_gpt61",
         "terminated_pids":pids,"retained_saved_arms":len(completed),
         "completed_episode_sha256":completed,"formal_requests":len(state["requests"]),
         "new_requests_sent_by_stop":0,"automatic_retry":False,
         "repaired_run":v["new"],"probe_receipts_retained":True}
(run/"stopped_for_model_binding_repair.json").write_text(json.dumps(stopped)+"\\n")
for pid in pids:
    try: os.kill(pid,signal.SIGTERM)
    except ProcessLookupError: pass
assert all(hashlib.sha256((run/n).read_bytes()).hexdigest()==h for n,h in completed.items())
print(json.dumps(stopped))
"""
    receipt = remote(script, {"run": SERVER_ROOT + "/runs/" + FIRST, "new": SECOND})
    write(ROOT / "runs" / FIRST / "stopped_for_model_binding_repair.json", receipt)
    return {k: v for k, v in receipt.items() if k != "completed_episode_sha256"}


def offline():
    script = """import json,os,pathlib,subprocess,sys,time
v=json.load(sys.stdin); base=pathlib.Path(v["base"]); parent=pathlib.Path(v["cwd"])
env=dict(os.environ)
for key in tuple(env):
    if any(t in key.upper() for t in ("API_KEY","TOKEN","ENDPOINT","BASE_URL","MODELS_KEY")): env.pop(key)
env.update(PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
target=parent/"offline-terra-model-binding"; target.mkdir(exist_ok=False)
rows=[]
for dataset in ("blink-multiview","cv3d","mmsi","mindcube-tiny"):
    for arm in ("direct","harness"):
        output=target/dataset/arm
        command=[v["python"],"-B","-m","GPTexp.terra_smoke",
                 "--preflight",str(base/"runs"/v["source"]/"preflight"/(dataset+".json")),
                 "--output",str(output),"--arm",arm]
        result=subprocess.run(command,cwd=parent,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
        row={"dataset":dataset,"arm":arm,"passed":result.returncode==0}
        file=output/"smoke.json"
        if file.exists():
            x=json.loads(file.read_text())
            row.update(status=x["result"]["status"],tool_calls=x["captured_tool_calls"],
                       authorization_receipts=x["captured_authorization_receipts"],
                       live_model_requests=x["live_model_requests"])
        binding=output/"model_binding.json"
        if binding.exists(): row["binding"]=json.loads(binding.read_text())
        if result.returncode:
            # Location-only failure evidence avoids exception text or input content.
            row["error_type"]="offline_smoke_failed"
            row["error_locations"]=[line.strip() for line in result.stderr.decode().splitlines()
                                    if line.strip().startswith('File "')][-8:]
        rows.append(row)
report={"at":time.time(),"passed":len(rows)==8 and all(r["passed"] for r in rows),
        "live_model_requests":0,"reference_answers_read":False,"rows":rows}
(target/"summary.json").write_text(json.dumps(report)+"\\n")
print(json.dumps(report))
"""
    receipt = remote(script, {"base": SERVER_ROOT, "cwd": PARENT, "python": PYTHON,
                              "source": FIRST}, timeout=600)
    write(ROOT / "deployment/terra-model-binding-offline.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["stop-first", "offline"])
    args = parser.parse_args()
    print(json.dumps({"stop-first": stop_first, "offline": offline}[args.action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
