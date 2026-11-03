"""One-shot CV3D completion gate; never retries a model request or run."""
import collections
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pilot
P=Path(__file__).resolve().parent

def process_state(start):
    pid=start.get("runner_pid",start.get("pid"))
    path=Path("/proc")/str(pid)
    if not path.exists():return {"alive":False,"identity_verified":True}
    fields=(path/"stat").read_text().split(") ",1)[1].split()
    identity=fields[19]==start["start_ticks"] and pilot.sha(path/"cmdline")==start["command_sha256"]
    return {"alive":identity and fields[0] not in {"Z","X"},"identity_verified":identity,"state":fields[0]}

def audit_cv3d(m,full=False):
    cv=Path(m["cv3d_root"])
    cm=pilot.read(cv/"manifest.json")
    assert cm["content_sha256"]==m["cv3d_manifest_sha256"]
    state=json.loads(Path(cm["shared_budget_root"],"state.json").read_text())
    assert state["policy"]["max_requests"]==m["original_total_cap"]
    prefix=json.loads((cv/"shared_budget_prefix.json").read_text())
    assert state["requests"][:len(prefix["requests"])]==prefix["requests"]
    new=state["requests"][len(prefix["requests"]):]
    assert all(r["context"].get("run_identity")==cm["run_identity"] for r in new)
    assert len(new)<=m["cv3d_request_cap"]
    start=pilot.read(cv/"startup.json");proc=process_state(start)
    out={"status":"waiting","at":pilot.now(),"cv3d_used":len(new),"original_ledger_used":len(state["requests"]),
        "process":proc,"mmsi_live_model_requests":len(json.loads((P/"requests/state.json").read_text())["requests"])}
    if (cv/"failure.json").exists():
        out["status"]="cv3d_stopped";return out
    if not (cv/"completion.json").exists():
        if not proc["alive"]:out["status"]="cv3d_process_missing"
        return out
    if not (cv/"exit.json").exists() or not (cv/"scores.json").exists():return out
    completion=pilot.read(cv/"completion.json");exit_record=pilot.read(cv/"exit.json");scores=pilot.read(cv/"scores.json")
    assert exit_record["returncode"]==0 and completion["status"]=="complete"
    assert completion["episodes"]==1680 and completion["questions"]==840
    assert completion["manifest_sha256"]==cm["content_sha256"]==scores["manifest_sha256"]
    assert scores["status"]=="complete" and len(scores["rows"])==1680
    assert all(scores["summary"][arm]["completed"]==840 for arm in ("skill","direct"))
    assert all(scores["summary"][arm]["accuracy_full_denominator"] is not None for arm in ("skill","direct"))
    assert all(r["status"]=="completed" for r in new)
    assert len(state["requests"])+m["shared_policy"]["max_requests"]<=m["original_total_cap"]
    frozen=pilot.read(cv/"predictions_frozen.json")
    assert frozen["episodes"]==1680 and len(frozen["predictions"])==1680
    if full or not (P/"cv3d_completion_audit.json").exists():
        for relative,expected in {**cm["source_files"],**cm["driver_files"],**frozen["predictions"]}.items():
            assert pilot.sha(cv/relative)==expected
        pairs={};request_count=0
        for job in cm["jobs"]:
            result=pilot.read(cv/"results"/job["arm"]/(job["operation_id"]+".json"))
            assert result["job_sha256"]==pilot.sha_value(job)
            for relative,expected in result["request_records"].items():
                assert pilot.sha(cv/relative)==expected
                if not relative.endswith(".response.json"):continue
                response=pilot.read(cv/relative);request_count+=1
                assert response["status"]=="received"
                identity=response["request_identity"];params=identity["parameters"]
                assert params["model"]=="qwen3.8-flash" and params["seed"]==137 and params["temperature"]==0
                assert params["enable_thinking"] is True and params["max_tokens"]==8192
                assert [r["jpeg_sha256"] for r in identity["images_in_order"]]==[r["sha256"] for r in job["images"]]
                assert response["response_identity"]["model"]=="qwen3.8-flash"
                assert response["response_identity"]["model_reported"] is True
                if relative.endswith("001.response.json"):
                    pairs.setdefault(job["sample_id"],{})[job["arm"]]=(job,identity)
        assert len(pairs)==840 and request_count==len(new)
        for arms in pairs.values():
            assert set(arms)=={"skill","direct"}
            a,ai=arms["skill"];b,bi=arms["direct"]
            assert a["input_sha256"]==b["input_sha256"] and ai["parameters"]==bi["parameters"]
            assert ai["images_in_order"]==bi["images_in_order"]
        pilot.update(P/"cv3d_completion_audit.json",{"status":"complete_verified","at":pilot.now(),
            "manifest_sha256":cm["content_sha256"],"episodes":1680,"paired_inputs_verified":840,
            "all_request_receipts_verified":request_count,"frozen_predictions_verified":1680,
            "scores_sha256":pilot.sha(cv/"scores.json"),"mmsi_requests_in_cv3d_ledger":0})
    out["status"]="complete_verified"
    return out
def main():
    with (P/"watcher.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=pilot.read(P/"manifest.json")
        assert pilot.read(P/"ready.json")["status"]=="ready"
        pilot.verify(m)
        assert not (P/"startup.json").exists() and not (P/"watcher_launch.json").exists()
        pid=os.getpid()
        pilot.write(P/"watcher_startup.json",{"pid":pid,"start_ticks":Path(f"/proc/{pid}/stat").read_text().split(") ",1)[1].split()[19],
            "command_sha256":pilot.sha(f"/proc/{pid}/cmdline"),"at":pilot.now(),"automatic_retry":False})
        while True:
            if (P/"STOP").exists():
                pilot.update(P/"watcher_progress.json",{"status":"operator_stopped","at":pilot.now()});return
            out=audit_cv3d(m)
            pilot.update(P/"watcher_progress.json",out)
            if out["status"]=="complete_verified":break
            if out["status"]!="waiting":raise RuntimeError("cv3d_requires_diagnosis")
            time.sleep(30)
        # Keep global online concurrency at four by requiring CV3D to have drained.
        assert not process_state(pilot.read(Path(m["cv3d_root"])/"startup.json"))["alive"]
        pilot.verify(m)
        assert not json.loads((P/"requests/state.json").read_text())["requests"]
        from provider import health
        h=health()
        assert h["schema_version"]==m["jobs"][0]["proxy_schema"]
        pilot.write(P/"watcher_launch.json",{"status":"launch_once","at":pilot.now(),
            "manifest_sha256":m["content_sha256"],"test_only":True,"online_train_requests":0})
        env={k:v for k,v in os.environ.items() if not any(t in k.upper() for t in ("API_KEY","TOKEN","BASE_URL","ENDPOINT","MODELS_KEY"))}
        with (P/"run.log").open("xb") as log:
            code=subprocess.call([sys.executable,"-B",str(P/"pilot.py")],env=env,stdout=log,stderr=log)
        pilot.update(P/"watcher_exit.json",{"returncode":code,"at":pilot.now(),"automatic_retry":False,"test_only":True})
if __name__=="__main__":
    try:main()
    except BaseException as exc:
        pilot.update(P/"watcher_failure.json",{"status":"stopped","error_type":type(exc).__name__,"at":pilot.now(),"automatic_retry":False})
        sys.exit(2)
