"""Read only test preparation/running receipts."""
import json
import os
from pathlib import Path
import sys
P=Path("/home/wangy/wangjh/harness3d/four-datasets-20261031/mmsi-365-test-multiagent-20261101")
for k in list(os.environ):
    if any(t in k.upper() for t in ("API_KEY","TOKEN","ENDPOINT","BASE_URL","MODELS_KEY")):os.environ.pop(k,None)
sys.path.insert(0,str(P))
import pilot
out={"path":str(P),"at":pilot.now(),"live_model_requests":0}
for name in ("preflight_progress.json","preflight_failure.json","preflight_exit.json","ready.json","smoke.json",
    "watcher_startup.json","watcher_progress.json","watcher_launch.json","watcher_failure.json","watcher_exit.json",
    "startup.json","progress.json","completion.json","failure.json","exit.json","cv3d_completion_audit.json","request_pairing_audit.json"):
    if (P/name).exists():out[name]=pilot.read(P/name)
if (P/"preflight.json").exists():
    d=pilot.read(P/"preflight.json")
    out["preflight"]={k:v for k,v in d.items() if k not in {"all_test_records","selected_full_records"}}
if (P/"requests/state.json").exists():
    state=json.loads((P/"requests/state.json").read_text());out["live_model_requests"]=len(state["requests"])
    out["budget_policy"]=state["policy"]
if (P/"scores.json").exists():
    d=pilot.read(P/"scores.json");out["scores"]={k:d[k] for k in ("status","at","summary","by_task")}
for name in ("watcher_startup.json","startup.json"):
    if name in out:
        s=out[name];proc=Path("/proc")/str(s["pid"])
        if proc.exists():
            f=(proc/"stat").read_text().split(") ",1)[1].split()
            out[name+"_process"]={"alive":f[0] not in {"Z","X"},
                "identity_verified":f[19]==s["start_ticks"] and pilot.sha(proc/"cmdline")==s["command_sha256"]}
        else:out[name+"_process"]={"alive":False}
print(json.dumps(out,ensure_ascii=False))
