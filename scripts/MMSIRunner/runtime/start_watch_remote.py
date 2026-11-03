"""Start the authorized test-only wait/launch entry once."""
import json
import os
from pathlib import Path
import subprocess
import sys
P=Path("/home/wangy/wangjh/harness3d/four-datasets-20261031/mmsi-365-test-multiagent-20261101")
sys.path.insert(0,str(P))
import pilot
assert pilot.read(P/"ready.json")["status"]=="ready"
assert not (P/"watcher_startup.json").exists()
pilot.verify(pilot.read(P/"manifest.json"))
env={k:v for k,v in os.environ.items() if not any(t in k.upper() for t in ("API_KEY","TOKEN","ENDPOINT","BASE_URL","MODELS_KEY"))}
py="/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"
code=subprocess.run(["tmux","-L","h3d-mmsi-test-only-20261101","new-session","-d","-s","watcher",
    py+" -B "+str(P/"watcher.py")+" > "+str(P/"watcher.log")+" 2>&1"],env=env,capture_output=True).returncode
print(json.dumps({"status":"watcher_started" if code==0 else "launch_failed","returncode":code,
    "model_requests":0,"path":str(P),"automatic_retry":False,"test_only":True}))
