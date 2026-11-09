"""Stop only the observed owned r03 Unity initialization, before any model request."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from target_detection.lifecycle import remote, load_config, CONFIG, REMOTE_STATE
from target_detection_r04.sources import canonical

OUT = ROOT / "target_detection_r04/verification"
task = ROOT / "target_detection/runs/target-detection-20261108-r03/alfred-base-000"
assert not list(task.glob("request-*.json")) and not list(task.glob("observation-*.png"))
config, _ = load_config(CONFIG)
state = remote(config, REMOTE_STATE)
pins = state["owned_trial_workers"]
assert len(pins) == 3 and sum(p["kind"] == "worker" for p in pins) == 1
code = r'''
import hashlib,json,os,signal,time
from pathlib import Path
selected=[]
for pin in PINS:
 p=Path("/proc")/str(pin["pid"])
 assert p.stat().st_uid==os.getuid()
 raw=(p/"cmdline").read_bytes()
 stat=(p/"stat").read_text().split(") ",1)[1].split()
 assert int(stat[19])==pin["start_ticks"]
 assert hashlib.sha256(raw).hexdigest()==pin["command_sha256"]
 argv=raw.split(b"\0")
 if pin["kind"]=="native_runtime" and b"thor" in argv[0].lower():
  selected.append(pin)
assert len(selected)==1
log=Path("/home/cvailab/wangjh/embodied-target-detection-20261108-r03/embodiedRunner/target_detection/runs/target-detection-20261108-r03/workers/alfred-base-000/unity.log")
assert time.time()-log.stat().st_mtime>=300
loghash=hashlib.sha256(log.read_bytes()).hexdigest()
pin=selected[0]
os.kill(pin["pid"],signal.SIGTERM)
print(json.dumps({"status":"owned_stale_initializer_stop_sent","process":pin,
 "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
 "log_sha256":loghash,"reason":"OpenGL initialization log unchanged for at least 300 seconds",
 "model_requests_before_stop":0,"automatic_retry":False}))
'''.replace("PINS", repr(pins))
result = remote(config, code)
(OUT / "r03-controlled-initializer-stop.json").write_bytes(canonical(result) + b"\n")
print(canonical(result).decode())
