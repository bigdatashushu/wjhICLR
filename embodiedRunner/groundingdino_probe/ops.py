"""Read-only discovery and credential-free SSH transport for screenshot probes."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_three_envs.transport import child_environment

OUT = Path(__file__).resolve().parent
SSH = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
       "-o", "UpdateHostKeys=no", "-o", "ConnectTimeout=10", "cvailab@10.10.0.198"]


def remote(code, *, python="/usr/bin/python3", prefix=(), timeout=240):
    argv = SSH + [shlex.join([*prefix, python, "-B", "-"])]
    result = subprocess.run(argv, input=code.encode(), stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, env=child_environment(), timeout=timeout)
    if result.returncode:
        raise RuntimeError("remote_process_failed")
    text = result.stdout.decode()
    value, end = json.JSONDecoder().raw_decode(text.lstrip())
    trailing = text.lstrip()[end:].strip()
    if trailing:
        value["native_trailing_output_lines"] = len(trailing.splitlines())
    if value.get("status") == "error":
        raise RuntimeError(f"{value.get('code', 'remote_error')} "
                           f"({value.get('exception_type', '')}; {value.get('stage', '')})")
    return value


INSPECT = r'''
import hashlib,json,re
from pathlib import Path
base=Path("/home/cvailab/wangjh/harness3d/detector-optimization-20261022")
paths=[base/"backends.json",base/"method_config_v2.json",
       Path("/home/cvailab/wangjh/harness3d/precompute-20261022-r04/receipts/harness3d-dino-pool-r04-auth-v1.json")]
out={}
for path in paths:
    raw=path.read_bytes(); value=json.loads(raw)
    safe={"keys":list(value),"file_sha256":hashlib.sha256(raw).hexdigest()}
    for name in ("pid","port","gpu","status","started_at","started_utc"):
        if name in value and isinstance(value[name],(str,int,float,bool)):
            safe[name]=value[name]
    safe["hashes"]={k:v for k,v in value.items() if isinstance(v,str) and re.fullmatch("[0-9a-f]{64}",v)}
    if isinstance(value.get("process"),dict):
        safe["process"]={k:v for k,v in value["process"].items() if k in ("pid","start_time","start_time_ticks")}
    if isinstance(value.get("backends"),list):
        safe["backends"]=[{k:v for k,v in row.items() if k in ("backend","receipt")} for row in value["backends"]]
    out[str(path)]=safe
for path in sorted(base.glob("native-g*/groundingdino_startup.json")):
    value=json.loads(path.read_text())
    safe={k:value[k] for k in ("status","gpu","port") if k in value}
    identity=value.get("generation_identity",{})
    safe["generation_identity"]={k:v for k,v in identity.items() if isinstance(v,(str,int,bool))}
    safe["model_sha256"]=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    out[str(path)]=safe
folder=Path("/home/cvailab/wangjh/embodied-rgbd-twenty-20261107-r01/embodiedRunner/rgbd_three_envs/runs/rgbd-twenty-20261107-r01/workers/manipulation-base-001")
out["manipulation_saved_pngs"]=[str(p.relative_to(folder)) for p in folder.rglob("*.png")][:30]
print(json.dumps({"status":"ok","inspection":out}))
'''


if __name__ == "__main__":
    value = remote(INSPECT, timeout=60)
    (OUT / "service-inspection.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))
