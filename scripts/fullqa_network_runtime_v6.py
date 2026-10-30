"""A new control lifetime that recognizes verified recovered ACK boundaries."""
import hashlib
import importlib.util
from pathlib import Path

PARENT="468057a1098fb9dd5da7358010086c924eb8e9d53bec7a8ae19912c3c99ec39e"


def source(parent_path):
    raw=Path(parent_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARENT:
        raise ValueError("network runtime v5 changed")
    spec=importlib.util.spec_from_file_location("_fullqa_network_v5",parent_path)
    parent=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    text=parent.text
    old='        if transition.exists():\n'
    new=('        if transition.exists():\n'
         '            released = read(transition)\n'
         '            current_path = WH / ("receiver-g" + str(gpu)) / "current.json"\n'
         '            current = read(current_path) if current_path.exists() else {}\n'
         '            skip_released = gpu == 3 or current.get("status") == "complete"\n'
         '        else:\n'
         '            skip_released = False\n'
         '        if skip_released:\n'
         '            if gpu != 3:\n'
         '                require(current["job_key"] == released["job_key"] '
         'and current["ack_sha256"] == released["ack_sha256"], "recovered ACK differs")\n'
         '                key = sha(current["job_key"].encode())\n'
         '                require(sha((WR / "worker_cache/acks" / (key + ".json")).read_bytes()) '
         '== released["ack_sha256"], "recovered original ACK missing")\n')
    if text.count(old)!=1:
        raise ValueError("transition site changed")
    text=text.replace(old,new)
    text=text.replace('"gateway_ready_v5.json"','"gateway_ready_v6.json"')
    text=text.replace('"controller_ready_v5.json"','"controller_ready_v6.json"')
    return parent,text


if __name__=="__main__":
    parent_path=Path(__file__).with_name("network_runtime_v5.py")
    if not parent_path.exists():
        parent_path=Path(__file__).parent.parent/"network_scratch_20261025_v1/network_runtime_v5.py"
    parent,text=source(parent_path)
    exec(compile(text,str(parent.parent.native_path),"exec"),{
        "__name__":"__main__","__file__":str(parent.parent.native_path),
        "run_hybrid":parent.hybrid.run_hybrid,
    })
