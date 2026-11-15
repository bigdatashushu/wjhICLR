"""Start the authorized completion with reused cache and wait for result events."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT.parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from control_small_comparison import remote, TITAN_PYTHON

NATIVE = json.loads((ROOT / "diagnostic.json").read_text())["native_root"]
GATEWAY = "/home/cvailab/wangjh/harness3d/vsi-direction20-s06v110-gateway-20261111-r18"
CACHE = "/tmp/vsi-rel-direction20-r16-prepared-cache-20261111"


def action(name, *extra):
    subprocess.run([sys.executable, str(SCRIPTS / "control_small_comparison.py"),
                    name, "--run-root", str(ROOT), "--gateway-root", GATEWAY, *extra],
                   check=True)


def wait_for(filename):
    # Wait for an external tmux task's filesystem event, never a sleep loop.
    program = f"""
import ctypes,json,os,select,time
from pathlib import Path
r=Path({NATIVE!r});name={filename!r}
lib=ctypes.CDLL(None,use_errno=True)
fd=lib.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
assert fd>=0
assert lib.inotify_add_watch(fd,os.fsencode(r),0x100|0x80|0x08)>=0
deadline=time.monotonic()+3600
try:
    while not (r/name).exists():
        assert time.monotonic()<deadline
        if (r/'exit.json').exists() and name!='exit.json':
            raise RuntimeError('completion stopped before requested receipt')
        ready,_,_=select.select([fd],[],[],45)
        if ready:os.read(fd,65536)
    value=json.loads((r/name).read_text())
    assert value['status']=='complete',value
    print(json.dumps({{'receipt':name,'status':value['status']}}))
finally:os.close(fd)
"""
    print(remote("wangy@10.10.0.34", TITAN_PYTHON + " -B -c " + shlex.quote(program)).decode(),
          flush=True)


def main():
    remote("wangy@10.10.0.34", "ls -ld " + shlex.quote(NATIVE) + " " + shlex.quote(CACHE))
    program = f"""
import hashlib,json,os
from pathlib import Path
r=Path({NATIVE!r});cache=Path({CACHE!r})
assert not (r/'prepared_cache').exists() and not (r/'pair_start.json').exists()
assert (cache/'derived_bindings_v4').is_dir()
assert os.statvfs(str(r)).f_bavail*os.statvfs(str(r)).f_frsize>2*1024**3
(r/'prepared_cache').symlink_to(cache,target_is_directory=True)
receipt={{'status':'reused','target':str(cache),
 'origin_identity':'572d9b35e467a95c966a4d9b261efd38545dfa95636489e1ec75674fd91af48e',
 'new_model_requests':0,'home_free_bytes':os.statvfs(str(r)).f_bavail*os.statvfs(str(r)).f_frsize,
 'cache_volume_free_bytes':os.statvfs(str(cache)).f_bavail*os.statvfs(str(cache)).f_frsize}}
(r/'prepared_cache_reuse_receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
"""
    print(remote("wangy@10.10.0.34", TITAN_PYTHON + " -B -c " + shlex.quote(program)).decode(),
          flush=True)
    action("gateway")
    action("health")
    action("preflight")
    wait_for("preflight_receipt.json")
    action("start")
    wait_for("exit.json")
    action("collect", "--collection-chunk-bytes", "1048576")


if __name__ == "__main__":
    main()
