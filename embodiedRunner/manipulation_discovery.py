"""Read-only task-scoped server inventory; no inference, simulator or secrets."""
import json
from pathlib import Path
import shutil
import subprocess


def main():
    root = Path("/home/cvailab/wangjh/embodiedRunner")
    parent = root.parent
    directories = {}
    for name in ("runtime", "cache", "data", "venv-nav", "venv-household"):
        directory = root / name
        directories[name] = sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []
    candidates = []
    for directory in (parent, Path("/home/cvailab/anaconda3/envs"), Path("/nas/wangjh")):
        if directory.is_dir():
            candidates.extend(str(p) for p in directory.iterdir()
                              if any(x in p.name.lower() for x in
                                     ("coppelia", "pyrep", "vlmbench", "embodied", "rlbench", "embench")))
    processes = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            data = (proc / "cmdline").read_bytes()
            if any(x in data.lower() for x in
                   (b"coppelia", b"eb_manipulation", b"embodied-action", b"xorg", b"xvfb")):
                # Command contents may carry authentication; publish a hash only.
                import hashlib
                processes.append({"pid": int(proc.name),
                                  "command_sha256": hashlib.sha256(data).hexdigest(),
                                  "executable": (proc / "comm").read_text().strip()})
        except (OSError, PermissionError):
            pass
    disk = shutil.disk_usage(root)
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    print(json.dumps({
        "schema": "embodied-manipulation-discovery/1", "directories": directories,
        "runtime_candidates": candidates, "related_processes": processes,
        "disk_free_bytes": disk.free, "gpu_processes": gpu.stdout.splitlines(),
        "model_requests": 0, "simulator_launches": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
