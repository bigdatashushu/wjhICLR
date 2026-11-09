"""Read-only final deployed-source, selected-asset and owned-process audit."""
import json
from pathlib import Path
import sys

CODE = Path("/home/cvailab/wangjh/embodied-three-rgbd-20261107-r02/embodiedRunner")
sys.path.insert(0, str(CODE))
from rgbd_three_envs.sources import Sources, canonical, file_digest, implementation_hashes
from rgbd_three_envs.worker import selected_assets
from rgbd_three_envs.protocol import Task


def main():
    package = CODE / "rgbd_three_envs"
    config = json.loads((package / "experiment-20261107-r02.json").read_text())
    frozen = json.loads((package / "verification/frozen-manifest-r02.json").read_text())
    assert implementation_hashes() == frozen["implementation_hashes"]
    assert file_digest(package / "experiment-20261107-r02.json") == frozen["config_sha256"]
    runtime = config["runtime"]
    sources = Sources(runtime["source_root"], runtime["manipulation_catalog"], runtime["manipulation_dataset"])
    counts = sources.verify_all()
    assert sources.hashes == frozen["source_hashes"]
    assets = {Task(**row).identity: selected_assets(sources, Task(**row))["dataset_sha256"]
              for row in config["episodes"]}
    old_code = Path("/home/cvailab/wangjh/embodied-three-rgbd-20261107-r01/embodiedRunner/rgbd_three_envs")
    old = json.loads((old_code / "verification/frozen-manifest.json").read_text())
    assert all(file_digest(old_code / name) == sha for name, sha in old["implementation_hashes"].items())
    roots = [str(CODE.parent).encode(), str(old_code.parents[1]).encode()]
    owned = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == __import__("os").getpid():
            continue
        try:
            command = (directory / "cmdline").read_bytes()
            cwd = str((directory / "cwd").resolve()).encode()
        except (OSError, PermissionError):
            continue
        if command and any(root in command or cwd.startswith(root) for root in roots):
            owned.append({"pid": int(directory.name), "executable": Path(command.split(b"\0", 1)[0].decode()).name})
    assert not owned, "owned_processes_still_active"
    print(canonical({"status": "verified", "deployed_r01_unchanged": True,
                     "deployed_r02_unchanged": True, "official_file_counts": counts,
                     "selected_assets_verified": assets, "active_owned_workers": owned,
                     "model_requests": 0, "simulator_launches": 0}).decode())


if __name__ == "__main__":
    main()
