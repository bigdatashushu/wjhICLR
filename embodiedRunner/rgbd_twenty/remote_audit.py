"""Read-only source, assets, and owned-process checks on the deployed server."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_three_envs.sources import canonical, file_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    frozen = json.loads((ROOT / "rgbd_twenty/frozen.json").read_text())
    assert str(ROOT) == frozen["remote_root"]
    packages, assets = {}, {}
    for package, pin in frozen["packages"].items():
        module = importlib.import_module(package + ".sources")
        assert module.implementation_hashes() == pin["implementation_hashes"]
        config_path = ROOT / "rgbd_twenty" / pin["config"]
        assert file_digest(config_path) == pin["config_sha256"]
        config = json.loads(config_path.read_text())
        runtime = config["runtime"]
        sources = module.Sources(runtime["source_root"], runtime["manipulation_catalog"], runtime["manipulation_dataset"])
        counts = sources.verify_all()
        assert sources.hashes == pin["source_hashes"]
        worker = importlib.import_module(package + ".worker")
        protocol = importlib.import_module(package + ".protocol")
        for row in pin["episodes"]:
            task = protocol.Task(**row)
            selected = worker.selected_assets(sources, task)
            assets[task.identity] = {k: selected[k] for k in
                                    ("dataset_sha256", "asset_sha256", "runtime_sha256") if k in selected}
        packages[package] = {"implementation_unchanged": True, "config_unchanged": True,
                             "source_unchanged": True, "official_file_counts": counts}
    previous = {
        "rgbd_trial": "/home/cvailab/wangjh/embodied-nav-rgbd-20261107-r01/embodiedRunner/rgbd_trial",
        "rgbd_three_envs": "/home/cvailab/wangjh/embodied-three-rgbd-20261107-r02/embodiedRunner/rgbd_three_envs",
    }
    for package, old in previous.items():
        assert all(file_digest(Path(old) / name) == sha for name, sha in
                   frozen["packages"][package]["implementation_hashes"].items())
    owned = []
    roots = [str(ROOT.parent).encode()] + [str(Path(path).parents[1]).encode() for path in previous.values()]
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            command = (directory / "cmdline").read_bytes()
            cwd = str((directory / "cwd").resolve()).encode()
        except (OSError, PermissionError):
            continue
        if command and any(root in command or cwd.startswith(root) for root in roots):
            owned.append({"pid": int(directory.name),
                          "executable": Path(command.split(b"\0", 1)[0].decode()).name})
    assert not owned, "owned_processes_still_active"
    print(canonical({
        "status": "verified", "phase": args.phase, "checked_at": datetime.now(timezone.utc).isoformat(),
        "node": "cvailab@10.10.0.198", "selected": len(assets),
        "packages": packages, "assets": assets, "previous_deployments_unchanged": True,
        "active_owned_workers": owned, "model_requests": 0, "simulator_launches": 0,
    }).decode())


if __name__ == "__main__":
    main()
