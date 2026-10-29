"""Admit a new independent Gate workspace from a real local seed archive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    from skill3d.storage.hot_cache import HotCache
    from skill3d.storage.state_workspace import StateWorkspace
    from skill3d.storage.framework_state_io import FrameworkStateIO

    args.archive_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    descriptor = {
        "cache_root": str(args.cache_root),
        "key": "reviser-successor-gate-20261029",
        "quota_bytes": 4_000_000_000,
        "min_free_bytes": 25_000_000_000,
        "workspace_bytes": 2_000_000_000,
        "seed_archive": str(args.archive_root / "successor_seed.tar"),
        "seed_archive_kind": "local_successor_seed",
        "preparer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    raw = (json.dumps(descriptor, sort_keys=True) + "\n").encode()
    archive_path = Path(descriptor["seed_archive"])
    with tarfile.open(archive_path, "x") as archive:
        info = tarfile.TarInfo("framework_io_binding.json")
        info.size, info.mode = len(raw), 0o600
        archive.addfile(info, io.BytesIO(raw))
    archive_path.chmod(0o600)
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    cache = HotCache(args.cache_root, quota_bytes=descriptor["quota_bytes"],
                     min_free_bytes=descriptor["min_free_bytes"])

    def populate(reservation):
        target = reservation.path / "payload/framework_io_binding.json"
        target.write_bytes(raw)
        target.chmod(0o600)

    root = cache.new_workspace(
        descriptor["key"], populate, initial_bytes=len(raw),
        growth_bytes=descriptor["workspace_bytes"] - len(raw))
    workspace = StateWorkspace.adopt(
        cache, descriptor["key"], baseline_anchor=archive_sha,
        baseline_members={"framework_io_binding.json": {
            "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}},
        mutable_roots=("campaigns", "request_governor", "preflight.json",
                       "startup.json", "final.json", "workspace_preparation.json"),
        workspace_bytes=descriptor["workspace_bytes"])
    bounded = StateWorkspace(cache, descriptor["key"], bounded=True)
    writer = FrameworkStateIO(
        bounded, job="reviser-successor-gate-20261029",
        max_workspace_bytes=descriptor["workspace_bytes"])
    receipt = {"status": "independent_gate_workspace_admitted", "root": str(root),
               "seed_archive_sha256": archive_sha, "descriptor": descriptor,
               "workspace_logical_ceiling": descriptor["workspace_bytes"],
               "model_requests": 0, "original_workspace_modified": False}
    writer.write_json_once(root / "workspace_preparation.json", receipt)
    with (args.archive_root / "workspace_receipt.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
