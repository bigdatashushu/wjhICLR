"""Verify native evidence transferred to Titan and freeze the actual SR solver inputs."""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile


def prepare(root):
    sys.path.insert(0, str(root / "source/scripts/BLINKRunner"))
    import sr_trial as trial
    sr = trial.sr
    geometry = sr.read(root / "geometry_audit.json")
    archive_path = root / "prepared_evidence.tar.gz"
    if sr.file_sha(archive_path) != geometry["prepared_archive_sha256"]:
        raise ValueError("received_native_archive_changed")
    if (root / "prepared").exists() or (root / "manifest.json").exists():
        raise ValueError("received_native_preparation_already_exists")
    members = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or ".." in name.parts
                    or len(name.parts) != 3 or name.parts[0] != "prepared"
                    or member.name in members):
                raise ValueError("received_native_archive_member_invalid")
            data = archive.extractfile(member).read()
            members[member.name] = hashlib.sha256(data).hexdigest()
            if members[member.name] != geometry["files"][member.name]:
                raise ValueError("received_native_member_changed")
            target = root.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    if members != geometry["files"]:
        raise ValueError("received_native_inventory_changed")
    receipt = trial.freeze(root)
    manifest = sr.read(root / "manifest.json")
    if {r["sample_id"]: r["artifact_sha256"] for r in geometry["inputs"]} != {
            r["problem"]["sample_id"]: r["prepared_artifact_sha256"]
            for r in manifest["inspection"]["inputs"]}:
        raise ValueError("received_native_artifact_changed")
    return {"questions": receipt["questions_per_arm"], "model_requests": 0,
            "manifest_sha256": manifest["content_sha256"],
            "actual_initial_prompts_checked": receipt["questions_per_arm"]}


if __name__ == "__main__":
    os.umask(0o077)
    for k in list(os.environ):
        if any(t in k.upper() for t in ("API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "PASSWORD", "SECRET")):
            os.environ.pop(k, None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(prepare(args.root.resolve())))
