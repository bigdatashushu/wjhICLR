"""Materialize a separate hash-verified official manipulation runtime snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

from fetch_sources import ROOT, REVISION, REPOSITORY


def main():
    entries = {x["path"]: x for x in json.loads((ROOT / "upstream_tree.json").read_bytes())["tree"]}
    wanted = {name for name, row in entries.items() if row["type"] == "blob" and (
        name.startswith("embodiedbench/envs/eb_manipulation/")
        or name in {
            "embodiedbench/evaluator/config/eb_manipulation_example.py",
            "embodiedbench/evaluator/config/system_prompts.py",
            "embodiedbench/planner/manip_planner.py",
            "embodiedbench/planner/planner_utils.py",
            "embodiedbench/configs/eb-man.yaml",
            "conda_envs/environment_eb-man.yaml", "README.md"})}
    archive = ROOT / "cache/household-official.tar.gz"
    records = []
    destination_root = ROOT / "upstream-manipulation"
    with tarfile.open(archive) as tar:
        for member in tar:
            name = member.name.partition("/")[2]
            if name not in wanted or not member.isfile():
                continue
            data = tar.extractfile(member).read()
            entry = entries[name]
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if blob != entry["sha"]:
                raise ValueError("official_manipulation_blob_changed")
            if data.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
                raise ValueError("unexpected_manipulation_lfs_pointer")
            path = destination_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError("existing_manipulation_source_changed")
            else:
                path.write_bytes(data)
            records.append({"path": name, "git_blob_sha": blob,
                            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    if {r["path"] for r in records} != wanted:
        raise ValueError("incomplete_manipulation_sources")
    manifest = {
        "repository": REPOSITORY, "revision": REVISION,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": sorted(records, key=lambda r: r["path"]),
    }
    path = ROOT / "manipulation_sources.json"
    text = json.dumps(manifest, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError("existing_manipulation_manifest_changed")
    path.write_text(text)
    print(json.dumps({"verified_files": len(records),
                      "verified_bytes": sum(r["bytes"] for r in records)}))


if __name__ == "__main__":
    main()
