"""Fetch pinned household sources without altering frozen Navigation sources."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import urllib.request

from fetch_sources import ROOT, REVISION, REPOSITORY


def main():
    entries = {x["path"]: x for x in json.loads((ROOT / "upstream_tree.json").read_text())["tree"]}
    prefixes = ("embodiedbench/envs/eb_alfred/", "embodiedbench/envs/eb_habitat/",
                "embodiedbench/evaluator/", "embodiedbench/planner/", "embodiedbench/configs/")
    wanted = {name for name, row in entries.items()
              if row["type"] == "blob" and (name.startswith(prefixes)
                  or name in ("README.md", "conda_envs/environment.yaml", "setup.py"))}
    environment = {k: v for k, v in os.environ.items()
                   if not any(s in k.upper() for s in ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL"))}
    environment["GH_TOKEN"] = "public-source-fetch"
    archive = ROOT / "cache/household-official.tar.gz"
    if not archive.exists():
        data = subprocess.run(
            ["gh", "api", f"repos/{REPOSITORY}/tarball/{REVISION}",
             "-H", "Authorization: anonymous"], capture_output=True,
            env=environment, check=True, timeout=180).stdout
        archive.write_bytes(data)
    records = []
    with tarfile.open(archive) as tar:
        for member in tar:
            name = member.name.partition("/")[2]
            if name not in wanted or not member.isfile():
                continue
            data = tar.extractfile(member).read()
            entry = entries[name]
            digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if digest != entry["sha"]:
                raise ValueError("Git blob changed: " + name)
            row = {"path": name, "git_blob_sha": digest}
            if data.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
                oid = re.search(rb"oid sha256:([a-f0-9]{64})", data).group(1).decode()
                size = int(re.search(rb"size (\d+)", data).group(1))
                row.update(lfs_sha256=oid, lfs_bytes=size, pointer_sha256=hashlib.sha256(data).hexdigest())
                destination = ROOT / "upstream-household" / name
                if destination.is_file() and destination.stat().st_size == size:
                    data = destination.read_bytes()
                else:
                    url = f"https://media.githubusercontent.com/media/{REPOSITORY}/{REVISION}/{name}"
                    data = urllib.request.urlopen(url, timeout=120).read()
                if len(data) != size or hashlib.sha256(data).hexdigest() != oid:
                    raise ValueError("LFS identity changed: " + name)
            destination = ROOT / "upstream-household" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            row.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            records.append(row)
    if {row["path"] for row in records} != wanted:
        raise ValueError("Incomplete source selection")
    (ROOT / "household_sources.json").write_text(json.dumps({
        "repository": REPOSITORY, "revision": REVISION,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": records}, indent=2) + "\n")
    print(json.dumps({"verified_files": len(records), "verified_bytes": sum(x["bytes"] for x in records)}))


if __name__ == "__main__":
    main()
