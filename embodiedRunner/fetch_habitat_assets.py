"""Fetch official Habitat 0.3 rearrangement assets by immutable identity."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.parse
import urllib.request

from protocol import ROOT, canonical

REPOS = (("ReplicaCAD_dataset", "v1.6", "replica_cad"),
         ("hab_fetch", "v2.0", "robots/hab_fetch"),
         ("ycb", "v1.2", "objects/ycb"))
DATA = ROOT / "cache/habitat-assets-data"
MANIFEST = ROOT / "habitat_assets.json"


def request(url):
    return urllib.request.urlopen(url, timeout=120)


def catalog():
    records = []
    for name, tag, destination in REPOS:
        api = "https://huggingface.co/api/datasets/ai-habitat/" + name
        with request(api + "/revision/" + tag) as response:
            revision = json.load(response)["sha"]
        url = api + "/tree/" + revision + "?recursive=true&limit=1000"
        files = []
        while url:
            with request(url) as response:
                rows = json.load(response)
                link = response.headers.get("Link", "")
            for row in rows:
                if row["type"] != "file":
                    continue
                item = {"path": row["path"], "bytes": row["size"],
                        "git_blob_sha1": row["oid"]}
                if row.get("lfs"):
                    item["sha256"] = row["lfs"]["oid"]
                files.append(item)
            url = next((x.split(">")[0].strip().lstrip("<") for x in link.split(",")
                        if 'rel="next"' in x), None)
        records.append({"repository": "ai-habitat/" + name, "tag": tag,
                        "revision": revision, "destination": destination, "files": files})
    MANIFEST.write_bytes(canonical({"version": 1, "sources": records}) + b"\n")
    print(json.dumps({"files": sum(len(x["files"]) for x in records),
                      "bytes": sum(r["bytes"] for x in records for r in x["files"])}), flush=True)


def digest(path, row):
    h = hashlib.sha256() if "sha256" in row else hashlib.sha1()
    if "sha256" not in row:
        h.update(b"blob " + str(row["bytes"]).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch():
    records = json.loads(MANIFEST.read_text())["sources"]
    jobs = [(repo, row) for repo in records for row in repo["files"]]
    def one(repo, row):
        destination = (DATA / repo["destination"] / row["path"]).resolve()
        if not destination.is_relative_to(DATA):
            raise ValueError("asset_path_escaped")
        expected = row.get("sha256", row["git_blob_sha1"])
        if destination.is_file():
            if destination.stat().st_size != row["bytes"] or digest(destination, row) != expected:
                raise ValueError("existing_asset_identity_changed")
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        url = f'https://huggingface.co/datasets/{repo["repository"]}/resolve/{repo["revision"]}/'
        url += urllib.parse.quote(row["path"])
        with request(url) as response, partial.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if partial.stat().st_size != row["bytes"] or digest(partial, row) != expected:
            raise ValueError("downloaded_asset_identity_changed")
        partial.replace(destination)
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(one, repo, row) for repo, row in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 25 == 0 or count == len(jobs):
                print(json.dumps({"verified_files": count, "total": len(jobs)}), flush=True)
    archive = ROOT / "cache/habitat-official-assets.tar"
    with tarfile.open(archive, "w") as tar:
        for repo, row in jobs:
            relative = Path(repo["destination"]) / row["path"]
            tar.add(DATA / relative, arcname=str(Path("data") / relative))
        tar.add(MANIFEST, arcname="habitat_assets.json")
    h = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    receipt = {"status": "ready", "verified_files": len(jobs), "archive_sha256": h.hexdigest(),
               "archive_bytes": archive.stat().st_size, "model_requests": 0}
    (ROOT / "cache/habitat-official-assets-receipt.json").write_bytes(canonical(receipt) + b"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("catalog", "fetch"))
    args = parser.parse_args()
    catalog() if args.stage == "catalog" else fetch()
