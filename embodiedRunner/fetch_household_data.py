"""Fetch three ALFRED trajectories and the official Habitat-Lab source."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tarfile
import urllib.parse
import urllib.request

from protocol import ROOT, sha256


def download(url):
    return urllib.request.urlopen(url, timeout=120).read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("alfred", "habitat-lab"))
    args = parser.parse_args()
    if args.kind == "alfred":
        repo = "EmbodiedBench/EB-ALFRED"
        info = json.loads(download("https://huggingface.co/api/datasets/" + repo))
        revision = info["sha"]
        split = json.loads((ROOT / "upstream-household/embodiedbench/envs/eb_alfred/data/splits/splits.json").read_text())
        records = []
        for task in split["base"][:3]:
            path = task["task"] + "/pp/ann_" + str(task["repeat_idx"]) + ".json"
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/" + urllib.parse.quote(path)
            data = download(url)
            json.loads(data)
            destination = ROOT / "upstream-household/embodiedbench/envs/eb_alfred/data/json_2.1.0" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            records.append({"path": path, "sha256": sha256(data), "bytes": len(data)})
        (ROOT / "alfred_pilot_data.json").write_text(json.dumps({
            "repository": repo, "revision": revision, "subset": "base",
            "indexes": [0, 1, 2], "files": records}, indent=2) + "\n")
        print(json.dumps({"fetched": len(records), "revision": revision}))
    else:
        env = {k: v for k, v in os.environ.items() if not any(
            s in k.upper() for s in ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL"))}
        env["GH_TOKEN"] = "public-source-fetch"
        command = ["gh", "api", "repos/facebookresearch/habitat-lab/tarball/v0.3.0",
                   "-H", "Authorization: anonymous"]
        data = subprocess.run(command, env=env, capture_output=True, check=True, timeout=180).stdout
        archive = ROOT / "cache/habitat-lab-v0.3.0.tar.gz"
        archive.write_bytes(data)
        destination = ROOT / "runtime/habitat-lab-v0.3.0"
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            for member in tar:
                path = member.name.partition("/")[2]
                if not path or not member.isfile():
                    continue
                target = (destination / path).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("archive_path_escaped")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
        (ROOT / "habitat_lab_source.json").write_text(json.dumps({
            "repository": "facebookresearch/habitat-lab", "tag": "v0.3.0",
            "archive_sha256": sha256(data), "archive_bytes": len(data)}, indent=2) + "\n")
        print(json.dumps({"habitat_lab_bytes": len(data)}))


if __name__ == "__main__":
    main()
