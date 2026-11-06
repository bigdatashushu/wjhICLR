"""Download only the three pinned development-episode files; never unpickle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

from protocol import ROOT, canonical, sha256

REVISION = "4b6c210a3c7539330c2f0d228e778a9400061025"
EPISODE = "data/base/eval/pick_cube_shape/variation0/episodes/episode0"
FILES = [
    ("configs.pkl", 7870709, "02c16b7498fab557072371592f366537c4e7c17b",
     "83bd68f84ba5ae25af4a1aa2595557b732f7bf2227717dcfd0e5d86585731c62"),
    ("task_base.ttm", 111010, "4794ab374623f29fa647bf92855e33a2cc51375d", None),
    ("waypoint_sets.ttm", 30542, "c664a33f175498f4af8b334ea647ee90d7645bfc", None),
]


def main():
    destination = ROOT / "cache/manipulation-assets"
    rows = []
    for name, size, git_blob, lfs in FILES:
        relative = EPISODE + "/" + name
        path = destination / relative
        if path.exists():
            data = path.read_bytes()
        else:
            url = f"https://huggingface.co/datasets/EmbodiedBench/EB-Manipulation/resolve/{REVISION}/{relative}"
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read(size + 1)
        if len(data) != size:
            raise ValueError("manipulation_asset_size_changed")
        digest = sha256(data)
        if lfs:
            if digest != lfs:
                raise ValueError("manipulation_asset_lfs_changed")
        elif hashlib.sha1(b"blob " + str(size).encode() + b"\0" + data).hexdigest() != git_blob:
            raise ValueError("manipulation_asset_blob_changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        rows.append({"path": relative, "sha256": digest, "bytes": size, "git_blob_sha": git_blob})
    catalog = {
        "schema": "embodied-manipulation-assets/1", "repository": "EmbodiedBench/EB-Manipulation",
        "revision": REVISION, "files": rows,
        "selection": [{"subset": "base", "index": 0, "runtime_index": 0,
                       "task_variation": "pick_cube_shape", "episode_directory": EPISODE}],
        "reference_answer_visible": False,
        "scope": "development episode only; simulator-private files are not deserialized on host",
    }
    path = ROOT / "manipulation_assets.json"
    wire = canonical(catalog) + b"\n"
    if path.exists() and path.read_bytes() != wire:
        raise ValueError("manipulation_asset_catalog_changed")
    path.write_bytes(wire)
    print(json.dumps({"files": len(rows), "bytes": sum(r["bytes"] for r in rows),
                      "catalog_sha256": sha256(canonical(catalog)), "unpickle_calls": 0}))


if __name__ == "__main__":
    main()
