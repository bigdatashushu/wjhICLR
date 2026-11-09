"""Prepare only the first five pinned Manipulation episodes; never deserialize pickle."""
import hashlib
import json
from pathlib import Path
import urllib.request

from .sources import ROOT, canonical, digest

REVISION = "4b6c210a3c7539330c2f0d228e778a9400061025"
PREFIX = "data/base/eval/pick_cube_shape"


def main():
    destination = ROOT / "official_direct/assets"
    if destination.exists():
        raise ValueError("independent_asset_directory_already_exists")
    destination.mkdir()
    api = f"https://huggingface.co/api/datasets/EmbodiedBench/EB-Manipulation/tree/{REVISION}/{PREFIX}?recursive=true&limit=1000"
    metadata = json.load(urllib.request.urlopen(api, timeout=120))
    # Confirm complete ordering from the pinned tree before selecting five.
    episodes = {str(Path(r["path"]).parent) for r in metadata
                if r["type"] == "file" and r["path"].endswith("/configs.pkl")}
    import re
    selected = sorted(episodes, key=lambda p: tuple(int(n) for n in re.search(
        r"variation(\d+)/episodes/episode(\d+)$", p).groups()))[:5]
    expected = [PREFIX + f"/variation{v}/episodes/episode{e}"
                for v, e in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))]
    if selected != expected:
        raise ValueError("official_episode_order_changed")
    (destination / "pinned-tree.json").write_bytes(canonical(metadata) + b"\n")
    rows = []
    for row in metadata:
        if row["type"] != "file" or str(Path(row["path"]).parent) not in selected:
            continue
        name, size = row["path"], row["size"]
        if Path(name).name not in ("configs.pkl", "task_base.ttm", "waypoint_sets.ttm"):
            continue
        cached = ROOT / "cache/manipulation-assets" / name
        if cached.exists():
            data = cached.read_bytes()
        else:
            url = f"https://huggingface.co/datasets/EmbodiedBench/EB-Manipulation/resolve/{REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read(size + 1)
        sha = digest(data)
        if len(data) != size:
            raise ValueError("asset_size_changed")
        lfs = row.get("lfs", {}).get("oid")
        blob = hashlib.sha1(b"blob " + str(size).encode() + b"\0" + data).hexdigest()
        if (lfs and sha != lfs) or (not lfs and blob != row["oid"]):
            raise ValueError("pinned_asset_hash_changed")
        path = destination / "dataset" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        rows.append({"path": name, "bytes": size, "sha256": sha, "git_blob_sha": row["oid"],
                     "lfs_sha256": lfs, "reused_local_cache": cached.exists()})
    if len(rows) != 15:
        raise ValueError("asset_set_incomplete")
    catalog = {
        "schema": "embodied-manipulation-assets/1", "repository": "EmbodiedBench/EB-Manipulation",
        "revision": REVISION, "files": rows, "reference_answer_visible": False,
        "scope": "official base indexes 0-4; private assets decoded only by simulator",
        "selection": [{"subset": "base", "index": i, "runtime_index": i,
                       "task_variation": "pick_cube_shape", "episode_directory": path}
                      for i, path in enumerate(selected)]}
    (destination / "manipulation_assets.json").write_bytes(canonical(catalog) + b"\n")
    print(json.dumps({"files": len(rows), "bytes": sum(r["bytes"] for r in rows),
                      "catalog_sha256": digest(canonical(catalog)), "unpickle_calls": 0}))


if __name__ == "__main__":
    main()
