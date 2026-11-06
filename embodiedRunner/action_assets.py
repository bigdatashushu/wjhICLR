"""Fetch the first five ALFRED inputs at the existing immutable dataset revision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.parse
import urllib.request

from protocol import ROOT, ContractError, canonical, sha256

MANIFEST = ROOT / "action_alfred_data.json"


def fetch(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ContractError("asset_output_directory_missing")
    original = json.loads((ROOT / "alfred_pilot_data.json").read_text(encoding="utf-8"))
    from household_protocol import source
    split = json.loads(source("embodiedbench/envs/eb_alfred/data/splits/splits.json"))
    pins = {row["path"]: row for row in original["files"]}
    rows = []
    for index, task in enumerate(split["base"][:5]):
        name = task["task"] + f"/pp/ann_{task['repeat_idx']}.json"
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ContractError("asset_path")
        if name in pins:
            raw = (ROOT / "upstream-household/embodiedbench/envs/eb_alfred/data/json_2.1.0" / name).read_bytes()
            if sha256(raw) != pins[name]["sha256"]:
                raise ContractError("original_alfred_asset_changed")
        else:
            url = (f"https://huggingface.co/datasets/{original['repository']}/resolve/"
                   f"{original['revision']}/" + urllib.parse.quote(name))
            with urllib.request.urlopen(url, timeout=120) as response:
                raw = response.read(8 * 1024**2 + 1)
            if len(raw) > 8 * 1024**2:
                raise ContractError("alfred_asset_cap")
        json.loads(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != raw:
                raise ContractError("asset_already_exists_with_different_content")
        else:
            with path.open("xb") as stream:
                stream.write(raw)
        rows.append({"index": index, "path": name, "sha256": sha256(raw), "bytes": len(raw)})
    value = {"schema": "embodied-alfred-assets/1", "repository": original["repository"],
             "revision": original["revision"], "subset": "base", "indexes": list(range(5)),
             "files": rows}
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_bytes()) != value:
            raise ContractError("alfred_asset_manifest_changed")
    else:
        with MANIFEST.open("xb") as stream:
            stream.write(canonical(value) + b"\n")
    return {"status": "ready", "indexes": value["indexes"],
            "manifest_sha256": sha256(MANIFEST.read_bytes()),
            "bytes": sum(r["bytes"] for r in rows), "model_requests": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    print(canonical(fetch(parser.parse_args().directory)).decode())
