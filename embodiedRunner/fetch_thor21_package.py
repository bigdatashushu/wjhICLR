"""Inspect and freeze the official AI2-THOR 2.1 package without installing it."""
from __future__ import annotations

import io
import json
import tarfile
import urllib.request
import zipfile

from protocol import ROOT, sha256


def main():
    info = json.load(urllib.request.urlopen("https://pypi.org/pypi/ai2thor/2.1.0/json", timeout=45))
    row = next(x for x in info["urls"] if x["filename"].endswith(".whl"))
    data = urllib.request.urlopen(row["url"], timeout=120).read()
    if sha256(data) != row["digests"]["sha256"]:
        raise ValueError("package_identity_mismatch")
    destination = ROOT / "runtime/ai2thor-2.1-source"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            if name.startswith("ai2thor/") and name.endswith(".py"):
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
    (ROOT / "cache" / row["filename"]).write_bytes(data)
    (destination / "package.json").write_text(json.dumps({
        "filename": row["filename"], "sha256": sha256(data), "bytes": len(data)}, indent=2) + "\n")
    print(json.dumps({"package_bytes": len(data)}))


if __name__ == "__main__":
    main()
