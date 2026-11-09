"""Stage the verified original GitHub archive without inventing a Git checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
ARCHIVE_SHA256 = "4fd4859991523b2fb0469c346382f5b7b02b69c2f57c05761acf87f747918508"
ORIGIN = "github.com/Lifelong-Robot-Learning/LIBERO"
RECEIPT = ".official-archive.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stage(archive: Path, root: Path):
    archive = archive.resolve()
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Archive does not match the verified official GitHub download.")
    if root.exists():
        raise ValueError("Choose a new official source directory.")
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        prefix = members[0].name.rstrip("/") + "/"
        files = {}
        destinations = set()
        for member in members:
            if member.name.rstrip("/") == prefix.rstrip("/") and member.isdir():
                continue
            if not member.name.startswith(prefix) or not (member.isfile() or member.isdir()):
                raise ValueError("Unexpected archive path or special/link member.")
            relative = PurePosixPath(member.name[len(prefix):])
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("Unsafe archive path.")
            name = str(relative)
            if name in destinations or name == RECEIPT:
                raise ValueError("Duplicate or reserved archive path.")
            destinations.add(name)
        root.mkdir(parents=True, exist_ok=False)
        for member in members:
            if not member.isfile():
                continue
            name = str(PurePosixPath(member.name[len(prefix):]))
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, path.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            files[name] = sha256(path)
    receipt = {"origin": ORIGIN, "commit": COMMIT, "archive": str(archive),
               "archive_sha256": ARCHIVE_SHA256,
               "url": f"https://api.github.com/repos/Lifelong-Robot-Learning/LIBERO/tarball/{COMMIT}",
               "files": files}
    (root / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"staged": True, "commit": COMMIT, "source_files": len(files)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage(args.archive, args.output.resolve())
