from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import uuid

from .config import ROOT, RunnerError


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":")
    ).encode()).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temp.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_identity() -> dict:
    paths = list(ROOT.glob("*.py")) + list((ROOT / "scripts").glob("*"))
    paths += list(ROOT.glob("*.txt"))
    paths += list((ROOT.parent / "xcodeapi").glob("*.py"))
    return {str(p.resolve()): file_digest(p) for p in sorted(paths) if p.is_file()}


def verify_files(identity: dict):
    for name, expected in identity.items():
        p = Path(name)
        if not p.is_file() or file_digest(p) != expected:
            raise RunnerError(f"Frozen input or source changed: {p.name}")


class Journal:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: dict):
        record = {"time_utc": datetime.now(timezone.utc).isoformat(), **event}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
