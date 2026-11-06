"""Include the MMSI package in deployment and immutable source identities."""
from __future__ import annotations

from pathlib import Path

from ..traces import file_digest


def source_files(directory):
    return tuple(sorted(
        path for path in Path(directory).rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    ))


def preflight_identity(root, dataset):
    root = Path(root)
    files = [root / name for name in ("adapters.py", "worker.py")]
    if dataset == "mmsi":
        files.extend(source_files(root / "MMSIGPT"))
    return {str(path.relative_to(root)): file_digest(path) for path in files}


def add_package(archive, directory):
    directory = Path(directory)
    for path in sorted(directory.rglob("*")):
        if (
            path.is_file()
            and path.suffix in {".py", ".md"}
            and "__pycache__" not in path.parts
        ):
            archive.add(path, arcname=str(Path("MMSIGPT") / path.relative_to(directory)))
