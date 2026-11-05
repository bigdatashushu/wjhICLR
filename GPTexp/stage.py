"""Collect only frozen public code/configuration from the authorized Titan."""
from __future__ import annotations

from pathlib import Path
import subprocess
import tarfile

from .traces import file_digest, write

ROOT = Path(__file__).resolve().parent
BASE = "/home/wangy/wangjh/harness3d/four-datasets-20261031"
SSH_CONFIG = "/Users/bytedance/wangjh/.trae/skills/harness3d-cluster-ops/assets/ssh_config"
SOURCES = {
    "blink-multiview": ("blink-paired93-20261104-r01",
                        ["code/source", "driver", "manifest.json"]),
    "cv3d": ("cv3d-box-review840-20261101-r01",
             ["source_frozen", "manifest.json", "run_config.json", "build.json"]),
    "mmsi": ("mmsi-365-test-rerun-20261101",
             ["source_frozen", "pilot.py", "manifest.json", "frozen_skills.json", "score.py"]),
    "mindcube-tiny": ("mindcube-tiny-test700-paired-20261104-r01",
                      ["source", "selection.json", "run.config.json"]),
    "blink-sr": ("blink-sr-paired20-20261104-r03",
                 ["source/src", "source/scripts", "source/BLINK/download_data.py",
                  "manifest.json", "split.json", "delivery_files.json"]),
}


def main():
    frozen = ROOT / "frozen"
    frozen.mkdir(exist_ok=True)
    receipts = {}
    for name, (run, files) in SOURCES.items():
        target = frozen / name
        target.mkdir(exist_ok=True)
        archive = target / "source.tar.gz"
        if not archive.exists():
            with archive.with_suffix(".partial").open("xb") as stream:
                subprocess.run(
                    ["ssh", "-F", SSH_CONFIG, "-T", "-o", "BatchMode=yes", "-o", "ControlMaster=no",
                     "h3d-titan", "tar", "--exclude=__pycache__", "--exclude=*.pyc", "-czf", "-",
                     "-C", BASE + "/" + run, *files],
                    stdout=stream, check=True, timeout=36000)
            archive.with_suffix(".partial").rename(archive)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                resolved = (target / member.name).resolve()
                if not resolved.is_relative_to(target.resolve()) or member.issym() or member.islnk():
                    raise ValueError("archive_path_not_public_regular_file")
            tar.extractall(target)
        hashes = {str(p.relative_to(target)): file_digest(p) for p in target.rglob("*")
                  if p.is_file() and p != archive}
        receipts[name] = {"origin": BASE + "/" + run, "archive_sha256": file_digest(archive),
                          "files": hashes}
        print("Staged", name, len(hashes), "files", flush=True)
    write(frozen / "sources.json", receipts)


if __name__ == "__main__":
    main()
