#!/usr/bin/env python3
"""Stream completed paired artifacts and their byte identities as a portable tar."""

import argparse
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import select
import sys
import tarfile
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=3600)
    parser.add_argument("--allow-stopped", action="store_true",
                        help="Collect infrastructure-failed runs without treating them as complete.")
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.wait:
        library = ctypes.CDLL(None, use_errno=True)
        watch_fd = library.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
        assert watch_fd >= 0
        assert library.inotify_add_watch(watch_fd, os.fsencode(root), 0x100 | 0x80 | 0x08) >= 0
        deadline = time.monotonic() + args.timeout_s
        try:
            while not (root / "exit.json").is_file():
                assert time.monotonic() < deadline, "paired run completion timeout"
                for name in ("preflight.log", "launcher.log"):
                    path = root / name
                    lines = path.read_text().splitlines() if path.is_file() else []
                    if lines:
                        try:
                            value = json.loads(lines[-1])
                        except ValueError:
                            continue
                        assert value.get("status") != "stopped", "paired run stopped before completion"
                ready, _, _ = select.select([watch_fd], [], [], 30)
                if ready:
                    os.read(watch_fd, 65536)
        finally:
            os.close(watch_fd)
    exit_status = json.loads((root / "exit.json").read_text())["status"]
    assert exit_status == "complete" or args.allow_stopped and exit_status == "stopped"
    manifest = json.loads((root / "diagnostic.json").read_text())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(manifest["runtime"]["perception_endpoint"] + "/health", timeout=10) as response:
        final_health = json.load(response)
    patterns = (
        "*_start.json", "*_exit.json", "*_receipt.json", "*_health.json",
        "exit.json", "execution_note.json", "*.log",
        "system/manifest.json", "system/summary.json", "system/preflight_summary.json",
        "system/episode_pool_receipt.json", "system/results/*.json",
        "system/preflight/*.json", "system/request_seeds/*.json",
        "system/variants/*/plan.json", "system/variants/*/outcome.json",
        "system/variants/*/trace/*.jsonl", "requests/state.json", "direct/**/*.json",
    )
    paths = sorted({p for pattern in patterns for p in root.glob(pattern) if p.is_file()})
    identities = {}
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
        for path in paths:
            data = path.read_bytes()
            relative = str(path.relative_to(root))
            identities[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            info = tarfile.TarInfo(relative)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
        data = json.dumps(final_health, sort_keys=True, indent=2).encode()
        relative = "gateway_final_health.json"
        identities[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        info = tarfile.TarInfo(relative)
        info.size, info.mode = len(data), 0o644
        archive.addfile(info, io.BytesIO(data))
        receipt = json.dumps({
            "status": "passed", "native_root": str(root), "files": identities,
            "run_status": exit_status, "diagnostic_identity": manifest["identity_sha256"],
        }, sort_keys=True, indent=2).encode()
        info = tarfile.TarInfo("collection_receipt.json")
        info.size, info.mode = len(receipt), 0o644
        archive.addfile(info, io.BytesIO(receipt))


if __name__ == "__main__":
    main()
