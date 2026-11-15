#!/usr/bin/env python3
"""Read paired evaluation progress without creating or resubmitting requests."""

import argparse
from collections import Counter
import ctypes
import json
import os
from pathlib import Path
import select
import time
import urllib.request


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wait-change", type=float, default=0)
    args = parser.parse_args()
    root = args.run_root
    if args.wait_change:
        assert 0 < args.wait_change <= 45
        library = ctypes.CDLL(None, use_errno=True)
        fd = library.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
        assert fd >= 0
        candidates = (root, root / "system/preflight", root / "system/results",
                      root / "direct/responses", root / "requests")
        try:
            for path in candidates:
                if path.is_dir():
                    assert library.inotify_add_watch(fd, os.fsencode(path), 0x100 | 0x80 | 0x08) >= 0
            select.select([fd], [], [], args.wait_change)
        finally:
            os.close(fd)
    manifest = read(root / "diagnostic.json")
    state = read(root / "requests/state.json", {})
    results = [read(p) for p in (root / "system/results").glob("*.json")]
    direct = [read(p) for p in (root / "direct/responses").glob("*.json")]
    value = {
        "observed_at": time.time(), "n_pairs": manifest["n_questions"],
        "preflights": len(list((root / "system/preflight").glob("*.json"))),
        "preflight_summary": read(root / "system/preflight_summary.json"),
        "system_completed": len(results), "direct_completed": len(direct),
        "system_by_task": dict(Counter(r["question_type"] for r in results)),
        "direct_by_task": dict(Counter(r["question_type"] for r in direct)),
        "system_statuses": dict(Counter(r["status"] for r in results)),
        "direct_statuses": dict(Counter(r["status"] for r in direct)),
        "request_statuses": dict(Counter(r["status"] for r in state.get("requests", []))),
        "request_count": len(state.get("requests", [])), "peak_inflight": state.get("peak_inflight"),
        "pair_started": (root / "pair_start.json").is_file(),
        "exit": read(root / "exit.json"),
    }
    for name in ("preflight.log", "launcher.log", "system.log", "direct.log"):
        path = root / name
        if path.is_file():
            lines = path.read_text().splitlines()
            if lines:
                try:
                    last = json.loads(lines[-1])
                except ValueError:
                    pass
                else:
                    if last.get("status") == "stopped":
                        value[name] = last
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(manifest["runtime"]["perception_endpoint"] + "/health", timeout=10) as response:
            health = json.load(response)
        value["gateway"] = {k: health[k] for k in ("status", "max_concurrency", "upstream")}
    except Exception as exc:
        value["gateway_error_type"] = type(exc).__name__
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
