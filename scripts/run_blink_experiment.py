#!/usr/bin/env python3
"""BLINK campaign. Default: full input/cache preflight with zero model requests."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main(argv=None):
    from skill3d.evaluation.blink_experiment import (
        Campaign, STAGES, create_run, file_sha, preflight, profile_at, read,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--frozen-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.frozen_child:
        if not (args.execute or args.check) or not args.output_dir:
            parser.error("frozen child requires --execute or --check and run directory")
        root = args.output_dir.resolve()
        if REPO != root/"code/source":
            raise ValueError("must_execute_from_run_frozen_source")
        manifest = read(root/"manifest.json")
        with (root/"operator.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            campaign = Campaign(root, manifest)
            # Preflight replays the full public cache identity on every resume.
            if preflight(manifest["profile"]) != manifest["preflight"]:
                raise ValueError("frozen_input_preflight_changed")
            campaign.verify()
            if args.check:
                print(json.dumps({"status": "ready", "questions": len(campaign.problems),
                                  "model_requests": 0, "reference_answers_read": False,
                                  "frozen_source_verified": True}))
                return 0
            completion = root/"completion.json"
            if completion.exists():
                result = read(completion)
                if result["campaign_sha256"] != manifest["content_sha256"]:
                    raise ValueError("completion_campaign_changed")
                if any(file_sha(root/p) != h for p, h in result["result_files"].items()):
                    raise ValueError("completed_results_changed")
            else:
                result = campaign.run()
        print(json.dumps({"status": result["status"], "scores": result["scores"],
                          "report": str(completion)}, ensure_ascii=False))
        return 0
    if args.resume:
        if not args.output_dir:
            parser.error("--resume requires --output-dir")
        root = args.output_dir.resolve()
        manifest = read(root/"manifest.json")
        # Saved source/profile govern resume, even if the development checkout changed.
        if args.config is not None or args.data_root is not None:
            profile = profile_at(args.config or REPO/"configs/experiments/blink.json", args.data_root)
            if args.max_requests is None:
                profile["runtime"]["max_requests"] = manifest["profile"]["runtime"]["max_requests"]
            if profile != manifest["profile"]:
                raise ValueError("resume_profile_changed")
        if args.max_requests is not None and args.max_requests != manifest["profile"]["runtime"]["max_requests"]:
            raise ValueError("resume_request_budget_changed")
        if args.dry_run:
            print(json.dumps({"status": "planned", "stages": STAGES,
                              "profile": manifest["profile"], "model_requests": 0,
                              "source": str(root/"code/source")}, ensure_ascii=False, indent=2))
            return 0
        mode = "--execute" if args.execute else "--check"
        command = [sys.executable, "-B", str(root/"code/source/scripts/run_blink_experiment.py"),
                   mode, "--frozen-child", "--output-dir", str(root)]
        return subprocess.run(command, check=False).returncode
    profile = profile_at(args.config or REPO/"configs/experiments/blink.json", args.data_root)
    if args.max_requests is not None:
        if args.max_requests < 1:
            parser.error("--max-requests must be positive")
        profile["runtime"]["max_requests"] = args.max_requests
    if args.dry_run:
        print(json.dumps({"status": "planned", "stages": STAGES, "profile": profile,
                          "model_requests": 0, "source_clean_required_for_execute": True},
                         ensure_ascii=False, indent=2))
        return 0
    inspection = preflight(profile)
    if not args.execute:
        print(json.dumps({"status": "ready", "questions": len(inspection["problems"]),
                          "preflight_sha256": inspection["content_sha256"],
                          "model_requests": 0, "reference_answers_read": False,
                          "stages": STAGES}, ensure_ascii=False))
        return 0
    if not args.output_dir:
        parser.error("--execute requires a new --output-dir and explicit request budget")
    root = args.output_dir.absolute()
    # A new formal run is frozen only from committed, clean development source.
    create_run(REPO, root, profile, inspection)
    command = [sys.executable, "-B", str(root/"code/source/scripts/run_blink_experiment.py"),
               "--execute", "--frozen-child", "--output-dir", str(root)]
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Do not print provider exception text or credentials.
        print(json.dumps({"status": "stopped", "error_type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(2)
