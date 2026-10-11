#!/usr/bin/env python3
"""Prepare a formal checkout or a frozen local development source for a GPU run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import yaml

from skill3d.infra.experiment_startup import (
    COMMIT_RE, RUN_ID_RE, StartupError, configure_run, environment_contract,
    freeze_working_tree, git, run_mode, write_json,
)


def read_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}",
        lambda match: os.environ.get(match.group(1), match.group(2) or ""), text)
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise StartupError("GPU configuration must be a YAML mapping")
    return cfg


def prepare(args) -> dict:
    root = Path(args.source_root).expanduser().resolve()
    original_root = root
    work = Path(args.work_dir).expanduser().resolve()
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise StartupError("Run ID may contain only letters, digits, dot, underscore and hyphen")
    run_root = work / "runs" / args.run_id
    if run_root.exists():
        raise StartupError(f"Run directory already exists: {run_root}")
    config_path = work / "launcher_configs" / f"{args.run_id}.yaml"
    if config_path.exists():
        raise StartupError(f"Generated run config already exists: {config_path}")
    cfg = read_config(Path(args.config).expanduser().resolve())
    cfg = configure_run(cfg, dev=args.dev, smoke=args.smoke, dry_run=args.dry_run)
    for key in ("qwen_gpu", "geometry_gpu"):
        value = getattr(args, key)
        if value is not None:
            cfg["runtime"][key] = value
    if args.detector_endpoint is not None:
        cfg["experiment"]["detector_endpoint"] = args.detector_endpoint
    mode = cfg["experiment"]["run_mode"]
    commit = git(root, "rev-parse", "HEAD")
    if not COMMIT_RE.fullmatch(commit):
        raise StartupError("Source has no full Git commit")
    if args.code_ref and args.code_ref != commit:
        raise StartupError(f"Source HEAD {commit} differs from requested {args.code_ref}")
    receipt_path = None
    if mode == "development":
        receipt = freeze_working_tree(root, work / "source_snapshots" / args.run_id)
        root = Path(receipt["source_root"])
        receipt_path = root.parent / "code_identity.json"
    elif git(root, "status", "--porcelain"):
        raise StartupError("Formal run requires a clean pinned source checkout")
    work.mkdir(parents=True, exist_ok=True)
    # Resolve relative inputs against the actual chosen source; keep output paths
    # and device/budget values unchanged.
    confirmation = cfg["experiment"].get("quality_confirmation")
    if confirmation:
        path = Path(confirmation).expanduser()
        if not path.is_absolute():
            cfg["experiment"]["quality_confirmation"] = str(root / path)
    video = cfg["experiment"].get("video_root")
    if video and not Path(video).expanduser().is_absolute():
        cfg["experiment"]["video_root"] = str(original_root / video)
    for model in cfg["weights"].values():
        existing = model.get("existing_path")
        if existing and not Path(existing).expanduser().is_absolute():
            model["existing_path"] = str(original_root / existing)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    install = environment_contract(root, cfg)
    plan = {
        "source_root": str(root), "code_ref": commit, "run_mode": mode,
        "config_path": str(config_path),
        "code_receipt": str(receipt_path) if receipt_path else "",
        "environment_contract": install,
        "dry_run": args.dry_run,
    }
    write_json(work / "launcher_configs" / f"{args.run_id}.json", plan)
    return plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["mode", "prepare"], default="prepare")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--code-ref")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--qwen-gpu", type=int)
    parser.add_argument("--geometry-gpu", type=int)
    parser.add_argument("--detector-endpoint")
    args = parser.parse_args(argv)
    try:
        if args.action == "mode":
            print(run_mode(read_config(Path(args.config)), dev=args.dev,
                           smoke=args.smoke, dry_run=args.dry_run))
            return 0
        if not args.work_dir or not args.run_id:
            parser.error("--work-dir and --run-id are required when preparing a run")
        plan = prepare(args)
        # Fixed-order lines consumed by bash, without eval or shell interpolation.
        for key in ("source_root", "config_path", "code_receipt", "run_mode"):
            print(plan[key] or "-")
        return 0
    except (StartupError, OSError, ValueError, KeyError) as exc:
        print(f"GPU source preparation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
