"""MMSI-only command entry using GPTexp's shared preparation and scheduler."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare or execute an MMSI GPT run.")
    parser.add_argument("action", choices=["prepare", "prepare-diagnostic", "execute"])
    parser.add_argument("--run", required=True, help="Independent run directory.")
    parser.add_argument("--config", help="GPTexp config for the MMSI-only request budget.")
    parser.add_argument("--source-run", help="Existing label-free MMSI source run.")
    parser.add_argument("--stage", choices=["first", "remaining", "all"], default="all")
    parser.add_argument("--per-type", type=int, default=5)
    parser.add_argument("--selection-seed", type=int, default=137)
    args = parser.parse_args(argv)
    if args.action == "prepare" and not args.config:
        parser.error("prepare requires --config")
    if args.action == "prepare-diagnostic" and not args.source_run:
        parser.error("prepare-diagnostic requires --source-run")

    from ..run import execute, prepare
    from ..traces import read

    directory = Path(args.run)
    if args.action == "prepare":
        prepare(directory, read(args.config), datasets=["mmsi"])
    elif args.action == "prepare-diagnostic":
        from .diagnostic import prepare_diagnostic
        prepare_diagnostic(directory, args.source_run, per_type=args.per_type,
                           seed=args.selection_seed, stage=args.stage)
    else:
        manifest = read(directory / "manifest.json")
        if (
            set(manifest.get("counts", {})) != {"mmsi"}
            or not manifest.get("pairs")
            or any(pair.get("dataset") != "mmsi" for pair in manifest["pairs"])
        ):
            raise ValueError("mmsigpt_requires_mmsi_only_manifest")
        diagnostic = manifest.get("diagnostic_only") is True
        if diagnostic:
            from .diagnostic import validate_diagnostic
            validate_diagnostic(manifest)
        execute(directory, score_results=not diagnostic)


if __name__ == "__main__":
    main()
