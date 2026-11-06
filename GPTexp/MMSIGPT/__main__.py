"""MMSI-only command entry using GPTexp's shared preparation and scheduler."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare or execute an MMSI GPT run.")
    parser.add_argument("action", choices=["prepare", "execute"])
    parser.add_argument("--run", required=True, help="Independent run directory.")
    parser.add_argument("--config", help="GPTexp config for the MMSI-only request budget.")
    args = parser.parse_args(argv)
    if args.action == "prepare" and not args.config:
        parser.error("prepare requires --config")

    from ..run import execute, prepare
    from ..traces import read

    directory = Path(args.run)
    if args.action == "prepare":
        prepare(directory, read(args.config), datasets=["mmsi"])
    else:
        manifest = read(directory / "manifest.json")
        if (
            set(manifest.get("counts", {})) != {"mmsi"}
            or not manifest.get("pairs")
            or any(pair.get("dataset") != "mmsi" for pair in manifest["pairs"])
        ):
            raise ValueError("mmsigpt_requires_mmsi_only_manifest")
        execute(directory)


if __name__ == "__main__":
    main()

