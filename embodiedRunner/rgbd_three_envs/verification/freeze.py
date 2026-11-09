"""Freeze this independent version and preserve previously completed artifacts."""
import ast
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rgbd_three_envs.__main__ import check
from rgbd_three_envs.config import load_config
from rgbd_three_envs.sources import Sources, canonical, file_digest, implementation_hashes

PACKAGE = ROOT / "rgbd_three_envs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="r01", choices=("r01", "r02"))
    args = parser.parse_args()
    suffix = "" if args.revision == "r01" else "-" + args.revision
    receipt = PACKAGE / f"verification/frozen-manifest{suffix}.json"
    if receipt.exists():
        raise ValueError("already_frozen")
    config_path = PACKAGE / f"experiment-20261107-{args.revision}.json"
    config, tasks = load_config(config_path)
    sources = Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
    offline = check(config, tasks, sources)
    for path in PACKAGE.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 9))
    parent = json.loads((PACKAGE / "verification/parent-identity.json").read_text())
    assert all(file_digest(ROOT / "rgbd_trial" / name) == sha
               for name, sha in parent["implementation_hashes"].items())
    assert parent["source_hashes"] == sources.hashes
    preserved = {}
    for old in ("official_direct", "manipulation_minimal", "rgbd_trial"):
        for path in sorted((ROOT / old).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.name != ".DS_Store":
                preserved[str(path.relative_to(ROOT))] = file_digest(path)
    # Include earlier legacy-baseline receipts already selected by the Navigation audit.
    legacy = json.loads((ROOT / "rgbd_trial/verification/preserved-runs.json").read_text())
    for name, sha in legacy.items():
        assert file_digest(ROOT / name) == sha
        preserved[name] = sha
    (PACKAGE / f"verification/preserved-files{suffix}.json").write_bytes(canonical(preserved) + b"\n")
    (PACKAGE / f"verification/offline-check{suffix}.json").write_bytes(canonical(offline) + b"\n")
    frozen = {"implementation_hashes": implementation_hashes(), "source_hashes": sources.hashes,
              "config_sha256": file_digest(config_path),
              "python39_syntax_pass": True, "old_files_preserved": len(preserved),
              "selected_tasks": [t.as_dict() for t in tasks], "model_requests": 0,
              "simulator_launches": 0}
    receipt.write_bytes(canonical(frozen) + b"\n")
    print(json.dumps({"status": "frozen", "old_files_preserved": len(preserved),
                      "implementation_files": len(frozen["implementation_hashes"])}))


if __name__ == "__main__":
    main()
