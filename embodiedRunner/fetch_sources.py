"""Download a pinned, small selection of official sources for feasibility analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REVISION = "9be4e980e9cd6bcb38373cd4aab7c32724bdd401"
REPOSITORY = "EmbodiedBench/EmbodiedBench"
PATHS = [
    "README.md",
    "Docker/Dockerfile",
    "conda_envs/environment.yaml",
    "conda_envs/environment_eb-nav.yaml",
    "conda_envs/environment_eb-man.yaml",
    "embodiedbench/main.py",
    "embodiedbench/configs/config.yaml",
    "embodiedbench/configs/eb-alf.yaml",
    "embodiedbench/configs/eb-hab.yaml",
    "embodiedbench/configs/eb-nav.yaml",
    "embodiedbench/configs/eb-man.yaml",
    "embodiedbench/envs/eb_alfred/EBAlfEnv.py",
    "embodiedbench/envs/eb_alfred/README.md",
    "embodiedbench/envs/eb_habitat/EBHabEnv.py",
    "embodiedbench/envs/eb_navigation/EBNavEnv.py",
    "embodiedbench/envs/eb_manipulation/EBManEnv.py",
    "embodiedbench/envs/eb_manipulation/eb_man_utils.py",
    "embodiedbench/evaluator/eb_alfred_evaluator.py",
    "embodiedbench/evaluator/eb_habitat_evaluator.py",
    "embodiedbench/evaluator/eb_navigation_evaluator.py",
    "embodiedbench/evaluator/eb_manipulation_evaluator.py",
    "embodiedbench/evaluator/summarize_result.py",
    "embodiedbench/evaluator/config/system_prompts.py",
    "embodiedbench/planner/custom_model.py",
    "embodiedbench/planner/remote_model.py",
    "embodiedbench/planner/nav_planner.py",
    "embodiedbench/planner/manip_planner.py",
    "embodiedbench/planner/vlm_planner.py",
    "embodiedbench/planner/planner_config/generation_guide.py",
    "embodiedbench/planner/planner_config/generation_guide_manip.py",
]
NAVIGATION_DATA = [
    f"embodiedbench/envs/eb_navigation/datasets/{subset}.json"
    for subset in ("base", "common_sense", "complex_instruction",
                   "visual_appearance", "long_horizon")
]
PATHS += NAVIGATION_DATA
NAVIGATION_RUNTIME = ["embodiedbench/envs/eb_navigation/utils.py"]
PATHS += NAVIGATION_RUNTIME
NAVIGATION_BASELINE = [
    "embodiedbench/planner/planner_utils.py",
    "embodiedbench/evaluator/config/eb_navigation_example.py",
]
PATHS += NAVIGATION_BASELINE


def fetch(paths: list[str]) -> None:
    tree = json.loads((ROOT / "upstream_tree.json").read_text(encoding="utf-8"))
    if tree["truncated"]:
        raise RuntimeError("Incomplete official source index")
    entries = {entry["path"]: entry for entry in tree["tree"]}
    # Public GitHub requests need no real token. A nonempty anonymous header
    # prevents gh from injecting its automation token into these public requests.
    environment = {
        key: value for key, value in os.environ.items()
        if not any(part in key.upper() for part in
                   ("TOKEN", "API_KEY", "PASSWORD", "SECRET", "CREDENTIAL"))
    }
    environment["GH_TOKEN"] = "public-source-fetch"
    for name in paths:
        entry = entries[name]
        if entry["type"] != "blob" or entry["size"] > 150_000:
            raise ValueError(f"Source selection is not a small text file: {name}")
        destination = ROOT / "upstream" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            data = destination.read_bytes()
        else:
            result = subprocess.run(
                ["gh", "api", f"repos/{REPOSITORY}/git/blobs/{entry['sha']}",
                 "-H", "Authorization: anonymous", "-H",
                 "Accept: application/vnd.github.raw+json"],
                check=True, capture_output=True, env=environment, timeout=45,
            )
            data = result.stdout
        blob_hash = hashlib.sha1(
            b"blob " + str(len(data)).encode("ascii") + b"\0" + data
        ).hexdigest()
        if blob_hash != entry["sha"]:
            raise ValueError(f"Official Git blob identity mismatch: {name}")
        data.decode("utf-8")
        destination.write_bytes(data)
        print(f"Fetched {name} ({len(data)} bytes)", flush=True)


def manifest() -> None:
    tree = json.loads((ROOT / "upstream_tree.json").read_text(encoding="utf-8"))
    entries = {entry["path"]: entry for entry in tree["tree"]}
    records = []
    for name in PATHS:
        data = (ROOT / "upstream" / name).read_bytes()
        records.append({
            "path": name, "git_blob_sha": entries[name]["sha"],
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "url": f"https://github.com/{REPOSITORY}/blob/{REVISION}/{name}",
        })
    (ROOT / "official_sources.json").write_text(
        json.dumps({"repository": REPOSITORY, "revision": REVISION,
                    "files": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Verified {len(records)} official files; "
          f"{sum(record['bytes'] for record in records)} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, choices=range(4))
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--navigation", action="store_true")
    parser.add_argument("--navigation-runtime", action="store_true")
    parser.add_argument("--navigation-baseline", action="store_true")
    args = parser.parse_args()
    if args.manifest:
        manifest()
    elif args.navigation_baseline:
        fetch(NAVIGATION_BASELINE)
        manifest()
    elif args.navigation_runtime:
        fetch(NAVIGATION_RUNTIME)
        manifest()
    elif args.navigation:
        fetch(NAVIGATION_DATA)
        manifest()
    elif args.group is not None:
        fetch(PATHS[args.group::4])
    else:
        fetch(PATHS)
        manifest()
