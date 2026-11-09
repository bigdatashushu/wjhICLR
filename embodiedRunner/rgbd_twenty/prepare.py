"""Bind the original twenty tasks to the two already accepted RGB-D packages."""
import ast
import importlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rgbd_three_envs.sources import canonical, file_digest

REMOTE = "/home/cvailab/wangjh/embodied-rgbd-twenty-20261107-r01/embodiedRunner"
KINDS = ("navigation", "alfred", "habitat", "manipulation")


def main():
    assert not (OUT / "frozen.json").exists(), "already_prepared"
    baseline = json.loads((ROOT / "official_direct/baseline-five-results-20261107.json").read_text())
    selected = [{"environment": k, "subset": "base", "index": i} for k in KINDS for i in range(5)]
    assert baseline["completed"] == 20
    assert [r["task"] for r in baseline["results"]] == selected
    configurations, packages, offline = {}, {}, {}
    for package, revision, environments in (
        ("rgbd_trial", "r01", ("navigation",)),
        ("rgbd_three_envs", "r02", KINDS[1:]),
    ):
        source = ROOT / package / f"experiment-20261107-{revision}.json"
        config = json.loads(source.read_text())
        config["episodes"] = [t for t in selected if t["environment"] in environments]
        config["runtime"]["code_root"] = REMOTE
        config["runtime"]["work_root"] = REMOTE + f"/{package}/runs"
        config["execution"]["max_episodes"] = 1
        config["execution"]["per_environment"] = {k: 1 for k in KINDS}
        name = f"config-{package}.json"
        (OUT / name).write_bytes(canonical(config) + b"\n")
        sources_module = importlib.import_module(package + ".sources")
        sources = sources_module.Sources(manipulation_catalog=ROOT / "official_direct/assets/manipulation_assets.json")
        checked_config, tasks = importlib.import_module(package + ".config").load_config(OUT / name)
        offline[package] = importlib.import_module(package + ".__main__").check(checked_config, tasks, sources)
        suffix = "" if revision == "r01" else "-r02"
        previous = json.loads((ROOT / package / f"verification/frozen-manifest{suffix}.json").read_text())
        assert sources_module.implementation_hashes() == previous["implementation_hashes"]
        assert sources.hashes == previous["source_hashes"]
        assert config["model"] == json.loads((ROOT / "official_direct/baseline-20261107-r03.json").read_text())["model"]
        for path in (ROOT / package).glob("*.py"):
            ast.parse(path.read_text(), feature_version=(3, 9))
        packages[package] = {
            "implementation_hashes": sources_module.implementation_hashes(),
            "source_hashes": sources.hashes,
            "config": name, "config_sha256": file_digest(OUT / name),
            "run_id": "rgbd-twenty-20261107-r01",
            "episodes": config["episodes"],
        }
        configurations[package] = config
    preserved = {}
    for directory in ("official_direct", "rgbd_trial", "rgbd_three_envs", "manipulation_minimal"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.name != ".DS_Store":
                preserved[str(path.relative_to(ROOT))] = file_digest(path)
    for relative, sha in json.loads((ROOT / "rgbd_trial/verification/preserved-runs.json").read_text()).items():
        assert file_digest(ROOT / relative) == sha
        preserved[relative] = sha
    (OUT / "preserved.json").write_bytes(canonical(preserved) + b"\n")
    frozen = {
        "schema": "rgbd-twenty-frozen/1", "selected": selected,
        "packages": packages, "baseline_sha256": file_digest(ROOT / "official_direct/baseline-five-results-20261107.json"),
        "remote_root": REMOTE, "preserved_old_files": len(preserved),
        "scheduler": "Navigation then ALFRED then Habitat then Manipulation; one active episode",
        "automatic_retry": False, "reference_answer_visible": False,
        "model_requests": 0, "simulator_launches": 0,
    }
    (OUT / "offline-check.json").write_bytes(canonical(offline) + b"\n")
    (OUT / "frozen.json").write_bytes(canonical(frozen) + b"\n")
    with tarfile.open(OUT / "deployment.tar.gz", "w:gz") as archive:
        for package in packages:
            for path in sorted((ROOT / package).glob("*.py")):
                archive.add(path, arcname=str(path.relative_to(ROOT)))
        for name in ("frozen.json", "config-rgbd_trial.json", "config-rgbd_three_envs.json", "remote_audit.py"):
            archive.add(OUT / name, arcname="rgbd_twenty/" + name)
    print(json.dumps({"status": "frozen", "selected": 20, "preserved_old_files": len(preserved),
                      "deployment_bytes": (OUT / "deployment.tar.gz").stat().st_size,
                      "model_requests": 0, "simulator_launches": 0}))


if __name__ == "__main__":
    main()
