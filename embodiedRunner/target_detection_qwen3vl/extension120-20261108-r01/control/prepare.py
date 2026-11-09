"""Freeze the extension, preserving the completed twenty-task experiment."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

TMP = Path(__file__).parent
ROOT = Path("/Users/bytedance/wangjh/wjhICLR/embodiedRunner")
BASE = ROOT / "target_detection_qwen3vl/twenty-20261108-r01"
OUT = ROOT / "target_detection_qwen3vl/extension120-20261108-r01"
ASSET_ROOT = "/home/cvailab/wangjh/embodied-qwen3vl-extension120-assets-20261108-r01"
sys.path.insert(0, str(BASE / "source_frozen/embodiedRunner"))
from target_detection_qwen3vl.config import load_config
from target_detection_qwen3vl.detector import Detector
from target_detection_qwen3vl.sources import Sources, canonical, file_digest, implementation_hashes
from target_detection_r05.__main__ import check
from target_detection_r05.lifecycle import remote
from target_detection_r05.transport import preflight


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def state_code():
    # Reuse the already verified process-ownership scanner without importing its main.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "old_prepare", "/private/tmp/qwen3vl-twenty-20261108/prepare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.REMOTE_STATE


def main():
    inventory = load(TMP / "inventory.json")
    original = load(BASE / "config.json")
    old_frozen = load(BASE / "frozen.json")
    assert file_digest(BASE / "results.json") == inventory["existing_results_sha256"]
    assert implementation_hashes() == old_frozen["implementation_hashes"]
    for name, value in old_frozen["snapshot_files"].items():
        assert file_digest(BASE / name) == value
    resume = len(sys.argv) > 1 and sys.argv[1] == "resume-installed"
    if resume:
        assert OUT.exists() and not (OUT / "frozen.json").exists()
        preserved = load(OUT / "verification/original20-preserved.json")
        assert all(file_digest(BASE / n) == h for n, h in preserved.items())
        assert load(OUT / "sampling.json") == inventory
    else:
        assert not OUT.exists()
        OUT.mkdir()
        (OUT / "verification").mkdir()
        preserved = {str(p.relative_to(BASE)): file_digest(p)
                     for p in sorted(BASE.rglob("*")) if p.is_file()
                     and "__pycache__" not in p.parts and p.name != ".DS_Store"}
        save(OUT / "verification/original20-preserved.json", preserved)
        save(OUT / "sampling.json", inventory)
    state = remote(original, state_code(), timeout=60)
    assert not state["owned_trial_workers"] and state["disk_free_bytes"] >= 4 * 1024**3
    assert state["worker_implementation_hashes"] == {
        n.split("/", 1)[1]: h for n, h in old_frozen["implementation_hashes"].items()
        if n.startswith("target_detection_r05/")}
    save(OUT / "verification/server-before.json", state)
    save(OUT / "verification/detector-before.json", Detector(original).status())
    splits = load(ROOT / "upstream-household/embodiedbench/envs/eb_alfred/data/splits/splits.json")
    old_alfred = load(ROOT / "action_alfred_data.json")
    alfred = {
        "repository": old_alfred["repository"], "revision": old_alfred["revision"],
        "files": [{"index": t["index"], "path": splits["base"][t["index"]]["task"]
                  + "/pp/ann_" + str(splits["base"][t["index"]]["repeat_idx"]) + ".json"}
                 for t in inventory["selected_tasks"] if t["environment"] == "alfred"],
    }
    if resume:
        receipt = load(OUT / "verification/assets-installed.json")
    else:
        print(canonical({"phase": "asset_download", "files": 120, "model_requests": 0}).decode(), flush=True)
        receipt = remote(original, (TMP / "assets_remote.py").read_text(),
                         canonical({"inventory": inventory, "alfred": alfred, "asset_root": ASSET_ROOT,
                                    "old_source_root": original["runtime"]["source_root"]}), timeout=1800)
        save(OUT / "verification/assets-installed.json", receipt)
    inputs = OUT / "inputs"
    inputs.mkdir()
    source = inputs / "embodiedRunner"
    source.mkdir()
    for name in ("upstream", "upstream-household", "upstream-manipulation", "official_sources.json",
                 "household_sources.json", "manipulation_sources.json", "alfred_pilot_data.json"):
        (source / name).symlink_to(ROOT / name, target_is_directory=(ROOT / name).is_dir())
    save(source / "action_alfred_data.json", receipt["alfred_catalog"])
    catalog = inputs / "manipulation_assets.json"
    save(catalog, receipt["manipulation_catalog"])
    config = load(BASE / "config.json")
    config["episodes"] = inventory["selected_tasks"]
    config["runtime"].update(source_root=receipt["source_root"],
                             manipulation_dataset=receipt["manipulation_dataset"],
                             manipulation_catalog=ASSET_ROOT + "/manipulation_assets.json")
    save(OUT / "config.json", config)
    config, tasks = load_config(OUT / "config.json")
    sources = Sources(source, catalog)
    save(OUT / "verification/offline.json", check(config, tasks, sources))
    differences = {k: sources.hashes.get(k) for k, v in old_frozen["official_source_hashes"].items()
                   if sources.hashes.get(k) != v}
    assert set(differences) == {"action_alfred_data.json", "manipulation_assets.json"}
    reports = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(preflight, config, task, sources): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            reports[task.identity] = future.result()
            print(canonical({"phase": "zero_request_preflight", "ready": len(reports),
                             "selected": 120, "task": task.identity}).decode(), flush=True)
    save(OUT / "verification/preflight.json",
         {"reports": reports, "model_requests": 0, "simulator_launches": 0,
          "cpu_preflight_concurrency": 4, "episode_execution_concurrency": 1})
    frozen_source = OUT / "source_frozen/embodiedRunner"
    files = {}
    for name, expected in old_frozen["snapshot_files"].items():
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith("/experiment.json"):
            target.write_bytes(canonical(config) + b"\n")
        else:
            target.write_bytes((BASE / name).read_bytes())
            assert file_digest(target) == expected
        target.chmod(0o444)
        files[name] = file_digest(target)
    for path in (OUT / "config.json", catalog, source / "action_alfred_data.json"):
        path.chmod(0o444)
    run_id = "qwen3vl-extension120-20261108-r01"
    result = frozen_source / "target_detection_qwen3vl/runs" / run_id
    (OUT / "run").symlink_to(result, target_is_directory=True)
    save(OUT / "frozen.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
        "source_root": str(frozen_source), "run_directory": str(result),
        "local_input_source_root": str(source), "local_manipulation_catalog": str(catalog),
        "config_sha256": file_digest(OUT / "config.json"), "snapshot_files": files,
        "implementation_hashes": old_frozen["implementation_hashes"],
        "official_source_hashes": sources.hashes,
        "changed_input_manifests": differences,
        "same_method_as_original20": {k: config[k] == original[k]
                                    for k in ("model", "n_shots", "max_requests", "execution", "detection")},
        "selected_tasks": [t.as_dict() for t in tasks], "new_selected": 120,
        "combined_selected": 140, "planning_model_request_cap": 3450,
        "automatic_retry": False, "reference_answer_visible": False,
        "generative_preflight_requests": 0, "sampling_sha256": file_digest(OUT / "sampling.json"),
    })
    print(canonical({"status": "prepared", "campaign": str(OUT), "selected": 120,
                     "combined": 140, "planning_model_request_cap": 3450}).decode(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(canonical({"status": "prepare_failed", "exception_type": type(exc).__name__,
                         "error_code": getattr(exc, "code", None)}).decode(), flush=True)
        raise SystemExit(2)
