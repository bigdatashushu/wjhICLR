"""Package a completed SR trial with source checks and full archive readback."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile


def package(root, backup, trial_module="sr_trial"):
    root, backup = Path(root).resolve(), Path(backup).resolve()
    sys.path[:0] = [str(root / "source/scripts/BLINKRunner")]
    if trial_module not in {"sr_trial", "sr_retest"}:
        raise ValueError("sr_invalid_trial_module")
    trial = importlib.import_module(trial_module)
    sr = trial.sr
    manifest = sr.read(root / "manifest.json")
    trial.verify(root, manifest)
    completed = sr.read(root / "completed.json")
    size = len(manifest["inspection"]["inputs"])
    exit_record = json.loads((root / "exit.json").read_text())
    if (completed["manifest_sha256"] != manifest["content_sha256"]
            or completed["episodes"] != len(manifest["jobs"]) or exit_record["returncode"] != 0
            or any(completed["budget"][k] for k in
                   ("error_requests", "unknown_requests", "active_requests"))
            or completed["budget"]["reserved_requests"] > manifest["request_cap"]):
        raise ValueError("sr_collection_not_complete")
    preparation = sr.read(root / "preparation.json")
    delivery = sr.read(root / "zero_request_delivery.json")
    if (preparation["zero_request_delivery_sha256"] != delivery["content_sha256"]
            or len(delivery["checked"]) != size or delivery["model_requests"] != 0):
        raise ValueError("sr_delivery_preflight_binding")
    if "continuation_sha256" in manifest:
        from sr_continuation import verify
        verify(root, manifest)
    for job in manifest["jobs"]:
        receipt = sr.read(root / "episodes" / (job["operation_id"] + ".json"))
        if receipt["job_sha256"] != sr.sha_value(job):
            raise ValueError("sr_collection_episode_identity")
    paths = [
        root / name for name in (
            "manifest.json", "split.json", "preparation.json", "zero_request_delivery.json",
            "completed.json", "launch.json", "startup.json", "exit.json", "provider_auth.json",
            "paid_run.json", "run.log", "budget/state.json")
    ]
    paths += [root / name for name in ("continuation.json", "direct_verified.json",
                                     "cohort.json", "parent_manifest.json",
                                     "precompute-0.json", "precompute-1.json",
                                     "native_startup.json", "native-exit.json",
                                     "native-start-0.json", "native-start-1.json",
                                     "native-exit-0.json", "native-exit-1.json",
                                     "geometry_audit.json", "native_start.py",
                                     "package_prepared.py", "collect_sr.py",
                                     "prepare_received.py", "status_sr.py")
              if (root / name).exists()]
    paths += [root / name for name in (
        "parent20_manifest.json", "parent80_manifest.json", "cache_reuse.json",
        "retest_cohort.json", "parent_retest20_manifest.json", "parent_retest20_completed.json")
              if (root / name).exists()]
    paths += sorted((root / "episodes").glob("*.json"))
    paths += sorted((root / "requests").rglob("*.json"))
    paths += [root / "source" / name for name in manifest["inspection"]["sources"]]
    files = {str(p.relative_to(root)): sr.file_sha(p) for p in paths}
    inventory = sr.write_once(root / "delivery_files.json", {
        "files": files, "manifest_sha256": manifest["content_sha256"],
        "reference_files_included": False, "source_and_prepared_identity_checked": True,
    })
    paths.append(root / "delivery_files.json")
    archive_path = root / "completed_traces.tar.gz"
    if archive_path.exists() or backup.exists():
        raise ValueError("sr_collection_destination_already_exists")
    with tarfile.open(archive_path, "x:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    expected = {**files, "delivery_files.json": sr.file_sha(root / "delivery_files.json")}
    with tarfile.open(archive_path, "r:gz") as archive:
        if {member.name for member in archive.getmembers()} != set(expected):
            raise ValueError("sr_archive_member_identity")
        for member in archive.getmembers():
            digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            if digest != expected[member.name]:
                raise ValueError("sr_archive_readback_mismatch")
    with tarfile.open(root / "public_runtime.tar.gz", "x:gz", dereference=True) as archive:
        archive.add(root / "split.json", arcname="split.json")
        for name in ("cohort.json", "parent_manifest.json",
                     "parent20_manifest.json", "parent80_manifest.json",
                     "retest_cohort.json", "parent_retest20_manifest.json", "parent_retest20_completed.json"):
            if (root / name).exists():
                archive.add(root / name, arcname=name, recursive=False)
        for name in manifest["inspection"]["sources"]:
            archive.add(root / "source" / name, arcname="source/" + name, recursive=False)
        for name in ("manifest.json", "inputs.jsonl"):
            archive.add(root / "dataset/data" / name, arcname="dataset/data/" + name)
        archive.add(root / "dataset/data/images", arcname="dataset/data/images")
    archives = {name: sr.file_sha(root / name) for name in
                ("completed_traces.tar.gz", "public_runtime.tar.gz")}
    receipt = sr.write_once(root / "delivery_receipt.json", {
        "archives": archives, "inventory_sha256": inventory["content_sha256"],
        "full_trace_archive_readback": True, "files": len(expected),
        "completed_episodes": len(manifest["jobs"]), "model_requests": completed["budget"]["reserved_requests"],
        "reference_answers_read": False, "backup_root": str(backup),
        "collector_sha256": sr.file_sha(Path(__file__)),
    })
    shutil.copytree(root, backup, symlinks=False)
    for name, digest in {**files, **archives,
                         "delivery_files.json": expected["delivery_files.json"],
                         "delivery_receipt.json": sr.file_sha(root / "delivery_receipt.json")}.items():
        if sr.file_sha(backup / name) != digest:
            raise ValueError("sr_durable_backup_mismatch")
    return {"status": "verified", "files": len(expected), "archives": archives,
            "backup_root": str(backup), "receipt_sha256": receipt["content_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--trial-module", choices=("sr_trial", "sr_retest"), default="sr_trial")
    args = parser.parse_args()
    os.umask(0o077)
    for name in list(os.environ):
        if any(t in name.upper() for t in (
                "API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "PASSWORD", "SECRET")):
            os.environ.pop(name, None)
    print(json.dumps(package(args.root, args.backup, args.trial_module)))


if __name__ == "__main__":
    main()
