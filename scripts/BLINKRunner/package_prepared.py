"""Validate native SR completion, package evidence, and fully read back the archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile


def package(root, archive_path):
    sys.path.insert(0, str(root / "source/scripts/BLINKRunner"))
    import sr_trial as trial
    sr = trial.sr
    selected = trial.selected_ids(root)
    exit_record = json.loads((root / "native-exit.json").read_text())
    if exit_record["returncodes"] != [0, 0]:
        raise ValueError("native_preparation_not_complete")
    config_path = root / "source/scripts/BLINKRunner/sr_precompute.yaml"
    startup = sr.read(root / "native_startup.json")
    if (startup["config_sha256"] != sr.file_sha(config_path)
            or startup["cohort_sha256"] != trial.cohort(root)["content_sha256"]
            or startup["questions"] != len(selected)):
        raise ValueError("native_preparation_identity_changed")
    _, items = sr.load_inputs(root / "dataset", ("val",))
    by_id = {p.sample_id: p for p, _ in items}
    receipts = []
    expected_shards = {}
    for shard in (0, 1):
        receipt = sr.read(root / f"precompute-{shard}.json")
        if (receipt["config_sha256"] != sr.file_sha(config_path)
                or [r["sample_id"] for r in receipt["inputs"]] != selected[shard::2]
                or receipt["reference_answers_read"] is not False):
            raise ValueError("native_shard_identity_changed")
        expected_shards.update({r["sample_id"]: r for r in receipt["inputs"]})
    for sid in selected:
        problem = by_id[sid]
        value = sr.prepared_for(problem, root / "prepared")
        expected = expected_shards[sid]
        if (value.artifact_sha256 != expected["artifact_sha256"]
                or problem.input_sha256 != expected["input_sha256"]
                or value.errors or not value.has_relative_depth
                or len(value.candidates) > 4):
            raise ValueError("native_prepared_evidence_changed")
        trace = value.provenance["detection_selection"]
        if len(trace) != 1 or trace[0]["policy"]["max_per_category"] != 2:
            raise ValueError("native_candidate_policy_changed")
        receipts.append({**expected, "relative_depth_only": True,
                         "category_confirmation_is_surface_validation": False})
    members = {}
    for p in sorted((root / "prepared").rglob("*")):
        if p.is_file():
            members[str(p.relative_to(root))] = sr.file_sha(p)
    with tarfile.open(archive_path, "x:gz") as archive:
        for name in members:
            archive.add(root / name, arcname=name, recursive=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        if [m.name for m in archive.getmembers()] != list(members):
            raise ValueError("native_archive_inventory_changed")
        for member in archive.getmembers():
            if hashlib.sha256(archive.extractfile(member).read()).hexdigest() != members[member.name]:
                raise ValueError("native_archive_readback_failed")
    audit = sr.write_once(root / "geometry_audit.json", {
        "schema": "blink-sr-native-evidence-audit/1", "at": trial.now(),
        "inputs": receipts, "files": members, "status": "passed",
        "prepared_archive_sha256": sr.file_sha(archive_path),
        "full_archive_readback": True, "model_requests": 0,
        "reference_answers_read": False, "physical_gpus": [3, 4],
        "processor_sha256": sr.file_sha(Path(__file__)),
    })
    print(json.dumps({"status": "passed", "questions": len(receipts),
                      "files": len(members), "archive_sha256": audit["prepared_archive_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    for k in list(os.environ):
        if any(t in k.upper() for t in ("API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "PASSWORD", "SECRET")):
            os.environ.pop(k, None)
    package(args.root.resolve(), args.archive.resolve())
