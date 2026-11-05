"""Synthetic report fixtures verify disjoint aggregation and protocol rejection."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sr import dataset
import sr
import sr_trial
from aggregate_sr import aggregate


def components(dataset, tmp_path):
    split = sr.write_once(tmp_path / "split.json", sr_trial.split_manifest(dataset))
    source = sr.source_identity()
    directories = []
    for name, ids in (
            ("FAKE_ONLY_20", split["diagnostic20"]),
            ("FAKE_ONLY_80", [s for s in split["internal_test"] if s not in split["diagnostic20"]])):
        directory = tmp_path / name
        directory.mkdir()
        archive = directory / "prepared_evidence.tar.gz"
        archive.write_bytes(b"FAKE_ONLY_ARCHIVE_FIXTURE")
        inputs = [{"problem": {"sample_id": sid},
                   "prepared_artifact_sha256": f"FAKE_ONLY_{sid}"} for sid in ids]
        manifest = sr.write_once(directory / "raw/manifest.json", {
            "inspection": {"configuration": sr.configuration()[0],
                           "skill_sha256": sr.file_sha(sr.SKILL),
                           "protocol": sr.VERSION, "sources": source, "inputs": inputs}})
        sr.write_once(directory / "local_verified.json", {
            "status": "passed", "manifest_sha256": manifest["content_sha256"]})
        sr.write_once(directory / "geometry_audit.json", {
            "prepared_archive_sha256": sr.file_sha(archive),
            "inputs": [{"sample_id": sid, "artifact_sha256": f"FAKE_ONLY_{sid}"} for sid in ids]})
        pairs = []
        for sid in ids:
            trajectory = {"sample_id": sid, "answer": "A", "status": "answered",
                          "request_count": 1, "attempted_tool_calls": 0,
                          "succeeded_tool_calls": 0, "execution_trace": [],
                          "rounds": [], "request_trace": []}
            pairs.append({"sample_id": sid, "question": "FAKE_ONLY_QUESTION", "reference": "A",
                          "relation": {"family": "depth"},
                          **{arm: {"correct": True, "trajectory": trajectory}
                             for arm in ("direct", "skill_tools")}})
        sr.write_once(directory / "results_and_full_trajectories.json", {
            "split": split, "manifest_sha256": manifest["content_sha256"],
            "groups": {}, "paired_outcomes": {"both_correct": len(ids)},
            "audit": {"status": "passed", "checked_payloads": len(ids) * 2},
            "budget": {"reserved_requests": len(ids) * 2}, "pairs": pairs})
        directories.append(directory)
    return directories


def test_aggregation_requires_exact100_pairs_and_preserves_all_trajectories(dataset, tmp_path):
    first, remaining = components(dataset, tmp_path)
    output = tmp_path / "FAKE_ONLY_COMBINED"
    result = aggregate(first, remaining, output, dataset)
    report = sr.read(output / "results_and_full_trajectories.json")
    assert report["audit"]["checked_episodes"] == 200
    assert len(report["pairs"]) == 100
    assert all(g["questions"] == 100 and g["requests"] == 100 for g in result["groups"].values())
    assert report["comparison_requests"] == report["total_paid_requests"] == 200


def test_aggregation_rejects_a_changed_model_before_creating_output(dataset, tmp_path):
    first, remaining = components(dataset, tmp_path)
    path = remaining / "raw/manifest.json"
    manifest = sr.read(path)
    manifest["inspection"]["configuration"]["model"]["model"] = "FAKE_ONLY_CHANGED_MODEL"
    path.unlink()
    sr.write_once(path, {k: v for k, v in manifest.items() if k != "content_sha256"})
    output = tmp_path / "FAKE_ONLY_REJECTED"
    with pytest.raises(ValueError, match="scientific_source_changed"):
        aggregate(first, remaining, output, dataset)
    assert not output.exists()
