"""Verify the local collector rejects changed native bytes before scoring."""
import importlib
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
collector = importlib.import_module("collect_score_pilot40")


def test_full_collection_checks_native_bytes_before_scoring(tmp_path):
    ids = [str(index) for index in range(40)]
    (tmp_path / "sampling_manifest.json").write_text(json.dumps({
        "questions": [{"qa_id": qa} for qa in ids]}))
    source = tmp_path / "native.json"
    source.write_text('{"status":"returned"}\n')
    sources = [{"file": source.name, "bytes": source.stat().st_size,
                "sha256": collector.digest(source)}]
    trajectories = tmp_path / "trajectories"
    trajectories.mkdir()
    rows = {}
    for qa in ids:
        full = {
            "qa_id": qa, "reference_answer_visible": False,
            "completeness": {"outcome_written": True, "all_model_requests_returned": True,
                             "missing_reviewer_request_files": [], "incomplete_jsonl_lines": []},
            "source_files": sources, "preparation": {"mask_assets": []},
        }
        path = trajectories / f"{qa}.json"
        path.write_text(json.dumps(full))
        rows[qa] = {"file": str(path.relative_to(tmp_path)), "sha256": collector.digest(path)}
    (tmp_path / "trajectory_index.json").write_text(json.dumps({"questions": rows}))
    collector.validate(tmp_path)
    receipt = json.loads((tmp_path / "local_collection_verified.json").read_text())
    assert receipt["selected_questions"] == len(receipt["verified"]) == 40
    assert receipt["reference_answers_read_during_collection"] is False
    # Equal length is insufficient: the collected native bytes must retain their SHA.
    source.write_text('{"status":"modified"}\n')
    with pytest.raises(ValueError, match="journal or mask identity mismatch"):
        collector.validate(tmp_path)
