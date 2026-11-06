"""Synthetic checks for immutable source/role quotas and label-free proposals."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_metric_coverage as audit
import propose_metric_replacements as proposer


def candidate(root, *, split="induction", ground_truth=""):
    episode = {"qa_id": "new", "scene_name": "good", "question_type": "object_counting",
               "ground_truth": ground_truth}
    body = {"episode": episode, "artifact_path": "/original/artifact.json"}
    identity = {**body, "sha256": hashlib.sha256(audit.canonical(body)).hexdigest()}
    migration_body = {
        "logical_input_identity": identity, "destination_root": str(root),
        "artifact": {"target_ref": "artifact.json"}}
    migration = {**migration_body,
                 "receipt_sha256": hashlib.sha256(audit.canonical(migration_body)).hexdigest()}
    validation = {"variants": {"metric": {"migration_receipts": {"new": migration}}}}
    for name, value in (("identity.json", identity), ("validation.json", validation)):
        (root / name).write_text(json.dumps(value))
    return {
        "qa_id": "new", "scene_id": "good", "split": split,
        "reference": {"scene_artifact_ref": "/original/artifact.json",
                      "scene_input_identity_sha256": identity["sha256"]},
        "publication_root": str(root), "scene_artifact": str(root / "artifact.json"),
        "migration_receipt_file": str(root / "validation.json"),
        "migration_receipt_file_sha256": hashlib.sha256(
            (root / "validation.json").read_bytes()).hexdigest(),
        "scene_input_identity_file": str(root / "identity.json"),
        "scene_input_identity_file_sha256": hashlib.sha256(
            (root / "identity.json").read_bytes()).hexdigest(),
    }


def inputs(entry):
    inventory = {
        "selection_id": "synthetic", "seed": 137,
        "anchors": {"catalog_identity_hash": "fixture"},
        "selected": [
            {"qa_id": "old", "scene_id": "bad", "raw_type": "object_counting",
             "source_split": "induction", "experiment_split": "train"},
            {"qa_id": "kept", "scene_id": "good", "raw_type": "object_counting",
             "source_split": "induction", "experiment_split": "test"}]}
    report = {
        "status": "complete", "anchors": inventory["anchors"],
        "scene_checks": [{"scene_id": "bad", "status": "fail"},
                         {"scene_id": "good", "status": "pass"}]}
    catalog = {"status": "verified", "no_labels": True, "identity_hash": "fixture",
               "variants": {"metric": {"new": entry}}}
    return inventory, report, catalog


class ReplacementTests(unittest.TestCase):
    def test_replacement_preserves_role_and_source_and_exposes_scene_loss(self):
        with tempfile.TemporaryDirectory() as path:
            result = proposer.propose(*inputs(candidate(Path(path).resolve())))
        self.assertEqual(result["n_replacements_found"], 1)
        self.assertTrue(result["exact_source_and_role_joint_quotas_preserved"])
        self.assertEqual(result["original_scene_count"], 2)
        self.assertEqual(result["proposed_scene_count"], 1)
        self.assertEqual(result["replacement_pairs"][0]["new"]["experiment_split"], "train")

    def test_wrong_source_split_cannot_fill_vacancy(self):
        with tempfile.TemporaryDirectory() as path:
            result = proposer.propose(*inputs(candidate(Path(path).resolve(), split="final_test")))
        self.assertEqual(result["status"], "insufficient_candidates")
        self.assertEqual(result["n_replacements_found"], 0)

    def test_nonempty_reference_answer_is_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(audit.AuditError):
                proposer.propose(*inputs(candidate(
                    Path(path).resolve(), ground_truth="synthetic-reference")))


if __name__ == "__main__":
    unittest.main()
