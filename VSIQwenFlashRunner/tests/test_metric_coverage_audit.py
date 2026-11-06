"""Synthetic quota audits; no service, model, dataset or reference-answer access."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_metric_coverage.py"
SPEC = importlib.util.spec_from_file_location("metric_coverage_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture():
    return {
        "selection_id": "synthetic", "seed": 137, "anchors": {},
        "selected": [
            {"qa_id": "a", "scene_id": "s1", "raw_type": "object_counting",
             "source_split": "induction", "experiment_split": "train"},
            {"qa_id": "b", "scene_id": "s2", "raw_type": "object_counting",
             "source_split": "final_test", "experiment_split": "test"},
        ],
        "scenes": [
            {"scene_id": "s1", "qa_ids": ["a", "c"], "m4_passed": True},
            {"scene_id": "s2", "qa_ids": ["b", "d"], "m4_passed": True},
        ],
        "m4_raw_type_upper_bounds": {"object_counting": 4},
        "published_states": {"g1m0": 2},
    }


def row(qa, status, split="induction", scene="s1"):
    return {"qa_id": qa, "status": status, "raw_type": "object_counting",
            "source_split": split, "scene_id": scene}


class QuotaAuditTests(unittest.TestCase):
    def test_unread_evidence_is_unknown_not_failure(self):
        result = audit.make_report(fixture())
        self.assertEqual(result["current_metric_qualification"],
                         {"pass": 0, "fail": 0, "unknown": 2})
        self.assertEqual(result["raw_type_quota_feasibility"], "unknown")
        self.assertEqual(result["raw_types"][0]["eligible_upper_bound"], 4)

    def test_same_type_replacements_preserve_quota(self):
        rows = [row("a", "fail"), row("b", "pass", "final_test", "s2"),
                row("c", "pass"), row("d", "pass", "final_test", "s2")]
        result = audit.make_report(fixture(), rows, scanned=True)
        self.assertEqual(result["raw_type_quota_feasibility"], "enough")
        self.assertEqual(result["source_split_and_raw_type_quota_feasibility"], "enough")
        self.assertEqual(result["raw_types"][0]["confirmed_replacements_outside_current"], 2)

    def test_aggregate_count_does_not_hide_source_split_shortage(self):
        rows = [row("a", "pass"), row("b", "fail", "final_test", "s2"),
                row("c", "pass"), row("d", "fail", "final_test", "s2")]
        result = audit.make_report(fixture(), rows, scanned=True)
        self.assertEqual(result["raw_type_quota_feasibility"], "enough")
        self.assertEqual(result["source_split_and_raw_type_quota_feasibility"], "insufficient")

    def test_confirmed_failures_reduce_upper_bound(self):
        rows = [row("a", "fail"), row("b", "fail", "final_test", "s2"),
                row("c", "fail"), row("d", "unknown", "final_test", "s2")]
        result = audit.make_report(fixture(), rows, scanned=True)
        self.assertEqual(result["raw_type_quota_feasibility"], "insufficient")
        self.assertEqual(result["raw_types"][0]["eligible_upper_bound"], 1)

    def test_scene_loss_is_reported_separately(self):
        result = audit.make_report(fixture(), scenes=[
            {"scene_id": "s1", "status": "fail"}, {"scene_id": "s2", "status": "pass"}])
        self.assertEqual(result["current_scene_qualification"],
                         {"pass": 1, "fail": 1, "unknown": 0})

    def test_selected_source_split_cannot_change(self):
        with self.assertRaises(audit.AuditError):
            audit.make_report(fixture(), [row("a", "pass", "final_test")])

    def test_direction_difficulty_stays_separate_in_raw_quotas(self):
        inventory = fixture()
        inventory["selected"][0]["raw_type"] = "object_rel_direction_easy"
        inventory["selected"][1]["raw_type"] = "object_rel_direction_hard"
        inventory["m4_raw_type_upper_bounds"] = {
            "object_rel_direction_easy": 3, "object_rel_direction_hard": 1}
        rows = [
            {**row("a", "pass"), "raw_type": "object_rel_direction_easy"},
            {**row("b", "fail", "final_test", "s2"), "raw_type": "object_rel_direction_hard"},
            {**row("c", "pass"), "raw_type": "object_rel_direction_easy"},
        ]
        result = audit.make_report(inventory, rows)
        self.assertEqual(result["task_types"][0]["quota_feasibility"], "enough")
        self.assertEqual(result["raw_type_quota_feasibility"], "insufficient")


class NativeEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPT.parents[1] / "src"))
        try:
            import numpy
            import pydantic
        except ImportError:
            raise unittest.SkipTest("Native evidence checks require the project environment")

    def fixture(self, root, n_frames=3):
        import numpy as np
        from skill3d.infra.m5_identity import content_sha256, file_identity
        from skill3d.reconstruction import metric_evidence as me, metric_fusion as mf
        from skill3d.schemas.reconstruction import QualityMetrics, ReconstructionArtifact
        from skill3d.storage.relocation import (
            artifact_dependency_refs, create_relocation_receipt)

        arrays = {
            "depth_maps": np.ones((n_frames, 3, 4), np.float32),
            "metric_depth_maps": np.full((n_frames, 3, 4), 2.0, np.float32),
            "metric_valid_masks": np.ones((n_frames, 3, 4), bool),
            "intrinsics": np.tile(np.eye(3), (n_frames, 1, 1)),
            "c2w_list": np.tile(np.eye(4), (n_frames, 1, 1)),
            "point_map": np.zeros((n_frames, 3, 4, 3), np.float32),
            "point_conf": np.ones((n_frames, 3, 4), np.float32),
        }
        old_refs, new_paths = {}, {}
        for name, array in arrays.items():
            new_paths[name] = root / f"{name}.npy"
            np.save(new_paths[name], array, allow_pickle=False)
            old_refs[name] = f"/fixture-worker/scene/{name}.npy"
        q = QualityMetrics(
            warp_inlier_ratio=.9, warp_photometric_inlier_ratio=.9,
            cloud_overlap_ratio=.9, point_depth_relative_p95=.02,
            cloud_neighbor_tail_ratio=1.5, main_gate_passed=True,
            g1_blur_ok=1., g2_brightness=1., g3_motion_blur=0.,
            g4_frame_count=n_frames, g6_depth_var_coeff=.1, g7_dynamic_ratio=0.,
            g9_tracker_consistency=.9, g10_baseline_quality=.5, overall_quality=.9)
        art = ReconstructionArtifact(
            artifact_id="fixture", artifact_version="fixture", scene_name="scene",
            frame_ids=[str(i) for i in range(n_frames)], frame_set_hash="frame-fixture",
            source_frame_indices=list(range(n_frames)), timestamps=[float(i) for i in range(n_frames)],
            quality=q, quality_status="computed",
            confidence={"per_point_confidence": old_refs["point_conf"]}, **old_refs)
        request = {
            "input_identity": "pixel-fixture",
            "sources": {k: file_identity(new_paths[k])["sha256"]
                        for k in me.GEOMETRY_REFS if k in new_paths},
            "frame_ids": art.frame_ids, "frame_set_hash": art.frame_set_hash,
            "scene_name": art.scene_name, "source_frame_indices": art.source_frame_indices,
            "timestamps": art.timestamps,
        }
        fusion = mf.fuse_metric_scale(
            arrays["metric_depth_maps"], arrays["depth_maps"],
            arrays["metric_valid_masks"], model="moge2", min_valid_pixels=1)
        receipt = mf.per_frame_receipt(fusion)
        receipt.update(
            request=request, request_sha256=me.digest(request),
            evidence_sha256={k: file_identity(new_paths[k])["sha256"]
                             for k in ("metric_depth_maps", "metric_valid_masks")})
        new_paths["per_frame_scale_ref"] = root / "scale.json"
        new_paths["per_frame_scale_ref"].write_text(json.dumps(receipt))
        art = art.model_copy(update={
            "per_frame_scale_ref": "/fixture-worker/scene/scale.json",
            "reconstruction_input_identity": "pixel-fixture",
            "metric_input_identity": "pixel-fixture",
            "metric_receipt_sha256": file_identity(new_paths["per_frame_scale_ref"])["sha256"],
            "metric_request_sha256": receipt["request_sha256"],
            "metric_model": fusion.model, "metric_fusion_version": fusion.version,
            "scale_fusion_status": fusion.status, "metric_scale": fusion.metric_scale,
            "scale_self_consistency": fusion.scale_self_consistency,
        })
        (root / "artifact.json").write_text(art.model_dump_json())
        refs = artifact_dependency_refs(art)
        paths = {k: new_paths["point_conf"] if k.startswith("confidence.") else new_paths[k]
                 for k in refs}
        body = {
            "episode": {"qa_id": "qa", "scene_name": "scene", "dataset": "synthetic",
                        "question_type": "object_counting", "ground_truth": ""},
            "source": "synthetic", "video_path": None,
            "pixels": [{"shape": [3, 4, 3], "dtype": "uint8", "sha256": "a" * 64}
                       for _ in range(n_frames)],
            "artifact_path": "/fixture-worker/scene/artifact.json",
            "artifact_json_sha256": file_identity(root / "artifact.json")["sha256"],
            "artifact_id": art.artifact_id, "artifact_version": art.artifact_version,
            "files": {k: {"path": refs[k], "sha256": file_identity(p)["sha256"]}
                      for k, p in paths.items()},
        }
        identity = {**body, "sha256": content_sha256(body)}
        migration = create_relocation_receipt(
            logical_input_identity=identity, destination_root=root,
            artifact_ref="artifact.json", dependency_refs={k: p.name for k, p in paths.items()})
        entry = {
            "qa_id": "qa", "scene_id": "scene", "split": "induction",
            "publication_root": str(root), "scene_artifact": str(root / "artifact.json"),
            "reference": {"scene_artifact_ref": identity["artifact_path"],
                          "scene_input_identity_sha256": identity["sha256"]},
        }
        identity_path = root / "identity.json"
        identity_path.write_text(json.dumps(identity))
        entry.update(
            scene_input_identity_file=str(identity_path),
            scene_input_identity_file_sha256=file_identity(identity_path)["sha256"])
        return entry, identity, migration

    def test_native_relocated_metric_evidence_qualifies_nonmetric_type(self):
        with tempfile.TemporaryDirectory() as path:
            entry, identity, migration = self.fixture(Path(path).resolve())
            public_identity, public_migration, public = audit.checked_identity(
                entry, {"variants": {"metric": {"migration_receipts": {"qa": migration}}}})
            result = audit.native_scene_check(entry, public_identity, public_migration)
            self.assertEqual(public["raw_type"], "object_counting")
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["metric_scale"], 2.)
            self.assertEqual(result["valid_frame_ratio"], 1.)

    def test_changed_metric_array_is_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path).resolve()
            entry, identity, migration = self.fixture(root)
            (root / "metric_depth_maps.npy").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                audit.native_scene_check(entry, identity, migration)

    def test_failed_fusion_does_not_qualify_despite_full_frame_coverage(self):
        with tempfile.TemporaryDirectory() as path:
            entry, identity, migration = self.fixture(Path(path).resolve(), n_frames=2)
            result = audit.native_scene_check(entry, identity, migration)
            self.assertEqual(result["valid_frame_ratio"], 1.)
            self.assertEqual(result["status"], "fail")
            self.assertIsNone(result["metric_scale"])

    def test_native_scale_threshold_boundaries(self):
        from skill3d.reconstruction_gate.evidence_profile import evaluate_metric_gate
        from skill3d.schemas.evidence import metric_scale_state_from_gate
        values = dict(scene_route="full_3d", main_gate_passed=True, fusion_status="success",
                      scale_self_consistency=.15, valid_frame_ratio=.75, metric_scale=1.,
                      question_type="object_counting")
        for changed, expected in (
            ({}, "available"), ({"scale_self_consistency": .150001}, "unavailable"),
            ({"valid_frame_ratio": .749999}, "unavailable"),
            ({"metric_scale": 0.}, "unavailable")):
            gate = evaluate_metric_gate(**{**values, **changed})
            self.assertEqual(metric_scale_state_from_gate(gate), expected)


if __name__ == "__main__":
    unittest.main()
