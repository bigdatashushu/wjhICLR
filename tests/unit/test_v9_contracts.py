from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

# The repository's current SciPy/NumPy environment is incompatible at import
# time. Keep this compatibility shim local to the focused contract tests.
for _name, _value in {"long": np.int64, "ulong": np.uint64}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)

from skill3d.online.runner import OnlineRunConfig, _invalidate_metric_gate
from skill3d.schemas import MetricEvidenceGateResult, SkillLibraryCandidate
from skill3d.skills.library import (
    candidate_record_from_revision,
    candidate_revision_from_record,
)
from skill3d.skills.registry import active_snapshot_provenance


ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "skill_library"


def test_current_budget_fields_are_normalized_without_legacy_aliases():
    names = {field.name for field in fields(OnlineRunConfig)}
    assert {"max_agent_rounds", "reserve_final_rounds", "max_recovery"}.isdisjoint(names)
    explicit = OnlineRunConfig(
        max_solver_rounds=8, finalization_rounds=7, max_retries_per_operation=1)
    assert (explicit.max_solver_rounds, explicit.finalization_rounds,
            explicit.max_retries_per_operation) == (8, 7, 1)
    bounded = OnlineRunConfig(max_solver_rounds=1, finalization_rounds=99,
                              max_retries_per_operation=-3)
    assert (bounded.max_solver_rounds, bounded.finalization_rounds,
            bounded.max_retries_per_operation) == (1, 0, 0)


def test_metric_gate_invalidation_revalidates_partial_gate():
    gate = MetricEvidenceGateResult(
        gate_passed=True, gate_version="test", sub_results={"unrelated": True})
    invalidated = _invalidate_metric_gate(gate, "geometry_3d")
    assert invalidated is not None
    assert invalidated.gate_passed is False
    assert invalidated.sub_results["scene_route_full_3d"] is False
    assert invalidated.sub_results["unrelated"] is True
    assert "geometry_3d" in invalidated.invalidated_by
    assert "scene_route_full_3d" in invalidated.missing_subconditions


def test_candidate_record_revision_mapping_preserves_lineage(tmp_path: Path):
    raw = json.loads((LIBRARY / "generated" / "S01" / "1.0.0.json").read_text())
    spec_content = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    record = SkillLibraryCandidate(
        candidate_id="cand-map",
        operation="revise",
        canonical_question_type="object_counting",
        parent_skill_versions=["S01@0.9.0"],
        source_trace_refs=["trace-1"],
        source_split="learning",
        experience_relation="retrieved_and_delivered_used",
        hypothesis="减少漏计",
        patch="增加遮挡后重现检查",
        expected_scope="object_counting",
        expected_effect="降低漏计",
        known_risks="增加观察成本",
        source_path="skills/count-scene-objects/SKILL.md",
        source_sha256="source-digest",
        generated_spec_path="generated/S01/1.0.0.json",
        generated_sha256="generated-digest",
        manifest_ref="manifest-digest",
        created_at="2026-09-26T00:00:00Z",
    )
    record_path = tmp_path / "candidates" / "future" / "cand-map.json"
    revision = candidate_revision_from_record(
        record, record_path=record_path, spec_content=spec_content)
    assert revision.root_candidate_id == "cand-map"
    assert revision.parent_version == "S01@0.9.0"
    assert revision.induction_trace_refs == ["trace-1"]
    assert revision.candidate_record_ref == str(record_path)
    assert revision.source_sha256 == "source-digest"
    assert revision.manifest_ref == "manifest-digest"

    projected = candidate_record_from_revision(revision)
    assert projected.candidate_id == record.candidate_id
    assert projected.parent_skill_versions == ["S01@0.9.0"]
    assert projected.source_path == record.source_path
    assert projected.generated_sha256 == record.generated_sha256


def test_active_snapshot_provenance_reads_manifest_digest(tmp_path):
    # v9 provenance must be independent of which version is active in the workspace.
    store = tmp_path / "snapshots"
    store.mkdir()
    name = "snapshot_S0-seed-20260925-v1.json"
    (store / name).write_bytes((LIBRARY / "snapshots" / name).read_bytes())
    (store / "active_snapshot.json").write_text(
        json.dumps({"snapshot_id": "S0-seed-20260925-v1"}))
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "library_manifest.json").write_bytes(
        (LIBRARY / "manifests/library_manifest.json").read_bytes())
    snapshot_id, digest = active_snapshot_provenance(store)
    assert snapshot_id == "S0-seed-20260925-v1"
    assert digest == "61259a20580170f8bf3d0490972cd448a6a67cc5d6316b07c4a77b7db4708046"
