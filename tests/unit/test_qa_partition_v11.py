"""Synthetic public QA partitions; no model, server or credential access."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from skill3d.evaluation.qa_partition_v11 import (
    PUBLIC_SCOPE_SCHEMA, QA_PARTITION_SCHEMA, QAPartitionError,
    load_qa_partition_manifest, partition_qa_variants, select_qa_execution_variants,
)
from skill3d.routing.task_classifier import canonical_task
from skill3d.schemas.data_access import normalize_experiment_split


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group(rows, field):
    return {
        key: {
            "total": sum(row[field] == key for row in rows),
            **{role: sum(row[field] == key and row["experiment_split"] == role for row in rows)
               for role in ("learning", "validation")},
        } for key in {row[field] for row in rows}
    }


@pytest.fixture
def qa_partition_case(tmp_path):
    metadata = tmp_path / "synthetic-metadata.json"
    metadata_sha256 = _write(metadata, {"synthetic": "hash-only metadata"})
    rows = [
        {"qa_id": f"q{i}", "dataset": "scannet", "scene_name": "scene-a" if i < 4 else "scene-b",
         "question_type": ("object_rel_direction_hard" if i == 2 else
                           "object_rel_direction_easy" if i == 3 else "object_counting"),
         "question": f"Synthetic public question {i}?", "options": None,
         "source_split": "induction" if i < 2 else "inner_validation",
         "experiment_split": "learning" if i % 2 == 0 else "validation"}
        for i in range(6)
    ]
    source = {
        "schema": PUBLIC_SCOPE_SCHEMA, "reference_answers_in_scope": False,
        "metadata_sha256": metadata_sha256, "source_split_config_sha256": "a" * 64,
        "questions": [
            {k: v for k, v in row.items() if k != "experiment_split"} for row in rows
        ],
    }
    source["questions"].append({
        **source["questions"][0], "qa_id": "outside", "scene_name": "outside-scene",
    })
    source_path = tmp_path / "public_scope.json"
    source_sha256 = _write(source_path, source)
    scope = {
        "total_qa": 6, "scene_names": ["scene-a", "scene-b"],
        "public_scope_file": str(source_path), "public_scope_sha256": source_sha256,
        "metadata_sha256": metadata_sha256, "source_split_config_sha256": "a" * 64,
    }
    manifest = {
        "schema": QA_PARTITION_SCHEMA, "split_id": "synthetic-qa-split", "seed": 137,
        "scope": scope, "provenance": {"original_source_split_preserved": True},
        "allocation": {
            "unit": "unique_qa_id", "question_overlap": 0, "scene_overlap_allowed": True,
            "qa_assignment": {row["qa_id"]: row["experiment_split"] for row in rows},
            "actual_counts": {"learning": 3, "validation": 3},
            "actual_ratio": {"learning": 0.5, "validation": 0.5},
            "requested_ratio": {"learning": 0.7, "validation": 0.3},
            "scene_coverage": {"learning": 2, "validation": 2, "shared": 2},
        },
        "by_dataset": _group(rows, "dataset"),
        "by_original_source_split": _group(rows, "source_split"),
        "by_question_type": _group(rows, "question_type"),
        "by_scene": _group(rows, "scene_name"),
        "by_scene_question_type": [
            {"scene_name": scene, "question_type": task, "total": len(selected),
             **Counter(row["experiment_split"] for row in selected),
             **{role: sum(row["experiment_split"] == role for row in selected)
                for role in ("learning", "validation")}}
            for scene, task in sorted({(row["scene_name"], row["question_type"]) for row in rows})
            for selected in [[row for row in rows if row["scene_name"] == scene
                              and row["question_type"] == task]]
        ],
        "files": {},
    }
    documents = {}
    role_paths = {}
    for role in ("learning", "validation"):
        selected = [row for row in rows if row["experiment_split"] == role]
        documents[role] = {
            "schema": QA_PARTITION_SCHEMA, "split_id": manifest["split_id"], "seed": 137,
            "scope": deepcopy(scope), "experiment_split": role, "reference_answers_included": False,
            "counts": {"questions": len(selected), "scenes": 2,
                       "question_types": len({row["question_type"] for row in selected})},
            "questions": deepcopy(selected),
        }
        role_paths[role] = tmp_path / f"{role}.json"
        manifest["files"][role] = {
            "path": role_paths[role].name, "sha256": _write(role_paths[role], documents[role]),
        }
    path = tmp_path / "split_manifest.json"
    _write(path, manifest)
    variants = [
        {"variant_id": f"{row['qa_id']}@{state}", "qa_id": row["qa_id"],
         "scene_id": row["scene_name"], "question_type": canonical_task(row["question_type"]),
         "split": row["source_split"], "evidence_state": state,
         "artifact_ref": f"/synthetic/artifacts/{row['scene_name']}",
         "system_input_status": "ready"}
        for row in rows for state in ("g0m0", "g1m0")
    ]
    return SimpleNamespace(
        path=path, manifest=manifest, rows=rows, documents=documents,
        role_paths=role_paths, source=source, source_path=source_path,
        metadata=metadata, variants=variants,
    )


def _save(case, *, role=None):
    if role is not None:
        case.manifest["files"][role]["sha256"] = _write(case.role_paths[role], case.documents[role])
    _write(case.path, case.manifest)


def test_hash_anchored_unique_union_and_core_interface(qa_partition_case):
    case = qa_partition_case
    partition = load_qa_partition_manifest(case.path, metadata_path=case.metadata)
    overlay = partition.experiment_split
    assert overlay == normalize_experiment_split(overlay)
    assert overlay["qa_roles"] == case.manifest["allocation"]["qa_assignment"]
    assert overlay["protocol"] == overlay["split_id"] == "synthetic-qa-split"
    assert overlay["manifest_identity"]["sha256"] == hashlib.sha256(case.path.read_bytes()).hexdigest()
    assert set(partition.questions) == {f"q{i}" for i in range(6)}
    assert overlay["source_split_by_qa"]["q2"] == "inner_validation"
    assert overlay["statistics"]["actual_counts"] == {"learning": 3, "validation": 3}
    assert overlay["statistics"]["scene_coverage"]["shared"] == 2
    assert overlay["source_files"]["learning"]["sha256"] == case.manifest["files"]["learning"]["sha256"]


def test_same_scene_question_disjoint_all_derived_variants_and_no_mutation(qa_partition_case):
    case = qa_partition_case
    before = deepcopy(case.variants)
    learning, gate = partition_qa_variants(case.variants, load_qa_partition_manifest(case.path))
    assert case.variants == before
    assert len(learning) == len(gate) == 6
    assert {row["qa_id"] for row in learning} == {"q0", "q2", "q4"}
    assert {row["qa_id"] for row in gate} == {"q1", "q3", "q5"}
    assert {row["scene_id"] for row in learning} == {row["scene_id"] for row in gate}
    assert {row["experiment_role"] for row in learning} == {"learning"}
    assert {row["experiment_role"] for row in gate} == {"validation"}
    for row in [*learning, *gate]:
        assert row["split"] == row["source_split"] == case.rows[int(row["qa_id"][1:])]["source_split"]
    assert sum(row["split"] == "inner_validation" for row in learning) == 4


@pytest.mark.parametrize("file_name", ["learning", "validation", "source", "metadata"])
def test_reject_changed_file_hash(qa_partition_case, file_name):
    case = qa_partition_case
    path = (case.source_path if file_name == "source" else
            case.metadata if file_name == "metadata" else case.role_paths[file_name])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(QAPartitionError, match="hash changed"):
        load_qa_partition_manifest(case.path, metadata_path=case.metadata)


def test_missing_role_hash_is_not_unanchored_read(qa_partition_case):
    case = qa_partition_case
    del case.manifest["files"]["learning"]["sha256"]
    _save(case)
    with pytest.raises(QAPartitionError, match="SHA256"):
        load_qa_partition_manifest(case.path)


def test_duplicate_json_assignment_key_is_rejected(qa_partition_case):
    case = qa_partition_case
    data = case.path.read_text()
    assert '"q0": "learning"' in data
    case.path.write_text(data.replace('"q0": "learning"', '"q0": "learning", "q0": "validation"'))
    with pytest.raises(QAPartitionError, match="duplicate JSON key"):
        load_qa_partition_manifest(case.path)


@pytest.mark.parametrize("value", ["gate", None, 1])
def test_unknown_manifest_role_is_rejected(qa_partition_case, value):
    case = qa_partition_case
    case.manifest["allocation"]["qa_assignment"]["q0"] = value
    _save(case)
    with pytest.raises(QAPartitionError, match="unknown or invalid QA roles"):
        load_qa_partition_manifest(case.path)


@pytest.mark.parametrize("field,value", [
    ("experiment_split", "validation"), ("qa_id", "unknown"),
    ("source_split", "induction"), ("scene_name", "another-scene"),
    ("question", "altered public question"), ("question_type", "object_counting"),
    ("dataset", "changed-dataset"), ("options", ["changed-option"]),
])
def test_rehashed_role_file_cannot_change_original_fields_or_role(qa_partition_case, field, value):
    case = qa_partition_case
    case.documents["learning"]["questions"][1][field] = value
    _save(case, role="learning")
    with pytest.raises(QAPartitionError, match="mismatched|fields changed"):
        load_qa_partition_manifest(case.path)


def test_duplicate_and_missing_qa_union_rejected(qa_partition_case):
    case = qa_partition_case
    case.documents["validation"]["questions"][0] = deepcopy(case.documents["learning"]["questions"][0])
    _save(case, role="validation")
    with pytest.raises(QAPartitionError, match="duplicate"):
        load_qa_partition_manifest(case.path)


def test_missing_qa_union_rejected_even_when_role_counts_match(qa_partition_case):
    case = qa_partition_case
    case.documents["learning"]["questions"].pop(1)
    case.documents["learning"]["counts"].update(questions=2, question_types=1)
    _save(case, role="learning")
    with pytest.raises(QAPartitionError, match="union differs"):
        load_qa_partition_manifest(case.path)


@pytest.mark.parametrize("path,value", [
    (("allocation", "actual_counts", "learning"), 4),
    (("allocation", "actual_counts", "learning"), True),
    (("allocation", "scene_coverage", "shared"), 0),
    (("allocation", "actual_ratio", "learning"), 0.7),
    (("allocation", "requested_ratio", "learning"), float("nan")),
    (("allocation", "question_overlap"), 1),
    (("allocation", "scene_overlap_allowed"), False),
    (("by_original_source_split", "inner_validation", "total"), 0),
    (("by_dataset", "scannet", "validation"), 4),
    (("by_scene", "scene-a", "total"), 8),
    (("by_question_type", "object_counting", "learning"), 4),
    (("provenance", "original_source_split_preserved"), False),
])
def test_statistics_and_policy_are_verified(qa_partition_case, path, value):
    case = qa_partition_case
    target = case.manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _save(case)
    with pytest.raises(QAPartitionError):
        load_qa_partition_manifest(case.path)


def test_scene_type_statistics_duplicate_is_rejected(qa_partition_case):
    case = qa_partition_case
    case.manifest["by_scene_question_type"].append(deepcopy(case.manifest["by_scene_question_type"][0]))
    _save(case)
    with pytest.raises(QAPartitionError, match="duplicate scene/question-type"):
        load_qa_partition_manifest(case.path)


def test_role_document_counts_and_identity_are_verified(qa_partition_case):
    case = qa_partition_case
    case.documents["learning"]["counts"]["questions"] = 300
    _save(case, role="learning")
    with pytest.raises(QAPartitionError, match="count changed"):
        load_qa_partition_manifest(case.path)
    case.documents["learning"]["counts"]["questions"] = 3
    case.documents["learning"]["experiment_split"] = "validation"
    _save(case, role="learning")
    with pytest.raises(QAPartitionError, match="identity/role changed"):
        load_qa_partition_manifest(case.path)


def test_public_source_relocation_requires_exact_hash(qa_partition_case):
    case = qa_partition_case
    case.manifest["scope"]["public_scope_file"] = "/nonexistent/staged/public_scope.json"
    for role in ("learning", "validation"):
        case.documents[role]["scope"] = deepcopy(case.manifest["scope"])
        _save(case, role=role)
    partition = load_qa_partition_manifest(case.path)
    assert partition.experiment_split["source_scope_identity"]["path"] == str(case.source_path)
    partition = load_qa_partition_manifest(case.path, public_scope_path=case.source_path)
    assert partition.questions["q2"]["source_split"] == "inner_validation"


def test_new_manifest_bytes_create_new_split_identity(qa_partition_case):
    case = qa_partition_case
    before = load_qa_partition_manifest(case.path).experiment_split
    case.path.write_bytes(case.path.read_bytes() + b"\n")
    after = load_qa_partition_manifest(case.path).experiment_split
    assert before["qa_roles_sha256"] == after["qa_roles_sha256"]
    assert before["identity_sha256"] != after["identity_sha256"]
    assert before["manifest_identity"]["sha256"] != after["manifest_identity"]["sha256"]


@pytest.mark.parametrize("field,value", [
    ("qa_id", "unknown"), ("split", "induction"), ("source_split", "induction"),
    ("experiment_role", "validation"), ("experiment_split", "gate"),
    ("scene_id", "another-scene"), ("question_type", "object_counting"),
])
def test_panel_unknown_roles_and_forged_source_rejected(qa_partition_case, field, value):
    case = qa_partition_case
    variants = deepcopy(case.variants)
    variants[4][field] = value
    with pytest.raises(QAPartitionError):
        partition_qa_variants(variants, load_qa_partition_manifest(case.path))


def test_duplicate_variants_and_empty_role_selection_rejected(qa_partition_case):
    case = qa_partition_case
    partition = load_qa_partition_manifest(case.path)
    with pytest.raises(QAPartitionError, match="duplicate"):
        partition_qa_variants([*case.variants, case.variants[0]], partition)
    with pytest.raises(QAPartitionError, match="nonempty"):
        partition_qa_variants(case.variants[:2], partition)


def test_partial_evidence_panel_preserves_role_assignment(qa_partition_case):
    case = qa_partition_case
    learning, gate = partition_qa_variants(case.variants[:4], load_qa_partition_manifest(case.path))
    assert {row["qa_id"] for row in learning} == {"q0"}
    assert {row["qa_id"] for row in gate} == {"q1"}


def test_validation_execution_works_with_validation_only_panel(qa_partition_case):
    case = qa_partition_case
    partition = load_qa_partition_manifest(case.path)
    validation = [row for row in case.variants if row["qa_id"] in {"q1", "q3", "q5"}]
    selected = select_qa_execution_variants(validation, partition)
    assert len(selected) == 6
    assert all(row["experiment_role"] == "validation" for row in selected)
    assert sum(row["split"] == "inner_validation" for row in selected) == 4
    with pytest.raises(QAPartitionError, match="nonempty"):
        partition_qa_variants(validation, partition)
    with pytest.raises(QAPartitionError, match="unknown QA execution role"):
        select_qa_execution_variants(validation, partition, role="gate")
