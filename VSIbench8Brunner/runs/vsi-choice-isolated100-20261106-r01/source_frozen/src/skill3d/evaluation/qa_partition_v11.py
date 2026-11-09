"""Hash-anchored QA roles: same-scene questions may belong to different roles."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

from skill3d.routing.task_classifier import canonical_task
from skill3d.schemas.data_access import (
    DATA_SPLITS, TRAIN_TEST_PARTITION_SCHEMA, is_train_test_partition,
    normalize_experiment_split,
)


QA_PARTITION_SCHEMA = "harness3d-qa-overlapping-scene-split/1"
PUBLIC_SCOPE_SCHEMA = "harness3d-full-question-precompute-scope/1"
QA_ROLES = ("learning", "validation")
SOURCE_SPLITS = {"induction", "inner_validation"}
QUESTION_FIELDS = {
    "qa_id", "dataset", "scene_name", "question_type", "question", "options",
    "source_split",
}
TRAIN_TEST_PUBLIC_EXTRA_FIELDS = {"id", "split", "benchmark_split", "experiment_split"}


class QAPartitionError(ValueError):
    """The frozen role allocation or its source identity is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QAPartitionError(message)


def _digest(value) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             "QA partition requires a lowercase SHA256")
    return value


def _unique_object(pairs) -> dict:
    value = {}
    for key, child in pairs:
        _require(key not in value, f"duplicate JSON key in QA partition: {key}")
        value[key] = child
    return value


def _read_anchored(path: Path, expected: str | None = None) -> tuple[dict, dict]:
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    if expected is not None:
        _require(sha256 == _digest(expected), f"QA partition file hash changed: {path}")
    value = json.loads(data, object_pairs_hook=_unique_object)
    _require(isinstance(value, dict), f"QA partition file must be a JSON object: {path}")
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == sha256,
             f"QA partition file changed during reading: {path}")
    return value, {"path": str(path.resolve()), "size": len(data), "sha256": sha256}


def _file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(rows: Sequence[Mapping], field: str) -> dict:
    result = defaultdict(lambda: {"total": 0, "learning": 0, "validation": 0})
    for row in rows:
        result[row[field]]["total"] += 1
        result[row[field]][row["experiment_split"]] += 1
    return dict(result)


def _check_counts(claimed, actual, name: str) -> None:
    _require(isinstance(claimed, dict) and set(claimed) == set(actual),
             f"QA partition {name} keys changed")
    for key, count in actual.items():
        if isinstance(count, dict):
            _check_counts(claimed[key], count, f"{name}/{key}")
        else:
            _require(type(claimed[key]) is int and claimed[key] == count,
                     f"QA partition {name}/{key} count changed")


def _statistics(rows: Sequence[Mapping]) -> dict:
    scenes = {
        role: {row["scene_name"] for row in rows if row["experiment_split"] == role}
        for role in QA_ROLES
    }
    cells = defaultdict(lambda: {"total": 0, "learning": 0, "validation": 0})
    for row in rows:
        cell = cells[(row["scene_name"], row["question_type"])]
        cell["total"] += 1
        cell[row["experiment_split"]] += 1
    return {
        "actual_counts": dict(Counter(row["experiment_split"] for row in rows)),
        "scene_coverage": {
            "learning": len(scenes["learning"]), "validation": len(scenes["validation"]),
            "shared": len(scenes["learning"] & scenes["validation"]),
        },
        "by_dataset": _counts(rows, "dataset"),
        "by_original_source_split": _counts(rows, "source_split"),
        "by_question_type": _counts(rows, "question_type"),
        "by_scene": _counts(rows, "scene_name"),
        "by_scene_question_type": [
            {"scene_name": scene, "question_type": task, **counts}
            for (scene, task), counts in sorted(cells.items())
        ],
    }


@dataclass(frozen=True)
class QAPartition:
    """Verified public question rows and the core runner's hashed role overlay."""

    questions: dict[str, dict]
    experiment_split: dict


def _read_question_jsonl(path: Path, expected: str) -> tuple[list[dict], dict]:
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    _require(sha256 == _digest(expected), f"QA partition file hash changed: {path}")
    rows = [json.loads(line, object_pairs_hook=_unique_object)
            for line in data.splitlines() if line.strip()]
    _require(bool(rows) and all(isinstance(row, dict) for row in rows),
             f"QA partition requires nonempty question JSONL: {path}")
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == sha256,
             f"QA partition file changed during reading: {path}")
    return rows, {"path": str(path.resolve()), "size": len(data), "sha256": sha256}


def _load_train_test_partition(path, manifest, identity, metadata_path, public_scope_path):
    """Public train/test JSONL, with no allocation tables or legacy source scope."""
    split_id = manifest.get("split_id")
    _require(isinstance(split_id, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", split_id) is not None,
             "QA partition requires a distinct split_id")
    _require(type(manifest.get("seed")) is int, "QA partition seed must be an integer")
    _digest(manifest.get("metadata_sha256"))
    if metadata_path is not None:
        _require(_file_digest(metadata_path) == manifest["metadata_sha256"],
                 "QA partition metadata file hash changed")
    scenes = manifest.get("scene_names")
    _require(isinstance(scenes, list) and bool(scenes)
             and all(isinstance(scene, str) and scene for scene in scenes)
             and len(set(scenes)) == len(scenes), "invalid QA partition scene scope")
    _require(manifest.get("source_split_preserved") is True,
             "QA partition must preserve original source splits")
    files = manifest.get("files")
    _require(isinstance(files, dict) and set(files) == {"train", "test"},
             "train/test partition requires exactly two public role files")

    def resolve(ref):
        _require(isinstance(ref, str) and bool(ref), "QA partition requires a question file path")
        result = Path(ref)
        return result if result.is_absolute() else path.parent / result

    source_path = resolve(str(public_scope_path) if public_scope_path is not None
                          else manifest.get("questions_file"))
    source_rows, source_identity = _read_question_jsonl(
        source_path, manifest.get("questions_sha256"))
    originals = {}
    for row in source_rows:
        _require(QUESTION_FIELDS <= set(row)
                 and set(row) <= QUESTION_FIELDS | TRAIN_TEST_PUBLIC_EXTRA_FIELDS,
                 "train/test source question fields changed")
        _require(all(isinstance(row[key], str) and row[key]
                     for key in ("qa_id", "dataset", "scene_name", "question_type", "question")),
                 "invalid train/test source question identity")
        qa_id = row["qa_id"]
        _require(("id" not in row or str(row["id"]) == qa_id)
                 and ("split" not in row or row["split"] == row["source_split"])
                 and ("benchmark_split" not in row or row["benchmark_split"] == "test")
                 and ("experiment_split" not in row or row["experiment_split"] in {"train", "test"}),
                 f"{qa_id}: inconsistent public question extras")
        _require(qa_id not in originals, f"duplicate source QA: {qa_id}")
        _require(row["source_split"] in DATA_SPLITS, f"{qa_id}: unsupported original source_split")
        _require(row["options"] is None or (
            isinstance(row["options"], list) and all(isinstance(x, str) for x in row["options"])),
            f"{qa_id}: invalid public options")
        canonical_task(row["question_type"])
        originals[qa_id] = row
    _require({row["scene_name"] for row in source_rows} == set(scenes),
             "train/test source scene scope changed")
    questions, roles, file_identities = {}, {}, {}
    for public_role, core_role in (("train", "learning"), ("test", "validation")):
        entry = files[public_role]
        _require(isinstance(entry, dict), f"QA partition lacks {public_role} file")
        rows, file_identity = _read_question_jsonl(
            resolve(entry.get("path")), entry.get("sha256"))
        file_identities[public_role] = file_identity
        for row in rows:
            _require(QUESTION_FIELDS | {"experiment_split"} <= set(row)
                     and set(row) <= QUESTION_FIELDS | TRAIN_TEST_PUBLIC_EXTRA_FIELDS,
                     f"train/test {public_role} question fields changed")
            qa_id = row["qa_id"]
            _require(isinstance(qa_id, str) and qa_id and qa_id not in questions,
                     f"duplicate or invalid QA in role union: {qa_id!r}")
            _require(row["experiment_split"] == public_role,
                     f"{qa_id}: mismatched public experiment role")
            original = originals.get(qa_id)
            _require(original is not None and all(row[key] == original[key] for key in QUESTION_FIELDS),
                     f"{qa_id}: original question/source fields changed")
            _require(("id" not in row or str(row["id"]) == qa_id)
                     and ("split" not in row or row["split"] == row["source_split"])
                     and ("benchmark_split" not in row or row["benchmark_split"] == "test")
                     and original.get("experiment_split", public_role) == public_role,
                     f"{qa_id}: inconsistent public question extras")
            questions[qa_id] = deepcopy(row)
            roles[qa_id] = core_role
        _require({row["scene_name"] for row in rows} == set(scenes),
                 f"train/test {public_role} must cover every declared scene")
    _require(set(questions) == set(originals), "train/test role union differs from source questions")
    overlay = normalize_experiment_split({
        "protocol": TRAIN_TEST_PARTITION_SCHEMA, "partition_schema": TRAIN_TEST_PARTITION_SCHEMA,
        "split_id": split_id, "seed": manifest["seed"], "manifest_identity": identity,
        "qa_roles": roles, "source_split_preserved": True,
        "source_split_by_qa": {qa: row["source_split"] for qa, row in questions.items()},
        "source_files": file_identities, "source_scope_identity": source_identity,
        "scope": {"total_qa": len(questions), "scene_names": scenes,
                  "metadata_sha256": manifest["metadata_sha256"]},
    })
    return QAPartition(questions=questions, experiment_split=overlay)


def load_qa_partition_manifest(
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    public_scope_path: str | Path | None = None,
) -> QAPartition:
    """Verify role files, their unique union, source projection and all counts.

    The public source scope may be relocated beside the manifest using its
    original basename; its declared SHA256 must still match. ``metadata_path``
    additionally verifies the CLI's offline metadata without inspecting labels.
    No scene, model, binding or server assets are loaded here.
    """
    path = Path(path).resolve()
    manifest, manifest_identity = _read_anchored(path)
    if manifest.get("schema") == TRAIN_TEST_PARTITION_SCHEMA:
        return _load_train_test_partition(
            path, manifest, manifest_identity, metadata_path, public_scope_path)
    _require(manifest.get("schema") == QA_PARTITION_SCHEMA, "unknown QA partition schema")
    split_id = manifest.get("split_id")
    _require(isinstance(split_id, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", split_id) is not None,
             "QA partition requires a distinct split_id")
    _require(type(manifest.get("seed")) is int, "QA partition seed must be an integer")
    scope = manifest.get("scope")
    allocation = manifest.get("allocation")
    files = manifest.get("files")
    _require(isinstance(scope, dict) and isinstance(allocation, dict)
             and isinstance(files, dict) and set(files) == set(QA_ROLES),
             "QA partition lacks scope, allocation or role files")
    _require(type(scope.get("total_qa")) is int and scope["total_qa"] > 0,
             "QA partition total_qa must be positive")
    scene_names = scope.get("scene_names")
    _require(isinstance(scene_names, list) and bool(scene_names)
             and all(isinstance(scene, str) and scene for scene in scene_names)
             and len(set(scene_names)) == len(scene_names), "invalid QA partition scene scope")
    for name in ("metadata_sha256", "source_split_config_sha256", "public_scope_sha256"):
        _digest(scope.get(name))
    if metadata_path is not None:
        _require(_file_digest(metadata_path) == scope["metadata_sha256"],
                 "QA partition metadata file hash changed")
    _require(isinstance(scope.get("public_scope_file"), str) and scope["public_scope_file"],
             "QA partition requires the original public source scope")
    source_path = Path(public_scope_path or scope["public_scope_file"])
    if not source_path.is_absolute():
        source_path = path.parent / source_path
    if public_scope_path is None and not source_path.is_file():
        source_path = path.parent / source_path.name
    source, source_identity = _read_anchored(source_path, scope["public_scope_sha256"])
    _require(source.get("schema") == PUBLIC_SCOPE_SCHEMA
             and source.get("reference_answers_in_scope") is False,
             "QA partition source must be the label-free public scope")
    _require(all(source.get(key) == scope[key] for key in
                 ("metadata_sha256", "source_split_config_sha256")),
             "QA partition source metadata/split hashes changed")
    source_rows = source.get("questions")
    _require(isinstance(source_rows, list), "QA partition source questions must be a list")
    originals = {}
    for row in source_rows:
        _require(isinstance(row, dict) and QUESTION_FIELDS <= set(row),
                 "invalid QA partition source question")
        qa_id = row["qa_id"]
        _require(isinstance(qa_id, str) and qa_id and qa_id not in originals,
                 f"duplicate or invalid source QA: {qa_id!r}")
        originals[qa_id] = row
    assignments = allocation.get("qa_assignment")
    _require(isinstance(assignments, dict) and bool(assignments)
             and all(isinstance(qa, str) and qa and role in QA_ROLES
                     for qa, role in assignments.items()),
             "QA partition has unknown or invalid QA roles")
    _require(len(assignments) == scope["total_qa"],
             "QA partition assignment does not cover total_qa")
    _require(set(assignments) == {
        qa for qa, row in originals.items() if row["scene_name"] in scene_names
    }, "QA partition assignment does not exactly cover the original scene scope")
    _require(allocation.get("unit") == "unique_qa_id"
             and type(allocation.get("question_overlap")) is int
             and allocation["question_overlap"] == 0
             and allocation.get("scene_overlap_allowed") is True,
             "QA partition requires same-scene question-disjoint allocation")
    _require((manifest.get("provenance") or {}).get("original_source_split_preserved") is True,
             "QA partition must preserve original source splits")
    questions = {}
    file_identities = {}
    for role in QA_ROLES:
        entry = files[role]
        _require(isinstance(entry, dict) and isinstance(entry.get("path"), str)
                 and entry["path"], f"QA partition lacks {role} file")
        role_path = Path(entry["path"])
        if not role_path.is_absolute():
            role_path = path.parent / role_path
        document, identity = _read_anchored(role_path, _digest(entry.get("sha256")))
        file_identities[role] = identity
        _require(document.get("schema") == QA_PARTITION_SCHEMA
                 and document.get("split_id") == split_id
                 and type(document.get("seed")) is int
                 and document["seed"] == manifest["seed"]
                 and document.get("scope") == scope
                 and document.get("experiment_split") == role
                 and document.get("reference_answers_included") is False,
                 f"QA partition {role} document identity/role changed")
        rows = document.get("questions")
        _require(isinstance(rows, list) and bool(rows), f"QA partition {role} has no questions")
        for row in rows:
            _require(isinstance(row, dict) and set(row) == QUESTION_FIELDS | {"experiment_split"},
                     f"QA partition {role} question fields changed")
            qa_id = row["qa_id"]
            _require(isinstance(qa_id, str) and qa_id and qa_id not in questions,
                     f"duplicate or invalid QA in role union: {qa_id!r}")
            _require(row["experiment_split"] == role and assignments.get(qa_id) == role,
                     f"{qa_id}: unknown or mismatched experiment role")
            original = originals.get(qa_id)
            _require(original is not None and all(row[key] == original[key] for key in QUESTION_FIELDS),
                     f"{qa_id}: original question/source fields changed")
            _require(row["source_split"] in SOURCE_SPLITS,
                     f"{qa_id}: unsupported original source_split")
            canonical_task(row["question_type"])
            questions[qa_id] = deepcopy(row)
        _check_counts(document.get("counts"), {
            "questions": len(rows), "scenes": len({row["scene_name"] for row in rows}),
            "question_types": len({row["question_type"] for row in rows}),
        }, f"{role} file")
    _require(set(questions) == set(assignments),
             "QA partition role file union differs from allocation")
    stats = _statistics(list(questions.values()))
    for name in ("actual_counts", "scene_coverage"):
        _check_counts(allocation.get(name), stats[name], name)
    for name in ("by_dataset", "by_original_source_split", "by_question_type", "by_scene"):
        _check_counts(manifest.get(name), stats[name], name)
    declared_cells = manifest.get("by_scene_question_type")
    _require(isinstance(declared_cells, list)
             and all(isinstance(cell, dict) and set(cell) == {
                 "scene_name", "question_type", "total", "learning", "validation",
             } for cell in declared_cells), "invalid scene/question-type statistics")
    cell_map = {}
    for cell in declared_cells:
        key = (cell["scene_name"], cell["question_type"])
        _require(key not in cell_map, "duplicate scene/question-type statistic")
        cell_map[key] = {k: cell[k] for k in ("total", *QA_ROLES)}
    _check_counts(cell_map, {
        (cell["scene_name"], cell["question_type"]): {k: cell[k] for k in ("total", *QA_ROLES)}
        for cell in stats["by_scene_question_type"]
    }, "by_scene_question_type")
    for name in ("actual_ratio", "requested_ratio"):
        ratio = allocation.get(name)
        _require(isinstance(ratio, dict) and set(ratio) == set(QA_ROLES)
                 and all(type(ratio[role]) in (int, float) and math.isfinite(ratio[role])
                         and 0 < ratio[role] < 1 for role in QA_ROLES)
                 and math.isclose(sum(ratio.values()), 1.0, rel_tol=0, abs_tol=1e-12),
                 f"invalid QA partition {name}")
        if name == "actual_ratio":
            _require(all(math.isclose(ratio[role], stats["actual_counts"][role] / len(questions),
                                     rel_tol=0, abs_tol=1e-12) for role in QA_ROLES),
                     "QA partition actual_ratio changed")
    overlay = normalize_experiment_split({
        "protocol": split_id, "split_id": split_id, "seed": manifest["seed"],
        "manifest_identity": manifest_identity, "qa_roles": deepcopy(assignments),
        "source_files": file_identities, "source_scope_identity": source_identity,
        "source_split_by_qa": {qa: row["source_split"] for qa, row in questions.items()},
        "scope": deepcopy(scope), "statistics": stats,
    })
    return QAPartition(questions=questions, experiment_split=overlay)


def _qa_variant_groups(variants: Sequence[Mapping], partition: QAPartition) -> dict:
    overlay = normalize_experiment_split(partition.experiment_split)
    groups = {role: [] for role in QA_ROLES}
    seen = set()
    for raw in variants:
        row = deepcopy(dict(raw))
        variant_id = row.get("variant_id")
        qa_id = row.get("qa_id")
        _require(isinstance(variant_id, str) and variant_id and variant_id not in seen,
                 f"duplicate or invalid partition variant: {variant_id!r}")
        seen.add(variant_id)
        _require(isinstance(qa_id, str) and qa_id in partition.questions,
                 f"{variant_id}: unknown partition QA")
        question = partition.questions[qa_id]
        role = overlay["qa_roles"].get(qa_id)
        public_role = {"learning": "train", "validation": "test"}.get(role) if (
            is_train_test_partition(overlay)) else role
        _require(role in QA_ROLES and public_role == question["experiment_split"],
                 f"{variant_id}: unknown or mismatched partition role")
        for field in ("experiment_role", "experiment_split"):
            _require(field not in row or row[field] == role,
                     f"{variant_id}: mismatched {field}")
        source_split = question["source_split"]
        _require(row.get("split") == source_split
                 and row.get("source_split", source_split) == source_split
                 and overlay["source_split_by_qa"].get(qa_id) == source_split,
                 f"{variant_id}: original source_split changed")
        _require(row.get("scene_id") == question["scene_name"]
                 and row.get("question_type") == canonical_task(question["question_type"])
                 and ("dataset" not in row or row["dataset"] == question["dataset"]),
                 f"{variant_id}: original QA identity changed")
        row["source_split"] = source_split
        row["experiment_role"] = role
        groups[role].append(row)
    _require(not ({row["qa_id"] for row in groups["learning"]}
                  & {row["qa_id"] for row in groups["validation"]}),
             "QA partition learning/gate question overlap")
    return groups


def partition_qa_variants(
    variants: Sequence[Mapping], partition: QAPartition,
) -> tuple[list[dict], list[dict]]:
    """Assign every selected QA's derived variants to exactly one role.

    Scene overlap is allowed. QA overlap across learning and gate is not.
    ``split`` remains catalog provenance, never an alias for the experiment role.
    """
    groups = _qa_variant_groups(variants, partition)
    _require(all(groups.values()), "QA partition requires nonempty learning and validation variants")
    return groups["learning"], groups["validation"]


def select_qa_execution_variants(
    variants: Sequence[Mapping], partition: QAPartition, *, role: str = "validation",
) -> list[dict]:
    """Select a frozen execution role without creating a learning/gate stream."""
    if is_train_test_partition(partition.experiment_split):
        role = {"train": "learning", "test": "validation"}.get(role, role)
    _require(role in QA_ROLES, "unknown QA execution role")
    groups = _qa_variant_groups(variants, partition)
    _require(bool(groups[role]), f"QA partition has no {role} execution variants")
    return groups[role]
