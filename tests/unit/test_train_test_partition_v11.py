"""Opt-in train/test runtime contracts; public fixtures and stub clients only."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skill3d.evaluation.fitset_v11 import FitSetError, SystemVariantRunner
from skill3d.evaluation.qa_partition_v11 import (
    QAPartitionError, TRAIN_TEST_PARTITION_SCHEMA, load_qa_partition_manifest,
    partition_qa_variants, select_qa_execution_variants,
)
from skill3d.evolution.fitset_stream_v11 import FitSetWorkerEvolution, FitSetEvolutionConfig
from skill3d.schemas.data_access import (
    DATA_SPLITS, DataAccessRecord, learning_authorization_for, normalize_experiment_split,
    validate_learning_authorization,
)

from test_run_fitset_v11_cli_qa_partition import cli, cli_case
from test_fitset_stream_v11 import (
    ProbeRunner, full_evidence_factory, gate_factory, revision_factory,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(case, role=None):
    if role:
        case.manifest["files"][role]["sha256"] = _write_jsonl(
            case.role_paths[role], case.documents[role])
    case.path.write_text(json.dumps(case.manifest))


@pytest.fixture
def train_test_case(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text('{"synthetic":"hashed offline labels, not loaded"}\n')
    rows = [
        {"id": i, "qa_id": str(i), "dataset": "scannet",
         "scene_name": f"scene-{i % 2}",
         "question_type": "object_size_estimation" if 20 <= i < 24 else "object_counting",
         "question": f"How many chairs are visible in room {i}?", "options": None,
         "source_split": DATA_SPLITS[i % 4], "split": DATA_SPLITS[i % 4],
         "benchmark_split": "test", "experiment_split": "train" if i < 24 else "test"}
        for i in range(32)
    ]
    source_path = tmp_path / "questions.jsonl"
    manifest = {
        "schema": TRAIN_TEST_PARTITION_SCHEMA, "split_id": "synthetic-train-test", "seed": 137,
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "questions_file": source_path.name, "questions_sha256": _write_jsonl(source_path, rows),
        "scene_names": ["scene-0", "scene-1"], "source_split_preserved": True, "files": {},
    }
    documents, role_paths = {}, {}
    for role in ("train", "test"):
        documents[role] = deepcopy([row for row in rows if row["experiment_split"] == role])
        role_paths[role] = tmp_path / f"{role}.jsonl"
        manifest["files"][role] = {
            "path": role_paths[role].name,
            "sha256": _write_jsonl(role_paths[role], documents[role]),
        }
    case = SimpleNamespace(
        path=tmp_path / "split.json", manifest=manifest, metadata=metadata,
        rows=rows, documents=documents, role_paths=role_paths, source_path=source_path,
        variants=[
            {"variant_id": f"{row['qa_id']}@{state}", "qa_id": row["qa_id"],
             "scene_id": row["scene_name"], "question_type": row["question_type"],
             "split": row["source_split"], "evidence_state": state,
             "artifact_ref": f"/synthetic/{row['scene_name']}/artifact.json",
             "system_input_status": "ready"}
            for row in rows for state in ("g0m0", "g1m0")
        ],
    )
    _save(case)
    return case


@pytest.fixture
def qa_partition_case(train_test_case):
    # Reuse the sealed CLI fixture, with the new public protocol as input.
    return train_test_case


@pytest.fixture
def runtime(cli_case):
    value = cli_case
    assert cli.main([*value.argv, "--experiment-split-manifest", str(value.case.path),
                     "--gate-mode", "train_sample"]) == 0
    return SystemVariantRunner(**value.runners[-1].kwargs)


def _request(parent, attempt=1):
    return SimpleNamespace(
        slot_key=parent.slot_key, question_type=parent.question_type,
        evidence_state=parent.evidence_state, parent_skill_version=f"{parent.skill_id}@{parent.version}",
        attempt_index=attempt, cases=[SimpleNamespace(case_id=f"case-{i}") for i in range(12)],
    )


def test_public_files_map_to_hashed_core_roles_and_preserve_all_four_sources(train_test_case):
    case = train_test_case
    partition = load_qa_partition_manifest(case.path, metadata_path=case.metadata)
    overlay = partition.experiment_split
    assert overlay == normalize_experiment_split(overlay)
    assert overlay["protocol"] == TRAIN_TEST_PARTITION_SCHEMA
    assert set(overlay["source_files"]) == {"train", "test"}
    assert set(overlay["source_split_by_qa"].values()) == set(DATA_SPLITS)
    learning, validation = partition_qa_variants(case.variants, partition)
    assert len(learning) == 48 and len(validation) == 16
    assert {row["experiment_split"] for row in partition.questions.values()} == {"train", "test"}
    assert {row["experiment_role"] for row in learning} == {"learning"}
    assert {row["experiment_role"] for row in validation} == {"validation"}
    assert all(row["source_split"] == row["split"] for row in [*learning, *validation])
    assert select_qa_execution_variants(case.variants, partition, role="test") == validation
    assert select_qa_execution_variants(case.variants, partition, role="train") == learning


@pytest.mark.parametrize("file", ["train", "test", "questions", "metadata"])
def test_all_input_hashes_are_verified(train_test_case, file):
    case = train_test_case
    path = (case.role_paths[file] if file in case.role_paths else
            case.source_path if file == "questions" else case.metadata)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(QAPartitionError, match="hash changed"):
        load_qa_partition_manifest(case.path, metadata_path=case.metadata)


@pytest.mark.parametrize("field,value", [
    ("source_split", "inner_validation"), ("split", "inner_validation"),
    ("id", 999), ("benchmark_split", "train"), ("experiment_split", "test"),
    ("question", "changed"), ("ground_truth", "forbidden label"),
])
def test_rehashed_public_role_cannot_change_identity_or_add_labels(train_test_case, field, value):
    case = train_test_case
    case.documents["train"][0][field] = value
    _save(case, "train")
    with pytest.raises(QAPartitionError):
        load_qa_partition_manifest(case.path)


def test_missing_qa_duplicate_qa_and_missing_scene_are_rejected(train_test_case):
    case = train_test_case
    original = deepcopy(case.documents["test"])
    case.documents["test"].pop()
    _save(case, "test")
    with pytest.raises(QAPartitionError, match="union differs"):
        load_qa_partition_manifest(case.path)
    case.documents["test"] = [deepcopy(original[0]), deepcopy(original[0])]
    _save(case, "test")
    with pytest.raises(QAPartitionError, match="duplicate"):
        load_qa_partition_manifest(case.path)
    case.documents["test"] = [row for row in original if row["scene_name"] == "scene-0"]
    _save(case, "test")
    with pytest.raises(QAPartitionError, match="every declared scene"):
        load_qa_partition_manifest(case.path)


def test_new_authorization_is_bound_to_roles_sources_and_protocol(train_test_case):
    overlay = load_qa_partition_manifest(train_test_case.path).experiment_split
    for i, source in enumerate(DATA_SPLITS):
        receipt = learning_authorization_for(overlay, [str(i)])
        validate_learning_authorization(receipt, [str(i)], overlay)
        assert receipt["gate_mode"] == "train_sample"
        access = DataAccessRecord(
            record_id="test", split=source, purpose="skill_induction",
            component_role="inducer", n_items=1, label_access=True,
            learning_authorization=receipt)
        assert access.split == source
        changed = deepcopy(receipt)
        changed["source_split_by_qa"][str(i)] = DATA_SPLITS[(i + 1) % 4]
        with pytest.raises(ValueError, match="frozen QA role map"):
            validate_learning_authorization(changed, [str(i)], overlay)
        with pytest.raises(ValueError, match="source_split changed"):
            DataAccessRecord.model_validate({**access.model_dump(), "split": DATA_SPLITS[(i + 1) % 4]})
    with pytest.raises(ValueError, match="cannot enter learning"):
        learning_authorization_for(overlay, ["24"])
    legacy = normalize_experiment_split({
        "protocol": "legacy-qa", "manifest_identity": {"sha256": "a" * 64},
        "qa_roles": {"q": "learning"},
    })
    receipt = learning_authorization_for(legacy, ["q"])
    for source in ("outer_holdout", "final_test"):
        with pytest.raises(ValueError, match="legacy"):
            DataAccessRecord(
                record_id="test", split=source, purpose="skill_induction",
                component_role="inducer", n_items=1, label_access=True,
                learning_authorization=receipt)
    receipt["gate_mode"] = "train_sample"
    with pytest.raises(ValueError, match="invalid explicit"):
        validate_learning_authorization(receipt, ["q"])


def test_runtime_only_schedules_train_and_accepts_no_separate_gate(runtime, cli_case):
    assert len(runtime.variants) == 48 and runtime.gate_variant_rows == {}
    assert set(runtime.label_metadata) == {str(i) for i in range(24)}
    assert {row["qa_id"] for row in runtime.execution_variants} == set(runtime.label_metadata)
    assert len(runtime.gate_panels["object_counting/g0m0"]) == 20
    assert "training QA" in runtime.gate_deferred_reasons["object_size_estimation/g0m0"]
    kwargs = dict(cli_case.runners[-1].kwargs)
    kwargs["gate_variants"] = [dict(kwargs["variants"][0], variant_id="separate")]
    with pytest.raises(FitSetError, match="separate gate"):
        SystemVariantRunner(**kwargs)
    kwargs = dict(cli_case.runners[-1].kwargs)
    kwargs["experiment_split"] = None
    with pytest.raises(FitSetError, match="hashed train/test"):
        SystemVariantRunner(**kwargs)
    kwargs = dict(cli_case.runners[-1].kwargs)
    kwargs["variants"] = deepcopy(kwargs["variants"])
    kwargs["variants"][0]["split"] = "final_test"
    kwargs["variants"][0]["source_split"] = "final_test"
    with pytest.raises(FitSetError, match="source catalog"):
        SystemVariantRunner(**kwargs)


def test_sample_plan_is_unique_exact_slot_deterministic_and_resume_checked(runtime, tmp_path):
    runtime.run_identity = "a" * 64
    worker = FitSetWorkerEvolution(runtime, 0, runtime.variants)
    parent = next(s for s in runtime.seed_skills if s.slot_key == "object_counting/g0m0")
    request = _request(parent)
    root = tmp_path / "attempt"
    panel = worker._train_sample_panel(request, parent, root)
    assert len(panel) == len({worker.rows[value]["qa_id"] for value in panel}) == 8
    assert all(worker.rows[value]["slot_key"] == request.slot_key for value in panel)
    assert all(int(worker.rows[value]["qa_id"]) < 20 for value in panel)
    assert worker._train_sample_panel(request, parent, root) == panel
    plan = json.loads((root / "gate_plan.json").read_bytes())
    assert plan["gate_seeds"] == [137, 138]
    other = worker._train_sample_panel(_request(parent, 2), parent, tmp_path / "attempt-2")
    assert other != panel
    plan["panel_ids"].reverse()
    (root / "gate_plan.json").write_text(json.dumps(plan))
    with pytest.raises(FitSetError, match="plan changed"):
        worker._train_sample_panel(request, parent, root)


def test_original_final_source_is_admitted_only_by_the_new_current_role(runtime):
    row = next(row for row in runtime.variants if row["split"] == "final_test")
    assert runtime._allow_original_final_test(row)
    before = runtime.experiment_split
    runtime.experiment_split = None
    assert not runtime._allow_original_final_test(row)
    runtime._train_sample_gate = True
    runtime._train_sample_partition = before
    assert runtime._allow_original_final_test(row)
    changed = {**row, "split": "outer_holdout"}
    assert not runtime._allow_original_final_test(changed)
    qa = next(qa for qa, role in before["qa_roles"].items() if role == "validation")
    assert not runtime._allow_original_final_test({**row, "qa_id": qa})


def test_both_arms_and_seeds_share_persisted_train_panel_without_labels(runtime, tmp_path, monkeypatch):
    runtime.run_identity = "a" * 64
    runtime.published_inputs = None
    runtime.gate_client_factory = lambda _seed, _root: None
    worker = FitSetWorkerEvolution(runtime, 0, runtime.variants)
    parent = next(s for s in runtime.seed_skills if s.slot_key == "object_counting/g0m0")
    root = tmp_path / "attempt"
    captured = []

    class StopBeforeRequests(Exception):
        pass

    def capture(items):
        assert (root / "gate_plan.json").is_file()
        assert (root / "gate_schedule.json").is_file()
        captured.extend(items)
        raise StopBeforeRequests

    monkeypatch.setattr(runtime, "_run_gate_episodes", capture)
    for _ in range(2):
        captured.clear()
        with pytest.raises(StopBeforeRequests):
            worker._gate(_request(parent), parent, parent, root)
        panel = json.loads((root / "gate_plan.json").read_bytes())["panel_ids"]
        assert len(captured) == 32
        for seed in (137, 138):
            for arm in ("parent", "candidate"):
                rows = [row for runner, row in captured if runner.root == root / "gate" / str(seed) / arm]
                assert [row["variant_id"] for row in rows] == panel
        assert all(runner.label_metadata == {} and runner.experiment_split is None
                   and runner.evolution_config is None and runner._train_sample_gate
                   and not runner.capture_offline_case_input for runner, _row in captured)


@pytest.mark.parametrize("role", ["validation", "test"])
def test_final_test_executes_only_test_without_learning(cli_case, role):
    value = cli_case
    assert cli.main([*value.argv, "--phase", "final_system", "--skill-bundle", "synthetic-bundle",
                     "--experiment-split-manifest", str(value.case.path),
                     "--execution-role", role]) == 0
    runtime = SystemVariantRunner(**value.runners[-1].kwargs)
    assert {row["qa_id"] for row in runtime.variants} == {str(i) for i in range(24, 32)}
    assert {row["split"] for row in runtime.variants} == set(DATA_SPLITS)
    assert runtime.evolution_config is None and runtime.label_metadata == {}
    assert runtime.gate_variant_rows == {}
    overlay = runtime.experiment_split
    assert overlay["execution_role"] == "validation"
    with pytest.raises(ValueError, match="cannot enter learning"):
        learning_authorization_for(overlay, ["0"])
    with pytest.raises(FitSetError, match="cannot enter learning"):
        runtime.learning_authorization(runtime.variants[0])


@pytest.mark.parametrize("opts", [
    ["--gate-mode", "train_sample"],
    ["--experiment-split-manifest", "manifest", "--gate-mode", "qa_disjoint"],
    ["--experiment-split-manifest", "manifest", "--gate-mode", "train_sample",
     "--gate-variant-ids", "24@g0m0"],
    ["--experiment-split-manifest", "manifest", "--gate-mode", "train_sample",
     "--variant-ids", "24@g0m0"],
])
def test_cli_rejects_missing_opt_in_legacy_gate_and_test_selection(cli_case, opts):
    opts = [str(cli_case.case.path) if value == "manifest" else value for value in opts]
    assert cli.main([*cli_case.argv, *opts]) == 2
    assert cli_case.runners == [] and cli_case.config_reads == []


def test_rich_all_source_cases_revise_every_twelve_and_resume_without_test(cli_case):
    value = cli_case
    assert cli.main([*value.argv, "--experiment-split-manifest", str(value.case.path),
                     "--gate-mode", "train_sample", "--learning-input-mode", "full_cases_v1"]) == 0
    kwargs = dict(value.runners[-1].kwargs)
    kwargs.update(
        variants=[row for row in kwargs["variants"]
                  if row["question_type"] == "object_counting" and row["evidence_state"] == "g0m0"],
        client_factory=None, published_inputs=None, min_free_gib=0,
        revision_client_factory=full_evidence_factory, gate_client_factory=gate_factory,
        label_metadata={qa: {**label, "options": [], "reference_answer": "2"}
                        for qa, label in kwargs["label_metadata"].items()},
    )
    kwargs["evolution_config"] = replace(kwargs["evolution_config"], scheduling_mode="batch_v1")
    runtime = ProbeRunner(**kwargs)
    summary = runtime.run(resume=False)
    assert summary["completed_variants"] == 20 and summary["status"] == "complete"
    bundles = list(runtime.root.glob("workers/*/campaigns/*/experience_bundle.json"))
    assert len(bundles) == 1
    bundle = json.loads(bundles[0].read_bytes())
    assert len(bundle["cases"]) == 12
    assert set(bundle["source_split"]) == set(DATA_SPLITS)
    assert all(int(case["qa_id"]) < 20 for case in bundle["cases"])
    assert all(case["learning_authorization"]["gate_mode"] == "train_sample" for case in bundle["cases"])
    assert (bundles[0].parent / "gate_plan.json").is_file()
    assert runtime.run(resume=True)["completed_variants"] == 20
    assert list(runtime.root.glob("workers/*/campaigns/*/experience_bundle.json")) == bundles


def test_train_gate_stub_promotion_is_train_only_and_reuses_results_on_resume(cli_case):
    value = cli_case
    assert cli.main([*value.argv, "--experiment-split-manifest", str(value.case.path),
                     "--gate-mode", "train_sample", "--learning-input-mode", "aggregate_v1"]) == 0
    kwargs = dict(value.runners[-1].kwargs)
    kwargs.update(
        variants=[row for row in kwargs["variants"]
                  if row["question_type"] == "object_counting" and row["evidence_state"] == "g0m0"],
        client_factory=None, published_inputs=None, min_free_gib=0,
        revision_client_factory=revision_factory, gate_client_factory=gate_factory,
        label_metadata={qa: {**label, "options": [], "reference_answer": "2"}
                        for qa, label in kwargs["label_metadata"].items()},
    )
    kwargs["evolution_config"] = replace(kwargs["evolution_config"], scheduling_mode="batch_v1")
    runtime = ProbeRunner(**kwargs)
    assert runtime.run(resume=False)["status"] == "complete"
    decisions = list(runtime.root.glob("workers/*/campaigns/*/decision.json"))
    assert len(decisions) == 1
    decision = json.loads(decisions[0].read_bytes())
    assert decision["passed"]
    assert len(decision["panel_ids"]) == 8
    assert all(int(value.split("@")[0]) < 20 for value in decision["panel_ids"])
    results = list(runtime.root.glob("workers/*/campaigns/*/gate/*/*/results/*.json"))
    assert len(results) == 32
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in results}
    assert runtime.run(resume=True)["status"] == "complete"
    assert hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in results}


def test_new_mode_keeps_frozen_dual_seeds_and_twelve_trace_batch():
    assert FitSetEvolutionConfig(gate_mode="train_sample").batch_size == 12
    with pytest.raises(ValueError, match="137/138"):
        FitSetEvolutionConfig(gate_mode="train_sample", gate_seeds=(138, 139))
    with pytest.raises(ValueError):
        FitSetEvolutionConfig(gate_mode="train_sample", batch_size=8)
