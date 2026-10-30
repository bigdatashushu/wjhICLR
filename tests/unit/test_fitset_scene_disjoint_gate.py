"""Model-free contracts for separately frozen fit-set gate inputs."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import threading

import numpy as np
import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.evaluation import fitset_v11 as fitset
from skill3d.evolution.fitset_stream_v11 import FitSetEvolutionConfig, FrozenSkillProvider
from skill3d.online.runner import OnlineRunConfig
from skill3d.skills.registry import load_active_skills


ROOT = Path(__file__).resolve().parents[2]
LABEL = "synthetic-offline-reference-only"


def _forbidden(*_args, **_kwargs):
    pytest.fail("gate admission must not request models or recompute inputs")


def _row(name, scene, task="object_counting", state="g0m0", split="induction"):
    return {
        "variant_id": f"{name}@{state}", "qa_id": name, "scene_id": scene,
        "question_type": task, "evidence_state": state, "split": split,
        "artifact_ref": f"/frozen/{scene}/artifact.json",
        "slot_key": f"{task}/{state}",
    }


@pytest.fixture
def build(tmp_path, monkeypatch):
    monkeypatch.setattr(fitset, "_source_code_identity", lambda: {"frozen_code": "test"})
    skills, _, snapshot = load_active_skills(ROOT / "skill_library/snapshots")
    learning = [_row("learn-1", "learning-scene")]
    gates = [
        _row("gate-1", "heldout-1", split="inner_validation"),
        _row("gate-2", "heldout-2", split="inner_validation"),
    ]

    def construct(*, variants=None, gate_variants=None, config=None, **extra):
        selected = learning if variants is None else variants
        heldout = gates if gate_variants is None else gate_variants
        kwargs = {
            "output_root": tmp_path / "run",
            "panel": {"proposal_sha256": "a" * 64, "panel_file_sha256": "b" * 64},
            "panel_name": "ready_579", "variants": selected, "gate_variants": heldout,
            "item_provider": _forbidden, "input_receipt_provider": _forbidden,
            "input_contract_identity": {"frozen": "public"},
            "binding_refs": {
                row["variant_id"]: {"key": "c" * 64, "manifest_sha256": "d" * 64}
                for row in [*selected, *heldout]
            },
            "base_cfg": OnlineRunConfig(seed=137), "phase": "accumulation",
            "skills_for_variant": FrozenSkillProvider(skills, {
                "snapshot_ref": snapshot, "manifest_sha256": "e" * 64,
            }),
            "min_free_gib": 0,
            "evolution_config": config or FitSetEvolutionConfig(
                learning_input_mode="full_cases_v1", gate_mode="scene_disjoint", gate_size=2),
            "label_metadata": {
                row["qa_id"]: {
                    "scene_id": row["scene_id"], "question_type": row["question_type"],
                    "options": [], "reference_answer": LABEL,
                } for row in [*selected, *heldout]
            },
            "revision_client_factory": _forbidden, "gate_client_factory": _forbidden,
        }
        kwargs.update(extra)
        return fitset.SystemVariantRunner(**kwargs)

    return construct


def test_gate_lookup_is_merged_but_scheduling_and_manifest_execution_are_learning_only(build):
    runner = build()
    gate_ids = set(runner.gate_variant_rows)
    learning_ids = {row["variant_id"] for row in runner.variants}
    assert gate_ids.isdisjoint(learning_ids)
    assert set(runner.variant_rows) == gate_ids | learning_ids
    assert runner.gate_panels == {
        "object_counting/g0m0": ["gate-1@g0m0", "gate-2@g0m0"],
    }
    assert {row["variant_id"] for shard in runner._planned_slot_shards() for row in shard} == learning_ids
    manifest = runner._manifest()
    assert set(manifest["gate_variant_ids"]) == gate_ids
    assert manifest["gate_variant_rows"] == list(runner.gate_variant_rows.values())
    assert set(manifest["execution"]["variant_ids"]) == learning_ids
    assert manifest["request_input_config"]["evolution_config"]["gate_mode"] == "scene_disjoint"
    assert LABEL not in json.dumps(manifest)


@pytest.mark.parametrize("mode", [
    "analysis_synthesis_v1", "full_cases_v1", "diagnosis_v1", "diagnosis_multimodal_v1",
])
@pytest.mark.parametrize("split", ["inner_validation", "outer_holdout", "", "learning"])
def test_rich_learning_rejects_every_non_induction_split(build, mode, split):
    with pytest.raises(fitset.FitSetError, match="induction-only"):
        build(
            variants=[_row("learn", "scene", split=split)],
            config=FitSetEvolutionConfig(learning_input_mode=mode, gate_mode="scene_disjoint", gate_size=2),
        )


def test_scene_overlap_is_global_across_learning_question_types(build):
    learning = [
        _row("count", "count-scene"),
        _row("size", "heldout-1", task="object_size_estimation"),
    ]
    with pytest.raises(fitset.FitSetError, match="overlaps learning"):
        build(variants=learning)


@pytest.mark.parametrize("gates,message", [
    ([], "requires supplied"),
    ([_row("gate", "heldout")], "incomplete frozen gate_size"),
    ([_row("g1", "heldout"), _row("g2", "heldout")], "distinct scenes"),
    ([_row("gate", "heldout"), _row("gate", "heldout")], "duplicate"),
    ([_row("learn-1", "different-scene")], "duplicate"),
    ([_row("g1", "h1", task="object_size_estimation"),
      _row("g2", "h2", task="object_size_estimation")], "incomplete"),
    ([_row("g1", "h1", split="outer_holdout"), _row("g2", "h2")], "held-out"),
    ([_row("g1", "h1", state="g1m1"), _row("g2", "h2")], "unreachable"),
])
def test_invalid_frozen_gate_is_rejected_before_model_or_input_access(build, gates, message):
    with pytest.raises(fitset.FitSetError, match=message):
        build(gate_variants=gates)


def test_gate_requires_verified_label_and_binding_identities(build):
    with pytest.raises(fitset.FitSetError, match="label metadata"):
        build(label_metadata={})
    with pytest.raises(fitset.FitSetError, match="binding reference"):
        build(binding_refs={})
    row = _row("gate", "heldout")
    row["slot_key"] = "object_counting/g1m0"
    with pytest.raises(fitset.FitSetError, match="slot identity"):
        build(gate_variants=[row])


def test_explicit_legacy_mode_keeps_small_non_induction_same_question_fixtures(build):
    runner = build(
        variants=[_row("legacy", "scene", split="inner_validation")], gate_variants=[],
        config=FitSetEvolutionConfig(learning_input_mode="aggregate_v1", gate_mode="same_question"),
    )
    assert runner.gate_panels == {"object_counting/g0m0": ["legacy@g0m0"]}
    assert runner.gate_variant_rows == {}


def test_same_question_cannot_silently_ignore_supplied_heldout_rows(build):
    with pytest.raises(fitset.FitSetError, match="cannot accept"):
        build(config=FitSetEvolutionConfig(gate_mode="same_question"))


@pytest.mark.parametrize("action", ["run", "preflight"])
def test_all_gate_inputs_are_prepared_before_any_learning_execution(build, monkeypatch, action):
    runner = build()
    events = []

    def prepare(row, *, preflight):
        assert preflight and row["variant_id"] in runner.gate_variant_rows
        events.append(("gate", row["variant_id"]))
        return {
            "identity": {"sha256": "f" * 64}, "state": row["evidence_state"],
            "reference": runner.binding_refs[row["variant_id"]],
        }

    monkeypatch.setattr(runner, "_prepare_one", prepare)
    monkeypatch.setattr(runner, "_execute_process_pending",
                        lambda pending, _accept: events.append(("learning", [r["variant_id"] for r in pending])))
    monkeypatch.setattr(runner, "_execute_preflight_pending",
                        lambda pending, _execute, _accept: events.append(("learning", [r["variant_id"] for r in pending])))
    getattr(runner, action)(resume=False, max_variants=1)
    assert events == [
        ("gate", "gate-1@g0m0"), ("gate", "gate-2@g0m0"),
        ("learning", ["learn-1@g0m0"]),
    ]
    receipts = list((runner.root / "gate_preflight").glob("*.json"))
    assert len(receipts) == 2 and not list((runner.root / "results").glob("*.json"))


def test_parallel_gate_preparation_finishes_every_input_before_learning(build, monkeypatch):
    runner = build(workers=2)
    barrier = threading.Barrier(2)
    prepared = set()
    learning_started = []

    def prepare(row, *, preflight):
        assert preflight and row["variant_id"] in runner.gate_variant_rows
        barrier.wait(timeout=3)
        prepared.add(row["variant_id"])
        return {
            "identity": {"sha256": "f" * 64}, "state": row["evidence_state"],
            "reference": runner.binding_refs[row["variant_id"]],
        }

    def execute(pending, _accept):
        assert prepared == set(runner.gate_variant_rows)
        assert len(list((runner.root / "gate_preflight").glob("*.json"))) == 2
        learning_started.extend(row["variant_id"] for row in pending)

    monkeypatch.setattr(runner, "_prepare_one", prepare)
    monkeypatch.setattr(runner, "_execute_process_pending", execute)
    runner.run(resume=False, max_variants=1)
    assert learning_started == ["learn-1@g0m0"]


@pytest.mark.parametrize("workers", [1, 2])
def test_missing_or_wrong_runtime_gate_evidence_stops_before_learning_requests(build, monkeypatch, workers):
    runner = build(workers=workers)
    monkeypatch.setattr(runner, "_execute_process_pending", _forbidden)

    def fail(*_args, **_kwargs):
        raise fitset.FitSetError("frozen gate runtime evidence state changed")

    monkeypatch.setattr(runner, "_prepare_one", fail)
    with pytest.raises(fitset.FitSetError, match="runtime evidence"):
        runner.run(resume=False)


def test_prepared_gate_state_must_match_the_frozen_slot(build, monkeypatch):
    runner = build()
    monkeypatch.setattr(runner, "_execute_process_pending", _forbidden)
    monkeypatch.setattr(runner, "_prepare_one", lambda *_args, **_kwargs: {"state": "g1m0"})
    with pytest.raises(fitset.FitSetError, match="runtime evidence state changed"):
        runner.run(resume=False)


@pytest.mark.parametrize("field,value", [
    ("learning_input_mode", "diagnosis_multimodal_v1"),
    ("diagnosis_max_cases", 3), ("diagnosis_max_input_chars", 32000),
    ("diagnosis_max_images", 4), ("diagnosis_max_output_tokens", 2048),
])
def test_frozen_request_input_configuration_cannot_change_on_resume(build, field, value):
    runner = build()
    runner.initialize(resume=False)
    runner.evolution_config = replace(runner.evolution_config, **{field: value})
    with pytest.raises(fitset.FitSetError, match="manifest changed"):
        runner.initialize(resume=True)


def test_gate_row_identity_cannot_change_on_resume(build):
    runner = build()
    runner.initialize(resume=False)
    runner.gate_variant_rows["gate-1@g0m0"]["artifact_ref"] = "/different/artifact.json"
    with pytest.raises(fitset.FitSetError, match="manifest changed"):
        runner.initialize(resume=True)


def test_gate_binding_identity_cannot_change_on_resume(build):
    runner = build()
    runner.initialize(resume=False)
    runner.binding_refs["gate-1@g0m0"]["key"] = "f" * 64
    with pytest.raises(fitset.FitSetError, match="manifest changed"):
        runner.initialize(resume=True)


def test_heldout_gate_arm_copies_use_union_lookup_and_do_not_carry_labels(build, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from skill3d.evolution.fitset_stream_v11 import FitSetWorkerEvolution

    runner = build()
    runner.run_identity = "f" * 64
    runner.gate_client_factory = lambda _seed, _root: _forbidden
    worker = FitSetWorkerEvolution(runner, 0, runner.variants)
    parent = next(skill for skill in runner.seed_skills if skill.slot_key == "object_counting/g0m0")
    captured = []

    class StopBeforeRequests(Exception):
        pass

    def capture(_runner, items):
        captured.extend(items)
        raise StopBeforeRequests

    monkeypatch.setattr(fitset.SystemVariantRunner, "_run_gate_episodes", capture)
    with pytest.raises(StopBeforeRequests):
        worker._gate(SimpleNamespace(slot_key=parent.slot_key), parent, parent, tmp_path / "attempt")
    assert len(captured) == 8
    for arm, row in captured:
        assert row["variant_id"] in runner.gate_variant_rows
        assert row["variant_id"] not in {r["variant_id"] for r in runner.variants}
        assert arm.label_metadata == {} and arm.evolution_config is None
        assert LABEL.encode() not in pickle.dumps(arm)
    assert runner.label_metadata["gate-1"]["reference_answer"] == LABEL


def test_offline_input_freezes_public_question_and_real_rgb_without_labels(build):
    runner = build()
    item = load_synthetic_items(
        "induction", question_types=["object_counting"], n_frames=32, frame_size=(8, 8),
    )[0]
    item.episode.ground_truth = LABEL
    row = runner.variants[0]
    runner._freeze_offline_case_input(row, item)
    value = runner.offline_case_input(row)
    assert value["question_text"] == item.episode.question
    assert value["options"] == list(item.episode.options or [])
    assert value["frame_set_hash"] == item.episode.frame_set.frame_set_hash
    assert len(value["image_refs"]) == 32
    assert LABEL not in json.dumps(value)
    from PIL import Image
    for index, ref in enumerate(value["image_refs"]):
        assert set(ref) == {
            "path", "file_sha256", "content_sha256", "image_id", "kind",
            "frame_id", "source_frame_index", "source_hw", "frame_set_hash",
        }
        assert hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() == (
            ref["file_sha256"]) == ref["content_sha256"]
        assert ref["image_id"] == f"{row['variant_id']}:frame:{ref['frame_id']}"
        assert ref["source_hw"] == [8, 8]
        assert ref["source_frame_index"] == item.episode.frame_set.source_frame_indices[index]
        with Image.open(ref["path"]) as image:
            assert np.array_equal(np.asarray(image), item.pixels[index])
    from skill3d.evolution.case_records_v11 import _image_ref
    from skill3d.evolution.diagnosis_v11 import FrozenCaseImageProvider
    image_provider = FrozenCaseImageProvider([runner.root])
    for ref in value["image_refs"]:
        normalized = _image_ref(ref, source_ref="offline_input.json")
        assert image_provider(None, normalized)["content_sha256"] == ref["file_sha256"]
    assert LABEL.encode() not in pickle.dumps(value)
    runner._freeze_offline_case_input(row, item)
    Path(value["image_refs"][0]["path"]).write_bytes(b"changed")
    # Either the transport or the offline reader can detect the same mutation.
    with pytest.raises(fitset.FitSetError, match="(offline image|logical file content) changed"):
        runner.offline_case_input(row)


def test_offline_input_missing_does_not_decode_download_or_reconstruct(build):
    runner = build()
    with pytest.raises(fitset.FitSetError, match="offline_input.json is missing"):
        runner.offline_case_input(runner.variants[0])


def test_offline_input_rejects_unknown_data_fields_and_changed_completed_input(build):
    runner = build()
    item = load_synthetic_items(
        "induction", question_types=["object_counting"], n_frames=32, frame_size=(8, 8),
    )[0]
    row = runner.variants[0]
    runner._freeze_offline_case_input(row, item)
    path = runner.root / "variants" / fitset._variant_key(row["variant_id"]) / "offline_input.json"
    source = json.loads(path.read_text())
    fitset._write_create_once(runner._result_path(row["variant_id"]), {
        "offline_input_sha256": fitset.file_identity(path)["sha256"],
    })
    assert runner.offline_case_input(row)["question_text"] == item.episode.question
    path.write_text(json.dumps({**source, "ground_truth": LABEL}))
    with pytest.raises(fitset.FitSetError, match="identity changed"):
        runner.offline_case_input(row)
    path.write_text(json.dumps({**source, "question_text": "changed"}))
    with pytest.raises(fitset.FitSetError, match="differs from completed result"):
        runner.offline_case_input(row)
