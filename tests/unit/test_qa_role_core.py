"""Model-free QA role overlay, canonical evidence and deferred admission tests."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

from skill3d.evaluation import fitset_v11 as fitset
from skill3d.evolution.case_records_v11 import (
    V11ExperienceBuildError, case_from_fitset_outcome,
)
from skill3d.evolution.fitset_revision_v11 import FitSetSlotReviser
from skill3d.evolution.fitset_stream_v11 import (
    FitSetEvolutionConfig, FitSetWorkerEvolution, FrozenSkillProvider, freeze_fitset_stream,
)
from skill3d.evolution.revision_input_v11 import render_case_evidence
from skill3d.schemas import DataAccessRecord, V11ExperienceBundle, V11ExperienceCase
from skill3d.schemas.data_access import (
    learning_authorization_for, normalize_experiment_split,
)
from skill3d.online.runner import OnlineRunConfig
from skill3d.skills.registry import load_active_skills
from test_fitset_executor_boundaries import (
    PublicInputRunner, forbidden, public_item, public_solver_factory,
)
from test_v11_case_records import _inputs


ROOT = Path(__file__).resolve().parents[2]


def overlay(learning, validation=()):
    return {
        "protocol": "524learning225validation",
        "qa_roles": {**{qa: "learning" for qa in learning},
                     **{qa: "validation" for qa in validation}},
        "manifest_identity": {"sha256": "a" * 64, "schema_version": "synthetic/1.0"},
    }


def row(qa, *, split="inner_validation", state="g0m0", task="object_counting"):
    return {
        "variant_id": f"{qa}@{state}", "qa_id": qa, "scene_id": "overlapping-scene",
        "question_type": task, "evidence_state": state, "split": split,
        "source_split": split,
    }


def build(tmp_path, *, n=1, gates=8, learning=None, heldout=None, role_map=None, **extra):
    learning = [row(f"learn-{i}") for i in range(n)] if learning is None else learning
    heldout = [row(f"gate-{i}") for i in range(gates)] if heldout is None else heldout
    skills, _, snapshot = load_active_skills(ROOT / "skill_library/snapshots")
    kwargs = dict(
        output_root=tmp_path / "run",
        panel={"proposal_sha256": "a" * 64, "panel_file_sha256": "b" * 64},
        panel_name="synthetic", variants=learning, gate_variants=heldout,
        item_provider=public_item, input_receipt_provider=forbidden,
        input_contract_identity={"public": "synthetic", "offline_metadata": {"sha256": "f" * 64}},
        published_inputs=SimpleNamespace(), client_factory=public_solver_factory,
        binding_refs={r["variant_id"]: {
            "key": "d" * 64, "manifest_sha256": "e" * 64, "request_identity": {"seed": 137},
        } for r in [*learning, *heldout]},
        base_cfg=OnlineRunConfig(seed=137), phase="accumulation",
        skills_for_variant=FrozenSkillProvider(
            skills, {"snapshot_ref": snapshot, "manifest_sha256": "e" * 64}),
        min_free_gib=0, scheduler="episode_pool", workers=1, skill_owners=1,
        evolution_config=FitSetEvolutionConfig(
            learning_input_mode="full_cases_v1", gate_mode="qa_disjoint"),
        label_metadata={r["qa_id"]: {
            "scene_id": r["scene_id"], "question_type": r["question_type"],
            "options": [], "reference_answer": "2",
        } for r in [*learning, *heldout]},
        revision_client_factory=forbidden, gate_client_factory=forbidden,
        experiment_split=role_map if role_map is not None else overlay(
            [r["qa_id"] for r in learning], [r["qa_id"] for r in heldout]),
    )
    kwargs.update(extra)
    return PublicInputRunner(**kwargs)


def canonical(qa="qa-1", source="inner_validation", role_map=None):
    inputs = _inputs()
    inputs["variant"].update(
        qa_id=qa, variant_id=f"{qa}@g1m1", split=source, source_split=source)
    inputs["outcome"].update(qa_id=qa, split=source, source_split=source)
    inputs["label"]["qa_id"] = qa
    return case_from_fitset_outcome(
        **inputs, experiment_split=role_map if role_map is not None else overlay([qa]))


def bundle(cases, role_map):
    sources = sorted({case.source_split for case in cases})
    records = []
    for source in sources:
        selected = [case for case in cases if case.source_split == source]
        records.append(DataAccessRecord(
            record_id=f"synthetic-{source}", at="synthetic-read-time", split=source,
            purpose="skill_induction", component_role="inducer", n_items=len(selected),
            label_access=True, input_manifest_sha256=fitset.content_sha256([
                case.model_dump(mode="json") for case in selected]),
            learning_authorization=learning_authorization_for(
                role_map, [case.qa_id for case in selected]),
        ))
    parent = cases[0]
    return V11ExperienceBundle(
        campaign_id="synthetic", parent_snapshot_id="synthetic",
        parent_skill_version=parent.delivered_skill_version,
        parent_content_sha256=parent.delivered_content_sha256,
        question_type=parent.question_type, evidence_state=parent.evidence_state,
        source_split=sources[0] if len(sources) == 1 else sources,
        split="learning", label_access=True, source_run_ref="synthetic",
        learning_authorization=learning_authorization_for(role_map, [c.qa_id for c in cases]),
        label_access_record=records[0], label_access_records=records, cases=cases,
    )


def test_all_validation_variants_are_used_despite_scene_overlap(tmp_path):
    runner = build(tmp_path, gates=10)
    assert runner.gate_panels == {
        "object_counting/g0m0": [f"gate-{i}@g0m0" for i in range(10)]}
    assert runner.gate_deferred_reasons == {}
    assert runner.variants[0]["split"] == runner.variants[0]["source_split"] == "inner_validation"
    manifest = runner.initialize(resume=False)
    assert manifest["experiment_split"]["qa_roles"]["learn-0"] == "learning"
    assert manifest["experiment_split"]["qa_roles_sha256"]
    assert manifest["experiment_split_sha256"] == runner.learning_authorization(
        runner.variants[0])["experiment_split_sha256"]
    assert manifest["evolution"]["gate_selection_policy"] == "all_validation_variants_per_slot"
    assert manifest["evolution"]["config"]["gate_size"] == 8
    assert manifest["evolution"]["config"]["gate_seeds"] == [137, 138]
    assert manifest["evolution"]["config"]["learning_input_mode"] == "full_cases_v1"


@pytest.mark.parametrize("state,task", [
    ("g1m0", "object_counting"), ("g0m0", "object_size_estimation"),
])
def test_qa_overlap_is_rejected_across_variant_ids_and_slots(tmp_path, state, task):
    gate = row("learn-0", state=state, task=task)
    gate["variant_id"] = "different-variant-id"
    with pytest.raises(fitset.FitSetError, match="gate QA overlaps"):
        build(tmp_path, heldout=[gate])


@pytest.mark.parametrize("bad", ["missing", "validation", "renamed_source", "outer"])
def test_ingest_role_and_source_checks_precede_input_or_models(tmp_path, bad):
    learning = [row("learn-0")]
    roles = overlay(["learn-0"], ["gate-0"])
    if bad == "missing":
        del roles["qa_roles"]["learn-0"]
    elif bad == "validation":
        roles["qa_roles"]["learn-0"] = "validation"
    elif bad == "renamed_source":
        learning[0]["source_split"] = "induction"
    else:
        learning[0]["source_split"] = learning[0]["split"] = "outer_holdout"
    with pytest.raises(fitset.FitSetError):
        build(tmp_path, learning=learning, heldout=[row("gate-0")], role_map=roles)


@pytest.mark.parametrize("gates", [0, 7])
def test_missing_or_small_gate_continues_full_case_accumulation_and_resume(tmp_path, gates):
    runner = build(tmp_path, n=25, gates=gates)
    key = "object_counting/g0m0"
    assert key in runner.gate_deferred_reasons
    assert runner.run(resume=False)["completed_variants"] == 25
    worker = runner.root / "workers/00"
    state = json.loads((worker / "buffer/stream.json").read_text())
    slot = state["slots"][key]
    assert slot["deferred_reason"] == ""
    assert slot["new_since_attempt"] == 1 and len(slot["case_refs"]) == 25
    assert len(state["attempts"]) == 2
    assert all(attempt["outcome"] == "no_change" for attempt in state["attempts"])
    deferred = list(worker.glob("campaigns/*/gate_deferred.json"))
    assert len(deferred) == 2
    assert all(json.loads(path.read_text())["admission_status"] == "deferred" for path in deferred)
    assert len(list((worker / "canonical_cases").glob("*.json"))) == 25
    assert not list(worker.glob("campaigns/*/revision"))
    assert not list(worker.glob("campaigns/*/gate"))
    parent, _ = FitSetWorkerEvolution(runner, 0, runner.variants).provider(runner.variants[0])
    assert parent[0].content_sha256 == next(
        skill.content_sha256 for skill in runner.seed_skills if skill.slot_key == key)
    assert build(tmp_path, n=25, gates=gates).run(resume=True)["completed_variants"] == 25
    frozen = freeze_fitset_stream(runner.root, tmp_path / "frozen")
    assert frozen["manifest_sha256"]


def test_sparse_learning_does_not_tail_revise(tmp_path):
    runner = build(tmp_path, n=11)
    runner.run(resume=False)
    state = json.loads((runner.root / "workers/00/buffer/stream.json").read_text())
    assert state["attempts"] == []
    assert len(state["slots"]["object_counting/g0m0"]["case_refs"]) == 11


def test_full_case_revision_windows_both_source_splits_without_aggregation(tmp_path, monkeypatch):
    learning = [row(f"learn-{i}", split=(
        "induction" if i % 2 == 0 else "inner_validation")) for i in range(25)]
    runner = build(tmp_path, learning=learning)
    runner.revision_client_factory = lambda: SimpleNamespace(chat=forbidden)
    observed = []

    def receive(reviser, *, experience, **_kwargs):
        assert reviser.experiment_split == runner.experiment_split
        assert experience.source_split == ["induction", "inner_validation"]
        assert {access.split for access in experience.label_access_records} == {
            "induction", "inner_validation"}
        assert all(case.rounds and case.response_texts and case.reference_answer == "2"
                   for case in experience.cases)
        render_case_evidence(experience, experiment_split=reviser.experiment_split)
        observed.append(len(experience.cases))
        return None

    monkeypatch.setattr(FitSetSlotReviser, "_revise_slot", receive)
    assert runner.run(resume=False)["completed_variants"] == 25
    assert observed == [12, 12]
    paths = list(runner.root.glob("workers/00/campaigns/*/experience_bundle.json"))
    assert len(paths) == 2
    assert sorted(len(json.loads(path.read_text())["cases"]) for path in paths) == [12, 12]


def test_final_system_accepts_both_source_splits_without_overlay_or_evolution(tmp_path):
    runner = build(
        tmp_path, learning=[row("validate-1", split="induction"), row("validate-2")],
        heldout=[], experiment_split=None, evolution_config=None, phase="final_system",
        label_metadata={},
    )
    manifest = runner.initialize(resume=False)
    assert manifest["evolution"] is None and "experiment_split" not in manifest
    assert not runner.capture_offline_case_input
    assert {variant["split"] for variant in runner.variants} == {
        "induction", "inner_validation"}


def test_real_final_constructor_accepts_validation_overlay_and_freezes_identity(tmp_path):
    roles = overlay(["unused-learning"], ["validate-1", "validate-2"])
    roles["execution_role"] = "validation"
    runner = build(
        tmp_path, learning=[row("validate-1", split="induction"), row("validate-2")],
        heldout=[], experiment_split=normalize_experiment_split(roles),
        evolution_config=None, phase="final_system", label_metadata={},
    )
    manifest = runner.initialize(resume=False)
    assert manifest["experiment_split"]["execution_role"] == "validation"
    assert manifest["experiment_split_sha256"] == runner.experiment_split["identity_sha256"]
    assert manifest["evolution"] is None
    assert manifest["gate_variant_ids"] == []
    assert runner.evolution_config is None and not runner.capture_offline_case_input
    assert runner.gate_panels == {} and runner.seed_skills == []
    roles["manifest_identity"]["sha256"] = "f" * 64
    runner.experiment_split = normalize_experiment_split(roles)
    with pytest.raises(fitset.FitSetError, match="manifest changed"):
        runner.initialize(resume=True)


@pytest.mark.parametrize("bad", ["execution_role", "qa_role", "gate", "evolution"])
def test_real_final_constructor_refuses_non_validation_or_evolution_overlay(tmp_path, bad):
    roles = overlay(["unused-learning"], ["validate-1", "validate-2"])
    roles["execution_role"] = "learning" if bad == "execution_role" else "validation"
    if bad == "qa_role":
        roles["qa_roles"]["validate-1"] = "learning"
    with pytest.raises(fitset.FitSetError, match="final_system"):
        build(
            tmp_path, learning=[row("validate-1", split="induction"), row("validate-2")],
            heldout=[row("validate-2", state="g1m0")] if bad == "gate" else [],
            experiment_split=normalize_experiment_split(roles),
            evolution_config=FitSetEvolutionConfig(gate_mode="qa_disjoint")
            if bad == "evolution" else None,
            phase="final_system", label_metadata={},
        )


def test_repeated_variants_cannot_satisfy_eight_distinct_validation_questions(tmp_path):
    gates = [row(f"gate-{i}") for i in range(7)]
    gates.append({**gates[0], "variant_id": "second-variant-for-gate-0"})
    runner = build(tmp_path, heldout=gates)
    key = "object_counting/g0m0"
    assert len(runner.gate_panels[key]) == 8
    assert key in runner.gate_deferred_reasons
    runner.initialize(resume=False)
    worker = FitSetWorkerEvolution(runner, 0, runner.variants)
    parent = worker.provider(runner.variants[0])[0][0]
    assert worker._gate(
        SimpleNamespace(slot_key=key), parent, parent, tmp_path / "gate")["passed"] is False


def test_validation_ingest_refused_before_outcome_or_label_read(tmp_path):
    runner = build(tmp_path)
    runner.initialize(resume=False)
    worker = FitSetWorkerEvolution(runner, 0, runner.variants)
    with pytest.raises(fitset.FitSetError, match="cannot enter learning ingest"):
        worker.ingest_result(next(iter(runner.gate_variant_rows.values())), {})


def test_inner_source_case_and_label_access_keep_source_and_original_rounds():
    case = canonical()
    assert case.source_split == "inner_validation"
    assert case.learning_authorization["experiment_role"] == "learning"
    access = DataAccessRecord.model_validate(
        case.rounds[0]["experience_provenance"]["offline_label_access_record"])
    assert access.split == "inner_validation" and access.label_access
    assert access.learning_authorization == case.learning_authorization
    assert case.rounds[1]["results"] == _inputs()["outcome"]["rounds"][1]["results"]
    assert case.response_texts == [r["response_text"] for r in _inputs()["outcome"]["rounds"]]
    assert case.rounds[0]["experience_provenance"]["result_links"]
    assert case.reference_answer == "B"
    damaged = case.model_dump(mode="json")
    damaged["learning_authorization"] = {}
    with pytest.raises(ValueError, match="explicit learning authorization"):
        V11ExperienceCase.model_validate(damaged)
    with pytest.raises(ValueError, match="只对学习划分"):
        DataAccessRecord(
            record_id="unscoped", split="inner_validation", purpose="skill_induction",
            component_role="inducer", label_access=True)


def test_validation_case_refused_before_any_label_access():
    inputs = _inputs()
    inputs["variant"]["split"] = "inner_validation"

    class Unreadable(dict):
        def get(self, *_args):
            pytest.fail("validation labels must not be read by the case builder")

    inputs["label"] = Unreadable()
    with pytest.raises(V11ExperienceBuildError, match="validation or unmapped QA"):
        case_from_fitset_outcome(**inputs, experiment_split=overlay([], ["qa-1"]))


def test_declared_validation_role_cannot_override_learning_overlay(tmp_path):
    learning = [row("learn-0")]
    learning[0]["experiment_role"] = "validation"
    with pytest.raises(fitset.FitSetError, match="declared QA role"):
        build(tmp_path, learning=learning)
    inputs = _inputs()
    inputs["variant"]["experiment_role"] = "validation"
    with pytest.raises(V11ExperienceBuildError, match="Validation role"):
        case_from_fitset_outcome(**inputs, experiment_split=overlay(["qa-1"]))
    with pytest.raises(fitset.FitSetError, match="Validation role"):
        FitSetSlotReviser(
            output_root=tmp_path / "reviser", panel={}, panel_name="synthetic",
            variants=[inputs["variant"]], accumulation_root=tmp_path, label_metadata={},
            library_root=ROOT / "skill_library", client=SimpleNamespace(chat=forbidden),
            experiment_split=overlay(["qa-1"]))


def test_mixed_source_bundle_requires_bound_learning_roles_at_reviser(tmp_path):
    roles = overlay(["qa-1", "qa-2"], ["heldout"])
    cases = [canonical("qa-1", "induction", roles), canonical("qa-2", role_map=roles)]
    experience = bundle(cases, roles)
    text = render_case_evidence(experience, experiment_split=roles)
    assert '"source_split": ["induction", "inner_validation"]' in text
    assert '"reference_answer": "B"' in text and '"custom_audit"' in text
    with pytest.raises(ValueError, match="authorized induction"):
        render_case_evidence(experience)
    changed = deepcopy(roles)
    changed["qa_roles"]["qa-2"] = "validation"
    with pytest.raises(ValueError, match="validation or unmapped"):
        render_case_evidence(experience, experiment_split=changed)
    reviser = FitSetSlotReviser(
        output_root=tmp_path / "revision", panel={}, panel_name="synthetic",
        variants=[], accumulation_root=tmp_path, label_metadata={},
        library_root=ROOT / "skill_library", client=SimpleNamespace(chat=forbidden),
        experiment_split=changed,
    )
    with pytest.raises(ValueError, match="validation or unmapped"):
        reviser._revise_slot(
            slot=(cases[0].question_type, cases[0].evidence_state),
            parent=_inputs()["parent"], feedback={}, forbidden_ids=set(), experience=experience)


@pytest.mark.parametrize("change", ["role", "manifest", "protocol", "validation_input"])
def test_resume_binds_roles_manifest_protocol_and_validation_inputs(tmp_path, change):
    runner = build(tmp_path)
    runner.initialize(resume=False)
    if change == "validation_input":
        runner.gate_variant_rows["gate-0@g0m0"]["artifact_ref"] = "changed-public-input"
    else:
        roles = deepcopy(runner.experiment_split)
        roles.pop("identity_sha256")
        roles.pop("qa_roles_sha256")
        if change == "role":
            roles["qa_roles"]["learn-0"] = "validation"
        elif change == "manifest":
            roles["manifest_identity"]["sha256"] = "b" * 64
        else:
            roles["protocol"] = "changed-protocol"
        runner.experiment_split = normalize_experiment_split(roles)
    with pytest.raises(fitset.FitSetError, match="manifest changed"):
        runner.initialize(resume=True)


def test_gate_copies_have_no_labels_or_learning_authorization(tmp_path, monkeypatch):
    runner = build(tmp_path, gates=10)
    runner.initialize(resume=False)
    worker = FitSetWorkerEvolution(runner, 0, runner.variants)
    parent = worker.provider(runner.variants[0])[0][0]
    captured = []

    class StopBeforeRequests(Exception):
        pass

    def capture(_runner, items):
        captured.extend(items)
        raise StopBeforeRequests

    runner.gate_client_factory = lambda *_args: forbidden
    monkeypatch.setattr(fitset.SystemVariantRunner, "_run_gate_episodes", capture)
    with pytest.raises(StopBeforeRequests):
        worker._gate(SimpleNamespace(slot_key=parent.slot_key), parent, parent, tmp_path / "gate")
    assert len(captured) == 40
    for arm, variant in captured:
        assert variant["qa_id"].startswith("gate-")
        assert not arm.capture_offline_case_input and arm.label_metadata == {}
        assert arm.evolution_config is None and arm.experiment_split is None
        assert arm._qa_disjoint_gate is True
        assert "offline_metadata" not in arm.input_contract_identity
        assert b"reference_answer" not in pickle.dumps(arm)


def test_gate_executor_refuses_item_labels_before_program_or_model(tmp_path, monkeypatch):
    runner = build(
        tmp_path, heldout=[], experiment_split=None, evolution_config=None,
        phase="final_system", label_metadata={})
    runner._qa_disjoint_gate = True
    item = public_item(runner.variants[0])
    item.episode.ground_truth = "must-not-be-used"
    monkeypatch.setattr(runner, "_prepare_one", lambda row: {
        "variant_id": row["variant_id"], "item": item})
    with pytest.raises(fitset.FitSetError, match="label-free public input"):
        runner._run_one_locked(runner.variants[0])


def test_overlay_hash_and_opt_in_are_enforced(tmp_path):
    frozen = normalize_experiment_split(overlay(["learn-0"]))
    frozen["qa_roles"]["learn-0"] = "validation"
    with pytest.raises(ValueError, match="role map hash changed"):
        normalize_experiment_split(frozen)
    with pytest.raises(fitset.FitSetError, match="requires opt-in"):
        build(tmp_path, evolution_config=FitSetEvolutionConfig())
    with pytest.raises(fitset.FitSetError, match="requires explicit experiment_split"):
        build(tmp_path, experiment_split=None)
    with pytest.raises(ValueError, match="137/138"):
        replace(FitSetEvolutionConfig(gate_mode="qa_disjoint"), gate_seeds=(1, 2))
