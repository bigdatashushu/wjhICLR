"""Public-input capture through the real spawned episode executor boundary."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.evaluation.fitset_v11 import (
    FitSetError, SystemVariantRunner, _variant_key, _write_create_once, content_sha256,
)
from skill3d.evolution.fitset_stream_v11 import (
    FitSetEvolutionConfig, FitSetWorkerEvolution, FrozenSkillProvider,
)
from skill3d.online.runner import EpisodeOutcome, OnlineRunConfig
from skill3d.schemas import EvidenceProfile, FrameSet, SceneState, VSIBenchEpisode
from skill3d.skills.registry import load_active_skills
from skill3d.synthesis.request_governor import RequestGovernorConfig


LIBRARY = Path(__file__).resolve().parents[2] / "skill_library"
METADATA = {"path": "/synthetic/coordinator-only-metadata.json", "sha256": "f" * 64}
MODES = ["full_cases_v1", "diagnosis_v1", "diagnosis_multimodal_v1"]


def forbidden(*_args, **_kwargs):
    pytest.fail("this fixture must not call a model or read external inputs")


def public_solver_factory(_variant_id):
    return SimpleNamespace(chat=forbidden)


def public_item(row):
    frames = list(range(32))
    return EpisodeItem(
        episode=VSIBenchEpisode(
            qa_id=row["qa_id"], scene_name=row["scene_id"], dataset="synthetic",
            question_type=row["question_type"], question="How many chairs are visible?",
            options=[], ground_truth="", frames=[], split=row["split"],
            frame_set=FrameSet(
                frame_ids=frames, source_frame_indices=frames,
                timestamps=[float(i) for i in frames], frame_set_hash="c" * 64,
            ),
        ),
        pixels=[np.full((4, 4, 3), i, dtype=np.uint8) for i in frames],
        source="synthetic",
    )


def synthetic_run_split(items, cfg, **_kwargs):
    from skill3d.sandbox.kernel import RestrictedNamespaceKernel
    from skill3d.tools import REGISTRY, SceneHandle

    item = items[0]
    assert item.episode.ground_truth == ""
    initial = {"geometry_3d": "unavailable", "metric_scale": "unavailable"}
    handle = SceneHandle(scene_state=SceneState(
        artifact_ref="synthetic", scene_route="fallback_2d_only",
        available_artifacts={"frames"}, evidence_profile=EvidenceProfile(**initial),
        question_type=item.episode.question_type,
    ))
    kernel = RestrictedNamespaceKernel(REGISTRY, handle, frames=item.pixels)
    cell = kernel.run_cell("ReturnAnswer('2')")
    assert cell.error_code is None
    parent = next(skill for skill in cfg.skills
                  if skill.slot_key == f"{item.episode.question_type}/g0m0")
    version = f"{parent.skill_id}@{parent.version}"
    outcome = EpisodeOutcome(
        qa_id=item.episode.qa_id, scene_name=item.episode.scene_name,
        question_type=item.episode.question_type, final_state="answer",
        answer=cell.answer, predicted=cell.answer, is_mca=False,
        correct=None, mra_value=None, answer_source="tool_program",
        episode_status="answered", initial_evidence_states=initial,
        evidence_history=[{"evidence_states": initial}],
        initial_skill_selection={
            "skill_version": version, "content_sha256": parent.content_sha256,
            "evidence_state": "g0m0", "question_type": parent.question_type,
        },
        delivered_skill_versions=[version],
        retrieval_records=[{
            "candidates": [{"skill_version": version, "delivered": True}],
            "delivered_skill_versions": [version],
            "delivered_content_sha256": {version: parent.content_sha256},
        }],
        rounds=[{
            "index": 0, "synthesis_source": "online_model_ok",
            "executed": True, "response_text": "synthetic program",
            "program_source": "ReturnAnswer('2')", "results": [], "error_code": "",
        }],
    )
    return [outcome], SimpleNamespace(model_dump=lambda **_kw: {"synthetic": True})


class PublicInputRunner(SystemVariantRunner):
    """Replace preparation/inference only; retain capture, result and reader code."""

    def _prepare_one(self, variant, *, preflight=False):
        if not preflight and getattr(self, "_request_category", "solver") == "solver":
            assert self.evolution_config is None
            assert self.label_metadata == {}
            assert self.seed_skills == [] and self.gate_panels == {}
            assert self.revision_client_factory is None
            assert self.gate_client_factory is None
        item = public_item(variant)
        skills, identity = self.skills_for_variant(variant)
        root = self.root / "variants" / _variant_key(variant["variant_id"])
        plan = {"variant": dict(variant), "skill_identity": identity,
                "public_input_sha256": content_sha256(variant)}
        if not preflight:
            _write_create_once(root / "plan.json", plan)
            _write_create_once(root / "execution.json", {
                "public_only": True,
                "offline_metadata_visible": "offline_metadata" in self.input_contract_identity,
            })
        return {
            "variant_id": variant["variant_id"], "item": item,
            "artifact_path": root / "synthetic-artifact.json",
            "artifact_relocation": None, "state": variant["evidence_state"],
            "identity": {"sha256": content_sha256(variant)},
            "reference": self.binding_refs[variant["variant_id"]],
            "prepared": SimpleNamespace(), "runtime_base_cfg": self.base_cfg,
            "skills": skills, "skill_identity": identity, "variant_root": root,
            "plan": plan,
        }

    def _run_one_locked(self, variant):
        from skill3d.online import runner

        prior = runner.run_split
        runner.run_split = synthetic_run_split
        try:
            return super()._run_one_locked(variant)
        finally:
            runner.run_split = prior


def build_runner(tmp_path, mode="full_cases_v1", *, n=1):
    skills, warnings, snapshot = load_active_skills(LIBRARY / "snapshots")
    assert not warnings
    learning = [{
        "variant_id": f"public-{i}@g0m0", "qa_id": f"public-{i}",
        "scene_id": f"learning-{i}", "question_type": "object_counting",
        "evidence_state": "g0m0", "split": "induction",
    } for i in range(n)]
    gates = [{
        **learning[0], "variant_id": f"gate-{i}@g0m0", "qa_id": f"gate-{i}",
        "scene_id": f"heldout-{i}", "split": "inner_validation",
    } for i in range(8)]
    return PublicInputRunner(
        output_root=tmp_path / "run",
        panel={"proposal_sha256": "a" * 64, "panel_file_sha256": "b" * 64},
        panel_name="synthetic", variants=learning, gate_variants=gates,
        item_provider=public_item, input_receipt_provider=forbidden,
        input_contract_identity={"public": "synthetic", "offline_metadata": METADATA},
        published_inputs=SimpleNamespace(),
        binding_refs={row["variant_id"]: {
            "key": "d" * 64, "manifest_sha256": "e" * 64, "request_identity": {"seed": 137},
        }
                      for row in [*learning, *gates]},
        base_cfg=OnlineRunConfig(seed=137), phase="accumulation",
        client_factory=public_solver_factory,
        skills_for_variant=FrozenSkillProvider(
            skills, {"snapshot_ref": snapshot, "manifest_sha256": "e" * 64}),
        min_free_gib=0, scheduler="episode_pool", workers=1, skill_owners=1,
        request_governor_config=RequestGovernorConfig(max_requests=8, max_inflight=1),
        evolution_config=FitSetEvolutionConfig(
            learning_input_mode=mode, gate_mode="scene_disjoint", scheduling_mode="batch_v1"),
        label_metadata={row["qa_id"]: {
            "scene_id": row["scene_id"], "question_type": row["question_type"],
            "options": [], "reference_answer": "2",
        } for row in [*learning, *gates]},
        revision_client_factory=forbidden, gate_client_factory=forbidden,
    )


@pytest.mark.parametrize("mode", MODES)
def test_spawn_captures_public_input_and_owner_ingests_once_on_resume(tmp_path, mode):
    runner = build_runner(tmp_path, mode)
    assert runner.run(resume=False)["completed_variants"] == 1
    variant = runner.variants[0]
    public = runner.offline_case_input(variant)
    assert public["question_text"] == public_item(variant).episode.question
    assert public["options"] == [] and public["split"] == "induction"
    assert "reference_answer" not in public and "ground_truth" not in public
    assert len(public["image_refs"]) == 32
    root = runner.root / "variants" / _variant_key(variant["variant_id"])
    assert json.loads((root / "execution.json").read_text())["offline_metadata_visible"] is False
    row = runner.completed()[variant["variant_id"]]
    assert row["offline_input_sha256"] == hashlib.sha256(
        (root / "offline_input.json").read_bytes()).hexdigest()
    stored = json.loads(next((runner.root / "workers/00/canonical_cases").glob("*.json")).read_text())
    assert stored["source_split"] == "induction" and stored["reference_answer"] == "2"
    assert len(stored["image_refs"]) == len(public["image_refs"])
    from PIL import Image

    for index, (recorded, source) in enumerate(zip(stored["image_refs"], public["image_refs"])):
        assert {key: recorded[key] for key in source} == source
        assert recorded["experience_provenance"] == {
            "source_ref": f"adapter:image_refs/{index}",
            "content_identity": "recorded", "observation_status": "unknown",
        }
        with Image.open(source["path"]) as image:
            assert np.array_equal(np.asarray(image), public_item(variant).pixels[index])
    assert runner.input_contract_identity["offline_metadata"] == METADATA
    assert runner.label_metadata[variant["qa_id"]]["reference_answer"] == "2"
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    resumed = build_runner(tmp_path, mode)
    assert resumed.run(resume=True)["completed_variants"] == 1
    assert resumed.completed()[variant["variant_id"]] == row
    assert {path: path.read_bytes() for path in before} == before
    receipt = json.loads((runner.root / "episode_pool_receipt.json").read_text())
    assert receipt["executor_count"] == 0
    state = json.loads((runner.root / "workers/00/buffer/stream.json").read_text())
    assert state["slots"]["object_counting/g0m0"]["new_since_attempt"] == 1
    assert len(state["slots"]["object_counting/g0m0"]["case_refs"]) == 1


def test_aggregate_executor_does_not_capture_offline_input(tmp_path):
    runner = build_runner(tmp_path, "aggregate_v1")
    assert runner.run(resume=False)["completed_variants"] == 1
    assert not list(runner.root.glob("variants/*/offline_input.json"))
    assert "offline_input_sha256" not in next(iter(runner.completed().values()))


def test_owner_configuration_is_frozen_at_initialization_not_construction(tmp_path):
    runner = build_runner(tmp_path, "aggregate_v1")
    assert runner.capture_offline_case_input is False
    runner.evolution_config = replace(
        runner.evolution_config, learning_input_mode="full_cases_v1")
    assert runner.run(resume=False)["completed_variants"] == 1
    manifest = json.loads((runner.root / "manifest.json").read_text())
    assert manifest["request_input_config"]["capture_offline_case_input"] is True
    assert runner.offline_case_input(runner.variants[0])["split"] == "induction"
    assert build_runner(tmp_path).run(resume=True)["completed_variants"] == 1


def test_capture_policy_is_frozen_into_resume_identity(tmp_path):
    runner = build_runner(tmp_path)
    manifest = runner.initialize(resume=False)
    assert manifest["request_input_config"]["capture_offline_case_input"] is True
    runner.capture_offline_case_input = False
    changed = runner._manifest()
    assert changed["identity_sha256"] != manifest["identity_sha256"]
    with pytest.raises(FitSetError, match="manifest changed"):
        runner.initialize(resume=True)


def test_gate_copy_keeps_capture_disabled_and_coordinator_metadata_private(tmp_path, monkeypatch):
    runner = build_runner(tmp_path)
    runner.initialize(resume=False)
    owner = FitSetWorkerEvolution(runner, 0, runner._planned_slot_shards()[0])
    parent = owner.provider(runner.variants[0])[0][0]
    request = SimpleNamespace(slot_key=parent.slot_key)

    class GateCopiesCaptured(Exception):
        pass

    def capture(items):
        assert len(items) == 32
        for arm, _row in items:
            assert arm.evolution_config is None and arm.label_metadata == {}
            assert arm.capture_offline_case_input is False
            assert "offline_metadata" not in arm.input_contract_identity
        raise GateCopiesCaptured

    runner.gate_client_factory = lambda *_args: forbidden
    monkeypatch.setattr(runner, "_run_gate_episodes", capture)
    with pytest.raises(GateCopiesCaptured):
        owner._gate(request, parent, parent, tmp_path / "gate-check")
    assert runner.capture_offline_case_input is True
    assert runner.input_contract_identity["offline_metadata"] == METADATA
