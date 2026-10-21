"""流式 fit-set 的版本边界、双 seed、恢复和冻结行为回归。"""

from dataclasses import dataclass
import json
import threading

import pytest

from skill3d.evaluation.fitset_v11 import (
    SystemVariantRunner, _variant_key, _write_create_once, content_sha256,
)
from skill3d.evolution.fitset_revision_v11 import load_active_parent_slots
from skill3d.evolution.fitset_stream_v11 import (
    FitSetEvolutionConfig, FrozenSkillProvider, SeededGateClient,
    freeze_fitset_stream,
)
from skill3d.online.runner import OnlineRunConfig
from skill3d.schemas import EvidenceProfile, SceneState
from skill3d.skills.registry import load_active_skills
from pathlib import Path

LIBRARY = Path(__file__).resolve().parents[2] / "skill_library"


class RevisionClient:
    def chat(self, messages, **kwargs):
        text = messages[-1]["content"]
        parent = text.split("Parent complete SKILL.md:\n")[1].split(
            "\nAuthorized Tool interfaces for this exact slot:\n")[0]
        return json.dumps({
            "action": "revise",
            "full_skill_md": parent.rstrip("\n") + "\n\nUse FITREV checks.\n",
            "modification_reason": "Add process checks.",
            "change_hypothesis": "Process checks may improve completeness.",
            "changed_sections": ["__body__"],
            "expected_process_effect": "Check completeness.",
            "evidence_signature_refs": [],
        })


def revision_factory():
    return RevisionClient()


class NoChangeRevisionClient:
    def chat(self, messages, **kwargs):
        return json.dumps({"action": "no_change", "modification_reason": "No supported method change."})


def no_change_revision_factory():
    return NoChangeRevisionClient()


class AnswerClient:
    def chat(self, _messages, **kwargs):
        return str(kwargs["seed"])


@dataclass
class SeedFactory:
    seed: int
    output: Path

    def __call__(self, variant_id):
        return SeededGateClient(
            AnswerClient(), self.seed,
            self.output / "request_seeds" / (_variant_key(variant_id) + ".json"))


def gate_factory(seed, output):
    return SeedFactory(seed, output)


class ProbeRunner(SystemVariantRunner):
    """使用真实 kernel；输入与模型用确定性小夹具替代。"""

    def _run_one(self, variant):
        from skill3d.sandbox.kernel import RestrictedNamespaceKernel
        from skill3d.tools import REGISTRY, SceneHandle

        assert threading.current_thread() is threading.main_thread()
        self._require_storage_reserve(variant["variant_id"])
        skills, identity = self.skills_for_variant(variant)
        spec = next(s for s in skills if s.slot_key ==
                    f"{variant['question_type']}/{variant['evidence_state']}")
        initial = {"geometry_3d": "unavailable", "metric_scale": "unavailable"}
        handle = SceneHandle(
            scene_state=SceneState(
                artifact_ref="fixture",
                scene_route="fallback_2d_only",
                available_artifacts={"frames"},
                evidence_profile=EvidenceProfile(**initial),
                question_type="object_counting",
            ),
        )
        kernel = RestrictedNamespaceKernel(REGISTRY, handle, frames=[])
        value = "2" if "FITREV" in spec.skill_md else "1"
        cell = kernel.run_cell(f"ReturnAnswer({value!r})")
        assert cell.error_code is None
        if self.client_factory:
            client = self.client_factory(variant["variant_id"])
            client.chat([], seed=self.base_cfg.seed)
        outcome = {
            "qa_id": variant["qa_id"], "final_state": "answer", "answer": cell.answer,
            "answer_source": "tool_program", "episode_status": "answered",
            "delivered_skill_versions": [f"{spec.skill_id}@{spec.version}"],
            "initial_skill_selection": {
                "skill_version": f"{spec.skill_id}@{spec.version}",
                "content_sha256": spec.content_sha256,
                "evidence_state": spec.evidence_state,
            },
            "initial_evidence_states": initial,
            "evidence_history": [{"evidence_states": initial}],
            "rounds": [{
                "executed": True, "response_text": "a program",
                "program_source": "ReturnAnswer()", "results": [], "error_code": "",
            }],
        }
        variant_root = self.root / "variants" / _variant_key(variant["variant_id"])
        _write_create_once(variant_root / "plan.json", {
            "variant": dict(variant), "skill_identity": identity,
            "input_hash": content_sha256(variant),
        })
        _write_create_once(variant_root / "outcome.json", outcome)
        row = {
            "variant_id": variant["variant_id"], "qa_id": variant["qa_id"],
            "question_type": variant["question_type"],
            "evidence_state": variant["evidence_state"],
            "initial_skill_selection": outcome["initial_skill_selection"],
            "answer_source": "tool_program",
            "answer": value, "status": "completed", "final_state": "answer",
            "skill_identity": identity,
        }
        _write_create_once(self._result_path(variant["variant_id"]), row)
        return row


def make_runner(tmp_path, *, workers=1, n=13, reference="2", episode_concurrency=None):
    skills, _, snapshot = load_active_skills(LIBRARY / "snapshots")
    variants = [{
        "variant_id": f"fit-{i:03d}@g0m0", "qa_id": f"fit-{i:03d}",
        "scene_id": f"scene-{i:03d}", "question_type": "object_counting",
        "evidence_state": "g0m0", "split": "inner_validation",
    } for i in range(n)]
    labels = {row["qa_id"]: {
        "question_type": "object_counting", "scene_id": row["scene_id"],
        "options": [], "reference_answer": reference,
    } for row in variants}
    return ProbeRunner(
        output_root=tmp_path / "run",
        panel={"proposal_sha256": "p" * 64, "panel_file_sha256": "f" * 64},
        panel_name="ready_579", variants=variants,
        item_provider=None, input_receipt_provider=None, input_contract_identity={},
        binding_refs={}, base_cfg=OnlineRunConfig(seed=137), phase="accumulation",
        skills_for_variant=FrozenSkillProvider(skills, {
            "snapshot_ref": snapshot, "manifest_sha256": "a" * 64,
        }), workers=workers, min_free_gib=0,
        episode_concurrency=episode_concurrency,
        evolution_config=FitSetEvolutionConfig(),
        label_metadata=labels, revision_client_factory=revision_factory,
        gate_client_factory=gate_factory,
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_twelve_case_boundary_promotes_then_next_uses_new_version(tmp_path, workers):
    seed = load_active_parent_slots(LIBRARY)[0]
    runner = make_runner(tmp_path, workers=workers)
    first = runner.run(resume=False, max_variants=12)
    assert first["completed_variants"] == 12
    result = make_runner(tmp_path, workers=workers).run(resume=True)
    assert result["completed_variants"] == 13
    rows = runner.completed()
    before = rows["fit-011@g0m0"]["initial_skill_selection"]["skill_version"]
    after = rows["fit-012@g0m0"]["initial_skill_selection"]["skill_version"]
    assert before == "S01-g0m0@1.0.0"
    assert after == "S01-g0m0@1.1.0"
    stream_path = next((runner.root / "workers").glob("*/buffer/stream.json"))
    stream = json.loads(stream_path.read_text())
    assert stream["slots"]["object_counting/g0m0"]["new_since_attempt"] == 1
    assert len(stream["attempts"]) == 1
    decision = json.loads(next(
        (runner.root / "workers").glob("*/campaigns/*/decision.json")).read_text())
    assert decision["conditions"] == {"137": True, "138": True}
    assert make_runner(tmp_path, workers=workers).run(resume=True)["completed_variants"] == 13
    bundle = freeze_fitset_stream(runner.root, tmp_path / "bundle")
    assert bundle["revised_slot_count"] == 1
    assert bundle["preserved_unreachable_slot_count"] == 5
    assert load_active_parent_slots(LIBRARY)[0] == seed


@pytest.mark.parametrize("workers", [1, 2])
def test_parallel_gate_matches_serial_admission(tmp_path, workers):
    """A global episode-concurrency cap must not change the A/B decision.

    The gate episodes run in spawned children bounded by one global permit, so
    this proves the parallel path reaches the same promotion and the same
    per-seed conditions as the serial gate.
    """
    seed = load_active_parent_slots(LIBRARY)[0]
    runner = make_runner(tmp_path, workers=workers, episode_concurrency=4)
    result = runner.run(resume=False)
    assert result["completed_variants"] == 13
    rows = runner.completed()
    assert rows["fit-011@g0m0"]["initial_skill_selection"][
        "skill_version"] == "S01-g0m0@1.0.0"
    assert rows["fit-012@g0m0"]["initial_skill_selection"][
        "skill_version"] == "S01-g0m0@1.1.0"
    decision = json.loads(next(
        (runner.root / "workers").glob("*/campaigns/*/decision.json")).read_text())
    assert decision["conditions"] == {"137": True, "138": True}
    assert decision["passed"] is True
    bundle = freeze_fitset_stream(runner.root, tmp_path / "bundle")
    assert bundle["revised_slot_count"] == 1
    assert load_active_parent_slots(LIBRARY)[0] == seed


def test_equal_scores_reject_and_keep_seed(tmp_path):
    runner = make_runner(tmp_path, reference="1")
    runner.run(resume=False)
    stream = json.loads(next(
        (runner.root / "workers").glob("*/buffer/stream.json")).read_text())
    assert stream["attempts"][0]["outcome"] == "rejected"
    assert runner.completed()["fit-012@g0m0"]["initial_skill_selection"][
        "skill_version"] == "S01-g0m0@1.0.0"
    bundle = freeze_fitset_stream(runner.root, tmp_path / "bundle")
    assert bundle["revised_slot_count"] == 0


def test_no_change_skips_gate_and_next_episode_keeps_parent(tmp_path):
    runner = make_runner(tmp_path)
    runner.revision_client_factory = no_change_revision_factory
    runner.run(resume=False)
    stream = json.loads(next(
        (runner.root / "workers").glob("*/buffer/stream.json")).read_text())
    assert stream["attempts"][0]["outcome"] == "no_change"
    assert not list((runner.root / "workers").glob("*/campaigns/*/decision.json"))
    assert runner.completed()["fit-012@g0m0"]["initial_skill_selection"][
        "skill_version"] == "S01-g0m0@1.0.0"
    assert freeze_fitset_stream(runner.root, tmp_path / "bundle")["revised_slot_count"] == 0


def test_short_slot_does_not_trigger_tail_revision(tmp_path):
    runner = make_runner(tmp_path, n=3)
    runner.run(resume=False)
    stream = json.loads(next(
        (runner.root / "workers").glob("*/buffer/stream.json")).read_text())
    assert stream["attempts"] == []
    assert freeze_fitset_stream(runner.root, tmp_path / "bundle")[
        "revised_slot_count"] == 0


class InterruptRunner(ProbeRunner):
    interrupted_once = False

    def _write_summary(self, manifest, completed):
        if not self.interrupted_once:
            self.interrupted_once = True
            raise KeyboardInterrupt()
        return super()._write_summary(manifest, completed)


def test_concurrent_slot_owners_publish_without_overwriting(tmp_path):
    runner = make_runner(tmp_path, workers=2, n=12)
    other = [{
        **row, "qa_id": row["qa_id"] + "-other",
        "variant_id": row["variant_id"].replace("@", "-other@"),
        "question_type": "object_abs_distance",
    } for row in runner.variants]
    for row in other:
        runner.label_metadata[row["qa_id"]] = {
            "question_type": row["question_type"], "scene_id": row["scene_id"],
            "options": [], "reference_answer": "2",
        }
    runner.variants.extend(other)
    runner.execution_variants = sorted(runner.variants, key=lambda row: row["scene_id"])
    runner.gate_panels["object_abs_distance/g0m0"] = [
        row["variant_id"] for row in other[:8]]
    runner.run(resume=False)
    bundle = freeze_fitset_stream(runner.root, tmp_path / "bundle")
    assert bundle["revised_slot_count"] == 2
    assert bundle["slots"]["object_counting/g0m0"]["version"] == "1.1.0"
    assert bundle["slots"]["object_abs_distance/g0m0"]["version"] == "1.1.0"


def test_parent_callback_interrupt_drains_workers_and_refreshes_summary(tmp_path):
    runner = make_runner(tmp_path, workers=2, n=4)
    runner.__class__ = InterruptRunner
    with pytest.raises(KeyboardInterrupt):
        runner.run(resume=False)
    summary = json.loads((runner.root / "summary.json").read_text())
    assert summary["completed_variants"] == len(runner.completed())
    assert summary["pending_variants"] + summary["completed_variants"] == 4
    assert make_runner(tmp_path, workers=2, n=4).run(resume=True)["status"] == "complete"


def test_freeze_rejects_missing_result_and_changed_worker_ownership(tmp_path):
    runner = make_runner(tmp_path, n=3)
    runner.run(resume=False)
    marker = runner.root / "workers" / "00" / "worker_identity.json"
    data = json.loads(marker.read_text())
    data["owned_slots"] = ["object_counting/g1m1"]
    marker.write_text(json.dumps(data))
    with pytest.raises(Exception, match="所有权"):
        freeze_fitset_stream(runner.root, tmp_path / "bundle")


def test_worker_cannot_complete_summary_before_evolution_finishes(tmp_path):
    runner = make_runner(tmp_path, n=3)
    manifest = runner.initialize(resume=False)
    completed = {
        row["variant_id"]: {"status": "completed"} for row in runner.variants
    }
    assert runner._write_summary(manifest, completed)["status"] == "partial"
    runner.evolution_finished = True
    assert runner._write_summary(manifest, completed)["status"] == "complete"


def test_immutable_ab_decision_is_required_for_freezing(tmp_path):
    runner = make_runner(tmp_path)
    runner.run(resume=False)
    path = next((runner.root / "workers").glob("*/campaigns/*/decision.json"))
    value = json.loads(path.read_text())
    value["candidate_hash"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(Exception, match="A/B"):
        freeze_fitset_stream(runner.root, tmp_path / "bundle")


def test_reviser_rejects_new_tool_from_another_evidence_state():
    from skill3d.evolution.fitset_revision_v11 import _static_validate

    parent = load_active_parent_slots(LIBRARY)[0][("object_abs_distance", "g0m0")]
    _, problems = _static_validate(
        parent=parent,
        candidate_text=parent.skill_md.rstrip("\n") + "\n\nobject_distance_m('x', 'y')\n",
        forbidden_sample_ids=set(), method_context_max_chars=24000,
    )
    assert any("unavailable exact-slot tool" in problem for problem in problems)


class SeedRegressionRunner(ProbeRunner):
    def _run_one(self, variant):
        spec = self.skills_for_variant(variant)[0][0]
        if (self.client_factory is not None
                and self.client_factory.seed == 138
                and "FITREV" in spec.skill_md):
            # 只改变该评估臂的生成结果，不改变候选身份。
            root = self.root
            row = super()._run_one(variant)
            outcome_path = root / "variants" / _variant_key(
                variant["variant_id"]) / "outcome.json"
            outcome = json.loads(outcome_path.read_text())
            outcome["answer"] = "1"
            outcome_path.write_text(json.dumps(outcome))
            row["answer"] = "1"
            self._result_path(variant["variant_id"]).write_text(json.dumps(row))
            return row
        return super()._run_one(variant)


def test_improving_one_seed_cannot_publish_if_other_seed_regresses(tmp_path):
    runner = make_runner(tmp_path)
    runner.__class__ = SeedRegressionRunner
    runner.run(resume=False)
    decision = json.loads(next(
        (runner.root / "workers").glob("*/campaigns/*/decision.json")).read_text())
    assert decision["conditions"] == {"137": True, "138": False}
    assert freeze_fitset_stream(runner.root, tmp_path / "bundle")[
        "revised_slot_count"] == 0
