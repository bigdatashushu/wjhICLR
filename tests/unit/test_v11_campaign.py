"""CPU-only tests for the resumable v11 evolution campaign."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import skill3d.evolution.campaign_v11 as campaign_module
from skill3d.evolution.campaign_v11 import (
    V11CampaignBlocked,
    V11CampaignConfig,
    V11CampaignRunner,
)
from skill3d.schemas import (
    DataAccessRecord,
    V11EvaluationArm,
    V11ExperienceBundle,
    V11ExperienceCase,
    V11PairedEvaluationReceipt,
    V11PostPublishObservation,
    V11RevisionProposal,
    V11StaticValidationReceipt,
)
from skill3d.online.submission import EXECUTION_PROTOCOL_VERSION
from skill3d.synthesis.prompt_builder import PROMPT_TEMPLATE_VERSION
from skill3d.tools.docs_v11 import TOOL_DOCS_VERSION
from skill3d.tools.registry import TOOL_FACE_VERSION
from skill3d.skills.registry import load_active_skills
from skill3d.skills.v11_library import (
    build_v11_snapshot,
    load_skill_source_v11,
    write_v11_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "skill_library"
    / "versions"
    / "S03"
    / "1.1.0"
    / "rank-object-distances"
    / "SKILL.md"
)
SOURCE_REF = "versions/S03/1.1.0/rank-object-distances/SKILL.md"


def _activate_parent(library: Path):
    target = library / SOURCE_REF
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(SOURCE.read_bytes())
    loaded = load_skill_source_v11(
        library,
        source_ref=SOURCE_REF,
        skill_id="S03",
        version="1.1.0",
        question_type="object_rel_distance",
    )
    snapshot, manifest = build_v11_snapshot(
        [loaded],
        snapshot_id="S0-v11-campaign-test",
    )
    write_v11_snapshot(library, snapshot, manifest, activate=True)
    return loaded.spec, snapshot


def _candidate_md(parent) -> str:
    return parent.skill_md.replace(
        "重点核查会改变前两名的错误绑定",
        "优先核查会改变前两名的错误绑定",
        1,
    )


class _Callbacks:
    def __init__(
        self,
        *,
        candidate_score: float = 0.75,
        candidate_runtime_errors: int = 1,
        candidate_legal_answer_rate: float = 1.0,
        post_body=None,
        revision_values=None,
        evaluator_error: Exception | None = None,
    ):
        self.candidate_score = candidate_score
        self.candidate_runtime_errors = candidate_runtime_errors
        self.candidate_legal_answer_rate = candidate_legal_answer_rate
        self.post_body = post_body
        self.revision_values = list(revision_values or [])
        self.evaluator_error = evaluator_error
        self.calls = {"collect": 0, "revise": 0, "evaluate": 0, "post": 0}
        self.revision_experiences = []

    def collect(self, **kwargs):
        self.calls["collect"] += 1
        parent = kwargs["parent"]
        key = f"{parent.skill_id}@{parent.version}"
        return V11ExperienceBundle(
            campaign_id=kwargs["campaign_id"],
            parent_snapshot_id=kwargs["parent_snapshot_id"],
            parent_skill_version=key,
            parent_content_sha256=parent.content_sha256,
            question_type=parent.question_type,
            source_split="induction",
            split="learning",
            label_access=True,
            source_run_ref="run:parent-learning",
            label_access_record=DataAccessRecord(
                record_id="label-access-campaign-test",
                at="2026-10-09T00:00:00+00:00",
                split="induction",
                purpose="skill_induction",
                component_role="inducer",
                run_id="parent-learning",
                input_manifest_sha256="1" * 64,
                n_items=2,
                label_access=True,
                source_refs=["trace"],
            ),
            cases=[
                V11ExperienceCase(
                    case_id="case-alpha",
                    qa_id="qa-alpha",
                    scene_id="scene-alpha",
                    question_type=parent.question_type,
                    question_text="Which object is closest to the chair?",
                    options=["lamp", "table", "sofa", "door"],
                    source_split="induction",
                    split="learning",
                    label_access=True,
                    outcome="failure",
                    episode_status="run_error",
                    delivered_skill_version=key,
                    delivered_content_sha256=parent.content_sha256,
                    delivered_skill_md=parent.skill_md,
                    response_text="candidate response",
                    response_texts=["candidate response"],
                    program_source="ReturnAnswer('A')",
                    tool_observations=[{"tool": "relative_distance_rank", "ok": False}],
                    model_answer="A",
                    reference_answer="B",
                    score=0.0,
                    failure_categories=["run_error"],
                    error_direction="run_error",
                    trace_ref="trace:qa-alpha",
                ),
                V11ExperienceCase(
                    case_id="case-beta",
                    qa_id="qa-beta",
                    scene_id="scene-beta",
                    question_type=parent.question_type,
                    question_text="Which object is nearest to the desk?",
                    options=["cabinet", "chair", "plant", "window"],
                    source_split="induction",
                    split="learning",
                    label_access=True,
                    outcome="success",
                    episode_status="answered",
                    delivered_skill_version=key,
                    delivered_content_sha256=parent.content_sha256,
                    delivered_skill_md=parent.skill_md,
                    response_text="candidate response",
                    response_texts=["candidate response"],
                    program_source="ReturnAnswer('B')",
                    tool_observations=[{"tool": "relative_distance_rank", "ok": True}],
                    model_answer="B",
                    reference_answer="B",
                    score=1.0,
                    trace_ref="trace:qa-beta",
                ),
            ],
        )

    def revise(self, **kwargs):
        self.calls["revise"] += 1
        self.revision_experiences.append(kwargs["experience"].model_copy(deep=True))
        parent = kwargs["parent"]
        index = kwargs["attempt"] - 1
        text = (
            self.revision_values[index]
            if index < len(self.revision_values)
            else _candidate_md(parent)
        )
        return V11RevisionProposal(
            full_skill_md=text,
            modification_reason="clarify binding checks",
            source_run_ref=f"revision:attempt-{kwargs['attempt']}",
        )

    def evaluate(self, **kwargs):
        self.calls["evaluate"] += 1
        if self.evaluator_error is not None:
            error = self.evaluator_error
            self.evaluator_error = None
            raise error
        parent = kwargs["parent"]
        candidate = kwargs["candidate"]
        parent_key = f"{parent.skill_id}@{parent.version}"
        candidate_key = candidate.candidate_skill_version
        return V11PairedEvaluationReceipt(
            campaign_id=kwargs["campaign_id"],
            evaluation_id="eval-inner-fixed",
            panel_id="panel-object-rel-distance",
            panel_sha256="a" * 64,
            question_type=parent.question_type,
            seed=kwargs["seed"],
            request_seed_observed=True,
            model_id=kwargs["model_id"],
            model_config_sha256=kwargs["model_config_sha256"],
            quality_contract_sha256=kwargs["quality_contract_sha256"],
            solver_config_sha256=kwargs["solver_config_sha256"],
            template_version=PROMPT_TEMPLATE_VERSION,
            execution_protocol_version=EXECUTION_PROTOCOL_VERSION,
            tool_docs_version=TOOL_DOCS_VERSION,
            tool_face_version=TOOL_FACE_VERSION,
            status="completed",
            formal_result_eligible=True,
            frozen_panel=True,
            independent_arms=True,
            n_pairs=4,
            result_refs=[f"pair:{index}" for index in range(4)],
            parent=V11EvaluationArm(
                skill_version=parent_key,
                content_sha256=parent.content_sha256,
                delivered_skill_version=parent_key,
                delivered_content_sha256=parent.content_sha256,
                delivery_observed=True,
                n_scored=4,
                mean_score=0.5,
                runtime_error_count=1,
                program_error_count=1,
                untrusted_geometry_use_count=0,
                legal_answer_rate=0.75,
            ),
            candidate=V11EvaluationArm(
                skill_version=candidate_key,
                content_sha256=candidate.full_skill_spec.content_sha256,
                delivered_skill_version=candidate_key,
                delivered_content_sha256=candidate.full_skill_spec.content_sha256,
                delivery_observed=True,
                n_scored=4,
                mean_score=self.candidate_score,
                runtime_error_count=self.candidate_runtime_errors,
                program_error_count=self.candidate_runtime_errors,
                untrusted_geometry_use_count=0,
                legal_answer_rate=self.candidate_legal_answer_rate,
            ),
        )

    def post(self, **kwargs):
        self.calls["post"] += 1
        candidate = kwargs["candidate"]
        body = (
            candidate.full_skill_spec.skill_md
            if self.post_body is None
            else self.post_body
        )
        return V11PostPublishObservation(
            question_type=kwargs["question_type"],
            snapshot_id=kwargs["snapshot_id"],
            retrieved_skill_version=candidate.candidate_skill_version,
            delivered_skill_version=candidate.candidate_skill_version,
            delivered_skill_md=body,
            learning_event_refs=["experience:new-learning"],
            source_run_ref="run:post-publish",
        )


def _runner(tmp_path: Path, callbacks: _Callbacks, *, campaign_id="campaign-one"):
    library = tmp_path / "library"
    runs = tmp_path / "runs"
    cfg = V11CampaignConfig(
        campaign_id=campaign_id,
        question_type="object_rel_distance",
        library_root=str(library),
        run_root=str(runs),
        max_revision_attempts=3,
        seed=137,
        model_id="qwen3vl-test",
        model_config_sha256="2" * 64,
        quality_contract_sha256="3" * 64,
        solver_config_sha256="4" * 64,
    )
    return V11CampaignRunner(
        cfg,
        collector=callbacks.collect,
        reviser=callbacks.revise,
        evaluator=callbacks.evaluate,
        post_publish_verifier=callbacks.post,
    )


def _pointer(library: Path) -> dict:
    return json.loads(
        (library / "snapshots" / "active_snapshot.json").read_text(encoding="utf-8")
    )


def test_v11_campaign_rejects_without_moving_active_pointer(tmp_path):
    _parent, original = _activate_parent(tmp_path / "library")
    callbacks = _Callbacks(candidate_score=0.5)

    result = _runner(tmp_path, callbacks).run()

    assert result.status == "rejected"
    assert result.decision == "reject"
    assert _pointer(tmp_path / "library") == {
        "snapshot_id": original["snapshot_id"]
    }
    assert callbacks.calls == {
        "collect": 1,
        "revise": 1,
        "evaluate": 1,
        "post": 0,
    }
    decision = json.loads(
        (
            tmp_path
            / "runs"
            / "campaign-one"
            / "receipts"
            / "decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["conditions"]["strict_score_improvement"] is False


@pytest.mark.parametrize(
    ("runtime_errors", "legal_rate", "failed_condition"),
    [
        (2, 1.0, "runtime_errors_nonincreasing"),
        (1, 0.5, "legal_answer_rate_nondecreasing"),
    ],
)
def test_v11_campaign_rejects_runtime_or_legal_answer_regression(
    tmp_path,
    runtime_errors,
    legal_rate,
    failed_condition,
):
    _parent, original = _activate_parent(tmp_path / "library")
    callbacks = _Callbacks(
        candidate_score=0.75,
        candidate_runtime_errors=runtime_errors,
        candidate_legal_answer_rate=legal_rate,
    )

    result = _runner(tmp_path, callbacks).run()

    assert result.status == "rejected"
    assert _pointer(tmp_path / "library")["snapshot_id"] == original["snapshot_id"]
    decision = json.loads(
        (
            tmp_path
            / "runs"
            / "campaign-one"
            / "receipts"
            / "decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["conditions"][failed_condition] is False


def test_v11_campaign_promotes_strict_improvement_and_verifies_new_learning(tmp_path):
    _activate_parent(tmp_path / "library")
    callbacks = _Callbacks()

    result = _runner(tmp_path, callbacks).run()

    assert result.status == "promoted"
    assert result.decision == "promote"
    assert result.candidate_skill_version == "S03@1.2.0"
    assert result.published_snapshot_id == _pointer(tmp_path / "library")["snapshot_id"]
    skills, warnings, snapshot_id = load_active_skills(
        tmp_path / "library" / "snapshots"
    )
    assert warnings == []
    assert snapshot_id == result.published_snapshot_id
    assert [(skill.skill_id, skill.version) for skill in skills] == [("S03", "1.2.0")]
    post = json.loads(
        (
            tmp_path
            / "runs"
            / "campaign-one"
            / "receipts"
            / "post_publish.json"
        ).read_text(encoding="utf-8")
    )
    assert post["verified"] is True
    assert post["expected_content_sha256"] == result.candidate_content_sha256
    visible = callbacks.revision_experiences[0].cases[0]
    assert visible.question_text == "Which object is closest to the chair?"
    assert visible.model_answer == "A"
    assert visible.reference_answer == "B"
    revision = json.loads(
        (
            tmp_path
            / "runs"
            / "campaign-one"
            / "receipts"
            / "revision_attempt_01.json"
        ).read_text(encoding="utf-8")
    )
    assert len(revision["experience_sha256"]) == 64


def test_v11_campaign_retries_static_failure_with_feedback(tmp_path):
    parent, original = _activate_parent(tmp_path / "library")
    invalid = _candidate_md(parent).rstrip("\n") + "\n\n调用 `invented_tool()`。\n"
    callbacks = _Callbacks(
        candidate_score=0.5,
        revision_values=[invalid, _candidate_md(parent)],
    )

    result = _runner(tmp_path, callbacks).run()

    assert result.status == "rejected"
    assert result.revision_attempt == 2
    assert callbacks.calls["revise"] == 2
    first = V11StaticValidationReceipt.model_validate_json(
        (
            tmp_path
            / "runs"
            / "campaign-one"
            / "receipts"
            / "static_validation_01.json"
        ).read_bytes()
    )
    second = V11StaticValidationReceipt.model_validate_json(
        (
            tmp_path
            / "runs"
            / "campaign-one"
            / "receipts"
            / "static_validation_02.json"
        ).read_bytes()
    )
    assert first.passed is False
    assert first.checks["tools_known"] is False
    assert "invented_tool" in " ".join(first.problems)
    assert second.passed is True
    assert _pointer(tmp_path / "library")["snapshot_id"] == original["snapshot_id"]


def test_v11_campaign_retries_candidate_that_copies_a_question(tmp_path):
    parent, _original = _activate_parent(tmp_path / "library")
    copied = (
        _candidate_md(parent).rstrip("\n")
        + "\n\nWhich object is closest to the chair?\n"
    )
    callbacks = _Callbacks(
        candidate_score=0.5,
        revision_values=[copied, _candidate_md(parent)],
    )

    result = _runner(tmp_path, callbacks).run()

    assert result.status == "rejected"
    assert result.revision_attempt == 2
    first = V11StaticValidationReceipt.model_validate_json(
        (
            tmp_path
            / "runs"
            / "campaign-one"
            / "receipts"
            / "static_validation_01.json"
        ).read_bytes()
    )
    assert first.checks["no_leakage"] is False
    assert "question_text:case-alpha" in " ".join(first.problems)


def test_v11_campaign_resume_does_not_repeat_completed_callbacks(tmp_path):
    _activate_parent(tmp_path / "library")
    first = _Callbacks(evaluator_error=RuntimeError("evaluation interrupted"))
    runner = _runner(tmp_path, first)

    with pytest.raises(RuntimeError, match="evaluation interrupted"):
        runner.run()
    assert first.calls == {
        "collect": 1,
        "revise": 1,
        "evaluate": 1,
        "post": 0,
    }

    second = _Callbacks()

    def should_not_run(**_kwargs):
        raise AssertionError("completed callback was repeated")

    resumed = _runner(tmp_path, second)
    resumed.collector = should_not_run
    resumed.reviser = should_not_run
    result = resumed.run()

    assert result.status == "promoted"
    assert second.calls["evaluate"] == 1
    assert second.calls["post"] == 1


def test_v11_campaign_blocks_when_post_publish_body_does_not_match(tmp_path):
    parent, _snapshot = _activate_parent(tmp_path / "library")
    callbacks = _Callbacks(post_body=parent.skill_md)
    runner = _runner(tmp_path, callbacks)

    with pytest.raises(V11CampaignBlocked, match="发布后新 learning 验证失败"):
        runner.run()

    checkpoint = json.loads(runner.checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "blocked"
    assert checkpoint["candidate_skill_version"] == "S03@1.2.0"
    skills, warnings, _snapshot_id = load_active_skills(
        tmp_path / "library" / "snapshots"
    )
    assert warnings == []
    assert [skill.version for skill in skills] == ["1.1.0"]
    rollback = json.loads(
        (
            tmp_path
            / "runs"
            / "campaign-one"
            / "receipts"
            / "rollback.json"
        ).read_text(encoding="utf-8")
    )
    assert rollback["snapshot_from"] != rollback["snapshot_to"]
    assert rollback["pointer_after"] == rollback["snapshot_to"]
    assert rollback["verified"] is True
    receipt = json.loads(
        (
            tmp_path
            / "runs"
            / "campaign-one"
            / "receipts"
            / "post_publish.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["verified"] is False
    assert receipt["checks"]["delivered_body_exact"] is False


def test_v11_campaign_recovers_publication_without_republishing(
    tmp_path,
    monkeypatch,
):
    _activate_parent(tmp_path / "library")
    first = _Callbacks()
    runner = _runner(tmp_path, first)
    original_record = runner._record_receipt

    class SimulatedProcessExit(BaseException):
        pass

    def crash_after_library_publish(key, receipt):
        if key == "publication":
            raise SimulatedProcessExit("crash before campaign publication receipt")
        return original_record(key, receipt)

    runner._record_receipt = crash_after_library_publish
    with pytest.raises(SimulatedProcessExit):
        runner.run()

    assert not (
        tmp_path
        / "runs"
        / "campaign-one"
        / "receipts"
        / "publication.json"
    ).exists()
    assert (
        tmp_path
        / "library"
        / "validation"
        / "promotion"
        / f"{runner.checkpoint.candidate_id}.json"
    ).exists()

    def no_republish(*_args, **_kwargs):
        raise AssertionError("publish_v11_candidate was called again")

    monkeypatch.setattr(campaign_module, "publish_v11_candidate", no_republish)
    second = _Callbacks()
    resumed = _runner(tmp_path, second)

    def should_not_run(**_kwargs):
        raise AssertionError("completed callback was repeated")

    resumed.collector = should_not_run
    resumed.reviser = should_not_run
    resumed.evaluator = should_not_run
    result = resumed.run()

    assert result.status == "promoted"
    assert second.calls["post"] == 1


def test_post_publish_exception_rolls_back_candidate(tmp_path):
    _parent, snapshot = _activate_parent(tmp_path / "library")
    campaign = _runner(tmp_path, _Callbacks())

    def unavailable(**kwargs):
        raise RuntimeError("post service unavailable")

    campaign.post_publish_verifier = unavailable
    with pytest.raises(V11CampaignBlocked, match="post service unavailable"):
        campaign.run()
    assert _pointer(tmp_path / "library")["snapshot_id"] == snapshot["snapshot_id"]
    assert campaign.checkpoint.status == "blocked"
    assert campaign._receipt_path("rollback").is_file()


@pytest.mark.parametrize("change", [
    {"seed": 999}, {"max_revision_attempts": 4},
    {"model_config_sha256": "9" * 64},
    {"solver_config_sha256": "9" * 64},
])
def test_resume_rejects_changed_contract_even_after_completion(tmp_path, change):
    _activate_parent(tmp_path / "library")
    campaign = _runner(tmp_path, _Callbacks(candidate_score=0.5))
    assert campaign.run().status == "rejected"
    resumed = _runner(tmp_path, _Callbacks())
    resumed.cfg = replace(resumed.cfg, **change)
    with pytest.raises(V11CampaignBlocked, match="contract"):
        resumed.run()


def test_revision_must_keep_stable_name(tmp_path):
    parent, _ = _activate_parent(tmp_path / "library")
    renamed = _candidate_md(parent).replace(
        "name: rank-object-distances", "name: new-distance-method")
    callbacks = _Callbacks(
        candidate_score=0.5, revision_values=[renamed, _candidate_md(parent)])
    campaign = _runner(tmp_path, callbacks)
    assert campaign.run().revision_attempt == 2
    first = json.loads(campaign._receipt_path("static_validation_01").read_bytes())
    assert first["checks"]["stable_name"] is False


def test_revision_format_error_consumes_one_attempt_and_keeps_response_ref(tmp_path):
    from skill3d.evolution.campaign_v11 import V11RevisionFormatError

    _activate_parent(tmp_path / "library")
    callbacks = _Callbacks(candidate_score=0.5)
    campaign = _runner(tmp_path, callbacks)
    calls = []

    def revise(**kwargs):
        calls.append(kwargs["attempt"])
        if kwargs["attempt"] == 1:
            raise V11RevisionFormatError(
                "invalid JSON", source_run_ref="revision:captured-response")
        return callbacks.revise(**kwargs)

    campaign.reviser = revise
    checkpoint = campaign.run()
    assert checkpoint.status == "rejected"
    assert checkpoint.revision_attempt == 2
    assert calls == [1, 2]
    assert callbacks.calls["evaluate"] == 1
    proposal = campaign._receipt_path("revision_attempt_01").read_text()
    assert "revision:captured-response" in proposal
