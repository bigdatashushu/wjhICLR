"""CPU-only tests for the resumable v11 evolution campaign."""

from __future__ import annotations

import hashlib
import json
import re
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
        evidence_state="g1m0",
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
        candidate_program_errors: int | None = None,
        candidate_legal_answer_rate: float = 1.0,
        post_body=None,
        revision_values=None,
        evaluator_error: Exception | None = None,
        n_pairs: int = 8,
    ):
        self.candidate_score = candidate_score
        self.candidate_runtime_errors = candidate_runtime_errors
        self.candidate_program_errors = candidate_program_errors
        self.candidate_legal_answer_rate = candidate_legal_answer_rate
        self.post_body = post_body
        self.revision_values = list(revision_values or [])
        self.evaluator_error = evaluator_error
        self.n_pairs = int(n_pairs)
        self.calls = {"collect": 0, "revise": 0, "evaluate": 0, "post": 0}
        self.revision_experiences = []
        self.revision_requests = []

    def collect(self, **kwargs):
        self.calls["collect"] += 1
        parent = kwargs["parent"]
        key = f"{parent.skill_id}@{parent.version}"
        evidence = {"geometry_3d": "available", "metric_scale": "unavailable"}
        return V11ExperienceBundle(
            campaign_id=kwargs["campaign_id"],
            parent_snapshot_id=kwargs["parent_snapshot_id"],
            parent_skill_version=key,
            parent_content_sha256=parent.content_sha256,
            question_type=parent.question_type,
            evidence_state=parent.evidence_state,
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
                    evidence_state=parent.evidence_state,
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
                    initial_evidence_states=evidence,
                    evidence_states=evidence,
                    evidence_history=[{"trigger": "initial", "evidence_states": evidence}],
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
                    evidence_state=parent.evidence_state,
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
                    initial_evidence_states=evidence,
                    evidence_states=evidence,
                    evidence_history=[{"trigger": "initial", "evidence_states": evidence}],
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
        self.revision_requests.append(kwargs)
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
            evidence_state=parent.evidence_state,
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
            n_pairs=self.n_pairs,
            result_refs=[f"pair:{index}" for index in range(self.n_pairs)],
            parent=V11EvaluationArm(
                skill_version=parent_key,
                content_sha256=parent.content_sha256,
                delivered_skill_version=parent_key,
                delivered_content_sha256=parent.content_sha256,
                delivery_observed=True,
                n_scored=self.n_pairs,
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
                n_scored=self.n_pairs,
                mean_score=self.candidate_score,
                runtime_error_count=self.candidate_runtime_errors,
                program_error_count=(
                    self.candidate_runtime_errors if self.candidate_program_errors is None
                    else self.candidate_program_errors),
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
            evidence_state=kwargs["evidence_state"],
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
        evidence_state="g1m0",
        library_root=str(library),
        run_root=str(runs),
        max_revision_attempts=3,
        seed=137,
        model_id="qwen3.8-flash",
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


def test_v11_campaign_rejects_too_small_paired_panel(tmp_path):
    _parent, original = _activate_parent(tmp_path / "library")
    callbacks = _Callbacks(candidate_score=0.75, n_pairs=4)

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
    assert decision["conditions"]["minimum_evaluation_pairs"] is False
    assert "minimum_evaluation_pairs" in decision["reasons"]


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


def test_v11_campaign_program_error_increase_is_descriptive_not_admission(tmp_path):
    _activate_parent(tmp_path / "library")
    callbacks = _Callbacks(candidate_program_errors=4)

    result = _runner(tmp_path, callbacks).run()

    assert result.status == "promoted"
    decision = json.loads(
        (tmp_path / "runs/campaign-one/receipts/decision.json").read_text())
    assert "program_errors_nonincreasing" not in decision["conditions"]
    assert decision["conditions"]["runtime_errors_nonincreasing"]
    assert callbacks.calls["post"] == 1


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
    repair = callbacks.revision_requests[1]["repair_context"]
    assert repair["candidate_text"] == invalid
    assert repair["candidate_sha256"] == hashlib.sha256(invalid.encode()).hexdigest()
    assert repair["fully_delivered"] is True
    assert {"code": "unknown_tool", "line": len(invalid.splitlines()),
            "rule": "registered_tool"} in repair["errors"]
    assert not re.search(r"[\u4e00-\u9fff]", " ".join(first.problems))
    assert second.passed is True
    assert _pointer(tmp_path / "library")["snapshot_id"] == original["snapshot_id"]


def test_unknown_tool_scan_ignores_prose_parentheticals():
    prose = "Use this evidence state (g0m0) and inspect structural elements (walls, doors)."
    assert campaign_module._unknown_tool_calls(prose) == []
    assert campaign_module._unknown_tool_calls(
        "Never call `invented_tool()`."
    ) == ["invented_tool"]


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
    repair = callbacks.revision_requests[1]["repair_context"]
    assert repair["candidate_sha256"] == hashlib.sha256(copied.encode()).hexdigest()
    assert "Which object is closest to the chair?" not in json.dumps(repair)
    assert "case-alpha" not in json.dumps(repair)
    assert repair["redacted_lines"] == [len(copied.splitlines())]
    assert repair["fully_delivered"] is False


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


def test_no_change_is_terminal_without_static_gate_or_publication(tmp_path):
    _parent, snapshot = _activate_parent(tmp_path / "library")
    callbacks = _Callbacks()
    campaign = _runner(tmp_path, callbacks)
    requests = []

    def unchanged(**kwargs):
        requests.append(kwargs)
        return V11RevisionProposal(
            action="no_change",
            modification_reason="The evidence does not justify a method change.",
            source_run_ref="revision:no-change",
        )

    campaign.reviser = unchanged
    result = campaign.run()
    assert result.status == result.decision == "no_change"
    assert result.revision_attempt == 1
    assert result.candidate_id == result.candidate_skill_version == ""
    assert not result.published_snapshot_id
    assert set(result.receipts) == {
        "experience_bundle", "revision_attempt_01", "decision",
    }
    decision = json.loads(campaign._receipt_path("decision").read_bytes())
    assert decision["stage"] == "revision"
    assert decision["outcome"] == "no_change"
    assert decision["reasons"] == ["The evidence does not justify a method change."]
    proposal = json.loads(campaign._receipt_path("revision_attempt_01").read_bytes())
    assert proposal["proposal"]["source_run_ref"] == "revision:no-change"
    assert "full_skill_md" not in proposal["proposal"]
    assert callbacks.calls == {"collect": 1, "revise": 0, "evaluate": 0, "post": 0}
    assert _pointer(tmp_path / "library")["snapshot_id"] == snapshot["snapshot_id"]
    resumed = _runner(tmp_path, callbacks)

    def repeated(**_kwargs):
        raise AssertionError("no_change was regenerated")

    resumed.collector = resumed.reviser = resumed.evaluator = resumed.post_publish_verifier = repeated
    assert resumed.run().status == "no_change"
    assert len(requests) == 1


def test_interrupted_no_change_resumes_from_proposal_without_regenerating(tmp_path):
    _activate_parent(tmp_path / "library")
    campaign = _runner(tmp_path, _Callbacks())
    campaign.reviser = lambda **_kwargs: V11RevisionProposal(
        action="no_change", modification_reason="Keep the existing method.",
        source_run_ref="revision:unchanged",
    )
    record = campaign._record_receipt

    class ProcessExit(BaseException):
        pass

    def interrupted(key, receipt):
        record(key, receipt)
        if key == "decision":
            raise ProcessExit()

    campaign._record_receipt = interrupted
    with pytest.raises(ProcessExit):
        campaign.run()
    resumed = _runner(tmp_path, _Callbacks())

    def repeated(**_kwargs):
        raise AssertionError("a completed revision was repeated")

    resumed.reviser = resumed.evaluator = resumed.post_publish_verifier = repeated
    assert resumed.run().status == "no_change"


@pytest.mark.parametrize("malformed", ['{"full_skill_md": "incomplete', "", "   "])
def test_malformed_response_delivers_actual_prior_candidate_to_repair(tmp_path, malformed):
    from skill3d.evolution.campaign_v11 import V11RevisionFormatError

    _activate_parent(tmp_path / "library")
    callbacks = _Callbacks(candidate_score=0.5)
    campaign = _runner(tmp_path, callbacks)
    requests = []

    def revise(**kwargs):
        requests.append(kwargs)
        if kwargs["attempt"] == 1:
            raise V11RevisionFormatError(
                "invalid JSON", source_run_ref="revision:malformed",
                candidate_text=malformed,
            )
        return callbacks.revise(**kwargs)

    campaign.reviser = revise
    assert campaign.run().revision_attempt == 2
    repair = requests[1]["repair_context"]
    assert repair["candidate_text"] == malformed
    assert repair["candidate_sha256"] == hashlib.sha256(malformed.encode()).hexdigest()
    first = json.loads(campaign._receipt_path("revision_attempt_01").read_bytes())
    assert first["format_candidate_text"] == malformed
    if malformed.strip():
        assert first["proposal"]["full_skill_md"] == malformed
    assert first["format_problems"] == ["invalid JSON"]


def test_fresh_collection_cannot_write_legacy_experience_bundle(tmp_path):
    _activate_parent(tmp_path / "library")
    callbacks = _Callbacks()
    campaign = _runner(tmp_path, callbacks)

    def legacy(**kwargs):
        bundle = callbacks.collect(**kwargs)
        return bundle.model_copy(update={"schema_version": "skill-experience-bundle-v11/5.0"})

    campaign.collector = legacy
    with pytest.raises(V11CampaignBlocked, match="read-only"):
        campaign.run()
    assert not campaign._receipt_path("experience_bundle").exists()


def test_experience_schema6_defaults_and_legacy_read_preserves_identity(tmp_path):
    parent, snapshot = _activate_parent(tmp_path / "library")
    bundle = _Callbacks().collect(
        campaign_id="schema", parent=parent, parent_snapshot_id=snapshot["snapshot_id"],
    )
    assert bundle.schema_version == "skill-experience-bundle-v11/6.0"
    assert bundle.cases[0].rounds == bundle.cases[0].image_refs == []
    assert bundle.cases[0].frame_set_hash == ""
    data = bundle.model_dump(mode="json")
    data["schema_version"] = "skill-experience-bundle-v11/5.0"
    for row in data["cases"]:
        for field in ("rounds", "image_refs", "frame_set_hash"):
            del row[field]
    raw = json.dumps(data).encode()
    legacy = V11ExperienceBundle.model_validate_json(raw)
    assert legacy.schema_version == "skill-experience-bundle-v11/5.0"
    assert legacy.cases[0].rounds == []
    assert legacy.parent_content_sha256 == parent.content_sha256
    case = bundle.cases[0].model_dump(mode="json")
    case.update(rounds=[{"round": 1, "program_source": "print('observed')"}],
                image_refs=[{"frame_slot": 0}], frame_set_hash="e" * 64)
    assert V11ExperienceCase.model_validate(case).rounds == case["rounds"]
    case["frame_set_hash"] = "not-a-hash"
    with pytest.raises(ValueError, match="frame_set_hash"):
        V11ExperienceCase.model_validate(case)


def test_resume_cannot_revise_from_an_existing_legacy_bundle_receipt(tmp_path):
    _activate_parent(tmp_path / "library")
    callbacks = _Callbacks()
    campaign = _runner(tmp_path, callbacks)
    parent, _snapshot = campaign._initialize()
    bundle = callbacks.collect(
        campaign_id=campaign.campaign_id, parent=parent,
        parent_snapshot_id=campaign.checkpoint.parent_snapshot_id)
    campaign._record_receipt("experience_bundle", bundle.model_copy(
        update={"schema_version": "skill-experience-bundle-v11/5.0"}))
    with pytest.raises(V11CampaignBlocked, match="read-only"):
        _runner(tmp_path, callbacks).run()
    assert callbacks.calls["revise"] == callbacks.calls["evaluate"] == 0


@pytest.mark.parametrize("declaration", ["changed", "unchanged", "nonexistent"])
def test_provided_sections_must_exist_and_actually_change(tmp_path, declaration):
    parent, _snapshot = _activate_parent(tmp_path / "library")
    headings = re.findall(r"^## (.+)$", parent.skill_md, flags=re.MULTILINE)
    section = {
        "changed": headings[0], "unchanged": headings[1], "nonexistent": "Invented heading",
    }[declaration]
    callbacks = _Callbacks(candidate_score=0.5)
    campaign = _runner(tmp_path, callbacks)

    def revise(**kwargs):
        proposal = callbacks.revise(**kwargs)
        if kwargs["attempt"] == 1:
            proposal = proposal.model_copy(update={
                "change_hypothesis": "Clarify the ambiguity trigger.",
                "changed_sections": [section],
                "expected_process_effect": "Check before ranking.",
            })
        return proposal

    campaign.reviser = revise
    result = campaign.run()
    assert result.status == "rejected"
    assert result.revision_attempt == (1 if declaration == "changed" else 2)
    first = json.loads(campaign._receipt_path("static_validation_01").read_bytes())
    assert first["checks"]["declared_sections_changed"] is (declaration == "changed")
    receipt = json.loads(campaign._receipt_path("revision_attempt_01").read_bytes())
    assert receipt["proposal"]["changed_sections"] == [section]
    assert receipt["proposal"]["change_hypothesis"] == "Clarify the ambiguity trigger."
    if declaration != "changed":
        errors = callbacks.revision_requests[1]["repair_context"]["errors"]
        assert any(error["code"] == "declared_change_not_present" and "line" in error
                   for error in errors)
    assert callbacks.calls["evaluate"] == 1


def test_governed_text_diagnosis_and_repair_share_one_frozen_campaign(tmp_path):
    from types import SimpleNamespace
    from skill3d.evolution.adapters_v11 import V11OfflineReviser
    from skill3d.evolution.diagnosis_v11 import OfflineDiagnosisConfig, V11CaseDiagnoser
    from skill3d.synthesis.request_governor import (
        GovernedClient, RequestGovernor, RequestGovernorConfig,
    )

    parent, snapshot = _activate_parent(tmp_path / "library")
    invalid = _candidate_md(parent).rstrip("\n") + "\n\n`invented_tool()`\n"
    requests = []

    class FakeClient:
        revision_count = 0

        def require_service(self):
            pass

        def chat_with_meta(self, messages, **kwargs):
            requests.append((messages, kwargs))
            if "evidence analyst" in messages[0]["content"]:
                value = {
                    "observations": [{
                        "statement": "The observed answer received an offline score.",
                        "evidence_refs": ["/cases/0/score"],
                    }],
                    "hypotheses": [],
                    "unknowns": ["The score does not establish a cause."],
                }
            else:
                self.revision_count += 1
                value = (
                    {"full_skill_md": invalid, "modification_reason": "Clarify checks."}
                    if self.revision_count == 1 else
                    {"action": "no_change", "modification_reason": "Keep the parent method."}
                )
            return SimpleNamespace(
                text=json.dumps(value), request_id=f"fake-{len(requests)}", truncated=False,
                usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            )

    client = FakeClient()
    governor = RequestGovernor(
        tmp_path / "governor", RequestGovernorConfig(max_requests=3, max_inflight=1),
    )
    diagnosis_config = OfflineDiagnosisConfig(mode="text")
    diagnoser = V11CaseDiagnoser(
        GovernedClient(client, governor, category="diagnosis"),
        tmp_path / "diagnosis", diagnosis_config,
    )
    reviser = V11OfflineReviser(
        GovernedClient(client, governor, category="reviser"),
        audit_root=tmp_path / "offline", diagnoser=diagnoser,
        diagnosis_config=diagnosis_config,
    )
    callbacks = _Callbacks()
    campaign = _runner(tmp_path, callbacks)
    revision_kwargs = []

    def revise(**kwargs):
        revision_kwargs.append(kwargs)
        return reviser(**kwargs)

    campaign.reviser = revise
    result = campaign.run()
    assert result.status == "no_change"
    assert result.revision_attempt == 2
    assert not campaign._receipt_path("static_validation_02").exists()
    assert callbacks.calls["evaluate"] == callbacks.calls["post"] == 0
    assert _pointer(tmp_path / "library")["snapshot_id"] == snapshot["snapshot_id"]
    summary = governor.summary()
    assert summary["categories"] == {"diagnosis": 1, "reviser": 2}
    assert summary["usage"]["total_tokens"] == 9
    assert summary["unknown_requests"] == 0
    assert len(requests) == 3
    assert invalid in revision_kwargs[1]["repair_context"]["candidate_text"]
    for messages, _kwargs in requests[1:]:
        text = messages[1]["content"]
        assert "Which object is closest to the chair?" in text
        assert "The score does not establish a cause." in text
    resumed = V11OfflineReviser(
        GovernedClient(client, governor, category="reviser"),
        audit_root=tmp_path / "offline", diagnoser=diagnoser,
        diagnosis_config=diagnosis_config,
    )
    assert resumed(**revision_kwargs[1]).action == "no_change"
    assert len(requests) == 3
    assert governor.summary()["reserved_requests"] == 3


def test_oversized_prior_candidate_stops_without_a_truncated_repair(tmp_path):
    parent, _snapshot = _activate_parent(tmp_path / "library")
    oversized = _candidate_md(parent) + "\n" + "Generic reasoning. " * 500
    callbacks = _Callbacks(revision_values=[oversized])
    campaign = _runner(tmp_path, callbacks)
    campaign.cfg = replace(
        campaign.cfg, method_context_max_chars=len(parent.skill_md) + 100,
    )
    result = campaign.run()
    assert result.status == "rejected"
    assert result.revision_attempt == 1
    assert callbacks.calls["revise"] == 1
    assert callbacks.calls["evaluate"] == 0
    assert not campaign._receipt_path("revision_attempt_02").exists()
