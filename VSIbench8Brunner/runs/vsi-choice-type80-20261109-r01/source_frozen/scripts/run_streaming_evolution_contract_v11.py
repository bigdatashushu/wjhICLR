#!/usr/bin/env python3
"""CPU contract experiment for buffered continuous v11 Skill evolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from skill3d.evolution.campaign_v11 import V11CampaignConfig, V11CampaignRunner
from skill3d.evolution.experience_v11 import V11BufferedCollector
from skill3d.evolution.streaming_v11 import (
    StreamingAttemptRequest,
    StreamingAttemptResult,
    StreamingEvolutionConfig,
    StreamingEvolutionRunner,
)
from skill3d.online.submission import EXECUTION_PROTOCOL_VERSION
from skill3d.schemas import (
    SkillSpecV11,
    V11EvaluationArm,
    V11ExperienceCase,
    V11PairedEvaluationReceipt,
    V11PostPublishObservation,
    V11RevisionProposal,
)
from skill3d.skills.registry import bump_semver, load_active_skills
from skill3d.synthesis.prompt_builder import PROMPT_TEMPLATE_VERSION
from skill3d.tools.docs_v11 import TOOL_DOCS_VERSION
from skill3d.tools.registry import TOOL_FACE_VERSION


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _case(
    parent: SkillSpecV11,
    *,
    index: int,
    prefix: str,
    score: float = 0.0,
) -> V11ExperienceCase:
    evidence = {
        "geometry_3d": (
            "unavailable" if parent.evidence_state == "g0m0" else "available"
        ),
        "metric_scale": (
            "available" if parent.evidence_state == "g1m1" else "unavailable"
        ),
    }
    answer = "B" if score == 1.0 else "A"
    return V11ExperienceCase(
        case_id=f"case-{prefix}-{index:03d}",
        qa_id=f"qa-{prefix}-{index:03d}",
        scene_id=f"scene-{prefix}-{index:03d}",
        question_type=parent.question_type,
        evidence_state=parent.evidence_state,
        question_text=f"Which option satisfies the spatial relation in example {index}?",
        options=["chair", "table", "lamp", "sofa"],
        source_split="induction",
        split="learning",
        label_access=True,
        outcome="success" if score == 1.0 else "failure",
        episode_status="answered",
        delivered_skill_version=f"{parent.skill_id}@{parent.version}",
        delivered_content_sha256=parent.content_sha256,
        delivered_skill_md=parent.skill_md,
        response_text=f"ReturnAnswer({answer!r})",
        response_texts=[f"ReturnAnswer({answer!r})"],
        program_source=f"ReturnAnswer({answer!r})",
        tool_observations=[],
        initial_evidence_states=evidence,
        evidence_states=evidence,
        evidence_history=[{"trigger": "initial", "evidence_states": evidence}],
        candidate_objects=[],
        selected_object_ids=[],
        image_observations={"observed_image_ids": ["frame-0"]},
        model_answer=answer,
        reference_answer="B",
        score=score,
        failure_categories=[] if score == 1.0 else ["answer_incorrect"],
        error_direction="" if score == 1.0 else "wrong_option",
        trace_ref=f"contract-trace:{prefix}:{index:03d}",
    )


class _InjectedCampaignCallbacks:
    def __init__(
        self,
        request: StreamingAttemptRequest,
        *,
        candidate_score: float,
    ) -> None:
        self.request = request
        self.candidate_score = candidate_score
        self.candidate = None
        self.post_cases: tuple[V11ExperienceCase, ...] = ()

    def revise(self, **kwargs) -> V11RevisionProposal:
        parent = kwargs["parent"]
        marker = (
            "\n## Buffered evolution check\n\n"
            "- Recheck object identity, evidence availability, and answer mapping "
            f"for buffered revision attempt {self.request.attempt_index}.\n"
        )
        return V11RevisionProposal(
            full_skill_md=parent.skill_md.rstrip("\n") + marker,
            modification_reason="Add a generic buffered-evolution verification step.",
            source_run_ref=(
                f"contract-reviser:{self.request.stream_id}:"
                f"{self.request.slot_key}:{self.request.attempt_index}"
            ),
        )

    def evaluate(self, **kwargs) -> V11PairedEvaluationReceipt:
        parent = kwargs["parent"]
        candidate = kwargs["candidate"]
        self.candidate = candidate
        parent_key = f"{parent.skill_id}@{parent.version}"
        candidate_key = candidate.candidate_skill_version
        n_pairs = 8

        def arm(
            spec_key: str,
            digest: str,
            score: float,
        ) -> V11EvaluationArm:
            return V11EvaluationArm(
                skill_version=spec_key,
                content_sha256=digest,
                delivered_skill_version=spec_key,
                delivered_content_sha256=digest,
                delivery_observed=True,
                n_scored=n_pairs,
                mean_score=score,
                runtime_error_count=0,
                program_error_count=0,
                untrusted_geometry_use_count=0,
                legal_answer_rate=1.0,
            )

        return V11PairedEvaluationReceipt(
            campaign_id=kwargs["campaign_id"],
            evaluation_id=f"eval-{kwargs['campaign_id']}",
            panel_id=f"panel-{self.request.slot_key.replace('/', '-')}",
            panel_sha256=hashlib.sha256(
                f"{self.request.slot_key}:8".encode("utf-8")
            ).hexdigest(),
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
            n_pairs=n_pairs,
            result_refs=[
                f"contract-pair:{self.request.slot_key}:{index}"
                for index in range(n_pairs)
            ],
            parent=arm(parent_key, parent.content_sha256, 0.5),
            candidate=arm(
                candidate_key,
                candidate.full_skill_spec.content_sha256,
                self.candidate_score,
            ),
        )

    def post(self, **kwargs) -> V11PostPublishObservation:
        candidate = kwargs["candidate"]
        post_case = _case(
            candidate.full_skill_spec,
            index=self.request.attempt_index,
            prefix=(
                "post-"
                + self.request.question_type.replace("_", "-")
                + "-"
                + self.request.evidence_state
            ),
            score=1.0,
        )
        self.post_cases = (post_case,)
        return V11PostPublishObservation(
            question_type=self.request.question_type,
            evidence_state=self.request.evidence_state,
            snapshot_id=kwargs["snapshot_id"],
            retrieved_skill_version=candidate.candidate_skill_version,
            delivered_skill_version=candidate.candidate_skill_version,
            delivered_skill_md=candidate.full_skill_spec.skill_md,
            learning_event_refs=[post_case.trace_ref],
            source_run_ref=f"contract-post:{kwargs['campaign_id']}",
        )


class _ContractExperiment:
    def __init__(self, root: Path, *, stream_id: str) -> None:
        self.root = root
        self.stream_id = stream_id
        self.library = root / "library"
        self.campaign_root = root / "campaigns"
        self.stream = StreamingEvolutionRunner(
            root / "stream",
            stream_id=stream_id,
            config=StreamingEvolutionConfig(batch_size=12),
        )
        self.slot_attempts: dict[str, int] = {}

    def active(self, question_type: str, evidence_state: str) -> tuple[SkillSpecV11, str]:
        skills, warnings, snapshot = load_active_skills(self.library / "snapshots")
        if warnings:
            raise RuntimeError(f"active library warnings: {warnings}")
        matching = [
            skill
            for skill in skills
            if skill.question_type == question_type
            and skill.evidence_state == evidence_state
        ]
        if len(matching) != 1:
            raise RuntimeError(
                f"expected one active Skill for {question_type}/{evidence_state}"
            )
        return matching[0], snapshot

    def attempt(self, request: StreamingAttemptRequest) -> StreamingAttemptResult:
        parent, snapshot = self.active(request.question_type, request.evidence_state)
        if (
            snapshot != request.parent_snapshot_id
            or f"{parent.skill_id}@{parent.version}" != request.parent_skill_version
            or parent.content_sha256 != request.parent_content_sha256
        ):
            raise RuntimeError("stream request is not bound to the current active parent")
        slot_attempt = self.slot_attempts.get(request.slot_key, 0) + 1
        self.slot_attempts[request.slot_key] = slot_attempt
        campaign_id = (
            f"{self.stream_id}-{request.question_type}-"
            f"{request.evidence_state}-a{slot_attempt:02d}"
        )
        candidate_score = (
            0.75
            if request.question_type == "object_rel_distance" and slot_attempt == 2
            else 0.5
        )
        callbacks = _InjectedCampaignCallbacks(
            request,
            candidate_score=candidate_score,
        )
        collector = V11BufferedCollector(
            cases=request.cases,
            source_run_ref=(
                f"stream-buffer:{self.stream_id}:{request.slot_key}:"
                f"{request.parent_skill_version}:a{slot_attempt}"
            ),
            source_refs=tuple(case.trace_ref for case in request.cases),
        )
        config = V11CampaignConfig(
            campaign_id=campaign_id,
            question_type=request.question_type,
            evidence_state=request.evidence_state,
            library_root=str(self.library),
            run_root=str(self.campaign_root),
            min_evaluation_pairs=8,
            seed=137,
            model_id="qwen3.8-flash-contract",
            model_config_sha256="2" * 64,
            quality_contract_sha256="3" * 64,
            solver_config_sha256="4" * 64,
            resume=True,
        )
        checkpoint = V11CampaignRunner(
            config,
            collector=collector,
            reviser=callbacks.revise,
            evaluator=callbacks.evaluate,
            post_publish_verifier=callbacks.post,
        ).run()
        if checkpoint.status == "rejected":
            return StreamingAttemptResult(
                outcome="rejected",
                campaign_id=campaign_id,
                reason="strict paired admission rejected the candidate",
            )
        if checkpoint.status != "promoted":
            return StreamingAttemptResult(
                outcome="deferred",
                campaign_id=campaign_id,
                reason=f"campaign ended with status={checkpoint.status}",
            )
        active, active_snapshot = self.active(
            request.question_type, request.evidence_state
        )
        return StreamingAttemptResult(
            outcome="promoted",
            campaign_id=campaign_id,
            active_snapshot_id=active_snapshot,
            active_skill_version=f"{active.skill_id}@{active.version}",
            active_content_sha256=active.content_sha256,
            post_publish_cases=callbacks.post_cases,
        )


def run(output: Path, library_source: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(library_source, output / "library")
    experiment = _ContractExperiment(
        output,
        stream_id="streaming-contract-v11",
    )

    parent, parent_snapshot = experiment.active(
        "object_rel_distance", "g1m0"
    )
    original_rel_version = parent.version
    for index in range(1, 25):
        experiment.stream.ingest(
            _case(parent, index=index, prefix="s03-g1m0"),
            parent_snapshot_id=parent_snapshot,
            attempt=experiment.attempt,
        )

    promoted, promoted_snapshot = experiment.active(
        "object_rel_distance", "g1m0"
    )
    expected_version = bump_semver(original_rel_version, "MINOR")
    if promoted.version != expected_version:
        raise RuntimeError(
            f"stream did not promote the expected version: {promoted.version}"
        )

    tail_parent, tail_snapshot = experiment.active("object_counting", "g0m0")
    tail_original_version = tail_parent.version
    for index in range(1, 6):
        experiment.stream.ingest(
            _case(tail_parent, index=index, prefix="s01-g0m0"),
            parent_snapshot_id=tail_snapshot,
        )
    tail_result = experiment.stream.flush_tail(
        "object_counting/g0m0",
        attempt=experiment.attempt,
    )
    if tail_result is None or tail_result.outcome != "rejected":
        raise RuntimeError("tail campaign did not execute and reject as expected")

    experiment.stream.record_operational_failure(
        qa_id="qa-service-unavailable",
        reason="service_unavailable",
        detail={"stage": "M8", "counted_in_skill_buffer": False},
    )

    final_tail, final_snapshot = experiment.active("object_counting", "g0m0")
    stream_checkpoint = json.loads(
        (output / "stream" / "stream.json").read_text(encoding="utf-8")
    )
    summary = {
        "schema_version": "streaming-evolution-contract-result-v11/1.0",
        "status": "passed",
        "batch_size": 12,
        "attempt_count": len(stream_checkpoint["attempts"]),
        "attempt_outcomes": [
            attempt["outcome"] for attempt in stream_checkpoint["attempts"]
        ],
        "relative_distance": {
            "initial_version": original_rel_version,
            "final_version": promoted.version,
            "final_snapshot": promoted_snapshot,
            "post_publish_buffer_cases": len(
                stream_checkpoint["slots"]["object_rel_distance/g1m0"][
                    "case_refs"
                ]
            ),
        },
        "counting_tail": {
            "initial_version": tail_original_version,
            "final_version": final_tail.version,
            "final_snapshot": final_snapshot,
            "buffer_cases": len(
                stream_checkpoint["slots"]["object_counting/g0m0"]["case_refs"]
            ),
        },
        "operational_failure_count": len(
            stream_checkpoint["operational_failures"]
        ),
        "checks": {
            "first_batch_rejected": (
                stream_checkpoint["attempts"][0]["outcome"] == "rejected"
            ),
            "cumulative_second_batch_promoted": (
                stream_checkpoint["attempts"][1]["outcome"] == "promoted"
            ),
            "post_publish_case_seeded_new_parent": (
                len(
                    stream_checkpoint["slots"]["object_rel_distance/g1m0"][
                        "case_refs"
                    ]
                )
                == 1
            ),
            "tail_attempt_rejected_without_pointer_change": (
                final_tail.version == tail_original_version
            ),
            "operational_failure_not_counted": (
                len(stream_checkpoint["operational_failures"]) == 1
            ),
        },
    }
    if not all(summary["checks"].values()):
        raise RuntimeError(f"contract experiment failed: {summary['checks']}")
    _json_write(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the CPU buffered-evolution contract experiment"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--library-source", default="skill_library")
    args = parser.parse_args(argv)
    summary = run(
        Path(args.output).resolve(),
        Path(args.library_source).resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
