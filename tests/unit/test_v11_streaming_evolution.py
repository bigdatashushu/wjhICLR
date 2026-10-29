"""Persistent streaming v11 evolution buffer contracts."""

from __future__ import annotations

import hashlib
import json

from skill3d.evolution.streaming_v11 import (
    StreamingAttemptResult,
    StreamingEvolutionConfig,
    StreamingEvolutionRunner,
)
from skill3d.schemas import V11ExperienceCase


def _skill_md(label: str) -> str:
    return (
        "---\n"
        "name: rank-object-distances-g1m0\n"
        f"description: Reusable distance method {label}.\n"
        "---\n\n"
        "# Method\n\n"
        f"Use explicit object identities and validate the ranking ({label}).\n"
    )


def _case(index: int, *, version: str = "1.0.0", label: str = "parent"):
    body = _skill_md(label)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    evidence = {"geometry_3d": "available", "metric_scale": "unavailable"}
    return V11ExperienceCase(
        case_id=f"case-{label}-{index}",
        qa_id=f"qa-{label}-{index}",
        scene_id=f"scene-{label}-{index}",
        question_type="object_rel_distance",
        evidence_state="g1m0",
        question_text=f"Which object is closest in example {index}?",
        options=["chair", "table", "lamp", "sofa"],
        source_split="induction",
        split="learning",
        label_access=True,
        outcome="failure",
        episode_status="answered",
        delivered_skill_version=f"S03-g1m0@{version}",
        delivered_content_sha256=digest,
        delivered_skill_md=body,
        response_text="ReturnAnswer('A')",
        response_texts=["ReturnAnswer('A')"],
        program_source="ReturnAnswer('A')",
        initial_evidence_states=evidence,
        evidence_states=evidence,
        evidence_history=[{"trigger": "initial", "evidence_states": evidence}],
        model_answer="A",
        reference_answer="B",
        score=0.0,
        failure_categories=["answer_incorrect"],
        error_direction="wrong_option",
        trace_ref=f"trace:{label}:{index}",
    )


def _checkpoint(tmp_path):
    return json.loads((tmp_path / "stream.json").read_text(encoding="utf-8"))


def test_rejection_keeps_all_cases_and_waits_for_new_batch(tmp_path):
    runner = StreamingEvolutionRunner(
        tmp_path,
        stream_id="stream-one",
        config=StreamingEvolutionConfig(batch_size=3),
    )
    requests = []

    def reject(request):
        requests.append(request)
        return StreamingAttemptResult(
            outcome="rejected",
            campaign_id=f"campaign-{len(requests)}",
            reason="candidate did not improve",
        )

    for index in range(1, 3):
        assert runner.ingest(
            _case(index), parent_snapshot_id="snapshot-parent", attempt=reject
        ) is None
    first = runner.ingest(
        _case(3), parent_snapshot_id="snapshot-parent", attempt=reject
    )
    assert first.outcome == "rejected"
    assert len(requests) == 1
    assert len(requests[0].cases) == 3
    assert requests[0].new_case_count == 3

    for index in range(4, 6):
        assert runner.ingest(
            _case(index), parent_snapshot_id="snapshot-parent", attempt=reject
        ) is None
    second = runner.ingest(
        _case(6), parent_snapshot_id="snapshot-parent", attempt=reject
    )
    assert second.outcome == "rejected"
    assert len(requests) == 2
    assert len(requests[1].cases) == 6
    assert requests[1].new_case_count == 3
    slot = _checkpoint(tmp_path)["slots"]["object_rel_distance/g1m0"]
    assert len(slot["case_refs"]) == 6
    assert slot["attempt_count"] == 2
    assert slot["new_since_attempt"] == 0


def test_promotion_resets_parent_buffer_and_seeds_new_learning(tmp_path):
    runner = StreamingEvolutionRunner(
        tmp_path,
        stream_id="stream-promote",
        config=StreamingEvolutionConfig(batch_size=2),
    )
    post = _case(99, version="1.1.0", label="candidate")

    def promote(request):
        return StreamingAttemptResult(
            outcome="promoted",
            campaign_id="campaign-promote",
            active_snapshot_id="snapshot-candidate",
            active_skill_version=post.delivered_skill_version,
            active_content_sha256=post.delivered_content_sha256,
            post_publish_cases=(post,),
        )

    old_case = _case(1)
    runner.ingest(old_case, parent_snapshot_id="snapshot-parent", attempt=promote)
    result = runner.ingest(
        _case(2), parent_snapshot_id="snapshot-parent", attempt=promote
    )
    assert result.outcome == "promoted"
    slot = _checkpoint(tmp_path)["slots"]["object_rel_distance/g1m0"]
    assert slot["parent_snapshot_id"] == "snapshot-candidate"
    assert slot["parent_skill_version"] == "S03-g1m0@1.1.0"
    assert len(slot["case_refs"]) == 1
    assert slot["new_since_attempt"] == 1
    assert slot["attempted_ever"] is True

    resumed = StreamingEvolutionRunner(
        tmp_path,
        stream_id="stream-promote",
        config=StreamingEvolutionConfig(batch_size=2),
    )
    assert resumed.ingest(
        post, parent_snapshot_id="snapshot-candidate", attempt=promote
    ) is None
    assert resumed.ingest(
        old_case, parent_snapshot_id="snapshot-parent", attempt=promote
    ) is None
    assert len(_checkpoint(tmp_path)["slots"]["object_rel_distance/g1m0"][
        "case_refs"]) == 1


def test_other_slot_publication_can_advance_snapshot_without_losing_cases(tmp_path):
    runner = StreamingEvolutionRunner(
        tmp_path,
        stream_id="stream-cross-slot",
        config=StreamingEvolutionConfig(batch_size=12),
    )
    runner.ingest(_case(1), parent_snapshot_id="snapshot-before")
    runner.ingest(_case(2), parent_snapshot_id="snapshot-after-other-slot")

    slot = _checkpoint(tmp_path)["slots"]["object_rel_distance/g1m0"]
    assert slot["parent_snapshot_id"] == "snapshot-after-other-slot"
    assert len(slot["case_refs"]) == 2


def test_tail_attempt_only_runs_for_never_attempted_slot(tmp_path):
    runner = StreamingEvolutionRunner(
        tmp_path,
        stream_id="stream-tail",
        config=StreamingEvolutionConfig(batch_size=12),
    )
    for index in range(1, 5):
        runner.ingest(_case(index), parent_snapshot_id="snapshot-parent")
    requests = []

    def reject(request):
        requests.append(request)
        return StreamingAttemptResult(
            outcome="rejected",
            campaign_id="campaign-tail",
            reason="tail candidate rejected",
        )

    result = runner.flush_tail(
        "object_rel_distance/g1m0",
        attempt=reject,
    )
    assert result.outcome == "rejected"
    assert requests[0].trigger == "tail"
    assert len(requests[0].cases) == 4
    assert runner.flush_tail(
        "object_rel_distance/g1m0",
        attempt=reject,
    ) is None


def test_tail_waits_for_configured_minimum_and_runs_once(tmp_path):
    runner = StreamingEvolutionRunner(
        tmp_path, stream_id="minimum-tail",
        config=StreamingEvolutionConfig(batch_size=12, minimum_tail_cases=4),
    )
    requests = []

    def reject(request):
        requests.append(request)
        return StreamingAttemptResult(outcome="rejected", campaign_id="minimum-tail")

    for index in range(3):
        runner.ingest(_case(index), parent_snapshot_id="snapshot-parent")
    assert runner.flush_tail("object_rel_distance/g1m0", attempt=reject) is None
    runner.ingest(_case(3), parent_snapshot_id="snapshot-parent")
    assert runner.flush_tail("object_rel_distance/g1m0", attempt=reject).outcome == "rejected"
    assert runner.flush_tail("object_rel_distance/g1m0", attempt=reject) is None
    assert len(requests) == 1


def test_deferred_slot_accumulates_until_explicit_retry(tmp_path):
    runner = StreamingEvolutionRunner(
        tmp_path,
        stream_id="stream-deferred",
        config=StreamingEvolutionConfig(batch_size=1),
    )
    calls = []

    def defer(request):
        calls.append(request)
        return StreamingAttemptResult(
            outcome="deferred",
            campaign_id="campaign-deferred",
            reason="validation shard unavailable",
        )

    first = runner.ingest(
        _case(1), parent_snapshot_id="snapshot-parent", attempt=defer)
    assert first.outcome == "deferred"
    assert runner.ingest(
        _case(2), parent_snapshot_id="snapshot-parent", attempt=defer) is None
    assert len(calls) == 1

    def reject(request):
        calls.append(request)
        return StreamingAttemptResult(
            outcome="rejected",
            campaign_id="campaign-resumed",
            reason="candidate did not improve",
        )

    resumed = runner.retry_deferred(
        "object_rel_distance/g1m0", attempt=reject)
    assert resumed.outcome == "rejected"
    assert len(calls[-1].cases) == 2
    assert calls[-1].new_case_count == 2


def test_operational_failures_do_not_increment_skill_buffer(tmp_path):
    runner = StreamingEvolutionRunner(tmp_path, stream_id="stream-failure")
    ref = runner.record_operational_failure(
        qa_id="qa-broken",
        reason="service_unavailable",
        detail={"stage": "M8"},
    )
    checkpoint = _checkpoint(tmp_path)
    assert checkpoint["slots"] == {}
    assert checkpoint["operational_failures"] == [ref]


def test_ready_attempt_can_be_recovered_after_callback_boundary(tmp_path):
    runner = StreamingEvolutionRunner(
        tmp_path,
        stream_id="stream-recover-ready",
        config=StreamingEvolutionConfig(batch_size=2),
    )
    runner.ingest(_case(1), parent_snapshot_id="snapshot-parent")
    runner.ingest(_case(2), parent_snapshot_id="snapshot-parent")
    calls = []

    def reject(request):
        calls.append(request)
        return StreamingAttemptResult(
            outcome="rejected",
            campaign_id="campaign-recovered",
            reason="candidate did not improve",
        )

    result = runner.attempt_ready(
        "object_rel_distance/g1m0",
        attempt=reject,
    )
    assert result.outcome == "rejected"
    assert len(calls) == 1
    assert calls[0].new_case_count == 2
    assert runner.attempt_ready(
        "object_rel_distance/g1m0",
        attempt=reject,
    ) is None


def test_no_change_keeps_parent_and_does_not_block_next_batch_on_resume(tmp_path):
    config = StreamingEvolutionConfig(batch_size=1)
    runner = StreamingEvolutionRunner(tmp_path, stream_id="no-change", config=config)

    def no_change(request):
        return StreamingAttemptResult(
            outcome="no_change", campaign_id=f"no-change-{request.attempt_index}",
            reason="No new method hypothesis")

    runner.ingest(_case(1), parent_snapshot_id="parent", attempt=no_change)
    resumed = StreamingEvolutionRunner(tmp_path, stream_id="no-change", config=config)
    result = resumed.ingest(_case(2), parent_snapshot_id="parent", attempt=no_change)
    assert result.outcome == "no_change"
    slot = _checkpoint(tmp_path)["slots"]["object_rel_distance/g1m0"]
    assert slot["parent_skill_version"] == "S03-g1m0@1.0.0"
    assert slot["new_since_attempt"] == 0
    assert slot["deferred_reason"] == ""
    assert len(slot["case_refs"]) == 2
    assert slot["attempt_count"] == 2
