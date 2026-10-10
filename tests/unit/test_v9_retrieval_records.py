"""Current v11 lookup records, complete delivery and trace identity checks."""

from __future__ import annotations

import json

import pytest

from skill3d.routing.retrieval_policy import (
    DEFAULT_METHOD_CONTEXT_MAX_CHARS,
    RetrievalPolicy,
)
from skill3d.routing.skill_retriever import retrieve_ex
from skill3d.schemas import SceneState, SkillSpecV11
from skill3d.schemas.evidence import GATE_SUBCONDITIONS, GATE_VERSION, EvidenceProfile, MetricEvidenceGateResult
from skill3d.schemas.retrieval import (
    SkillCandidateRecord,
    SkillRetrievalRecord,
)
from skill3d.skills.delivery import (
    check_service_limit,
    plan_delivery,
    render_skill_entry,
    skill_content_sha256,
)

QUESTION = "How many chairs are in this room?"


# ------------------------------------------------------------------ 构造助手 ----


def _profile(states=None) -> EvidenceProfile:
    base = {"geometry_3d": "available", "world_frame": "available",
            "metric_scale": "available", "object_detection": "available",
            "track_consensus": "available", "object_grounding": "available"}
    base.update(states or {})
    return EvidenceProfile(**base)


def _gate(passed=True, version=GATE_VERSION) -> MetricEvidenceGateResult:
    return MetricEvidenceGateResult(
        gate_passed=bool(passed), gate_version=version,
        sub_results={name: bool(passed) for name in GATE_SUBCONDITIONS},
        values={}, missing_subconditions=[] if passed else list(GATE_SUBCONDITIONS))


def _scene(states=None) -> SceneState:
    return SceneState(artifact_ref="a", scene_route="full_3d",
                      evidence_profile=_profile(states), metric_evidence_gate_result=_gate(),
                      objects=[], summary="s")


# --------------------------------------------------- §13.6 检索记录字段齐备 ----

def test_retrieval_record_carries_every_required_field():
    """§13.6 点名的字段一个都不能少（规范题型/证据版本/候选/原因/分数/选中/配置版本）。"""
    policy = RetrievalPolicy(label="ret-test-1")
    hits, rec = retrieve_ex(QUESTION, _scene(), [_skill()], question_type="object_counting",
                            policy=policy, evidence_version="9.0",
                            snapshot_ref="S0-seed-20260925-v1",
                            snapshot_manifest_sha256="61259a")
    assert [h.skill_version for h in hits] == ["sk-1@1.0.0"]
    assert rec.canonical_question_type == "object_counting"
    assert rec.question_type_raw == "object_counting"
    assert rec.evidence_version == "9.0"
    assert rec.config_version == "ret-test-1"          # 人类标签
    assert rec.config_sha256 == policy.sha256()        # 内容摘要（可复算）
    assert rec.config_source == "default"
    assert rec.policy["ranking"] is False
    assert rec.policy["method_context_max_chars"] == DEFAULT_METHOD_CONTEXT_MAX_CHARS
    assert rec.active_snapshot_ref == "S0-seed-20260925-v1"
    assert rec.active_snapshot_manifest_sha256 == "61259a"
    assert rec.partition_policy == "v11_unique_active_by_question_type"
    assert rec.n_skills_offered == 1
    row = rec.candidates[0]
    assert row.reason_code == "hit" and row.hard_filter_passed
    assert row.selected is True and row.rank == 1 and row.score is not None
    assert row.content_sha256 == skill_content_sha256(_skill())
    assert row.content_chars == len(render_skill_entry(_skill()))
    assert rec.eligible_skill_versions == ["sk-1@1.0.0"]
    assert rec.retrieved_skill_versions == ["sk-1@1.0.0"]
    # 还没发出任何请求 → 一条都不算已交付（§13.6）
    assert rec.delivered_skill_versions == []
    assert rec.delivery_channel == "not_sent"
    assert rec.delivered_content_sha256 == {}


def test_unknown_question_type_records_why_nothing_was_retrieved():
    """题型不可知 → fail-closed 不检索；记录必须能回答"为什么一条都没有"。"""
    hits, rec = retrieve_ex(QUESTION, _scene(), [_skill()], question_type="not_a_task")
    assert hits == []
    assert rec.question_type_known is False
    assert rec.n_skills_offered == 1
    assert [r.reason_code for r in rec.candidates] == ["question_type_unknown"]
    assert rec.delivery_note


# ------------------------------------------------- §13.5 上下文上限与静态检查 ----


def test_plan_delivery_rejects_non_positive_cap():
    """上限必须为正：把 0 静默当成"无上限"等于把上限撤销。"""
    with pytest.raises(ValueError):
        plan_delivery([_skill()], max_chars=0)
    with pytest.raises(ValueError):
        plan_delivery([_skill()], max_chars=-5)


def test_service_limit_check_rejects_oversized_single_candidate():
    """§13.5：单条正文超过服务限制 → 该候选永远无法完整交付，静态检查拒绝。"""
    big = _skill("sk-big", desc="X" * 500)
    small = _skill("sk-small", desc="ok")
    cap = len(render_skill_entry(small)) + 10
    assert check_service_limit(small, max_chars=cap) == []
    problems = check_service_limit(big, max_chars=cap)
    assert problems and "服务限制" in problems[0]


# --------------------------------------------------------- §13.5 策略冻结 ----


# --------------------------------------------- §13.6 交付（选中 ≠ 已交付）----

def test_add_delivery_only_counts_entries_in_a_sent_request():
    """§13.6：prompt 里放了方法但**没发出请求** → retrieved 非空、delivered 为空。"""
    hits, rec = retrieve_ex(QUESTION, _scene(), [_skill("sk-a")],
                            question_type="object_counting")
    assert rec.retrieved_skill_versions == ["sk-a@1.0.0"]
    plan = plan_delivery([_skill("sk-a")], max_chars=8000)
    rec.add_delivery(plan, round_index=1, skills_by_key={"sk-a@1.0.0": _skill("sk-a")})
    assert rec.delivered_skill_versions == []
    assert rec.candidates[0].delivered is False
    assert rec.candidates[0].delivery_reason == "no_model_request"
    assert rec.method_summaries[-1]["channel"] == "not_sent"

    plan.mark_delivered()
    rec.add_delivery(plan, round_index=1, skills_by_key={"sk-a@1.0.0": _skill("sk-a")})
    assert rec.delivered_skill_versions == ["sk-a@1.0.0"]
    assert rec.candidates[0].delivered is True
    assert rec.candidates[0].delivery_reason == "delivered"
    assert rec.delivered_content_sha256 == {"sk-a@1.0.0": skill_content_sha256(_skill("sk-a"))}
    assert rec.method_summaries[-1]["channel"] == "model_request"


def test_request_failure_is_not_recorded_as_delivered():
    """§13.6：请求失败不能记"已交付"（模型可能根本没收到正文）。"""
    _hits, rec = retrieve_ex(QUESTION, _scene(), [_skill("sk-a")],
                             question_type="object_counting")
    plan = plan_delivery([_skill("sk-a")], max_chars=8000)
    plan.mark_request_failed("ServiceUnavailable: connection refused")
    rec.add_delivery(plan, round_index=2)
    assert rec.delivered_skill_versions == []
    assert rec.delivered_content_sha256 == {}
    assert rec.candidates[0].delivery_reason == "request_failed"
    assert rec.delivery_channel == "request_failed"


def test_record_validators_reject_inconsistent_delivery():
    """自洽性 fail-closed：delivered ⊆ retrieved，且 delivered 必须有正文 hash。"""
    with pytest.raises(ValueError):
        SkillCandidateRecord(skill_id="s", version="1", skill_version="s@1",
                             hard_filter_passed=True, selected=True, delivered=True,
                             delivery_reason="no_model_request")
    with pytest.raises(ValueError):
        SkillCandidateRecord(skill_id="s", version="1", skill_version="s@1",
                             hard_filter_passed=False, selected=True)
    with pytest.raises(ValueError):
        SkillRetrievalRecord(retrieved_skill_versions=["s@1"], eligible_skill_versions=[],
                             delivered_skill_versions=["s@1"])
    with pytest.raises(ValueError):
        # delivered 必须带正文 hash（§13.6"实际交付版本与正文 hash"）
        SkillRetrievalRecord(eligible_skill_versions=["s@1"],
                             retrieved_skill_versions=["s@1"],
                             delivered_skill_versions=["s@1"])
    with pytest.raises(ValueError):
        SkillRetrievalRecord(trigger="not_a_trigger")
    with pytest.raises(ValueError):
        SkillCandidateRecord(skill_id="s", version="1", reason_code="made_up")


def test_trace_schema_carries_the_four_states_separately():
    """trace 落盘字段（§13.6 三个版本清单 + 每次检索的完整记录）。"""
    from skill3d.schemas import EpisodeTrace, TraceRecord

    for model in (EpisodeTrace, TraceRecord):
        fields = set(model.model_fields)
        assert {"retrieval_records", "retrieved_skill_versions",
                "delivered_skill_versions",
                "declared_selected_skill_versions"} <= fields


def test_retrieval_record_round_trips_through_json():
    """记录要能原样落盘/回读（NaN 与嵌套模型都不能在往返里变形）。"""
    _hits, rec = retrieve_ex(QUESTION, _scene(), [_skill("sk-a")],
                             question_type="object_counting")
    plan = plan_delivery([_skill("sk-a")], max_chars=8000)
    plan.mark_delivered()
    rec.add_delivery(plan, round_index=1, skills_by_key={"sk-a@1.0.0": _skill("sk-a")})
    blob = json.dumps(rec.model_dump(), ensure_ascii=False)
    back = SkillRetrievalRecord.model_validate(json.loads(blob))
    assert back.model_dump() == rec.model_dump()


def test_plan_to_dict_does_not_leak_full_bodies():
    """交付计划落盘的是 hash（身份），不是正文（体积）——正文只在 prompt 里。"""
    plan = plan_delivery([_skill("sk-a")], max_chars=8000)
    dumped = json.dumps(plan.to_dict(), ensure_ascii=False)
    assert "content_sha256" in dumped
    assert render_skill_entry(_skill("sk-a")) not in dumped


def _skill(skill_id="sk-1", version="1.0.0", task="object_counting", desc="计数方法", template="", **kwargs):
    return SkillSpecV11(skill_id=skill_id, version=version, question_type=task,
                        skill_md=f"---\nname: count-objects\ndescription: {desc}\n---\n\n{template or desc}\n")
