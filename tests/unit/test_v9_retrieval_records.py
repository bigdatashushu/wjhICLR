"""P6 单测：§13.5/§13.6 检索记录、四态分离与交付正文身份。

覆盖的规范原文（§13.6）：

    「每次检索记录规范题型、evidence_version、候选及过滤原因、排序分数、选中版本、
    实际交付版本与正文 hash、配置版本。区分"检索选中但未送达模型"与"已交付"。」
    「记录 `retrieved_skill_versions`、`delivered_skill_versions`、
    `declared_selected_skill_versions`、可观察的程序使用线索，以及每轮短方法摘要。」
    「模型自称选择要与程序和观察交叉检查，不能当作方法成功或因果贡献的充分证明。」

以及 §13.5：

    「top-k、排序规则和方法上下文上限在演化开始前写入配置并冻结。」
    「上下文放不下时先减少完整条目，不能截掉检查、局部条件或来源后仍称
    "完整 Skill 已交付"。」
    「超过服务限制的候选在静态检查中拒绝或修订。」

本文件只测"记录与交付"这一层：**不**声称任何真实模型/真实 Skill 晋升。
"""

from __future__ import annotations

import json

import pytest

from skill3d.routing.retrieval_policy import (
    DEFAULT_METHOD_CONTEXT_MAX_CHARS,
    RetrievalPolicy,
    retrieval_policy_from_config,
)
from skill3d.routing.skill_retriever import retrieve, retrieve_ex
from skill3d.schemas import SceneState, SkillSpec
from skill3d.schemas.evidence import GATE_SUBCONDITIONS, GATE_VERSION, EvidenceProfile, MetricEvidenceGateResult
from skill3d.schemas.retrieval import (
    CANDIDATE_REASON_CODES,
    SkillCandidateRecord,
    SkillRetrievalRecord,
)
from skill3d.skills.delivery import (
    DEFAULT_METHOD_CONTEXT_MAX_CHARS as DELIVERY_DEFAULT_CAP,
    check_service_limit,
    plan_delivery,
    render_skill_entry,
    skill_content_sha256,
)

QUESTION = "How many chairs are in this room?"


# ------------------------------------------------------------------ 构造助手 ----

def _skill(skill_id="sk-1", version="1.0.0", task="object_counting", desc="计数方法",
           template="ReturnAnswer(str(count_objects('chair')))", family="counting",
           signature=None, metric=False, gate_version=None, state=None) -> SkillSpec:
    s = SkillSpec(
        skill_id=skill_id, version=version, applicable_question_types=[task],
        required_evidence_signature=dict(signature or {}),
        requires_metric_evidence=bool(metric),
        applicable_gate_version=(gate_version or (GATE_VERSION if metric else None)),
        skill_family=family, source="real", description=desc,
        call_graph_template=template)
    if state is not None:
        object.__setattr__(s, "state", state)
    return s


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
    policy = RetrievalPolicy(top_k=1, label="ret-test-1")
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
    assert rec.policy["top_k"] == 1
    assert rec.policy["method_context_max_chars"] == DEFAULT_METHOD_CONTEXT_MAX_CHARS
    assert rec.active_snapshot_ref == "S0-seed-20260925-v1"
    assert rec.active_snapshot_manifest_sha256 == "61259a"
    assert rec.partition_policy == "question_type_partition_then_hard_requirements"
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


def test_every_rejected_candidate_keeps_a_filter_reason():
    """§13.6"候选及过滤原因"：被拦下的候选必须留在记录里并带原因码（此前只写日志）。"""
    skills = [
        _skill("count-1", task="object_counting"),
        _skill("metric-1", task="object_abs_distance", family="metric", metric=True,
               signature={"metric_scale": "available"}),
        _skill("gate-shut", task="object_abs_distance", family="metric", metric=True,
               signature={"metric_scale": "available"}),
    ]
    scene = SceneState(artifact_ref="a", scene_route="fallback_2d_only",
                       evidence_profile=_profile({"metric_scale": "unavailable"}),
                       metric_evidence_gate_result=_gate(passed=False),
                       objects=[], summary="s")
    hits, rec = retrieve_ex(QUESTION, scene, skills, question_type="object_counting")
    assert [h.skill_version for h in hits] == ["count-1@1.0.0"]
    reasons = {r.skill_version: r.reason_code for r in rec.candidates}
    # §13.5 先精确过滤题型分区：另两条是米制题型的，压根不参与本题竞争
    assert reasons["metric-1@1.0.0"] == "question_type_mismatch"
    assert reasons["gate-shut@1.0.0"] == "question_type_mismatch"
    assert len(rec.candidates) == 3 and rec.n_skills_offered == 3
    for row in rec.candidates:
        if not row.hard_filter_passed:
            assert row.reason and row.reason_code in CANDIDATE_REASON_CODES


def test_metric_skill_rejection_reason_codes_are_distinguished():
    """米制 Skill 的三类拒绝（门未过／版本不符／连 gate 都没有）必须可区分。

    签名取 `metric_scale>=degraded`：这条**能**在当前证据下通过，于是判定会走到
    §13.6 的米制 gate 双重校验（否则先被签名拦下，看不出 gate 的原因码）。
    """
    skill = _skill("metric-1", task="object_abs_distance", family="metric", metric=True,
                   signature={"metric_scale": "degraded"})
    s_degraded = _profile({"metric_scale": "degraded"})

    s_unmet = SceneState(artifact_ref="a", scene_route="fallback_2d_only",
                         evidence_profile=s_degraded,
                         metric_evidence_gate_result=_gate(passed=False),
                         objects=[], summary="s")
    _hits, rec = retrieve_ex("How far?", s_unmet, [skill],
                             question_type="object_abs_distance")
    assert rec.candidates[0].reason_code == "metric_gate_not_passed"

    s_bad_version = SceneState(artifact_ref="a", scene_route="full_3d",
                               evidence_profile=s_degraded,
                               metric_evidence_gate_result=_gate(version="gate-v0"),
                               objects=[], summary="s")
    _hits, rec2 = retrieve_ex("How far?", s_bad_version, [skill],
                              question_type="object_abs_distance")
    assert rec2.candidates[0].reason_code == "metric_gate_version_mismatch"

    s_no_gate = SceneState(artifact_ref="a", scene_route="full_3d",
                           evidence_profile=s_degraded,
                           metric_evidence_gate_result=None, objects=[], summary="s")
    _hits, rec3 = retrieve_ex("How far?", s_no_gate, [skill],
                              question_type="object_abs_distance")
    assert rec3.candidates[0].reason_code == "gate_result_missing"
    assert rec3.eligible_skill_versions == []


def test_unknown_question_type_records_why_nothing_was_retrieved():
    """题型不可知 → fail-closed 不检索；记录必须能回答"为什么一条都没有"。"""
    hits, rec = retrieve_ex(QUESTION, _scene(), [_skill()], question_type="not_a_task")
    assert hits == []
    assert rec.question_type_known is False
    assert rec.n_skills_offered == 1
    assert [r.reason_code for r in rec.candidates] == ["question_type_unknown"]
    assert rec.delivery_note


def test_not_selected_top_k_is_recorded_with_rank_and_score():
    """top-k 落选者也要留痕：分数、名次、原因码（§13.6"排序分数、选中版本"）。"""
    skills = [_skill("sk-a"), _skill("sk-b"), _skill("sk-c")]
    hits, rec = retrieve_ex("How many chairs?", _scene(), skills,
                            question_type="object_counting",
                            policy=RetrievalPolicy(top_k=2))
    assert len(hits) == 2
    rows = sorted(rec.candidates, key=lambda r: r.rank)
    assert [r.rank for r in rows] == [1, 2, 3]
    assert all(r.score is not None for r in rows)
    assert sum(1 for r in rows if r.selected) == 2
    loser = rows[2]
    assert loser.selected is False and loser.reason_code == "not_selected_top_k"
    # 交付状态词表里"没被选中"只有 `not_selected`（不细分到 top-k）；
    # 为什么没入选看 `reason_code`，两者含义不同。
    assert loser.delivery_reason == "not_selected"
    assert rec.retrieved_skill_versions == [r.skill_version for r in rows[:2]]


def test_retrieve_wrapper_matches_retrieve_ex():
    """§13.5：带记录与不带记录的两条路径不得行为分叉。"""
    skills = [_skill("sk-a"), _skill("sk-b", desc="另一个计数方法",
                                     template="ReturnAnswer(str(2))")]
    scene = _scene()
    plain = retrieve(QUESTION, scene, skills, question_type="object_counting", top_k=2)
    detailed, _rec = retrieve_ex(QUESTION, scene, skills, question_type="object_counting",
                                 top_k=2)
    assert [h.model_dump() for h in plain] == [h.model_dump() for h in detailed]


# ------------------------------------------------- §13.5 上下文上限与静态检查 ----

def test_plan_delivery_drops_whole_entries_and_never_truncates():
    """§13.5：放不下时**先减少完整条目**，绝不截断正文后仍称完整交付。"""
    skills = [_skill("sk-a", desc="A" * 400), _skill("sk-b", desc="B" * 400),
              _skill("sk-c", desc="C" * 400)]
    sizes = [len(render_skill_entry(s)) for s in skills]
    cap = sizes[0] + 1 + sizes[1]          # 只放得下前两条（含分隔符）
    plan = plan_delivery(skills, max_chars=cap)
    assert [e.skill_version for e in plan.entries] == ["sk-a@1.0.0", "sk-b@1.0.0"]
    assert [d.skill_version for d in plan.dropped] == ["sk-c@1.0.0"]
    assert plan.dropped[0].reason == "context_cap_exceeded"
    assert plan.used_chars <= cap
    # 交付正文字节级完整：每条 entry 的 text 就是 render_skill_entry 的输出
    for e, s in zip(plan.entries, skills):
        assert e.text == render_skill_entry(s)
        assert e.chars == len(e.text)


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

def test_policy_version_is_content_derived():
    """§13.6"配置版本"：改任何一个参数，内容摘要必变（标签不能抵赖内容）。"""
    base = RetrievalPolicy()
    same = RetrievalPolicy()
    assert base.sha256() == same.sha256()
    for changed in (RetrievalPolicy(top_k=1), RetrievalPolicy(rerank=False),
                    RetrievalPolicy(candidates=7),
                    RetrievalPolicy(method_context_max_chars=999),
                    RetrievalPolicy(rank_weights={**base.rank_weights, "keyword": 2.0})):
        assert changed.sha256() != base.sha256()
    # 标签只影响人类可读版本名，不影响内容摘要
    labelled = RetrievalPolicy(label="ret-frozen-7")
    assert labelled.version() == "ret-frozen-7"
    assert labelled.sha256() == base.sha256()
    assert base.version().startswith("ret-")


def test_policy_rejects_unknown_weights_and_bad_values():
    with pytest.raises(ValueError):
        RetrievalPolicy(rank_weights={"semantic": 1.0, "keyword_typo": 1.0})
    with pytest.raises(ValueError):
        RetrievalPolicy(top_k=0)
    with pytest.raises(ValueError):
        RetrievalPolicy(method_context_max_chars=0)


def test_policy_read_from_main_config_and_marks_source():
    """§13.5：策略写在 configs/config.yaml 并在演化前冻结；缺段时如实记 default。"""
    from skill3d.online.config import DEFAULT_CONFIG, load_config

    policy = retrieval_policy_from_config(load_config(DEFAULT_CONFIG))
    assert policy.source == "config"
    assert policy.top_k == 3
    assert policy.method_context_max_chars == DELIVERY_DEFAULT_CAP
    assert policy.weight("semantic") == 1.0 and policy.weight("semantic_mix_keyword") == 0.0
    assert policy.label == "ret-v11-deterministic-1"
    fallback = retrieval_policy_from_config({})
    assert fallback.source == "default"


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


def test_context_cap_drop_is_recorded_per_candidate():
    """被上限整条丢弃的条目：delivered=False 且原因是 context_cap_exceeded。"""
    skills = [_skill("sk-a"), _skill("sk-b")]
    _hits, rec = retrieve_ex(QUESTION, _scene(), skills, question_type="object_counting",
                             policy=RetrievalPolicy(top_k=2))
    tiny = len(render_skill_entry(skills[0]))
    plan = plan_delivery(skills, max_chars=tiny)
    plan.mark_delivered()
    rec.add_delivery(plan, round_index=1)
    rows = {r.skill_version: r for r in rec.candidates}
    assert rows["sk-a@1.0.0"].delivered is True
    assert rows["sk-b@1.0.0"].delivered is False
    assert rows["sk-b@1.0.0"].delivery_reason == "context_cap_exceeded"
    assert [d["skill_version"] for d in rec.dropped_for_context] == ["sk-b@1.0.0"]


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
