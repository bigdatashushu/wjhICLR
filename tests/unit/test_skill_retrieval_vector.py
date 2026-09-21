"""G-20 单测：语义检索（hashing 嵌入 + 可选 LanceDB）与 v6 证据签名硬过滤。

v6 验收口径（§17.1/§17.2/§13.6）：
- 语义排序 top-1 与关键词打分 top-1 **不同**（证明语义排序生效）；
- 检索硬条件 = **题型匹配 ∧ EvidenceProfile 证据签名匹配**：
  声明 `{"metric_scale": "degraded"}` 的 Skill 在 `metric_scale=unavailable` 时**不得**被检索，
  在 `degraded`/`available` 时**应当**被检索；无签名声明的 Skill 任意证据状态下都可检索；
- 米制 Skill **双重 fail-closed**：`requires_metric_evidence=True` 时还要 gate 通过且
  `applicable_gate_version` 与当前 `gate_version` 一致，不匹配 → 不可检索；
- `matched_evidence_signature` = 当前画像在 Skill 签名键上的投影（§17.2 分开积累、互不污染）；
- 未 promoted/consolidated 的 skill 不出现在结果中。
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from skill3d.memory.vector_index import (
    HashingEmbedder,
    SkillVectorIndex,
    cosine,
    lancedb_available,
    rerank_scores,
    top_k_by_score,
)
from skill3d.routing.skill_retriever import (
    build_index,
    hard_filter,
    keyword_only_ranking,
    retrieve,
    skill_text,
)
from skill3d.schemas import SceneState, SkillSpec
from skill3d.schemas.evidence import (
    GATE_SUBCONDITIONS,
    GATE_VERSION,
    EvidenceProfile,
    MetricEvidenceGateResult,
)
from skill3d.schemas.skill import FAMILY_QUESTION_TYPES

QUESTION = "How many chairs are in this room?"

# 规范题型 → Skill 族（SkillSpec 构造期强制"族的题型覆盖"，测试里自动补齐）
_TASK_FAMILY: dict[str, str] = {
    task: fam for fam, tasks in FAMILY_QUESTION_TYPES.items() for task in tasks
}


def _skill(skill_id="sk-1", version="1.0.0", task="object_counting", desc="",
           template="", family=None, signature=None, metric=False, gate_version=None,
           frames=("world",), source="real", state=None) -> SkillSpec:
    """构造 v6 SkillSpec：题型 + 证据签名（+ 米制 Skill 的 gate 版本声明）。"""
    tasks = [task] if isinstance(task, str) else list(task)
    s = SkillSpec(
        skill_id=skill_id, version=version, applicable_question_types=tasks,
        required_evidence_signature=dict(signature or {}),
        requires_metric_evidence=bool(metric),
        applicable_gate_version=(gate_version or (GATE_VERSION if metric else None)),
        skill_family=family or _TASK_FAMILY[tasks[0]],
        source=source, description=desc,
        call_graph_template=template or "ReturnAnswer(str(len(scene.list_objects())))",
        supported_coordinate_frames=list(frames), validation_assertions=[])
    if state is not None:
        object.__setattr__(s, "state", state)      # duck-typing：registry 层注入
    return s


def _profile(states=None) -> EvidenceProfile:
    """默认"健康场景"画像：8 项能力全 available；`states` 覆盖指定能力。"""
    base = {
        "geometry_3d": "available", "world_frame": "available",
        "metric_scale": "available", "object_detection": "available",
        "track_consensus": "available", "object_grounding": "available",
    }
    base.update(states or {})
    return EvidenceProfile(**base)


def _gate(passed=True, version=GATE_VERSION) -> MetricEvidenceGateResult:
    """6 项子条件同真/同假的米制门（`gate_passed` 与 `sub_results` 必须自洽）。"""
    return MetricEvidenceGateResult(
        gate_passed=bool(passed), gate_version=version,
        sub_results={name: bool(passed) for name in GATE_SUBCONDITIONS},
        values={}, missing_subconditions=[] if passed else list(GATE_SUBCONDITIONS))


def _scene(scene_route="full_3d", states=None, gate="default") -> SceneState:
    """v6 SceneState：scene_route + EvidenceProfile + 米制门。"""
    g = _gate() if gate == "default" else gate
    return SceneState(
        artifact_ref="a", scene_route=scene_route,
        evidence_profile=_profile(states), metric_evidence_gate_result=g,
        objects=[], summary="s")


def _no_gate_scene() -> SceneState:
    """连 MetricEvidenceGateResult 都没有的场景（旧 artifact）。"""
    return SceneState(artifact_ref="a", scene_route="full_3d",
                      evidence_profile=_profile(), objects=[], summary="s")


# ------------------------------------------------------------------ 嵌入索引 ----

def test_hashing_embedder_is_deterministic_and_normalized():
    e = HashingEmbedder(dim=256)
    a, b = e.embed("count the chairs"), e.embed("count the chairs")
    assert np.array_equal(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert e.embed("").sum() == 0.0


def test_embedding_similarity_reflects_morphology_not_only_exact_tokens():
    """字符 n-gram 参与 → 形态相近的词也能产生相似度（语义排序的基础）。"""
    e = HashingEmbedder(dim=512)
    q = e.embed("chairs seating furniture")
    same_root = e.embed("chair seat furniture")
    unrelated = e.embed("triangle area computation geometry")
    assert cosine(q, same_root) > cosine(q, unrelated)


def test_index_search_and_where_filter():
    idx = SkillVectorIndex()
    idx.add_text("a", "counting objects in a room", task_type="object_counting")
    idx.add_text("b", "measuring room area in square meters",
                 task_type="room_size_estimation")
    hits = idx.search_text("how many objects are present", top_k=2)
    assert hits[0][0] == "a" and hits[0][1] > hits[1][1]
    only_room = idx.search_text("how many objects", where={"task_type":
                                                          "room_size_estimation"})
    assert [h[0] for h in only_room] == ["b"]
    assert idx.search_text("x", where={"task_type": "nonexistent"}) == []


def test_index_is_deterministic_across_insert_order():
    e = HashingEmbedder(dim=256)
    a = SkillVectorIndex(e)
    a.add_text("x", "alpha beta", k=1)
    a.add_text("y", "gamma delta", k=2)
    b = SkillVectorIndex(e)
    b.add_text("y", "gamma delta", k=2)
    b.add_text("x", "alpha beta", k=1)
    assert a.search_text("alpha", top_k=2) == b.search_text("alpha", top_k=2)


def test_rerank_scores_falls_back_to_embedding_without_model():
    scores = rerank_scores("count the chairs", ["counting seats", "room area"])
    assert len(scores) == 2 and scores[0] > scores[1]


def test_top_k_by_score_is_deterministic_on_ties():
    got = top_k_by_score([("b", 1.0), ("a", 1.0), ("c", 0.5)], 3)
    assert [k for k, _ in got] == ["a", "b", "c"]
    assert top_k_by_score([("a", 1.0)], 0) == []


def test_lancedb_roundtrip_when_available(tmp_path):
    if not lancedb_available():
        pytest.skip("lancedb 未安装（可选依赖）")
    idx = SkillVectorIndex()
    idx.add_text("a", "counting objects", task_type="object_counting")
    idx.add_text("b", "room area", task_type="room_size_estimation")
    assert idx.to_lancedb(str(tmp_path / "db")) is True
    back = SkillVectorIndex.from_lancedb(str(tmp_path / "db"))
    assert len(back) == 2
    assert back.search_text("counting objects", top_k=1)[0][0] == "a"
    assert back.metadata_of("b")["task_type"] == "room_size_estimation"


# --------------------------------------------------- v6 硬过滤：题型 + 证据签名 ----

def test_hard_filter_requires_question_type_match():
    """§17.1：题型不在 `applicable_question_types` 内 → 不检索（证据再全也不行）。"""
    s = _skill(task="object_counting")
    assert hard_filter(s, _scene(), question_type="object_counting")
    assert hard_filter(s, _scene(), task_type="object_counting")        # 规范题型入口
    assert not hard_filter(s, _scene(), question_type="route_planning")
    assert not hard_filter(s, _scene(), task_type="room_size_estimation")
    # 官方变体题型（三档变体）先规范化为 8 题型之一
    d = _skill(task="object_rel_direction")
    assert hard_filter(d, _scene(), question_type="object_rel_direction_hard")


def test_hard_filter_fails_closed_when_question_type_unknown_or_missing():
    """题型不可知（缺失 / 未知取值）→ 不检索任何 Skill（fail-closed，§17.1）。"""
    s = _skill()
    assert not hard_filter(s, _scene())                                   # 没给题型
    assert not hard_filter(s, _scene(), question_type="not_a_real_type")
    assert not hard_filter(s, _scene(), question_type="")


def test_hard_filter_requires_evidence_signature_state():
    """§17.1：签名要求 `metric_scale>=degraded` → unavailable 时不得检索。"""
    s = _skill(task="room_size_estimation", signature={"metric_scale": "degraded"})
    qt = "room_size_estimation"
    assert not hard_filter(s, _scene(states={"metric_scale": "unavailable"}),
                           question_type=qt)
    assert hard_filter(s, _scene(states={"metric_scale": "degraded"}),      # 等于要求 → 满足
                       question_type=qt)
    assert hard_filter(s, _scene(states={"metric_scale": "available"}),     # 高于要求 → 满足
                       question_type=qt)


def test_hard_filter_requires_geometry_evidence_for_relative_geometry_skill():
    """几何能力 unavailable 时，依赖它的相对几何 Skill 不得被检索（§7.2 单项收窄）。"""
    s = _skill(task="object_rel_distance",
               signature={"geometry_3d": "available", "object_grounding": "degraded"})
    qt = "object_rel_distance"
    assert not hard_filter(s, _scene(states={"geometry_3d": "degraded"}), question_type=qt)
    assert not hard_filter(s, _scene(states={"object_grounding": "unavailable"}),
                           question_type=qt)
    assert hard_filter(s, _scene(states={"geometry_3d": "available",
                                         "object_grounding": "degraded"}), question_type=qt)


def test_hard_filter_empty_signature_is_state_agnostic():
    """无证据签名声明 = 不额外增加证据硬条件 → 任意证据状态下都可检索（§17.1）。"""
    s = _skill()
    for state in ("available", "degraded", "unavailable"):
        assert hard_filter(s, _scene(states={
            "geometry_3d": state, "world_frame": state, "metric_scale": state,
            "object_detection": state, "track_consensus": state,
            "object_grounding": state}), question_type="object_counting"), state
    # v6 不再按 scene_route / scale_confidence 一刀切：连证据画像都没有（旧 artifact）
    # 的 2D-only 场景也不拦"无签名"Skill
    bare = SceneState(artifact_ref="a", scene_route="fallback_2d_only")
    assert hard_filter(s, bare, question_type="object_counting")


def test_hard_filter_metric_skill_requires_gate_and_version():
    """§13.6 双重 fail-closed：米制 Skill 还要 gate 通过 + `applicable_gate_version` 匹配。"""
    s = _skill(task="object_abs_distance", metric=True,
               signature={"metric_scale": "available"})
    qt = "object_abs_distance"
    assert s.requires_metric_evidence and s.applicable_gate_version == GATE_VERSION
    assert hard_filter(s, _scene(), question_type=qt)                     # gate 过 + 版本一致
    assert not hard_filter(s, _scene(gate=_gate(passed=False)), question_type=qt)
    assert not hard_filter(s, _scene(gate=_gate(version="metric-gate-v5")),
                           question_type=qt)                              # 版本不匹配
    assert not hard_filter(s, _no_gate_scene(), question_type=qt)         # 无 gate 结果


def test_hard_filter_rejects_non_active_state():
    """未 promoted/consolidated 的候选不得生效（§7）。"""
    assert hard_filter(_skill(state="promoted"), _scene(), question_type="object_counting")
    for bad in ("draft", "shadow", "canary", "quarantined"):
        assert not hard_filter(_skill(state=bad), _scene(),
                               question_type="object_counting"), bad


# ------------------------------------------------------------------ retrieve ----

def test_retrieve_filters_and_ranks_with_semantics():
    """语义排序把"模板真正解此题"的 skill 排前，即使它的描述用词与问题不同。"""
    decoy = _skill("sk-decoy", desc="furniture report routine",
                   template="ReturnAnswer('unrelated')")
    right = _skill("sk-count", desc="inventory of furniture",
                   template=("# object_counting: how many chairs are in this room\n"
                             "n = len(scene.list_objects())\nReturnAnswer(str(n))"))
    got = retrieve(QUESTION, _scene(), [decoy, right], question_type="object_counting")
    assert got and got[0].skill_id == "sk-count"
    assert got[0].skill_version == "sk-count@1.0.0"
    assert all(r.hard_filter_passed for r in got)
    assert got[0].score > got[1].score


def test_semantic_top1_differs_from_keyword_top1():
    """G-20 验收：语义排序 top-1 与关键词打分 top-1 不同。

    构造差异的来源是**索引文本不同**：关键词打分只看 `题型 + description`，
    语义索引还包含 `call_graph_template`（"这道题怎么做"）。于是：
    - skill A 描述里含问题词（关键词排它第一），但模板与题意无关；
    - skill B 描述用词完全不同，模板却是正确的计数程序 → 语义排它第一。
    """
    a = _skill("sk-keyword", desc="chairs in this room counting tables",
               template="ReturnAnswer('unrelated')")
    b = _skill("sk-semantic", desc="furniture inventory routine",
               template=("# how many chairs are in this room\n"
                         "n = len(scene.list_objects())\nReturnAnswer(str(n))"))
    sem = retrieve(QUESTION, _scene(), [a, b], question_type="object_counting", top_k=2)
    kw = keyword_only_ranking(QUESTION, [a, b])
    assert sem and sem[0].skill_id == "sk-semantic"
    assert kw[0] == "sk-keyword@1.0.0"
    assert sem[0].skill_version != kw[0]                       # 两条口径确实不同


def test_retrieve_rerank_false_uses_keyword_ranking():
    a = _skill("sk-keyword", desc="chairs chairs table lamp")
    b = _skill("sk-semantic", desc="count the seating furniture items present")
    got = retrieve(QUESTION, _scene(), [a, b], question_type="object_counting",
                   rerank=False, top_k=2)
    assert [r.skill_version for r in got] == keyword_only_ranking(QUESTION, [a, b])


def test_retrieve_returns_empty_when_evidence_signature_unmet():
    """证据签名不满足 → 一条都不检索（§17.1：不检索、不执行）。"""
    got = retrieve(QUESTION, _scene(states={"metric_scale": "unavailable"}),
                   [_skill(signature={"metric_scale": "degraded"})],
                   question_type="object_counting")
    assert got == []


def test_retrieve_returns_empty_when_metric_gate_version_mismatches():
    """§13.6：米制 Skill 的 gate 版本不匹配 → 不检索（即使证据画像看起来齐全）。"""
    metric = _skill("sk-metric", task="object_abs_distance", metric=True,
                    signature={"metric_scale": "available"})
    q = "How far is the chair from the table?"
    bad = _scene(gate=_gate(version="metric-evidence-gate-v5"))
    assert retrieve(q, bad, [metric], question_type="object_abs_distance") == []
    assert retrieve(q, _scene(gate=_gate(passed=False)), [metric],
                    question_type="object_abs_distance") == []
    got = retrieve(q, _scene(), [metric],
                   question_type="object_abs_distance")
    assert [r.skill_id for r in got] == ["sk-metric"]


def test_retrieve_no_question_type_match_returns_empty():
    got = retrieve(QUESTION, _scene(), [_skill(task="room_size_estimation")],
                   question_type="object_counting")
    assert got == []


def test_retrieve_respects_top_k_and_determinism():
    skills = [_skill(f"sk-{i}", desc=f"count objects variant {i}") for i in range(5)]
    a = retrieve(QUESTION, _scene(), skills, question_type="object_counting", top_k=3)
    b = retrieve(QUESTION, _scene(), skills, question_type="object_counting", top_k=3)
    assert len(a) == 3
    assert [r.skill_version for r in a] == [r.skill_version for r in b]


def test_retrieve_records_matched_evidence_signature_projection():
    """§17.2：`matched_evidence_signature` 是当前画像在 Skill 签名键上的投影（可审计）。"""
    s = _skill(signature={"geometry_3d": "available", "object_grounding": "degraded"})
    scene = _scene(states={"object_grounding": "degraded", "metric_scale": "unavailable"})
    got = retrieve(QUESTION, scene, [s], question_type="object_counting")
    assert [r.skill_id for r in got] == ["sk-1"]
    assert got[0].matched_evidence_signature == {"geometry_3d": "available",
                                                 "object_grounding": "degraded"}
    assert got[0].gate_version_matched is None        # 非米制 Skill：无 gate 语义


def test_retrieve_metric_skill_records_gate_version_matched():
    """§13.6：米制 Skill 的 gate 校验结果落进 `RetrievedSkill.gate_version_matched`。"""
    s = _skill("sk-metric", task="object_size_estimation", metric=True,
               signature={"metric_scale": "available", "geometry_3d": "available"})
    got = retrieve("What is the size of the table?", _scene(), [s],
                   question_type="object_size_estimation")
    assert [r.skill_id for r in got] == ["sk-metric"]
    assert got[0].gate_version_matched is True
    assert got[0].matched_evidence_signature == {"metric_scale": "available",
                                                 "geometry_3d": "available"}


def test_retrieve_separates_skills_by_evidence_signature():
    """§17.2：同一题型/同族的 Skill 按证据签名分开检索，互不污染。

    "metric_scale=unavailable" 签名下的相对几何 Skill 与"全 available"签名下的米制
    Skill 各走各的；尺度不可用时只有前者被检索到。
    """
    rel = _skill("sk-rel", task="object_rel_distance",
                 signature={"geometry_3d": "available", "metric_scale": "unavailable"},
                 desc="compare relative distance between two objects")
    metric = _skill("sk-metric", task="object_abs_distance", metric=True,
                    signature={"geometry_3d": "available", "metric_scale": "available"},
                    desc="absolute distance in meters")
    q = "Which object is closer, the chair or the sofa?"
    no_scale = _scene(states={"metric_scale": "unavailable"}, gate=_gate(passed=False))

    got = retrieve(q, no_scale, [rel, metric], question_type="object_rel_distance",
                   top_k=5)
    assert [r.skill_id for r in got] == ["sk-rel"]
    assert got[0].matched_evidence_signature == {"geometry_3d": "available",
                                                 "metric_scale": "unavailable"}

    # 同一 Skill 在"尺度可用"签名下被检索时，投影随当前画像变化（签名分桶的依据）
    got2 = retrieve(q, _scene(), [rel, metric], question_type="object_rel_distance",
                    top_k=5)
    assert [r.skill_id for r in got2] == ["sk-rel"]
    assert got2[0].matched_evidence_signature == {"geometry_3d": "available",
                                                  "metric_scale": "available"}

    # 米制题：只有米制 Skill 命中，且 gate 校验记入输出
    got3 = retrieve("How far is the chair from the table?", _scene(), [rel, metric],
                    question_type="object_abs_distance", top_k=5)
    assert [r.skill_id for r in got3] == ["sk-metric"] and got3[0].gate_version_matched


def test_build_index_and_skill_text_include_template():
    s = _skill("sk-x", desc="d", template="room_size_m2()")
    assert "room_size_m2" in skill_text(s)
    idx = build_index([s])
    assert len(idx) == 1
    md = idx.metadata_of("sk-x@1.0.0")
    assert md["task_type"] == "object_counting"
    assert md["skill_family"] == "counting"


def test_direction_variant_question_type_maps_to_canonical_task():
    """官方 question_type 变体（object_rel_direction_hard）应命中规范题型 Skill。"""
    s = _skill("sk-dir", task="object_rel_direction", desc="relative direction left right")
    got = retrieve("Which direction is the sofa from the chair?",
                   _scene(), [s], question_type="object_rel_direction_hard")
    assert [r.skill_version for r in got] == ["sk-dir@1.0.0"]


def test_retrieval_has_no_vlm_adjudication_channel():
    """§16.1：检索只用可观测证据（EvidenceProfile + grounding + question_type），

    不接任何 VLM/离线模型客户端，也没有"让模型挑 Skill"的入参。
    """
    from skill3d.routing import skill_retriever as sr

    tree = ast.parse(inspect.getsource(sr))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    banned = ("vllm", "openai", "governance", "deepseek", "gpt")
    assert not [m for m in modules if any(b in m.lower() for b in banned)], modules
    params = set(inspect.signature(sr.retrieve).parameters)
    assert not [p for p in params if any(k in p.lower() for k in ("llm", "vlm", "judge"))]
