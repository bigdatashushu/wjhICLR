"""G-20 单测：语义检索（hashing 嵌入 + 可选 LanceDB）与硬过滤。

验收口径（§7.1 G-20）：
- 语义排序 top-1 与关键词打分 top-1 **不同**（证明语义排序生效）；
- `metric_scale_required=True` 的 skill 在 `scale_unknown` scene 上被硬过滤；
- 未 promoted/consolidated 的 skill 不出现在结果中。
"""

from __future__ import annotations

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

QUESTION = "How many chairs are in this room?"


def _skill(skill_id="sk-1", semver="1.0.0", task="object_counting", desc="",
           template="", min_q=0.0, metric=False, artifacts=(), frames=("world",),
           state=None) -> SkillSpec:
    s = SkillSpec(
        skill_id=skill_id, semver=semver, task_type=task, description=desc,
        call_graph_template=template or "ReturnAnswer(str(len(scene.list_objects())))",
        requires_artifacts=list(artifacts), minimum_quality=min_q,
        supported_coordinate_frames=list(frames), metric_scale_required=metric,
        validation_assertions=[])
    if state is not None:
        object.__setattr__(s, "state", state)      # duck-typing：registry 层注入
    return s


def _scene(route="full_3d", scale_known=True, frame="world") -> SceneState:
    return SceneState(artifact_ref="a", route=route, frame=frame,
                      scale_known=scale_known, objects=[], summary="s",
                      scale_confidence="high" if scale_known else "low", scale_ci_rel=0.02)


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


# ------------------------------------------------------------------ 硬过滤 ----

def test_hard_filter_metric_scale_required():
    assert hard_filter(_skill(metric=True), _scene(scale_known=True))
    assert not hard_filter(_skill(metric=True), _scene(scale_known=False))


def test_hard_filter_requires_artifacts_by_route():
    needs_objects = _skill(artifacts=["objects"])
    assert hard_filter(needs_objects, _scene(route="full_3d"))
    assert not hard_filter(needs_objects, _scene(route="fallback_2d_only"))


def test_hard_filter_task_type_and_frames_and_quality():
    assert not hard_filter(_skill(task="object_counting"), _scene(),
                           task_type="room_size_estimation")
    assert not hard_filter(_skill(frames=["camera"]), _scene(frame="world"))
    assert not hard_filter(_skill(min_q=0.9), _scene(), scene_quality=0.5)


def test_hard_filter_rejects_non_active_state():
    """未 promoted/consolidated 的候选不得生效（§7）。"""
    assert hard_filter(_skill(state="promoted"), _scene())
    for bad in ("draft", "shadow", "canary", "quarantined"):
        assert not hard_filter(_skill(state=bad), _scene()), bad


# ------------------------------------------------------------------ retrieve ----

def test_retrieve_filters_and_ranks_with_semantics():
    """语义排序把"模板真正解此题"的 skill 排前，即使它的描述用词与问题不同。"""
    decoy = _skill("sk-room", desc="measure the area of the room in square meters")
    right = _skill("sk-count", desc="inventory of furniture",
                   template=("# object_counting: how many chairs are in this room\n"
                             "n = len(scene.list_objects())\nReturnAnswer(str(n))"))
    got = retrieve(QUESTION, _scene(), [decoy, right], question_type="object_counting")
    assert got and got[0].skill_semver == "sk-count@1.0.0"
    assert all(r.hard_filter_passed for r in got)
    assert got[0].score > got[1].score


def test_semantic_top1_differs_from_keyword_top1():
    """G-20 验收：语义排序 top-1 与关键词打分 top-1 不同。

    构造差异的来源是**索引文本不同**：关键词打分只看 `task_type + description`，
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
    assert sem and sem[0].skill_semver == "sk-semantic@1.0.0"
    assert kw[0] == "sk-keyword@1.0.0"
    assert sem[0].skill_semver != kw[0]                       # 两条口径确实不同


def test_retrieve_rerank_false_uses_keyword_ranking():
    a = _skill("sk-keyword", desc="chairs chairs table lamp")
    b = _skill("sk-semantic", desc="count the seating furniture items present")
    got = retrieve(QUESTION, _scene(), [a, b], question_type="object_counting",
                   rerank=False, top_k=2)
    assert [r.skill_semver for r in got] == keyword_only_ranking(QUESTION, [a, b])


def test_retrieve_returns_empty_when_hard_filter_kills_all():
    got = retrieve(QUESTION, _scene(scale_known=False),
                   [_skill(metric=True)], question_type="object_counting")
    assert got == []


def test_retrieve_no_task_type_match_returns_empty():
    got = retrieve(QUESTION, _scene(), [_skill(task="room_size_estimation")],
                   question_type="object_counting")
    assert got == []


def test_retrieve_respects_top_k_and_determinism():
    skills = [_skill(f"sk-{i}", desc=f"count objects variant {i}") for i in range(5)]
    a = retrieve(QUESTION, _scene(), skills, question_type="object_counting", top_k=3)
    b = retrieve(QUESTION, _scene(), skills, question_type="object_counting", top_k=3)
    assert len(a) == 3
    assert [r.skill_semver for r in a] == [r.skill_semver for r in b]


def test_build_index_and_skill_text_include_template():
    s = _skill("sk-x", desc="d", template="room_size_m2()")
    assert "room_size_m2" in skill_text(s)
    idx = build_index([s])
    assert len(idx) == 1
    assert idx.metadata_of("sk-x@1.0.0")["task_type"] == "object_counting"


def test_direction_variant_question_type_maps_to_canonical_task():
    """官方 question_type 变体（object_rel_direction_hard）应命中规范题型 Skill。"""
    s = _skill("sk-dir", task="object_rel_direction", desc="relative direction left right")
    got = retrieve("Which direction is the sofa from the chair?",
                   _scene(), [s], question_type="object_rel_direction_hard")
    assert [r.skill_semver for r in got] == ["sk-dir@1.0.0"]
