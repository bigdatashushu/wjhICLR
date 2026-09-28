"""v10 §8.2/§8.3/§5.3 集成测试：固定注入臂与谱系内版本选择（真实 runner 路径）。

规范原文（§8.2）："评测目标 Skill 时跳过检索选择，但不跳过 Skill 的适用性、内容长度、
工具权限和 Schema 校验。"

规范原文（§8.3）："禁止：A 使用空 Skill、B 使用候选 Skill；……将固定注入记录成正常
检索命中。"

规范原文（§5.3）："2. 检索先按题型过滤，再按谱系分组，最后在谱系内选版本；
3. `top_k` 针对方法谱系，而不是任意版本条目；……7. 检索分数、稳定 tie-break 和版本
选择原因必须落盘。"

本文件走**真实 runner 路径**（mock_light 全链 + real 模式的 `_synthesize`），不触网。
"""

from __future__ import annotations

import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.online.runner import (
    FixedSkillInjectionError,
    OnlineRunConfig,
    _synthesize,
    run_episode,
)
from skill3d.routing.skill_retriever import retrieve_ex
from skill3d.schemas import SkillSpec
from skill3d.skills.delivery import render_skill_entry, skill_content_sha256
from skill3d.schemas.evolution import SkillEvaluationBinding

FRAME_SIZE = (120, 160)

from test_tool_contract_recovery import _FakeClient  # noqa: E402 - 同目录测试夹具


def _counting_skill(version="1.0.0", desc=None) -> SkillSpec:
    return SkillSpec(
        skill_id="S01", version=version,
        applicable_question_types=["object_counting"], skill_family="counting",
        source="real",
        description=desc or f"数出目标对象：先定类别，再跨帧覆盖（v{version}）。",
        call_graph_template=("n = detect_objects(img, category)\n"
                             "ReturnAnswer(str(len(n)))"))


@pytest.fixture(scope="module")
def counting_item():
    return load_synthetic_items("inner_validation", question_types=["object_counting"],
                                frame_size=FRAME_SIZE, seed=0)[0]


def _binding(arm: str, spec: SkillSpec) -> SkillEvaluationBinding:
    return SkillEvaluationBinding(
        mode="fixed_skill_evaluation", arm=arm, skill_id=spec.skill_id,
        skill_version=f"{spec.skill_id}@{spec.version}",
        content_sha256=skill_content_sha256(spec),
        bypassed_component="retrieval_selection")


# ---------------------------------------------------------------- §8.2 固定注入 ----

def test_fixed_injection_is_recorded_as_injection_not_a_retrieval_hit(counting_item,
                                                                      tmp_path):
    """§8.3：固定注入必须显式标记，候选行**不得**记成 `hit`。"""
    spec = _counting_skill()
    cfg = OnlineRunConfig(mode="mock_light", skills=[spec], memory_dir="",
                          trace_dir=str(tmp_path / "t"), active_snapshot_ref="S1-test",
                          evaluation_binding=_binding("candidate", spec))
    out = run_episode(counting_item.episode, counting_item.pixels, cfg,
                      geometry=counting_item.geometry)

    record = out.retrieval_records[0]
    assert record.evaluation_binding is not None
    assert record.evaluation_binding["mode"] == "fixed_skill_evaluation"
    assert record.evaluation_binding["arm"] == "candidate"
    assert record.evaluation_binding["bypassed_component"] == "retrieval_selection"
    row = record.candidates[0]
    assert row.reason_code == "fixed_injection_selected"
    assert row.reason_code != "hit"
    assert row.hard_filter_passed and row.selected
    assert record.lineage_selections == []          # 固定注入没有"检索选择"这回事
    assert record.selection_mode == "fixed_skill_evaluation"


def test_fixed_injection_still_checks_applicability(counting_item, tmp_path):
    """§8.2：跳过的是**检索选择**，不是适用性校验；不适用要吵出来（不得跑成空臂）。"""
    wrong = SkillSpec(skill_id="S07", version="1.0.0",
                      applicable_question_types=["room_size_estimation"],
                      skill_family="metric", source="real")
    cfg = OnlineRunConfig(mode="mock_light", skills=[wrong], memory_dir="",
                          trace_dir=str(tmp_path / "t"),
                          evaluation_binding=_binding("parent", wrong))
    with pytest.raises(FixedSkillInjectionError):
        run_episode(counting_item.episode, counting_item.pixels, cfg,
                    geometry=counting_item.geometry)


def test_fixed_injection_requires_exactly_one_skill(counting_item, tmp_path):
    """§8.2：每次请求只包含该臂被评测的一条完整 Skill（多条/零条都非法）。"""
    a, b = _counting_skill("1.0.0"), _counting_skill("1.1.0")
    with pytest.raises(ValueError):
        OnlineRunConfig(mode="mock_light", skills=[a, b], memory_dir="",
                        evaluation_binding=_binding("parent", a))
    with pytest.raises(ValueError):
        OnlineRunConfig(mode="mock_light", skills=[], memory_dir="",
                        evaluation_binding=_binding("parent", a))
    # 绑定版本与被注入 Skill 不一致也要拒（§8.2：绑定是这一臂的身份）
    with pytest.raises(ValueError):
        OnlineRunConfig(mode="mock_light", skills=[a], memory_dir="",
                        evaluation_binding=_binding("parent", b))


def test_fixed_injection_delivers_exact_arm_body_in_real_mode(counting_item, tmp_path):
    """§16.2：A/B 两臂各自把**自己的**完整正文送进请求，正文 hash 逐臂可核对。"""
    parent = _counting_skill("1.0.0")
    candidate = _counting_skill("1.1.0", desc="候选：空检出错时先复核再计数(1.1.0)")
    program = "ReturnAnswer(str(len(detect_objects(img, 'chair'))))\n"
    for arm, spec in (("parent", parent), ("candidate", candidate)):
        client = _FakeClient([program])
        cfg = OnlineRunConfig(mode="real", vllm_endpoints=["http://fake"], skills=[spec],
                              seed=0, evaluation_binding=_binding(arm, spec))
        res = _synthesize(counting_item.episode, scene=None, handle=None, skills=[spec],
                          cfg=cfg, llm=client, pixels=list(counting_item.pixels))
        assert res.program is not None, res.note
        assert res.delivery.channel == "model_request"
        assert res.delivery.delivered_skill_versions == [
            f"{spec.skill_id}@{spec.version}"]
        assert res.delivery.delivered_content_sha256 == {
            f"{spec.skill_id}@{spec.version}": skill_content_sha256(spec)}
        sent = "\n".join(
            part.get("text", "")
            for m in client.calls[0] for part in (m.get("content") or [])
            if isinstance(part, dict)) if isinstance(client.calls[0][0].get("content"),
                                                     list) else str(
            client.calls[0][0].get("content", ""))
        assert render_skill_entry(spec) in sent                    # 本臂正文逐字入请求
        other = render_skill_entry(candidate if arm == "parent" else parent)
        assert other not in sent                                   # 另一臂正文不在


# ---------------------------------------------------------------- §5.3 谱系选择 ----

def test_lineage_picks_one_version_and_records_why():
    """§5.3-1/2/7：同谱系每题只选一个版本，分数与 tie-break 原因落盘。"""
    from skill3d.schemas import SceneState

    scene = SceneState(artifact_ref="a", scene_route="fallback_2d_only",
                       evidence_profile=None)
    v1, v2 = _counting_skill("1.0.0"), _counting_skill("1.1.0")
    hits, record = retrieve_ex("数一下有几个椅子", scene, [v1, v2],
                               question_type="object_counting", rerank=False)
    assert len(hits) == 1                                  # 一个谱系只交付一个版本
    assert record.retrieved_skill_versions == [hits[0].skill_version]
    rows = {r.skill_version: r for r in record.candidates}
    assert rows[hits[0].skill_version].selected is True
    assert len(record.lineage_selections) == 1
    selection = record.lineage_selections[0]
    assert selection["lineage"] == "S01"
    assert selection["selected_version"] == hits[0].skill_version
    assert selection["ranking_policy"] == "score_desc_then_stable_key"
    assert selection["reason"] in ("score_tie_break_stable_key", "higher_ranking_score",
                                   "single_eligible_version")
    assert {c["skill_version"] for c in selection["candidates"]} == {
        "S01@1.0.0", "S01@1.1.0"}
    assert all(c["score"] is not None for c in selection["candidates"])
    other = [r for r in record.candidates if not r.selected][0]
    assert other.reason_code == "lineage_version_not_selected"
    assert "最多交付一个版本" in other.reason


def test_lineage_top_k_counts_lineages_not_versions():
    """§5.3-3：`top_k` 针对方法谱系 —— 两个谱系各两版本仍应各交付一个版本。"""
    from skill3d.schemas import SceneState
    from skill3d.schemas.evidence import EvidenceProfile

    scene = SceneState(
        artifact_ref="a", scene_route="full_3d",
        evidence_profile=EvidenceProfile(
            geometry_3d="available", world_frame="available", metric_scale="available",
            object_detection="available", track_consensus="available",
            object_grounding="available"))
    skills = [
        _counting_skill("1.0.0"), _counting_skill("1.1.0"),
        SkillSpec(skill_id="S09", version="1.0.0",
                  applicable_question_types=["object_counting"], skill_family="counting",
                  source="real", description="另一个计数方法：先数再复核",
                  call_graph_template="n = detect_objects(img, 'chair')"),
    ]
    hits, record = retrieve_ex("数一下有几个椅子", scene, skills,
                               question_type="object_counting", rerank=False, top_k=2)
    assert sorted(h.skill_id for h in hits) == ["S01", "S09"]
    assert len(record.lineage_selections) == 2


def test_third_active_version_historizes_the_oldest_at_publish(tmp_path):
    """§5.3-4/5：同谱系最多两个在线竞争版本；第三个进入时最旧转 historical。

    （发布侧的完整断言在 `tests/unit/test_v10_campaign.py`；这里单独锁"最旧"的
    定义 = 版本号最小，且历史版本仍保留在快照里。）
    """
    import json
    from pathlib import Path

    from skill3d.schemas import SkillCandidate
    from skill3d.skills.promote_atomic import publish_candidate_snapshot
    from skill3d.skills.registry import load_active_skills

    lib = tmp_path / "lib"
    store = lib / "snapshots"
    store.mkdir(parents=True)
    spec = _counting_skill("1.0.0")
    (store / "snapshot_S0.json").write_text(json.dumps({
        "snapshot_id": "S0", "generation": 0, "manifest_hash": "x",
        "entries": {"S01@1.0.0": {
            "root_candidate_id": "S01", "candidate_type": "skill",
            "skill_version": "S01@1.0.0",
            "spec_content": json.dumps(spec.model_dump(mode="json"))}}},
        ensure_ascii=False), encoding="utf-8")
    (store / "active_snapshot.json").write_text(json.dumps({"snapshot_id": "S0"}),
                                                encoding="utf-8")

    def _candidate(version: str, parent: str, parent_snapshot: str, generation: int):
        child = _counting_skill(version, desc=f"v{version} 的方法")
        return SkillCandidate(
            candidate_id=f"c-{version}", campaign_id="C", generation=generation,
            operation="revise", parent_snapshot_id=parent_snapshot,
            parent_skill_version=f"S01@{parent}",
            candidate_skill_version=f"S01@{version}",
            canonical_question_type="object_counting", full_skill_spec=child,
            structured_diff=[{"field": "description", "from": "x", "to": "y"}])

    snap1, _ = publish_candidate_snapshot(store, _candidate("1.1.0", "1.0.0", "S0", 1))
    snap2, _ = publish_candidate_snapshot(
        store, _candidate("1.2.0", "1.1.0", snap1["snapshot_id"], 2))
    snap3, _ = publish_candidate_snapshot(
        store, _candidate("1.3.0", "1.2.0", snap2["snapshot_id"], 3))
    assert snap3["active_skill_versions"] == ["S01@1.2.0", "S01@1.3.0"]
    assert snap3["historical_skill_versions"] == ["S01@1.0.0", "S01@1.1.0"]
    assert set(snap3["entries"]) == {"S01@1.0.0", "S01@1.1.0", "S01@1.2.0", "S01@1.3.0"}
    skills, warnings, _sid = load_active_skills(store)
    assert [f"{s.skill_id}@{s.version}" for s in skills] == ["S01@1.2.0", "S01@1.3.0"]
    assert len(warnings) == 2                       # 历史版本保留但被明确跳过


# ---------------------------------------------------------------- §8.4 两阶段选择 ----

class _SelectionClient:
    """只回答"版本选择"请求的假模型（记录收到的请求，不触网）。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list = []

    def chat(self, messages, max_tokens: int = 4096):
        self.calls.append(messages)
        return self.reply


def _selection_fixture():
    """构造一个"同谱系两个版本都过硬条件"的检索记录（确定性选中 1.0.0）。"""
    from skill3d.online.runner import _select_version_with_model
    from skill3d.schemas import SkillCandidateRecord, SkillRetrievalRecord

    v1, v2 = _counting_skill("1.0.0"), _counting_skill("1.1.0", desc="候选 1.1.0 的方法描述")
    record = SkillRetrievalRecord(
        canonical_question_type="object_counting", question_type_raw="object_counting",
        n_skills_offered=2,
        candidates=[
            SkillCandidateRecord(skill_id="S01", version="1.0.0",
                                 skill_version="S01@1.0.0", hard_filter_passed=True,
                                 selected=True, reason_code="hit", score=0.9, rank=1,
                                 content_sha256="a", delivery_reason="no_model_request"),
            SkillCandidateRecord(skill_id="S01", version="1.1.0",
                                 skill_version="S01@1.1.0", hard_filter_passed=True,
                                 selected=False, reason_code="lineage_version_not_selected",
                                 score=0.8, rank=2, content_sha256="b"),
        ],
        eligible_skill_versions=["S01@1.0.0", "S01@1.1.0"],
        retrieved_skill_versions=["S01@1.0.0"])
    record.record_lineage_selection(
        "S01", "S01@1.0.0",
        [{"skill_version": "S01@1.0.0", "score": 0.9, "rank": 1, "hard_filter_passed": True},
         {"skill_version": "S01@1.1.0", "score": 0.8, "rank": 2, "hard_filter_passed": True}],
        reason="higher_ranking_score")
    episode = type("E", (), {"question": "数一下场景里有几把椅子"})()
    outcome = type("O", (), {"m7_notes": []})()
    cfg = OnlineRunConfig(mode="real", skills=[v1, v2], vllm_endpoints=["http://fake"])
    return _select_version_with_model, record, episode, outcome, cfg


def test_two_stage_selection_takes_the_model_choice():
    """§8.4：模型按摘要在谱系内选版本；选择阶段与执行阶段分别记请求 hash。"""
    select, record, episode, outcome, cfg = _selection_fixture()
    client = _SelectionClient('{"selected_skill_version": "S01@1.1.0"}')
    select(episode, record, cfg, outcome, client)

    assert client.calls, "选择请求必须真的发出去"
    assert record.selection_request_sha256 and record.selection_response_sha256
    assert record.retrieved_skill_versions == ["S01@1.1.0"]     # 记录跟着选择走
    assert record.lineage_selections[0]["selected_version"] == "S01@1.1.0"
    assert record.lineage_selections[0]["reason"] == "model_selected_from_summaries"
    rows = {r.skill_version: r for r in record.candidates}
    assert rows["S01@1.1.0"].selected is True and rows["S01@1.0.0"].selected is False
    # 摘要进了选择请求，完整正文**没有**进（§8.4 只有第二阶段才放完整正文）
    prompt = client.calls[0][0]["content"]
    assert "S01@1.0.0" in prompt and "S01@1.1.0" in prompt
    assert render_skill_entry(_counting_skill("1.0.0")) not in prompt


def test_two_stage_selection_falls_back_honestly_on_garbage():
    """§8.4：模型给不出合法版本时确定性回落，并**如实**记录这次失败。"""
    select, record, episode, outcome, cfg = _selection_fixture()
    client = _SelectionClient("我看不出区别，随便选一个吧")
    select(episode, record, cfg, outcome, client)

    assert record.retrieved_skill_versions == ["S01@1.0.0"]      # 确定性结果保留
    assert record.lineage_selections[0]["reason"] == "deterministic_fallback"
    assert record.selection_request_sha256 and record.selection_response_sha256
    assert any("确定性回落" in n for n in outcome.m7_notes)
