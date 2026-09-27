"""P6 端到端：§13.5/§13.6 的"检索选中 ≠ 已交付"、交付正文身份与证据更新后重检索。

规范原文（§13.6）：

    「每次检索记录规范题型、evidence_version、候选及过滤原因、排序分数、选中版本、
    实际交付版本与正文 hash、配置版本。区分"检索选中但未送达模型"与"已交付"。」

规范原文（§13.5）：

    「上下文放不下时先减少完整条目，不能截掉检查、局部条件或来源后仍称
    "完整 Skill 已交付"。」
    「证据更新后可在同一快照中重检索，更新实际交付记录；不在 episode 中发布新库。」

本文件走**真实 runner 路径**（不是替身）：mock_light 全链、real 模式的 `_synthesize`
与 M10 恢复层的级联降级。真实模型与真实 vLLM 服务不在本文件覆盖范围内。
"""

from __future__ import annotations

import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.online.runner import OnlineRunConfig, _synthesize, run_episode
from skill3d.routing.retrieval_policy import RetrievalPolicy
from skill3d.schemas import SkillSpec
from skill3d.skills.delivery import render_skill_entry, skill_content_sha256

FRAME_SIZE = (120, 160)

# 交付层的真实路径需要"合法 v6 artifact"（real 模式复用产物），复用恢复层测试的
# 夹具构造器与假客户端 —— 它们与真实链路同一套加载/路由/执行，不另造一套替身。
from test_tool_contract_recovery import (  # noqa: E402 - 同目录测试夹具
    _FakeClient,
    _write_v6_artifact,
)


def _prompt_text(messages) -> str:
    """取多模态 messages 里的文本段（交付正文的包含判断必须看**原始文本**）。"""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out.extend(p.get("text", "") for p in content
                       if isinstance(p, dict) and p.get("type") == "text")
    return "\n".join(out)


def _room_skill(desc="房间面积：先用 plane_fit_room_size() 取平面，再报告面积（平方米）。",
                template='area = plane_fit_room_size()\nReturnAnswer(str(area["room_area_m2"]))',
                signature=None, skill_id="sk-room") -> SkillSpec:
    return SkillSpec(
        skill_id=skill_id, version="1.0.0",
        applicable_question_types=["room_size_estimation"],
        required_evidence_signature=dict(signature or {}),
        requires_metric_evidence=False, applicable_gate_version=None,
        skill_family="metric", source="real", description=desc,
        call_graph_template=template)


@pytest.fixture(scope="module")
def room_item():
    return load_synthetic_items("inner_validation",
                                question_types=["room_size_estimation"],
                                frame_size=FRAME_SIZE, seed=0)[0]


# ------------------------------------------------ "检索选中" ≠ "已交付" ----

def test_mock_light_retrieval_is_recorded_but_nothing_is_delivered(room_item, tmp_path):
    """§13.6：mock_light 不生成模型请求 → retrieved 可非空，delivered 必须为空。

    这正是规范要区分的那件事：方法被**检索选中**了，但一条都没**送达模型**。
    此前 trace 里只有 `selected_skill_semvers`，看起来像"模型见过这些方法"。
    """
    skill = _room_skill()
    cfg = OnlineRunConfig(mode="mock_light", skills=[skill], memory_dir="",
                          trace_dir=str(tmp_path / "t"),
                          active_snapshot_ref="S0-seed-20260925-v1")
    out = run_episode(room_item.episode, room_item.pixels, cfg, geometry=room_item.geometry)

    assert out.final_state == "answer" and out.synthesis_source == "mock_stub"
    assert out.selected_skill_semvers == ["sk-room@1.0.0"]      # 检索选中
    assert out.retrieved_skill_versions == ["sk-room@1.0.0"]
    assert out.delivered_skill_versions == []                   # 未交付（无模型请求）
    assert len(out.retrieval_records) == 1
    rec = out.retrieval_records[0]
    assert rec.trigger == "initial" and rec.retrieval_index == 1
    assert rec.canonical_question_type == "room_size_estimation"
    assert rec.config_source == "default"                       # 未从 yaml 传策略时如实标注
    assert rec.delivery_channel == "not_sent"
    assert rec.candidates[0].selected is True
    assert rec.candidates[0].delivered is False
    assert rec.candidates[0].delivery_reason == "no_model_request"
    # 落盘的 trace 与内存口径一致（§13.6 的记录必须真的进 trace）
    assert out.episode_trace.retrieval_records == [rec.model_dump()]
    assert out.episode_trace.delivered_skill_versions == []
    assert out.episode_trace.retrieved_skill_versions == ["sk-room@1.0.0"]


def test_real_synthesis_delivers_the_exact_method_body(room_item, tmp_path):
    """§13.6：交付版本必须带**正文 hash**，且那段正文真的在发出的请求里。"""
    skill = _room_skill()
    body = render_skill_entry(skill)
    program = ('# 参考 sk-room@1.0.0\n'
               'area = plane_fit_room_size()\n'
               'ReturnAnswer(str(round(area["room_area_m2"], 2)))\n')
    client = _FakeClient([program])
    cfg = OnlineRunConfig(mode="real", vllm_endpoints=["http://fake"],
                          skills=[skill], seed=0)
    res = _synthesize(room_item.episode, scene=None, handle=None, skills=[skill],
                      cfg=cfg, llm=client, pixels=list(room_item.pixels))

    assert res.program is not None, res.note
    assert res.delivery is not None
    assert res.delivery.channel == "model_request"
    assert res.delivery.delivered_skill_versions == ["sk-room@1.0.0"]
    # 交付身份 = 完整正文的 sha256（与 prompt 里的文本同源）
    assert res.delivery.delivered_content_sha256 == {
        "sk-room@1.0.0": skill_content_sha256(skill)}
    sent = _prompt_text(client.calls[0])
    assert body in sent                     # 正文逐字进了发出的请求
    # 程序对象携带的是"实际交付集"（v8 上下文口径），不是"检索选中集"
    assert res.program.skill_semver_used == ["sk-room@1.0.0"]


def test_context_cap_drops_the_whole_method_and_never_truncates(room_item, tmp_path):
    """§13.5：放不下就**整条不放**，不得截断正文后仍记"已交付"。"""
    skill = _room_skill()
    huge = _room_skill(desc="很长的方法描述。" * 200, skill_id="sk-huge")
    cap = len(render_skill_entry(skill)) + 20      # 只放得下第一条
    policy = RetrievalPolicy(top_k=2, method_context_max_chars=cap)
    client = _FakeClient(["ReturnAnswer(1)"])
    cfg = OnlineRunConfig(mode="real", vllm_endpoints=["http://fake"],
                          skills=[skill, huge], seed=0, retrieval_policy=policy)
    res = _synthesize(room_item.episode, scene=None, handle=None,
                      skills=[skill, huge], cfg=cfg, llm=client,
                      pixels=list(room_item.pixels))

    assert res.program is not None, res.note
    assert res.delivery.delivered_skill_versions == ["sk-room@1.0.0"]
    assert [d.skill_version for d in res.delivery.dropped] == ["sk-huge@1.0.0"]
    assert res.delivery.dropped[0].reason == "context_cap_exceeded"
    sent = _prompt_text(client.calls[0])
    assert render_skill_entry(skill) in sent            # 完整条目在
    # 被丢弃的那条：一个字都不该出现在 prompt 里（截断也是违规）
    assert "很长的方法描述" not in sent
    assert res.program.skill_semver_used == ["sk-room@1.0.0"]


def test_retrieval_record_reaches_the_trace_with_candidates_and_reasons(room_item, tmp_path):
    """§13.6：被拦下的候选 + 原因 + 排序分数一起落盘（不再只写日志）。"""
    eligible = _room_skill(skill_id="sk-room")
    # 一条米制题型的 Skill：本题是 room_size_estimation 分区，它必被题型分区拦下
    other = _room_skill(skill_id="sk-count")
    other = other.model_copy(update={
        "applicable_question_types": ["object_counting"], "skill_family": "counting"})
    cfg = OnlineRunConfig(mode="mock_light", skills=[eligible, other], memory_dir="",
                          trace_dir=str(tmp_path / "t"))
    out = run_episode(room_item.episode, room_item.pixels, cfg, geometry=room_item.geometry)

    rec = out.retrieval_records[0]
    rows = {r.skill_version: r for r in rec.candidates}
    assert rows["sk-room@1.0.0"].reason_code == "hit"
    assert rows["sk-room@1.0.0"].score is not None and rows["sk-room@1.0.0"].rank == 1
    assert rows["sk-count@1.0.0"].reason_code == "question_type_mismatch"
    assert "题型不匹配" in rows["sk-count@1.0.0"].reason
    assert rec.n_skills_offered == 2
    assert any("检索#" in n for n in out.notes)
    assert any("sk-count@1.0.0 被拦下" in n and "question_type_mismatch" in n
               for n in out.notes)


# --------------------------------- 证据更新后在同一快照内重检索（§13.5）----

def test_evidence_update_triggers_a_second_retrieval_in_the_same_snapshot(room_item,
                                                                         tmp_path):
    """§13.5：级联降级（证据更新）后重检索，且**不**重新加载 Skill 库。

    证据能力 `geometry_3d: available → degraded` 后，声明
    `required_evidence_signature={"geometry_3d": "available"}` 的方法不再可检索：
    这正是"重检索必须发生"的可观测后果。
    """
    skill = _room_skill(signature={"geometry_3d": "available"})
    # 共享前提失效（缺 poses）→ 级联撤销 + 能力降级（同 §14.1）
    # 模型自称使用了 sk-room@1.0.0（写在注释里）→ §13.6 的"declared"线索
    program = ('# 按 sk-room@1.0.0 的步骤\n'
               'area = plane_fit_room_size()\n'
               'uv = reproject([0.0, 0.0, 1.0], 0)\n'
               'ReturnAnswer(str(round(area["room_area_m2"], 2)))\n')
    cfg = OnlineRunConfig(mode="real", reuse_artifact=_write_v6_artifact(tmp_path,
                                                                        with_poses=False),
                          vllm_endpoints=["http://fake"], deterministic_replay=True,
                          trace_dir=str(tmp_path / "t"), memory_dir="",
                          skills=[skill], max_recovery=1,
                          active_snapshot_ref="S0-seed-20260925-v1")
    out = run_episode(room_item.episode, room_item.pixels, cfg,
                      llm=_FakeClient([program, 'ReturnAnswer("1")\n']))

    assert len(out.retrieval_records) == 2, [n for n in out.notes if "检索#" in n]
    first, second = out.retrieval_records
    assert first.trigger == "initial" and first.retrieval_index == 1
    assert second.trigger == "evidence_update" and second.retrieval_index == 2
    # 同一快照：两次检索指向同一 active snapshot，没有中途发布新库
    assert second.active_snapshot_ref == first.active_snapshot_ref == "S0-seed-20260925-v1"
    assert second.config_sha256 == first.config_sha256 == cfg.retrieval_policy.sha256()
    # 证据确实变了（profile_version 是合同版本，不变；三值快照才看得出变化）
    assert first.evidence_states["geometry_3d"] == "available"
    assert second.evidence_states["geometry_3d"] == "degraded"
    # 重检索的后果：这条方法在降级后的证据下不再可检索
    assert first.eligible_skill_versions == ["sk-room@1.0.0"]
    assert second.eligible_skill_versions == []
    assert second.candidates[0].reason_code == "evidence_signature_unmet"
    assert second.retrieved_skill_versions == []
    # 第一次检索的实际交付记录仍留在第一条记录里（不是被覆盖）
    assert first.delivered_skill_versions == ["sk-room@1.0.0"]
    assert first.candidates[0].delivered is True
    # §13.6：模型自称（程序里点名）+ 可观察线索（模板点名的 Tool 与程序字面交叠）
    assert first.declared_selected_skill_versions == ["sk-room@1.0.0"]
    assert out.declared_selected_skill_versions == ["sk-room@1.0.0"]
    clue = first.usage_clues[0]
    assert clue["skill_version"] == "sk-room@1.0.0"
    assert clue["declared_in_program"] is True
    assert "plane_fit_room_size" in clue["template_tool_overlap"]
    assert "不构成" in clue["note"]

