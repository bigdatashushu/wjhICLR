"""v9 P1：多轮执行的账本可追溯性与轮次事实落盘（§11.1/§11.2/§12/§17.1）。

守两类此前的失真：

1. **跨轮账本被抹掉**：`reset_user_namespace()` 曾连 `tool_results` 一起清空，而
   runner 在 yield 路径上"清空后再塞回去"的写法会被下一次 `_execute_program` 的
   reset 立刻抹掉。后果是 `collect_validated` / `used_result_ids` / 级联撤销都只看
   得到最后一轮，前几轮的观测**无法追溯**——v9 §11.1 要求点名结果可回灌、§17.1
   要求每条 Tool 结果可追溯，两者都失效。
2. **轮次事实只在内存里**：`agent_rounds` / `yield_count` / `round_trace_refs` 不曾
   落盘，恢复轮还不计入求解轮预算（§11.2 禁止"在每轮之外无限追加恢复请求"）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.online import synthetic as syn
from skill3d.online.runner import OnlineRunConfig, run_episode
from skill3d.reconstruction.metric_fusion import write_per_frame_receipt
from skill3d.reconstruction_gate.quality_metrics import compute_quality
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas import ConfidenceMap, ObjectInstance, ReconstructionArtifact, SceneState
from skill3d.schemas.evidence import (
    GATE_SUBCONDITIONS,
    GATE_VERSION,
    EvidenceProfile,
    MetricEvidenceGateResult,
)
from skill3d.schemas.reconstruction import METRIC_TASK_TYPES
from skill3d.schemas.trace import KNOWN_SYNTHESIS_SOURCES
from skill3d.tools import REGISTRY
from skill3d.tools.contract import available_artifacts_for, question_tool_scope_of
from skill3d.tools.scene_handle import SceneHandle

FRAME_SIZE = (120, 160)


class _ScriptedClient:
    """按序返回预设 program 的假客户端；非 M8 调用返回空串（M5 走降级分支）。"""

    def __init__(self, programs: list[str]) -> None:
        self.programs = list(programs)
        self.calls: list[list[dict]] = []

    def chat(self, messages, max_tokens: int = 512, **kwargs) -> str:
        blob = json.dumps(messages, ensure_ascii=False)
        if "ReturnAnswer" not in blob and "YieldObservations" not in blob:
            return ""
        self.calls.append(messages)
        idx = min(len(self.calls) - 1, len(self.programs) - 1)
        return self.programs[idx]


# --------------------------------------------------- 内核层：账本跨轮存活 ----

def _kernel() -> RestrictedNamespaceKernel:
    gate = MetricEvidenceGateResult(gate_passed=True, gate_version=GATE_VERSION,
                                    sub_results={n: True for n in GATE_SUBCONDITIONS})
    profile = EvidenceProfile(geometry_3d="available", world_frame="available",
                              metric_scale="available", object_detection="available",
                              track_consensus="available", object_grounding="available")
    qtype = "room_size_estimation"
    scene = SceneState(
        artifact_ref="test", scene_route="full_3d",
        question_tool_scope=question_tool_scope_of("full_3d", metric_question=True,
                                                   gate_passed=True),
        available_artifacts=set(available_artifacts_for(
            "full_3d", question_type=qtype, gate_passed=True,
            metric_question=qtype in METRIC_TASK_TYPES)),
        evidence_profile=profile, metric_evidence_gate_result=gate,
        objects=["obj_0"], summary="s", question_type=qtype)
    handle = SceneHandle(
        scene,
        objects=[ObjectInstance(
            obj_id="obj_0", category_name="table", mask_per_frame="",
            pointcloud_world="", centroid_world=[0.0, 0.0, 1.0],
            bbox=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], det_conf=0.9)],
        c2w_list=np.tile(np.eye(4), (2, 1, 1)),
        intrinsics=np.tile(np.eye(3), (2, 1, 1)))
    return RestrictedNamespaceKernel(REGISTRY, handle, frames=[], mode="real")


def test_ledger_survives_namespace_reset_while_cell_view_is_per_round():
    """命名空间清空 ≠ 账本清空：episode 账本保留，逐轮视图只报本轮。"""
    kernel = _kernel()
    kernel.run_cell('a = object_centroid("obj_0")\n')
    first = [r.result_id for r in kernel.cell_results]
    assert len(first) == 1

    kernel.reset_user_namespace()          # 回灌重执行前的清理
    assert [r.result_id for r in kernel.tool_results] == first, \
        "episode 账本不得被命名空间重置抹掉"

    kernel.run_cell('b = object_centroid("obj_0")\n')
    assert len(kernel.cell_results) == 1, "逐轮视图只应含本轮结果"
    assert len(kernel.tool_results) == 2, "账本应累计两轮结果"
    # result_id 是**内容寻址**的（请求 + 场景状态摘要），因此同一场景里重复同一次
    # 测量得到同一个 id —— 这是"两次测的是同一件事"的正确表达，不是碰撞。
    assert kernel.cell_results[0].result_id == first[0]

    # 换一个请求则必须是不同的 id
    kernel.run_cell('c = object_3d_extent("obj_0")\n')
    assert kernel.cell_results[0].result_id != first[0]
    assert len(kernel.tool_results) == 3


def test_ledger_reset_is_explicit_and_available():
    """确实要复用同一 kernel 开新 episode 时，清账本必须是显式动作。"""
    kernel = _kernel()
    kernel.run_cell('a = object_centroid("obj_0")\n')
    kernel.reset_episode_ledger()
    assert kernel.tool_results == [] and kernel.cell_results == []


# ------------------------------------------- 端到端：连续两次 yield ----

def _programs() -> list[str]:
    return [
        'r1 = plane_fit_room_size()\n'
        'return YieldObservations([r1["result_id"]], "第一次让出：需要看尺寸")\n',
        'r2 = plane_fit_room_size()\n'
        'return YieldObservations([r2["result_id"]], "第二次让出：再看一次")\n',
        'return ReturnAnswer("20")\n',
    ]


def _real_cfg(tmp_path, art_path: str, **over) -> OnlineRunConfig:
    """注入客户端的 episode 必须用 mode="real"（mock_light 会忽略注入的 client），
    而 real 模式要求真实重建产物 → 走 `reuse_artifact`（与恢复测试同一路径）。"""
    return OnlineRunConfig(mode="real", reuse_artifact=art_path,
                           vllm_endpoints=["http://fake"], deterministic_replay=True,
                           trace_dir=str(tmp_path / "t"), memory_dir="", **over)


def _write_v6_artifact(tmp_path) -> str:
    """把一个合法 v6 artifact 真落盘（quality 真算、逐帧尺度 receipt 真写）。"""
    se = syn.make_synthetic_episode("room_size_estimation", scene_name="rl-scene",
                                    qa_id="rl-0", frame_size=FRAME_SIZE)
    g = se.geometry
    d = tmp_path / "recon"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "depth.npy", g.depth_maps)
    np.save(d / "pm.npy", g.point_map)
    np.save(d / "k.npy", g.intrinsics)
    np.save(d / "conf.npy", g.depth_conf)
    np.save(d / "c2w.npy", g.c2w)
    q = compute_quality(None, frames=se.frames, depth_maps=g.depth_maps,
                        c2w_list=g.c2w, intrinsics=g.intrinsics,
                        point_map=g.point_map, depth_conf=g.depth_conf)
    assert q.main_gate_passed, q.diagnostic_warnings
    fusion = syn._synthetic_metric_fusion(g)
    receipt = write_per_frame_receipt(fusion, d / "per_frame_scale.json")
    art = ReconstructionArtifact(
        artifact_id="rl-artifact", artifact_version="v1", scene_name="rl-scene",
        frame_ids=list(range(32)), source_frame_indices=list(range(32)),
        timestamps=[i / 30.0 for i in range(32)],
        frame_set_hash=se.episode.frame_set.frame_set_hash,
        c2w_list=str(d / "c2w.npy"), intrinsics=str(d / "k.npy"),
        depth_maps=str(d / "depth.npy"), point_map=str(d / "pm.npy"), point_conf="",
        depth_conf=str(d / "conf.npy"),
        world_up=g.world_up, handedness=g.handedness,
        world_frame_status=g.world_frame_status,
        metric_scale=(float(fusion.metric_scale)
                      if fusion.metric_scale is not None else 1.0),
        scale_self_consistency=fusion.scale_self_consistency,
        per_frame_scale_ref=str(receipt), metric_model="none",
        metric_fusion_version=fusion.version, scale_fusion_status="success",
        quality_status="computed", quality=q,
        confidence=ConfidenceMap(per_point_confidence="", coverage_count_per_frame=""),
    )
    p = tmp_path / "artifact.json"
    p.write_text(art.model_dump_json(indent=2), encoding="utf-8")
    return str(p), se


def test_two_consecutive_yields_stay_traceable_and_rounds_are_persisted(tmp_path):
    """连续两次 yield：轮次事实落盘，且**两轮**的观测都可在最终答案归因里追溯。"""
    art_path, se = _write_v6_artifact(tmp_path)
    client = _ScriptedClient(_programs())
    cfg = _real_cfg(tmp_path, art_path)
    out = run_episode(se.episode, se.frames, cfg, llm=client)
    trace = out.episode_trace
    assert trace is not None

    # ---- 轮次事实落盘（§17.1「检索与 Round」层）----
    assert out.yield_count == 2, out.notes
    assert trace.yield_count == 2
    assert trace.agent_rounds == out.agent_rounds >= 3, (trace.agent_rounds, out.notes)
    triggers = [r["trigger"] for r in trace.rounds]
    assert triggers[:3] == ["initial", "observation", "observation"], triggers
    assert [r["index"] for r in trace.rounds] == sorted(r["index"] for r in trace.rounds)
    assert trace.round_trace_refs == [f"round:{r['index']}" for r in trace.rounds]
    assert trace.budget["max_solver_rounds"] >= 1
    assert trace.finalization_used is False

    # 每轮真实执行的程序文本与哈希都要留档（只留最后一轮是不够的）
    for rec in trace.rounds:
        assert rec["program_source"] and rec["program_sha256"]
        assert rec["tool_calls"] == len(rec["observed_result_ids"])

    # ---- 跨轮账本可追溯（§11.1/§14.1）----
    round_ids = [rid for rec in trace.rounds for rid in rec["observed_result_ids"]]
    assert len(round_ids) == 2, (round_ids, [r["program_source"] for r in trace.rounds])
    for rid in round_ids:
        assert rid in trace.used_result_ids, \
            f"{rid} 未进入最终归因账本：早先轮次的观测不可追溯"

    # 第二次 yield 的回灌必须真的带上了第一轮的观测内容（不是只有"成功"标记）
    second_call = json.dumps(client.calls[2], ensure_ascii=False)
    assert round_ids[0] in second_call, "第二轮请求未拿到第一轮的观测 payload"


def test_forced_answer_round_is_recorded_with_finalize_trigger(tmp_path):
    """收口轮（while 之外）也不能漏记：轮次记录与 finalization 事实一致。"""
    art_path, se = _write_v6_artifact(tmp_path)
    programs = _programs()
    # 预算只够一轮 + 收口：首轮 yield 即触及边界 → 进 finalization
    client = _ScriptedClient(programs[:1] + programs[2:])
    cfg = _real_cfg(tmp_path, art_path, max_solver_rounds=1, finalization_rounds=0)
    out = run_episode(se.episode, se.frames, cfg, llm=client)
    trace = out.episode_trace
    assert trace.finalization_used is True, out.notes
    assert trace.round_trigger == "finalize"
    assert any(r["trigger"] == "finalize" for r in trace.rounds), trace.rounds
    # 收口轮的程序来源照实记：必须是**已知来源**，且不得把轮次原因
    # （finalize / forced_answer）写成来源 —— 这正是 v8 造成的失真。
    assert trace.synthesis_source in KNOWN_SYNTHESIS_SOURCES, trace.synthesis_source
    assert trace.synthesis_source not in ("finalize", "forced_answer",
                                          "forced_answer_final")
    for rec in trace.rounds:
        assert rec["synthesis_source"] in KNOWN_SYNTHESIS_SOURCES, rec["synthesis_source"]


def test_recovery_rounds_consume_solver_budget(tmp_path):
    """恢复轮必须计入求解轮预算（§11.2 禁止在每轮之外无限追加恢复请求）。"""
    items = load_synthetic_items("inner_validation",
                                 question_types=["room_size_estimation"],
                                 frame_size=FRAME_SIZE, degrade="blur_all",
                                 out_dir=str(tmp_path / "obj"))
    # v9 §5.1：诊断默认关闭 → 路由不再被 M2 权重拉低，也就不会触发契约恢复。
    # 本用例要验证"恢复轮计入轮预算"，故显式开启诊断（独立诊断实验口径）。
    cfg = OnlineRunConfig(mode="mock_light", trace_dir=str(tmp_path / "t"),
                          memory_dir="", input_diagnostics=True)
    out = run_episode(items[0].episode, items[0].pixels, cfg, geometry=items[0].geometry)
    trace = out.episode_trace
    assert out.tool_contract_hits >= 1
    # 至少发生过一次恢复，且它被记进了轮数
    assert trace.agent_rounds >= 2, (trace.agent_rounds, out.notes)
    assert len(trace.rounds) == trace.agent_rounds, (len(trace.rounds), trace.agent_rounds)
