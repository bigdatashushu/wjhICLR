"""v6 §14 `partial_tool_recovery` 的端到端测试（在线链 runner，M10 块）。

覆盖（逐条对齐 §14.1/§6.4，**替代 v5 的"回灌→裁剪"两档阶梯**）：

- (a) **局部失败不级联**：域值错误（DomainValueError）不撤销既有成功结果，
  已成功结果作为 validated observations 回灌进重生成的 prompt；
- (b) **共享前提失效级联撤销**：缺 `poses`/`depth` 这类共享前提 → 撤销依赖它的既有结果、
  记录 `invalidated_result_ids`、把 EvidenceProfile 对应能力降级、并重新派生
  `question_tool_scope`；
- (c) **恢复次数有限**：由 `cfg.max_retries_per_operation` 约束，之后进入终结轮；
  不再重入恢复环，执行失败记 `run_error`（主榜按错计）；
- (d) **契约污染过的答案永不采纳**：program 自捕获 ToolContractError 后仍 ReturnAnswer
  → 答案作废，`answer_untrusted=True`，不得进分。

数据源：合成几何写成**合法 v6 artifact**（数组真落盘、质量用真门算、
逐帧尺度 receipt 真写），走 `reuse_artifact` 路径 —— 与真实链路同一套加载/路由/执行。
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
from skill3d.schemas import (
    ConfidenceMap,
    ReconstructionArtifact,
)

FRAME_SIZE = (120, 160)


class _FakeClient:
    """假 vLLM 客户端：**只对 M8（program 合成）调用**按序返回预设 program。

    真实链路里 M5 的对象清单阶段会复用同一个客户端（`_vlm_inventory_pass`）：
    这里对它返回空串，让 M5 走"绑定不可用"的降级分支（与缺 SAM2 checkpoint 的
    真实情形一致），从而 `calls` 只记录 M8 的调用次数（恢复次数断言才有意义）。
    """

    def __init__(self, programs: list[str]) -> None:
        self.programs = list(programs)
        self.calls: list[list[dict]] = []

    def chat(self, messages, max_tokens: int = 512, **kwargs) -> str:
        blob = json.dumps(messages, ensure_ascii=False)
        if "ReturnAnswer" not in blob:        # 非 M8 调用（M5 清单等）
            return ""
        self.calls.append(messages)
        idx = min(len(self.calls) - 1, len(self.programs) - 1)
        return self.programs[idx]


def _write_v6_artifact(tmp_path, *, with_poses: bool) -> str:
    """把合成几何写成**合法 v6 artifact**（quality 真算、receipt 真写）。

    `with_poses=False` 时 `c2w_list` 留空 → 运行时句柄没有 poses/intrinsics →
    调用依赖与位姿的 Tool 会抛 `ArtifactUnavailableError(missing=['poses'])`
    —— 正是 §14.1 的"共享前提失效"（geometry_3d）。
    """
    se = syn.make_synthetic_episode("room_size_estimation", scene_name="tc-scene",
                                    qa_id="tc-0", frame_size=FRAME_SIZE)
    g = se.geometry
    d = tmp_path / "recon"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "depth.npy", g.depth_maps)
    np.save(d / "pm.npy", g.point_map)
    np.save(d / "k.npy", g.intrinsics)
    np.save(d / "conf.npy", g.depth_conf)
    if with_poses:
        np.save(d / "c2w.npy", g.c2w)
    q = compute_quality(None, frames=se.frames, depth_maps=g.depth_maps,
                        c2w_list=g.c2w, intrinsics=g.intrinsics,
                        point_map=g.point_map, depth_conf=g.depth_conf)
    assert q.main_gate_passed, q.diagnostic_warnings
    fusion = syn._synthetic_metric_fusion(g)
    receipt = write_per_frame_receipt(fusion, d / "per_frame_scale.json")
    art = ReconstructionArtifact(
        artifact_id="tc-artifact", artifact_version="v1", scene_name="tc-scene",
        frame_ids=list(range(32)), source_frame_indices=list(range(32)),
        timestamps=[i / 30.0 for i in range(32)],
        frame_set_hash=se.episode.frame_set.frame_set_hash,
        c2w_list=str(d / "c2w.npy") if with_poses else "",
        intrinsics=str(d / "k.npy"),
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
    return str(p)


def _blob(messages) -> str:
    """把一轮 messages 拍成字符串（M8 prompt 是"文本 + 32 帧图像"的多模态结构）。"""
    return json.dumps(messages, ensure_ascii=False)


@pytest.fixture(scope="module")
def episode_item():
    return load_synthetic_items("inner_validation", question_types=["room_size_estimation"],
                                frame_size=FRAME_SIZE, seed=0)[0]


def _cfg(tmp_path, art_path: str, *, max_recovery: int) -> OnlineRunConfig:
    return OnlineRunConfig(mode="real", reuse_artifact=art_path,
                           vllm_endpoints=["http://fake"], deterministic_replay=True,
                           trace_dir=str(tmp_path / "t"), memory_dir="",
                           max_retries_per_operation=max_recovery, max_images=32)


# ------------------------------------------------------- (a) 局部失败不级联 ----

LOCAL_FAIL_THEN_OK = [
    # 第一次：成功的 plane_fit_room_size + 一个域值错误（局部：参数错，不涉及共享前提）
    ('area = plane_fit_room_size()\n'
     'bad = euclidean_distance([0.0, 0.0], 1.0)\n'
     'ReturnAnswer(str(round(area["room_area_m2"], 2)))\n'),
    # 第二次（重生成后）：只用仍然可信的既有结果作答
    ('area = plane_fit_room_size()\n'
     'ReturnAnswer(str(round(area["room_area_m2"], 2)))\n'),
]


def test_local_failure_keeps_validated_observations(tmp_path, episode_item):
    """domain_value（局部）不撤销既有结果，且 validated observations 被回灌。"""
    client = _FakeClient(LOCAL_FAIL_THEN_OK)
    cfg = _cfg(tmp_path, _write_v6_artifact(tmp_path, with_poses=True), max_recovery=2)
    out = run_episode(episode_item.episode, episode_item.pixels, cfg, llm=client)

    assert out.tool_contract_hits == 1                    # 命中一次契约失败
    assert out.partial_tool_recovery is True
    assert out.recovery_count == 1
    # (a) 局部失败**不**级联撤销：没有任何 result 被标 invalidated
    assert out.invalidated_result_ids == []
    assert not any("级联撤销" in n for n in out.notes)
    # 恢复成功 → 程序路径作答
    assert out.final_state == "answer" and out.answer_source == "tool_program"
    assert out.answer not in (None, "")

    # 回灌内容：失败信息 + validated observations 摘要（含 result_id）都在 prompt 里
    assert len(client.calls) == 2
    text = _blob(client.calls[1])
    assert "局部错误" in text
    assert "仍然可信的既有观测" in text and "result_id=" in text
    # 局部失败**不得**出现"已级联撤销"这类断言
    assert "级联撤销" not in text
    # 恢复后答案关联到本次用的 result（§14.1：trace 记 used_result_ids）
    assert out.used_result_ids


# ------------------------------------------------- (b) 共享前提失效级联撤销 ----

PREMISE_FAIL_THEN_OK = [
    # 第一次：先成功拿到房间面积，再调用依赖位姿的 reproject（poses 缺失 → 共享前提失效）
    ('area = plane_fit_room_size()\n'
     'uv = reproject([0.0, 0.0, 1.0], 0)\n'
     'ReturnAnswer(str(round(area["room_area_m2"], 2)))\n'),
    # 第二次：只用一个不依赖被撤销证据的 Tool 作答（不碰已失效的数值）
    ('ReturnAnswer(str(round(euclidean_distance([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 2)))\n'),
]


def test_shared_premise_failure_cascades_and_downgrades(tmp_path, episode_item):
    """缺 poses → 撤销依赖 geometry_3d 的既有结果 + 降级证据能力 + 重派生 scope。"""
    client = _FakeClient(PREMISE_FAIL_THEN_OK)
    cfg = _cfg(tmp_path, _write_v6_artifact(tmp_path, with_poses=False), max_recovery=2)
    out = run_episode(episode_item.episode, episode_item.pixels, cfg, llm=client)

    # 契约失败（ArtifactUnavailableError: poses/intrinsics 缺失）
    assert out.tool_contract_hits == 1 and out.partial_tool_recovery is True
    # 级联撤销与能力降级写入结构化轮次记录。
    failed_round = next(r for r in out.rounds if r.get("error_code"))
    assert failed_round["downgraded_capabilities"]["geometry_3d"] == (
        "available→degraded")
    assert failed_round["invalidated_result_ids"]
    assert len(out.invalidated_result_ids) >= 1
    # EvidenceProfile 能力降级（available → degraded），几何/世界系/米制一起降
    assert out.evidence_profile is not None
    assert out.evidence_profile.state("geometry_3d") == "degraded"
    assert out.evidence_profile.state("metric_scale") == "degraded"
    # 降级后的证据画像必须真的把米制 Tool 收回去（§7.2 单项失败只收回依赖它的工具）
    from skill3d.tools import REGISTRY

    docs = REGISTRY.docs(out.question_tool_scope, evidence_profile=out.evidence_profile)
    assert "plane_fit_room_size" not in docs
    # 回灌文本明确写着"不要再用被撤销的数值"，且此时已无任何可信既有观测
    text = _blob(client.calls[1])
    assert "级联撤销" in text and "被撤销的数值" in text
    assert "当前没有任何可信的既有观测" in text
    # 恢复后仍能作答（用未被污染的工具），但答案绝不复用被撤销的房间面积
    assert out.final_state == "answer" and out.answer_source == "tool_program"
    assert out.answer not in ("17.11",)                   # 被撤销的旧数值不得复用


# ------------------------------------------------------- (c) 恢复次数有限 ----

ALWAYS_FAIL = ('uv = reproject([0.0, 0.0, 1.0], 0)\n'
               'ReturnAnswer("0.0")\n')


@pytest.mark.parametrize("max_recovery", [0, 1, 2])
def test_recovery_attempts_bounded_then_zero_tool_answer(tmp_path, episode_item, max_recovery):
    """恢复次数有界；用尽后只调用一次零工具终结程序。"""
    direct = 'ReturnAnswer("3.0")\n'
    client = _FakeClient([ALWAYS_FAIL] * (max_recovery + 1) + [direct])
    cfg = _cfg(tmp_path, _write_v6_artifact(tmp_path, with_poses=False),
               max_recovery=max_recovery)
    out = run_episode(episode_item.episode, episode_item.pixels, cfg, llm=client)

    assert len(client.calls) == max_recovery + 2
    assert out.recovery_count == max_recovery + 1
    assert out.final_state == "answer" and out.answer == "3.0"
    assert out.answer_source == "tool_program"
    assert "finalization" in out.answer_flags
    assert out.mra_value is not None
    assert out.episode_trace.failure is None
    assert out.program_trace.calls == []


def test_caught_forced_tool_call_is_not_retried(tmp_path, episode_item):
    """finalization 通过 ctx 别名调用工具也会被运行时禁止，且不会重试。"""
    forced_with_caught_tool_call = (
        "def solve(ctx):\n"
        "    try:\n"
        "        ctx.tools.reproject([0.0, 0.0, 1.0], 0)\n"
        "    except Exception:\n"
        "        pass\n"
        "    return ReturnAnswer('3.0')\n"
    )
    client = _FakeClient([ALWAYS_FAIL, forced_with_caught_tool_call])
    cfg = _cfg(tmp_path, _write_v6_artifact(tmp_path, with_poses=False), max_recovery=0)
    out = run_episode(episode_item.episode, episode_item.pixels, cfg, llm=client)

    assert len(client.calls) == 2
    assert out.final_state == "run_error"
    assert out.answer is None
    assert out.mra_value == pytest.approx(0.0)
    assert out.episode_trace.failure.categories == ["run_error"]


# --------------------------------------------- (d) 契约污染过的答案不采纳 ----

SNEAKY_CATCH = (
    'answer = "0.0"\n'
    'try:\n'
    '    uv = reproject([0.0, 0.0, 1.0], 0)\n'
    'except Exception:\n'
    '    pass\n'
    'ReturnAnswer(answer)\n'
)


def test_answer_after_caught_contract_error_is_preserved(tmp_path, episode_item):
    """自捕获工具异常后的预测可保留，但必须记录契约违规并保守归因。"""
    client = _FakeClient([SNEAKY_CATCH])
    cfg = _cfg(tmp_path, _write_v6_artifact(tmp_path, with_poses=False), max_recovery=0)
    out = run_episode(episode_item.episode, episode_item.pixels, cfg, llm=client)

    assert out.answer_untrusted is True
    assert out.answer == "0.0"
    assert out.answer_source == "tool_program"
    assert out.final_state == "answer"
    assert out.mra_value is not None
    assert "tool_contract_observed" in out.answer_flags
    assert out.tool_contract_hits == 0


def test_validated_observation_is_not_replayed_after_invalidation(tmp_path,
                                                                  episode_item):
    """级联撤销后，被撤销的 result 不得再出现在回灌的 validated observations 里。"""
    client = _FakeClient(PREMISE_FAIL_THEN_OK)
    cfg = _cfg(tmp_path, _write_v6_artifact(tmp_path, with_poses=False), max_recovery=2)
    out = run_episode(episode_item.episode, episode_item.pixels, cfg, llm=client)

    text = _blob(client.calls[1])
    # 撤销列表里的 result_id 必须出现在"已撤销"提示中，而不得作为可信观测回灌
    for rid in out.invalidated_result_ids:
        assert f"[result_id={rid}]" not in text
