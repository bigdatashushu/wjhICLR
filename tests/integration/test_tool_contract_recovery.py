"""D-3 恢复阶梯的端到端测试（在线链 runner，§5.1 / §4 M6 字段 9）。

覆盖：
- route 降级后 tool_contract 命中 → 回灌一次（多轮 messages）→ 恢复执行；
- 回灌未修复 → 裁剪 prompt（强制 fallback_2d_only）重生成一次；
- 两次机会用尽 → 显式 abstain，答案**不得采纳**，主榜按错计；
- 未启用回灌层时只走裁剪轮（[Conditional Go] 默认关闭）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.online.runner import OnlineRunConfig, run_episode
from skill3d.schemas import (
    ConfidenceMap,
    QualityMetrics,
    ReconstructionArtifact,
)


class _FakeClient:
    """假 vLLM 客户端：按调用次数依次返回预设 program（记录收到的 messages）。"""

    def __init__(self, programs: list[str]) -> None:
        self.programs = list(programs)
        self.calls: list[list[dict]] = []

    def chat(self, messages, max_tokens: int = 512, **kwargs) -> str:
        self.calls.append(messages)
        idx = min(len(self.calls) - 1, len(self.programs) - 1)
        return self.programs[idx]


def _artifact_json(tmp_path) -> str:
    """写一个 quality 已算、route=full_3d 的 artifact，但**不落位姿/内参数组**。

    这样 `reproject` 需要的 poses/intrinsics 产物在句柄上不可用 →
    Tool 执行期 fail-closed 触发 tool_contract（而 exists_in_scene 仍可用）。
    """
    q = QualityMetrics(g1_blur_ok=1.0, g2_brightness=1.0, g3_motion_blur=0.5,
                       g4_frame_count=32, g5_reproj_err_median=float("nan"),
                       g5_reproj_err_p95=float("nan"), g6_depth_var_coeff=0.5,
                       g7_dynamic_ratio=0.0, g9_tracker_consistency=0.95, g10_baseline_quality=0.3,
                       g11_scale_ci=0.05, overall_quality=0.9)
    art = ReconstructionArtifact(
        artifact_id="frozen", artifact_version="v", scene_name="synth-scene",
        recon_method="vggt", frame_ids=list(range(32)),
        source_frame_indices=list(range(32)),
        timestamps=[float(i) for i in range(32)], frame_set_hash="hash-32",
        c2w_list="", intrinsics="", depth_maps="", point_map="", point_conf="",
        track_list=None, metric_scale=1.0, scale_known=True,
        quality_status="computed", quality=q,
        confidence=ConfidenceMap(per_point_confidence="", coverage_count_per_frame=""),
        scale_confidence="medium", scale_source="camera_height_prior",
    )
    p = tmp_path / "art.json"
    p.write_text(art.model_dump_json(indent=2), encoding="utf-8")
    return str(p)


BAD = ('c = reproject([0.0, 0.0, 1.0], 0)\n'
       'ReturnAnswer(c)\n')
GOOD = ('ReturnAnswer(2)\n')


@pytest.fixture(scope="module")
def episode_item():
    return load_synthetic_items("inner_validation", question_types=["object_counting"],
                                frame_size=(120, 160), seed=0)[0]


def test_replay_recovers_after_tool_contract(tmp_path, episode_item):
    """D-3：命中 tool_contract → 回灌一次（多轮 messages）→ 修复后正常作答。"""
    client = _FakeClient([BAD, GOOD])
    cfg = OnlineRunConfig(mode="real", reuse_artifact=_artifact_json(tmp_path),
                          vllm_endpoints=["http://fake"], deterministic_replay=True,
                          trace_dir=str(tmp_path / "t"), memory_dir="",
                          enable_tool_contract_replay=True, max_images=32)
    out = run_episode(episode_item.episode, episode_item.pixels, cfg, llm=client)

    assert out.tool_contract_hits == 1 and out.replay_used
    assert out.abstained is False and out.final_state == "answer"
    assert out.answer == "2"
    # 回灌的第二次调用必须带 assistant(上一轮 program) + user(裁剪 traceback)
    second = client.calls[1]
    roles = [m["role"] for m in second]
    assert roles[-2:] == ["assistant", "user"]
    assert "产物缺失" in second[-1]["content"] or "硬约束 23" in second[-1]["content"]
    assert any("tool_contract" in n for n in out.notes)


def test_trimmed_regeneration_then_abstain_when_replay_disabled(tmp_path, episode_item):
    """未启用回灌层 → 直接走裁剪 prompt 轮；仍失败 → abstain 且不采纳答案。"""
    client = _FakeClient([BAD, BAD, BAD])
    cfg = OnlineRunConfig(mode="real", reuse_artifact=_artifact_json(tmp_path),
                          vllm_endpoints=["http://fake"], deterministic_replay=True,
                          trace_dir=str(tmp_path / "t"), memory_dir="",
                          enable_tool_contract_replay=False)
    out = run_episode(episode_item.episode, episode_item.pixels, cfg, llm=client)

    assert out.trimmed_regen_used and not out.replay_used
    assert out.abstained is True and out.final_state == "unanswerable"
    assert out.answer is None and out.answer_untrusted
    assert out.mra_value == pytest.approx(0.0)            # 主榜按错计
    assert "tool_contract" in out.answer_flags
    assert out.episode_trace.failure.categories == ["tool_contract"]
    # 裁剪轮的 prompt 必须显式声明受限 route（只能编排该 route 下可用的 Tool）
    trimmed = client.calls[1][0]["content"]
    text = trimmed if isinstance(trimmed, str) else json.dumps(trimmed)
    assert "fallback_2d_only" in text


def test_replay_then_trim_then_abstain_full_ladder(tmp_path, episode_item):
    """完整阶梯：回灌一次 → 裁剪一次 → 仍失败 → abstain（整个 episode 最多回灌 1 次）。"""
    client = _FakeClient([BAD, BAD, BAD, BAD])
    cfg = OnlineRunConfig(mode="real", reuse_artifact=_artifact_json(tmp_path),
                          vllm_endpoints=["http://fake"], deterministic_replay=True,
                          trace_dir=str(tmp_path / "t"), memory_dir="",
                          enable_tool_contract_replay=True)
    out = run_episode(episode_item.episode, episode_item.pixels, cfg, llm=client)

    assert out.replay_used and out.trimmed_regen_used
    assert out.tool_contract_hits >= 2
    assert out.abstained and out.final_state == "unanswerable" and out.answer is None
    assert len(client.calls) <= 3          # 不无限重试：回灌 1 + 裁剪 1


def test_answer_from_program_touching_missing_artifact_is_never_scored(tmp_path, episode_item):
    """答案依赖过契约失败的 Tool → 即使 program 先 ReturnAnswer 也不采纳（D-3）。"""
    # 先 ReturnAnswer 再调用缺失产物的 Tool：program 里答案已写入槽位，
    # 但 tool_contract 一旦命中必须把答案作废
    sneaky = 'ReturnAnswer(7)\nc = reproject([0.0, 0.0, 1.0], 0)\n'
    client = _FakeClient([sneaky, sneaky, sneaky, sneaky])
    cfg = OnlineRunConfig(mode="real", reuse_artifact=_artifact_json(tmp_path),
                          vllm_endpoints=["http://fake"], deterministic_replay=True,
                          trace_dir=str(tmp_path / "t"), memory_dir="",
                          enable_tool_contract_replay=True)
    out = run_episode(episode_item.episode, episode_item.pixels, cfg, llm=client)
    assert out.abstained and out.final_state == "unanswerable"
    assert out.answer is None and out.mra_value == pytest.approx(0.0)
