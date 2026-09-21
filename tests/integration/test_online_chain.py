"""端到端集成测试：M1→M13 在线链（§12 M0/M1 验收口径）。

覆盖：
- 8 题型端到端跑通（mock_light 管道验证，非精度结论）；
- 同 seed 字节级一致（硬约束 11/§4 M8/M17 验收）；
- M2 被动观测（blur_all/blur_some 都**不删帧**，只打 flag/降权，硬约束 21）；
  blur_all 降权后 route 落 fallback_2d_only → 需要 objects 的 Tool 触发
  fail-closed（ArtifactUnavailableError）→ 显式 abstain 且主榜按错计（D-3）；
- 硬约束 9：final_test 默认拒绝；
- 硬约束 1：real 模式缺 vLLM → 记 unavailable（不伪造答案）；
- C0 baseline 不经沙箱（无 program）。
"""

from __future__ import annotations

import json

import pytest

from skill3d.adapters.episode_source import ALL_QUESTION_TYPES, load_synthetic_items
from skill3d.online.runner import OnlineRunConfig, run_episode, run_split
from skill3d.trace.store import TraceStore

# 小尺寸合成帧：保持与 VSI-Bench 同构（32 帧），测试跑得快
FRAME_SIZE = (120, 160)
FULL_CHAIN = ["INGEST", "INPUT_GATE", "RECONSTRUCT", "QUALITY_GATE", "CLASSIFY_TASK",
              "RETRIEVE_SKILL", "SYNTHESIZE_PROGRAM", "STATIC_CHECK", "SANDBOX_EXECUTE",
              "GEOMETRY_VERIFY", "BENCHMARK_EVAL", "ANSWER", "LOG_TRACE"]


@pytest.fixture(scope="module")
def items():
    return load_synthetic_items("inner_validation", seed=0, frame_size=FRAME_SIZE)


def test_all_question_types_run_end_to_end(items, tmp_path):
    """8 题型全部跑完在线链并产出答案（M1 端到端可运行）。"""
    cfg = OnlineRunConfig(mode="mock_light", trace_dir=str(tmp_path / "traces"))
    outcomes, run = run_split(items, cfg)
    assert run.n_episodes == len(ALL_QUESTION_TYPES) == 8
    for o in outcomes:
        assert o.final_state == "answer", f"{o.question_type}: {o.final_state} {o.notes}"
        assert o.answer not in (None, "")
        assert o.states == FULL_CHAIN, f"{o.question_type} 状态链不完整: {o.states}"
        assert o.receipts_ok, "receipt 哈希链校验失败"
        assert o.episode_trace is not None and o.episode_trace.failure is None
    # MCA 走 Accuracy、NA 走 MRA（§5.7 EvaluationRun）
    assert run.accuracy is not None and run.mra is not None


def test_program_actually_calls_tools(items):
    """C1：program 经 M9 AST 后进沙箱，Tool 被真实调用且结果进 trace（§5.4）。"""
    it = next(i for i in items if i.episode.question_type == "room_size_estimation")
    cfg = OnlineRunConfig(mode="mock_light")
    out = run_episode(it.episode, it.pixels, cfg, geometry=it.geometry)
    assert out.synthesis_source == "deterministic_stub"
    trace = out.program_trace
    assert trace is not None and trace.steps >= 1
    assert [r.tool for r in trace.results] == ["room_size_m2"]
    assert trace.results[0].error is None
    # mock_light 下 source 必须显式标 mock（绝不冒充 real，§9.2）
    assert trace.results[0].source == "mock_light"
    assert out.verify is not None and out.verify.passed


def test_same_seed_is_byte_identical(items, tmp_path):
    """同 seed 重放字节级一致（§4 M8 字段 11 / §4 M17 字段 11）。"""
    def dump(tag: str) -> str:
        cfg = OnlineRunConfig(mode="mock_light", deterministic_replay=True, seed=7,
                              trace_dir=str(tmp_path / tag))
        outcomes, run = run_split(items, cfg)
        return json.dumps({
            "episodes": [[o.episode_trace.model_dump(), o.program_trace.model_dump(),
                          [r.model_dump() for r in o.receipts]] for o in outcomes],
            "run": run.model_dump(),
        }, sort_keys=True, ensure_ascii=False)

    assert dump("a") == dump("b")


def test_input_gate_blur_all_downgrades_then_abstains_on_tool_contract(tmp_path):
    """M2 全帧模糊 → 只降权 + route 降级 → Tool fail-closed → abstain（D-3）。

    v3 语义（硬约束 21/23）：M2 不再删帧；质量权重拉低 route 到 fallback_2d_only，
    此时需要 objects 的 Tool 必须抛 ArtifactUnavailableError（不静默返回 False），
    恢复层用尽 → 显式 abstain，且主榜按错计（MRA=0）。
    """
    items = load_synthetic_items("inner_validation", question_types=["room_size_estimation"],
                                 frame_size=FRAME_SIZE, degrade="blur_all",
                                 out_dir=str(tmp_path / "obj"))
    n_before = len(items[0].pixels)
    cfg = OnlineRunConfig(mode="mock_light", trace_dir=str(tmp_path / "t"))
    out = run_episode(items[0].episode, items[0].pixels, cfg, geometry=items[0].geometry)

    # 帧集不变（M2 被动观测）
    assert n_before == 32 and len(items[0].pixels) == 32
    assert len(items[0].episode.frames) == 32
    assert any("M2 被动观测" in n and "不" not in n[:2] for n in out.notes)
    # route 降级 + 契约 fail-closed + 显式 abstain
    assert out.scene_route == "fallback_2d_only"
    assert out.abstained and out.tool_contract_hits >= 1
    assert "tool_contract" in out.answer_flags
    assert out.episode_trace.failure.categories == ["tool_contract"]
    assert out.final_state == "unanswerable" and out.answer is None
    assert out.mra_value == pytest.approx(0.0)          # 主榜按错计，不刷分


def test_input_gate_blur_some_keeps_all_frames_and_answers(tmp_path):
    """M2 局部低质 → 打 flag + 降权后继续；帧数/帧序不变（硬约束 21）。"""
    items = load_synthetic_items("inner_validation", question_types=["room_size_estimation"],
                                 frame_size=FRAME_SIZE, degrade="blur_some",
                                 out_dir=str(tmp_path / "obj"))
    cfg = OnlineRunConfig(mode="mock_light", trace_dir=str(tmp_path / "t"))
    out = run_episode(items[0].episode, items[0].pixels, cfg, geometry=items[0].geometry)
    assert len(items[0].pixels) == 32                   # 没删帧
    assert out.final_state == "answer"
    assert any("被动观测" in n for n in out.notes)
    assert not any("屏蔽" in n for n in out.notes)      # 旧的 drop_and_refill 已废弃
    assert out.episode_trace.failure is None


def test_final_test_refused_by_default(items):
    """硬约束 9：final_test 默认不进在线链，需显式 allow_final_test。"""
    it = load_synthetic_items("final_test", question_types=["room_size_estimation"],
                              frame_size=FRAME_SIZE)[0]
    out = run_episode(it.episode, it.pixels, OnlineRunConfig(mode="mock_light"),
                      geometry=it.geometry)
    assert out.final_state == "unanswerable"
    assert "final_test" in " ".join(out.notes)

    out2 = run_episode(it.episode, it.pixels,
                       OnlineRunConfig(mode="mock_light", allow_final_test=True),
                       geometry=it.geometry)
    assert out2.final_state == "answer"


def _test_double_artifact(path) -> str:
    """测试替身：一个合法 ReconstructionArtifact（数组 ref 不存在，走 NaN 兜底）。

    仅用于驱动 runner 的 real 代码路径；不是系统产出，不得进入任何实验记录。
    """
    import math

    from skill3d.schemas import ConfidenceMap, QualityMetrics, ReconstructionArtifact

    nan = float("nan")
    art = ReconstructionArtifact(
        artifact_id="test-double", artifact_version="test-double",
        scene_name="test-double-scene", recon_method="vggt",
        c2w_list="", intrinsics="", depth_maps="", point_map="", point_conf="",
        track_list=None, metric_scale=1.0, scale_known=True,
        quality=QualityMetrics(
            g1_blur_ok=nan, g2_brightness=nan, g3_motion_blur=nan, g4_frame_count=32,
            g5_reproj_err_median=nan, g5_reproj_err_p95=nan, g6_depth_var_coeff=nan,
            g7_dynamic_ratio=nan, g9_tracker_consistency=nan,
            g10_baseline_quality=nan, g11_scale_ci=nan, overall_quality=nan),
        confidence=ConfidenceMap(per_point_confidence="", coverage_count_per_frame=""),
    )
    p = path / "test_double_artifact.json"
    p.write_text(art.model_dump_json(), encoding="utf-8")
    return str(p)


def test_real_mode_without_vllm_is_unavailable_not_fake(items, tmp_path):
    """硬约束 1/诚实性：real 模式缺 vLLM → 记 unavailable，绝不伪造答案。

    用测试替身 artifact 走 reuse_artifact 路径（等价于已有真实重建产物的场景）。
    """
    it = items[0]
    cfg = OnlineRunConfig(mode="real", vllm_endpoints=[],
                          reuse_artifact=_test_double_artifact(tmp_path))
    out = run_episode(it.episode, it.pixels, cfg, geometry=it.geometry)
    assert out.states == FULL_CHAIN[:-1] + ["LOG_TRACE"] or "SYNTHESIZE_PROGRAM" in out.states
    assert out.final_state == "unavailable"
    assert out.answer is None
    assert out.synthesis_source == "none"
    assert out.episode_trace.failure.categories == ["evaluator_noanswer"]
    assert any("复用既有 artifact" in n for n in out.notes)


def test_c0_baseline_skips_sandbox(items):
    """§16.1 C0：direct VLM 基线无 program → 不经沙箱执行。"""
    it = next(i for i in items if i.episode.question_type == "object_counting")
    cfg = OnlineRunConfig(mode="mock_light", baseline="C0_direct_vlm")
    out = run_episode(it.episode, it.pixels, cfg, geometry=it.geometry)
    assert out.final_state == "answer"
    assert out.program is not None and out.program.program_source == ""
    assert out.program_trace.steps == 0 and out.program_trace.results == []
    assert "SANDBOX_EXECUTE" in out.states


def test_trace_store_records_all_topics(items, tmp_path):
    """M13：episode_trace / program_trace / geometry_check / evaluation_result 落盘。"""
    store = TraceStore(tmp_path / "traces")
    cfg = OnlineRunConfig(mode="mock_light")
    it = next(i for i in items if i.episode.question_type == "room_size_estimation")
    run_episode(it.episode, it.pixels, cfg, geometry=it.geometry, trace_store=store)
    for topic in ("episode_trace", "program_trace", "geometry_check", "evaluation_result"):
        p = tmp_path / "traces" / f"{topic}.jsonl"
        assert p.is_file() and p.read_text(encoding="utf-8").strip(), f"缺 topic {topic}"
    # 在线只写不读：trace 内容不得回灌进在线链（由构造保证：runner 不读 store）
    rec = json.loads((tmp_path / "traces" / "episode_trace.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    assert rec["qa_id"] == it.episode.qa_id
