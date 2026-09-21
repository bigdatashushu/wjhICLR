"""端到端集成测试：M1→M13 在线链（§12 M0/M1 验收口径）。

覆盖（v6 语义）：

- 8 题型端到端跑通（mock_light 管道验证，**M4 主门在合成数据上真算**）；
- `synthesis_source` 的 v6 取值（mock 路径 = `mock_stub`；§19.3 六类 + mock 单列）；
- `scene_route`（只由 M4 质量决定）× `question_tool_scope`（逐题派生收窄）解耦（D4）；
- `answer_source` 四值词汇（`tool_program` / `direct_vlm_routed` / `abstain` /
  `tool_contract`，§5.3/§6.3）；
- 同 seed 字节级一致（硬约束 11/§4 M8/M17 验收）；
- M2 被动观测（blur_all/blur_some 都**不删帧**，只打 flag/降权，硬约束 21）：
  blur_all 降权后 route 落 fallback_2d_only → 需要 3D 产物的 Tool 触发
  fail-closed（ArtifactUnavailableError）→ partial_tool_recovery 用尽 → 显式 abstain
  且主榜按错计；
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

# v6 §5.3：answer_source 四值（不再有 v5 的 program / direct_vlm）
ANSWER_SOURCES_V6 = {"tool_program", "direct_vlm_routed", "abstain", "tool_contract"}


@pytest.fixture(scope="module")
def items():
    return load_synthetic_items("inner_validation", seed=0, frame_size=FRAME_SIZE)


def test_all_question_types_run_end_to_end(items, tmp_path):
    """8 题型全部跑完在线链并产出答案（M1 端到端可运行）。"""
    cfg = OnlineRunConfig(mode="mock_light", trace_dir=str(tmp_path / "traces"),
                          memory_dir=str(tmp_path / "mem"))
    outcomes, run = run_split(items, cfg)
    assert run.n_episodes == len(ALL_QUESTION_TYPES) == 8
    for o in outcomes:
        assert o.final_state == "answer", f"{o.question_type}: {o.final_state} {o.notes}"
        assert o.answer not in (None, "")
        assert o.states == FULL_CHAIN, f"{o.question_type} 状态链不完整: {o.states}"
        assert o.receipts_ok, "receipt 哈希链校验失败"
        assert o.episode_trace is not None and o.episode_trace.failure is None
        # v6 §19.3：mock_light 的程序来源显式标 mock_stub（绝不冒充 vllm_ok）
        assert o.synthesis_source == "mock_stub", o.question_type
        # v6 §5.3：answer_source 只用四值词汇
        assert o.answer_source in ANSWER_SOURCES_V6, o.question_type
        # 程序路径作答 → tool_program（direct_vlm_routed 只属于直答/C0）
        assert o.answer_source == "tool_program", o.question_type
    # MCA 走 Accuracy、NA 走 MRA（§5.7 EvaluationRun）
    assert run.accuracy is not None and run.mra is not None


def test_synthetic_scene_passes_real_m4_main_gate(items):
    """mock_light 的 scene_route 必须由**真算**的 M4 主门给出（不许绕过门）。"""
    for it in items:
        cfg = OnlineRunConfig(mode="mock_light", memory_dir="")
        out = run_episode(it.episode, it.pixels, cfg, geometry=it.geometry)
        assert out.main_gate_passed is True, it.episode.question_type
        assert out.scene_route == "full_3d", it.episode.question_type
        # 三项证据门都过 → 米制题 scope 收窄到 metric_enabled（§5.3/D4）
        if it.episode.question_type in ("object_abs_distance", "object_size_estimation",
                                        "room_size_estimation"):
            assert out.question_tool_scope == "metric_enabled"
        else:
            assert out.question_tool_scope == "full_3d"
        # 逐题 scope ⊆ scene_route（只收窄不新增）
        assert out.episode_trace.scene_route == "full_3d"


def test_program_actually_calls_tools(items):
    """C1：program 经 M9 AST 后进沙箱，Tool 被真实调用且结果进 trace（§5.4）。"""
    it = next(i for i in items if i.episode.question_type == "room_size_estimation")
    cfg = OnlineRunConfig(mode="mock_light", memory_dir="")
    out = run_episode(it.episode, it.pixels, cfg, geometry=it.geometry)
    assert out.synthesis_source == "mock_stub"
    trace = out.program_trace
    assert trace is not None and trace.steps >= 1
    assert [r.tool for r in trace.results] == ["plane_fit_room_size"]
    assert trace.results[0].error is None
    # mock_light 下 source 必须显式标 mock（绝不冒充 real，§9.2）
    res = trace.results[0]
    assert res.source == "mock_light"
    # §5.7：每个结果自带 result_id / status / evidence_version（partial recovery 的撤销依据）
    assert res.result_id and res.status == "ok" and res.evidence_version
    assert res.invalidated_by is None
    assert out.verify is not None and out.verify.passed


def test_same_seed_is_byte_identical(items, tmp_path):
    """同 seed 重放字节级一致（§4 M8 字段 11 / §4 M17 字段 11）。

    注意：**任何** episode 的 program_trace 都可能是 None（M2 硬失败 / M8 生成失败
    这类"没走到 M10"的 episode 没有程序轨迹）—— 序列化必须显式处理，不能想当然。
    本用例同时断言 mock 路径 8 题**都**走到了 M10（否则说明管道半路断了）。
    """
    def dump(tag: str) -> str:
        cfg = OnlineRunConfig(mode="mock_light", deterministic_replay=True, seed=7,
                              trace_dir=str(tmp_path / tag),
                              memory_dir=str(tmp_path / f"mem-{tag}"))
        outcomes, run = run_split(items, cfg)
        assert all(o.program_trace is not None for o in outcomes), \
            "mock_light 8 题都应执行过 program（program_trace 不得为 None）"
        return json.dumps({
            "episodes": [[o.episode_trace.model_dump(),
                          (None if o.program_trace is None else o.program_trace.model_dump()),
                          [(None if r is None else r.model_dump()) for r in o.receipts]]
                         for o in outcomes],
            "run": run.model_dump(),
        }, sort_keys=True, ensure_ascii=False)

    assert dump("a") == dump("b")


def test_input_gate_blur_all_downgrades_then_abstains_on_tool_contract(tmp_path):
    """M2 全帧模糊 → 只降权 + route 降级 → Tool fail-closed → abstain（§6.4/§14）。

    v6 语义（硬约束 21）：M2 不再删帧；质量权重拉低 route 到 fallback_2d_only，
    此时需要 3D 产物的 Tool 必须抛 ArtifactUnavailableError（不静默返回 False），
    partial_tool_recovery 用尽 → 显式 abstain（`answer_source="tool_contract"`），
    且主榜按错计（MRA=0，不刷分）。
    """
    items = load_synthetic_items("inner_validation", question_types=["room_size_estimation"],
                                 frame_size=FRAME_SIZE, degrade="blur_all",
                                 out_dir=str(tmp_path / "obj"))
    n_before = len(items[0].pixels)
    cfg = OnlineRunConfig(mode="mock_light", trace_dir=str(tmp_path / "t"),
                          memory_dir="")
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
    assert out.answer_source == "tool_contract"          # §6.3：契约失败终止单列
    assert out.mra_value == pytest.approx(0.0)          # 主榜按错计，不刷分


def test_input_gate_blur_some_keeps_all_frames_and_answers(tmp_path):
    """M2 局部低质 → 打 flag + 降权后继续；帧数/帧序不变（硬约束 21）。"""
    items = load_synthetic_items("inner_validation", question_types=["room_size_estimation"],
                                 frame_size=FRAME_SIZE, degrade="blur_some",
                                 out_dir=str(tmp_path / "obj"))
    cfg = OnlineRunConfig(mode="mock_light", trace_dir=str(tmp_path / "t"),
                          memory_dir="")
    out = run_episode(items[0].episode, items[0].pixels, cfg, geometry=items[0].geometry)
    assert len(items[0].pixels) == 32                   # 没删帧
    assert out.final_state == "answer"
    assert any("被动观测" in n for n in out.notes)
    assert not any("屏蔽" in n for n in out.notes)      # 旧的 drop_and_refill 已废弃
    assert out.episode_trace.failure is None
    assert out.input_degradation_flags                   # 局部劣化被如实记录


def test_final_test_refused_by_default(items):
    """硬约束 9：final_test 默认不进在线链，需显式 allow_final_test。"""
    it = load_synthetic_items("final_test", question_types=["room_size_estimation"],
                              frame_size=FRAME_SIZE)[0]
    out = run_episode(it.episode, it.pixels,
                      OnlineRunConfig(mode="mock_light", memory_dir=""),
                      geometry=it.geometry)
    assert out.final_state == "unanswerable"
    assert "final_test" in " ".join(out.notes)

    out2 = run_episode(it.episode, it.pixels,
                       OnlineRunConfig(mode="mock_light", allow_final_test=True,
                                       memory_dir=""),
                       geometry=it.geometry)
    assert out2.final_state == "answer"


def _test_double_artifact(path) -> str:
    """测试替身：一个合法 **v6** ReconstructionArtifact（数组 ref 不存在 → 走 NaN 兜底）。

    v6 口径：`QualityMetrics` 不再有 G5/G11 字段、artifact 不再有 `scale_/scale_confidence`
    等校准路线字段（§20）；米制尺度由 `metric_scale` + `scale_fusion_status` 表达。
    仅用于驱动 runner 的 real 代码路径；不是系统产出，不得进入任何实验记录。
    """
    from skill3d.schemas import ConfidenceMap, QualityMetrics, ReconstructionArtifact

    nan = float("nan")
    art = ReconstructionArtifact(
        artifact_id="test-double", artifact_version="test-double",
        scene_name="test-double-scene", recon_method="vggt",
        frame_ids=list(range(32)), source_frame_indices=list(range(32)),
        timestamps=[float(i) for i in range(32)], frame_set_hash="test-double-hash",
        c2w_list="", intrinsics="", depth_maps="", point_map="", point_conf="",
        track_list=None,
        quality_status="computed",
        quality=QualityMetrics(
            warp_inlier_ratio=nan, warp_photometric_inlier_ratio=nan,
            cloud_overlap_ratio=nan, main_gate_passed=False,
            g1_blur_ok=nan, g2_brightness=nan, g3_motion_blur=nan, g4_frame_count=32,
            g6_depth_var_coeff=nan, g7_dynamic_ratio=nan, g9_tracker_consistency=nan,
            g10_baseline_quality=nan, overall_quality=nan),
        confidence=ConfidenceMap(per_point_confidence="", coverage_count_per_frame=""),
    )
    # 该替身必须**能被运行时加载**（v6 的唯一 artifact 入口）
    from skill3d.legacy.readers import load_artifact_v5

    p = path / "test_double_artifact.json"
    p.write_text(art.model_dump_json(), encoding="utf-8")
    assert load_artifact_v5(p).artifact_id == "test-double"
    return str(p)


def test_real_mode_without_vllm_is_unavailable_not_fake(items, tmp_path):
    """硬约束 1/诚实性：real 模式缺 vLLM → 记 unavailable，绝不伪造答案。

    用测试替身 artifact 走 reuse_artifact 路径（等价于已有真实重建产物的场景）。
    """
    it = items[0]
    cfg = OnlineRunConfig(mode="real", vllm_endpoints=[],
                          reuse_artifact=_test_double_artifact(tmp_path),
                          memory_dir="")
    out = run_episode(it.episode, it.pixels, cfg, geometry=it.geometry)
    assert out.states == FULL_CHAIN[:-1] + ["LOG_TRACE"] or "SYNTHESIZE_PROGRAM" in out.states
    assert out.final_state == "unavailable"
    assert out.answer is None
    assert out.synthesis_source == "none"
    # v6 归因：M8 拿不到模型输出 → `synthesis`（v5 的 evaluator_noanswer 已随口径更新）
    assert out.episode_trace.failure.categories == ["synthesis"]
    assert "unavailable" in out.episode_trace.failure.note
    assert any("复用既有 artifact" in n for n in out.notes)
    # 兜底 artifact 的数组 ref 全是空串 → 主门算不出 → route 必落 fallback（fail-closed）
    assert out.scene_route == "fallback_2d_only"


def test_c0_baseline_skips_sandbox(items):
    """§16.1 C0：direct VLM 基线无 program → 不经沙箱执行。"""
    it = next(i for i in items if i.episode.question_type == "object_counting")
    cfg = OnlineRunConfig(mode="mock_light", baseline="C0_direct_vlm", memory_dir="")
    out = run_episode(it.episode, it.pixels, cfg, geometry=it.geometry)
    assert out.final_state == "answer"
    assert out.program is not None and out.program.program_source == ""
    assert out.program_trace.steps == 0 and out.program_trace.results == []
    assert "SANDBOX_EXECUTE" in out.states
    # C0 的答案由生成阶段直接产出 → direct_vlm_routed（§6.3；不属于工具/程序路径贡献）
    assert out.answer_source == "direct_vlm_routed"


def test_trace_store_records_all_topics(items, tmp_path):
    """M13：episode_trace / program_trace / geometry_check / evaluation_result 落盘。"""
    store = TraceStore(tmp_path / "traces")
    cfg = OnlineRunConfig(mode="mock_light", memory_dir="")
    it = next(i for i in items if i.episode.question_type == "room_size_estimation")
    run_episode(it.episode, it.pixels, cfg, geometry=it.geometry, trace_store=store)
    for topic in ("episode_trace", "program_trace", "geometry_check", "evaluation_result",
                  "trace_record"):
        p = tmp_path / "traces" / f"{topic}.jsonl"
        assert p.is_file() and p.read_text(encoding="utf-8").strip(), f"缺 topic {topic}"
    # 在线只写不读：trace 内容不得回灌进在线链（由构造保证：runner 不读 store）
    rec = json.loads((tmp_path / "traces" / "episode_trace.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    assert rec["qa_id"] == it.episode.qa_id
    # §5.9：TraceRecord 必含版本字段与证据/路由全状态（不依赖重跑即可归因）
    trec = json.loads((tmp_path / "traces" / "trace_record.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    assert trec["scene_route"] == "full_3d"
    assert trec["question_tool_scope"] == "metric_enabled"
    assert trec["answer_source"] == "tool_program"
    assert trec["synthesis_source"] == "mock_stub"
    assert trec["evidence_profile"]["geometry_3d"] == "available"
    assert trec["metric_evidence_gate_result"]["gate_passed"] is True
    assert trec["template_version"] and trec["tool_face_version"]
    assert trec["gate_version"] and trec["distance_primitive_params"]
