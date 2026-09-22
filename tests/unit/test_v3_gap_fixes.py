"""v3 规格偏差修复的回归护栏（**v6 口径**：逐条对应《系统架构v6.md》条款）。

每条测试都钉住一个**曾经不满足规格**的具体行为，防止回退。v5 期间这些测试用的是
被 v6 废止的机制（`route` 一词三义、`scale_known` / `allowed_metric_tasks` 授权、
M5 统计→quality 的增量补写），相应条目已归档到 `tests/archive_v5/`
（见本文件末尾"归档"段与 `tests/archive_v5/README.md` 的 v6 替代物对照）。

| 测试 | 规格条款 |
|---|---|
| `test_program_swallowing_contract_error_is_not_scored` | §6.3/M10：答案依赖过契约失败的 Tool → 不得采纳 |
| `test_m8_never_silently_drops_frames` | M8：不得纯文本、不得静默丢帧 |
| `test_exists_in_scene_semantics_and_zero_detection_fail_closed` | §9.1/§7.1：False 当且仅当 objects 可用且真无；零检出 → fail-closed |
| `test_g1_uses_episode_relative_criterion` | 附录 A G1 双判据（B-3） |
| `test_g3_motion_blur_on_depth_grid` | §9 / C-8：光流必须在深度网格上算 |
| `test_quality_status_failed_when_no_data_source` | §6.2 硬约束 22：算不出质量不得标 `computed` |
| `test_frozen_artifact_is_never_rewritten_by_quality_gate` | 硬约束 18：paired A/B 的 frozen artifact 只读 |
| `test_scale_artifact_requires_metric_gate_and_metric_question` | §5.3/D3：`scale` 只在米制题 ∧ gate 通过时可用 |
| `test_prompt_header_reports_actually_loaded_artifacts` | §5.3/M8：prompt 头部"可用产物"不得高报 |
| `test_docs_trim_respects_actually_available_artifacts` | §7.2：`docs()` 按证据 + 实际装载产物裁剪 |
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.reconstruction_gate import quality_metrics as qm
from skill3d.reconstruction_gate.evidence_profile import M5EvidenceSummary
from skill3d.reconstruction_gate.scene_state import build_scene_state, quality_gate
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas import (
    ConfidenceMap,
    EvidenceProfile,
    ObjectRecord,
    QualityMetrics,
    ReconstructionArtifact,
    SceneState,
)
from skill3d.tools import REGISTRY
from skill3d.tools.contract import (
    ArtifactUnavailableError,
    available_artifacts_for,
)
from skill3d.tools.scene_handle import SceneHandle

# ------------------------------------------------------------------ 夹具 ----

def _quality(**over) -> QualityMetrics:
    """v6 QualityMetrics：主门字段必填；G5/G8/G11 已不存在（出现即 hard fail）。"""
    base = dict(warp_inlier_ratio=0.9, warp_photometric_inlier_ratio=0.9,
                cloud_overlap_ratio=0.9, main_gate_passed=True,
                g1_blur_ok=1.0, g2_brightness=1.0, g3_motion_blur=0.0,
                g4_frame_count=32, g6_depth_var_coeff=0.5, g7_dynamic_ratio=float("nan"),
                g9_tracker_consistency=float("nan"), g10_baseline_quality=0.3,
                overall_quality=0.9)
    base.update(over)
    return QualityMetrics(**base)


def _artifact(tmp_path) -> ReconstructionArtifact:
    """v6 artifact：世界系契约齐备、米制融合未跑（`metric_scale=None`）。

    位姿/内参落盘（reproject 可用），但 `depth_maps/point_map` 留空 —— 本文件
    关心的不是几何精度，而是**契约与路由**（fail-closed 语义）。
    """
    p = tmp_path / "a.npy"
    np.save(p, np.tile(np.eye(4), (32, 1, 1)))
    kn = tmp_path / "k.npy"
    np.save(kn, np.tile(np.eye(3), (32, 1, 1)))
    return ReconstructionArtifact(
        artifact_id="art-gap", artifact_version="v1", scene_name="scene-x",
        frame_ids=list(range(32)), source_frame_indices=list(range(32)),
        timestamps=[float(i) for i in range(32)], frame_set_hash="h32",
        c2w_list=str(p), intrinsics=str(kn), depth_maps="", point_map="",
        point_conf="", track_list=None,
        world_up=[0.0, 1.0, 0.0], handedness="right", world_frame_status="available",
        confidence=ConfidenceMap(per_point_confidence="", coverage_count_per_frame=""))


def _m5(n_objects: int = 3) -> M5EvidenceSummary:
    """M5 摘要：检测/跟踪都有正常统计（否则 object_detection 会是 unavailable）。"""
    return M5EvidenceSummary(n_objects=n_objects, n_tracks=n_objects,
                             track_stable_ratio=0.9)


def _scene_state(tmp_path, *, question_type="object_counting",
                 art=None, objects=("obj_0",)) -> SceneState:
    """M4 的 v6 生产路径：artifact → SceneState（scene_route × scope × 证据画像）。"""
    art = art or _artifact(tmp_path).model_copy(
        update={"quality_status": "computed", "quality": _quality()})
    return build_scene_state(art, m5=_m5(len(objects)), objects=list(objects),
                             artifact_ref="a", question_type=question_type)


def _obj(oid: str = "obj_0", category: str = "table") -> ObjectRecord:
    """v6 自描述对象记录（`ObjectInstance` 只是别名；构造用 `obj_id=` 等 v6 字段）。"""
    return ObjectRecord(obj_id=oid, category_name=category, mask_per_frame="",
                        pointcloud_world="", centroid_world=[0.0, 0.0, 0.0],
                        bbox=[0.0] * 6, det_conf=0.9)


def _fallback_scene() -> SceneState:
    """只装了 frames 的 2D-only 场景（objects/poses 都不可用）。"""
    return SceneState(
        artifact_ref="a", scene_route="fallback_2d_only",
        question_tool_scope="fallback_2d_only",
        available_artifacts=set(available_artifacts_for("fallback_2d_only")),
        evidence_profile=EvidenceProfile(
            geometry_3d="unavailable", world_frame="unavailable",
            metric_scale="unavailable", object_detection="unavailable",
            track_consensus="unavailable", object_grounding="unavailable"),
        objects=[], summary="s", question_type="object_counting")


# ------------------------------------------------ 硬约束 23：自捕获契约异常 ----

def test_program_swallowing_contract_error_is_not_scored():
    """程序 `try/except` 吞掉契约异常后再作答 → 按 tool_contract 归因，答案不采纳。

    规格（§6.3 M10 / 硬约束 23）："一旦 ReturnAnswer 依赖的某次 Tool 抛
    ToolContractError，最终答案不得采纳"。`error_code` 为空（异常被程序吞掉）
    不能成为绕过 fail-closed 的口子。
    """
    scene = _fallback_scene()          # objects/poses 均不可用
    handle = SceneHandle(scene, objects=[], c2w_list=None, intrinsics=None)
    kernel = RestrictedNamespaceKernel(REGISTRY, handle, frames=[], mode="real")
    cell = kernel.run_cell(
        'try:\n'
        '    n = exists_in_scene("table")\n'
        'except Exception:\n'
        '    n = 99          # 猜一个\n'
        'ReturnAnswer(n)\n')
    assert cell.answer == "99"                 # 程序确实写了答案
    assert cell.error_code is None             # 但异常被程序自己吞掉了
    assert cell.answer_untrusted is True       # 引擎把它标成不可信
    assert cell.contract_violations            # 契约违规有据可查
    assert cell.contract_violations[0]["error_code"] == "tool_contract"


# ------------------------------------------------ 硬约束 26：M8 多模态帧 ----

def test_m8_never_silently_drops_frames():
    """M8 图像通道：超上限 / 编码失败 / 零帧都必须报错，不得静默丢帧或退化纯文本。"""
    from skill3d.synthesis.prompt_builder import build_image_messages

    frames = [(np.random.default_rng(i).random((16, 16, 3)) * 255).astype(np.uint8)
              for i in range(4)]
    msgs = build_image_messages("p", frames, max_images=4)
    assert len(msgs[0]["content"]) == 5                      # 1 文本 + 4 图

    with pytest.raises(ValueError, match="禁止静默丢帧"):
        build_image_messages("p", frames, max_images=2)
    with pytest.raises(ValueError, match="退化为纯文本"):
        build_image_messages("p", [], max_images=4)

    bad = frames + [np.zeros((0, 0, 3), dtype=np.uint8)]      # 空帧 → 编码失败
    with pytest.raises(ValueError):
        build_image_messages("p", bad, max_images=8)


def test_m8_real_mode_requires_full_frame_set():
    """real 模式帧数必须等于统一 FrameSet（硬约束 21/26）。"""
    from skill3d.online.runner import OnlineRunConfig, _prompt_messages

    cfg = OnlineRunConfig(mode="real")
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]
    assert len(_prompt_messages("p", frames, cfg, expected_frames=4)[0]["content"]) == 5
    with pytest.raises(ValueError, match="禁止双帧集"):
        _prompt_messages("p", frames, cfg, expected_frames=32)
    with pytest.raises(ValueError, match="多模态"):
        _prompt_messages("p", [], cfg, expected_frames=4)


# ------------------------------------------- §9.1：objects 产物语义 ----

def test_exists_in_scene_semantics_and_zero_detection_fail_closed(tmp_path):
    """§9.1：False 当且仅当 objects 产物可用且真无此实例；产物未产出 → fail-closed 抛错。

    v6 比 v5 更严的一处：**零检出**（基础清单为空）在证据层与"检测器故障"不可区分
    → `object_detection=unavailable`（§7.1）→ Tool 被收回，**不得**返回 False
    假装"场景里没有"。这也是把 `exists_in_scene` 当布尔计数器用的最后一层刹车。
    """
    healthy = _scene_state(tmp_path, objects=("obj_0", "obj_1", "obj_2"))
    inventory = [_obj("obj_0", "table"), _obj("obj_1", "chair"), _obj("obj_2", "door")]
    c2w = np.tile(np.eye(4), (2, 1, 1))
    k = np.tile(np.eye(3), (2, 1, 1))

    # ① 检测健康 + 产物已产出：真无此实例 → False；有 → True
    materialized = SceneHandle(healthy, objects=inventory, c2w_list=c2w, intrinsics=k,
                               objects_materialized=True)
    assert "objects" in materialized.available_artifacts
    assert REGISTRY.call_tool("exists_in_scene", {"name": "table"}, materialized,
                              mode="real").value == "true"
    assert REGISTRY.call_tool("exists_in_scene", {"name": "spaceship"}, materialized,
                              mode="real").value == "false"

    # ② M5 未跑（objects 产物未产出）→ 必须 fail-closed 抛错（不得假装"场景里没有"）
    absent = SceneHandle(healthy, objects=[], c2w_list=c2w, intrinsics=k,
                         objects_materialized=False)
    assert "objects" not in absent.available_artifacts
    with pytest.raises(ArtifactUnavailableError):
        REGISTRY.call_tool("exists_in_scene", {"name": "table"}, absent, mode="real")

    # ③ 未显式声明时保持老语义（空列表 = 未产出）
    legacy = SceneHandle(healthy, objects=[])
    assert "objects" not in legacy.available_artifacts

    # ④ 零检出（M5 跑过但清单为空）→ 证据层 fail-closed，而不是"False"
    empty = _scene_state(tmp_path, objects=())
    assert empty.evidence_state("object_detection") == "unavailable"
    empty_handle = SceneHandle(empty, objects=[], c2w_list=c2w, intrinsics=k,
                               objects_materialized=True)
    with pytest.raises(ArtifactUnavailableError) as ei:
        REGISTRY.call_tool("exists_in_scene", {"name": "table"}, empty_handle, mode="real")
    # v6：missing 逐项带原因后缀（`object_detection(unavailable)`）
    assert len(ei.value.missing) == 1
    assert ei.value.missing[0].startswith("object_detection")


# ------------------------------------------------- Appendix A：G1 / G3 ----

def test_g1_uses_episode_relative_criterion():
    """G1 双判据：绝对下界 or episode 中位×0.35，取较严者（单靠绝对值会漏判）。"""
    import cv2

    from skill3d.gates.input_gate import TH_BLUR_VAR_ABS

    rng = np.random.default_rng(0)
    # 高纹理帧（Laplacian 方差远大于绝对下界 10）
    sharp = [(rng.random((64, 64, 3)) * 200 + 30).astype(np.uint8) for _ in range(9)]
    # 一张"相对本 episode 明显模糊"但绝对值仍 > 10 的帧
    weak = cv2.GaussianBlur(sharp[0], (9, 9), 0)
    med = float(np.median([qm.iqa.laplacian_var(f) for f in sharp]))
    assert qm.iqa.laplacian_var(weak) > TH_BLUR_VAR_ABS        # 绝对判据抓不到
    assert qm.iqa.laplacian_var(weak) < med * 0.35             # 相对判据抓得到

    assert qm.g1_blur_ok(sharp + [weak]) == pytest.approx(9 / 10)
    assert qm.g1_blur_ok(sharp) == 1.0


def test_g3_motion_blur_on_depth_grid():
    """G3 光流在 VGGT-depth-grid 上算（C-8）；给定深度网格时量纲随网格缩放。"""
    rng = np.random.default_rng(1)
    base = (rng.random((96, 128, 3)) * 255).astype(np.uint8)
    frames = [base, np.roll(base, shift=2, axis=1)]        # 水平位移 2px

    plain = qm.g3_motion_blur(frames)
    assert np.isfinite(plain) and plain > 0

    # 深度网格只有一半分辨率 → 位移在缩放后减半（证明确实先缩放再算光流）
    half = qm.g3_motion_blur(frames, (48, 64))
    assert np.isfinite(half)
    assert half < plain
    assert np.isnan(qm.g3_motion_blur([base]))


# -------------------------------- 硬约束 22/18：质量单一事实源与 frozen 只读 ----

def test_quality_status_failed_when_no_data_source(tmp_path):
    """无任何数据源时不得伪造 "computed"（质量算不出 → failed → route 必落 fallback）。"""
    art = _artifact(tmp_path)
    out = qm.compute_and_store_quality(art, artifact_path=str(tmp_path / "x.json"))
    assert out.quality_status == "failed" and out.quality is None
    assert not qm.quality_is_computed(out)


def test_frozen_artifact_is_never_rewritten_by_quality_gate(tmp_path):
    """硬约束 18：paired A/B 共用的 frozen artifact 不得被 A/B 之间的写回改动。

    v6 的冻结语义比 v5 更强：`quality_status="computed"` 的 artifact 直接复用
    （质量是 M4 的单一事实源，不再有"M5 统计回写"）；即使未算过，`persist_quality=False`
    也一个字节都不写。
    """
    art = _artifact(tmp_path).model_copy(
        update={"quality_status": "computed", "quality": _quality()})
    path = tmp_path / "frozen.json"
    path.write_text(art.model_dump_json(indent=2), encoding="utf-8")
    before = path.read_bytes()

    scene = quality_gate(art, artifact_path=str(path), persist_quality=False)
    assert scene.quality is not None                 # 内存里照常给出质量
    assert scene.scene_route == "full_3d"
    assert path.read_bytes() == before               # 已算过 → 不重算、不落盘

    # 未算过（且无数据源）→ 只能在内存里判 failed，frozen 文件仍不许动
    fresh = _artifact(tmp_path)
    scene2 = quality_gate(fresh, artifact_path=str(path), persist_quality=False)
    assert scene2.quality is None
    assert scene2.scene_route == "fallback_2d_only"  # fail-closed，不得停在 full_3d
    assert path.read_bytes() == before


# ------------------------------------------- §5.3/D3：scale 可用性 ----

def test_scale_artifact_requires_metric_gate_and_metric_question(tmp_path):
    """`scale` 可用 = 米制题型 ∧ MetricEvidenceGate 通过 ∧ scope=metric_enabled（§5.3）。

    这是 v5 `scale_known` 中间量 + `allowed_metric_tasks` 逐题型授权的 v6 替代物
    （§20：未标定恒 low → 米制题恒 0 的根源由 MetricEvidenceGate 显式表达）。
    """
    # ① 融合未跑（metric_scale=None / scale_fusion_status="not_run"）→ 任何题型都没有 scale
    art = _artifact(tmp_path).model_copy(
        update={"quality_status": "computed", "quality": _quality()})
    for qt in ("object_abs_distance", "object_counting"):
        scene = build_scene_state(art, m5=_m5(), objects=["obj_0"],
                                  artifact_ref="a", question_type=qt)
        handle = SceneHandle(scene, objects=[_obj()],
                             c2w_list=np.tile(np.eye(4), (2, 1, 1)),
                             intrinsics=np.tile(np.eye(3), (2, 1, 1)),
                             objects_materialized=True)
        assert "scale" not in handle.available_artifacts, qt
        # 收回 scale 不连累非米制 3D 产物（§7.2：单项失败只收回依赖它的工具）
        assert {"depth", "poses", "point_cloud", "objects"} <= handle.available_artifacts

    # ② 融合成功 + receipt（valid_frame_ratio ≥ τ_frames）+ 自洽 → 米制题 gate 过 → scale 可用
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"valid_frame_ratio": 0.9, "per_frame_scale": [1.0]}',
                       encoding="utf-8")
    fused = _artifact(tmp_path).model_copy(update={
        "quality_status": "computed", "quality": _quality(),
        "metric_scale": 5.0, "scale_fusion_status": "success",
        "scale_self_consistency": 0.05, "per_frame_scale_ref": str(receipt),
        "metric_model": "moge2", "metric_fusion_version": "metric-fusion-v6"})
    metric_scene = build_scene_state(fused, m5=_m5(), objects=["obj_0"],
                                     artifact_ref="a", question_type="object_abs_distance")
    assert metric_scene.metric_gate_passed is True
    assert metric_scene.question_tool_scope == "metric_enabled"
    metric_handle = SceneHandle(metric_scene, objects=[_obj()],
                                c2w_list=np.tile(np.eye(4), (2, 1, 1)),
                                intrinsics=np.tile(np.eye(3), (2, 1, 1)),
                                objects_materialized=True)
    assert "scale" in metric_handle.available_artifacts
    assert metric_handle.metric_scale == pytest.approx(5.0)

    # ③ 同一 artifact、非米制题 → 第 6 项子条件不过 → gate 不过 → 仍不给 scale（逐题）
    counting_scene = build_scene_state(fused, m5=_m5(), objects=["obj_0"],
                                       artifact_ref="a", question_type="object_counting")
    assert counting_scene.metric_gate_passed is False
    assert counting_scene.question_tool_scope == "full_3d"
    assert "scale" not in SceneHandle(
        counting_scene, objects=[_obj()],
        c2w_list=np.tile(np.eye(4), (2, 1, 1)),
        intrinsics=np.tile(np.eye(3), (2, 1, 1)),
        objects_materialized=True).available_artifacts


# ------------------------------------------- §5.3/M8：prompt 头部不得高报 ----

def test_prompt_header_reports_actually_loaded_artifacts(tmp_path):
    """prompt 的"可用产物"用句柄实际装载集合（否则模型照着写必撞 fail-closed）。"""
    from skill3d.adapters.episode_source import load_synthetic_items
    from skill3d.online.runner import _build_prompt

    scene = _scene_state(tmp_path, objects=())
    # 句柄只装载了 frames（无 c2w/intrinsics，objects 也未产出）
    handle = SceneHandle(scene, objects=[], objects_materialized=False)
    item = load_synthetic_items("inner_validation", question_types=["object_counting"],
                                frame_size=(32, 32), seed=0)[0]
    text = _build_prompt(item.episode, scene, handle, [], None)
    line = next(l for l in text.splitlines() if "可用重建产物=" in l)
    assert "frames" in line
    for absent in ("objects", "poses", "intrinsics", "scale"):
        assert absent not in line, f"未装载的产物 {absent} 不应出现在 prompt 头部: {line}"
    # 头部必须写明 v6 的 scope 字段（不再是 v5 的 route 一词三义）
    assert "question_tool_scope=" in line


def test_docs_trim_respects_actually_available_artifacts(tmp_path):
    """静态裁剪双条件：scope 声明 ∩ 实际装载产物（prompt 不再列必然抛错的 Tool）。"""
    scene = _scene_state(tmp_path, objects=())
    # full_3d 声明里含 objects/poses/intrinsics，但句柄只装了 frames/depth/point_cloud
    handle = SceneHandle(scene, objects=[], objects_materialized=False)
    facts = REGISTRY.docs(scene.question_tool_scope, available=handle.available_artifacts)
    assert "exists_in_scene" not in facts          # 需要 objects → 不该出现
    assert "reproject" not in facts                # 需要 poses/intrinsics → 不该出现
    assert "euclidean_distance" in facts           # 纯算术 → 保留
    # 不传 available 时保持 scope 口径（属性测试的单调性不受影响）
    assert "exists_in_scene" in REGISTRY.docs(scene.question_tool_scope)


# ------------------------------------------------------------------ 归档 ----
#
# 以下 v5 测试测的是 v6 已废止的机制，已整体移入 `tests/archive_v5/`（保持原样）：
# - `test_quality_gate_enriches_g7_g9_from_m5`（v5 的 "M5 统计 → quality 增量补写"）
#   → v6 替代物：`compute_quality(..., dynamic_masks=…, track_ious=…)` 一次算清 G7/G9
#   （§10.2 诊断项只产告警，不存在跨模块回写）；
# - `test_scale_artifact_hidden_when_scale_unusable`（v5 的 `scale_known` +
#   `allowed_metric_tasks` 授权）
#   → v6 替代物：`MetricEvidenceGate` 驱动 `scale` 可用性（见本文件
#   `test_scale_artifact_requires_metric_gate_and_metric_question`）。
