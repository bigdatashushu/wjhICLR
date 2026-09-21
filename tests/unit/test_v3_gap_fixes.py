"""v3 规格偏差修复的回归护栏（逐条对应《系统架构3.md》条款）。

每条测试都钉住一个**曾经不满足规格**的具体行为，防止回退：

| 测试 | 规格条款 |
|---|---|
| `test_program_swallowing_contract_error_is_not_scored` | §3 M10 硬约束 23：答案依赖过契约失败的 Tool → 不得采纳 |
| `test_m8_never_silently_drops_frames` | §3 M8 硬约束 26：不得纯文本、不得静默丢帧 |
| `test_exists_in_scene_false_when_objects_empty_but_materialized` | §3 M6 硬约束 23：False 当且仅当产物可用且真无 |
| `test_g1_uses_episode_relative_criterion` | Appendix A G1 双判据（B-3） |
| `test_g3_motion_blur_on_depth_grid` | §9 / C-8：光流必须在深度网格上算 |
| `test_quality_gate_enriches_g7_g8_g9_from_m5` | §3 M4 / D-5：quality 单一事实源，M5 产物不得被丢弃 |
| `test_quality_status_failed_when_no_data_source` | §4.1 / 硬约束 22：三态之一 `failed` 必须有生产者 |
| `test_prompt_header_reports_actually_loaded_artifacts` | §3 M6：prompt 头部"可用产物"不得高报 |
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.reconstruction_gate import quality_metrics as qm
from skill3d.reconstruction_gate.scene_state import quality_gate
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas import ConfidenceMap, ReconstructionArtifact, SceneState
from skill3d.tools import REGISTRY
from skill3d.tools.scene_handle import SceneHandle
from skill3d.tools.contract import ArtifactUnavailableError


# ------------------------------------------------------------------ 夹具 ----

def _artifact(tmp_path) -> ReconstructionArtifact:
    """route=full_3d 的 artifact；位姿/内参数组不落位（reproject 会 fail-closed）。"""
    c2w = np.tile(np.eye(4), (32, 1, 1))
    k = np.tile(np.eye(3), (32, 1, 1))
    p = tmp_path / "a.npy"
    np.save(p, c2w)
    kn = tmp_path / "k.npy"
    np.save(kn, k)
    return ReconstructionArtifact(
        artifact_id="art-gap", artifact_version="v1", scene_name="scene-x",
        recon_method="vggt", frame_ids=list(range(32)),
        source_frame_indices=list(range(32)),
        timestamps=[float(i) for i in range(32)], frame_set_hash="h32",
        c2w_list=str(p), intrinsics=str(kn), depth_maps="", point_map="",
        point_conf="", track_list=None, metric_scale=None, scale_known=False,
        confidence=ConfidenceMap(per_point_confidence="", coverage_count_per_frame=""))


def _quality(**over) -> qm.QualityMetrics:
    base = dict(g1_blur_ok=1.0, g2_brightness=1.0, g3_motion_blur=1.0,
                g4_frame_count=32, g5_reproj_err_median=float("nan"),
                g5_reproj_err_p95=float("nan"), g6_depth_var_coeff=0.5,
                g7_dynamic_ratio=float("nan"), g9_tracker_consistency=float("nan"), g10_baseline_quality=0.3,
                g11_scale_ci=float("nan"), overall_quality=0.9)
    base.update(over)
    return qm.QualityMetrics(**base)


def _scene(route="full_3d", scale_known=False, metric_tasks=None) -> SceneState:
    return SceneState(artifact_ref="a", route=route, frame="world",
                      scale_known=scale_known, objects=[], summary="s",
                      allowed_metric_tasks=set(metric_tasks or set()))


# ------------------------------------------------ 硬约束 23：自捕获契约异常 ----

def test_program_swallowing_contract_error_is_not_scored():
    """程序 `try/except` 吞掉契约异常后再作答 → 按 tool_contract 归因，答案不采纳。

    规格（§3 M10 / 硬约束 23）："一旦 ReturnAnswer 依赖的某次 Tool 抛
    ArtifactUnavailableError，最终答案不得采纳"。`error_code` 为空（异常被吞）
    不能成为绕过 fail-closed 的口子。
    """
    scene = _scene("fallback_2d_only")   # objects/poses 均不可用
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
    assert cell.contract_violations              # 契约违规有据可查


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


# ------------------------------------------- 硬约束 23：objects 产物语义 ----

def test_exists_in_scene_false_when_objects_empty_but_materialized():
    """M5 跑过但没绑到对象 → 产物**存在且为空** → False，而不是抛"产物缺失"。"""
    scene = _scene("full_3d")
    materialized = SceneHandle(scene, objects=[], c2w_list=np.tile(np.eye(4), (2, 1, 1)),
                               intrinsics=np.tile(np.eye(3), (2, 1, 1)),
                               objects_materialized=True)
    assert "objects" in materialized.available_artifacts
    assert REGISTRY.call_tool("exists_in_scene", {"name": "table"}, materialized,
                              mode="real").value == "false"

    # M5 未跑（产物未产出）→ 必须 fail-closed 抛错（不得假装"场景里没有"）
    absent = SceneHandle(scene, objects=[], objects_materialized=False)
    with pytest.raises(ArtifactUnavailableError):
        REGISTRY.call_tool("exists_in_scene", {"name": "table"}, absent, mode="real")

    # 未显式声明时保持老语义（空列表 = 未产出）
    legacy = SceneHandle(scene, objects=[])
    assert "objects" not in legacy.available_artifacts


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


# --------------------------------------- D-5：M5 指标补写进 quality ----

def test_quality_gate_enriches_g7_g9_from_m5(tmp_path):
    """方案 X 落盘的 quality 缺 G7/G9 → P2 拿到 M5 统计后补写并原子落盘。

    （G8 已按附录 A 删除，不再参与补写。）
    """
    art = _artifact(tmp_path).model_copy(
        update={"quality_status": "computed", "quality": _quality()})
    path = tmp_path / "art.json"
    path.write_text(art.model_dump_json(indent=2), encoding="utf-8")

    masks = np.zeros((32, 8, 8), dtype=bool)
    masks[:8] = True                                        # 动态占比 1/4
    scene = quality_gate(art, dynamic_masks=masks,
                         track_ious=[0.9, 0.8], artifact_path=str(path))

    q = scene.quality
    assert q.g7_dynamic_ratio == pytest.approx(0.25)
    assert q.g9_tracker_consistency == pytest.approx(0.85)
    # 补写后的 overall 必须重算（不再是"只有 G1-G4/G6/G10"的那份）
    assert q.overall_quality == pytest.approx(qm.overall_from_metrics(q))
    # 原子写回：落盘文件里也能读到补写后的值（单一事实源，不是内存幻觉）
    import json

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["quality"]["g7_dynamic_ratio"] == pytest.approx(0.25)


def test_quality_gate_does_not_persist_for_frozen_ab_artifact(tmp_path):
    """硬约束 18：paired A/B 共用的 frozen artifact 不得被 A/B 之间的补写改动。"""
    art = _artifact(tmp_path).model_copy(
        update={"quality_status": "computed", "quality": _quality()})
    path = tmp_path / "frozen.json"
    path.write_text(art.model_dump_json(indent=2), encoding="utf-8")
    before = path.read_bytes()

    scene = quality_gate(art, dynamic_masks=np.ones((4, 4, 4), dtype=bool),
                         artifact_path=str(path), persist_quality=False)
    assert np.isfinite(scene.quality.g7_dynamic_ratio)      # 内存里补上了
    assert path.read_bytes() == before                     # 文件没被动过


def test_quality_status_failed_when_no_data_source(tmp_path):
    """无任何数据源时不得伪造 "computed"（overall 会被 G4 一项撑起来）。"""
    art = _artifact(tmp_path)
    out = qm.compute_and_store_quality(art, artifact_path=str(tmp_path / "x.json"))
    assert out.quality_status == "failed" and out.quality is None
    assert not qm.quality_is_computed(out)


# ------------------------------------------- §3 M6：prompt 头部不得高报 ----

def test_scale_artifact_hidden_when_scale_unusable():
    """尺度不可用时 `scale` 不得声明为可用（硬约束 23：metric Tool 必须 fail-closed）。

    v4 HC33：`scale` 可用性还要求**至少一个米制题型被授权**（逐题型授权），
    所以"scale_known=True 但 allowed_metric_tasks=∅"同样必须收回 `scale`。
    """
    unusable = SceneHandle(_scene("full_3d"), objects=[_obj()],
                           c2w_list=np.tile(np.eye(4), (2, 1, 1)),
                           intrinsics=np.tile(np.eye(3), (2, 1, 1)),
                           objects_materialized=True)
    assert "scale" not in unusable.available_artifacts     # scale_known=False
    # scale_known=True 但无任何米制授权 → 仍不可用（v4 HC33）
    no_auth = SceneHandle(_scene("full_3d", scale_known=True), objects=[_obj()],
                          c2w_list=np.tile(np.eye(4), (2, 1, 1)),
                          intrinsics=np.tile(np.eye(3), (2, 1, 1)),
                          objects_materialized=True)
    assert "scale" not in no_auth.available_artifacts
    # HC33：收回 scale 不影响非米制 3D 产物
    assert {"depth", "poses", "point_cloud", "objects"} <= no_auth.available_artifacts
    # 有授权 → 可用
    usable = SceneHandle(_scene("full_3d", scale_known=True,
                                metric_tasks={"object_abs_distance"}), objects=[_obj()],
                         c2w_list=np.tile(np.eye(4), (2, 1, 1)),
                         intrinsics=np.tile(np.eye(3), (2, 1, 1)),
                         objects_materialized=True)
    assert "scale" in usable.available_artifacts


def _obj():
    from skill3d.schemas import ObjectInstance

    return ObjectInstance(instance_id="obj_0", class_hint="table", mask_per_frame="",
                          pointcloud_world="", centroid_world=[0.0, 0.0, 0.0],
                          bbox=[0.0] * 6, confidence=0.9)


def test_prompt_header_reports_actually_loaded_artifacts():
    """prompt 的"可用产物"用句柄实际装载集合（否则模型照着写必撞 fail-closed）。"""
    from skill3d.adapters.episode_source import load_synthetic_items
    from skill3d.online.runner import _build_prompt

    scene = _scene("full_3d")
    # 句柄只装载了 frames（无 c2w/intrinsics，objects 也未产出）
    handle = SceneHandle(scene, objects=[], objects_materialized=False)
    item = load_synthetic_items("inner_validation", question_types=["object_counting"],
                                frame_size=(32, 32), seed=0)[0]
    text = _build_prompt(item.episode, scene, handle, [], None)
    line = next(l for l in text.splitlines() if "可用重建产物=" in l)
    assert "frames" in line
    for absent in ("objects", "poses", "intrinsics"):
        assert absent not in line, f"未装载的产物 {absent} 不应出现在 prompt 头部: {line}"


def test_docs_trim_respects_actually_available_artifacts():
    """静态裁剪双条件：route 声明 ∩ 实际装载产物（prompt 不再列必然抛错的 Tool）。"""
    from skill3d.tools import REGISTRY

    # full_3d 声明里含 objects/poses/intrinsics，但句柄只装了 frames
    scene = _scene("full_3d")
    handle = SceneHandle(scene, objects=[], objects_materialized=False)
    facts = REGISTRY.docs(route="full_3d", available=handle.available_artifacts)
    assert "exists_in_scene" not in facts          # 需要 objects → 不该出现
    assert "reproject" not in facts                # 需要 poses/intrinsics → 不该出现
    assert "euclidean_distance" in facts           # 纯算术 → 保留
    # 不传 available 时保持 route 口径（属性测试的单调性不受影响）
    assert "exists_in_scene" in REGISTRY.docs(route="full_3d")
