"""质量门禁与逐题门控单测（v5：HC22 route fail-closed / HC33 逐题型授权 / HC38）。

v5 口径变更：
- `scale_ci`（旧绝对半宽）已按 HC39 从 Schema 移除，不再作回退；
- `coverage_ok` 已按 HC38 删除（G8 永久退役，本版不设替代 geometry coverage 门），
  `room_size_estimation` 的 medium 条件改为三态 `plane_quality_ok`（None/False 一律
  fail-closed 不授权）。
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.reconstruction_gate.confidence_map import (
    coverage_ratio,
    fuse_confidence,
    track_ious_from_masks,
)
from skill3d.reconstruction_gate.scene_state import (
    QuestionGateDecision,
    question_gate,
    route_from_quality,
    scale_is_usable,
)
from skill3d.schemas.reconstruction import SceneState


def _scene(route="full_3d", scale_known=True, conf=None, ci_rel=None,
           metric_tasks=None, conflict=False) -> SceneState:
    """v5：`metric_tasks` = HC33 逐题型授权集合（默认空 = 无米制授权）。"""
    return SceneState(artifact_ref="a", route=route, frame="world",
                      scale_known=scale_known, objects=[], summary="s",
                      scale_confidence=conf, scale_ci_rel=ci_rel,
                      scale_conflict=conflict,
                      allowed_metric_tasks=set(metric_tasks or set()))




def test_track_ious_from_masks():
    m = np.zeros((4, 4), dtype=bool)
    m[:2, :2] = True
    same = {0: m, 1: m.copy(), 2: m.copy()}
    assert track_ious_from_masks([same])[0] == pytest.approx(1.0)
    shifted = np.zeros((4, 4), dtype=bool)
    shifted[2:, 2:] = True
    assert track_ious_from_masks([{0: m, 1: shifted}])[0] == pytest.approx(0.0)
    # 单帧无相邻对 → 不产出 IoU
    assert track_ious_from_masks([{0: m}]) == []


# ------------------------------------------------------------------ G-11 尺度可用性 ----

def test_scale_is_usable_ci_and_confidence_rules():
    """v5 HC29：CI 只接受**分数口径半宽** `scale_ci_rel`；旧绝对字段不再入参。"""
    assert scale_is_usable(True, "high", scale_ci_rel=0.05)
    assert scale_is_usable(True, "medium", scale_ci_rel=0.05)
    assert not scale_is_usable(True, "high", scale_ci_rel=0.5)   # CI 过宽
    assert not scale_is_usable(True, "low", scale_ci_rel=0.05)   # 置信档不足（§7.1）
    assert not scale_is_usable(False, "high", scale_ci_rel=0.05)
    # 无置信档 = 没锚定过，不等于"置信度没问题"（fail-closed，§3 M7 / D-2）
    assert not scale_is_usable(True, None, scale_ci_rel=None)
    assert not scale_is_usable(True, None, scale_ci_rel=float("nan"))
    assert scale_is_usable(True, "medium", scale_ci_rel=None)    # CI 缺失但档位够 → 可用


def test_scale_confidence_none_is_normalized_to_low():
    """D-2：显式 null 的 scale_confidence 一律归一为 low（老 artifact 也不 fail-open）。"""
    from skill3d.schemas import ConfidenceMap, ReconstructionArtifact

    art = ReconstructionArtifact(
        artifact_id="a", artifact_version="v", scene_name="s", recon_method="vggt",
        c2w_list="", intrinsics="", depth_maps="", point_map="", point_conf="",
        track_list=None, metric_scale=None, scale_known=True,
        confidence=ConfidenceMap(per_point_confidence="", coverage_count_per_frame=""))
    payload = art.model_dump()
    payload["scale_confidence"] = None
    parsed = ReconstructionArtifact.model_validate(payload)
    assert parsed.scale_confidence == "low"
    assert not scale_is_usable(parsed.scale_known, parsed.scale_confidence,
                               scale_ci_rel=parsed.scale_ci_rel)


def test_route_from_quality_branches():
    """Appendix A route 总纲：route 只由质量决定，尺度不参与（走逐题通道）。"""
    class _Q:
        def __init__(self, n, q):
            self.g4_frame_count, self.overall_quality = n, q

    assert route_from_quality(_Q(0, 0.9)) == "unanswerable"
    assert route_from_quality(_Q(32, 0.2)) == "fallback_2d_only"
    assert route_from_quality(_Q(32, float("nan"))) == "fallback_2d_only"
    assert route_from_quality(None) == "fallback_2d_only"
    assert route_from_quality(_Q(32, float("nan")),
                              quality_status="not_computed") == "fallback_2d_only"
    assert route_from_quality(_Q(32, 0.9)) == "full_3d"


# ------------------------------------------------------------------ 逐题门控 ----



def test_question_gate_metric_task_conditional_on_g9():
    """v4 §3 M7：medium 档的尺寸题必须 G9 达标才授权（否则逐题收回米制 Tool）。

    G8 包围盒覆盖条件已随指标删除（附录 A），尺寸题不再有"覆盖率拒答门"。
    """
    scene = _scene(conf="medium", ci_rel=0.10,
                   metric_tasks={"object_size_estimation"})
    ok = question_gate(scene, "object_size_estimation", g9_tracker_consistency=0.8)
    assert ok.route == "full_3d" and ok.allowed_metric_tasks == {"object_size_estimation"}
    # G9 不达标 → 收回米制 Tool（逐题收回，不是整题拒答，**也不改 route**）
    bad_g9 = question_gate(scene, "object_size_estimation", g9_tracker_consistency=0.2)
    assert bad_g9.allowed and bad_g9.route == "full_3d"
    assert "v5_metric_task_not_authorized" in bad_g9.flags
    assert not bad_g9.allowed_metric_tasks
    # G9 未知 → 无法证明达标 → 同样收回
    unknown = question_gate(scene, "object_size_estimation")
    assert unknown.route == "full_3d" and not unknown.allowed_metric_tasks
    # room_size_estimation 的 medium 条件 = 平面质量三态（HC38：**没有** coverage 门）。
    # None（证据缺失）与 False 一律 fail-closed 不授权，只有显式 True 才放行。
    rs = _scene(conf="medium", ci_rel=0.10, metric_tasks={"room_size_estimation"})
    assert not question_gate(rs, "room_size_estimation").allowed_metric_tasks
    assert not question_gate(rs, "room_size_estimation",
                             plane_quality_ok=False).allowed_metric_tasks
    assert question_gate(rs, "room_size_estimation",
                         plane_quality_ok=True).allowed_metric_tasks \
        == {"room_size_estimation"}
    # 无论授权与否，route 都由质量决定（§3"尺度能力门控"：不得改变原本合格的 full_3d）
    assert question_gate(rs, "room_size_estimation").route == "full_3d"


def test_question_gate_v4_metric_task_not_authorized_withdraws_tools_only():
    """v4 HC33 + §3 已确认设计：未获米制授权的 measurement 题**只收回米制 Tool**。

    历史缺陷（2026-09-21 修）：此处曾把 route 改成 `fallback_2d_only`，连带砍掉
    `depth/poses/point_cloud/objects` → 非米制 3D Tool 全部消失（outer_holdout 里
    12/32 题的 prompt 只剩 `euclidean_distance` 一个工具）。
    """
    for task in ("object_abs_distance", "object_size_estimation", "room_size_estimation"):
        d = question_gate(_scene(route="full_3d", scale_known=False), task)
        assert d.allowed and d.route == "full_3d", task
        assert "v5_metric_task_not_authorized" in d.flags
        assert not d.allowed_metric_tasks
        assert any("HC33" in r for r in d.reasons)


def test_question_gate_v4_only_authorized_task_passes():
    """v4 HC33：授权了哪个题型，就只有那个题型放行（逐题而非全局）。"""
    scene = _scene(conf="high", ci_rel=0.03, metric_tasks={"object_abs_distance"})
    ok = question_gate(scene, "object_abs_distance")
    assert ok.route == "full_3d" and ok.allowed_metric_tasks == {"object_abs_distance"}
    other = question_gate(scene, "room_size_estimation")
    assert other.route == "full_3d"
    assert not other.allowed_metric_tasks


def test_question_gate_g11_low_confidence_keeps_route_and_withdraws_metrics():
    """scale_confidence=low（§7.1 验收）→ 只收回米制 Tool，route 不变。"""
    d = question_gate(_scene(scale_known=False, conf="low", ci_rel=0.3),
                      "object_abs_distance")
    assert d.route == "full_3d" and d.allowed and not d.allowed_metric_tasks


def test_coverage_gate_status_is_not_defined_never_passed():
    """v5 HC38：本版不设 geometric_coverage 门 → 字段恒 not_defined，不得写 passed。"""
    scene = _scene(metric_tasks={"room_size_estimation"})
    assert scene.coverage_gate_status == "not_defined"
    with pytest.raises(Exception):
        SceneState(artifact_ref="a", route="full_3d", frame="world", scale_known=False,
                   objects=[], summary="s", coverage_gate_status="passed")


def test_question_gate_mca_task_unaffected_by_scale():
    d = question_gate(_scene(scale_known=False), "object_rel_direction_hard")
    assert d.allowed and d.route == "full_3d" and not d.flags


def test_question_gate_scene_unanswerable_short_circuits():
    d = question_gate(_scene(route="unanswerable"), "object_counting")
    assert not d.allowed and d.route == "unanswerable"
    assert "scene_unanswerable" in d.flags


def test_question_gate_unknown_type_is_not_extra_gated():
    d = question_gate(_scene(), "not_a_real_type")
    assert d.allowed


def test_question_gate_note_is_human_readable():
    d = question_gate(_scene(), "object_counting")
    assert "通过" in d.note()
    d2 = QuestionGateDecision(route="unanswerable", allowed=False,
                              flags=["metric_task_not_authorized"], reasons=["x"])
    assert "metric_task_not_authorized" in d2.note()
