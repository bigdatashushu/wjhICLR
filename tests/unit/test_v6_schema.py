"""v6 Schema 6.0 + EvidenceProfile + 工具证据门 负向测试（v6 §5、§7.2、§13、§23.1 Phase 0 DoD）。

本文件是 v6 Phase 0 完成定义的硬门禁测试。以下情形**必须** fail-closed，
且不得通过删除测试、放宽断言或 mock 获得假绿：

1. `ReconstructionArtifact` 只接受 `schema_version="6.0"` / `recon_method="vggt"`；
   v5 尺度/BA/G8/pad 字段出现即 hard fail（`LEGACY_ONLY_FIELDS`）；
2. G5 字段**固定** `not_available` + `None`，不参与聚合，不得用代理值冒充；
3. 世界系契约：`world_frame_status="available"` 必须同时有 `world_up`（**单位**向量）
   与 `handedness`；非单位向量不得静默归一化；
4. 度量融合：`scale_fusion_status="success"` 必须有有限正 `metric_scale`；
   非 success 时 `metric_scale` 必须为 `None`（PoC 通过前不得冒充）；
5. `SceneState`：`docs(question_tool_scope) ⊆ docs(scene_route)`，逐题只收窄；
6. `EvidenceProfile`：三值有序、`temporal`/`image_2d` 恒 available、
   未知能力名不得静默放行；
7. 工具证据门：`requires_evidence` 声明缺失/未知能力/容忍列表越界 → 注册失败；
   `MetricEvidenceGate` 6 项子条件全真才通过，且 gate 只影响 `metric_scale` 分项；
8. 单项失败只收回依赖该项的 Tool（§7.2 的核心不变量）。
"""

from __future__ import annotations

import pytest

from skill3d.reconstruction_gate.evidence_profile import (
    M5EvidenceSummary,
    build_evidence_profile,
    evaluate_metric_gate,
    metric_scale_capability,
    world_frame_capability,
)
from skill3d.schemas import (
    LEGACY_ONLY_FIELDS,
    EvidenceProfile,
    MetricEvidenceGateResult,
    QualityMetrics,
    ReconstructionArtifact,
    SceneState,
    ToolSpec,
)
from skill3d.schemas.evidence import (
    CAPABILITY_ORDER,
    GATE_SUB_FINITE_INPUTS,
    GATE_SUB_FUSION_SUCCESS,
    GATE_SUB_M4_MAIN_GATE,
    GATE_SUB_METRIC_QUESTION,
    GATE_SUB_SCALE_SELF_CONSISTENCY,
    GATE_SUB_SCENE_ROUTE,
    capability_at_least,
)
from skill3d.tools.contract import (
    EVIDENCE_METRIC_SCALE,
    SCOPE_FALLBACK_2D_ONLY,
    SCOPE_FULL_3D,
    SCOPE_METRIC_ENABLED,
    available_artifacts_for,
    degraded_evidence_flags,
    evidence_visible,
    question_tool_scope_of,
    scope_allows,
)
from skill3d.tools.registry import ToolRegistry

UP = [0.0, 1.0, 0.0]


# --------------------------------------------------------------- 构造助手 ----

def _quality(**kw) -> QualityMetrics:
    base = dict(
        warp_inlier_ratio=0.9, warp_photometric_inlier_ratio=0.9,
        cloud_overlap_ratio=0.8, main_gate_passed=True,
        g1_blur_ok=1.0, g2_brightness=1.0, g3_motion_blur=0.0, g4_frame_count=32,
        g6_depth_var_coeff=0.1, g7_dynamic_ratio=0.0, g9_tracker_consistency=0.9,
        g10_baseline_quality=0.5, overall_quality=0.9)
    base.update(kw)
    return QualityMetrics(**base)


def _art(**kw) -> ReconstructionArtifact:
    base = dict(
        artifact_id="a", artifact_version="v", scene_name="s",
        c2w_list="c2w.npy", intrinsics="k.npy", depth_maps="d.npy",
        point_map="p.npy", point_conf="pc.npy",
        quality_status="computed", quality=_quality(),
        confidence={"per_point_confidence": "pc.npy", "coverage_count_per_frame": ""})
    base.update(kw)
    return ReconstructionArtifact(**base)


def _scene(**kw) -> SceneState:
    base = dict(artifact_ref="a", scene_route="full_3d",
                question_tool_scope="full_3d",
                available_artifacts=available_artifacts_for("full_3d"))
    base.update(kw)
    return SceneState(**base)


# --------------------------------------------- 1. Schema 6.0 与 legacy 隔离 ----

def test_schema_version_is_6_and_recon_method_only_vggt():
    a = _art()
    assert a.schema_version == "6.0"
    assert a.quality_metric_version == "v6-warp-overlap-no-g5"
    assert a.recon_method == "vggt"
    for bad in ("vggt_sparse_ba", "dust3r_mast3r", "colmap", "vggt_ba"):
        with pytest.raises(Exception):
            _art(recon_method=bad)


@pytest.mark.parametrize("field,value", [
    ("scale_known", True),
    ("scale_ci_rel", 0.1),
    ("scale_ci_abs_m", 0.1),
    ("scale_confidence", "high"),
    ("scale_confidence_level", 0.9),
    ("scale_anchor_fired", []),
    ("scale_conflict", False),
    ("scale_calibration_id", "cal-1"),
    ("allowed_metric_tasks", {"object_abs_distance"}),
    ("scale_method", "m"),
    ("scale_source", "src"),
    ("scale_ci", 0.1),
    ("relative_ci", 0.1),
    ("sparse_ba_receipt_ref", "r.json"),
    ("reproj_errors", "e.npy"),
    ("g8_bbox_coverage_min", 0.5),
    ("bbox_coverage_ratio", 0.5),
    ("CoverageMap", {}),
    ("grid_transform", {}),
])
def test_legacy_fields_hard_fail(field, value):
    """v5 尺度/BA/G8/pad 字段一律 hard fail（不静默忽略，不"读进来降级"）。"""
    assert field in LEGACY_ONLY_FIELDS
    with pytest.raises(Exception) as ei:
        _art(**{field: value})
    assert "legacy" in str(ei.value)


def test_quality_metrics_rejects_retired_metrics():
    """G5/G8/G11 不得出现在 QualityMetrics 里（禁止代理值冒充重投影残差）。"""
    for banned in ("g5_reproj_err_median", "g5_reproj_err_p95",
                   "g8_bbox_coverage_min", "g11_scale_ci"):
        with pytest.raises(Exception):
            _quality(**{banned: 0.1})


# ------------------------------------------------- 2. G5 永久 not_available ----

def test_g5_is_permanently_none_and_not_available():
    a = _art()
    assert a.reprojection_status == "not_available"
    assert a.g5_reproj_err_median is None and a.g5_reproj_err_p95 is None
    # 任何"声明 computed 但没有真 BA"的写法都必须写不进来
    with pytest.raises(Exception):
        _art(reprojection_status="computed")
    with pytest.raises(Exception):
        _art(g5_reproj_err_median=1.0)


# --------------------------------------------------- 3. 世界系契约（D5）----

def test_world_frame_available_requires_both_fields():
    for kw in ({"world_frame_status": "available", "handedness": "right"},
               {"world_frame_status": "available", "world_up": UP}):
        with pytest.raises(Exception):
            _art(**kw)
    a = _art(world_frame_status="available", world_up=UP, handedness="right")
    assert a.world_up == UP and a.handedness == "right"


def test_world_up_must_be_unit_and_finite():
    """非单位向量一律 fail-closed（**不做**静默归一化）。"""
    for bad in ([0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0],
                [float("inf"), 0.0, 0.0], [1.0, 1.0]):
        with pytest.raises(Exception):
            _art(world_frame_status="available", world_up=bad, handedness="right")


def test_world_frame_capability_three_values():
    assert world_frame_capability(world_frame_status="available", world_up=UP,
                                  handedness="right") == "available"
    assert world_frame_capability(world_frame_status="degraded", world_up=None,
                                  handedness="right") == "degraded"
    assert world_frame_capability(world_frame_status="unavailable", world_up=None,
                                  handedness=None) == "unavailable"
    # 声称 available 但没给 up → 一律 unavailable（自相矛盾的产物不得留半条证据）
    assert world_frame_capability(world_frame_status="available", world_up=None,
                                  handedness="right") == "unavailable"
    # 声称 degraded 但没有 handedness（连手性都不知道）→ 同样 unavailable
    assert world_frame_capability(world_frame_status="degraded", world_up=None,
                                  handedness=None) == "unavailable"


# ------------------------------------------------- 4. 度量融合字段（D1/D2）----

def test_metric_scale_none_until_fusion_succeeds():
    a = _art()
    assert a.scale_fusion_status == "not_run" and a.metric_scale is None
    with pytest.raises(Exception):
        _art(metric_scale=1.5)               # 非 success 却给了尺度 → hard fail
    with pytest.raises(Exception):
        _art(scale_fusion_status="success")  # success 必须有有限正尺度
    with pytest.raises(Exception):
        _art(scale_fusion_status="success", metric_scale=-1.0)
    ok = _art(scale_fusion_status="success", metric_scale=1.5,
              scale_self_consistency=0.05, metric_model="moge2")
    assert ok.metric_scale == 1.5


# ------------------------------------------- 5. SceneState 双路由字段（D4）----

def test_question_tool_scope_cannot_exceed_scene_route():
    """逐题只收窄：fallback_2d_only 场景不得出现 full_3d / metric_enabled 的 scope。"""
    with pytest.raises(Exception):
        _scene(scene_route="fallback_2d_only", question_tool_scope="full_3d")
    with pytest.raises(Exception):
        _scene(scene_route="fallback_2d_only", question_tool_scope="metric_enabled")
    with pytest.raises(Exception):
        _scene(scene_route="unanswerable", question_tool_scope="metric_enabled")


def test_scope_of_derivation_is_narrowing_only():
    assert question_tool_scope_of("full_3d", metric_question=False,
                                  gate_passed=False) == SCOPE_FULL_3D
    assert question_tool_scope_of("full_3d", metric_question=True,
                                  gate_passed=True) == SCOPE_METRIC_ENABLED
    assert question_tool_scope_of("full_3d", metric_question=True,
                                  gate_passed=False) == SCOPE_FULL_3D
    for route in ("fallback_2d_only", "unanswerable"):
        assert question_tool_scope_of(route, metric_question=True,
                                      gate_passed=True) == SCOPE_FALLBACK_2D_ONLY


def test_available_artifacts_scale_is_conditional():
    """`scale` 只在（米制题 ∧ gate 通过 ∧ full_3d）时进入可用产物集（§5.3）。"""
    base = available_artifacts_for("full_3d")
    assert "scale" not in base
    assert "depth" in base and "objects" in base
    with_scale = available_artifacts_for("full_3d", question_type="object_abs_distance",
                                         metric_question=True, gate_passed=True)
    assert "scale" in with_scale
    # 非米制题即便 gate 通过也不给 scale
    assert "scale" not in available_artifacts_for(
        "full_3d", question_type="object_counting", metric_question=False,
        gate_passed=True)
    # 2D-only 场景永远不给 scale
    assert "scale" not in available_artifacts_for(
        "fallback_2d_only", question_type="object_abs_distance",
        metric_question=True, gate_passed=True)


def test_metric_task_authorized_requires_scope_and_gate():
    gate_ok = MetricEvidenceGateResult(
        gate_passed=True, gate_version="g",
        sub_results={k: True for k in (GATE_SUB_SCENE_ROUTE, GATE_SUB_M4_MAIN_GATE,
                                       GATE_SUB_FUSION_SUCCESS,
                                       GATE_SUB_SCALE_SELF_CONSISTENCY,
                                       GATE_SUB_FINITE_INPUTS, GATE_SUB_METRIC_QUESTION)})
    s = _scene(scene_route="full_3d", question_tool_scope="metric_enabled",
               metric_evidence_gate_result=gate_ok)
    assert s.metric_gate_passed and s.metric_task_authorized("object_abs_distance")
    # 非米制题型不得被授权米制 Tool
    assert not s.metric_task_authorized("object_counting")
    # scope 不是 metric_enabled 时不得授权
    s2 = _scene(scene_route="full_3d", question_tool_scope="full_3d",
                metric_evidence_gate_result=gate_ok)
    assert s2.metric_gate_passed and not s2.metric_task_authorized("object_abs_distance")


# --------------------------------------------- 6. EvidenceProfile 三值纪律 ----

def test_evidence_profile_order_and_unknown_capability():
    assert CAPABILITY_ORDER["available"] > CAPABILITY_ORDER["degraded"] \
        > CAPABILITY_ORDER["unavailable"]
    assert capability_at_least("available", "degraded")
    assert capability_at_least("degraded", "degraded")
    assert not capability_at_least("unavailable", "degraded")
    # 未知取值一律当 unavailable（fail-closed）
    assert not capability_at_least("bogus", "available")
    assert not capability_at_least("bogus", "unavailable")


def test_evidence_profile_temporal_and_image_2d_never_degrade():
    """§5.4：统一 FrameSet 存在时二者恒 available，是 direct_vlm_routed 的兜底基础。"""
    p = EvidenceProfile(geometry_3d="unavailable", world_frame="unavailable",
                        metric_scale="unavailable", object_detection="unavailable",
                        track_consensus="unavailable")
    assert p.temporal == "available" and p.image_2d == "available"
    for bad in ("degraded", "unavailable"):
        with pytest.raises(Exception):
            EvidenceProfile(geometry_3d="available", world_frame="available",
                            metric_scale="available", object_detection="available",
                            track_consensus="available", temporal=bad)


def test_evidence_profile_signature_matching_fail_closed():
    p = EvidenceProfile(geometry_3d="available", world_frame="degraded",
                        metric_scale="unavailable", object_detection="available",
                        track_consensus="degraded")
    assert p.satisfies({"geometry_3d": "available"})
    assert not p.satisfies({"metric_scale": "degraded"})
    assert p.satisfies({"world_frame": "degraded"})
    # 未知能力名 → 不匹配（归纳器写错不得静默放行）
    assert not p.satisfies({"not_a_capability": "available"})
    with pytest.raises(KeyError):
        p.state("nope")


def test_metric_scale_capability_three_values():
    gate_ok = MetricEvidenceGateResult(
        gate_passed=True, gate_version="g",
        sub_results={k: True for k in (GATE_SUB_SCENE_ROUTE, GATE_SUB_M4_MAIN_GATE,
                                       GATE_SUB_FUSION_SUCCESS,
                                       GATE_SUB_SCALE_SELF_CONSISTENCY,
                                       GATE_SUB_FINITE_INPUTS, GATE_SUB_METRIC_QUESTION)})
    assert metric_scale_capability(gate=gate_ok, fusion_status="success",
                                   scale_self_consistency=0.05, valid_frame_ratio=1.0,
                                   metric_scale=1.5) == "available"
    # 融合成功但自洽松 → degraded（不是 unavailable，也不是 available）
    gate_loose = MetricEvidenceGateResult(
        gate_passed=False, gate_version="g",
        sub_results={**{k: True for k in (GATE_SUB_SCENE_ROUTE, GATE_SUB_M4_MAIN_GATE,
                                          GATE_SUB_FUSION_SUCCESS,
                                          GATE_SUB_FINITE_INPUTS)},
                     GATE_SUB_SCALE_SELF_CONSISTENCY: False,
                     GATE_SUB_METRIC_QUESTION: False},
        missing_subconditions=[GATE_SUB_SCALE_SELF_CONSISTENCY, GATE_SUB_METRIC_QUESTION])
    assert metric_scale_capability(gate=gate_loose, fusion_status="success",
                                   scale_self_consistency=0.9, valid_frame_ratio=1.0,
                                   metric_scale=1.5) == "degraded"
    assert metric_scale_capability(gate=None, fusion_status="failed",
                                   scale_self_consistency=None, valid_frame_ratio=0.0,
                                   metric_scale=None) == "unavailable"


# ------------------------------------------------ 7. MetricEvidenceGate（D3）----

def _gate(**kw):
    base = dict(scene_route="full_3d", main_gate_passed=True, fusion_status="success",
                scale_self_consistency=0.05, valid_frame_ratio=1.0, metric_scale=1.5,
                inputs_finite=True, question_type="object_abs_distance")
    base.update(kw)
    return evaluate_metric_gate(**base)


def test_metric_gate_all_six_conditions_required():
    g = _gate()
    assert g.gate_passed and not g.missing_subconditions
    cases = {
        GATE_SUB_SCENE_ROUTE: dict(scene_route="fallback_2d_only"),
        GATE_SUB_M4_MAIN_GATE: dict(main_gate_passed=False),
        GATE_SUB_FUSION_SUCCESS: dict(fusion_status="failed"),
        GATE_SUB_SCALE_SELF_CONSISTENCY: dict(scale_self_consistency=0.9),
        GATE_SUB_FINITE_INPUTS: dict(inputs_finite=False),
        GATE_SUB_METRIC_QUESTION: dict(question_type="object_counting"),
    }
    for sub, kw in cases.items():
        gi = _gate(**kw)
        assert gi.gate_passed is False, sub
        assert sub in gi.missing_subconditions, sub


def test_metric_gate_result_rejects_self_contradiction():
    """`gate_passed` 必须与 `sub_results` 自洽，否则伪造门通过。"""
    with pytest.raises(Exception):
        MetricEvidenceGateResult(gate_passed=True, gate_version="g",
                                 sub_results={"a": True, "b": False})
    with pytest.raises(Exception):
        MetricEvidenceGateResult(gate_passed=True, gate_version="g")


def test_gate_is_metric_only_and_does_not_touch_other_capabilities():
    """§13.8：MetricEvidenceGate 只设置 `metric_scale` 分项，不影响其他能力。"""
    art = _art(world_frame_status="available", world_up=UP, handedness="right")
    gate_fail = _gate(main_gate_passed=False,
                      question_type="object_abs_distance")
    p = build_evidence_profile(artifact=art, scene_route="full_3d",
                               question_type="object_abs_distance", gate=gate_fail)
    assert p.metric_scale == "unavailable"
    # world_frame 仍然 available —— 米制失败不波及其他能力（§7.3 示例 B）
    assert p.world_frame == "available"
    assert p.image_2d == "available" and p.temporal == "available"


# -------------------------------------------- 8. 工具证据门 / 单项失败收窄 ----

def _spec(**kw) -> ToolSpec:
    base = dict(name="t", description="d", args_schema_ref="a", returns_schema_ref="r",
                cost_estimate_ms=1.0, source_default="real", requires_artifacts=[])
    base.update(kw)
    return ToolSpec(**base)


def test_toolspec_requires_explicit_evidence_declaration():
    """§17.4 HC7：无 `requires_evidence` 声明 → 构造即失败。"""
    with pytest.raises(Exception):
        ToolSpec(name="t", description="d", args_schema_ref="a", returns_schema_ref="r",
                 cost_estimate_ms=1.0, source_default="real", requires_artifacts=[])


def test_registry_rejects_bad_evidence_declarations():
    reg = ToolRegistry()
    # 未知能力名
    with pytest.raises(ValueError):
        reg.register(_spec(name="x", requires_evidence=["not_a_capability"]))(lambda h: None)
    # 容忍列表越界（不在 requires_evidence 内）
    with pytest.raises(ValueError):
        reg.register(_spec(name="y", requires_evidence=[],
                           tolerates_degraded=["geometry_3d"]))(lambda h: None)
    # 米制证据依赖但产物里没有 scale
    with pytest.raises(ValueError):
        reg.register(_spec(name="z", requires_evidence=[EVIDENCE_METRIC_SCALE]))(
            lambda h: None)


def test_evidence_visible_per_tool_single_failure_isolation():
    """§7.2 核心不变量：某能力失败**只**隐藏依赖它的 Tool。"""
    p = EvidenceProfile(geometry_3d="unavailable", world_frame="available",
                        metric_scale="unavailable", object_detection="available",
                        track_consensus="available", object_grounding="available")
    geo_tool = _spec(requires_evidence=["geometry_3d"])
    det_tool = _spec(requires_evidence=["object_detection"])
    metric_tool = _spec(requires_evidence=["geometry_3d", EVIDENCE_METRIC_SCALE])
    assert not evidence_visible(geo_tool, p)
    assert evidence_visible(det_tool, p)          # 不连带隐藏
    assert not evidence_visible(metric_tool, p)
    # 无证据依赖的纯算术 Tool 在任何状态下都可用
    pure = _spec(requires_evidence=[])
    assert evidence_visible(pure, p)
    # profile 缺失 → 只放行无证据依赖的 Tool（fail-closed）
    assert not evidence_visible(det_tool, None)
    assert evidence_visible(pure, None)


def test_degraded_tool_visible_only_when_tolerated_and_flagged():
    p = EvidenceProfile(geometry_3d="available", world_frame="available",
                        metric_scale="available", object_detection="available",
                        track_consensus="degraded")
    strict = _spec(requires_evidence=["object_detection", "track_consensus"])
    tolerant = _spec(requires_evidence=["object_detection", "track_consensus"],
                     tolerates_degraded=["track_consensus"])
    assert not evidence_visible(strict, p)         # degraded 且不在容忍列表 → 隐藏
    assert evidence_visible(tolerant, p)           # 容忍 → 暴露但带标记
    assert degraded_evidence_flags(tolerant, p) == ["evidence_degraded:track_consensus"]
    assert degraded_evidence_flags(strict, p) == []


def test_scope_allows_narrowing_only():
    metric_tool = _spec(requires_evidence=[EVIDENCE_METRIC_SCALE],
                        requires_artifacts=["scale"])
    plain3d = _spec(requires_evidence=["geometry_3d"], requires_artifacts=["depth"])
    pure = _spec(requires_evidence=[])
    assert scope_allows(metric_tool, SCOPE_METRIC_ENABLED)
    assert not scope_allows(metric_tool, SCOPE_FULL_3D)
    assert not scope_allows(metric_tool, SCOPE_FALLBACK_2D_ONLY)
    assert scope_allows(plain3d, SCOPE_METRIC_ENABLED)
    assert scope_allows(plain3d, SCOPE_FULL_3D)
    assert not scope_allows(plain3d, SCOPE_FALLBACK_2D_ONLY)
    for scope in (SCOPE_METRIC_ENABLED, SCOPE_FULL_3D, SCOPE_FALLBACK_2D_ONLY):
        assert scope_allows(pure, scope)


def test_registry_docs_tracks_evidence_and_scope():
    """`docs()` 列出的 Tool 集合必须与证据/scope 判定一致（Phase 0 DoD）。"""
    import skill3d.tools  # noqa: F401 - 触发注册

    from skill3d.tools.registry import REGISTRY

    full = EvidenceProfile(geometry_3d="available", world_frame="available",
                           metric_scale="available", object_detection="available",
                           track_consensus="available", object_grounding="available")
    broken_geo = full.model_copy(update={"geometry_3d": "unavailable"})
    n_full = len(REGISTRY.names_for_scope(SCOPE_METRIC_ENABLED, evidence_profile=full))
    n_geo = len(REGISTRY.names_for_scope(SCOPE_METRIC_ENABLED, evidence_profile=broken_geo))
    # geometry_3d 失败收回了部分 Tool，但**不是**全部（单项失败只收回依赖它的）
    assert 0 < n_geo < n_full
    # list_objects 只依赖 object_detection → 几何失败不影响它
    names_geo = REGISTRY.names_for_scope(SCOPE_FULL_3D, evidence_profile=broken_geo)
    assert "list_objects" in names_geo
    assert "object_centroid" not in names_geo
    # scope 收窄：full_3d 下米制 Tool 不可见，metric_enabled 下可见
    names_nometric = REGISTRY.names_for_scope(SCOPE_FULL_3D, evidence_profile=full)
    names_metric = REGISTRY.names_for_scope(SCOPE_METRIC_ENABLED, evidence_profile=full)
    assert "camera_object_distance" not in names_nometric
    assert "camera_object_distance" in names_metric
    # 2D-only：只留不依赖深度/位姿/点云/对象的 Tool
    names_2d = REGISTRY.names_for_scope(SCOPE_FALLBACK_2D_ONLY, evidence_profile=full)
    assert "list_objects" not in names_2d
    assert "euclidean_distance" in names_2d


# ------------------------------- 9. degraded 容忍口径（真实 GPU 实测修正）----

def test_geometry_degraded_keeps_3d_tools_but_metric_degraded_does_not():
    """真实 corpus 实测口径：`degraded` 是"能用但有告警"，不是"不能用"。

    背景（本机真实 VGGT 数据，scene 41069043）：`warp_inlier=0.668` /
    `cloud_overlap=0.551` → M4 主门**通过**，但因"相邻帧旋转跳变 43.3°"这条
    **诊断告警**，`geometry_3d` 被判 `degraded`。若不容忍 degraded，
    `relative_direction_of`（方向题唯一可用原语）会被一起隐藏，与 §6.2
    "主门告警未崩 → 仍走 full_3d" 冲突。

    但 `metric_scale` 例外：D3 硬契约要求米制 Tool 只在 gate 通过
    （= `metric_scale == "available"`）时可用，`degraded` 也不放行。
    """
    import skill3d.tools  # noqa: F401 - 触发注册

    from skill3d.tools.registry import REGISTRY

    degraded_geo = EvidenceProfile(
        geometry_3d="degraded", world_frame="degraded", metric_scale="available",
        object_detection="available", track_consensus="available",
        object_grounding="available")
    names = REGISTRY.names_for_scope(SCOPE_METRIC_ENABLED, evidence_profile=degraded_geo)
    assert "relative_direction_of" in names     # 方向题必须仍然可用
    assert "object_centroid" in names
    assert "connectivity_graph" in names

    # metric_scale 降级 → 米制 Tool 一律收回（D3）
    degraded_metric = EvidenceProfile(
        geometry_3d="available", world_frame="available", metric_scale="degraded",
        object_detection="available", track_consensus="available",
        object_grounding="available")
    names2 = REGISTRY.names_for_scope(SCOPE_METRIC_ENABLED,
                                      evidence_profile=degraded_metric)
    for metric_tool in ("camera_object_distance", "object_3d_extent",
                        "plane_fit_room_size"):
        assert metric_tool not in names2, metric_tool
    # 非米制 3D Tool 不受影响（§7.2 单项失败只收回依赖它的）
    assert "object_centroid" in names2 and "relative_direction_of" in names2


def test_metric_tools_never_tolerate_degraded_metric_scale():
    """D3 硬契约：任何 requires `metric_scale` 的 Tool 都不得声明容忍它。"""
    import skill3d.tools  # noqa: F401

    from skill3d.tools.registry import REGISTRY

    for name in REGISTRY.names():
        spec = REGISTRY.spec(name)
        if EVIDENCE_METRIC_SCALE in (spec.requires_evidence or []):
            assert EVIDENCE_METRIC_SCALE not in (spec.tolerates_degraded or []), name


# ------------------- 10. track_consensus 判据（真实数据口径修正）----

def test_track_consensus_uses_fragmentation_not_visibility():
    """`track_consensus` 判据 = 碎片化率 ∧ 重复嫌疑占比，**不是**可见帧率。

    真实数据依据（scene 41069043 / arkitscenes）：可见帧率实测只有 0.14–0.19
    （手持扫描里单个物体本来只在少数帧出现），若拿它当门，`count_objects`
    会结构性不可用 —— counting 题的程序路径直接消失。而 v5 [已实测] 的计数
    失败根因是**重复实例**（清单 12 个重复）与**碎片化**。故改口径。
    """
    from skill3d.reconstruction_gate.evidence_profile import (
        TH_TRACK_FRAGMENTATION_DEGRADED,
        track_capability,
    )


    # 可见帧率很低但 track 干净 → 判 available（旧口径会误判 unavailable）
    clean = M5EvidenceSummary(n_objects=24, n_tracks=24, track_stable_ratio=0.15,
                              track_fragmentation_ratio=0.0,
                              duplicate_suspect_ratio=0.0)
    assert track_capability(clean)[0] == "available"

    # 碎片化/重复偏高 → degraded（计数可用但需带标记）
    messy = M5EvidenceSummary(n_objects=24, n_tracks=24, track_stable_ratio=0.19,
                              track_fragmentation_ratio=0.31,
                              duplicate_suspect_ratio=0.33)
    state, notes = track_capability(messy)
    assert state == "degraded" and notes

    # 碎片化极高 → unavailable（计数不可信）
    broken = M5EvidenceSummary(n_objects=24, n_tracks=24,
                               track_fragmentation_ratio=0.8,
                               duplicate_suspect_ratio=0.6)
    assert track_capability(broken)[0] == "unavailable"

    # 无统计 → unavailable（不伪造）
    assert track_capability(M5EvidenceSummary())[0] == "unavailable"
    assert TH_TRACK_FRAGMENTATION_DEGRADED > 0


def test_count_objects_visible_when_track_consensus_degraded():
    """§9.2：`count_objects` 容忍 track_consensus 降级（带 evidence_degraded 标记）。

    这条是 counting 题程序路径能否存在的前提；degraded 下必须**暴露**而不是隐藏。
    """
    import skill3d.tools  # noqa: F401

    from skill3d.tools.registry import REGISTRY

    p = EvidenceProfile(geometry_3d="available", world_frame="available",
                        metric_scale="available", object_detection="available",
                        track_consensus="degraded", object_grounding="available")
    names = REGISTRY.names_for_scope(SCOPE_FULL_3D, evidence_profile=p)
    assert "count_objects" in names
    flags = degraded_evidence_flags(REGISTRY.spec("count_objects"), p)
    assert "evidence_degraded:track_consensus" in flags


# ------------- 11. prompt 头部：非米制题不得误导模型 abstain（实测修正）----

def test_docs_header_does_not_scare_non_metric_tasks():
    """真实实测修正：非米制题不得打印"米制门未通过 + 缺失子条件"。

    依据（2026-09-21，32 题 inner_validation 真实运行）：header 此前**无条件**打印
    "米制证据门未通过（缺失子条件=[..., 'question_type_is_metric', ...]）"，
    而 `question_type_is_metric` 对非米制题恒 False 是**设计如此**；模型把它读成
    "证据不足 → abstain" → 4 个 rel_direction 全 abstain、4 个 rel_distance 3 个
    abstain，而这两类题的相对几何会把尺度 s 约掉，**根本不需要米制**。
    """
    import skill3d.tools  # noqa: F401

    from skill3d.tools.registry import REGISTRY

    p = EvidenceProfile(geometry_3d="degraded", world_frame="degraded",
                        metric_scale="unavailable", object_detection="available",
                        track_consensus="degraded", object_grounding="available")
    for task in ("object_rel_direction", "object_rel_distance", "route_planning",
                 "obj_appearance_order", "object_counting"):
        h = REGISTRY.docs_header("full_3d", ["depth", "objects"], question_type=task,
                                 gate_passed=False, evidence_profile=p,
                                 gate_missing=["question_type_is_metric",
                                               "scale_fusion_success"])
        assert "米制证据门未通过" not in h, task
        assert "不需要米制尺度" in h, task

    # 米制题仍然必须明确写清门未通过
    h = REGISTRY.docs_header("full_3d", ["depth", "objects"],
                             question_type="object_abs_distance", gate_passed=False,
                             evidence_profile=p, gate_missing=["scale_fusion_success"])
    assert "米制证据门未通过" in h and "scale_fusion_success" in h
    # 门通过时也不得再喊"未通过"
    h2 = REGISTRY.docs_header("full_3d", ["depth", "objects"],
                              question_type="object_abs_distance", gate_passed=True,
                              evidence_profile=p)
    assert "未通过" not in h2 and "已通过" in h2


def test_tool_descriptions_expose_return_shape():
    """工具描述必须给出确切返回结构（实测：模型把 list_objects 的 dict 当字符串用）。

    依据：qa 1905 的程序写 `for obj_id in list_objects(): 'trash' in obj_id.lower()`，
    而 v6 的 `list_objects` 返回 dict 列表 → 运行期失败 → abstain。描述里必须
    直接给可照抄的取值方式。
    """
    import skill3d.tools  # noqa: F401

    from skill3d.tools.registry import REGISTRY

    d = REGISTRY.spec("list_objects").description
    assert "category_name" in d and "dict" in d
    d2 = REGISTRY.spec("count_objects").description
    assert "result['count']" in d2 or "['count']" in d2
    d3 = REGISTRY.spec("relative_direction_of").description
    assert "direction" in d3 and "['direction']" in d3
