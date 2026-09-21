"""EvidenceProfile 构建 + MetricEvidenceGate（v6 §5.4/§5.5、§7、§13）。

本模块是 v6 的**证据判定单一事实源**：

- `evaluate_metric_gate(...)` 实现 §13.1 的 6 项子条件（确定性组件，非 VLM 裁决）；
- `build_evidence_profile(...)` 把 M3/M4/M5 的事实压成 8 项能力 × 三值；
- `metric_scale` 这一项由 gate 的前 5 项 + 融合结果决定（§13.8：它是**分项能力门**，
  不是全局 gate）；第 6 项（题型是米制）由 `question_tool_scope` 表达。

纪律：所有边界阈值 `[TODO_CALIBRATE]`；任何"证据不足"一律落 `unavailable`
（fail-closed），不做乐观默认。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from skill3d.schemas.evidence import (
    GATE_SUB_FINITE_INPUTS,
    GATE_SUB_FUSION_SUCCESS,
    GATE_SUB_M4_MAIN_GATE,
    GATE_SUB_METRIC_QUESTION,
    GATE_SUB_SCALE_SELF_CONSISTENCY,
    GATE_SUB_SCENE_ROUTE,
    GATE_VERSION,
    CapabilityState,
    EvidenceProfile,
    MetricEvidenceGateResult,
)
from skill3d.schemas.reconstruction import METRIC_TASK_TYPES

# ---- 阈值（全部 TODO_CALIBRATE）----
TH_TRACK_STABLE_RATIO: float = 0.6     # TODO_CALIBRATE: track 稳定占比（available 下限）
TH_TRACK_MIN_RATIO: float = 0.2        # TODO_CALIBRATE: track 稳定占比（degraded 下限）
TH_DETECTION_SPARSE: int = 3           # TODO_CALIBRATE: 检出对象数低于此视为稀疏
TH_GROUNDING_MIN_CONF: float = 0.5     # TODO_CALIBRATE: 补绑置信度下限
TH_SCALE_DISPERSION: float = 0.15      # TODO_CALIBRATE: τ_scale_disp（§13.1 子条件 4）
TH_MIN_VALID_FRAME_RATIO: float = 0.75  # TODO_CALIBRATE: τ_frames（§13.1 子条件 3）

EVIDENCE_PROFILE_BUILDER_VERSION: str = "evidence-builder-v6"

# 记录 world_frame 的候选状态命名（诊断用）
WORLD_FRAME_STATES = ("available", "degraded", "unavailable")


@dataclass
class M5EvidenceSummary:
    """M5 对象绑定对证据画像的输入（由在线链填充；字段缺失 = 证据不足 → fail-closed）。"""

    detection_fault: bool = False          # 检测器服务故障
    n_objects: int = 0                     # 基础清单对象数
    n_tracks: int = 0                      # 去重后 track 数
    track_stable_ratio: Optional[float] = None  # 稳定 track 占比（跨帧存活）
    grounding_pointed_hit: Optional[bool] = None  # 题面点名物是否命中
    grounding_conf: Optional[float] = None
    grounding_miss: bool = False           # 明确未命中（grounding_recall_miss）
    grounding_filled: bool = False         # 走的是 question-targeted 补绑
    notes: list[str] = field(default_factory=list)


def evaluate_metric_gate(
    *,
    scene_route: str,
    main_gate_passed: bool,
    fusion_status: str,
    scale_self_consistency: Optional[float],
    valid_frame_ratio: float,
    metric_scale: Optional[float],
    inputs_finite: bool = True,
    question_type: str = "",
    gate_version: str = GATE_VERSION,
) -> MetricEvidenceGateResult:
    """§13.1 六项子条件（全为真才 `gate_passed=True`）。

    逐项：

    1. `scene_route == "full_3d"`；
    2. M4 主门通过（warp 内点率 ≥ τ_warp **且** 分组点云重叠率 ≥ τ_cloud）；
    3. 融合成功（`s_global` 有限、有效帧占比 ≥ τ_frames）；
    4. 尺度自洽（32 帧 `s_k` 的 `std/median` ≤ τ_scale_disp）；
    5. 输入/内参/坐标/数值全部有限（无 NaN/Inf）；
    6. `question_type ∈ {object_abs_distance, object_size_estimation, room_size_estimation}`。

    `missing_subconditions` 列出未过的子条件名（§13.3 prompt 头部要写清楚）。
    """
    disp = scale_self_consistency
    fusion_ok = (
        str(fusion_status) == "success"
        and metric_scale is not None
        and np.isfinite(float(metric_scale))
        and float(metric_scale) > 0
        and float(valid_frame_ratio) >= TH_MIN_VALID_FRAME_RATIO
    )
    consistency_ok = (
        disp is not None and np.isfinite(float(disp))
        and float(disp) <= TH_SCALE_DISPERSION
    )
    subs: dict[str, bool] = {
        GATE_SUB_SCENE_ROUTE: str(scene_route) == "full_3d",
        GATE_SUB_M4_MAIN_GATE: bool(main_gate_passed),
        GATE_SUB_FUSION_SUCCESS: bool(fusion_ok),
        GATE_SUB_SCALE_SELF_CONSISTENCY: bool(consistency_ok),
        GATE_SUB_FINITE_INPUTS: bool(inputs_finite),
        GATE_SUB_METRIC_QUESTION: str(question_type) in METRIC_TASK_TYPES,
    }
    values: dict[str, float] = {
        "scale_self_consistency": (float(disp) if disp is not None
                                   and np.isfinite(float(disp)) else float("nan")),
        "valid_frame_ratio": float(valid_frame_ratio),
        "metric_scale": (float(metric_scale) if metric_scale is not None
                         and np.isfinite(float(metric_scale)) else float("nan")),
        "th_scale_dispersion": TH_SCALE_DISPERSION,
        "th_min_valid_frame_ratio": TH_MIN_VALID_FRAME_RATIO,
    }
    missing = sorted(k for k, v in subs.items() if not v)
    return MetricEvidenceGateResult(
        gate_passed=all(subs.values()),
        gate_version=gate_version,
        sub_results=subs,
        values=values,
        missing_subconditions=missing,
    )


def metric_scale_capability(
    *,
    gate: Optional[MetricEvidenceGateResult],
    fusion_status: str,
    scale_self_consistency: Optional[float],
    valid_frame_ratio: float,
    metric_scale: Optional[float],
) -> CapabilityState:
    """§5.5：gate 结果 → `EvidenceProfile.metric_scale` 的三值。

    - 全过 → `available`；
    - **融合成功但自洽松** → `degraded`（米制 Skill 可在 `degraded` 签名下积累）；
    - 融合失败 / 有限值不过 → `unavailable`。

    注意只用 gate 的**前 5 项**（题型子条件由 `question_tool_scope` 表达），
    这样同一 scene 在不同题型下的 `metric_scale` 能力值保持一致（§7.1：scene 级）。
    """
    if gate is None:
        return "unavailable"
    subs = gate.sub_results or {}
    prefix_ok = all(bool(subs.get(k, False)) for k in (
        GATE_SUB_SCENE_ROUTE, GATE_SUB_M4_MAIN_GATE, GATE_SUB_FUSION_SUCCESS,
        GATE_SUB_FINITE_INPUTS))
    if prefix_ok and bool(subs.get(GATE_SUB_SCALE_SELF_CONSISTENCY, False)):
        return "available"
    fusion_ok = (str(fusion_status) == "success" and metric_scale is not None
                 and np.isfinite(float(metric_scale)) and float(metric_scale) > 0)
    consistency_ok = (scale_self_consistency is not None
                      and np.isfinite(float(scale_self_consistency))
                      and float(scale_self_consistency) <= TH_SCALE_DISPERSION)
    if prefix_ok and fusion_ok and not consistency_ok:
        return "degraded"
    return "unavailable"


def world_frame_capability(
    *,
    world_frame_status: str,
    world_up: Optional[Sequence[float]],
    handedness: Optional[str],
) -> CapabilityState:
    """§7.1 `world_frame` 三值：估计成功 / 置信低 / 位姿不可用。"""
    st = str(world_frame_status or "unavailable")
    if st == "available" and world_up is not None and handedness is not None:
        return "available"
    if st == "degraded" and handedness is not None:
        return "degraded"
    return "unavailable"


def geometry_capability(
    *,
    main_gate_passed: bool,
    quality_status: str,
    overall_quality: Optional[float],
    diagnostic_warnings: Sequence[str] = (),
    conf_warp_monotonic: Optional[bool] = None,
) -> tuple[CapabilityState, list[str]]:
    """§7.1 `geometry_3d` 三值：主门通过 / 一项告警但未崩 / 严重失败。

    返回 `(状态, 告警列表)`。**质量未计算 / NaN 一律 unavailable**（fail-closed，
    对应 §6.2"不得停在 full_3d"）。
    """
    warns = list(diagnostic_warnings)
    if str(quality_status) != "computed":
        return "unavailable", warns + [f"quality_status={quality_status}"]
    if overall_quality is None or not np.isfinite(float(overall_quality)):
        return "unavailable", warns + ["overall_quality 非有限"]
    if not main_gate_passed:
        return "unavailable", warns + ["M4 主门未通过"]
    if conf_warp_monotonic is False:
        # conf 不单调 → 只降权（软权重），不否决几何
        warns.append("conf_warp 单调性自检未通过（conf 仅降权）")
    return ("degraded" if warns else "available"), warns


def detection_capability(summary: M5EvidenceSummary) -> tuple[CapabilityState, list[str]]:
    """§7.1 `object_detection`：服务正常+有检出 / 检出稀疏 / 服务故障或零检出。"""
    notes: list[str] = []
    if summary.detection_fault:
        return "unavailable", ["detector_fault"]
    if int(summary.n_objects) <= 0:
        return "unavailable", ["zero_detection_after_retry"]
    if int(summary.n_objects) < TH_DETECTION_SPARSE:
        notes.append(f"检出稀疏（{summary.n_objects} < {TH_DETECTION_SPARSE}）")
        return "degraded", notes
    return "available", notes


def track_capability(summary: M5EvidenceSummary) -> tuple[CapabilityState, list[str]]:
    """§7.1 `track_consensus`：稳定 track 占比高 / 中 / 几乎无 track。"""
    r = summary.track_stable_ratio
    if int(summary.n_tracks) <= 0 or r is None or not np.isfinite(float(r)):
        return "unavailable", ["no_track_statistics"]
    r = float(r)
    if r >= TH_TRACK_STABLE_RATIO:
        return "available", []
    if r >= TH_TRACK_MIN_RATIO:
        return "degraded", [f"track 稳定占比偏低（{r:.2f}）"]
    return "unavailable", [f"track 稳定占比过低（{r:.2f}）"]


def grounding_capability(summary: M5EvidenceSummary) -> CapabilityState:
    """题级 `object_grounding`：点名物命中 / 有候选未命中 / 明确未命中。"""
    if summary.grounding_miss:
        return "unavailable"
    if summary.grounding_pointed_hit is None:
        # 未走题面补绑（题目不点具体物件）→ 该能力对该题不适用，
        # 记为 available（基础清单已覆盖），并在 subvalues 里留痕
        return "available"
    if summary.grounding_pointed_hit:
        conf = summary.grounding_conf
        if conf is not None and np.isfinite(float(conf)) \
                and float(conf) < TH_GROUNDING_MIN_CONF:
            return "degraded"
        return "available"
    return "degraded" if summary.grounding_filled else "unavailable"


def build_evidence_profile(
    *,
    artifact,
    scene_route: str,
    question_type: str,
    gate: Optional[MetricEvidenceGateResult],
    m5: Optional[M5EvidenceSummary] = None,
    metric_scale_override: Optional[CapabilityState] = None,
) -> EvidenceProfile:
    """构建 8 项能力 × 三值的 EvidenceProfile（§5.4/§7.1）。

    `metric_scale_override` 供 mock_light/合成路径显式指定（真实路径一律走 gate）。
    `temporal` / `image_2d` 恒 `available`（统一 FrameSet 存在即成立，§5.4）。
    """
    m5 = m5 or M5EvidenceSummary()
    q = getattr(artifact, "quality", None)
    quality_status = str(getattr(artifact, "quality_status", "not_computed"))
    overall = getattr(q, "overall_quality", None)
    main_ok = bool(getattr(q, "main_gate_passed", False))
    diag = list(getattr(q, "diagnostic_warnings", None) or [])
    conf_mono = getattr(q, "conf_warp_monotonic", None)

    geom, geom_warns = geometry_capability(
        main_gate_passed=main_ok, quality_status=quality_status,
        overall_quality=overall, diagnostic_warnings=diag,
        conf_warp_monotonic=conf_mono)
    if str(scene_route) != "full_3d" and geom == "available":
        # §6.2：主门通过但 route 被输入质量压到 2D-only → 几何能力不得报 available
        geom = "degraded"
        geom_warns.append(f"scene_route={scene_route}")

    wf = world_frame_capability(
        world_frame_status=str(getattr(artifact, "world_frame_status", "unavailable")),
        world_up=getattr(artifact, "world_up", None),
        handedness=getattr(artifact, "handedness", None))

    det, det_warns = detection_capability(m5)
    trk, trk_warns = track_capability(m5)
    grd = grounding_capability(m5)

    if metric_scale_override is not None:
        metric = metric_scale_override
    else:
        valid_ratio = float(gate.values.get("valid_frame_ratio", 0.0)) if gate else 0.0
        metric = metric_scale_capability(
            gate=gate,
            fusion_status=str(getattr(artifact, "scale_fusion_status", "not_run")),
            scale_self_consistency=getattr(artifact, "scale_self_consistency", None),
            valid_frame_ratio=valid_ratio,
            metric_scale=getattr(artifact, "metric_scale", None))

    return EvidenceProfile(
        geometry_3d=geom,
        world_frame=wf,
        metric_scale=metric,
        object_detection=det,
        track_consensus=trk,
        temporal="available",
        image_2d="available",
        object_grounding=grd,
        producer_subvalues={
            "_builder": {"version": EVIDENCE_PROFILE_BUILDER_VERSION},
            "geometry_3d": {
                "main_gate_passed": main_ok,
                "warp_inlier_ratio": getattr(q, "warp_inlier_ratio", None),
                "warp_photometric_inlier_ratio": getattr(
                    q, "warp_photometric_inlier_ratio", None),
                "cloud_overlap_ratio": getattr(q, "cloud_overlap_ratio", None),
                "overall_quality": overall,
                "warnings": geom_warns,
            },
            "world_frame": {
                "status": str(getattr(artifact, "world_frame_status", "unavailable")),
                "up_consistency": None,
            },
            "metric_scale": {
                "fusion_status": str(getattr(artifact, "scale_fusion_status", "not_run")),
                "metric_scale": getattr(artifact, "metric_scale", None),
                "scale_self_consistency": getattr(
                    artifact, "scale_self_consistency", None),
                "gate_passed": bool(gate.gate_passed) if gate else False,
                "gate_missing": list(gate.missing_subconditions) if gate else [],
                "th_scale_dispersion": TH_SCALE_DISPERSION,
                "th_min_valid_frame_ratio": TH_MIN_VALID_FRAME_RATIO,
            },
            "object_detection": {"n_objects": int(m5.n_objects),
                                 "warnings": det_warns + list(m5.notes)},
            "track_consensus": {"n_tracks": int(m5.n_tracks),
                                "track_stable_ratio": m5.track_stable_ratio,
                                "warnings": trk_warns},
            "object_grounding": {
                "pointed_hit": m5.grounding_pointed_hit,
                "conf": m5.grounding_conf,
                "filled": m5.grounding_filled,
                "miss": m5.grounding_miss,
            },
            "_question": {"question_type": str(question_type)},
            "_conf_warp": {"monotonic": conf_mono},
        },
    )


__all__ = [
    "EVIDENCE_PROFILE_BUILDER_VERSION",
    "M5EvidenceSummary",
    "TH_SCALE_DISPERSION",
    "build_evidence_profile",
    "evaluate_metric_gate",
    "metric_scale_capability",
]
