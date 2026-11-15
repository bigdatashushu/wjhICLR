"""确定性公共证据门：M4 几何门与尺度门；M5 只提供候选诊断。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    metric_scale_state_from_gate,
)
from skill3d.schemas.reconstruction import METRIC_TASK_TYPES

TH_SCALE_DISPERSION: float = 0.15       # TODO_CALIBRATE
TH_MIN_VALID_FRAME_RATIO: float = 0.75  # TODO_CALIBRATE
EVIDENCE_PROFILE_BUILDER_VERSION = "evidence-builder-v11.6-two-capabilities"


@dataclass
class M5EvidenceSummary:
    """候选生成与去重的实测诊断，不证明候选是题目所指的实例。"""

    m5_ran: bool = False
    detection_fault: bool = False
    n_objects: int = 0
    n_tracks: int = 0
    track_stable_ratio: Optional[float] = None
    track_fragmentation_ratio: Optional[float] = None
    duplicate_suspect_ratio: Optional[float] = None
    detection_scope: str = "question_binding"
    track_scope: str = "question_binding"
    base_inventory: dict = field(default_factory=dict)
    question_supplement: dict = field(default_factory=dict)
    question_inventory: dict = field(default_factory=dict)
    question_binding: dict = field(default_factory=dict)
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
    """前五项确定尺度能力，第六项单独约束米制题型授权。"""
    disp = scale_self_consistency
    fusion_ok = (
        str(fusion_status) == "success"
        and metric_scale is not None
        and np.isfinite(float(metric_scale))
        and float(metric_scale) > 0
        and np.isfinite(float(valid_frame_ratio))
        and float(valid_frame_ratio) >= TH_MIN_VALID_FRAME_RATIO
    )
    consistency_ok = (
        disp is not None and np.isfinite(float(disp))
        and 0 <= float(disp) <= TH_SCALE_DISPERSION
    )
    subs = {
        GATE_SUB_SCENE_ROUTE: str(scene_route) == "full_3d",
        GATE_SUB_M4_MAIN_GATE: bool(main_gate_passed),
        GATE_SUB_FUSION_SUCCESS: bool(fusion_ok),
        GATE_SUB_SCALE_SELF_CONSISTENCY: bool(consistency_ok),
        GATE_SUB_FINITE_INPUTS: bool(inputs_finite),
        GATE_SUB_METRIC_QUESTION: str(question_type) in METRIC_TASK_TYPES,
    }
    values = {
        "scale_self_consistency": (float(disp) if disp is not None
                                   and np.isfinite(float(disp)) else float("nan")),
        "valid_frame_ratio": float(valid_frame_ratio),
        "metric_scale": (float(metric_scale) if metric_scale is not None
                         and np.isfinite(float(metric_scale)) else float("nan")),
        "th_scale_dispersion": TH_SCALE_DISPERSION,
        "th_min_valid_frame_ratio": TH_MIN_VALID_FRAME_RATIO,
    }
    return MetricEvidenceGateResult(
        gate_passed=all(subs.values()), gate_version=gate_version,
        sub_results=subs, values=values,
        missing_subconditions=sorted(k for k, v in subs.items() if not v),
    )


def metric_scale_capability(
    *,
    gate: Optional[MetricEvidenceGateResult],
    fusion_status: str,
    scale_self_consistency: Optional[float],
    valid_frame_ratio: float,
    metric_scale: Optional[float],
) -> CapabilityState:
    """使用已实算门的前五项；未通过的自洽门直接禁用尺度。"""
    return metric_scale_state_from_gate(gate)


def geometry_capability(
    *,
    main_gate_passed: bool,
    quality_status: str,
    overall_quality: Optional[float],
    diagnostic_warnings: Sequence[str] = (),
    conf_warp_monotonic: Optional[bool] = None,
) -> tuple[CapabilityState, list[str]]:
    """主门与有限性决定是否可用，软告警只保留在诊断中。"""
    warns = list(diagnostic_warnings)
    if str(quality_status) != "computed":
        return "unavailable", warns + [f"quality_status={quality_status}"]
    if overall_quality is None or not np.isfinite(float(overall_quality)):
        return "unavailable", warns + ["overall_quality 非有限"]
    if not main_gate_passed:
        return "unavailable", warns + ["M4 主门未通过"]
    if conf_warp_monotonic is False:
        warns.append("conf_warp 单调性自检未通过（conf 仅降权）")
    return "available", warns


def build_evidence_profile(
    *,
    artifact,
    scene_route: str,
    question_type: str,
    gate: Optional[MetricEvidenceGateResult],
    m5: Optional[M5EvidenceSummary] = None,
    metric_scale_override: Optional[CapabilityState] = None,
) -> EvidenceProfile:
    """构建两项二值画像；mock 显式覆盖也必须满足尺度依赖几何。"""
    q = getattr(artifact, "quality", None)
    quality_status = str(getattr(artifact, "quality_status", "not_computed"))
    overall = getattr(q, "overall_quality", None)
    main_ok = bool(getattr(q, "main_gate_passed", False))
    conf_mono = getattr(q, "conf_warp_monotonic", None)
    geom, geom_warns = geometry_capability(
        main_gate_passed=main_ok, quality_status=quality_status,
        overall_quality=overall,
        diagnostic_warnings=list(getattr(q, "diagnostic_warnings", None) or []),
        conf_warp_monotonic=conf_mono,
    )
    if str(scene_route) != "full_3d":
        geom = "unavailable"
        geom_warns.append(f"scene_route={scene_route}")
    metric = (metric_scale_override if metric_scale_override is not None
              else metric_scale_state_from_gate(gate))
    if geom != "available":
        metric = "unavailable"

    fusion_status = str(getattr(artifact, "scale_fusion_status", "not_run"))
    reasons: dict[str, str] = {}
    if geom != "available":
        reasons["geometry_3d"] = ("not_run" if quality_status != "computed"
                                  else "quality_gate_not_passed")
    if metric != "available":
        if geom != "available":
            reasons["metric_scale"] = "invalidated"
        elif fusion_status == "not_run":
            reasons["metric_scale"] = "not_run"
        elif fusion_status == "failed":
            reasons["metric_scale"] = "producer_failed"
        else:
            reasons["metric_scale"] = "quality_gate_not_passed"
    return EvidenceProfile(
        geometry_3d=geom, metric_scale=metric, state_reasons=reasons,
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
            "metric_scale": {
                "fusion_status": fusion_status,
                "metric_scale": getattr(artifact, "metric_scale", None),
                "scale_self_consistency": getattr(
                    artifact, "scale_self_consistency", None),
                "gate_passed": bool(gate.gate_passed) if gate else False,
                "gate_missing": list(gate.missing_subconditions) if gate else [],
                "th_scale_dispersion": TH_SCALE_DISPERSION,
                "th_min_valid_frame_ratio": TH_MIN_VALID_FRAME_RATIO,
            },
            "world_frame_diagnostics": {
                "status": str(getattr(artifact, "world_frame_status", "unavailable")),
                "world_up": getattr(artifact, "world_up", None),
                "handedness": getattr(artifact, "handedness", None),
            },
            "candidate_diagnostics": asdict(m5 or M5EvidenceSummary()),
            "_question": {"question_type": str(question_type)},
            "_conf_warp": {"monotonic": conf_mono},
        },
    )


__all__ = [
    "EVIDENCE_PROFILE_BUILDER_VERSION", "M5EvidenceSummary", "TH_SCALE_DISPERSION",
    "build_evidence_profile", "evaluate_metric_gate", "metric_scale_capability",
]
