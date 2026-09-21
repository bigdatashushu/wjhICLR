"""HC30 / HC33 尺度评估：置信度派生 + **逐题型授权**（v4 单一入口）。

背景（v4 实况，不得误报）：当前生产路径虽能输出尺度数值，但未完成独立标定，
`scale_confidence` 仍应为 `low`；且旧字段口径不自洽（`metric_scale=2.386`,
`relative_ci=21.875` 与 `scale_ci_abs≈32.81m` 不能由同一公式互推）。

本模块把 v4 的三条硬约束落成一个可判定的函数：

- **HC30**：`scale_confidence` 必须由 `已触发锚点 × 锚点冲突 × 经验校准覆盖率 ×
  逐题型授权` 共同派生。**未标定、口径异常、锚点冲突或非有限值一律 low**；
  不得靠常量或放宽门槛提升（`MIN_*` / `CONF_*` 阈值一律 `[TODO_CALIBRATE]`，
  且**校准器缺失时任何阈值都无法把 low 抬起来**）。
- **HC31**：锚点逐个留证（`ScaleAnchorEvidence`），鲁棒融合 + 显式冲突检测。
- **HC33**：尺度按题型授权而非全局摧毁 3D 能力 —— `allowed_metric_tasks` 决定
  哪些米制 Tool/Skill 可用；`low` 只收回米制工具，**不得**把可用的非米制
  `depth/poses/point_cloud/objects` 一并降级。

在线纪律：确定性 numpy + 只读冻结校准器；无 VLM、无 GPT-6（硬约束 1）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

from skill3d.schemas.reconstruction import (
    METRIC_TASK_TYPES,
    ReconstructionArtifact,
    ScaleAnchorEvidence,
)

from .metric_scale import (
    CONF_HIGH_MAX_REL_CI,
    CONF_HIGH_MIN_ANCHORS,
    CONF_MEDIUM_MAX_REL_CI,
    CONF_MEDIUM_MIN_ANCHORS,
    DEFAULT_CONFIDENCE_LEVEL,
    MAX_USABLE_REL_CI,
    MEASUREMENT_REL_NOISE,
    MIN_ACCEPTED_ANCHORS_HIGH,
    MIN_ACCEPTED_ANCHORS_MEDIUM,
    MEDIUM_CALIB_MAX_MEDIAN_REL_ERR,
    MEDIUM_CALIB_MIN_SPEARMAN,
    MEDIUM_VETO_MEDIAN_REL_ERR,
    MEDIUM_VETO_SPEARMAN,
    ScaleAnchor,
    ScaleEstimate,
    _grade_confidence,
    anchor_metric_scale,
    anchor_evidence_of,
    fuse_scale_anchors_robust,
)
from .scale_calibration import (
    CalibrationUnavailable,
    ConformalCalibrator,
    load_calibrator,
)
from .scale_units import (
    TASK_MIN_CONFIDENCE,
    check_ci_rel,
    ci_abs_m,
    confidence_at_least,
    pre_authorized_metric_tasks,
)

Confidence = Literal["high", "medium", "low"]

__all__ = [
    "Confidence",
    "ScaleAssessment",
    "TASK_MIN_CONFIDENCE",
    "apply_scale_assessment",
    "assess_scale",
    "grade_confidence_v4",
    "pre_authorized_metric_tasks",
]


@dataclass(frozen=True)
class ScaleAssessment:
    """尺度评估结果（v4；M3 产出 → 写进 artifact → M4/M7 逐题消费）。"""

    metric_scale: Optional[float] = None
    scale_known: bool = False
    ci_rel: Optional[float] = None
    ci_abs_m: Optional[float] = None
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    confidence: Confidence = "low"
    allowed_metric_tasks: frozenset[str] = frozenset()
    anchors: tuple[ScaleAnchorEvidence, ...] = ()
    conflict: bool = False
    calibration_id: Optional[str] = None
    # v5.1：校准器来源数据集与同源标记（透传到 artifact/trace/manifest）
    calibration_dataset: str = ""
    evaluation_datasets: tuple[str, ...] = ()
    empirical_coverage: Optional[float] = None
    nominal_coverage: Optional[float] = None
    n_accepted_anchors: int = 0
    method: str = ""
    source: str = ""
    reason_codes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_artifact_update(self) -> dict:
        """artifact 写回字段（与 §4.1 v4 字段一一对应）。"""
        return {
            "metric_scale": self.metric_scale,
            "scale_known": self.scale_known,
            "scale_ci_rel": self.ci_rel,
            "scale_ci_abs_m": self.ci_abs_m,
            "scale_confidence_level": self.confidence_level,
            "scale_confidence": self.confidence,
            "scale_anchor_fired": list(self.anchors),
            "scale_conflict": self.conflict,
            "scale_calibration_id": self.calibration_id,
            "scale_calibration_dataset": getattr(self, "calibration_dataset", "") or "",
            "scale_dataset_match": getattr(self, "dataset_match", None),
            "scale_empirical_coverage": self.empirical_coverage,
            "scale_nominal_coverage": self.nominal_coverage,
            "allowed_metric_tasks": set(self.allowed_metric_tasks),
        }

    def as_report(self) -> dict:
        """§7 / §10.2 报告字段（锚点触发率、冲突率、覆盖、逐题型授权）。"""
        return {
            "scale": self.metric_scale,
            "scale_known": self.scale_known,
            "scale_ci_rel": self.ci_rel,
            "scale_ci_abs_m": self.ci_abs_m,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level,
            "calibration_id": self.calibration_id,
            "empirical_coverage": self.empirical_coverage,
            "nominal_coverage": self.nominal_coverage,
            "n_anchors_accepted": self.n_accepted_anchors,
            "n_anchors_fired": len(self.anchors),
            "anchor_fire_rate": (1.0 if self.anchors else 0.0),
            "conflict": self.conflict,
            "allowed_metric_tasks": sorted(self.allowed_metric_tasks),
            "reason_codes": list(self.reason_codes),
            "notes": list(self.notes),
        }


def _calibration_verdict(
    cal: Optional[ConformalCalibrator],
    expected_level: Optional[float] = None,
) -> tuple[bool, list[str]]:
    """冻结校准器是否够格支撑 medium/high（§10.2 升级/否决条件 + §4.1 口径一致）。

    全满足才 True：**置信水平与本次运行一致**（§4.1："必须与校准器一致"）；
    经验覆盖在名义值容差内；中位真实相对误差 <25%；解析 rel_ci 与真实误差
    Spearman ρ>0.5；无否决项。

    置信水平不一致必须否决：校准器按 0.99 标定、却按 0.90 报 CI，属于 HC29 明确
    禁止的"不同置信水平混写"，且会让论文里的 coverage 声明对不上。
    """
    if cal is None:
        return False, ["未加载冻结校准器（§10.2：未标定一律 low）"]
    reasons: list[str] = []
    if expected_level is not None and \
            abs(float(cal.confidence_level) - float(expected_level)) > 1e-9:
        return False, [f"校准器置信水平 {cal.confidence_level} 与本次运行 "
                       f"{expected_level} 不一致（§4.1 要求一致；HC29 禁止混用）"]
    if not cal.coverage_within_tolerance:
        reasons.append(
            f"经验覆盖 {cal.empirical_coverage:.3f} 偏离名义 "
            f"{cal.confidence_level:.2f} 超容差")
    if not np.isfinite(cal.median_rel_error):
        reasons.append("中位真实相对误差未计算")
    elif cal.median_rel_error > MEDIUM_VETO_MEDIAN_REL_ERR:
        reasons.append(f"中位真实相对误差 {cal.median_rel_error:.3f} > "
                       f"{MEDIUM_VETO_MEDIAN_REL_ERR}（否决）")
    elif cal.median_rel_error >= MEDIUM_CALIB_MAX_MEDIAN_REL_ERR:
        reasons.append(f"中位真实相对误差 {cal.median_rel_error:.3f} ≥ "
                       f"{MEDIUM_CALIB_MAX_MEDIAN_REL_ERR}（未达升级门槛）")
    if not np.isfinite(cal.spearman_rho):
        reasons.append("Spearman ρ 未计算")
    elif cal.spearman_rho < MEDIUM_VETO_SPEARMAN:
        reasons.append(f"Spearman ρ {cal.spearman_rho:.3f} < "
                       f"{MEDIUM_VETO_SPEARMAN}（否决）")
    elif cal.spearman_rho <= MEDIUM_CALIB_MIN_SPEARMAN:
        reasons.append(f"Spearman ρ {cal.spearman_rho:.3f} ≤ "
                       f"{MEDIUM_CALIB_MIN_SPEARMAN}（未达升级门槛）")
    if cal.n_calibration < MIN_ACCEPTED_ANCHORS_HIGH:
        reasons.append(f"标定样本数 {cal.n_calibration} 过少")
    return (not reasons), reasons


def grade_confidence_v4(
    *,
    n_accepted: int,
    ci_rel: Optional[float],
    conflict: bool,
    has_plane: bool,
    has_object_anchor: bool,
    calibration: Optional[ConformalCalibrator],
    ci_unit_ok: bool = True,
    confidence_level: Optional[float] = None,
) -> tuple[Confidence, list[str]]:
    """v4 置信度派生（HC30）：`锚点 × 冲突 × 覆盖 × 单位口径` 共同决定。

    任何一项不满足即 **low**（fail-closed）。特别注意：
    - **没有冻结校准器 → 恒 low**，无论锚点多少/CI 多小（"未标定一律 low"）；
    - `scale_ci_rel` 口径异常（百分数/全宽/NaN）→ low（`ci_unit_ok=False`）；
    - 锚点冲突 → low（§3 M3：「不得返回虚假的 medium」）；
    - 高杠杆离群锚点被剔除后**接受锚点数**才是"已触发锚点"的真实计数。
    """
    reasons: list[str] = []
    if not ci_unit_ok:
        return "low", ["scale_ci_rel 口径异常（HC29）：fail-closed 为 low"]
    chk = check_ci_rel(ci_rel)
    if not chk.ok:
        return "low", [f"scale_ci_rel 不可用（{chk.reason_code}）：{chk.detail}"]
    rel = float(chk.value)  # type: ignore[arg-type]
    if not np.isfinite(rel):
        return "low", ["scale_ci_rel 非有限值 → low"]
    if rel > MAX_USABLE_REL_CI:
        reasons.append(f"相对 CI {rel:.3f} > {MAX_USABLE_REL_CI}（CI 过宽）")
    if conflict:
        return "low", ["锚点冲突（scale_conflict=True）→ low（§3 M3）"]
    ok, cal_reasons = _calibration_verdict(calibration, confidence_level)
    if not ok:
        return "low", ["未获经验校准支撑"] + cal_reasons

    if (n_accepted >= CONF_HIGH_MIN_ANCHORS and rel <= CONF_HIGH_MAX_REL_CI
            and has_plane and has_object_anchor and not reasons):
        return "high", reasons
    if (n_accepted >= CONF_MEDIUM_MIN_ANCHORS and rel <= CONF_MEDIUM_MAX_REL_CI
            and not reasons):
        return "medium", reasons
    if not reasons:
        reasons.append(
            f"锚点/CI 未达门槛（accepted={n_accepted}, rel_ci={rel:.3f}；"
            f"medium 需 ≥{CONF_MEDIUM_MIN_ANCHORS} 锚点且 CI ≤{CONF_MEDIUM_MAX_REL_CI}）")
    return "low", reasons


def assess_scale(
    point_map: Optional[np.ndarray] = None,
    c2w_list: Optional[np.ndarray] = None,
    *,
    objects: Optional[Sequence[object]] = None,
    object_points: Optional[dict[str, np.ndarray]] = None,
    depth_maps: Optional[np.ndarray] = None,
    scene_name: str = "",
    seed: int = 0,
    ground_plane: bool = True,
    fixed_scale: Optional[float] = None,
    calibration: Optional[ConformalCalibrator] = None,
    calibration_path: Optional[str] = None,
    calibration_dir: Optional[str] = None,
    evaluation_datasets: Optional[Sequence[str]] = None,
    confidence_level: Optional[float] = DEFAULT_CONFIDENCE_LEVEL,
    scale_estimate: Optional[ScaleEstimate] = None,
) -> ScaleAssessment:
    """v4 尺度评估主入口（M3 SCALE_ESTIMATE 调一次）。

    流程：锚点提取（`anchor_metric_scale`，含地平面/相机高与标准物体先验）→
    鲁棒融合（log 空间 Huber M-estimator）→ 冲突检测 → **冻结 conformal 校准** →
    v4 置信度派生 → 逐题型预授权。

    `calibration_path` / `calibration_dir` / `calibration` 三选一：给路径则在线只读加载，
    加载失败**不抛异常**，而是降级为 `calibration=None`（→ 恒 low）。
    `calibration_dir` 为 v5.1 多校准器目录：按 `evaluation_datasets` 选同源校准器
    （`<dir>/<dataset>.json`），找不到再退回该目录下的 `calibrator.json`。
    """
    est = scale_estimate if scale_estimate is not None else anchor_metric_scale(
        point_map, c2w_list, objects=objects, object_points=object_points,
        depth_maps=depth_maps, scene_name=scene_name, seed=seed,
        ground_plane=ground_plane, fixed_scale=fixed_scale)
    # 校准器自带 confidence_level；显式传入则以调用方为准（必须与校准器一致）
    if confidence_level is None:
        confidence_level = (calibration.confidence_level if calibration is not None
                            else DEFAULT_CONFIDENCE_LEVEL)

    notes: list[str] = list(est.notes)
    reason_codes: list[str] = []

    if fixed_scale is not None:
        # 消融档：固定尺度不产生任何锚点证据 → 不得授权任何米制题型
        return ScaleAssessment(
            metric_scale=est.scale, scale_known=est.scale_known,
            ci_rel=None, ci_abs_m=None, confidence_level=confidence_level,
            confidence="low", allowed_metric_tasks=frozenset(),
            anchors=(), conflict=False, calibration_id=None,
            method=est.method, source=est.scale_source,
            reason_codes=("fixed_scale_ablation_no_anchor_evidence",),
            notes=tuple(notes))

    if calibration is None and calibration_dir:
        from .scale_calibration import load_calibrator_for

        datasets = list(evaluation_datasets or [])
        _cal, _notes = load_calibrator_for(datasets, calibration_dir,
                                          fallback=Path(calibration_dir) / "calibrator.json"
                                          if Path(calibration_dir).is_dir() else None)
        calibration = _cal
    if calibration is None and calibration_path:
        try:
            calibration = load_calibrator(calibration_path)
        except CalibrationUnavailable as exc:
            notes.append(f"冻结校准器不可用 → 一律 low（HC30）：{exc}")
            reason_codes.append("calibration_unavailable")
            calibration = None

    fusion = fuse_scale_anchors_robust(est.anchors) if est.anchors else None
    evidence: list[ScaleAnchorEvidence] = (
        anchor_evidence_of(est.anchors, fusion) if est.anchors else [])
    notes.extend(fusion.notes if fusion is not None else [])

    scale = est.scale
    ci_rel: Optional[float] = None
    ci_rel_source = "none"
    if fusion is not None and fusion.ok:
        scale = fusion.scale
        ci_rel = fusion.ci_rel
        ci_rel_source = "analytic_robust_fusion"
        if calibration is not None:
            # conformal 校准只加宽不收窄：得到有经验覆盖保证的半宽
            _, calibrated = calibration.apply(scale, ci_rel)
            if calibrated is not None and np.isfinite(calibrated):
                ci_rel = float(calibrated)
                ci_rel_source = "conformal_calibrated"
    elif scale is not None and calibration is not None:
        # 融合不可用但 v3 路径给了 scale（例如单锚点且 rel_ci 来自估计器）
        _, calibrated = calibration.apply(scale, est.scale_rel_ci)
        if calibrated is not None and np.isfinite(calibrated):
            ci_rel = float(calibrated)
            ci_rel_source = "conformal_calibrated"

    chk = check_ci_rel(ci_rel, confidence_level=confidence_level)
    ci_unit_ok = chk.ok
    if not chk.ok and ci_rel is not None:
        reason_codes.append(chk.reason_code)

    has_plane = any(a.kind == "ground_plane_camera_height" for a in est.anchors)
    has_object = any(a.kind == "object_prior" for a in est.anchors)
    n_accepted = fusion.n_accepted if fusion is not None else len(est.anchors)
    conflict = bool(fusion.conflict) if fusion is not None else False

    confidence, conf_reasons = grade_confidence_v4(
        n_accepted=n_accepted, ci_rel=chk.value, conflict=conflict,
        has_plane=has_plane, has_object_anchor=has_object,
        calibration=calibration, ci_unit_ok=ci_unit_ok,
        confidence_level=confidence_level)
    notes.extend(conf_reasons)
    reason_codes.extend(f"low:{r}" for r in conf_reasons[:3])

    allowed = pre_authorized_metric_tasks(confidence, ci_rel=chk.value)
    if confidence == "low":
        reason_codes.append("metric_tasks_withdrawn")

    scale_out = scale if (scale is not None and np.isfinite(float(scale))
                          and float(scale) > 0) else None
    ci_rel_out = chk.value if (chk.ok and scale_out is not None) else None
    ci_abs_out = ci_abs_m(scale_out, ci_rel_out)

    return ScaleAssessment(
        metric_scale=scale_out,
        scale_known=bool(scale_out is not None),
        ci_rel=ci_rel_out,
        ci_abs_m=ci_abs_out,
        confidence_level=float(confidence_level),
        confidence=confidence,
        allowed_metric_tasks=allowed,
        anchors=tuple(evidence),
        conflict=conflict,
        calibration_id=(calibration.calibration_id if calibration is not None else None),
        calibration_dataset=(str(getattr(calibration, "calibration_dataset", "") or "")
                             if calibration is not None else ""),
        evaluation_datasets=(tuple(getattr(calibration, "evaluation_datasets", ()) or ())
                             if calibration is not None else
                             tuple(str(d) for d in (evaluation_datasets or ()))),
        empirical_coverage=(calibration.empirical_coverage
                            if calibration is not None else None),
        nominal_coverage=(calibration.confidence_level
                          if calibration is not None else None),
        n_accepted_anchors=int(n_accepted),
        method=f"{est.method} [{ci_rel_source}]",
        source=est.scale_source,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        notes=tuple(notes),
    )


def apply_scale_assessment(artifact: ReconstructionArtifact,
                           assessment: ScaleAssessment,
                           *, method_suffix: str = "") -> ReconstructionArtifact:
    """把评估写回 artifact（不可变；写前自洽断言，HC29）。

    `method_suffix` 用于附加**本条 route 的执行说明**（如 BA 的生效/跳过原因）——
    `scale_method` 是唯一记录这些事实的字段，调用方必须显式传入，
    否则 BA 的 skip_reason 会被覆盖丢失（PoC 审计需要它）。
    

    **写前调用 `assert_ci_consistent`**：生产端保证自洽，消费端（schema 校验器）
    才有资格只做兜底降级而不是抛异常。
    """
    from .scale_units import assert_ci_consistent, ci_consistency_status

    update = dict(assessment.to_artifact_update())
    update["scale_method"] = (assessment.method + (f" {method_suffix}" if method_suffix
                                                  else "")).strip()
    update["scale_source"] = assessment.source
    status = ci_consistency_status(update.get("metric_scale"),
                                   update.get("scale_ci_rel"),
                                   update.get("scale_ci_abs_m"))
    if status == "uncalibrated":
        # v5 HC30：有尺度点估计但**没有可用区间**（缺冻结校准器 / 融合不可用）
        # → fail-closed 为 low + 清空逐题授权，**不得**让重建失败，也不得
        # 携带任何 CI 声明（否则会被误读为"已有经验覆盖保证"）。
        update["scale_confidence"] = "low"
        update["allowed_metric_tasks"] = set()
        update["scale_ci_rel"] = None
        update["scale_ci_abs_m"] = None
        update["scale_method"] = (
            update["scale_method"] + " [uncalibrated: 无可用 CI → low（HC30）]").strip()
    assert_ci_consistent(update.get("metric_scale"), update.get("scale_ci_rel"),
                         update.get("scale_ci_abs_m"))
    new_art = artifact.model_copy(update=update)
    # model_copy 不重跑校验器：这里显式复核 HC29 不变量（防回归）
    from skill3d.schemas.reconstruction import _ci_inconsistency_reason

    bad = _ci_inconsistency_reason(new_art.model_dump())
    if bad:  # pragma: no cover - 生产端自洽断言已挡住
        # HC31：同样不把"口径降级"写成 scale_conflict（该字段专属锚点冲突）
        new_art = new_art.model_copy(update={
            "scale_confidence": "low", "allowed_metric_tasks": set()})
    return new_art
