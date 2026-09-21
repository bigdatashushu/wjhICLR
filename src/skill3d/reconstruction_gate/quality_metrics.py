"""M4 G1-G11 重建/输入质量指标计算（§10 指标表）。

阈值全部 TODO_CALIBRATE。能用 numpy/cv2 真实算的指标真实计算；
依赖外部产物（重投影残差、动态 mask、跟踪 IoU、基线、尺度 CI）的指标
从传入数据/ReconstructionArtifact 读取，缺失时给 NaN 并注释 TODO。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

from skill3d.gates import iqa
from skill3d.gates.input_gate import (
    MIN_FRAMES,
    TH_BLUR_VAR,
    TH_OVER_EXPOSED,
    TH_UNDER_EXPOSED,
    blur_floor_for,
)
from skill3d.schemas.reconstruction import QualityMetrics, ReconstructionArtifact

# ---- 阈值常量（全部 TODO_CALIBRATE，起始参考值见 §10）----
TH_G3_MOTION: float = 20.0        # TODO_CALIBRATE: G3 运动模糊光流幅值（px）
TH_G5_MEDIAN: float = 2.0         # TODO_CALIBRATE: G5 重投影残差 median（px）
TH_G5_P95: float = 5.0            # TODO_CALIBRATE: G5 重投影残差 p95（px）
TH_G6_DEPTH_CV: float = 1.0       # TODO_CALIBRATE: G6 深度方差系数 σ/μ
TH_G7_DYNAMIC: float = 0.3        # TODO_CALIBRATE: G7 动态物体占比
TH_G9_TRACK_IOU: float = 0.5      # TODO_CALIBRATE: G9 跟踪一致性 IoU
TH_G10_BASELINE: float = 0.1      # TODO_CALIBRATE: G10 基线质量（归一化）
TH_G11_SCALE_CI: float = 0.1      # TODO_CALIBRATE: G11 尺度**相对** CI 半宽（分数，HC29）

# v5 活动指标集合（附录 A 聚合不变量 / HC37/38）：G5 是**条件项**，G8 已永久退役。
# `overall_quality` 只能由本集合（+ 满足条件的 G5）计算；缺失项**不得**补默认分、
# 补历史值或沿用旧 overall_quality。
QUALITY_METRIC_VERSION: str = "v5-no-g8-g5-optional"
ALWAYS_ACTIVE_METRICS: tuple[str, ...] = (
    "g1_blur_ok", "g2_brightness", "g3_motion_blur", "g4_frame_count",
    "g6_depth_var_coeff", "g7_dynamic_ratio", "g9_tracker_consistency",
    "g10_baseline_quality", "g11_scale_ci",
)
CONDITIONAL_METRICS: tuple[str, ...] = ("g5_reproj_err_median", "g5_reproj_err_p95")
RETIRED_METRICS: tuple[str, ...] = ("g8_bbox_coverage_min",)

NaN = float("nan")


def _finite_or_none(v: Optional[float]) -> Optional[float]:
    """v5 G5 口径：NaN/Inf 一律归一为 `None`（"未计算"，不是"坏值"）。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def g1_blur_ok(frames: Sequence[np.ndarray]) -> float:
    """G1 帧模糊：通过**双判据**的帧占比，[0,1]（Appendix A / B-3）。

    双判据 = 绝对下界（`TH_BLUR_VAR_ABS`）与 episode 相对判据
    （`< 本 episode 中位 × TH_BLUR_REL`）取较严者；与 M2 同一实现
    （`gates.input_gate.blur_floor_for`，单一事实源），不重复一份阈值逻辑。
    实机实测：跨数据集 Laplacian 中位数相差近 10×（arkitscenes≈39 / scannetpp≈315），
    单靠绝对阈值不可用，故 G1 必须带 episode 相对判据。
    """
    if not frames:
        return NaN
    blurs = [iqa.laplacian_var(f) for f in frames]
    floor = blur_floor_for(blurs)
    ok = sum(b >= floor for b in blurs)
    return ok / len(frames)


def g2_brightness(frames: Sequence[np.ndarray]) -> float:
    """G2 曝光：曝光正常帧占比，[0,1]。"""
    if not frames:
        return NaN
    ok = 0
    for f in frames:
        p_over, p_under = iqa.exposure_ratios(f)
        if p_over <= TH_OVER_EXPOSED and p_under <= TH_UNDER_EXPOSED:
            ok += 1
    return ok / len(frames)


def g3_motion_blur(frames: Sequence[np.ndarray],
                   depth_shape: Optional[tuple[int, int]] = None) -> float:
    """G3 运动模糊：帧间光流幅值均值（px），越小越好。

    **光流必须在 VGGT-depth-grid 上计算**（§9 / C-8 / Appendix A G3）：
    在原始帧上算光流、再拿深度网格坐标去取值会越界。`depth_shape` 给定 (H, W) 时
    先把每帧缩放到深度网格再算光流；未给定时退化为原始分辨率（并已在返回值语义上
    标记为同一量纲），调用方应尽量传深度网格尺寸。
    """
    if len(frames) < 2:
        return NaN
    if depth_shape is None:
        mags = [iqa.motion_score(frames[i - 1], frames[i]) for i in range(1, len(frames))]
        return float(np.mean(mags))
    from skill3d import coords

    mags = []
    for i in range(1, len(frames)):
        flow = coords.optical_flow_on_depth_grid([frames[i - 1], frames[i]], depth_shape)
        mags.append(float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))))
    return float(np.mean(mags))


def g4_frame_count(frames: Sequence) -> int:
    """G4 帧数完整性：实际可用帧数。"""
    return len(frames)


def g5_reproj_err(reproj_errors: Optional[np.ndarray],
                  *, status: str = "not_available") -> tuple[Optional[float], Optional[float]]:
    """G5 重投影残差 median/p95（px）；**只有真 BA 才允许有值**（HC37）。

    - `status != "computed"` → 一律 `(None, None)`：正式 `vggt` 主线写
      `reprojection_status="not_available"`，禁止用 `depth_conf`/`point_conf`/
      pose smoothing 等代理值冒充重投影残差；
    - `status == "computed"` 但残差缺失/全非有限 → 同样 `(None, None)`
      （由调用方把 `reprojection_status` 降为 `failed`，不得补默认分）。
    """
    if status != "computed" or reproj_errors is None or len(reproj_errors) == 0:
        return None, None
    e = np.asarray(reproj_errors, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return None, None
    return float(np.median(e)), float(np.percentile(e, 95))


def g6_depth_var_coeff(depth_maps: np.ndarray) -> float:
    """G6 深度方差系数 σ/μ（无量纲），跨全部帧的全局统计。

    异常记深度漂移；深度无效（全 0/NaN）时返回 NaN。
    """
    d = np.asarray(depth_maps, dtype=np.float64)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return NaN
    mu = float(np.mean(d))
    if mu <= 0:
        return NaN
    return float(np.std(d) / mu)


def g7_dynamic_ratio(dynamic_masks: Optional[np.ndarray]) -> float:
    """G7 动态物体占比：动态 mask 像素占比。

    TODO: 动态 mask 来自 M5 SAM2；无数据时 NaN。
    """
    if dynamic_masks is None:
        return NaN
    m = np.asarray(dynamic_masks)
    if m.size == 0:
        return NaN
    return float(np.mean(m.astype(bool)))


def g9_tracker_consistency(track_ious: Optional[Sequence[float]]) -> float:
    """G9 跟踪一致性：SAM2 mask 跨帧 IoU 均值，[0,1]。

    TODO: IoU 来自 M5 SAM2 传播；无数据时 NaN。
    """
    if not track_ious:
        return NaN
    return float(np.mean(track_ious))


def g10_baseline_quality(c2w_list: Optional[np.ndarray]) -> float:
    """G10 基线质量：相机基线相对场景尺度的覆盖（归一化）。

    真实可算部分：由 c2w 平移分量估计相邻帧基线中位数 / 场景直径。
    无位姿数据时 NaN（TODO: 需 VGGT/COLMAP 位姿产物）。
    """
    if c2w_list is None:
        return NaN
    c2w = np.asarray(c2w_list, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[-2:] != (4, 4) or len(c2w) < 2:
        return NaN
    centers = c2w[:, :3, 3]
    baselines = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    diameter = float(np.max(np.linalg.norm(
        centers[None, :, :] - centers[:, None, :], axis=-1)))
    if diameter <= 0:
        return NaN
    return float(np.median(baselines) / diameter)


def g11_scale_ci(scale_ci: Optional[float]) -> float:
    """G11 尺度 CI：**相对** CI 半宽（分数口径，HC29）；未知时 NaN。

    v4 口径变更（§0.2 HC29 + 附录 A G11 行）：G11 记录的是 `scale_ci_rel`
    （无量纲分数），不再是旧公式的绝对米制半宽。阈值 `TH_G11_SCALE_CI=0.1` 因此
    读作"相对半宽 ≤10%"。旧绝对量一律经 `reconstruction.scale_units` 迁移，
    且迁移结果不可用于准入。
    """
    if scale_ci is None:
        return NaN
    return float(scale_ci)


def _norm_scores(q: dict) -> list[float]:
    """将各指标归一化为 [0,1] 的"越高越好"分数。

    v5 聚合不变量（附录 A / HC37/38）：
    - **G5 是条件项**：仅当 `q["g5_computed"]` 为真（= `reprojection_status=="computed"`
      且两个标量都有限）才进入分母；否则直接从分母排除，**不补 1.0/历史值/代理值**；
    - **G8 不参与**（永久退役）；
    - 其余活动指标在数据缺失时记 NaN 并跳过（NaN ≠ 0 分，也不冒充满分）。
    """
    scores = []
    for v in (q["g1_blur_ok"], q["g2_brightness"]):
        if np.isfinite(v):
            scores.append(v)
    if np.isfinite(q["g3_motion_blur"]):
        scores.append(1.0 if q["g3_motion_blur"] <= TH_G3_MOTION else 0.0)
    scores.append(1.0 if q["g4_frame_count"] >= MIN_FRAMES else q["g4_frame_count"] / MIN_FRAMES)
    for v, th in ((q["g6_depth_var_coeff"], TH_G6_DEPTH_CV),
                  (q["g7_dynamic_ratio"], TH_G7_DYNAMIC)):
        if np.isfinite(v):
            scores.append(1.0 if v <= th else 0.0)
    if q.get("g5_computed"):
        if np.isfinite(q["g5_reproj_err_median"]):
            scores.append(1.0 if q["g5_reproj_err_median"] <= TH_G5_MEDIAN else 0.0)
        if np.isfinite(q["g5_reproj_err_p95"]):
            scores.append(1.0 if q["g5_reproj_err_p95"] <= TH_G5_P95 else 0.0)
    for v, th in ((q["g9_tracker_consistency"], TH_G9_TRACK_IOU),
                  (q["g10_baseline_quality"], TH_G10_BASELINE)):
        if np.isfinite(v):
            scores.append(1.0 if v >= th else v / th if th > 0 else 0.0)
    if np.isfinite(q["g11_scale_ci"]):
        scores.append(1.0 if q["g11_scale_ci"] <= TH_G11_SCALE_CI else 0.0)
    return scores


def g5_is_computed(q: "QualityMetrics") -> bool:
    """G5 是否构成一个**可聚合**的观测（两标量都有限）。"""
    return (q.g5_reproj_err_median is not None and q.g5_reproj_err_p95 is not None
            and np.isfinite(float(q.g5_reproj_err_median))
            and np.isfinite(float(q.g5_reproj_err_p95)))


def overall_from_metrics(q: "QualityMetrics") -> float:
    """按 `_norm_scores` 口径重算 overall_quality（活动指标中有限值项的均值）。

    G5 仅在 `g5_is_computed` 为真时入分母（v5 HC37）；G8 永不入分母（HC38）。
    """
    scores = _norm_scores({
        "g1_blur_ok": q.g1_blur_ok,
        "g2_brightness": q.g2_brightness,
        "g3_motion_blur": q.g3_motion_blur,
        "g4_frame_count": q.g4_frame_count,
        "g5_reproj_err_median": (float("nan") if q.g5_reproj_err_median is None
                                 else q.g5_reproj_err_median),
        "g5_reproj_err_p95": (float("nan") if q.g5_reproj_err_p95 is None
                              else q.g5_reproj_err_p95),
        "g5_computed": g5_is_computed(q),
        "g6_depth_var_coeff": q.g6_depth_var_coeff,
        "g7_dynamic_ratio": q.g7_dynamic_ratio,
        "g9_tracker_consistency": q.g9_tracker_consistency,
        "g10_baseline_quality": q.g10_baseline_quality,
        "g11_scale_ci": q.g11_scale_ci,
    })
    return float(np.mean(scores)) if scores else 0.0


def needs_m5_enrichment(q: Optional["QualityMetrics"]) -> bool:
    """quality 里 G7/G9 是否仍空（P1 阶段 M5 未跑，这两项必然算不出）。

    这两项的数据源（动态 mask / track IoU）属 M5，P2 才产生；若不补写，
    artifact 里的 G7/G9 会永久是 NaN（"质量单一事实源"名不副实）。
    （G8 已删除，不再参与补写。）
    """
    if q is None:
        return False
    return any(not np.isfinite(float(v)) for v in (
        q.g7_dynamic_ratio, q.g9_tracker_consistency))


def enrich_with_m5(
    q: "QualityMetrics",
    *,
    dynamic_masks: Optional[np.ndarray] = None,
    track_ious: Optional[Sequence[float]] = None,
) -> "QualityMetrics":
    """把 M5 产物补进已落盘的 quality（只填 NaN 项，并重算 overall_quality）。

    返回新对象（不可变）；没有任何可补项时原样返回。
    """
    update: dict = {}
    if not np.isfinite(float(q.g7_dynamic_ratio)) and dynamic_masks is not None:
        val = g7_dynamic_ratio(dynamic_masks)
        if np.isfinite(val):
            update["g7_dynamic_ratio"] = val
    if not np.isfinite(float(q.g9_tracker_consistency)) and track_ious:
        val = g9_tracker_consistency(track_ious)
        if np.isfinite(val):
            update["g9_tracker_consistency"] = val
    if not update:
        return q
    merged = q.model_copy(update=update)
    return merged.model_copy(update={"overall_quality": overall_from_metrics(merged)})


def compute_g1_g11(
    artifact: Optional[ReconstructionArtifact] = None,
    *,
    frames: Optional[Sequence[np.ndarray]] = None,
    depth_maps: Optional[np.ndarray] = None,
    reproj_errors: Optional[np.ndarray] = None,
    reprojection_status: str = "not_available",
    dynamic_masks: Optional[np.ndarray] = None,
    track_ious: Optional[Sequence[float]] = None,
    c2w_list: Optional[np.ndarray] = None,
    scale_ci_rel: Optional[float] = None,
) -> QualityMetrics:
    """计算活动质量指标（附录 A：G1–G4、G6、G7、G9–G11 + 条件项 G5）。

    依赖外部产物的指标从参数读入；artifact 已携带 quality 时可作数据源。
    算不出的指标为 NaN（非 G5 项）或 `None`（G5 项），**不伪造**。

    `reprojection_status`：只有 `"computed"`（真 BA 跑通）时才读 `reproj_errors`
    产生 G5 标量；其余情况 G5 必须为 `None`，从 `overall_quality` 分母排除
    （HC37：正式 `vggt` 主线为 `not_available`，禁止代理值）。

    `scale_ci_rel` 为 **v5 分数口径**的相对 CI 半宽（HC29），直接写入
    `g11_scale_ci`；旧的绝对米制半宽字段已按 HC39 移除。
    """
    # artifact 中已有的 quality 字段作为兜底数据源
    art_q = artifact.quality if artifact is not None else None
    if artifact is not None and reprojection_status == "not_available":
        # 以 artifact 自身的重投影状态为准（BA 产物才可能 computed）
        reprojection_status = str(
            getattr(artifact, "reprojection_status", "not_available") or "not_available")
    depth_shape = None
    if depth_maps is not None:
        try:
            from skill3d.coords import depth_grid_shape

            depth_shape = depth_grid_shape(depth_maps)   # G3 必须在深度网格上算（C-8）
        except Exception:  # noqa: BLE001 - 深度数组异常：G3 退化为原始分辨率
            depth_shape = None

    def _fallback(val: float, name: str) -> float:
        return val if np.isfinite(val) or art_q is None else getattr(art_q, name)

    def _fallback_opt(val: Optional[float], name: str) -> Optional[float]:
        """G5 兜底：只接受已 computed 的旧值，否则 None（不补分）。"""
        if val is not None and np.isfinite(val):
            return float(val)
        if art_q is None:
            return None
        return _finite_or_none(getattr(art_q, name, None))

    g5_med, g5_p95 = g5_reproj_err(reproj_errors, status=reprojection_status)
    q = {
        "g1_blur_ok": _fallback(g1_blur_ok(frames) if frames else NaN, "g1_blur_ok"),
        "g2_brightness": _fallback(g2_brightness(frames) if frames else NaN, "g2_brightness"),
        "g3_motion_blur": _fallback(
            g3_motion_blur(frames, depth_shape) if frames else NaN, "g3_motion_blur"),
        "g4_frame_count": len(frames) if frames is not None else (
            art_q.g4_frame_count if art_q is not None else 0),
        "g5_reproj_err_median": _fallback_opt(g5_med, "g5_reproj_err_median"),
        "g5_reproj_err_p95": _fallback_opt(g5_p95, "g5_reproj_err_p95"),
        "g6_depth_var_coeff": _fallback(
            g6_depth_var_coeff(depth_maps) if depth_maps is not None else NaN,
            "g6_depth_var_coeff"),
        "g7_dynamic_ratio": _fallback(g7_dynamic_ratio(dynamic_masks), "g7_dynamic_ratio"),
        "g9_tracker_consistency": _fallback(g9_tracker_consistency(track_ious), "g9_tracker_consistency"),
        "g10_baseline_quality": _fallback(g10_baseline_quality(c2w_list), "g10_baseline_quality"),
        "g11_scale_ci": _fallback(g11_scale_ci(scale_ci_rel), "g11_scale_ci"),
    }
    scores = _norm_scores({
        **q,
        # G5 入分母的前提：状态 computed **且** 两个标量都有限
        "g5_computed": (reprojection_status == "computed"
                        and q["g5_reproj_err_median"] is not None
                        and q["g5_reproj_err_p95"] is not None),
    })
    q["overall_quality"] = float(np.mean(scores)) if scores else 0.0
    return QualityMetrics(**q)


# --------------------------------------------------- 质量写回（D-5 / 硬约束 22）----

def quality_is_computed(art: Optional[ReconstructionArtifact]) -> bool:
    """artifact 的质量是否可用（fail-closed 判定的唯一入口）。

    仅当 `quality_status=="computed"` 且 `quality` 非 None 且 `overall_quality`
    有限时才算"质量已计算"。非 computed 的 artifact **不得**停在 full_3d。
    """
    if art is None:
        return False
    if getattr(art, "quality_status", "not_computed") != "computed":
        return False
    q = getattr(art, "quality", None)
    if q is None:
        return False
    return bool(np.isfinite(float(q.overall_quality)))


def apply_quality(
    art: ReconstructionArtifact,
    q: Optional[QualityMetrics],
    *,
    status: Literal["computed", "failed"] = "computed",
    artifact_path: Optional[str | Path] = None,
) -> ReconstructionArtifact:
    """把质量写回 artifact（**不可变**：返回新版本，不原地改老对象）。

    - `status="computed"`：`quality=q`、`quality_status="computed"`；
    - `status="failed"`：`quality=None`、`quality_status="failed"`（route 必落
      `fallback_2d_only`，硬约束 22）；
    - `artifact_path` 给定则**原子写回** artifact JSON（方案 Y：序列化落盘，天然幂等）。

    v5 G5 口径（HC37）：G5 标量**只能**在 `reprojection_status=="computed"` 时为有限值；
    其余情况一律 `None`。若 artifact 声明 computed 但本次拿不到有限残差，则把状态降为
    `failed`（不得"声明算过却留空"）。

    实码核验结论：`schemas.Spec` 未配置 frozen → `model_copy` 路径成立；
    若将来改为 frozen，本函数回退到 deepcopy + object.__setattr__。
    """
    update: dict = {"quality_status": status,
                    "quality": q if status == "computed" else None}
    if q is not None and status == "computed":
        g5_ok = g5_is_computed(q)
        if g5_ok:
            update["reprojection_status"] = "computed"
            update["g5_reproj_err_median"] = float(q.g5_reproj_err_median)  # type: ignore[arg-type]
            update["g5_reproj_err_p95"] = float(q.g5_reproj_err_p95)        # type: ignore[arg-type]
        else:
            # 没有真 BA 残差 → not_available（正式 vggt 主线），G5 一律 None
            update["reprojection_status"] = (
                "failed" if str(getattr(art, "reprojection_status", "not_available"))
                == "computed" else "not_available")
            update["g5_reproj_err_median"] = None
            update["g5_reproj_err_p95"] = None
    try:
        new_art = art.model_copy(update=update)
    except Exception:  # noqa: BLE001 - frozen 实例的兜底路径
        import copy

        new_art = copy.deepcopy(art)
        for k, v in update.items():
            object.__setattr__(new_art, k, v)
    if artifact_path:
        write_artifact_json(new_art, artifact_path)
    return new_art


def write_artifact_json(art: ReconstructionArtifact, path: str | Path) -> Path:
    """原子写 artifact JSON（先写临时文件再 replace，避免半截文件被 P2 读到）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(art.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def compute_and_store_quality(
    art: ReconstructionArtifact,
    *,
    frames: Optional[Sequence[np.ndarray]] = None,
    depth_maps: Optional[np.ndarray] = None,
    reproj_errors: Optional[np.ndarray] = None,
    dynamic_masks: Optional[np.ndarray] = None,
    track_ious: Optional[Sequence[float]] = None,
    c2w_list: Optional[np.ndarray] = None,
    artifact_path: Optional[str | Path] = None,
) -> ReconstructionArtifact:
    """M4 单一入口：算活动质量指标 → 写回 artifact（可选落盘）。

    - **方案 X（优先）**：P1 重建阶段调用本函数，quality 随 artifact 持久化，
      P2 加载即得实算值、零重算；
    - **方案 Y（兜底）**：P2 读到 `quality_status != "computed"` 时调用本函数，
      并把结果原子写回 artifact 文件；
    - **禁止方案 Z**（只写内存不持久化）。

    G5 由 artifact 自身的 `reprojection_status` 决定（v5 HC37）：正式 `vggt` 主线为
    `not_available` → G5 恒 None 且不入 `overall_quality` 分母；只有真 BA 产物
    （`vggt_sparse_ba` + `reprojection_status="computed"`）才产生 G5 标量。

    数据源全缺（无 frames / depth）时**不得**标 "computed"：
    `overall_quality` 只对有限值求均值，此时会被 G4 一项撑起来，属于伪造质量。
    这种情况记 `quality_status="failed"` + `quality=None`（route 必落 fallback，硬约束 22）。
    """
    if frames is None and depth_maps is None:
        return apply_quality(art, None, status="failed", artifact_path=artifact_path)
    q = compute_g1_g11(
        art, frames=frames, depth_maps=depth_maps, reproj_errors=reproj_errors,
        reprojection_status=str(getattr(art, "reprojection_status", "not_available")),
        dynamic_masks=dynamic_masks,
        track_ious=track_ious, c2w_list=c2w_list,
        # v5：G11 = scale_ci_rel（分数口径，HC29）；旧绝对字段已按 HC39 移除
        scale_ci_rel=art.scale_ci_rel,
    )
    return apply_quality(art, q, status="computed", artifact_path=artifact_path)
