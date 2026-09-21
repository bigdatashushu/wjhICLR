"""M3 metric 尺度锚定：已知物体先验 + 地平面/相机高先验 + 鲁棒加权融合。

**本模块只做锚点提取与融合**（`ScaleAnchor` / `ScaleEstimate`）。v4 起，写进
`ReconstructionArtifact` 的尺度字段（含 CI 口径、置信档、逐题型授权）一律由
`reconstruction/scale_assessment.py` 的 `assess_scale` 产出 —— 那里才是
HC29/30/31/33 的唯一入口，本模块的 `ScaleEstimate.scale_confidence` 仅供内部
诊断，**不得**直接作为 artifact 的置信档（未标定的 medium/high 会被 Schema 兜底降为 low）。

v4 融合口径见 `fuse_scale_anchors_robust`（log 空间共识窗口 + Huber 精修 + 显式冲突检测）。

**为什么不用 PaGeR**（§4 §6.2、§7.1 G-11）：PaGeR（arXiv:2605.26368）输入单张
360° ERP 全景图，而 VSI-Bench 是 640×480 手持透视 RGB 视频（§1.2 已核验），
模态无直接转换路径——即使非商用可免费使用，接入也必然失败。故主线改为自研：

```
SAM2 对象绑定（M5）                    场景点云 + 相机位姿（VGGT/COLMAP）
        │                                        │
        ├──► 标准物体尺寸先验锚点 ──┐             │
        │    （门 2.05m / 桌面 0.75m│             │
        │     / 椅高 0.9m …）      ├─► 鲁棒融合（共识窗口 + Huber）→ scale + CI
        └──► 地平面 RANSAC ────────┤    （`fuse_scale_anchors_robust`）
             相机高 ~1.5m 先验 ────┘
```

`apply_scale` 对 SfM 点云做 uniform scale 变换。

纪律：本模块在线（确定性 numpy，无 VLM、无 GPT-6，硬约束 1/17）；
一切数值先验为起始参考值 `TODO_CALIBRATE`（§0.2 硬约束 3）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np

# ----------------------------------------------------------------- 物理先验 ----
# 手持相机高度先验：成人持机行走 ~1.5m（TODO_CALIBRATE）
CAMERA_HEIGHT_PRIOR_M: float = 1.5
CAMERA_HEIGHT_PRIOR_SIGMA_M: float = 0.15   # TODO_CALIBRATE：先验标准差

# 标准物体尺寸先验（室内常见家具/构件，TODO_CALIBRATE）
# measure ∈ {extent_up: 竖直总高, top_height: 顶面离地高, longest_horizontal: 水平最长边}
OBJECT_SIZE_PRIORS: dict[str, list[tuple[str, float, float, float]]] = {
    # class_hint -> [(measure, prior_m, sigma_m, weight)]
    "door": [(("extent_up", 2.05, 0.15, 1.0)), ],
    "doorway": [(("extent_up", 2.05, 0.15, 0.8)), ],
    "chair": [(("extent_up", 0.90, 0.12, 1.0)), (("top_height", 0.45, 0.05, 0.7))],
    "stool": [(("extent_up", 0.65, 0.10, 0.8)), ],
    "table": [(("top_height", 0.75, 0.05, 1.0)), (("longest_horizontal", 1.20, 0.35, 0.5))],
    "desk": [(("top_height", 0.75, 0.05, 1.0)), ],
    "countertop": [(("top_height", 0.92, 0.05, 0.9)), ],
    "counter": [(("top_height", 0.92, 0.05, 0.9)), ],
    "cabinet": [(("extent_up", 1.80, 0.20, 0.6)), ],
    "sofa": [(("extent_up", 0.85, 0.12, 0.8)), (("longest_horizontal", 2.00, 0.35, 0.7))],
    "couch": [(("extent_up", 0.85, 0.12, 0.8)), ],
    "bed": [(("longest_horizontal", 2.00, 0.20, 0.8)), (("extent_up", 0.55, 0.10, 0.6))],
}

# 认为是标准物体（可做锚点）的类别别名归一（SAM2 class_hint 来自 Qwen3-VL 提示）
_CLASS_ALIASES: dict[str, str] = {
    "doors": "door", "chairs": "chair", "tables": "table", "desks": "desk",
    "sofas": "sofa", "couches": "couch", "beds": "bed", "cabinets": "cabinet",
    "chairs_seat": "chair", "dining_table": "table", "coffee_table": "table",
    "kitchen_counter": "counter", "countertop": "countertop",
}

# 置信档位门槛（TODO_CALIBRATE，§7.1 G-11 验收）
CONF_HIGH_MIN_ANCHORS: int = 3
CONF_HIGH_MAX_REL_CI: float = 0.05
CONF_MEDIUM_MIN_ANCHORS: int = 2
CONF_MEDIUM_MAX_REL_CI: float = 0.15
# 尺度不确定度下限（测量噪声地板，避免单锚点给出 CI=0 的过度自信，TODO_CALIBRATE）
MIN_REL_UNCERTAINTY: float = 0.03
# D-2（§10.2）：只有地平面锚点时，相对 CI 超过该值一律强制 low（TODO_CALIBRATE）
PROVISIONAL_MEDIUM_MAX_REL_CI: float = 0.40
# D-2 升级 medium 的门槛（全部满足才可标 medium；任一否决条件命中即回 low）
MEDIUM_CALIB_MAX_MEDIAN_REL_ERR: float = 0.25   # 中位真实相对误差 < 25%
MEDIUM_CALIB_MIN_SPEARMAN: float = 0.5          # 解析 rel_ci 与真实误差 Spearman ρ > 0.5
MEDIUM_VETO_MEDIAN_REL_ERR: float = 0.50        # 中位真实相对误差 > 50% → 否决
MEDIUM_VETO_SPEARMAN: float = 0.2               # ρ < 0.2 → 否决
MEDIUM_VETO_PLANE_IDENTITY_ERR: float = 0.30    # 平面身份错误率 > 30% → 否决
# ARKitScenes 标定所需最小 episode 数（§10.2：≥30）
SCALE_CALIB_MIN_EPISODES: int = 30
# 尺度来源标签（写进 artifact.scale_source 与 RunManifest）
SCALE_SOURCE_PLANE = "camera_height_prior"
SCALE_SOURCE_OBJECTS = "known_object_prior"
SCALE_SOURCE_FUSED = "fused_anchors"
MEASUREMENT_REL_NOISE: float = 0.05   # 点云尺度测量的相对噪声（TODO_CALIBRATE）

# ---- v4 HC31 多锚点鲁棒融合（全部 TODO_CALIBRATE）----
# 有效锚点最小数量：少于它一律不得升级置信度（§10.2「有效锚点不足」）
MIN_ACCEPTED_ANCHORS_MEDIUM: int = 2
MIN_ACCEPTED_ANCHORS_HIGH: int = 3
# 冲突阈值：锚点两两尺度比的 max/min 超过它 → scale_conflict=True（§3 M3）。
# 1.5 含义：最乐观与最悲观锚点差 50% 即判冲突（起始参考值，须标定）
ANCHOR_CONFLICT_RATIO: float = 1.5
# 单锚点残差门：log 尺度距离超过它的锚点在稳健融合中被剔除（先剔除后复核冲突）
ANCHOR_OUTLIER_LOG_RESIDUAL: float = 0.4    # ≈ ±49% 尺度偏差
# 冲突判定的"多数"门槛：被拒锚点权重占比超过它 → 判定为锚点冲突（§10.2 L1
# "多数锚点冲突时必须 low"）。0.4 含义：被拒方权重不显著低于接受方即算冲突
CONFLICT_WEIGHT_FRACTION: float = 0.4
# Hubrid/M-estimator（Huber）的软阈值与最大迭代（在 log 尺度上做 IRLS）
HUBER_K: float = 1.345                      # 高斯下 95% 效率的经典取值
ROBUST_MAX_ITERS: int = 25
# 融合结果的 CI 上限：超过它视为"CI 过宽"（§10.2 否决项之一）
MAX_USABLE_REL_CI: float = 0.40
# 默认置信水平（研究目标；`[TODO_CALIBRATE]`，必须与冻结校准器一致）
DEFAULT_CONFIDENCE_LEVEL: float = 0.90

# 锚点类型归一（ScaleAnchor.kind/name → ScaleAnchorEvidence.anchor_type）
_ANCHOR_TYPE_BY_CLASS: dict[str, str] = {
    "door": "door", "doorway": "door",
    "table": "table", "desk": "table", "countertop": "table", "counter": "table",
    "coffee_table": "table", "dining_table": "table",
    "chair": "chair", "stool": "chair", "chairs_seat": "chair",
}

# 地平面 RANSAC 参数（TODO_CALIBRATE）
PLANE_MAX_ITERS: int = 200
PLANE_INLIER_REL_TOL: float = 0.01    # 内点阈值：相对场景直径
PLANE_MIN_INLIERS: int = 500
PLANE_NORMAL_HINT_MAX_ANGLE_DEG: float = 45.0   # 与相机 up 轴的最大夹角
# 单个物体锚点所需最少点数（TODO_CALIBRATE）
MIN_OBJECT_POINTS: int = 50


# ------------------------------------------------------------------ 数据结构 ----

@dataclass
class ScaleAnchor:
    """一个尺度观测：观测到的相对尺度尺寸 vs 物理先验。"""

    kind: Literal["ground_plane_camera_height", "object_prior"]
    name: str
    measured: float          # 相对单位下的观测尺寸
    prior_m: float           # 物理先验（m）
    ratio: float             # prior_m / measured = 候选尺度因子 (m / rel-unit)
    weight: float
    rel_sigma: float         # 该锚点的相对不确定度
    note: str = ""


@dataclass
class ScaleEstimate:
    """尺度锚定结果（§5.2 artifact 字段 + G-11 验收字段）。"""

    scale: Optional[float]                       # m / rel-unit；None = 锚定失败
    scale_ci: Optional[float]                    # CI 半宽（m，见 scale_ci_m 说明）
    scale_confidence: Literal["high", "medium", "low", "none"]
    scale_rel_ci: float                          # 尺度因子的相对 CI（论文报告用）
    method: str
    anchors: list[ScaleAnchor] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # 尺度来源标签（§7 RunManifest / D-2）：camera_height_prior / known_object_prior /
    # fused_anchors / fixed_scale / ""
    scale_source: str = ""

    @property
    def scale_known(self) -> bool:
        return self.scale is not None and self.scale > 0

    def summary(self) -> str:
        if not self.scale_known:
            return f"scale_unknown method={self.method} anchors={len(self.anchors)}"
        return (f"scale={self.scale:.4f} ci=±{self.scale_ci:.4f}m "
                f"rel_ci={self.scale_rel_ci:.3f} conf={self.scale_confidence} "
                f"anchors={len(self.anchors)}")


# ------------------------------------------------------------------ 基础工具 ----

def normalize_class_hint(hint: str) -> str:
    """类别名归一（小写、去空白、别名映射），用于查先验表。"""
    key = str(hint).strip().lower().replace(" ", "_")
    return _CLASS_ALIASES.get(key, key)


def known_object_priors(class_hint: str) -> list[tuple[str, float, float, float]]:
    """查标准物体尺寸先验；非标准物体返回空（不臆造尺寸）。"""
    return list(OBJECT_SIZE_PRIORS.get(normalize_class_hint(class_hint), []))


def apply_scale(point_map: np.ndarray, scale: float) -> np.ndarray:
    """对 SfM 点云做 uniform scale 变换（相对尺度 → metric，§7.1 G-11）。"""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale 必须为正有限值，收到 {scale}")
    return np.asarray(point_map, dtype=np.float64) * float(scale)


def ransac_plane(
    points: np.ndarray,
    *,
    seed: int = 0,
    max_iters: int = PLANE_MAX_ITERS,
    inlier_tol: Optional[float] = None,
    min_inliers: int = PLANE_MIN_INLIERS,
    normal_hint: Optional[np.ndarray] = None,
    max_angle_deg: float = PLANE_NORMAL_HINT_MAX_ANGLE_DEG,
) -> tuple[Optional[np.ndarray], Optional[float], np.ndarray]:
    """RANSAC 平面拟合（默认找地平面）。

    返回 `(normal, offset, inlier_mask)`，满足 `normal·p + offset = 0`；
    点数不足或无有效平面时返回 `(None, None, 全 False)`。

    `normal_hint` 给出期望法向（如相机 up 轴）时，只接受夹角在
    `max_angle_deg` 内的候选平面——手持视频中相机大致竖直，地面法向
    应与相机 up 轴同向，该约束显著提升地平面召回率。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < max(min_inliers, 3):
        return None, None, np.zeros(0, dtype=bool)

    if inlier_tol is None:
        extent = float(np.linalg.norm(np.percentile(pts, 95, axis=0)
                                      - np.percentile(pts, 5, axis=0)))
        inlier_tol = max(PLANE_INLIER_REL_TOL * extent, 1e-6)

    hint = None
    if normal_hint is not None:
        h = np.asarray(normal_hint, dtype=np.float64).ravel()[:3]
        n = float(np.linalg.norm(h))
        if n > 1e-9:
            hint = h / n

    rng = np.random.default_rng(seed)
    best_count, best_normal, best_offset = 0, None, None
    cos_limit = float(np.cos(np.deg2rad(max_angle_deg)))
    for _ in range(max_iters):
        idx = rng.choice(pts.shape[0], size=3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        offset = -float(normal @ p0)
        if hint is not None and abs(float(normal @ hint)) < cos_limit:
            continue  # 法向与相机 up 轴偏离过大 → 不是地平面
        dist = np.abs(pts @ normal + offset)
        count = int((dist <= inlier_tol).sum())
        if count > best_count:
            best_count, best_normal, best_offset = count, normal, offset

    if best_normal is None or best_count < min_inliers:
        return None, None, np.zeros(len(pts), dtype=bool)

    # 用全部内点做最小二乘精修（SVD → 最小奇异值方向为法向）
    mask = np.abs(pts @ best_normal + best_offset) <= inlier_tol
    inliers = pts[mask]
    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vt[-1]
    if best_normal @ normal < 0:
        normal = -normal
    offset = -float(normal @ centroid)
    return normal, offset, mask


def camera_up_hint(c2w_list: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """由第一帧相机姿态给出 up 轴先验（OpenCV 约定 +y 向下，故 up = -R[:,1]）。"""
    if c2w_list is None:
        return None
    c2w = np.asarray(c2w_list, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[-2:] != (4, 4) or len(c2w) < 1:
        return None
    up = -c2w[0][:3, 1]
    norm = float(np.linalg.norm(up))
    return up / norm if norm > 1e-9 else None


def gravity_from_planes(planes: Sequence[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    """多帧地面法向平均得到重力方向（单位向量）；无有效平面返回 None。"""
    acc = np.zeros(3, dtype=np.float64)
    n_valid = 0
    for n in planes:
        if n is None:
            continue
        arr = np.asarray(n, dtype=np.float64).ravel()[:3]
        norm = float(np.linalg.norm(arr))
        if norm < 1e-9:
            continue
        acc += arr / norm
        n_valid += 1
    if n_valid == 0 or float(np.linalg.norm(acc)) < 1e-9:
        return None
    return acc / float(np.linalg.norm(acc))


# ------------------------------------------------------------------ 锚点提取 ----

def ground_plane_anchor(
    point_map: Optional[np.ndarray],
    c2w_list: Optional[np.ndarray],
    *,
    seed: int = 0,
    prior_camera_height_m: float = CAMERA_HEIGHT_PRIOR_M,
    prior_sigma_m: float = CAMERA_HEIGHT_PRIOR_SIGMA_M,
) -> tuple[Optional[ScaleAnchor], Optional[tuple[np.ndarray, float]], list[str]]:
    """地平面 + 相机高先验锚点。

    步骤：RANSAC 拟合地平面（法向受相机 up 轴约束）→ 逐帧相机中心到平面的
    距离（相对单位）→ 取中位数为观测高度 → `ratio = 先验高度 / 观测高度`。
    返回 `(anchor, (normal, offset), notes)`。
    """
    notes: list[str] = []
    if point_map is None:
        return None, None, ["地平面锚定跳过：无点云"]
    if c2w_list is None:
        return None, None, ["地平面锚定跳过：无相机位姿"]

    c2w = np.asarray(c2w_list, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[-2:] != (4, 4):
        return None, None, ["地平面锚定跳过：c2w 形状非 (T,4,4)"]

    normal, offset, mask = ransac_plane(
        point_map, seed=seed, normal_hint=camera_up_hint(c2w))
    if normal is None or offset is None:
        return None, None, ["地平面 RANSAC 未找到有效平面（点数/内点不足或法向偏离 up 轴）"]

    centers = c2w[:, :3, 3]
    heights = np.abs(centers @ normal + offset)
    heights = heights[np.isfinite(heights) & (heights > 1e-9)]
    if heights.size == 0:
        return None, None, ["地平面锚定跳过：相机中心到平面距离全无效"]

    h_rel = float(np.median(heights))
    if h_rel <= 1e-9:
        return None, None, ["地平面锚定跳过：相机高观测为 0"]

    ratio = float(prior_camera_height_m) / h_rel
    # 权重：内点比例 × 高度观测的帧间一致性（相对 MAD）
    mad = float(np.median(np.abs(heights - h_rel)))
    consistency = float(np.clip(1.0 - 1.4826 * mad / h_rel, 0.0, 1.0))
    inlier_ratio = float(mask.sum()) / float(max(len(np.asarray(point_map).reshape(-1, 3)), 1))
    weight = max(consistency * min(1.0, inlier_ratio / 0.1), 1e-3)
    # D-2 解析不确定性传播（§10.2）：
    #   (σ_s/s)² ≈ (σ_h/h)² + (σ_plane/h_plane)²
    # σ_h = 先验标准差（默认 1.5±0.15m，约 10%）；σ_plane 取平面拟合的帧间离散
    # （相对 h_rel 的 MAD），再加测量噪声地板（避免单锚点 CI=0 的过度自信）。
    plane_rel_sigma = float(np.clip(1.4826 * mad / h_rel, 0.0, 1.0))
    sigma_h_rel = float(prior_sigma_m) / float(prior_camera_height_m)
    rel_sigma = float(np.sqrt(sigma_h_rel ** 2 + plane_rel_sigma ** 2))
    rel_sigma = max(rel_sigma / max(consistency, 1e-3), MEASUREMENT_REL_NOISE)
    notes.append(f"地平面内点={int(mask.sum())} 相机高观测={h_rel:.4f}(rel) "
                 f"帧间一致性={consistency:.3f} "
                 f"解析 σ_s/s=√((σ_h/h)²+(σ_plane/h_plane)²)="
                 f"√({sigma_h_rel ** 2:.5f}+{plane_rel_sigma ** 2:.5f})={rel_sigma:.4f}")
    anchor = ScaleAnchor(
        kind="ground_plane_camera_height", name="ground_plane_camera_height",
        measured=h_rel, prior_m=float(prior_camera_height_m), ratio=ratio,
        weight=weight, rel_sigma=rel_sigma,
        note=f"相机高先验 {prior_camera_height_m}m（TODO_CALIBRATE）",
    )
    return anchor, (normal, offset), notes


def object_anchor(
    points_world: Optional[np.ndarray],
    class_hint: str,
    *,
    up_axis: Optional[np.ndarray] = None,
    plane: Optional[tuple[np.ndarray, float]] = None,
    instance_id: str = "",
) -> tuple[list[ScaleAnchor], list[str]]:
    """标准物体尺寸先验锚点（门高 / 桌面高 / 椅高等）。

    `points_world` 为该对象的点云（相对尺度、世界坐标）。竖直方向优先取
    `up_axis`（重力方向），否则取相机 up 轴传入值。返回单个对象可产出多个
    锚点（不同 measure 各自一个观测）。
    """
    notes: list[str] = []
    priors = known_object_priors(class_hint)
    if not priors:
        return [], [f"对象 {instance_id or class_hint} 非标准物体（无尺寸先验，跳过）"]
    if points_world is None:
        return [], [f"对象 {instance_id or class_hint} 无点云（跳过）"]
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < MIN_OBJECT_POINTS:
        return [], [f"对象 {instance_id or class_hint} 点数不足 "
                    f"({pts.shape[0]}<{MIN_OBJECT_POINTS})，跳过"]

    if up_axis is None:
        up_axis = np.array([0.0, -1.0, 0.0])  # OpenCV 相机约定的 up（第一帧近似重力）
    up = np.asarray(up_axis, dtype=np.float64).ravel()[:3]
    norm = float(np.linalg.norm(up))
    if norm < 1e-9:
        return [], [f"对象 {instance_id or class_hint} up 轴无效，跳过"]
    up = up / norm

    # 沿 up 轴的投影（"离地高度"需要平面基准）
    proj = pts @ up
    p_min, p_max = float(proj.min()), float(proj.max())
    ground = None
    if plane is not None:
        normal, offset = plane
        n = np.asarray(normal, dtype=np.float64).ravel()[:3]
        nn = float(np.linalg.norm(n))
        if nn > 1e-9:
            # 平面上的最近点 p0 = -offset*n 沿 up 轴的投影即地面高度
            ground = float((-offset * (n / nn)) @ up)

    centered_h = pts - np.outer(proj, up)
    centered_h = centered_h - centered_h.mean(axis=0)
    if centered_h.shape[0] > 1:
        # 水平最长边：主轴方向上的 2/98 分位极差（对噪点鲁棒）
        _, _, vt = np.linalg.svd(centered_h, full_matrices=False)
        main = centered_h @ vt[0]
        longest_h = float(np.percentile(main, 98) - np.percentile(main, 2))
    else:
        longest_h = float("nan")

    extent_up = p_max - p_min
    top_height = (p_max - ground) if ground is not None else float("nan")

    out: list[ScaleAnchor] = []
    for measure, prior_m, sigma_m, w in priors:
        if measure == "extent_up":
            measured = extent_up
        elif measure == "top_height":
            measured = top_height
        elif measure == "longest_horizontal":
            measured = longest_h
        else:  # pragma: no cover - 先验表受控
            continue
        if not np.isfinite(measured) or measured <= 1e-9:
            continue
        rel_meas_sigma = MEASUREMENT_REL_NOISE * float(np.clip(
            400.0 / max(pts.shape[0], 1), 1.0, 3.0))
        rel_sigma = max(sigma_m / prior_m, rel_meas_sigma)
        # 有效样本量折减：点数越多越可信
        n_eff = float(np.clip(pts.shape[0] / 400.0, 0.2, 1.0))
        out.append(ScaleAnchor(
            kind="object_prior", name=f"{normalize_class_hint(class_hint)}:{measure}",
            measured=float(measured), prior_m=float(prior_m),
            ratio=float(prior_m) / float(measured),
            weight=float(w) * n_eff / (rel_sigma ** 2), rel_sigma=rel_sigma,
            note=f"{instance_id or class_hint} 先验 {prior_m}m±{sigma_m}（TODO_CALIBRATE）",
        ))
    if not out:
        notes.append(f"对象 {instance_id or class_hint} 先验维度无法测量（跳过）")
    return out, notes


# ------------------------------------------------------------------ 鲁棒融合 ----

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """加权中位数：排序后取累计权重过半处（权重非负，总和>0）。"""
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    total = float(cw[-1])
    if total <= 0:
        return float(np.median(values))
    idx = int(np.searchsorted(cw, 0.5 * total, side="left"))
    return float(v[min(idx, len(v) - 1)])


def fit_scale_robust(
    rel: np.ndarray,
    metric: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    min_samples: int = 100,
    sigma_floor: float = MIN_REL_UNCERTAINTY,
) -> tuple[float, float]:
    """鲁棒尺度拟合：`s = weighted_median(metric/rel)`，CI 由加权 MAD 估计（G-12）。

    返回 `(s, rel_ci)`，`rel_ci` 为**尺度因子的相对 CI**（无量纲）。
    单锚点或完全一致时用 `sigma_floor`/各锚点自身 `rel_sigma` 兜底，
    避免给出 CI=0 的过度自信（G11 门控依赖 CI 的可用性）。

    像素级调用（G-12：相对深度 vs metric 深度）与锚点级调用（G-11：多锚点融合）
    共用本函数；锚点级调用应放宽 `min_samples`。
    """
    r = np.asarray(rel, dtype=np.float64).ravel()
    m = np.asarray(metric, dtype=np.float64).ravel()
    if r.shape != m.shape:
        raise ValueError(f"rel/metric 形状不一致: {r.shape} vs {m.shape}")
    w = np.ones_like(r) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != r.shape:
        raise ValueError(f"weights 形状不一致: {w.shape} vs {r.shape}")
    mask = (np.abs(r) > 1e-9) & np.isfinite(r) & np.isfinite(m) & (m > 0) & np.isfinite(w) & (w > 0)
    if mask.sum() < min_samples:
        return float("nan"), float("inf")

    ratio = m[mask] / r[mask]
    ww = w[mask]
    s = _weighted_median(ratio, ww)
    if s <= 0 or not np.isfinite(s):
        return float("nan"), float("inf")

    # 加权 MAD → 相对标准误；再叠加样本量因子
    mad = _weighted_median(np.abs(ratio - s), ww)
    n_eff = float(ww.sum() ** 2 / max((ww ** 2).sum(), 1e-12))
    rel_mad = 1.4826 * mad / s
    rel_ci = 1.96 * rel_mad / np.sqrt(max(n_eff, 1.0))
    return s, max(float(rel_ci), float(sigma_floor))


def fuse_scale_anchors(
    anchors: Sequence[ScaleAnchor],
    *,
    sigma_floor: float = MIN_REL_UNCERTAINTY,
) -> tuple[Optional[float], float, float]:
    """多锚点加权融合（G-11/G-12）：返回 `(scale, rel_ci, ci_m)`。

    `ci_m` = 相对 CI × 锚点典型尺寸（m），即"在锚定物体典型尺寸上的 metric 半宽"，
    与 G11 门控阈值（`TH_G11_SCALE_CI=0.1m`）同单位、可直接比较。
    """
    usable = [a for a in anchors if np.isfinite(a.measured) and a.measured > 1e-9
              and np.isfinite(a.prior_m) and a.prior_m > 0]
    if not usable:
        return None, float("inf"), float("inf")

    measured = np.array([a.measured for a in usable], dtype=np.float64)
    prior = np.array([a.prior_m for a in usable], dtype=np.float64)
    weights = np.array([max(a.weight, 1e-9) for a in usable], dtype=np.float64)
    # 逐锚点自身不确定度也纳入地板：单锚点时 rel_ci 至少是其 rel_sigma
    floor = max(float(sigma_floor),
                float(np.average([a.rel_sigma for a in usable], weights=weights)))
    scale, rel_ci = fit_scale_robust(measured, prior, weights=weights,
                                    min_samples=1, sigma_floor=floor)
    if scale is None or not np.isfinite(scale) or rel_ci == float("inf"):
        return None, float("inf"), float("inf")
    # 去掉高杠杆离群锚点后再算一次（一致性差 → CI 自然变宽）
    if len(usable) >= 3:
        ratio = prior / measured
        med = float(np.median(ratio))
        keep = np.abs(ratio - med) / med <= 0.5
        if keep.sum() >= 2 and keep.sum() < len(usable):
            scale, rel_ci = fit_scale_robust(measured[keep], prior[keep],
                                            weights=weights[keep], min_samples=1,
                                            sigma_floor=floor)
            usable = [a for a, k in zip(usable, keep) if k]
    ci_m = float(rel_ci * float(np.median([a.prior_m for a in usable])))
    return float(scale), float(rel_ci), ci_m


# ============================================ v4 HC31：多锚点鲁棒融合（log 空间）====
#
# 与 v3 的两点实质差异：
# 1. 融合在 **log 尺度空间** 做（尺度是乘性量，加权中位数在 log 空间才等价于
#    几何中位数；线性空间的均值/中位数对"锚点差 2 倍"这类乘性偏差不稳健）；
# 2. 每个锚点必须留下 **来源 / 估计 / 不确定性 / 残差 / 是否被接受 / 原因码**
#    （`ScaleAnchorEvidence`），并**显式检测冲突**（锚点两两尺度比超阈）。
#
# 融合权重用 Huber IRLS（M-estimator）：残差在 `HUBER_K·σ` 内按权重 1 计，
# 超出则线性降权。这比"硬剔除 50% 离群点"更平滑，且对 **单个离群锚点拉偏** 有
# 明确抵抗（§10.2 L1 合成多锚点 PoC 的验收目标）。


@dataclass
class FusionResult:
    """多锚点鲁棒融合的输出（v4 HC31）。"""

    scale: Optional[float]                 # m / rel-unit；None = 融合失败
    ci_rel: Optional[float]                # 相对 CI 半宽（分数，HC29 口径）
    ci_abs_m: Optional[float]              # = scale × ci_rel
    accepted: list[str]                    # 被接受的锚点名（审计）
    residuals: dict[str, float]            # 锚点名 → log 尺度残差
    conflict: bool                         # 锚点间冲突（超阈值 / 冲突未消解）
    reason_codes: dict[str, str]           # 锚点名 → ok / outlier / conflict / …
    conflict_ratio: Optional[float]        # 接受的锚点两两尺度比 max/min
    n_accepted: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.scale is not None and np.isfinite(self.scale) and self.scale > 0
                and self.ci_rel is not None and np.isfinite(self.ci_rel))


def _anchor_ratio(a: ScaleAnchor) -> float:
    """锚点的尺度候选（m / rel-unit）；非法时返回 NaN。"""
    if a.measured is None or a.prior_m is None:
        return float("nan")
    m, p = float(a.measured), float(a.prior_m)
    if not (np.isfinite(m) and np.isfinite(p)) or m <= 1e-9 or p <= 0:
        return float("nan")
    return p / m


def _consensus_window(log_r: np.ndarray, weights: np.ndarray,
                      window: float) -> tuple[float, np.ndarray]:
    """共识窗口搜索（鲁棒位置估计）：返回 `(位置, 接受掩码)`。

    规则：对每个候选锚点 `i`，统计落在 `|log r_j − log r_i| <= window` 内的锚点
    **个数**（先按票数，再按权重和打破平局），取票数最多的窗口；位置用窗口内成员的
    **加权中位数**。

    为什么用"票数优先"而不是"权重和优先"：先验权重按 `1/σ²` 计算，单个先验很窄的
    锚点可能拿到与两个一致锚点相当的总权重（实测：门 2.05±0.15 vs 桌高 0.75±0.05），
    纯权重会把它单独选中、反而把多数派判成离群 —— 这正是 §10.2 L1 要防的"被单个
    离群锚点拉偏"。票数优先等价于"多数派获胜"，权重只在同票时生效。
    """
    n = log_r.size
    best_i, best_key = 0, (-1.0, -1.0)
    for i in range(n):
        in_w = np.abs(log_r - log_r[i]) <= window
        key = (float(in_w.sum()), float(weights[in_w].sum()))
        if key > best_key:
            best_i, best_key = i, key
    mask = np.abs(log_r - log_r[best_i]) <= window
    loc = _weighted_median(log_r[mask], weights[mask])
    return float(loc), mask


def _huber_weights(resid: np.ndarray, w0: np.ndarray,
                   *, huber_k: float = HUBER_K) -> np.ndarray:
    """Huber 软权重：`|r| <= k·s` 时为 1，超出按 `k·s/|r|` 线性降权。"""
    s = max(1.4826 * float(_weighted_median(resid, w0)), 1e-6)
    rw = np.where(resid <= huber_k * s, 1.0, huber_k * s / np.maximum(resid, 1e-12))
    return np.asarray(w0, dtype=np.float64) * rw


def _robust_log_location(log_r: np.ndarray, weights: np.ndarray,
                         *, huber_k: float = HUBER_K,
                         max_iters: int = ROBUST_MAX_ITERS) -> tuple[float, np.ndarray]:
    """log 尺度的 Huber M-estimator（IRLS）；返回 `(位置, 最终权重)`。

    供 `fuse_scale_anchors_robust` 在**共识集内**精修使用（外部工具，保留公开接口
    以便单测与消融对照"纯 Huber"与"共识窗口 + 加权中位数"两种融合口径）。
    """
    w0 = np.asarray(weights, dtype=np.float64)
    if w0.sum() <= 0:
        w0 = np.ones_like(log_r)
    loc = _weighted_median(log_r, w0)
    w = w0.copy()
    for _ in range(max_iters):
        w = _huber_weights(np.abs(log_r - loc), w0, huber_k=huber_k)
        if w.sum() <= 0:
            break
        loc_new = float(np.average(log_r, weights=w))
        if abs(loc_new - loc) < 1e-9:
            loc = loc_new
            break
        loc = loc_new
    return float(loc), w


def fuse_scale_anchors_robust(
    anchors: Sequence[ScaleAnchor],
    *,
    sigma_floor: float = MIN_REL_UNCERTAINTY,
    conflict_ratio: float = ANCHOR_CONFLICT_RATIO,
    outlier_log_residual: float = ANCHOR_OUTLIER_LOG_RESIDUAL,
    conflict_weight_fraction: float = CONFLICT_WEIGHT_FRACTION,
) -> FusionResult:
    """多锚点鲁棒融合（v4 HC31 唯一入口；阈值全部 `[TODO_CALIBRATE]`）。

    流程：
    1. 过滤非法锚点（measured/prior 非有限或非正）→ `reason_code="invalid"`；
    2. **共识窗口搜索**（票数优先 + 权重和破平）→ 多数派锚点集合；
    3. 共识集内加权中位数求尺度；CI = 1.96 × 加权 MAD / √n_eff，并与各锚点自身
       `rel_sigma` 的加权均值取较大者（地板，避免单锚点 CI=0 的过度自信）；
    4. **冲突检测（两路，任一成立即 `conflict=True`）**：
       - 被拒锚点的权重占比 > `conflict_weight_fraction`（"多数锚点冲突"，
         §10.2 L1 明文要求此时必须 low）；
       - 有效锚点两两尺度比 `max/min > conflict_ratio` **且**被拒权重不可忽略。
       单个小权重离群锚点只改标 `reason_code="outlier"`，不误报冲突
       （L1：注入单个离群锚点不得把融合拉偏）。
    """
    names = [a.name or f"anchor_{i}" for i, a in enumerate(anchors)]
    ratios = np.array([_anchor_ratio(a) for a in anchors], dtype=np.float64)
    weights = np.array([max(float(a.weight or 0.0), 1e-9) for a in anchors], dtype=np.float64)
    reason: dict[str, str] = {}
    notes: list[str] = []

    valid = np.isfinite(ratios) & (ratios > 0)
    for i, name in enumerate(names):
        if not valid[i]:
            reason[name] = "invalid"
    if not valid.any():
        return FusionResult(None, None, None, [], {}, False, reason, None, 0,
                            ["无有效锚点可融合"])

    valid_idx = np.flatnonzero(valid)
    log_r_all = np.log(ratios[valid])
    w_all = weights[valid]

    loc0, mask0 = _consensus_window(log_r_all, w_all, outlier_log_residual)
    # 共识集内 Huber 精修（得到最终位置）
    loc, _ = _robust_log_location(log_r_all[mask0], w_all[mask0])
    s = float(np.exp(loc))
    if not np.isfinite(s) or s <= 0:
        return FusionResult(None, None, None, [], {}, False, reason, None, 0,
                            ["log 尺度位置非法（融合失败）"])

    resid_all = np.abs(log_r_all - loc)
    # 最终软权重（全 valid 集上围绕最终位置的 Huber 权重），供 CI 与加权平均使用
    w_refined = _huber_weights(resid_all, w_all)
    keep_mask = resid_all <= outlier_log_residual
    if keep_mask.sum() == 0:
        best = int(np.argmin(resid_all))
        keep_mask = np.zeros_like(keep_mask)
        keep_mask[best] = True
        notes.append("全部锚点残差超阈 → 保底接受残差最小者（冲突已记录）")
    for local_i, global_i in enumerate(valid_idx):
        name = names[global_i]
        resid = float(resid_all[local_i])
        if keep_mask[local_i]:
            reason[name] = "ok"
        else:
            reason[name] = "outlier" if resid > outlier_log_residual else "low_weight"

    kept_ratios = ratios[valid][keep_mask]
    kept_names = [names[i] for i in valid_idx[keep_mask]]
    w_kept = w_refined[keep_mask]
    rejected_weight = float(w_all[~keep_mask].sum())
    total_weight = float(w_all.sum())
    rejected_frac = rejected_weight / total_weight if total_weight > 0 else 0.0

    cr_valid = float(ratios[valid].max() / ratios[valid].min())
    cr_accepted = float(kept_ratios.max() / kept_ratios.min())
    conflict = bool(rejected_frac > conflict_weight_fraction
                    or (cr_accepted > conflict_ratio and rejected_frac > 0.0))
    if conflict:
        notes.append(
            f"锚点冲突：valid 两两尺度比 max/min={cr_valid:.3f}（阈值 {conflict_ratio}），"
            f"被拒锚点权重占比={rejected_frac:.3f}（阈值 {conflict_weight_fraction}）"
            "——§3 M3 要求显式记录且不得返回虚假的 medium")
        for name in kept_names:
            if reason.get(name) == "ok":
                reason[name] = "conflict"

    log_kept = np.log(kept_ratios)
    final_scale = float(np.exp(_weighted_median(log_kept, w_kept)))
    mad = _weighted_median(np.abs(log_kept - float(np.log(final_scale))), w_kept)
    n_eff = float(w_kept.sum() ** 2 / max((w_kept ** 2).sum(), 1e-12))
    rel_ci = 1.96 * 1.4826 * float(mad) / np.sqrt(max(n_eff, 1.0))
    floor = max(float(sigma_floor),
                float(np.average([a.rel_sigma for a, k in zip(anchors, valid & keep_mask)
                                  if k], weights=w_kept)))
    rel_ci = float(max(rel_ci, floor))
    # 冲突未消解时把冲突幅度并入 CI：区间必须覆盖到冲突的另一侧，
    # 否则会低估不确定性（HC30：冲突不得被"更小的 CI"掩盖）
    if conflict and cr_valid > 1.0:
        rel_ci = float(max(rel_ci, (cr_valid - 1.0) / 2.0))

    residuals = {names[i]: float(resid_all[j]) for j, i in enumerate(valid_idx)}
    return FusionResult(
        scale=final_scale, ci_rel=rel_ci, ci_abs_m=final_scale * rel_ci,
        accepted=kept_names, residuals=residuals, conflict=conflict,
        reason_codes=reason, conflict_ratio=cr_valid, n_accepted=len(kept_names),
        notes=notes,
    )


def anchor_evidence_of(anchors: Sequence[ScaleAnchor],
                       fusion: Optional[FusionResult] = None) -> list:
    """`ScaleAnchor` → `ScaleAnchorEvidence`（schema 层证据，硬约束 31）。

    每个锚点保存来源类型、尺度估计、不确定性、残差、接受状态与原因码；
    `anchor_type` 按类别归一为 `camera_height_floor / door / table / chair / other`。
    """
    from skill3d.schemas.reconstruction import ScaleAnchorEvidence

    fusion = fusion or FusionResult(None, None, None, [], {}, False, {}, None, 0)
    accepted = set(fusion.accepted)
    out: list[ScaleAnchorEvidence] = []
    for a in anchors:
        name = a.name or ""
        if a.kind == "ground_plane_camera_height":
            atype = "camera_height_floor"
        else:
            cls = normalize_class_hint(name.split(":")[0]) if ":" in name else \
                normalize_class_hint(name)
            atype = _ANCHOR_TYPE_BY_CLASS.get(cls, "other")
        ratio = _anchor_ratio(a)
        out.append(ScaleAnchorEvidence(
            anchor_type=atype,  # type: ignore[arg-type]
            scale_estimate=float(ratio) if np.isfinite(ratio) else float("nan"),
            ci_rel=float(a.rel_sigma),
            residual=float(fusion.residuals.get(name, float("nan"))),
            accepted=name in accepted,
            reason_code=str(fusion.reason_codes.get(name, "not_fused")),
            source_frame_ids=[],
            anchor_name=name, measured=(float(a.measured) if a.measured is not None else None),
            prior_m=(float(a.prior_m) if a.prior_m is not None else None),
            weight=float(a.weight or 0.0), note=a.note or "",
        ))
    return out


def _grade_confidence(n_anchors: int, rel_ci: float,
                      has_plane: bool) -> Literal["high", "medium", "low"]:
    """置信档位（TODO_CALIBRATE，§7.1 G-11 验收口径）。

    v2 口径：high 需 ≥3 锚点 + 相对 CI ≤5% + 有地平面；medium 需 ≥2 锚点 + CI ≤15%。
    **只有地平面锚点（默认路径）时按此口径恒为 low** —— 这正是 D-2 要处理的问题：
    见 `grade_confidence_d2`（provisional_medium 档，需标定后才可用）。
    """
    if (n_anchors >= CONF_HIGH_MIN_ANCHORS and rel_ci <= CONF_HIGH_MAX_REL_CI and has_plane):
        return "high"
    if n_anchors >= CONF_MEDIUM_MIN_ANCHORS and rel_ci <= CONF_MEDIUM_MAX_REL_CI:
        return "medium"
    return "low"


# ------------------------------------------------------- D-2 provisional_medium ----

@dataclass
class ScaleCalibration:
    """尺度档位的**标定证据**（§10.2 升级/否决标准的可判定载体）。

    仅当标定通过（`provisional_medium=True`）时，单地平面锚点才允许标 medium；
    未标定前 `scale_confidence` 一律 low（v3：medium 不得进主表）。
    """

    n_episodes: int = 0
    median_rel_err: float = float("nan")      # 中位真实相对误差（ARKitScenes GT 位姿）
    spearman_rho: float = float("nan")        # 解析 rel_ci 与真实误差的秩相关
    plane_identity_err: float = float("nan")  # 平面身份错误率
    measurement_mra: Optional[float] = None   # medium 档 measurement MRA
    baseline_2d_mra: Optional[float] = None   # 2D-only baseline MRA
    method: str = "arkitscenes_pairwise_translation_ratio"

    @property
    def provisional_medium(self) -> bool:
        ok, _ = evaluate_calibration(self)
        return ok

    def summary(self) -> str:
        ok, reasons = evaluate_calibration(self)
        return (f"calib n={self.n_episodes} median_rel_err={self.median_rel_err:.3f} "
                f"rho={self.spearman_rho:.3f} plane_err={self.plane_identity_err:.3f} "
                f"→ provisional_medium={ok}"
                + (f"（{'；'.join(reasons)}）" if reasons else ""))


def evaluate_calibration(cal: Optional[ScaleCalibration]) -> tuple[bool, list[str]]:
    """§10.2 升级 medium 的条件判定（全部满足才 True）；返回 (是否通过, 原因)。

    升级条件（全满足）：中位真实相对误差 <25%；解析 rel_ci 与真实误差 Spearman ρ>0.5；
    medium 档 measurement MRA 不低于 2D-only baseline。
    否决条件（任一命中即 False）：中位真实相对误差 >50%；ρ<0.2；平面身份错误率 >30%。
    """
    if cal is None:
        return False, ["未提供标定证据（§10.2：未标定前 medium 不得进主表）"]
    reasons: list[str] = []
    if cal.n_episodes < SCALE_CALIB_MIN_EPISODES:
        reasons.append(f"标定 episode 数 {cal.n_episodes} < {SCALE_CALIB_MIN_EPISODES}")
    if not np.isfinite(cal.median_rel_err):
        reasons.append("中位真实相对误差未计算")
    elif cal.median_rel_err > MEDIUM_VETO_MEDIAN_REL_ERR:
        reasons.append(f"中位真实相对误差 {cal.median_rel_err:.3f} > "
                       f"{MEDIUM_VETO_MEDIAN_REL_ERR}（否决）")
    elif cal.median_rel_err >= MEDIUM_CALIB_MAX_MEDIAN_REL_ERR:
        reasons.append(f"中位真实相对误差 {cal.median_rel_err:.3f} ≥ "
                       f"{MEDIUM_CALIB_MAX_MEDIAN_REL_ERR}（未达升级门槛）")
    if not np.isfinite(cal.spearman_rho):
        reasons.append("Spearman ρ 未计算")
    elif cal.spearman_rho < MEDIUM_VETO_SPEARMAN:
        reasons.append(f"Spearman ρ {cal.spearman_rho:.3f} < {MEDIUM_VETO_SPEARMAN}（否决）")
    elif cal.spearman_rho <= MEDIUM_CALIB_MIN_SPEARMAN:
        reasons.append(f"Spearman ρ {cal.spearman_rho:.3f} ≤ {MEDIUM_CALIB_MIN_SPEARMAN}"
                       "（未达升级门槛）")
    if np.isfinite(cal.plane_identity_err) \
            and cal.plane_identity_err > MEDIUM_VETO_PLANE_IDENTITY_ERR:
        reasons.append(f"平面身份错误率 {cal.plane_identity_err:.3f} > "
                       f"{MEDIUM_VETO_PLANE_IDENTITY_ERR}（否决）")
    if cal.measurement_mra is not None and cal.baseline_2d_mra is not None \
            and cal.measurement_mra < cal.baseline_2d_mra:
        reasons.append(f"medium 档 measurement MRA {cal.measurement_mra:.3f} < "
                       f"2D-only baseline {cal.baseline_2d_mra:.3f}")
    return (not reasons), reasons


def grade_confidence_d2(
    n_anchors: int,
    rel_ci: float,
    has_plane: bool,
    has_object_anchor: bool,
    *,
    calibration: Optional[ScaleCalibration] = None,
) -> Literal["high", "medium", "low"]:
    """D-2 档位判定（§10.2）：只有地平面锚点时允许走 provisional_medium。

    - 有标准物体锚点 → 走 v2 口径 `_grade_confidence`（多锚点融合）；
    - **只有地平面**（默认路径）→ 仅当 `relative_ci <= 40%` **且**标定通过时才标
      medium；否则一律 low（v3：未标定前 medium 不得进主表）。
    """
    if has_object_anchor:
        return _grade_confidence(n_anchors, rel_ci, has_plane)
    if rel_ci > PROVISIONAL_MEDIUM_MAX_REL_CI:
        return "low"
    ok, _ = evaluate_calibration(calibration)
    return "medium" if ok else "low"


def calibrate_scale_on_arkitscenes(
    records: Sequence[dict],
    *,
    measurement_mra: Optional[float] = None,
    baseline_2d_mra: Optional[float] = None,
) -> ScaleCalibration:
    """ARKitScenes 尺度标定（§10.2）：用 **GT 位姿 pairwise 平移幅值比** 评误差。

    为什么不用深度对齐 / Umeyama：那类对齐本身会吸收尺度（自证循环），
    只有 GT 位姿的成对平移幅值比才给出与尺度无关的参照。

    每条 record：`{"scale_pred": float, "c2w_pred": (T,4,4), "c2w_gt": (T,4,4),
    "rel_ci": float, "plane_identity_ok": bool}`。
    """
    rel_errs: list[float] = []
    rel_cis: list[float] = []
    plane_bad = 0
    for rec in records:
        err = _pairwise_scale_error(rec.get("c2w_pred"), rec.get("c2w_gt"))
        if err is None:
            continue
        rel_errs.append(abs(err - 1.0))
        rel_cis.append(float(rec.get("rel_ci") or 0.0))
        if rec.get("plane_identity_ok") is False:
            plane_bad += 1
    n = len(rel_errs)
    if n == 0:
        return ScaleCalibration(n_episodes=0)
    rho = _spearman(rel_cis, rel_errs) if n >= 3 else float("nan")
    return ScaleCalibration(
        n_episodes=n,
        median_rel_err=float(np.median(rel_errs)),
        spearman_rho=rho,
        plane_identity_err=(plane_bad / len(records)) if records else float("nan"),
        measurement_mra=measurement_mra,
        baseline_2d_mra=baseline_2d_mra,
    )


def _pairwise_scale_error(c2w_pred, c2w_gt) -> Optional[float]:
    """GT 位姿 pairwise 平移幅值比的中位（预测/真值的尺度比估计）。"""
    if c2w_pred is None or c2w_gt is None:
        return None
    a = np.asarray(c2w_pred, dtype=np.float64)
    b = np.asarray(c2w_gt, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-2:] != (4, 4) or len(a) < 2:
        return None
    d_pred = np.linalg.norm(np.diff(a[:, :3, 3], axis=0), axis=1)
    d_gt = np.linalg.norm(np.diff(b[:, :3, 3], axis=0), axis=1)
    ok = np.isfinite(d_pred) & np.isfinite(d_gt) & (d_gt > 1e-9)
    if not ok.any():
        return None
    return float(np.median(d_pred[ok] / d_gt[ok]))


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """秩相关（优先 scipy；退化样本/无 scipy 时用 numpy 实现，仍不可算则 NaN）。"""
    try:
        from scipy.stats import spearmanr

        rho, _p = spearmanr(x, y)
        return float(rho) if np.isfinite(rho) else float("nan")
    except Exception:  # noqa: BLE001 - 无 scipy：手写秩相关
        xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if xs.size < 3:
            return float("nan")

        def _rank(v: np.ndarray) -> np.ndarray:
            order = np.argsort(v)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(v.size, dtype=np.float64)
            return ranks

        rx, ry = _rank(xs), _rank(ys)
        rx, ry = rx - rx.mean(), ry - ry.mean()
        denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
        return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def scale_source_of(anchors: Sequence[ScaleAnchor], has_object_anchor: bool) -> str:
    """尺度来源标签（写进 artifact.scale_source 与 RunManifest，§7 / D-2）。"""
    kinds = {a.kind for a in anchors}
    if has_object_anchor and "ground_plane_camera_height" in kinds:
        return SCALE_SOURCE_FUSED
    if has_object_anchor:
        return SCALE_SOURCE_OBJECTS
    if "ground_plane_camera_height" in kinds:
        return SCALE_SOURCE_PLANE
    return ""


# ------------------------------------------------------------------ 主入口 ----

def anchor_metric_scale(
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
    calibration: Optional[ScaleCalibration] = None,
) -> ScaleEstimate:
    """自研 metric 尺度锚定入口（G-11）。

    参数：
    - `point_map`：场景点云（相对尺度，世界坐标），(T,H,W,3) 或 (N,3)；
    - `c2w_list`：(T,4,4) 相机位姿（世界坐标 SE(3)）；
    - `objects`：M5 `ObjectInstance` 序列（用 `class_hint` 查尺寸先验）；
    - `object_points`：instance_id -> 该对象世界坐标点云（相对尺度）；
    - `depth_maps`：可选；仅在无点云时用于像素级拟合的降级路径；
    - `fixed_scale`：非 None 时直接返回该固定尺度（论文消融 "固定尺度 1.0" 档）；
    - `ground_plane`：是否启用"地平面 + 相机高"锚点（消融开关）；
    - `calibration`：尺度档位标定证据（D-2）。**未标定前只有地平面锚点一律 low**，
      标定通过后才允许 `provisional_medium`（`scale_source="camera_height_prior"`）。

    永不抛异常：锚定失败 → `scale_known=False` + `scale_confidence="none"`，
    由 M4 分流为 2D-only（§4 M3 字段 9）。
    """
    if fixed_scale is not None:
        return ScaleEstimate(
            scale=float(fixed_scale), scale_ci=None, scale_confidence="low",
            scale_rel_ci=0.0, method="fixed_scale_ablation", scale_source="fixed_scale",
            anchors=[], notes=[f"消融档：固定尺度 {fixed_scale}（无锚定，§7.1 G-11 对照）"],
        )

    anchors: list[ScaleAnchor] = []
    notes: list[str] = []
    plane: Optional[tuple[np.ndarray, float]] = None
    up_axis: Optional[np.ndarray] = None

    if ground_plane:
        pa, plane, pnotes = ground_plane_anchor(point_map, c2w_list, seed=seed)
        notes.extend(pnotes)
        if pa is not None:
            anchors.append(pa)
        if plane is not None:
            up_axis = np.asarray(plane[0], dtype=np.float64)
    if up_axis is None:
        up_axis = camera_up_hint(c2w_list)

    # 标准物体先验锚点
    n_objects = 0
    if objects:
        for obj in objects:
            iid = str(getattr(obj, "instance_id", "") or "")
            hint = str(getattr(obj, "class_hint", "") or "")
            pts = None
            if object_points and iid in object_points:
                pts = object_points[iid]
            else:
                ref = getattr(obj, "pointcloud_world", "") or ""
                if ref:
                    try:
                        pts = np.load(ref)
                    except Exception:  # noqa: BLE001 - ref 解析失败 → 该对象跳过
                        pts = None
            o_anchors, o_notes = object_anchor(pts, hint, up_axis=up_axis, plane=plane,
                                              instance_id=iid)
            notes.extend(o_notes)
            if o_anchors:
                n_objects += 1
                anchors.extend(o_anchors)

    if not anchors:
        # 无点云时用深度图做兜底：仅当存在相机高先验才可能成锚
        if point_map is None and depth_maps is not None and c2w_list is not None:
            notes.append("无点云：深度图兜底路径需要 SAM2 对象绑定才能建立物体锚点")
        return ScaleEstimate(
            scale=None, scale_ci=None, scale_confidence="none", scale_rel_ci=float("inf"),
            method="known_object_prior+ground_plane_camera_height", scale_source="",
            anchors=[], notes=notes + ["尺度锚定失败：无有效锚点 → scale_known=False，"
                                       "measurement 题降级 2D-only（§7.1 G-11 验收）"],
        )

    scale, rel_ci, ci_m = fuse_scale_anchors(anchors)
    if scale is None:
        return ScaleEstimate(
            scale=None, scale_ci=None, scale_confidence="none",
            scale_rel_ci=float("inf"), scale_source="",
            method="known_object_prior+ground_plane_camera_height",
            anchors=anchors, notes=notes + ["尺度融合失败（比值非正/非有限）"],
        )

    has_plane = any(a.kind == "ground_plane_camera_height" for a in anchors)
    has_object = n_objects > 0
    # D-2：只有地平面锚点时走 provisional_medium 通道（需标定通过 + rel_ci ≤ 40%）
    conf = grade_confidence_d2(len(anchors), rel_ci, has_plane, has_object,
                               calibration=calibration)
    source = scale_source_of(anchors, has_object)
    notes.append(f"锚点 {len(anchors)} 个（标准物体 {n_objects} 个，地平面 {'有' if has_plane else '无'}）"
                 f"；{scale:.4f} m/rel, rel_ci={rel_ci:.3f}, ci=±{ci_m:.4f}m → {conf}"
                 f"（scale_source={source}）")
    if conf == "low":
        if not has_object:
            notes.append(
                "scale_confidence=low：只有地平面锚点且"
                + ("相对 CI 超过 40% 或标定未通过" if rel_ci > PROVISIONAL_MEDIUM_MAX_REL_CI
                   or not evaluate_calibration(calibration)[0]
                   else "未获标定认可")
                + " → measurement 题降级 2D-only（§10.2 D-2 / §7.1 G-11 验收）")
        else:
            notes.append("scale_confidence=low：measurement 题降级 2D-only"
                         "（§7.1 G-11 验收）")
    return ScaleEstimate(
        scale=scale, scale_ci=ci_m, scale_confidence=conf, scale_rel_ci=rel_ci,
        method="known_object_prior+ground_plane_camera_height",
        anchors=anchors, notes=notes, scale_source=source,
    )
