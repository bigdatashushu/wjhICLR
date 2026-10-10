"""§12 稳健低分位距离原语（D8(4)）—— 纯 numpy / scipy 的**点集函数**。

本模块只做 3D 点集上的确定性与几何量计算，**不注册任何 Tool、不碰 REGISTRY、
不读重建产物、不做任何 I/O**；Tool 层（§9.4–9.7）负责取点集/取尺度后调用这里的函数，
再把结果组装成 `ToolResult` 与 trace。

核心定义（§12.1）
----------------
对 3D 点集 `P`，到参考 `R` 的距离集合 `D = {||p − R||}`，稳健距离 = `quantile(D, q)`，
`q` 默认候选 `0.01`（1%）。用稳健低分位**近似最近点**，对抗离群/飞点：
`q=0` 即真最近点（无稳健性），`q=0.5` 即中位数（被飞点拖走）。

统一预处理（§12.2，所有距离原语共用）
-------------------------------------
1. **体素降采样**（`voxel_size`，`[TODO_CALIBRATE]`）：压缩点集同时保留表面分布；
   实现为"按体素分组 → 取**离体素质心最近的真实点**"，全程 `np.lexsort` 定序，
   无未播种随机、无 dict 迭代序依赖（确定性硬要求）。
2. **VGGT `point_conf` 只作软权重/掩码**，且**必须先过 conf-warp 单调性自检**
   （§10.3、第 10 章）：`conf_warp_monotonic=None`（未自检）→ **完全不用 conf**
   （既不过滤也不加权）；`False`（自检未过）→ **降权**（幂次压平，绝不丢弃）；
   `True` → 可作软权重，且**仅在**调用方显式打开 `conf_optional_mask` 时启用
   `C>2` 可选掩码。`C>2` 从来不是硬阈值。
3. **排除 NaN/Inf 坐标**。
4. **最小点数 `N_min`**（`[TODO_CALIBRATE]`）：有效点 < `N_min` → `degraded`，
   **不输出伪精确值**（`distance_normalized=None`、`distance_metric=None`）。
   选"置 None"而不是"给值 + 标记"的理由：低分位距离在这个样本量下**没有统计学
   意义**（`q=0.01` 在 100 点以下退化成极小值，毫无稳健性），给数字等于邀请下游
   把它当精确值用；置 None 能把"测不出"和"测出来很差"区分开（fail-closed）。

两类几何原语与题型口径（v7 §2.2 修正）
--------------------------------
- 点到对象：`robust_distance_to_reference`，可用于相机或对象质心到点集的辅助测量。
- 对象到对象：`robust_distance_between_pointsets`，由 `object_distance_m` 与
  `relative_distance_rank` 使用。相对距离题比较题面参照对象到各候选类别最近实例的
  距离，不以相机替代参照对象；尺度在比较中约掉。绝对距离的米制值需要授权尺度。

降级条件（§12.4）与"值可否输出"的对应表
--------------------------------------
| flag | 触发 | 数值字段 |
|---|---|---|
| `degraded` | 有效点 < `N_min`（或输入非法/空） | `None` |
| `point_contamination_suspect` | 污染的极近距离占比 > `TH_CONTAM_FRACTION` | `None`（原值留在 `audit["suppressed_distance_normalized"]`） |
| `suspect_duplicate` | **调用方**传入 `duplicate_suspect=True`（两对象 mask 高重叠，§12.4 第 2 条） | 照常输出 + 标记（由 Tool 层决定降级） |
| `metric_scale_invalid` | 传了 `metric_scale` 但非有限/非正 | 归一化值照常；米制值 `None` |
| `plane_fit_low_quality` | 房间拟合质量 < `TH_FIT_QUALITY_MIN`（§9.5 过门判据） | 照常输出 + 标记 |
| `up_axis_ambiguous` / `no_ground_plane` / `ground_plane_axis_mismatch` | 房间几何退化 | 相应字段 `None` |

**"被污染的距离不作精确值输出"**（§12.4 末条）：污染时 `distance_normalized` 置 `None`，
但把被压制值写进 `audit["suppressed_distance_normalized"]` 供审计（不是"输出精确值"，
是"记录我们拒绝了什么"）。

`TAU_CONTAM` 的口径（与 §10.1"报率不报绝对距离"一致）
---------------------------------------------------
`TAU_CONTAM` 是**无量纲比例**，乘的是点云**自身的表面采样间距**（集合内最近邻距离的
中位数，由数据算出）：`极近距离 ≜ d < TAU_CONTAM × 采样间距`。故判据与 VGGT 点云的
任意归一化倍数无关，**绝不是米制阈值**。

输出/trace 字段（§12.5）
------------------------
`DistanceResult` 携带 `distance_normalized / distance_metric / scale_version /
quantile_q / voxel_size / conf_warp_version / n_valid_points / degradation_flags`，
`to_trace()` 直接给出可落盘的 dict（另附审计字段）；`DistancePrimitiveParams.snapshot()`
给阈值快照。`scale_version` 的单一事实源是 §11 的度量融合版本号
（`skill3d.reconstruction.metric_fusion.METRIC_FUSION_VERSION`），本模块不写第二份。

确定性
------
同一输入两次调用结果**逐位一致**：体素降采样走 `np.lexsort`（排序后归约），
无 RNG、无时间戳、无集合/字典迭代序、无多线程归约；KD-tree 只做距离查询。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

# `scale_version` 单一事实源（§11/D2）：直接引用度量融合版本号，
# 避免 trace 里出现第二份"尺度版本"字符串。
from skill3d.reconstruction.metric_fusion import (
    METRIC_FUSION_VERSION as SCALE_VERSION_DEFAULT,
)

# ---------------------------------------------------------------------------
# 阈值常量（**全部 TODO_CALIBRATE**，起始参考值只为让链路能跑；见 §12.1/§12.2/§12.4）
# ---------------------------------------------------------------------------

# §12.1：稳健低分位候选。q=0.01（1%）。**仅 inner 选择、outer 冻结**（D8）。
# 与官方 GT 口径的系统性偏差见 `ablation_quantiles`（§12.5 消融）。
QUANTILE_Q_DEFAULT: float = 0.01              # TODO_CALIBRATE: §12.1 低分位默认候选
# §12.2-1 体素降采样边长（**世界单位**；VGGT 点云是中位归一化、场景尺度 ~1，
# 故 0.01 ≈ 场景尺度的 1%）。须与"点云的归一化口径"联标：调用方若拿到的是别的
# 尺度（米/厘米），必须显式传 `params.voxel_size`，否则降采样会过粗或失效。
VOXEL_SIZE_DEFAULT: float = 0.01              # TODO_CALIBRATE: §12.2 体素边长（世界单位）
# §12.2-4 最小有效点数。取 100 的依据：q=0.01 需要至少 ~1/q 个样本，否则低分位
# 退化成"极小值"，稳健性为零（与 `m4_main_gate.MIN_CLOUD_POINTS` 同量级）。
N_MIN_DEFAULT: int = 100                      # TODO_CALIBRATE: §12.2 最小有效点数
# §12.4-3 污染判据：极近距离 ≜ `d < TAU_CONTAM × 表面采样间距`（无量纲比例）。
TAU_CONTAM: float = 0.1                       # TODO_CALIBRATE: §12.4 污染距离比例
# §12.4-3 占比门：极近距离占比超过它才判"污染"。指标可以是**率**（不是绝对距离）。
TH_CONTAM_FRACTION: float = 0.5               # TODO_CALIBRATE: §12.4 污染占比门
# §12.2-2 conf-warp 自检**未通过**时的降权方式：w ← w**0.5（向均匀压平；不丢弃）。
CONF_DOWNWEIGHT_POWER: float = 0.5            # TODO_CALIBRATE: §10.3 降权幂次
# §12.2-2 软权重的下界：conf 只降权、不把点权重置零（避免软权重变相成硬门）。
CONF_WEIGHT_FLOOR: float = 0.05               # TODO_CALIBRATE: conf 软权重下界
# §10.3 `C>2` **可选掩码**（不是硬阈值）：仅在 conf-warp 自检通过且调用方显式打开
# `conf_optional_mask=True` 时生效。数值与 `m4_main_gate.CONF_OPTIONAL_MASK_C`
# 同口径（单测交叉校验两者一致）。
CONF_OPTIONAL_MASK_C: float = 2.0             # TODO_CALIBRATE: §10.3 可选掩码阈值
# §12.5 消融档：min(q=0) vs q ∈ {0.5%, 1%, 2%, 5%}。
ABLATION_QUANTILES_Q: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05)
# §9.5 房间几何：地面/天花候选带取高度分布的最低/最高分位比例。
GROUND_BAND_Q: float = 0.02                   # TODO_CALIBRATE: §9.5 平面候选带分位
# §9.5 平面内点带容差（相对**场景稳健尺度**的百分比；不是米制阈值）。
PLANE_TOL_REL: float = 0.01                   # TODO_CALIBRATE: §9.5 平面内点带**下限**
# §9.5 平面身份判据：拟合出的地面法向与"上"方向的最小对齐（cos≈0.86 ⇒ 约 30° 内）。
PLANE_MIN_ALIGN_COS: float = 0.86             # TODO_CALIBRATE: §9.5 平面身份门
# §9.5 拟合质量门：低于则 `plane_fit_low_quality`（§9.5"拟合质量不过门时降级"）。
# §9.5 拟合质量门。**未经标定**（2026-09-21 实测：真实场景的 fit_quality 分布与
# 人工退化场景的分布尚未做过分离度标定，故它现在只能作**审计标记**）。
# 纪律：**不得**拿它当 abstain 的硬门（红线 2：未标定阈值不得充当结论）；
# 要用它做门必须先按 §10.6 的最小 PoC 完成分离度标定。
TH_FIT_QUALITY_MIN: float = 0.3               # TODO_CALIBRATE: §9.5 拟合质量门（未标定）
# §9.5 上方向可辨识判据：PCA 最小方差轴与次小方差轴的特征值比下限（房间场景里竖直
# 轴方差最小）。低于该比值 → `up_axis_ambiguous`（审计与标记，不静默假装确定）。
MIN_UP_AXIS_ANISOTROPY: float = 1.05          # TODO_CALIBRATE: §9.5 上方向可辨识性

# ---- 结构性下限（性能/退化保护，**不是**标定阈值）----
N_MIN_PLANE_FIT: int = 3                      # 平面拟合所需最少点（TLS 的秩条件）
N_MIN_ABLATION_PROBLEMS: int = 1              # 消融表至少一个样本
PLANE_REFIT_ITERS: int = 2                    # 地面平面内点带迭代精修次数（固定 → 可复现）
PLANE_BAND_TOL_MULT: float = 3.0              # 候选带宽度 = 该倍数 × 平面内点带容差
MIN_SPACING_EPS: float = 1e-12                # 除零保护（采样间距退化时的 eps）

# conf-warp 自检口径版本（§10.3/§12.5：进 trace，便于审计"这批数字按哪版纪律算的"）
CONF_WARP_VERSION: str = "conf-warp-v6"

__all__ = [
    "ABLATION_QUANTILES_Q",
    "CONF_OPTIONAL_MASK_C",
    "CONF_WARP_VERSION",
    "N_MIN_DEFAULT",
    "QUANTILE_Q_DEFAULT",
    "SCALE_VERSION_DEFAULT",
    "TAU_CONTAM",
    "TH_CONTAM_FRACTION",
    "TH_FIT_QUALITY_MIN",
    "VOXEL_SIZE_DEFAULT",
    "DistancePrimitiveParams",
    "DistanceResult",
    "ablation_quantiles",
    "object_extent",
    "planarity_and_ground",
    "preprocess_points",
    "robust_distance_between_pointsets",
    "robust_distance_to_reference",
    "room_size_from_planes",
]


# ---------------------------------------------------------------------------
# 参数与结果（§12.5：进 trace）
# ---------------------------------------------------------------------------

@dataclass
class DistancePrimitiveParams:
    """距离原语参数（§12.5：`snapshot()` 全进 trace，可审计"这批数字按哪套参数算的"）。

    全部字段默认值来自模块级常量；`quantile_q` 的**选择**只在 inner 做、outer 冻结
    （D8/§12.1）——本类是"被冻结的那套参数"的载体，不在这里做任何选择。
    """

    quantile_q: float = QUANTILE_Q_DEFAULT
    voxel_size: float = VOXEL_SIZE_DEFAULT
    conf_warp_version: str = CONF_WARP_VERSION
    n_min: int = N_MIN_DEFAULT

    def __post_init__(self) -> None:
        q = float(self.quantile_q)
        if not np.isfinite(q) or q < 0.0 or q > 1.0:
            raise ValueError(f"quantile_q 必须落在 [0,1]（收到 {self.quantile_q!r}）")
        v = float(self.voxel_size)
        if not np.isfinite(v):
            raise ValueError(
                f"voxel_size 必须有限（收到 {self.voxel_size!r}）；"
                "关闭降采样请显式传 0（<=0 视为关闭）")
        if int(self.n_min) < 1:
            raise ValueError(f"n_min 必须 >= 1（收到 {self.n_min!r}）")
        if not isinstance(self.conf_warp_version, str) or not self.conf_warp_version:
            raise ValueError("conf_warp_version 必须是非空字符串（进 trace）")

    def snapshot(self) -> dict:
        """阈值/口径快照（§12.5：进 `ToolResult` 与 trace）。"""
        return {
            "quantile_q": float(self.quantile_q),
            "voxel_size": float(self.voxel_size),
            "conf_warp_version": str(self.conf_warp_version),
            "n_min": int(self.n_min),
        }


@dataclass
class DistanceResult:
    """稳健距离结果（§12.5 输出字段 + 审计附加字段）。

    `distance_normalized` 为 `None` ⇔ 该量不可测或已被压制（见模块头"值可否输出"表）：
    宁可显式空，也不给伪精确值（fail-closed）。
    """

    distance_normalized: Optional[float]
    distance_metric: Optional[float]
    n_valid_points: int
    degradation_flags: list[str] = field(default_factory=list)
    quantile_q: float = QUANTILE_Q_DEFAULT
    voxel_size: float = VOXEL_SIZE_DEFAULT
    conf_warp_version: str = CONF_WARP_VERSION
    scale_version: Optional[str] = None
    # ---- 审计附加（§9.7 需要 n_nn_samples；其余供 trace 复核，不进正式口径）----
    metric_scale: Optional[float] = None
    n_points_raw: int = 0
    n_nn_samples: int = 0            # 仅 `robust_distance_between_pointsets` 有意义
    contamination_ratio: Optional[float] = None
    duplicate_suspect: bool = False
    conf_usage: str = "disabled"     # disabled / ignored_unverified / weighted / downweighted / masked_optional
    suppressed_distance_normalized: Optional[float] = None
    audit: dict = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        return "degraded" in self.degradation_flags

    def to_trace(self) -> dict:
        """§12.5 trace 字段（含审计附加项，全部可 JSON 化）。"""
        return {
            "distance_normalized": self.distance_normalized,
            "distance_metric": self.distance_metric,
            "scale_version": self.scale_version,
            "quantile_q": float(self.quantile_q),
            "voxel_size": float(self.voxel_size),
            "conf_warp_version": str(self.conf_warp_version),
            "n_valid_points": int(self.n_valid_points),
            "degradation_flags": list(self.degradation_flags),
            "metric_scale": self.metric_scale,
            "n_points_raw": int(self.n_points_raw),
            "n_nn_samples": int(self.n_nn_samples),
            "contamination_ratio": self.contamination_ratio,
            "duplicate_suspect": bool(self.duplicate_suspect),
            "conf_usage": str(self.conf_usage),
            "suppressed_distance_normalized": self.suppressed_distance_normalized,
            "audit": _jsonable(self.audit),
        }


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """把审计 dict 压成可 JSON 落盘的形式（numpy 标量 → python 标量，NaN → None）。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _params(params: Optional[Any]) -> DistancePrimitiveParams:
    """接受 `DistancePrimitiveParams` / dict / None（trace 回读友好）。"""
    if params is None:
        return DistancePrimitiveParams()
    if isinstance(params, DistancePrimitiveParams):
        return params
    if isinstance(params, Mapping):
        return DistancePrimitiveParams(
            quantile_q=params.get("quantile_q", QUANTILE_Q_DEFAULT),
            voxel_size=params.get("voxel_size", VOXEL_SIZE_DEFAULT),
            conf_warp_version=params.get("conf_warp_version", CONF_WARP_VERSION),
            n_min=params.get("n_min", N_MIN_DEFAULT),
        )
    raise TypeError(
        f"params 必须是 DistancePrimitiveParams / dict / None，收到 {type(params).__name__}")


def _as_points(points: Any, name: str = "points") -> np.ndarray:
    """任意点集输入 → `(N, 3) float64`（支持 `(N,3)` / `(H,W,3)` / 嵌套 list）。"""
    a = np.asarray(points, dtype=np.float64)
    if a.ndim < 2 or a.shape[-1] != 3:
        raise ValueError(
            f"{name} 必须是 (N,3) 或 (H,W,3) 的点集（末维=3），实际 shape={a.shape}")
    return np.ascontiguousarray(a.reshape(-1, 3), dtype=np.float64)


def _as_vector3(v: Any, name: str) -> np.ndarray:
    """3 维向量入参校验（非 3 维或非有限 → 抛错，不静默兜底）。"""
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.shape != (3,):
        raise ValueError(f"{name} 必须是 3 维向量（收到 shape={np.asarray(v).shape}）")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} 含 NaN/Inf（收到 {v!r}）：参考点非法 → 拒算，不猜")
    return a


def _flags(*items: Optional[str]) -> list[str]:
    """降级标记去重 + 定序（确定性：同一情形两次调用给出同一列表）。"""
    return sorted({str(i) for i in items if i})


def _robust_scale(points: np.ndarray) -> float:
    """点集稳健尺度 = 各点到**中位中心**距离的中位数（无量纲，与 §10.1 同口径）。

    与 VGGT 的任意归一化倍数无关（换尺度不改判据），**绝不**解释成米。
    """
    if points.shape[0] == 0:
        return float("nan")
    center = np.median(points, axis=0)
    rad = np.linalg.norm(points - center, axis=1)
    return float(np.median(rad))


def _self_spacing(points: np.ndarray, tree: Optional[cKDTree] = None) -> Optional[float]:
    """表面采样间距 = 集合内**最近邻距离**的中位数（k=2 取第 2 近，排除自身）。

    `TAU_CONTAM` 乘的就是它（§12.4 污染判据的无量纲基准）；
    点数 < 2 或间距退化 → `None`（调用方按"判据不可用"处理，不假装不污染）。
    """
    n = int(points.shape[0])
    if n < 2:
        return None
    t = tree if tree is not None else cKDTree(points)
    dist, _ = t.query(points, k=2)
    s = float(np.median(np.asarray(dist)[:, 1]))
    return s if np.isfinite(s) and s > 0 else None


def _weighted_quantile(values: np.ndarray, q: float,
                       weights: Optional[np.ndarray] = None) -> float:
    """低分位（加权版走 Hazen 型绘图位置线性插值；不含权重时与 `np.quantile` 一致）。

    - 无权重：`np.quantile(v, q, method="linear")`（`q=0` 精确等于最小值）；
    - 有权重：按权重累积插值，保证 `q=0 → min`、`q=1 → max`，且权重全部相同时
      与无权重的结果同量级（软权重只微调分位位置，不改变语义）。
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return float("nan")
    if weights is None:
        return float(np.quantile(v, q, method="linear"))
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != v.shape or not np.all(np.isfinite(w)) or float(w.sum()) <= 0.0:
        return float(np.quantile(v, q, method="linear"))
    order = np.argsort(v, kind="stable")
    vs, ws = v[order], np.clip(w[order], 0.0, None)
    total = float(ws.sum())
    if total <= 0.0:
        return float(np.quantile(v, q, method="linear"))
    cw = np.cumsum(ws)
    pos = (cw - 0.5 * ws) / total          # Hazen 绘图位置 ∈ (0,1)，单调不减
    return float(np.interp(float(q), pos, vs))


def _conf_weights(
    conf: Optional[Any],
    monotonic: Optional[bool],
    *,
    n_expected: int,
    optional_mask: bool,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str, list[str]]:
    """§10.3/§12.2-2 conf 纪律：只作软权重，且**必须先过 conf-warp 单调性自检**。

    返回 `(weights|None, keep_mask|None, usage, notes)`：

    - `conf is None` → `("disabled", [])`：没有 conf 输入，不用它；
    - `monotonic is None`（未自检）→ `("ignored_unverified", ...)`：**完全不用 conf**
      （既不过滤也不加权）。未知 ≠ 坏值，但更 ≠ 可用的权重；
    - `monotonic is False`（自检未过）→ **降权**：`w ← w**CONF_DOWNWEIGHT_POWER`
      向均匀压平，**不丢弃任何点**（§12.2-2 明写"降权而非丢弃"）；
    - `monotonic is True`（自检通过）→ 可作软权重；`optional_mask=True` 时**额外**
      给出 `C>2` 掩码（掩码只在这条分支里存在，且从不当硬门用）。

    软权重一律夹到 `[CONF_WEIGHT_FLOOR, ∞)` 再归一到均值 1：conf 只调整各点相对
    话语权，不把点权重置零（否则"软权重"就变成硬门了）。
    """
    if conf is None:
        return None, None, "disabled", []
    c = np.asarray(conf, dtype=np.float64).reshape(-1)
    if c.size != n_expected:
        raise ValueError(
            f"point_conf 长度 {c.size} 与点数 {n_expected} 不一致（必须逐点对齐）")
    if monotonic is None:
        return None, None, "ignored_unverified", [
            "conf_warp_monotonic=None（未自检）→ 完全不用 point_conf（§12.2-2/§10.3）"]
    notes: list[str] = []
    finite = np.isfinite(c)
    if not np.all(finite):
        med = float(np.median(c[finite])) if np.any(finite) else 1.0
        notes.append(
            f"point_conf 含 {int((~finite).sum())} 个非有限值 → 用有限值中位数 {med:.6g} 顶替"
            "（软权重的中性填充，不丢弃点）")
        c = np.where(finite, c, med)
    w = np.clip(c, 0.0, None)
    mean_w = float(np.mean(w)) if w.size else 0.0
    if not np.isfinite(mean_w) or mean_w <= 0.0:
        return None, None, "unusable", notes + [
            "point_conf 全为 0/非正 → 无法形成权重，退回无权重口径"]
    w = w / mean_w
    usage = "weighted"
    if monotonic is False:
        w = w ** float(CONF_DOWNWEIGHT_POWER)
        w = w / float(np.mean(w))
        usage = "downweighted"
        notes.append(
            f"conf_warp 单调自检未过 → conf 只降权（w**{CONF_DOWNWEIGHT_POWER}），不丢弃点"
            "（§10.3/§12.2-2）")
    w = np.clip(w, float(CONF_WEIGHT_FLOOR), None)
    w = w / float(np.mean(w))
    keep: Optional[np.ndarray] = None
    if optional_mask:
        if monotonic is True:
            keep = (c >= float(CONF_OPTIONAL_MASK_C))
            usage = "masked_optional"
            notes.append(
                f"启用 C>={CONF_OPTIONAL_MASK_C:g} **可选掩码**（§10.3：可选，不是硬阈值）"
                f"，滤掉 {int((~keep).sum())} 点")
        else:
            notes.append("conf-warp 自检未过 → 拒绝启用 C>2 可选掩码（§10.3）")
    return w, keep, usage, notes


def _voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
    """§12.2-1 体素降采样：每个体素取**离体素质心最近的真实点**（确定性）。

    为什么取真实点而不是体素质心：质心会被"凹面/薄壳"拉离表面（甚至落到物体内部），
    而低分位距离要的是**表面**；取真实点还天然保持"点集＝表面样本"的语义。

    确定性：`np.lexsort` 三级键（体素 → 到质心距离 → 原始索引）定序，
    组内平局取**最小原始索引**；不依赖字典序、不依赖 RNG。
    """
    n = int(points.shape[0])
    info = {"voxel_applied": False, "n_voxels": n, "n_points_in": n}
    if n == 0 or not np.isfinite(voxel_size) or voxel_size <= 0.0:
        info["reason"] = "voxel_disabled"
        return points, weights, info
    origin = points.min(axis=0)
    keys = np.floor((points - origin) / float(voxel_size)).astype(np.int64)
    idx = np.arange(n, dtype=np.int64)
    order = np.lexsort((idx, keys[:, 2], keys[:, 1], keys[:, 0]))
    k_sorted = keys[order]
    p_sorted = points[order]
    new_group = np.ones(n, dtype=bool)
    new_group[1:] = np.any(k_sorted[1:] != k_sorted[:-1], axis=1)
    group_id = np.cumsum(new_group) - 1
    n_groups = int(group_id[-1]) + 1
    counts = np.bincount(group_id, minlength=n_groups)
    centroids = np.empty((n_groups, 3), dtype=np.float64)
    for ax in range(3):
        centroids[:, ax] = np.bincount(group_id, weights=p_sorted[:, ax],
                                       minlength=n_groups) / counts
    d2 = np.einsum("ij,ij->i", p_sorted - centroids[group_id],
                   p_sorted - centroids[group_id])
    # 组内取 d2 最小者；平局取最小原始索引（idx_sorted 随 order 单调信息保留）
    order2 = np.lexsort((idx[order], d2, group_id))
    g_sorted = group_id[order2]
    first = np.ones(g_sorted.size, dtype=bool)
    first[1:] = g_sorted[1:] != g_sorted[:-1]
    sel = np.sort(order2[first])          # 按体素字典序输出（确定性定序）
    pts_out = p_sorted[sel]
    info.update({"voxel_applied": True, "n_voxels": n_groups,
                 "n_points_in": n, "reason": ""})
    w_out = None
    if weights is not None:
        w_sorted = np.asarray(weights, dtype=np.float64).reshape(-1)[order]
        w_out = w_sorted[sel]
    return pts_out, w_out, info


def preprocess_points(
    points: Any,
    *,
    params: Optional[Any] = None,
    point_conf: Optional[Any] = None,
    conf_warp_monotonic: Optional[bool] = None,
    conf_optional_mask: bool = False,
) -> dict:
    """§12.2 统一预处理（所有距离原语共用）—— 独立暴露，便于 Tool 层与测试复核。

    返回 `{"points", "weights", "n_raw", "n_finite", "n_used", "voxel_size",
    "degraded", "degradation_flags", "conf_usage", "robust_scale", "notes", "audit"}`。

    顺序（§12.2 逐条对齐）：排除 NaN/Inf → conf 可选掩码 → conf 软权重 →
    体素降采样 → `N_min` 判定。`N_min` 判的是**最终参与计算的点数**
    （降采样之后），因为分位数的样本量就是它。
    """
    p = _params(params)
    pts_raw = _as_points(points)
    n_raw = int(pts_raw.shape[0])
    notes: list[str] = []
    empty = np.zeros((0, 3), dtype=np.float64)
    if n_raw == 0:
        return {
            "points": empty, "weights": None, "n_raw": 0, "n_finite": 0, "n_used": 0,
            "voxel_size": float(p.voxel_size), "degraded": True,
            "degradation_flags": ["degraded"], "conf_usage": "disabled",
            "robust_scale": float("nan"), "notes": ["点集为空"], "audit": {},
        }

    finite = np.isfinite(pts_raw).all(axis=1)
    n_drop_nonfinite = int(n_raw - np.count_nonzero(finite))
    if n_drop_nonfinite:
        notes.append(f"排除 NaN/Inf 坐标 {n_drop_nonfinite}/{n_raw} 个（§12.2-3）")
    pts = pts_raw[finite]

    weights, keep_mask, conf_usage, conf_notes = _conf_weights(
        point_conf, conf_warp_monotonic, n_expected=n_raw, optional_mask=conf_optional_mask)
    notes.extend(conf_notes)
    n_masked = 0
    if keep_mask is not None:
        keep = keep_mask[finite]
        n_masked = int(np.count_nonzero(~keep))
        pts = pts[keep]
        if weights is not None:
            weights = weights[finite][keep]
    elif weights is not None:
        weights = weights[finite]

    n_finite = int(pts.shape[0])
    scale = _robust_scale(pts) if n_finite else float("nan")
    pts_ds, w_ds, vinfo = _voxel_downsample(pts, float(p.voxel_size), weights)
    n_used = int(pts_ds.shape[0])
    degraded = n_used < int(p.n_min)
    if degraded:
        notes.append(
            f"有效点 {n_used} < N_min={int(p.n_min)} → degraded（§12.2-4：不输出伪精确值）")
    voxel_rel = (float(p.voxel_size) / scale) if (np.isfinite(scale) and scale > 0) else float("nan")
    audit = {
        "n_raw": n_raw, "n_finite": n_finite, "n_used": n_used,
        "n_drop_nonfinite": n_drop_nonfinite, "n_masked_by_conf": n_masked,
        "n_min": int(p.n_min), "voxel_size": float(p.voxel_size),
        "voxel_applied": bool(vinfo.get("voxel_applied", False)),
        "n_voxels": int(vinfo.get("n_voxels", n_used)),
        "voxel_size_rel_to_scene_scale": voxel_rel,
        "robust_scale": float(scale) if np.isfinite(scale) else None,
        "conf_usage": conf_usage, "notes": notes,
    }
    return {
        "points": pts_ds, "weights": w_ds, "n_raw": n_raw, "n_finite": n_finite,
        "n_used": n_used, "voxel_size": float(p.voxel_size), "degraded": degraded,
        "degradation_flags": _flags("degraded" if degraded else None),
        "conf_usage": conf_usage, "robust_scale": float(scale),
        "notes": notes, "audit": audit,
    }


def _metric_scale_check(metric_scale: Optional[float]) -> tuple[Optional[float], list[str]]:
    """米制换算系数：`None` = 无米制证据（正常，不标降级）；给了但非法 → 标记 + 不给米制值。"""
    if metric_scale is None:
        return None, []
    try:
        v = float(metric_scale)
    except (TypeError, ValueError):
        return None, ["metric_scale_invalid"]
    if not np.isfinite(v) or v <= 0.0:
        return None, ["metric_scale_invalid"]
    return v, []


def _contamination(
    nn_distances: np.ndarray,
    *,
    reference_spacing: Optional[float],
    tau: float,
    fraction: float,
) -> tuple[Optional[float], bool, str]:
    """§12.4-3 污染判据：极近距离（`< tau × 表面采样间距`）占比异常高 ⇒ mask 泄漏。

    返回 `(极近距离占比|None, 是否可疑, 说明)`。基准间距来自点云自身（无量纲），
    故与 VGGT 的任意归一化倍数无关，**不是米制阈值**（§10.1 纪律）。
    基准不可用（点数太少/间距退化）→ 占比 `None`、可疑 `False`：宁可"判不了"，
    也不把"没有证据"说成"干净"（判不了的情形会写进 `audit`）。
    """
    d = np.asarray(nn_distances, dtype=np.float64).reshape(-1)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None, False, "无有效 NN 距离"
    if reference_spacing is None or not np.isfinite(reference_spacing) or reference_spacing <= 0:
        return None, False, "表面采样间距不可用 → 污染判据判不了（不假装干净）"
    thr = float(tau) * float(reference_spacing)
    ratio = float(np.mean(d <= thr))
    suspect = ratio > float(fraction)
    return ratio, suspect, (
        f"极近距离(< {tau:g}×{reference_spacing:.6g} = {thr:.6g}) 占比 {ratio:.3f}"
        f"（门 {fraction:g}）")


def _result(
    *,
    value: Optional[float],
    metric: Optional[float],
    prep: dict,
    params: DistancePrimitiveParams,
    flags: Iterable[Optional[str]],
    scale_version: Optional[str],
    metric_scale: Optional[float],
    n_nn_samples: int = 0,
    contamination_ratio: Optional[float] = None,
    duplicate_suspect: bool = False,
    suppressed: Optional[float] = None,
    extra_audit: Optional[dict] = None,
) -> DistanceResult:
    """组装 `DistanceResult`（统一带上 §12.5 的 trace 字段 + 预处理审计）。"""
    audit = dict(prep.get("audit", {}))
    if extra_audit:
        audit.update(extra_audit)
    return DistanceResult(
        distance_normalized=None if value is None else float(value),
        distance_metric=None if metric is None else float(metric),
        n_valid_points=int(prep.get("n_used", 0)),
        degradation_flags=_flags(*flags),
        quantile_q=float(params.quantile_q),
        voxel_size=float(params.voxel_size),
        conf_warp_version=str(params.conf_warp_version),
        scale_version=scale_version,
        metric_scale=metric_scale,
        n_points_raw=int(prep.get("n_raw", 0)),
        n_nn_samples=int(n_nn_samples),
        contamination_ratio=contamination_ratio,
        duplicate_suspect=bool(duplicate_suspect),
        conf_usage=str(prep.get("conf_usage", "disabled")),
        suppressed_distance_normalized=None if suppressed is None else float(suppressed),
        audit=audit,
    )


# ---------------------------------------------------------------------------
# §12.1 距离原语之一：点集 → 参考点（相机/质心辅助测量）
# ---------------------------------------------------------------------------

def robust_distance_to_reference(
    points: Any,
    reference_xyz: Any,
    *,
    metric_scale: Optional[float] = None,
    params: Optional[Any] = None,
    point_conf: Optional[Any] = None,
    conf_warp_monotonic: Optional[bool] = None,
    duplicate_suspect: bool = False,
    conf_optional_mask: bool = False,
    scale_version: Optional[str] = None,
    tau_contam: float = TAU_CONTAM,
    th_contam_fraction: float = TH_CONTAM_FRACTION,
) -> DistanceResult:
    """§12.1：`quantile({||p − R||}, q)`——稳健低分位距离。

    `points` 是对象 mask 内 3D 点，`reference_xyz` 为调用方提供的相机中心或参考点。
    `metric_scale` 给出时 `distance_metric = distance_normalized × metric_scale`。
    VSI-Bench 题面对象间的相对/绝对距离由点集间原语提供，不以相机距离替代。

    参数
    ----
    metric_scale : 世界单位→米的换算系数（§11/D1）。`None` = 无米制证据 → `distance_metric=None`
        （**绝不**默认 1.0 冒充米制）；给了但非有限/非正 → `metric_scale_invalid` + 米制值 None。
    point_conf : VGGT 逐点置信度（与 `points` 逐点对齐；`(N,)` 或 `(H,W)`）。
    conf_warp_monotonic : conf-warp 单调性自检结果（§10.3）。`None`=未自检 → 不用 conf；
        `False`=未过 → 降权；`True`=通过 → 软权重（可选掩码需 `conf_optional_mask=True`）。
    duplicate_suspect : 调用方已知"两对象 mask 高重叠/疑似重复绑定"（§12.4-2）→ 原样标进
        `degradation_flags`（是否据此降级由 Tool 层决定）。
    scale_version : 米制尺度来源版本（§12.5 进 trace）；缺省用 §11 的度量融合版本号。

    返回
    ----
    `DistanceResult`。**降级纪律**：有效点 < `N_min` → `distance_normalized=None` +
    `degraded`；极近距离占比超门 → `point_contamination_suspect` 且**不输出精确值**
    （被压制值在 `audit["suppressed_distance_normalized"]`）。

    参考点非 3 维或含 NaN/Inf → 抛 `ValueError`（参考点非法属于调用方 bug，不静默降级）。
    """
    p = _params(params)
    ref = _as_vector3(reference_xyz, "reference_xyz")
    prep = preprocess_points(points, params=p, point_conf=point_conf,
                             conf_warp_monotonic=conf_warp_monotonic,
                             conf_optional_mask=conf_optional_mask)
    flags: list[Optional[str]] = list(prep["degradation_flags"])
    if duplicate_suspect:
        flags.append("suspect_duplicate")
    mscale, mflags = _metric_scale_check(metric_scale)
    flags.extend(mflags)
    sv = (scale_version or SCALE_VERSION_DEFAULT) if mscale is not None else None

    if prep["degraded"]:
        return _result(value=None, metric=None, prep=prep, params=p, flags=flags,
                       scale_version=sv, metric_scale=mscale,
                       duplicate_suspect=duplicate_suspect,
                       extra_audit={"reason": "insufficient_points"})

    pts = prep["points"]
    dist = np.linalg.norm(pts - ref[None, :], axis=1)
    ratio, suspect, note = _contamination(
        dist, reference_spacing=_self_spacing(pts), tau=tau_contam,
        fraction=th_contam_fraction)
    if suspect:
        # §12.4-3：mask 泄漏（对象点贴死在参考点上，例如观察者自身被并进 mask）
        flags.append("point_contamination_suspect")
    raw = _weighted_quantile(dist, p.quantile_q, prep["weights"])
    value = None if suspect else raw           # §12.4：污染不作精确值输出
    metric = (value * mscale) if (value is not None and mscale is not None) else None
    return _result(
        value=value, metric=metric, prep=prep, params=p, flags=flags,
        scale_version=sv, metric_scale=mscale,
        contamination_ratio=ratio, duplicate_suspect=duplicate_suspect,
        suppressed=(raw if suspect else None),
        extra_audit={"contamination_note": note,
                     "reference_xyz": [float(x) for x in ref],
                     "distance_min": float(np.min(dist)),
                     "distance_quantile_q": float(raw)},
    )


# ---------------------------------------------------------------------------
# §9.7 距离原语之二：点集 ↔ 点集（双向 NN，低分位）
# ---------------------------------------------------------------------------

def robust_distance_between_pointsets(
    points_a: Any,
    points_b: Any,
    *,
    params: Optional[Any] = None,
    metric_scale: Optional[float] = None,
    point_conf_a: Optional[Any] = None,
    point_conf_b: Optional[Any] = None,
    conf_warp_monotonic_a: Optional[bool] = None,
    conf_warp_monotonic_b: Optional[bool] = None,
    conf_optional_mask: bool = False,
    duplicate_suspect: bool = False,
    scale_version: Optional[str] = None,
    tau_contam: float = TAU_CONTAM,
    th_contam_fraction: float = TH_CONTAM_FRACTION,
) -> DistanceResult:
    """§9.7 `surface_distance_between_objects` 的内核：**双向 NN 距离集合的低分位**。

    对 A 每点在 B 中查最近邻（KD-tree），反之亦然，距离集合取低分位
    `quantile(D_AB ∪ D_BA, q)`；复杂度 `O((n_A+n_B)·log·max)`，
    **不做全笛卡尔积**（`n_A × n_B` 矩阵在本模块里根本不会出现）。

    **定位**（v7 §2.2）：对象↔对象表面距离代理；相对距离工具比较题面参照对象
    与各候选实例，候选完整性和排名唯一性由 Tool 层校验。

    污染判据（§12.4-3）用**双向 NN 集合自身**：极近距离占比（基准 = 两侧集合内采样
    间距的中位数）超门 ⇒ `point_contamination_suspect`，此时不输出精确值
    （两个 mask 互相咬合/重复绑定会表现为此形态）。

    任一集合有效点 < `N_min` → `degraded`、`distance_normalized=None`。
    """
    p = _params(params)
    prep_a = preprocess_points(points_a, params=p, point_conf=point_conf_a,
                               conf_warp_monotonic=conf_warp_monotonic_a,
                               conf_optional_mask=conf_optional_mask)
    prep_b = preprocess_points(points_b, params=p, point_conf=point_conf_b,
                               conf_warp_monotonic=conf_warp_monotonic_b,
                               conf_optional_mask=conf_optional_mask)
    merged = {
        "n_used": int(prep_a["n_used"] + prep_b["n_used"]),
        "n_raw": int(prep_a["n_raw"] + prep_b["n_raw"]),
        "conf_usage": f"A:{prep_a['conf_usage']}|B:{prep_b['conf_usage']}",
        "audit": {"a": prep_a["audit"], "b": prep_b["audit"]},
    }
    flags: list[Optional[str]] = list(prep_a["degradation_flags"]) + \
        list(prep_b["degradation_flags"])
    if duplicate_suspect:
        flags.append("suspect_duplicate")
    mscale, mflags = _metric_scale_check(metric_scale)
    flags.extend(mflags)
    sv = (scale_version or SCALE_VERSION_DEFAULT) if mscale is not None else None

    if prep_a["degraded"] or prep_b["degraded"]:
        return _result(value=None, metric=None, prep=merged, params=p, flags=flags,
                       scale_version=sv, metric_scale=mscale,
                       duplicate_suspect=duplicate_suspect,
                       extra_audit={"reason": "insufficient_points",
                                    "n_used_a": prep_a["n_used"],
                                    "n_used_b": prep_b["n_used"]})

    a, b = prep_a["points"], prep_b["points"]
    tree_a, tree_b = cKDTree(a), cKDTree(b)
    d_ab, _ = tree_b.query(a, k=1)          # A 的点 → B 的最近面
    d_ba, _ = tree_a.query(b, k=1)          # B 的点 → A 的最近面
    nn = np.concatenate([np.asarray(d_ab, dtype=np.float64),
                         np.asarray(d_ba, dtype=np.float64)])
    w = None
    wa, wb = prep_a["weights"], prep_b["weights"]
    if wa is not None or wb is not None:
        wa = np.ones_like(a[:, 0]) if wa is None else np.asarray(wa, dtype=np.float64)
        wb = np.ones_like(b[:, 0]) if wb is None else np.asarray(wb, dtype=np.float64)
        w = np.concatenate([wa, wb])
    spacing = _merged_spacing(a, b, tree_a, tree_b)
    ratio, suspect, note = _contamination(nn, reference_spacing=spacing,
                                          tau=tau_contam, fraction=th_contam_fraction)
    if suspect:
        flags.append("point_contamination_suspect")
    raw = _weighted_quantile(nn, p.quantile_q, w)
    value = None if suspect else raw            # §12.4：污染不作精确值输出
    metric = (value * mscale) if (value is not None and mscale is not None) else None
    return _result(
        value=value, metric=metric, prep=merged, params=p, flags=flags,
        scale_version=sv, metric_scale=mscale, n_nn_samples=int(nn.size),
        contamination_ratio=ratio, duplicate_suspect=duplicate_suspect,
        suppressed=(raw if suspect else None),
        extra_audit={
            "surface_distance_note": "双向 NN 低分位（§9.7）；对象间表面最近距离代理",
            "contamination_note": note,
            "reference_spacing": None if spacing is None else float(spacing),
            "n_used_a": prep_a["n_used"], "n_used_b": prep_b["n_used"],
            "distance_min": float(np.min(nn)),
            "distance_median": float(np.median(nn)),
            "distance_quantile_q": float(raw),
        },
    )


def _merged_spacing(a: np.ndarray, b: np.ndarray,
                    tree_a: cKDTree, tree_b: cKDTree) -> Optional[float]:
    """双侧集合内采样间距的中位数（污染判据的无量纲基准；两侧都要可用）。"""
    sa = _self_spacing(a, tree_a)
    sb = _self_spacing(b, tree_b)
    vals = [s for s in (sa, sb) if s is not None and np.isfinite(s) and s > 0]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


# ---------------------------------------------------------------------------
# §9.4 object_3d_extent
# ---------------------------------------------------------------------------

def object_extent(
    points: Any,
    *,
    metric_scale: Optional[float] = None,
    params: Optional[Any] = None,
    point_conf: Optional[Any] = None,
    conf_warp_monotonic: Optional[bool] = None,
    conf_optional_mask: bool = False,
    extent_tail_q: Optional[float] = None,
    scale_version: Optional[str] = None,
) -> dict:
    """§9.4 `object_3d_extent` 内核：对象点集的 **3 轴稳健 extent**（包围盒边长）。

    稳健口径：每轴取 `[q, 1−q]` 分位差（`q = extent_tail_q or params.quantile_q`），
    故飞点/离群点不会撑大 extent；`q=0` 退化为真包围盒（min/max）。
    世界轴对齐（与既有 `ObjectInstance.bbox` 语义一致；不做 PCA 旋转，
    旋转物体在轴对齐口径下会被高估——这是与现有 bbox 语义保持一致的选择）。

    米制（§9.4）：`extent_metric = extent_normalized × metric_scale`（逐轴）；
    **面积按平方**：`area_metric = area_normalized × metric_scale²`。
    无 `metric_scale` → 两个米制字段均为 `None`（fail-closed，不缺系数冒充）。

    返回 `{"extent_normalized", "extent_metric", "extent_longest_normalized",
    "extent_longest_metric", "area_normalized", "area_metric", "n_valid_points",
    + §12.5 trace 字段 + "degradation_flags" + "audit"}`；
    有效点 < `N_min` → `degraded` 且所有数值字段 `None`。
    """
    p = _params(params)
    prep = preprocess_points(points, params=p, point_conf=point_conf,
                             conf_warp_monotonic=conf_warp_monotonic,
                             conf_optional_mask=conf_optional_mask)
    q = float(p.quantile_q if extent_tail_q is None else extent_tail_q)
    if not np.isfinite(q) or q < 0.0 or q > 0.5:
        raise ValueError(f"extent_tail_q 必须落在 [0, 0.5]（收到 {extent_tail_q!r}）")
    mscale, mflags = _metric_scale_check(metric_scale)
    flags: list[Optional[str]] = list(prep["degradation_flags"]) + list(mflags)
    sv = (scale_version or SCALE_VERSION_DEFAULT) if mscale is not None else None
    base = {
        "extent_normalized": None, "extent_metric": None,
        "extent_longest_normalized": None, "extent_longest_metric": None,
        "area_normalized": None, "area_metric": None,
        "extent_tail_q": q,
        "n_valid_points": int(prep["n_used"]),
        "quantile_q": float(p.quantile_q), "voxel_size": float(p.voxel_size),
        "conf_warp_version": str(p.conf_warp_version), "scale_version": sv,
        "metric_scale": mscale,
        "degradation_flags": _flags(*flags),
    }
    if prep["degraded"]:
        base["audit"] = {**prep["audit"], "reason": "insufficient_points"}
        return base
    pts = prep["points"]
    lo = np.quantile(pts, q, axis=0, method="linear")
    hi = np.quantile(pts, 1.0 - q, axis=0, method="linear")
    ext = np.clip(hi - lo, 0.0, None)
    order = np.argsort(-ext, kind="stable")          # 最长的两个轴（面积用）
    area = float(ext[order[0]] * ext[order[1]])
    ext_list = [float(x) for x in ext]
    ext_metric = ([float(x) * mscale for x in ext] if mscale is not None else None)
    longest = float(ext[order[0]])
    area_metric = (area * mscale * mscale) if mscale is not None else None
    base.update({
        "extent_normalized": ext_list,
        "extent_metric": ext_metric,
        "extent_longest_normalized": longest,
        "extent_longest_metric": (longest * mscale) if mscale is not None else None,
        "area_normalized": area,
        "area_metric": area_metric,
        "audit": {**prep["audit"],
                  "bbox_low": [float(x) for x in lo], "bbox_high": [float(x) for x in hi],
                  "note": "世界轴对齐稳健 extent；面积按平方换算（§9.4）"},
    })
    return base


# ---------------------------------------------------------------------------
# §9.5 房间几何：平面/地面/墙（planarity_and_ground ← room_size_from_planes 共用）
# ---------------------------------------------------------------------------

def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """TLS 平面拟合（SVD）：返回 `(单位法向 n, 偏移 d)`，使 `n·x ≈ d`。"""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c[None, :], full_matrices=False)
    n = np.asarray(vt[-1], dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("平面拟合退化：点集无法张成 2 维（法向未定义）")
    n = n / norm
    return n, float(np.dot(n, c))


def _planarity(points: np.ndarray) -> Optional[float]:
    """平面性（本模块口径）= `1 − λ1/λ3`（λ1≤λ2≤λ3 为协方差特征值），∈[0,1]。

    1 = 完美平面（法向厚度为零），0 = 各向同性团块。
    **不采用** PCL 的 `(λ2−λ1)/λ3`：它把"长条形平面"判低（λ2/λ3 在 5:4 的矩形上
    只有 0.64），而地板/路面本来就常是长条形 —— 这里要回答的是"平不平"，不是"方不方"。
    点不足或退化 → `None`。
    """
    if points.shape[0] < N_MIN_PLANE_FIT:
        return None
    cov = np.cov(points.T, bias=True)
    if not np.all(np.isfinite(cov)):
        return None
    ev = np.sort(np.linalg.eigvalsh(cov))
    l1, l3 = (float(ev[0]), float(ev[2]))
    if l3 <= MIN_SPACING_EPS:
        return None
    return float(np.clip(1.0 - l1 / l3, 0.0, 1.0))


def _orient_up(points: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, bool]:
    """给"上"方向定符号：地面一侧点更密（地板面积/采样通常大于天花板）。

    确定性规则：比较 `h = points·up` 两端各 `GROUND_BAND_Q` 分位带内的点数，
    高侧更多则翻转。不可判定（两侧相等）→ 保持原符号（不猜）。
    """
    h = points @ up
    lo = float(np.quantile(h, GROUND_BAND_Q, method="linear"))
    hi = float(np.quantile(h, 1.0 - GROUND_BAND_Q, method="linear"))
    n_low = int(np.count_nonzero(h <= lo))
    n_high = int(np.count_nonzero(h >= hi))
    if n_high > n_low:
        return -up, True
    return up, False


def _in_plane_basis(points: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """由水平面内 PCA 定出两轴（e1 为水平主轴），返回 `(e1, e2, 各向异性比)`。

    符号约定（确定性）：使每轴**绝对值最大的分量取正**；轴向按特征值降序。
    """
    proj = points - np.outer(points @ up, up)        # 投到与 up 垂直的平面
    cov = np.cov(proj.T, bias=True)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(-ev, kind="stable")
    e1 = np.asarray(evec[:, order[0]], dtype=np.float64)
    e2 = np.asarray(evec[:, order[1]], dtype=np.float64)
    for e in (e1, e2):
        k = int(np.argmax(np.abs(e)))
        if e[k] < 0:
            e *= -1.0
    e1 = e1 - float(np.dot(e1, up)) * up             # 数值上再正交化一次
    e1 = e1 / max(float(np.linalg.norm(e1)), MIN_SPACING_EPS)
    e2 = np.cross(up, e1)
    e2 = e2 / max(float(np.linalg.norm(e2)), MIN_SPACING_EPS)
    lam = np.sort(ev)[::-1]
    aniso = float(lam[0] / lam[1]) if (lam.size > 1 and lam[1] > MIN_SPACING_EPS) else float("inf")
    return e1, e2, aniso


def planarity_and_ground(
    points: Any,
    *,
    params: Optional[Any] = None,
    point_conf: Optional[Any] = None,
    conf_warp_monotonic: Optional[bool] = None,
    conf_optional_mask: bool = False,
    up_hint: Optional[Any] = None,
) -> dict:
    """平面性 + 地面/天花/墙平面（供 §9.5 房间尺寸与 §9.10 连通性共用）。

    步骤（确定性、无 RANSAC 随机）：

    1. `up`：优先用调用方给的 `up_hint`（D5：世界系契约里的 `world_up`，单一事实源）；
       否则用点云协方差**最小方差轴**（房间场景里竖直轴方差最小，等价于既有
       `_up_axis` 的"最小 extent 轴"但更稳），并按 `_orient_up` 定符号；
       随后用**地面平面法向精修** `up`（PCA 的最小方差轴只有 ~0.5° 量级精度，
       会把大房间远端的地面点挤出内点带）；
    2. 地面：取高度最低 `GROUND_BAND_Q` 分位的候选带做 TLS 平面拟合，再用
       `PLANE_TOL_REL × 稳健尺度` 的内点带迭代精修；法向与 `up` 的对齐低于
       `PLANE_MIN_ALIGN_COS` → `ground_plane_axis_mismatch`（"这不是地板"）；
    3. 天花：最高的同一分位带同样处理（不合法向 → `ceiling=None`）；
    4. 墙：水平面内 PCA 定两轴，四张墙面 = 两轴的稳健极值平面；
    5. `plane_inlier_ratio` = **地面**平面内点带内的有效点占比（§9.5 的拟合质量关心的是
       定义了房间与可走面的那张平面；墙内点率另见 `wall_inlier_ratio` 与审计里的
       `room_surface_inlier_ratio`。注意地面占比在真实房间里通常只有 0.2–0.5，
       拿它当"是否房间"的门时请用 `fit_quality` 而不是裸比阈值）；
       `fit_quality = sqrt(floor_fit_ratio × 地面带 planarity)`。其中
       `floor_fit_ratio = 地面平面内点数 / 地面候选带点数` —— 回答"**看起来像地板的点
       里有多少真的被这张平面解释**"，与"地板占整幅点云多少"无关。

       > 2026-09-21 实测修正：原式用 `plane_inlier_ratio`（地板点 / 全部点）作因子，
       > 于是 `fit_quality` 被"地板占点云比例"主导 —— 真实室内点云里地板只占
       > 2%–11%（其余是墙/天花/家具），导致拟合**准确**（房间面积与 GT 差 4–19%）
       > 的场景 `fit_quality` 也只有 0.14–0.34，低于门限 0.3 → 模型据它 abstain，
       > 把本来能拿分的题丢掉。`plane_inlier_ratio` 仍按 §9.5 原样上报（地板占比
       > 本身是有用信息），只是不再拿它当"拟合质量"。

    返回
    ----
    `{"up", "up_axis", "up_source", "up_anisotropy", "ground_plane", "floor_height_normalized",
    "ceiling_height_normalized", "room_height_normalized", "wall_directions",
    "horizontal_extent_normalized", "wall_inlier_ratio", "plane_inlier_ratio",
    "ground_planarity", "fit_quality", "n_valid_points", + §12.5 trace 字段 +
    "degradation_flags" + "audit"}`。

    点不足/几何退化 → 相应字段 `None` + `degraded`（绝不用 0 或假平面搪塞）。
    """
    p = _params(params)
    prep = preprocess_points(points, params=p, point_conf=point_conf,
                             conf_warp_monotonic=conf_warp_monotonic,
                             conf_optional_mask=conf_optional_mask)
    flags: list[Optional[str]] = list(prep["degradation_flags"])
    out: dict = {
        "up": None, "up_axis": None, "up_source": None, "up_flipped": False,
        "up_anisotropy": None,
        "ground_plane": None, "floor_height_normalized": None,
        "ceiling_height_normalized": None, "room_height_normalized": None,
        "wall_directions": None, "horizontal_extent_normalized": None,
        "wall_inlier_ratio": None, "plane_inlier_ratio": None,
        "ground_planarity": None, "fit_quality": None,
        "n_valid_points": int(prep["n_used"]),
        "quantile_q": float(p.quantile_q), "voxel_size": float(p.voxel_size),
        "conf_warp_version": str(p.conf_warp_version), "scale_version": None,
        "degradation_flags": [],
    }
    if prep["degraded"]:
        out["degradation_flags"] = _flags(*flags)
        out["audit"] = {**prep["audit"], "reason": "insufficient_points"}
        return out

    pts = prep["points"]
    scale = float(prep["robust_scale"])
    if not np.isfinite(scale) or scale <= 0:
        out["degradation_flags"] = _flags(*flags, "degraded")
        out["audit"] = {**prep["audit"], "reason": "degenerate_scale"}
        return out
    tol = float(PLANE_TOL_REL) * scale

    # ---- 1. up 方向 ----
    anisotropy = float("nan")
    if up_hint is not None:
        up = _as_vector3(up_hint, "up_hint")
        n = float(np.linalg.norm(up))
        if n <= MIN_SPACING_EPS:
            raise ValueError("up_hint 是零向量：上方向未定义 → 拒算（fail-closed）")
        up = up / n
        out["up_source"] = "provided"
    else:
        cov = np.cov(pts.T, bias=True)
        ev, evec = np.linalg.eigh(cov)
        up = np.asarray(evec[:, 0], dtype=np.float64)      # 最小方差轴
        lam = np.sort(ev)
        anisotropy = float(lam[1] / lam[0]) if lam[0] > MIN_SPACING_EPS else float("inf")
        if np.isfinite(anisotropy) and anisotropy < MIN_UP_AXIS_ANISOTROPY:
            flags.append("up_axis_ambiguous")
        out["up_source"] = "pca_min_variance"
    up, flipped = _orient_up(pts, up)
    out["up"] = [float(x) for x in up]
    out["up_flipped"] = bool(flipped)
    out["up_axis"] = int(np.argmax(np.abs(up)))
    out["up_anisotropy"] = None if not np.isfinite(anisotropy) else float(anisotropy)
    up_initial = up.copy()

    # ---- 2. 地面平面 ----
    h = pts @ up
    h_lo = float(np.quantile(h, GROUND_BAND_Q, method="linear"))
    band_tol = max(tol, PLANE_BAND_TOL_MULT * tol)
    near_floor = np.abs(h - h_lo) <= band_tol
    if int(np.count_nonzero(near_floor)) < N_MIN_PLANE_FIT:
        flags.extend(["no_ground_plane", "degraded"])
        out["degradation_flags"] = _flags(*flags)
        out["audit"] = {**prep["audit"], "reason": "ground_band_too_small"}
        return out
    n_g, d_g = _fit_plane(pts[near_floor])
    for _ in range(PLANE_REFIT_ITERS):                    # 内点带定值迭代精修
        inl = near_floor & (np.abs(pts @ n_g - d_g) <= tol)
        if int(np.count_nonzero(inl)) < N_MIN_PLANE_FIT:
            break
        n_g, d_g = _fit_plane(pts[inl])
    # 说明（2026-09-21）：曾试过把内点带按残差 MAD 自适应放宽，未采纳 ——
    # 那会在**无地板点团**上把内点率抬到跟真实房间一样高，削弱负例分离度，
    # 属未标定的度量改动（红线 2）。当前 `tol` 保持 PLANE_TOL_REL × 场景尺度，
    # 其标定登记在 §10.6。
    align = abs(float(np.dot(n_g, up)))
    if align < float(PLANE_MIN_ALIGN_COS):
        flags.extend(["ground_plane_axis_mismatch", "degraded"])
        out["ground_plane"] = {"normal": [float(x) for x in n_g], "offset": float(d_g),
                               "inlier_ratio": None, "n_inliers": 0,
                               "axis_alignment": float(align)}
        out["degradation_flags"] = _flags(*flags)
        out["audit"] = {**prep["audit"], "reason": "ground_plane_not_horizontal"}
        return out
    if float(np.dot(n_g, up)) < 0:                        # 法向统一朝上（可复现）
        n_g, d_g = -n_g, -d_g
    # `up` 精修：地面是场景里最大的一张平面，其 TLS 法向精度远高于 PCA 最小方差轴
    # （后者只有 ~0.5° 量级），不精修的话大房间远端的地面点会掉出内点带。
    up = n_g
    h = pts @ up
    out["up"] = [float(x) for x in up]
    out["up_axis"] = int(np.argmax(np.abs(up)))
    out["up_source"] = str(out["up_source"]) + "+ground_refined"
    out["up_flipped"] = bool(flipped or np.dot(up, up_initial) < 0.0)
    ground_in = np.abs(pts @ n_g - d_g) <= tol
    n_ground = int(np.count_nonzero(ground_in))
    n_near_floor = int(np.count_nonzero(near_floor))
    ground_plane = {
        "normal": [float(x) for x in n_g], "offset": float(d_g),
        "inlier_ratio": float(n_ground / pts.shape[0]), "n_inliers": n_ground,
        "n_near_floor": n_near_floor,
        "axis_alignment": float(align),
        # 审计：实际使用的平面内点容差（相对场景尺度），便于事后标定与复核
        "inlier_tol": float(tol),
        "tol_relative_to_scale": float(tol / scale) if scale else None,
    }
    out["ground_plane"] = ground_plane
    out["floor_height_normalized"] = float(np.median(h[ground_in]))
    out["ground_planarity"] = _planarity(pts[ground_in])

    # ---- 3. 天花平面（同一分位带的另一端；不合法向即 None）----
    h_hi = float(np.quantile(h, 1.0 - GROUND_BAND_Q, method="linear"))
    near_ceil = np.abs(h - h_hi) <= band_tol
    if int(np.count_nonzero(near_ceil)) >= N_MIN_PLANE_FIT:
        try:
            n_c, d_c = _fit_plane(pts[near_ceil])
            if abs(float(np.dot(n_c, up))) >= float(PLANE_MIN_ALIGN_COS):
                ceil_in = np.abs(pts @ n_c - d_c) <= tol
                if int(np.count_nonzero(ceil_in)) >= N_MIN_PLANE_FIT:
                    out["ceiling_height_normalized"] = float(np.median(h[ceil_in]))
        except ValueError:
            pass
    if (out["ceiling_height_normalized"] is not None
            and out["floor_height_normalized"] is not None):
        out["room_height_normalized"] = float(out["ceiling_height_normalized"]
                                              - out["floor_height_normalized"])

    # ---- 4. 墙平面（水平两轴上的稳健极值）----
    e1, e2, aniso_h = _in_plane_basis(pts, up)
    c1 = pts @ e1
    c2 = pts @ e2
    q = float(p.quantile_q)
    c1_lo, c1_hi = (float(np.quantile(c1, q, method="linear")),
                    float(np.quantile(c1, 1.0 - q, method="linear")))
    c2_lo, c2_hi = (float(np.quantile(c2, q, method="linear")),
                    float(np.quantile(c2, 1.0 - q, method="linear")))
    d1 = max(c1_hi - c1_lo, 0.0)
    d2 = max(c2_hi - c2_lo, 0.0)
    above_floor = np.abs(h - out["floor_height_normalized"]) > tol
    wall_in = above_floor & ((np.abs(c1 - c1_lo) <= tol) | (np.abs(c1 - c1_hi) <= tol)
                             | (np.abs(c2 - c2_lo) <= tol) | (np.abs(c2 - c2_hi) <= tol))
    n_wall = int(np.count_nonzero(wall_in))
    out["wall_directions"] = [[float(x) for x in e1], [float(x) for x in e2]]
    out["horizontal_extent_normalized"] = [float(d1), float(d2)]
    out["wall_inlier_ratio"] = float(n_wall / pts.shape[0])

    n_covered = int(np.count_nonzero(ground_in | wall_in))
    # §9.5 的 `plane_inlier_ratio` = **地面**平面内点率（地面定义了房间与可走面；
    # 墙只作诊断，见 `wall_inlier_ratio` / 审计里的 `room_surface_inlier_ratio`）。
    out["plane_inlier_ratio"] = float(n_ground / pts.shape[0])
    planarity = out["ground_planarity"]
    fit_quality = None
    if planarity is not None:
        # 口径保持 §9.5 原设计：`sqrt(plane_inlier_ratio × planarity)` ——
        # "多少点落在地面平面上" × "地面本身平不平"。
        #
        # 2026-09-21 实测记录（**结论：不改公式，改用法**）：在真实 VGGT 点云上
        # 地板只占 2%–11%（其余是墙/天花/家具），于是**拟合准确**（房间面积与 GT
        # 差 4%–19%）的场景 `fit_quality` 也只有 0.14–0.34，低于 `TH_FIT_QUALITY_MIN`。
        # 但该公式在**合成房间**（地板占比大）与**无地板点团**上是正确分离的
        # （tests/unit/test_distance_primitives.py 的两个用例）—— 也就是说
        # "低值"不是公式算错，而是**阈值对真实点云未标定**。
        # 按红线 2（未标定阈值不得充当结论 / 不得为分数放宽阈值）：
        #   - **不**重定义公式（那等于为了让真实数据通过而改度量）；
        #   - **不**放宽 TH_FIT_QUALITY_MIN；
        #   - 真正造成丢分的是**模型拿它当 abstain 硬门**，已在工具描述里写明
        #     "仅供审计，不要拿它当 abstain 的硬门"，并在 §10.6 待标定清单里登记。
        fit_quality = float(np.sqrt(max(out["plane_inlier_ratio"], 0.0)
                                    * max(float(planarity), 0.0)))
        if fit_quality < float(TH_FIT_QUALITY_MIN):
            flags.append("plane_fit_low_quality")
    out["fit_quality"] = fit_quality
    out["degradation_flags"] = _flags(*flags)
    out["audit"] = {
        **prep["audit"],
        "plane_tol": float(tol), "robust_scale": float(scale),
        "n_ground_inliers": n_ground, "n_wall_inliers": n_wall,
        "room_surface_inlier_ratio": float(n_covered / pts.shape[0]),
        "wall_plane_offsets": {"u1": [c1_lo, c1_hi], "u2": [c2_lo, c2_hi]},
        "horizontal_anisotropy": None if not np.isfinite(aniso_h) else float(aniso_h),
        "note": ("地面/墙平面为确定性 TLS 拟合（无 RANSAC 随机）；"
                 "plane_inlier_ratio 是**地面**平面内点率，墙内点率见 wall_inlier_ratio"),
    }
    return out


def room_size_from_planes(
    point_map: Any,
    *,
    metric_scale: Optional[float] = None,
    params: Optional[Any] = None,
    point_conf: Optional[Any] = None,
    conf_warp_monotonic: Optional[bool] = None,
    conf_optional_mask: bool = False,
    up_hint: Optional[Any] = None,
    scale_version: Optional[str] = None,
) -> dict:
    """§9.5 `plane_fit_room_size` 内核：拟合地平面/墙平面 → 房间对角线/面积。

    - `room_diagonal_normalized`：**地面矩形对角线** `√(d1² + d2²)`（d1、d2 是地面平面内
      两主轴的稳健 extent）。选"楼面对角线"而不是"三维体对角线"：房间尺寸题问的是
      房间大小/面积，竖直维由墙高决定、与"房间多大"不是同一量；楼面对角线口径也
      与平面图(GT 来源)一致。三维 extent 与墙高都在 `audit`/`room_height_normalized` 里可查。
    - `room_area_normalized` = `d1 × d2`（地面矩形面积）；`room_area_m2` = 面积 ×
      `metric_scale²`（**面积按平方换算**，§9.4/§9.5）。无 `metric_scale` → `room_area_m2=None`。
    - `plane_inlier_ratio` / `fit_quality`：见 `planarity_and_ground`；`fit_quality` 低于
      `TH_FIT_QUALITY_MIN` → `plane_fit_low_quality`（§9.5"不过门时降级"，数值仍输出但带标记；
      真正不可测才置 `None` + `degraded`）。

    返回 `{"room_diagonal_normalized", "room_diagonal_metric", "room_area_normalized",
    "room_area_m2", "room_height_normalized", "wall_extents_normalized",
    "plane_inlier_ratio", "fit_quality", "ground_planarity", "n_valid_points",
    + §12.5 trace 字段 + "degradation_flags" + "audit"}`。
    """
    p = _params(params)
    ground = planarity_and_ground(point_map, params=p, point_conf=point_conf,
                                  conf_warp_monotonic=conf_warp_monotonic,
                                  conf_optional_mask=conf_optional_mask, up_hint=up_hint)
    mscale, mflags = _metric_scale_check(metric_scale)
    flags: list[Optional[str]] = list(ground["degradation_flags"]) + list(mflags)
    sv = (scale_version or SCALE_VERSION_DEFAULT) if mscale is not None else None
    ext = ground.get("horizontal_extent_normalized")
    out: dict = {
        "room_diagonal_normalized": None, "room_diagonal_metric": None,
        "room_area_normalized": None, "room_area_m2": None,
        "room_height_normalized": ground.get("room_height_normalized"),
        "wall_extents_normalized": ext,
        "plane_inlier_ratio": ground.get("plane_inlier_ratio"),
        "fit_quality": ground.get("fit_quality"),
        "ground_planarity": ground.get("ground_planarity"),
        "up": ground.get("up"), "up_axis": ground.get("up_axis"),
        "up_source": ground.get("up_source"),
        "ground_plane": ground.get("ground_plane"),
        "n_valid_points": int(ground.get("n_valid_points", 0)),
        "quantile_q": float(p.quantile_q), "voxel_size": float(p.voxel_size),
        "conf_warp_version": str(p.conf_warp_version), "scale_version": sv,
        "metric_scale": mscale,
        "degradation_flags": _flags(*flags),
        "audit": dict(ground.get("audit", {})),
    }
    if ext is None or ground.get("ground_plane") is None:
        out["audit"] = {**out["audit"], "reason": "no_ground_plane"}
        return out
    d1, d2 = float(ext[0]), float(ext[1])
    diag = float(np.hypot(d1, d2))
    area = float(d1 * d2)
    out["room_diagonal_normalized"] = diag
    out["room_area_normalized"] = area
    out["room_diagonal_metric"] = (diag * mscale) if mscale is not None else None
    out["room_area_m2"] = (area * mscale * mscale) if mscale is not None else None
    out["audit"] = {**out["audit"],
                    "wall_extents_normalized": [d1, d2],
                    "area_metric_note": "面积按 metric_scale 的**平方**换算（§9.5）",
                    "diagonal_note": "地面矩形对角线（非三维体对角线）"}
    return out


# ---------------------------------------------------------------------------
# §12.5 消融：稳健低分位 vs 官方最近点口径
# ---------------------------------------------------------------------------

def _normalize_problems(problems: Any) -> list[dict]:
    """消融样本归一化：接受单个/一列 `{"points", "reference_xyz"}` 或 `(points, ref)` 二元组。"""
    if isinstance(problems, Mapping) or (
            isinstance(problems, (tuple, list)) and len(problems) == 2
            and not isinstance(problems[0], (Mapping, tuple, list))):
        problems = [problems]
    rows: list[dict] = []
    for i, item in enumerate(problems):
        if isinstance(item, Mapping):
            if "points" not in item or "reference_xyz" not in item:
                raise ValueError(f"第 {i} 个消融样本缺 points / reference_xyz：{sorted(item)}")
            rows.append({
                "name": str(item.get("name", f"p{i}")),
                "points": item["points"],
                "reference_xyz": item["reference_xyz"],
                "official_nearest": item.get("official_nearest",
                                             item.get("official_nearest_distance")),
                "point_conf": item.get("point_conf"),
                "conf_warp_monotonic": item.get("conf_warp_monotonic"),
                "metric_scale": item.get("metric_scale"),
            })
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            rows.append({"name": f"p{i}", "points": item[0], "reference_xyz": item[1],
                         "official_nearest": item[2] if len(item) > 2 else None,
                         "point_conf": None, "conf_warp_monotonic": None,
                         "metric_scale": None})
        else:
            raise TypeError(f"第 {i} 个消融样本类型不支持：{type(item).__name__}")
    return rows


def ablation_quantiles(
    problems: Any,
    *,
    params: Optional[Any] = None,
    quantiles: Sequence[float] = ABLATION_QUANTILES_Q,
    official_reference: str = "min_q0",
) -> dict:
    """§12.5 消融表：`min(q=0)` vs `q ∈ {0.5%, 1%, 2%, 5%}` 对**官方最近点口径**的偏差。

    用途（论文口径）：论文必须能报告"用稳健低分位近似最近点，与官方 GT 的系统性偏差
    见消融"。本函数给出逐样本偏差表 + 逐 q 汇总表，**不做任何选择**（D8：q 只在 inner
    选、outer 冻结），只把数字摆出来。

    参数
    ----
    problems : 单个或一列样本，每项 `{"points", "reference_xyz", "name"(可选),
        "official_nearest"(可选，官方 GT 最近点距离), "point_conf"(可选),
        "conf_warp_monotonic"(可选)}`；也接受 `(points, reference_xyz[, official_nearest])`。
    quantiles : 消融档（默认 `ABLATION_QUANTILES_Q`，全部为比例而非百分数）。
    official_reference : `"min_q0"`（默认，用同一份点云的 `q=0` = 真最近点作参照）或
        `"provided"`（用样本里的 `official_nearest`；缺该字段的样本按 `min_q0` 并标注）。

    返回
    ----
    `{"quantiles", "official_reference", "rows", "summary", "params", "note"}`：

    - `rows`：逐样本逐 q 的 `distance_normalized / official_nearest_normalized /
      deviation_abs / deviation_rel / n_valid_points / degradation_flags`（降级样本的
      偏差为 `None`，不参与汇总，也不伪造）；
    - `summary`：逐 q 的 `n_measured / n_degraded / mean_deviation_abs /
      median_deviation_abs / mean_deviation_rel / median_deviation_rel /
      max_deviation_abs`（相对偏差只对 `official_nearest > 0` 的样本算）；
    - 对干净点集，同一 q 的 `distance_normalized` 随 q 单调不减（分位数的性质）。

    预处理（体素/conf/N_min）在**每个 q 之间保持一致**：先预处理一次、算一次距离集合，
    再在各 q 上取分位，故表内差异只来自 q，不来自采样口径（消融的可比性要求）。
    """
    p = _params(params)
    qs = [float(x) for x in quantiles]
    for x in qs:
        if not np.isfinite(x) or x < 0.0 or x > 1.0:
            raise ValueError(f"消融档必须是 [0,1] 内的比例（收到 {x!r}）")
    rows_in = _normalize_problems(problems)
    if len(rows_in) < N_MIN_ABLATION_PROBLEMS:
        raise ValueError("消融至少需要 1 个样本")
    if official_reference not in ("min_q0", "provided"):
        raise ValueError(f"official_reference 只能是 min_q0 / provided（收到 {official_reference!r}）")

    rows: list[dict] = []
    n_provided = 0
    measured: dict[float, list[float]] = {q: [] for q in qs}
    rel: dict[float, list[float]] = {q: [] for q in qs}
    degraded_count: dict[float, int] = {q: 0 for q in qs}
    for spec in rows_in:
        # 每个样本**只预处理一次**，各 q 共用同一份距离集合：表内差异只来自 q，
        # 不来自采样口径（消融的可比性要求）。
        prep = preprocess_points(spec["points"], params=p,
                                 point_conf=spec["point_conf"],
                                 conf_warp_monotonic=spec["conf_warp_monotonic"])
        ref = None
        ref_src = ""
        if official_reference == "provided" and spec["official_nearest"] is not None:
            try:
                ref = float(spec["official_nearest"])
            except (TypeError, ValueError):
                ref = None
            if ref is not None and np.isfinite(ref):
                n_provided += 1
                ref_src = "provided"
            else:
                ref = None
        if prep["degraded"]:
            for q in qs:
                degraded_count[q] += 1
                rows.append({
                    "name": spec["name"], "q": q, "distance_normalized": None,
                    "official_nearest_normalized": None if ref is None else float(ref),
                    "official_reference_source": ref_src or "unavailable_degraded",
                    "deviation_abs": None, "deviation_rel": None,
                    "n_valid_points": int(prep["n_used"]),
                    "degradation_flags": list(prep["degradation_flags"]),
                })
            continue
        refv = _as_vector3(spec["reference_xyz"], "reference_xyz")
        dist = np.linalg.norm(prep["points"] - refv[None, :], axis=1)
        if ref is None:
            ref = float(np.min(dist)) if dist.size else None
            ref_src = "min_q0_same_cloud"
        for q in qs:
            v = _weighted_quantile(dist, q, prep["weights"])
            dev_abs = (abs(v - ref) if (ref is not None and np.isfinite(v)) else None)
            dev_rel = (dev_abs / ref if (dev_abs is not None and ref > 0) else None)
            rows.append({
                "name": spec["name"], "q": float(q),
                "distance_normalized": float(v),
                "official_nearest_normalized": None if ref is None else float(ref),
                "official_reference_source": ref_src,
                "deviation_abs": None if dev_abs is None else float(dev_abs),
                "deviation_rel": None if dev_rel is None else float(dev_rel),
                "n_valid_points": int(prep["n_used"]),
                "degradation_flags": list(prep["degradation_flags"]),
            })
            measured[q].append(float(dev_abs) if dev_abs is not None else float("nan"))
            if dev_rel is not None:
                rel[q].append(float(dev_rel))

    summary: list[dict] = []
    for q in qs:
        arr = np.asarray([x for x in measured[q] if np.isfinite(x)], dtype=np.float64)
        rarr = np.asarray(rel[q], dtype=np.float64)
        summary.append({
            "q": float(q),
            "n_problems": len(rows_in),
            "n_measured": int(arr.size),
            "n_degraded": int(degraded_count[q]),
            "mean_deviation_abs": float(arr.mean()) if arr.size else None,
            "median_deviation_abs": float(np.median(arr)) if arr.size else None,
            "max_deviation_abs": float(arr.max()) if arr.size else None,
            "mean_deviation_rel": float(rarr.mean()) if rarr.size else None,
            "median_deviation_rel": float(np.median(rarr)) if rarr.size else None,
            "mean_signed_bias_rel": (float(np.mean([x["deviation_rel"] for x in rows
                                                     if x["q"] == q
                                                     and x["deviation_rel"] is not None]))
                                     if any(x["q"] == q and x["deviation_rel"] is not None
                                            for x in rows) else None),
        })
    if official_reference == "provided" and n_provided == 0:
        src = "min_q0_same_cloud（样本未提供 official_nearest）"
    elif official_reference == "provided":
        src = f"provided（{n_provided}/{len(rows_in)} 个样本提供了官方值，其余回退 min_q0）"
    else:
        src = "min_q0_same_cloud（q=0 = 真最近点，§12.5 官方最近点口径的可复现代理）"
    return {
        "quantiles": qs,
        "official_reference": {"requested": official_reference, "source": src},
        "rows": rows,
        "summary": summary,
        "params": p.snapshot(),
        "note": ("稳健低分位近似最近点与官方 GT 口径的系统性偏差见本表；"
                 "q 的选择只在 inner 做、outer 冻结（D8/§12.1）"),
    }


# ---------------------------------------------------------------------------
# 实例整合（v7 §9.1/§9.2：计数必须按**实例**去重，不能数清单长度/记录数）
# ---------------------------------------------------------------------------

def consolidate_instances(
    point_sets: dict[str, np.ndarray],
    *,
    min_overlap: float = 0.30,
    eps: float = 0.01,
    max_points: int = 3000,
) -> dict:
    """把"同一物理实例被重复检出"的记录合并成簇（并查集）。

    为什么需要它（真实实测）：M5 的 `_dedupe_by_world_centroid` 要求**时序重叠**
    才合并，因此同一个显示器被 SAM2 分裂成支撑帧互不相交的多条 track 时不会被合并。
    实测 scene `7b6477cb95`：问 "How many monitor(s)"，GT=5，而清单里有 12 条
    monitor/tv-monitor 记录，`count_objects` 直接返回 12（MRA=0）。

    判据只用**几何重合**（与帧无关，正好补上时序判据的缺口）：两条记录的点集在
    `eps` 邻域内**双向**重合比例都 ≥ `min_overlap` → 判为同一实例。
    该阈值有实测分离度支撑（同一显示器 0.38–0.69，不同对象 ≤0.14）：
    取 0.30 落在间隙内，两侧各留 >2 倍余量。

    只做**同类别内**的合并由调用方保证（跨类别不合并，避免把桌上的显示器与
    桌面并成一个）。返回 `{"clusters", "n_clusters", "n_input", "merge_edges"}`，
    `clusters` 是簇内键的列表（顺序稳定：按输入键序）。
    """
    keys = list(point_sets.keys())
    n = len(keys)
    if n == 0:
        return {"clusters": [], "n_clusters": 0, "n_input": 0, "merge_edges": []}
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps 必须是正的有限值（收到 {eps!r}）")
    if not (0.0 < float(min_overlap) <= 1.0):
        raise ValueError(f"min_overlap 必须落在 (0,1]（收到 {min_overlap!r}）")

    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)      # 保持确定性（小下标为根）

    # 点集降采样：计数只需要"是否重合"，3000 点已足够稳定，且把 O(n²·query) 压住
    prepped: list[Optional[np.ndarray]] = []
    for k in keys:
        a = np.asarray(point_sets[k], dtype=np.float64).reshape(-1, 3)
        a = a[np.all(np.isfinite(a), axis=1)]
        if a.shape[0] > max_points:
            stride = int(np.ceil(a.shape[0] / max_points))
            a = a[::stride][:max_points]
        prepped.append(a if a.shape[0] else None)

    from scipy.spatial import cKDTree

    merge_edges: list[dict] = []
    for i in range(n):
        ai = prepped[i]
        if ai is None:
            continue
        for j in range(i + 1, n):
            aj = prepped[j]
            if aj is None:
                continue
            ti, tj = cKDTree(ai), cKDTree(aj)
            d_ij, _ = tj.query(ai)
            d_ji, _ = ti.query(aj)
            ovl_ij = float((d_ij < eps).mean())
            ovl_ji = float((d_ji < eps).mean())
            both = min(ovl_ij, ovl_ji)          # 双向：单向高重合可能是包含关系
            if both >= float(min_overlap):
                _union(i, j)
                merge_edges.append({"a": keys[i], "b": keys[j],
                                    "overlap_ab": round(ovl_ij, 4),
                                    "overlap_ba": round(ovl_ji, 4)})

    groups: dict[int, list[str]] = {}
    for idx, k in enumerate(keys):
        groups.setdefault(_find(idx), []).append(k)
    clusters = [groups[r] for r in sorted(groups)]
    return {
        "clusters": clusters,
        "n_clusters": len(clusters),
        "n_input": n,
        "merge_edges": merge_edges,
        "min_overlap": float(min_overlap),
        "eps": float(eps),
        "method": "bidirectional_nn_pointcloud_overlap_v7",
    }
