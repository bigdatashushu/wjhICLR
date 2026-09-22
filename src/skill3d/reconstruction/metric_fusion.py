"""M3 度量尺度融合（v6 §11，首个 PoC = MoGe-2）—— **全模块 [待实验]**。

**证据等级声明（§1.2 / §11 / §23.2）**：本模块实现的是第 11 章那条**待实验**路线。
它只负责"把逐帧零样本度量深度与 VGGT 相对深度熔出一个全局尺度"，
**不证明、也不宣称**该尺度在真实数据上准确、可用或能涨分——C1–C6 六项验收
（§11.4）未在自有数据上跑完之前，`metric_scale` 一律不得当作已验证事实。

算法（§11.1，逐字对齐）：

- 逐帧 ``s_k = median(d_metric / d_VGGT_norm)``，只在
  ``(VGGT conf 经 conf-warp 单调自检可用) & (度量模型高置信) & (非黑边 valid mask)``
  的像素上计算；任一条件算不出 → 该帧 ``s_k = None``（**不是** 0、不是 NaN 冒充值）。
- 跨帧 ``s_global = median_k(s_k)``（鲁棒融合）；``scale_self_consistency = std(s_k)/median(s_k)``。
- 离群帧用 MAD 标注（不静默丢弃）：进 ``outlier_frames``、进 receipt、**排除出最终 median**。

**硬边界（D1，不得越界）**：无 GT 位姿、无标定池、无 LiDAR、无 BA、无场景级拟合。
本模块**没有任何**接收以上输入的参数或代码路径；任何"用先验补一个尺度"的写法都被禁止。

**失败纪律（§11.4）**：六项验收任一不过 → 关闭尺度融合支路，系统退回纯相对几何；
**绝不回退多锚点、绝不回退校准池**（§20 已废止）。因此 ``status="failed"`` 必须
``metric_scale=None``——本模块不存在任何"从先验造尺度"的分支。

**确定性**：无未播种随机、无墙钟依赖（receipt 内也不含时间戳，因此字节可复现；
审计时间由外层 trace（§19）负责）。

调用契约（本模块**不隐式**做，缺哪条就 fail-closed）：

1. 两侧深度必须**同网格同帧序**（``d_metric`` 与 ``d_VGGT_norm`` 形状一致）。
   本模块不做隐式 resize/插值——插值会在物体边缘造出不存在的比值；
2. ``depth_metric`` 必须是**米制**；VGGT 侧传**归一化**深度（全帧中位距归一化那个量纲）。
3. 传给本模块的 ``valid_masks`` 已是"度量模型高置信 ∧ 非黑边"的合取结果
   （生产者见 `_MoGe2MetricModel.infer`）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, Union

import numpy as np

# ---------------------------------------------------------------------------
# 版本与模型标识（进 trace / artifact，§5.2 / §19.2）
# ---------------------------------------------------------------------------

METRIC_FUSION_VERSION: str = "metric-fusion-v6"
METRIC_MODEL_MOGE2 = "moge2"
METRIC_MODEL_METRIC3D_V2 = "metric3d_v2"
METRIC_MODEL_NONE = "none"

# ---------------------------------------------------------------------------
# 阈值常量（**全部 TODO_CALIBRATE**，起始参考值见 §11.1 / §13.1；不得当已验证值）
# ---------------------------------------------------------------------------

# 一帧最少有效像素数（§11 未给参考值：取 518×392 网格的约 0.5% 作起始参考）
MIN_VALID_PIXELS: int = 1000          # TODO_CALIBRATE: 单帧参与 s_k 的最小像素数
# MAD 离群倍数（鲁棒 σ = 1.4826×MAD 的倍数；3.0 = 常规 3σ 口径）
OUTLIER_MAD_K: float = 3.0            # TODO_CALIBRATE: MAD 离群倍数
# τ_scale_disp：跨帧 s_k 的 std/median 上限（§13.1 子条件 4 起始参考 0.15）
MAX_SCALE_DISPERSION: float = 0.15    # TODO_CALIBRATE: τ_scale_disp 起始参考
# τ_frames：有效帧占比下界（§13.1 子条件 3 起始参考 0.75）
MIN_VALID_FRAME_RATIO: float = 0.75   # TODO_CALIBRATE: τ_frames 起始参考
# 成功融合所需的最小有效帧数（防"2 帧也算融合成功"）
MIN_VALID_FRAMES_FOR_SUCCESS: int = 3  # TODO_CALIBRATE: 成功所需最小有效帧数
# MAD 离群检测的最小样本数（更少时 MAD 退化，不做离群判定并写明）
MIN_FRAMES_FOR_MAD: int = 5           # TODO_CALIBRATE: MAD 离群检测最小帧数
# 正深度数值护栏（防 0/负/极小值放大比值噪声）
MIN_POSITIVE_DEPTH: float = 1e-6      # TODO_CALIBRATE: 正深度下界（数值护栏）

# MAD → 鲁棒 σ 的一致性常数（数学常数，不是可标定量）
MAD_TO_SIGMA: float = 1.4826

# conf 软权重（§10.3：conf **只作软权重，不作硬门**；C>2 不是固定硬阈值）：
# 这里用**逐帧相对分位**（丢弃每帧 conf 最低的 20% 像素），不是绝对 conf 门；
# 备选口径是 `reconstruction_gate.m4_main_gate.conf_optional_mask`（绝对 C 下界，
# 该模块明文"不参与任何门"）——改用哪个属于 C1–C6 标定内容，**不得在标定前定死**。
CONF_SOFT_KEEP_QUANTILE: float = 0.2  # TODO_CALIBRATE: per-frame conf 相对分位下界

# ---- MoGe-2（§11.2，MIT，arXiv:2507.02546）----
MOGE2_CHECKPOINT: str = "Ruicheng/moge-2-vitl"  # HF repo id
MOGE2_USE_FP16: bool = True           # §11.2：FP16 ViT-L 约 60ms/帧（文献值，非本系统实测）
MOGE2_BLACK_LUMA_MAX: int = 12        # TODO_CALIBRATE: 非黑边像素的 luma 下界
# MoGe 的 `infer(fov_x=...)` 单位按官方文档为**度**。**已核验**（2026-09-21：
# 本机装上 moge 后读 `moge/model/v2.py::MoGeModel.infer` 源码，fov_x 走
# `torch.deg2rad(fov_x/2)`；实测传入 66.7 时输出 intrinsics 反推 fov 亦为 66.7）；
# 若单位不一致**只改这一处**（全模块唯一换算点）。
MOGE2_FOV_UNIT_DEGREES: bool = True

# ---- C1–C6 验收用阈值（§11.4；只服务于 acceptance_report，不参与融合计算）----
C4_MIN_QUESTIONS: int = 3             # TODO_CALIBRATE: C4 paired 比较的最小题数
C5_MAX_LATENCY_MS_PER_FRAME: float = 200.0  # TODO_CALIBRATE: C5 单帧延迟上限（ms）
C5_MAX_PEAK_GPU_GIB: float = 24.0     # TODO_CALIBRATE: C5 峰值显存上限（24 GiB 卡预算）
PAPER_MIN_SEEDS: int = 5              # TODO_CALIBRATE: §18.6 paper 档 seed 下界（≥5）


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class PerFrameScale:
    """单帧尺度估计（§11.1 的 ``s_k``）——PoC receipt 的一条记录。

    ``s_k=None`` 表示**这一帧算不出**（深度缺失/形状不一致/有效像素不足/非有限），
    与 ``s_k=0`` 有本质区别：前者是"无证据"，任何下游都不得把它当数值用。
    """

    frame_idx: int
    s_k: Optional[float]                 # median(d_metric / d_VGGT_norm)；算不出为 None
    n_valid_pixels: int
    outlier: bool = False


@dataclass
class ScaleFusionResult:
    """跨帧融合结果（§5.2 的尺度字段来源）。

    与 `ReconstructionArtifact` 的字段映射（§5.2）：

    - ``status`` → ``scale_fusion_status``；
    - ``metric_scale`` → ``metric_scale``（**只有 status="success" 才非 None**）；
    - ``scale_self_consistency`` → ``scale_self_consistency``（32 帧 s_k 的 std/median）；
    - ``version`` → ``metric_fusion_version``；``model`` → ``metric_model``；
    - ``per_frame`` / ``mad`` / ``outlier_frames`` → 经 `write_per_frame_receipt` 落盘后
      由调用方写入 ``per_frame_scale_ref``（本模块不写 artifact，不猜 ref 路径）。

    口径说明（实现选择，见模块文档"开放项"）：``scale_self_consistency`` 在
    **全部有效帧**的 s_k 上算（含被标注为离群的帧）——它是自洽门（§13.1 子条件 4）
    与退化识别（C2）的信号，"离群被平滑掉"会让自洽门失去意义；
    ``metric_scale`` 才只在**未被标注离群**的帧上取 median。
    """

    status: str                          # "success" | "failed" | "not_run"
    metric_scale: Optional[float]        # s_global = median_k(s_k)
    scale_self_consistency: Optional[float]  # std/median of s_k
    n_frames_valid: int
    n_frames_total: int
    valid_frame_ratio: float
    outlier_frames: list[int]
    per_frame: list[PerFrameScale]
    mad: Optional[float]                 # median absolute deviation of s_k
    version: str = METRIC_FUSION_VERSION
    model: str = METRIC_MODEL_NONE
    note: str = ""


@dataclass
class MetricDepthOutput:
    """零样本度量深度模型的一次前向输出（单位：米）。

    ``valid_mask`` 是"高置信 ∧ 非黑边"的合取（§11.1 第三条像素条件）；
    ``fov_x_deg`` / ``intrinsics`` 供相机对齐与审计（§11.2）。
    """

    depth_metric: np.ndarray             # (H,W) 米制深度
    valid_mask: np.ndarray               # (H,W) bool：高置信 且 非黑边
    point_map: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    fov_x_deg: Optional[float] = None    # 供相机对齐与审计


class MetricDepthModel(Protocol):
    """零样本单目度量深度模型的可替换接口（首个 PoC = MoGe-2）。"""

    name: str

    def infer(self, rgb: np.ndarray,
              intrinsics: Optional[np.ndarray]) -> MetricDepthOutput:
        ...


# ---------------------------------------------------------------------------
# 逐帧 s_k
# ---------------------------------------------------------------------------


def _empty_frame(idx: int, n_pixels: int = 0) -> PerFrameScale:
    """构造"该帧无证据"的记录（fail-closed：``s_k=None``，绝不填 0/NaN）。"""
    return PerFrameScale(frame_idx=int(idx), s_k=None, n_valid_pixels=int(n_pixels))


def per_frame_scale(
    frame_idx: int,
    depth_metric: Optional[np.ndarray],
    depth_vggt: Optional[np.ndarray],
    valid_mask: Optional[np.ndarray] = None,
    vggt_conf: Optional[np.ndarray] = None,
    *,
    use_conf: bool = False,
    min_valid_pixels: int = MIN_VALID_PIXELS,
) -> PerFrameScale:
    """单帧 ``s_k = median(d_metric / d_VGGT_norm)``（§11.1 逐帧估计）。

    像素集合 = ``有限 ∧ 正`` ∧ ``valid_mask``（高置信∧非黑边）∧（``use_conf`` 时）
    ``conf 高于本帧相对分位下界``。任何形状/数值异常 → ``s_k=None``，**不猜、不插值**。

    ``use_conf`` 只允许在 conf-warp 单调自检通过后置 True（§10.3）；见 `fuse_metric_scale`。
    """
    if depth_metric is None or depth_vggt is None:
        return _empty_frame(frame_idx)
    dm = np.asarray(depth_metric, dtype=np.float64)
    dv = np.asarray(depth_vggt, dtype=np.float64)
    if dm.shape != dv.shape or dm.ndim != 2:
        return _empty_frame(frame_idx)          # 非同网格 → fail-closed
    ok = (np.isfinite(dm) & np.isfinite(dv)
          & (dm > MIN_POSITIVE_DEPTH) & (dv > MIN_POSITIVE_DEPTH))
    if valid_mask is not None:
        vm = np.asarray(valid_mask)
        if vm.shape != dm.shape:
            return _empty_frame(frame_idx)      # 掩码不同网格 → fail-closed
        ok &= vm.astype(bool)
    if use_conf and vggt_conf is not None:
        cf = np.asarray(vggt_conf, dtype=np.float64)
        if cf.shape == dm.shape:
            ok &= np.isfinite(cf)
            kept = cf[ok]
            if kept.size:
                # 逐帧**相对**分位下界（软权重口径，不是绝对 C>2 硬门，§10.3）
                floor = float(np.quantile(kept, CONF_SOFT_KEEP_QUANTILE))
                ok &= (cf >= floor)
    n_pixels = int(ok.sum())
    floor_px = max(int(min_valid_pixels), 1)
    if n_pixels < floor_px:
        return _empty_frame(frame_idx, n_pixels)
    ratio = dm[ok] / dv[ok]
    ratio = ratio[np.isfinite(ratio) & (ratio > MIN_POSITIVE_DEPTH)]
    if ratio.size < floor_px:
        return _empty_frame(frame_idx, int(ratio.size))
    s_k = float(np.median(ratio))
    if not np.isfinite(s_k) or s_k <= 0.0:
        return _empty_frame(frame_idx, int(ratio.size))
    return PerFrameScale(frame_idx=int(frame_idx), s_k=s_k, n_valid_pixels=n_pixels)


def _flag_outliers(s_values: Sequence[float], mad: Optional[float],
                   k: float) -> list[bool]:
    """MAD 离群标注（§11.1）：``|s_k − median| > k × 1.4826 × MAD`` 即标注。

    标注**不等于丢弃**——离群帧照常进 receipt、进 `outlier_frames`，只是不进最终
    median（"flagged, not silently dropped"）。MAD=0（全体一致）或样本不足
    （< `MIN_FRAMES_FOR_MAD`）时不做判定，返回全 False。
    """
    n = len(s_values)
    if mad is None or n < int(MIN_FRAMES_FOR_MAD):
        return [False] * n
    sigma = MAD_TO_SIGMA * float(mad)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return [False] * n
    med = float(np.median(np.asarray(s_values, dtype=np.float64)))
    return [bool(abs(float(v) - med) > float(k) * sigma) for v in s_values]


# ---------------------------------------------------------------------------
# 跨帧融合
# ---------------------------------------------------------------------------


def fuse_metric_scale(
    depth_metric_list: Sequence[Optional[np.ndarray]],
    depth_vggt_list: Sequence[np.ndarray],
    valid_masks: Optional[Sequence[Optional[np.ndarray]]],
    vggt_conf_list: Optional[Sequence[Optional[np.ndarray]]] = None,
    *,
    model: str = METRIC_MODEL_NONE,
    conf_warp_monotonic: Optional[bool] = None,
    min_valid_pixels: int = MIN_VALID_PIXELS,
    outlier_mad_k: float = OUTLIER_MAD_K,
    max_scale_dispersion: float = MAX_SCALE_DISPERSION,
    min_valid_frame_ratio: float = MIN_VALID_FRAME_RATIO,
) -> ScaleFusionResult:
    """跨帧度量尺度融合（§11.1）：逐帧 ``s_k`` → ``median_k`` → 自洽度（fail-closed）。

    参数
    ----
    depth_metric_list
        逐帧**米制**深度 ``(H,W)``（度量模型输出；``None`` = 该帧无度量深度）。
    depth_vggt_list
        逐帧 VGGT **归一化**深度 ``(H,W)``，必须与前者同网格同帧序。
    valid_masks
        逐帧"高置信 ∧ 非黑边"掩码（`MetricDepthOutput.valid_mask`）；``None`` 表示该帧
        未提供掩码 → 该帧只按"有限正深度"取像素，并在 ``note`` 里写明（不静默放宽）。
    vggt_conf_list
        VGGT ``depth_conf``；**只在** ``conf_warp_monotonic is True`` 时参与（§10.3）。
    model
        产出 ``depth_metric`` 的模型标识（`METRIC_MODEL_MOGE2` / `..._METRIC3D_V2` /
        `..._NONE`），原样写进结果与 receipt，供 trace 审计。
    conf_warp_monotonic
        来自 `m4_main_gate.conf_warp_monotonic`（§10.3 自检）。``True`` 才允许把 conf
        当软权重；``False``/``None`` 一律"不把 conf 当门"（保守口径，宁少用不误用）。
    min_valid_pixels / outlier_mad_k / max_scale_dispersion / min_valid_frame_ratio
        阈值，默认取模块级 `TODO_CALIBRATE` 常量。

    返回
    ----
    `ScaleFusionResult`：``status`` ∈ {success, failed}；``metric_scale`` **只有**
    success 才非 None。失败时仍返回完整 ``per_frame`` / ``mad`` / ``outlier_frames``
    供诊断与 receipt 落盘（§23.1 Phase 2 DoD）。本函数不发散、不抛数值异常：
    输入坏 → 该帧 ``s_k=None``；证据不足 → ``status="failed"`` + ``metric_scale=None``。
    """
    notes: list[str] = []
    n_total = len(depth_vggt_list)
    n_metric = len(depth_metric_list)
    n_mask = n_total if valid_masks is None else len(valid_masks)

    # ---- 契约前置检查：序列长度不一致 → 直接失败（不猜对齐方式）----
    if len({n_total, n_metric, n_mask}) != 1:
        return ScaleFusionResult(
            status="failed", metric_scale=None, scale_self_consistency=None,
            n_frames_valid=0, n_frames_total=int(n_total), valid_frame_ratio=0.0,
            outlier_frames=[], per_frame=[], mad=None, model=str(model),
            note=(f"输入序列长度不一致（vggt={n_total} / metric={n_metric} / "
                  f"mask={n_mask}）→ fail-closed，不猜帧对齐"))
    if n_total == 0:
        return ScaleFusionResult(
            status="failed", metric_scale=None, scale_self_consistency=None,
            n_frames_valid=0, n_frames_total=0, valid_frame_ratio=0.0,
            outlier_frames=[], per_frame=[], mad=None, model=str(model),
            note="输入帧数为 0 → fail-closed（无证据不出尺度）")

    # ---- conf 使用口径（§10.3）----
    use_conf = conf_warp_monotonic is True
    if vggt_conf_list is None:
        notes.append("未提供 VGGT conf → conf 不参与")
    elif conf_warp_monotonic is True:
        notes.append("conf-warp 单调自检通过 → conf 作逐帧相对分位软下界"
                     f"（q={CONF_SOFT_KEEP_QUANTILE}，非绝对 conf 硬门，§10.3）")
    elif conf_warp_monotonic is False:
        notes.append("conf-warp 单调自检未通过 → VGGT conf 不作门、不参与（§10.3 降权）")
    else:
        notes.append("conf_warp_monotonic 未提供（M4 主门未跑）→ 保守口径：conf 不参与")

    # ---- 逐帧 s_k ----
    per_frame: list[PerFrameScale] = []
    n_mask_missing = 0
    for k in range(n_total):
        mask_k = None if valid_masks is None else valid_masks[k]
        if mask_k is None:
            n_mask_missing += 1
        conf_k = None if vggt_conf_list is None else vggt_conf_list[k]
        per_frame.append(per_frame_scale(
            k, depth_metric_list[k], depth_vggt_list[k], mask_k, conf_k,
            use_conf=use_conf, min_valid_pixels=int(min_valid_pixels)))
    if n_mask_missing:
        notes.append(f"{n_mask_missing}/{n_total} 帧缺 valid_mask → 该帧只按有限正深度取"
                     "像素（未提供高置信/非黑边保证，收据里如实记录）")

    valid = [p for p in per_frame if p.s_k is not None]
    s_values = [float(p.s_k) for p in valid]  # type: ignore[arg-type]
    n_valid = len(s_values)
    ratio = (n_valid / n_total) if n_total else 0.0

    # ---- 跨帧统计（全部有效帧；诊断量在失败时也照算）----
    mad: Optional[float] = None
    dispersion: Optional[float] = None
    flags = [False] * n_valid
    if n_valid:
        s_arr = np.asarray(s_values, dtype=np.float64)
        med_all = float(np.median(s_arr))
        if np.isfinite(med_all) and med_all > 0.0:
            dispersion = float(np.std(s_arr) / med_all)
        if np.all(np.isfinite(s_arr)):
            mad = float(np.median(np.abs(s_arr - med_all)))
        flags = _flag_outliers(s_values, mad, float(outlier_mad_k))
        if mad is not None and n_valid < int(MIN_FRAMES_FOR_MAD):
            notes.append(f"有效帧 {n_valid} < {int(MIN_FRAMES_FOR_MAD)} → 跳过 MAD 离群检测"
                         "（样本不足时 MAD 退化，不硬判）")
        for p, flag in zip(valid, flags):
            p.outlier = bool(flag)
    outlier_frames = [int(p.frame_idx) for p in valid if p.outlier]
    retained = [float(p.s_k) for p in valid if not p.outlier]  # type: ignore[arg-type]

    # ---- 失败纪律：任一不达标 → failed + metric_scale=None（绝不造尺度）----
    failed_reasons: list[str] = []
    if not s_values:
        failed_reasons.append("无任何帧算出可用 s_k")
    else:
        if n_valid < int(MIN_VALID_FRAMES_FOR_SUCCESS):
            failed_reasons.append(
                f"有效帧数 {n_valid} < 成功下界 {int(MIN_VALID_FRAMES_FOR_SUCCESS)}")
        if ratio < float(min_valid_frame_ratio):
            failed_reasons.append(
                f"有效帧占比 {ratio:.4f} < τ_frames {float(min_valid_frame_ratio)}")
        if dispersion is None or not np.isfinite(dispersion):
            failed_reasons.append("尺度自洽度不可计算（median 非正/非有限）")
        elif dispersion > float(max_scale_dispersion):
            failed_reasons.append(
                f"尺度离散度 {dispersion:.4f} > τ_scale_disp {float(max_scale_dispersion)}"
                "（自洽门不过）")
        if not retained:
            failed_reasons.append("离群剔除后无可融合帧（不做静默回退）")

    metric_scale: Optional[float] = None
    if not failed_reasons:
        s_global = float(np.median(np.asarray(retained, dtype=np.float64)))
        if np.isfinite(s_global) and s_global > 0.0:
            metric_scale = s_global
        else:
            failed_reasons.append(f"融合尺度非有限/非正（{s_global!r}）")
    status = "failed" if failed_reasons else "success"
    if outlier_frames:
        notes.append(f"离群帧 {outlier_frames} 已标注并排除出最终 median（未静默丢弃）")
    if failed_reasons:
        notes.append("失败纪律（§11.4）：关闭尺度融合支路、退回纯相对几何；"
                     "不回退多锚点、不回退校准池（§20）")

    return ScaleFusionResult(
        status=status,
        metric_scale=metric_scale,
        scale_self_consistency=dispersion,
        n_frames_valid=n_valid,
        n_frames_total=int(n_total),
        valid_frame_ratio=float(ratio),
        outlier_frames=outlier_frames,
        per_frame=per_frame,
        mad=mad,
        model=str(model),
        note="；".join([*failed_reasons, *notes]),
    )


# ---------------------------------------------------------------------------
# PoC receipt 落盘（§11.4 / §23.1 Phase 2 DoD）
# ---------------------------------------------------------------------------


def threshold_snapshot() -> dict[str, float]:
    """当前模块级阈值快照（全是 `TODO_CALIBRATE` 起始参考）→ 进 receipt/trace。

    收据必须自带"这份 s_k 是按哪套阈值判的"，否则事后无法审计某次 success/failed。
    """
    return {
        "min_valid_pixels": float(MIN_VALID_PIXELS),
        "outlier_mad_k": float(OUTLIER_MAD_K),
        "max_scale_dispersion": float(MAX_SCALE_DISPERSION),
        "min_valid_frame_ratio": float(MIN_VALID_FRAME_RATIO),
        "min_valid_frames_for_success": float(MIN_VALID_FRAMES_FOR_SUCCESS),
        "min_frames_for_mad": float(MIN_FRAMES_FOR_MAD),
        "min_positive_depth": float(MIN_POSITIVE_DEPTH),
        "conf_soft_keep_quantile": float(CONF_SOFT_KEEP_QUANTILE),
    }


def _json_safe(v: Any) -> Any:
    """非有限浮点 → None（收据里不允许出现 NaN/Inf 冒充数值）。"""
    if v is None:
        return None
    if isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def per_frame_receipt(result: ScaleFusionResult,
                      thresholds: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """把融合结果组织成可落盘/可 diff 的收据（纯函数，无墙钟、无随机）。"""
    thr = threshold_snapshot() if thresholds is None else dict(thresholds)
    return {
        "metric_fusion_version": str(result.version),
        "metric_model": str(result.model),
        "status": str(result.status),
        "n_frames_total": int(result.n_frames_total),
        "n_frames_valid": int(result.n_frames_valid),
        "valid_frame_ratio": _json_safe(result.valid_frame_ratio),
        "metric_scale": _json_safe(result.metric_scale),
        "scale_self_consistency": _json_safe(result.scale_self_consistency),
        "mad": _json_safe(result.mad),
        "outlier_frames": [int(i) for i in result.outlier_frames],
        "thresholds": {str(k): _json_safe(v) for k, v in sorted(thr.items())},
        "per_frame": [
            {
                "frame_idx": int(p.frame_idx),
                "s_k": _json_safe(p.s_k),
                "n_valid_pixels": int(p.n_valid_pixels),
                "outlier": bool(p.outlier),
            }
            for p in result.per_frame
        ],
        "note": str(result.note),
    }


def write_per_frame_receipt(result: ScaleFusionResult, path: Union[str, Path],
                            *,
                            thresholds: Optional[Mapping[str, Any]] = None) -> Path:
    """原子写 PoC receipt（32 个 s_k、median、MAD、离群帧列表、阈值快照）。

    §11.4 / §23.1 Phase 2 DoD 的"PoC receipt 落盘"就是这一个文件：
    **成功也写、失败也写**（失败收据是止损纪律的凭证，不许只留成功那一份）。
    写入用临时文件 + `replace`，避免半截 JSON；内容不含时间戳，字节可复现。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = per_frame_receipt(result, thresholds)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    return p


def read_per_frame_receipt(path: Union[str, Path]) -> dict[str, Any]:
    """读回收据（只读；审计/回归对比用）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# C1–C6 验收（§11.4）—— **只做证据核查，不产生证据、不写死 True**
# ---------------------------------------------------------------------------


def _num(v: Any) -> Optional[float]:
    """任意输入 → 有限 float，否则 None（收据/证据缺失一律 None）。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _num_or_bool(v: Any) -> Optional[bool]:
    return None if v is None else bool(v)


def acceptance_report(
    result: Optional[ScaleFusionResult] = None,
    *,
    c1_dispersion: Optional[float] = None,
    c1_max_dispersion: Optional[float] = None,
    c2_degradation: Optional[Mapping[str, Any]] = None,
    c3_m4_main_gate: Optional[bool] = None,
    c4_paired: Optional[Mapping[str, Any]] = None,
    c5_resource: Optional[Mapping[str, Any]] = None,
    c6_cross_model: Optional[Mapping[str, Any]] = None,
    paper_evidence: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """按 §11.4 核查 C1–C6 六项验收，返回 ``{item: bool|None}`` + 四级 Readiness 标志。

    **本函数不生产证据，只判传入的证据**：没给的项一律 ``None``（= 未跑），
    **绝不写死 True**。四级 Readiness（§23.2）只能由证据逐级推动，
    且 ``implemented`` 只表示"代码已真跑过一次并产出结构化结果"，
    **不含任何效果声明**。

    传参 → 判据（全部要求证据齐备，缺一项即 ``None``）：

    - ``c1_dispersion``：跨帧 s_k 离散度 ``std/median``，``≤ c1_max_dispersion``
      （默认 τ_scale_disp）才 True。未给时**仅当** result 是真跑（``status="success"``
      且 ``model != "none"``）才回落到 ``result.scale_self_consistency``。
    - ``c2_degradation``：``{"baseline_dispersion": x, "degraded_dispersion": y|[y1,y2,...]}``
      ——注入退化（抽稀/跨场景帧/运动模糊）后离散度应**上升**：单个 y 需 ``y > x``；
      给 mapping/序列时**每一项**都要 ``> x``。
    - ``c3_m4_main_gate``：直接传 M4 主门布尔（§10.1 warp 内点率 ∧ 分组点云重叠率）。
    - ``c4_paired``：``{"n_questions": n, "not_worse_than_unscaled": b1,
      "not_worse_than_same_frame_direct": b2}``——米制三题 MRA 的 paired 比较（不劣）
      且题数 ``≥ C4_MIN_QUESTIONS`` 才 True。
    - ``c5_resource``：``{"latency_ms_per_frame": t, "peak_gpu_gib": g}``——
      ``t ≤ C5_MAX_LATENCY_MS_PER_FRAME`` 且 ``g ≤ C5_MAX_PEAK_GPU_GIB``。
    - ``c6_cross_model``：``{"correct_k_dispersion": d_ok, "wrong_k_dispersion": d_bad,
      "cross_model_compared": b}``——错 K 应**发散**（``d_bad > d_ok``）且已做
      MoGe-2 / Metric3D v2 同场景对照。
    - ``paper_evidence``（§18.6）：``{"n_seeds", "split_isolated", "statistical_gate",
      "non_mock"}`` 且 ``n_seeds ≥ PAPER_MIN_SEEDS``。

    ``branch_closed`` 是失败纪律的显式输出（§11.4）：有任一 False → True（关闭尺度
    融合支路、退回纯相对几何）；有任一项 ``None`` → ``None``（"还没法判"，不是通过）。
    """
    res = result
    tau_disp = (MAX_SCALE_DISPERSION if c1_max_dispersion is None
                else float(c1_max_dispersion))

    # ---- C1 跨帧 s 稳定 ----
    disp = _num(c1_dispersion)
    if disp is None and res is not None and res.status == "success" \
            and str(res.model) != METRIC_MODEL_NONE:
        disp = _num(res.scale_self_consistency)
    c1 = None if disp is None else bool(disp <= tau_disp)

    # ---- C2 人工退化敏感性（离散度应上升）----
    c2: Optional[bool] = None
    if c2_degradation is not None:
        base = _num(c2_degradation.get("baseline_dispersion"))
        degraded = c2_degradation.get("degraded_dispersion")
        if base is not None:
            if isinstance(degraded, Mapping):
                vals = [_num(v) for v in degraded.values()]
            elif isinstance(degraded, Sequence) and not isinstance(degraded, (str, bytes)):
                vals = [_num(v) for v in degraded]
            else:
                vals = [_num(degraded)]
            if vals and all(v is not None for v in vals):
                c2 = bool(all(float(v) > base for v in vals))  # type: ignore[arg-type]

    # ---- C3 过 M4 几何自洽门（warp + 重叠率主门）----
    c3 = _num_or_bool(c3_m4_main_gate)

    # ---- C4 米制三题 MRA 的 paired 比较（不劣）----
    c4: Optional[bool] = None
    if c4_paired is not None:
        n_q = _num(c4_paired.get("n_questions"))
        b1 = _num_or_bool(c4_paired.get("not_worse_than_unscaled"))
        b2 = _num_or_bool(c4_paired.get("not_worse_than_same_frame_direct"))
        if n_q is not None and b1 is not None and b2 is not None:
            c4 = bool(n_q >= C4_MIN_QUESTIONS and b1 and b2)

    # ---- C5 资源/延迟可接受 ----
    c5: Optional[bool] = None
    if c5_resource is not None:
        lat = _num(c5_resource.get("latency_ms_per_frame"))
        peak = _num(c5_resource.get("peak_gpu_gib"))
        if lat is not None and peak is not None:
            c5 = bool(lat <= C5_MAX_LATENCY_MS_PER_FRAME
                      and peak <= C5_MAX_PEAK_GPU_GIB)

    # ---- C6 错 K 应发散 + MoGe-2/Metric3D v2 同场景对照 ----
    c6: Optional[bool] = None
    if c6_cross_model is not None:
        d_ok = _num(c6_cross_model.get("correct_k_dispersion"))
        d_bad = _num(c6_cross_model.get("wrong_k_dispersion"))
        compared = _num_or_bool(c6_cross_model.get("cross_model_compared"))
        if d_ok is not None and d_bad is not None and compared is not None:
            c6 = bool(d_bad > d_ok and compared)

    items: dict[str, Optional[bool]] = {"C1": c1, "C2": c2, "C3": c3,
                                        "C4": c4, "C5": c5, "C6": c6}
    n_passed = sum(1 for v in items.values() if v is True)
    n_failed = sum(1 for v in items.values() if v is False)
    n_not_run = sum(1 for v in items.values() if v is None)
    all_six = n_passed == len(items)

    # ---- 四级 Readiness（§23.2：只能由证据逐级推动，缺证据即 False）----
    ran = bool(res is not None and res.status in ("success", "failed")
               and len(res.per_frame) > 0)
    implemented = ran                                   # 代码真跑过并产出结构化结果
    connected = bool(implemented and c3 is True)        # 已接进主线（过 M4 自洽门）
    real_model = bool(res is not None and str(res.model) != METRIC_MODEL_NONE)
    real_poc_verified = bool(all_six and real_model)    # 六项全过 + 真度量模型
    paper_eligible = False
    if real_poc_verified and paper_evidence is not None:
        seeds = _num(paper_evidence.get("n_seeds"))
        iso = _num_or_bool(paper_evidence.get("split_isolated"))
        gate = _num_or_bool(paper_evidence.get("statistical_gate"))
        non_mock = _num_or_bool(paper_evidence.get("non_mock"))
        paper_eligible = bool(seeds is not None and seeds >= PAPER_MIN_SEEDS
                              and iso and gate and non_mock)

    if n_failed:
        branch_closed: Optional[bool] = True
        note = ("§11.4 失败纪律：有验收项不达标 → 关闭尺度融合支路，系统退回纯相对"
                "几何；不回退多锚点、不回退校准池（§20）。metric_scale 保持 None。")
    elif n_not_run:
        branch_closed = None
        note = (f"尚有 {n_not_run} 项验收未跑（None）→ 既不算通过也不算关闭；"
                "PoC 未跑完前 metric_scale 不得当作已验证事实（§11 [待实验]）。")
    else:
        branch_closed = False
        note = ("六项验收按**传入证据**全部为 True；本函数不生产证据，"
                "真实性与非 mock 属性由 §18.6 的隔离/seed/统计门单独把关。")

    return {
        **items,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_not_run": n_not_run,
        "implemented": implemented,
        "connected": connected,
        "real_poc_verified": real_poc_verified,
        "paper_eligible": paper_eligible,
        "branch_closed": branch_closed,
        "metric_fusion_version": METRIC_FUSION_VERSION,
        "metric_model": (str(res.model) if res is not None else METRIC_MODEL_NONE),
        "scale_fusion_status": (str(res.status) if res is not None else "not_run"),
        "metric_scale": (None if res is None else _json_safe(res.metric_scale)),
        "fusion_note": ("" if res is None else str(res.note)),
        "note": note,
    }


# ---------------------------------------------------------------------------
# MoGe-2 包装（§11.2；懒加载，包缺失一律 RuntimeError，**绝不返回假深度**）
# ---------------------------------------------------------------------------


def fov_x_deg_from_intrinsics(intrinsics: Optional[np.ndarray],
                              image_width: Optional[int] = None) -> Optional[float]:
    """由内参 K 换算水平视场角（度），供 §11.2 的相机对齐使用。

    **优先由主点推**：主点位于图像中心 ⇒ 同一网格内有 ``W = 2·cx``，故
    ``fov_x = 2·atan(cx / fx)`` —— 只用 K 内部量，**天然与"K 在哪套分辨率网格"
    无关**，不需要调用方知道图像宽。

    2026-09-21 实测修正（真实 MoGe-2 + 真实 VGGT K）：原实现写成
    ``fx_norm = fx / W`` 再 ``2·atan(1/(2·fx_norm))``，展开就是 ``2·atan(W/(2·fx))``
    —— **依赖 W**。而 VGGT 的 K 是**深度网格**（518 宽，cx=259=518/2）的，
    调用方按**原图**宽（640）传入 ⇒ 得 78.2°，正确值是 66.7°（差 11.5°）。
    用 MoGe-2 自估 fov 作独立参照实测：自估 68.7°、主点推导 66.7°（差 2°）、
    原实现 78.2°（差 9.5°）→ 原实现错。深度中位数随之从 1.72 m 变成 1.98 m
    （尺度差 15%，直接影响米制三题）。

    `image_width` 退化为**兜底**：仅当主点不可用（非有限 / ≤0）时才使用，
    此时调用方**必须**传入与 K 同一网格的宽度。K 非法 → ``None``
    （**不猜视场角**：宁可让调用方 fail-closed，也不给一个错 K 换算出的 fov）。
    """
    if intrinsics is None:
        return None
    k = np.asarray(intrinsics, dtype=np.float64)
    while k.ndim > 2 and k.shape[0] == 1:   # 去 batch 维 (1,3,3) → (3,3)
        k = k[0]
    if k.ndim == 3:                          # 多帧 K：只认第一帧（对齐按帧调用）
        k = k[0]
    if k.shape != (3, 3):
        return None
    fx, cx = float(k[0, 0]), float(k[0, 2])
    if not np.isfinite(fx) or fx <= 0.0:
        return None
    if np.isfinite(cx) and cx > 0.0:
        return float(np.degrees(2.0 * np.arctan(cx / fx)))
    if image_width is None or int(image_width) <= 0:
        return None
    return float(np.degrees(2.0 * np.arctan(float(image_width) / (2.0 * fx))))


def _to_moge_tensor(rgb: np.ndarray, torch_mod: Any) -> Any:
    """RGB 帧 → MoGe 需要的 ``(1,3,H,W)`` float32 [0,1] 张量（口径显式，不猜）。"""
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"rgb 必须是 (H,W,3)，收到 shape={a.shape}")
    if a.dtype == np.uint8:
        f = a.astype(np.float32) / 255.0
    else:
        f = a.astype(np.float32)
        if float(np.max(f)) > 1.0 + 1e-3:  # 浮点输入必须是 [0,1]，不替调用方猜 0-255
            raise ValueError("浮点 rgb 必须已归一到 [0,1]；疑似 0–255 输入 → fail-closed")
    t = torch_mod.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).unsqueeze(0)
    return t.contiguous()


def _black_border_mask(rgb: np.ndarray) -> np.ndarray:
    """非黑边掩码（§11.1 第三条像素条件）：luma 低于 `MOGE2_BLACK_LUMA_MAX` 视为边框。"""
    a = np.asarray(rgb)
    if a.dtype == np.uint8:
        luma = a.astype(np.float64).mean(axis=2)
    else:
        luma = a.astype(np.float64).mean(axis=2) * (255.0 if float(np.max(a)) <= 1.0 else 1.0)
    return luma > float(MOGE2_BLACK_LUMA_MAX)


def _first_finite_scalar(v: Any) -> float:
    """从张量/数组/标量里取第一个有限数（拿不到 → 1.0，即不额外折算）。

    MoGe 的 `metric_scale` 是**逐图一个标量**（v1 口径 `depth × metric_scale`）；
    这里只取第一个有限值，不做任何聚合猜测。
    """
    a = v.detach().float().cpu().numpy() if hasattr(v, "detach") else v
    arr = np.asarray(a, dtype=np.float64).reshape(-1)
    for x in arr:
        if np.isfinite(x):
            return float(x)
    return 1.0


class _MoGe2MetricModel:
    """MoGe-2 的进程内包装（懒加载 + 输出合取掩码 + fov 对齐）。

    **不产生任何伪数据**：包/权重/输出不合法一律抛 `RuntimeError`；模型输出里的
    非有限像素标为 invalid（不填 0），全图非有限则直接报错。
    """

    name: str = METRIC_MODEL_MOGE2

    def __init__(self, model: Any, torch_mod: Any, device: str, checkpoint: str) -> None:
        self._model = model
        self._torch = torch_mod
        self.device = str(device)
        self.checkpoint = str(checkpoint)

    def infer(self, rgb: np.ndarray,
              intrinsics: Optional[np.ndarray] = None) -> MetricDepthOutput:
        """单帧前向：米制深度 + (高置信 ∧ 非黑边) 掩码 + 内参/fov 审计字段。

        `fov_x` 由 VGGT 的 K 换算后传入（§11.2 相机对齐）；权重自带的
        ``metric_scale`` 系数按 MoGe 官方口径 ``depth × metric_scale`` 折算成米
        （MoGe-2 该系数应≈1.0；若装包后与官方 API 不符，**只改这一处**）。
        """
        torch = self._torch
        img = _to_moge_tensor(rgb, torch)
        # 主点推导（尺度无关）；宽度仅作兜底，且必须是 K 所在网格的宽度。
        # 注意 `img` 是**原图**分辨率，与 VGGT 深度网格不同 —— 只有主点路径
        # 才对这个不一致免疫（见 fov_x_deg_from_intrinsics 的实测说明）。
        fov_x_deg = fov_x_deg_from_intrinsics(intrinsics)
        fov_arg = fov_x_deg
        if fov_arg is not None and not MOGE2_FOV_UNIT_DEGREES:  # pragma: no cover
            fov_arg = float(np.radians(fov_x_deg))
        with torch.no_grad():
            kwargs: dict[str, Any] = {}
            if fov_arg is not None:
                kwargs["fov_x"] = fov_arg
            try:
                out = self._model.infer(img.to(self.device), use_fp16=MOGE2_USE_FP16,
                                        **kwargs)
            except TypeError:  # 老/新签名差异（如无 use_fp16）→ 退化为最小调用
                out = self._model.infer(img.to(self.device), **kwargs)
        if not isinstance(out, Mapping) or "depth" not in out:
            raise RuntimeError(
                "MoGe-2 输出缺 depth 键（API 变更？）→ 拒绝返回假深度")
        depth = _to_numpy_2d(out["depth"])
        if depth is None:
            raise RuntimeError("MoGe-2 输出 depth 形状非法 → 拒绝返回假深度")
        if "metric_scale" in out:
            depth = depth * _first_finite_scalar(out["metric_scale"])
        finite = np.isfinite(depth) & (depth > 0.0)
        if not bool(finite.any()):
            raise RuntimeError("MoGe-2 输出 depth 全为非有限/非正 → 拒绝返回假深度")
        if "mask" not in out:
            raise RuntimeError(
                "MoGe-2 输出缺 mask（高置信/非黑边掩码缺失）→ 拒绝伪造掩码")
        mask = _to_numpy_2d(out["mask"])
        if mask is None or mask.shape != depth.shape:
            raise RuntimeError("MoGe-2 输出 mask 与 depth 形状不一致 → 拒绝伪造掩码")
        valid = finite & mask.astype(bool) & _black_border_mask(rgb)
        points = _to_numpy_3d(out.get("points"))
        intr = _to_numpy_2d(out.get("intrinsics"), allow_batch=True)
        return MetricDepthOutput(
            depth_metric=depth.astype(np.float64),
            valid_mask=valid,
            point_map=points,
            intrinsics=intr,
            fov_x_deg=fov_x_deg,
        )


def _to_numpy_2d(x: Any, allow_batch: bool = False) -> Optional[np.ndarray]:
    """torch/np → float ndarray（去 ``(1,·,·)``/``(·,·,1)`` 退化维；形状不合 → None）。

    ``allow_batch=False``（depth/mask）：必须归到 2D；``allow_batch=True``（intrinsics）：
    允许 2D ``(3,3)`` 或 3D（batch 维为 1 时也被压掉，避免下游把 ``cy`` 当 ``fx``）。
    """
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    a = np.asarray(x, dtype=np.float64)
    while a.ndim > 2 and a.shape[0] == 1:   # 去 batch 维
        a = a[0]
    if a.ndim == 3 and a.shape[-1] == 1:    # (H,W,1) → (H,W)
        a = a[..., 0]
    if not allow_batch and a.ndim != 2:
        return None
    return a if a.ndim in (2, 3) else None


def _to_numpy_3d(x: Any) -> Optional[np.ndarray]:
    """torch/np → (H,W,3) float ndarray；不可用则 None（点图非必需，不做伪数据）。"""
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    a = np.asarray(x, dtype=np.float64)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    return a if (a.ndim == 3 and a.shape[-1] == 3) else None


def make_moge2_model(device: str = "cuda",
                     checkpoint: Optional[str] = None) -> MetricDepthModel:
    """MoGe-2（MIT，arXiv:2507.02546，HF `Ruicheng/moge-2-vitl`）懒加载包装。

    真实推理走 lazy import（`import moge` / `from moge.model.v2 import MoGeModel`）；
    包不可用时 raise RuntimeError 并给出安装提示，**绝不返回假深度**。

    - 权重 ID 默认 `MOGE2_CHECKPOINT`；`checkpoint` 可覆盖（本地目录 / 其它 repo id）。
    - ``device="cuda"`` 时要求 CUDA 真的可用：**不静默降级到 CPU**——否则 C5 的
      资源/延迟证据会变成另一个硬件的数字（§11.4 C5）。
    - 本函数只在调用期 import torch/moge（模块 import 期不拉重依赖）。
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "MoGe-2 需要 torch：`pip install torch` 后重试（当前环境未安装）。"
            f"原始错误：{exc}") from exc
    try:
        from moge.model.v2 import MoGeModel  # lazy import：包不可用时才报错
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "MoGe-2 不可用：未安装 `moge` 包（§11.2 首个 PoC 模型）。安装提示：\n"
            "  pip install git+https://github.com/microsoft/MoGe.git   # MIT\n"
            "  然后确保权重可离线加载：HF `Ruicheng/moge-2-vitl`\n"
            f"原始错误：{exc}") from exc

    dev = str(device)
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={dev!r} 但 CUDA 不可用 → 拒绝静默降到 CPU"
            "（会把 C5 的资源/延迟证据换成另一台硬件；要跑 CPU 请显式传 device='cpu'）")
    ckpt = str(checkpoint or MOGE2_CHECKPOINT)
    try:
        model = MoGeModel.from_pretrained(ckpt)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"MoGe-2 权重加载失败（checkpoint={ckpt!r}）：{exc}\n"
            "离线环境请先预下载权重；**不提供任何随机初始化/占位模型**，"
            "因为那会产出看着像米制、实则无意义的深度。") from exc
    model = model.to(dev).eval()
    return _MoGe2MetricModel(model=model, torch_mod=torch, device=dev, checkpoint=ckpt)


__all__ = [
    "C4_MIN_QUESTIONS",
    "C5_MAX_LATENCY_MS_PER_FRAME",
    "C5_MAX_PEAK_GPU_GIB",
    "CONF_SOFT_KEEP_QUANTILE",
    "MAD_TO_SIGMA",
    "MAX_SCALE_DISPERSION",
    "METRIC_FUSION_VERSION",
    "METRIC_MODEL_METRIC3D_V2",
    "METRIC_MODEL_MOGE2",
    "METRIC_MODEL_NONE",
    "MIN_FRAMES_FOR_MAD",
    "MIN_POSITIVE_DEPTH",
    "MIN_VALID_FRAMES_FOR_SUCCESS",
    "MIN_VALID_FRAME_RATIO",
    "MIN_VALID_PIXELS",
    "MOGE2_CHECKPOINT",
    "MOGE2_FOV_UNIT_DEGREES",
    "OUTLIER_MAD_K",
    "PAPER_MIN_SEEDS",
    "MetricDepthModel",
    "MetricDepthOutput",
    "PerFrameScale",
    "ScaleFusionResult",
    "acceptance_report",
    "fov_x_deg_from_intrinsics",
    "fuse_metric_scale",
    "make_moge2_model",
    "per_frame_receipt",
    "per_frame_scale",
    "read_per_frame_receipt",
    "threshold_snapshot",
    "write_per_frame_receipt",
]
