"""M4 无真值质量门（v6 §10，D11）—— 主门 = warp 内点率 **且** 分组点云重叠率。

纯几何实现：只吃 `frames / depth_maps / c2w_list / intrinsics / point_map /
depth_conf`，不用真值、不加载任何额外模型、不读题目（§10.1/§10.5）。
点云最近邻走 `scipy.spatial.cKDTree`，其余是纯 numpy 向量化；同一输入两次调用
逐位一致（任何抽样都走固定种子，见 `_SUBSAMPLE_SEED`），32 帧 @518×392 秒级可跑。

四条纪律（照抄 §10，不得"顺手优化"掉）：

1. **多指标不得单挑**（§10.1）：`main_gate_passed` 恒为"warp 内点率 ≥ τ_warp"
   **且**"分组点云重叠率 ≥ τ_cloud"的逻辑与。SysCON3D 依据：前馈 backbone 会
   幻觉出跨视图一致性，任何单项都能被自洽的幻觉骗过。
2. **报率不报绝对距离**（§10.1/§10.6）：VGGT 点云是"中位归一化、无米制尺度"，
   点云重叠一律用**相对阈值**（场景稳健尺度的百分比）判内点，代码里**永不**出现
   米制绝对阈值，也禁止拿"室内 1–3m"当硬门。
3. **G5 永久 not_available**（§10.4）：绝不用 `depth_conf/point_conf`、点云密度等
   代理值冒充 BA 重投影残差。`depth_conf` 只在 **conf-warp 单调性自检通过后**作
   软权重；不单调（或自检证据不足）**只降权、不否决**主门（§10.3）。
4. **未知 ≠ 坏值**（fail-closed，§10.5 诚实边界）：算不出来的量返回 `NaN`（不是 0、
   不是 False）并把原因写进 `warnings`/`reason`，由调用方决定怎么处置。

阈值全部 `[TODO_CALIBRATE]`：给出的起始参考值只是让链路能跑起来，**必须**按 §10.6
的最小 PoC（1 个好 episode + 抽稀 / 跨场景帧 / 运动模糊三种注入退化）在自有数据上
重标后才能当门用。
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

# 版本标识单一事实源（§5.2：`quality_metric_version="v6-warp-overlap-no-g5"`）
from skill3d.schemas.reconstruction import QUALITY_METRIC_VERSION as M4_GATE_VERSION

NaN = float("nan")
_EPS = 1e-12
# 抽样随机种子（硬要求：确定性，禁止时间戳/无种子随机；见 §0 实现纪律 7）
_SUBSAMPLE_SEED: int = 0

# ---------------------------------------------------------------------------
# 阈值常量（§10.1/§10.3：全部 TODO_CALIBRATE，须按 §10.6 最小 PoC 重标）
# ---------------------------------------------------------------------------

# ---- 主门阈值（§10.1）----
TH_WARP_INLIER: float = 0.5       # TODO_CALIBRATE: warp 相对深度内点率下限（起始参考）
TH_CLOUD_OVERLAP: float = 0.3     # TODO_CALIBRATE: 前/后子云双向重叠率下限（起始参考）

# ---- warp 口径 ----
N_NEIGHBOR_FRAMES: int = 5        # 每帧取约 5 个近邻帧（§10.1 起始参考；是采样口径不是阈值）
TH_WARP_REL_DEPTH_TOL: float = 0.05   # TODO_CALIBRATE: 相对深度差容差 |z−d|/d
TH_WARP_PHOTO_TOL: float = 0.10       # TODO_CALIBRATE: 光度差容差（[0,1]，= 25.5/255 灰度）

# ---- 点云重叠口径 ----
TH_CLOUD_REL_TOL: float = 0.03    # TODO_CALIBRATE: 内点判据 = 场景稳健尺度的百分比
MAX_CLOUD_POINTS: int = 100000    # TODO_CALIBRATE: 每个子云下采样上限（与 τ_cloud **联标**：
                                  #   采样越稀 → 最近邻距离越大 → 重叠率越低。经验关系：
                                  #   采样间距 ≈ sqrt(可见表面面积/max_points)，要不被它吃掉，
                                  #   需 max_points ≳ 面积/(rel_tol·场景尺度)²，故两者必须同标）

# ---- conf-warp 自检（§10.3）----
TH_CONF_DOWNWEIGHT: float = 0.5   # TODO_CALIBRATE: conf 不可信时的**软权重**（降权，不否决）
CONF_OPTIONAL_MASK_C: float = 2.0  # TODO_CALIBRATE: `C>2` 仅作**可选**掩码（§10.3），不是硬阈值

# ---- 诊断告警线（§10.2：只产告警，不决定主门）----
TH_PHOTO_WARN: float = 0.2        # TODO_CALIBRATE: warp 光度内点率过低告警线

# ---- 结构性下限（性能/退化保护，非标定阈值）----
N_WARP_SAMPLES_PER_PAIR: int = 4096   # 每对帧的采样像素上限（步长网格抽样，确定性）
MIN_WARP_PAIR_SAMPLES: int = 64       # 单对有效可比像素下限，低于则该对不参与聚合
MIN_CLOUD_POINTS: int = 100           # 子云点数下限，低于则重叠率记 NaN（不伪造）
MIN_CONF_WARP_SAMPLES: int = 100      # conf-warp 自检的有效样本下限
MIN_CONF_WARP_BINS: int = 3           # conf-warp 自检的可用分桶数下限（低于 → None）
MIN_CONF_BIN_SAMPLES: int = 20        # 单个 conf 分桶的样本下限（低于则弃桶）

__all__ = [
    "CONF_OPTIONAL_MASK_C",
    "M4_GATE_VERSION",
    "MAX_CLOUD_POINTS",
    "MIN_CONF_WARP_BINS",
    "MIN_CONF_WARP_SAMPLES",
    "N_NEIGHBOR_FRAMES",
    "N_WARP_SAMPLES_PER_PAIR",
    "TH_CLOUD_OVERLAP",
    "TH_CLOUD_REL_TOL",
    "TH_CONF_DOWNWEIGHT",
    "TH_WARP_INLIER",
    "TH_WARP_PHOTO_TOL",
    "TH_WARP_REL_DEPTH_TOL",
    "conf_optional_mask",
    "conf_warp_monotonic",
    "cross_view_warp_inliers",
    "default_thresholds",
    "grouped_cloud_overlap",
    "main_gate",
    "warp_residual_map",
]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _to_float(v: Any) -> float:
    """任意输入 → float；不可解析记 NaN（**不抛异常**：门必须能对坏值 fail-closed）。"""
    if v is None or isinstance(v, (bool, np.bool_)):
        return NaN
    try:
        return float(v)
    except (TypeError, ValueError):
        return NaN


def _finite_ratio(num: int, den: int) -> float:
    """内点率；分母为 0 → NaN（"未知"≠ 0，"0 个可比像素"既不是全内点也不是零内点）。"""
    return float(num) / float(den) if den > 0 else NaN


def _sample_grid(h: int, w: int, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """确定性步长网格抽样：返回采样像素的 (gy, gx)（覆盖整幅图，不依赖随机数）。

    抽样是**性能手段**（每对帧 O(采样数) 而非 O(HW)）；同一输入两次调用采样点相同。
    """
    total = max(1, int(h) * int(w))
    step = max(1, int(np.floor(np.sqrt(total / max(1, int(max_samples))))))
    ys = np.arange(0, int(h), step, dtype=np.int64)
    xs = np.arange(0, int(w), step, dtype=np.int64)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return gy.ravel(), gx.ravel()


def _neighbor_pairs(n_frames: int, n_neighbors: int) -> list[tuple[int, int]]:
    """每帧取 |i−j| 最小的 `n_neighbors` 个近邻帧（并列时取小下标）→ 有序对列表。

    用**帧序近邻**（视频帧序即采集序）而非位姿距离：位姿本身来自 VGGT，用它的噪声
    去挑配对会把门变成"自己判自己"。n_neighbors ≤ 0 或帧数 < 2 → 空列表。
    """
    k = int(n_neighbors)
    if k <= 0 or n_frames < 2:
        return []
    pairs: list[tuple[int, int]] = []
    for i in range(n_frames):
        order = sorted((abs(j - i), j) for j in range(n_frames) if j != i)[:k]
        pairs.extend((i, j) for _, j in order)
    return pairs


def _frames_on_grid(frames: Sequence[np.ndarray], hw: tuple[int, int]) -> np.ndarray:
    """帧缩放到深度网格（C-8 教训：坐标必须与深度同网格），返回 uint8 (N,H,W,3)。"""
    h, w = int(hw[0]), int(hw[1])
    out = np.empty((len(frames), h, w, 3), dtype=np.uint8)
    for idx, f in enumerate(frames):
        a = np.asarray(f)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        if a.ndim != 3:
            raise ValueError(f"frames[{idx}] 维度异常: {a.shape}")
        if a.shape[2] == 4:
            a = a[..., :3]
        if a.shape[2] != 3:
            raise ValueError(f"frames[{idx}] 必须是 RGB（3 通道），收到 {a.shape}")
        if a.shape[0] != h or a.shape[1] != w:
            import cv2

            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        out[idx] = np.clip(a, 0, 255).astype(np.uint8)
    return out


def _validate_geometry(
    frames: Optional[Sequence[np.ndarray]],
    depth_maps: np.ndarray,
    c2w_list: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], Optional[np.ndarray]]:
    """校验并规范化几何输入 → (depth(N,H,W), c2w(N,4,4), K(N,3,3), (H,W), frames(N,H,W,3)|None)。

    **形状/内参非法一律抛 ValueError**（这是输入契约被破坏，属接线 bug，不该被静默吞掉）；
    深度里的 NaN/Inf 是**合法的逐像素无效值**，由各函数逐像素 fail-closed 处理。
    """
    d = np.asarray(depth_maps, dtype=np.float64)
    if d.ndim != 3:
        raise ValueError(f"depth_maps 必须是 (N,H,W)，收到 {d.shape}")
    n, h, w = int(d.shape[0]), int(d.shape[1]), int(d.shape[2])
    if n < 1:
        raise ValueError("depth_maps 帧数为 0")

    c2w = np.asarray(c2w_list, dtype=np.float64)
    if c2w.shape != (n, 4, 4):
        raise ValueError(f"c2w_list 必须是 (N,4,4)（N={n}），收到 {c2w.shape}")
    if not np.all(np.isfinite(c2w)):
        raise ValueError("c2w_list 含 NaN/Inf（位姿契约被破坏，拒绝静默降级）")

    k3 = np.asarray(intrinsics, dtype=np.float64)
    if k3.shape == (3, 3):
        k3 = np.broadcast_to(k3, (n, 3, 3)).copy()
    if k3.shape != (n, 3, 3):
        raise ValueError(f"intrinsics 必须是 (3,3) 或 (N,3,3)（N={n}），收到 {k3.shape}")
    if not np.all(np.isfinite(k3)):
        raise ValueError("intrinsics 含 NaN/Inf")
    if np.any(k3[:, 0, 0] <= 0) or np.any(k3[:, 1, 1] <= 0):
        raise ValueError(f"intrinsics 焦距必须为正，收到 fx={k3[:, 0, 0]!r} fy={k3[:, 1, 1]!r}")

    frames_arr: Optional[np.ndarray] = None
    if frames is not None:
        if len(frames) != n:
            raise ValueError(f"frames 帧数 {len(frames)} 与 depth_maps {n} 不一致")
        frames_arr = _frames_on_grid(list(frames), (h, w))
    return d, c2w, k3, (h, w), frames_arr


def _pair_warp_stats(
    i: int,
    j: int,
    gy: np.ndarray,
    gx: np.ndarray,
    depth: np.ndarray,
    frames: Optional[np.ndarray],
    c2w: np.ndarray,
    k: np.ndarray,
    hw: tuple[int, int],
    *,
    rel_depth_tol: float,
    photo_tol: float,
) -> dict:
    """单对帧 (i→j) 的 warp 统计（数组级，向量化）。

    口径（§10.1-1）：把第 i 帧采样像素按 `z = depth[i]` 反投影到世界系 → 投进第 j 帧
    → 比较**投影深度 z_j 与第 j 帧深度图在该像素的取值 d_j** 的**相对**差
    `|z_j − d_j|/d_j ≤ rel_depth_tol` 记深度内点（全部无量纲，无米制含义）；
    同时比较两帧该像素的 RGB 均值绝对差 ≤ photo_tol 记光度内点。
    不相交/深度无效/投影越界的采样**不计入分母**（分母 = 两帧都有有效深度的可比像素数）。
    """
    h, w = int(hw[0]), int(hw[1])
    d_i = np.asarray(depth[i][gy, gx], dtype=np.float64)
    finite_i = np.isfinite(d_i) & (d_i > 0.0)

    fx_i, fy_i, cx_i, cy_i = (float(k[i, 0, 0]), float(k[i, 1, 1]),
                              float(k[i, 0, 2]), float(k[i, 1, 2]))
    fx_j, fy_j, cx_j, cy_j = (float(k[j, 0, 0]), float(k[j, 1, 1]),
                              float(k[j, 0, 2]), float(k[j, 1, 2]))

    # 反投影：像素 × 相机系 z 深度 → 世界系（VGGT 世界系 = 首帧相机系）
    z_cam = np.where(finite_i, d_i, 0.0)
    x_cam = (gx.astype(np.float64) - cx_i) / fx_i * z_cam
    y_cam = (gy.astype(np.float64) - cy_i) / fy_i * z_cam
    p_w = np.stack([x_cam, y_cam, z_cam], axis=1) @ c2w[i, :3, :3].T + c2w[i, :3, 3]

    # 世界系 → 第 j 帧相机系（R_jᵀ(p − t_j)）
    p_c = (p_w - c2w[j, :3, 3]) @ c2w[j, :3, :3]
    z_j = p_c[:, 2]
    z_ok = np.isfinite(z_j) & (z_j > _EPS)
    denom = np.where(z_ok, z_j, 1.0)          # 只为了让被掩码的像素不产生 NaN/告警
    u = fx_j * p_c[:, 0] / denom + cx_j
    v = fy_j * p_c[:, 1] / denom + cy_j

    inside = finite_i & z_ok & (u >= 0.0) & (u <= w - 1) & (v >= 0.0) & (v <= h - 1)
    n_valid = int(np.count_nonzero(inside))
    ui = np.clip(np.rint(u[inside]).astype(np.int64), 0, w - 1)
    vi = np.clip(np.rint(v[inside]).astype(np.int64), 0, h - 1)

    d_j = np.asarray(depth[j][vi, ui], dtype=np.float64)
    ok = np.isfinite(d_j) & (d_j > 0.0)
    z_in = z_j[inside]
    resid = np.full(ok.shape, np.inf, dtype=np.float64)      # 未定义处记 +inf（显式，不是 0）
    resid[ok] = np.abs(z_in[ok] - d_j[ok]) / d_j[ok]
    depth_in = ok & (resid <= float(rel_depth_tol))
    # 采样网格长度（n_src）的残差图，无观测处 NaN —— 供 `warp_residual_map` 聚合
    resid_full = np.full(int(gy.size), NaN, dtype=np.float64)
    idx_in = np.flatnonzero(inside)
    resid_full[idx_in[ok]] = resid[ok]

    n_photo_in = 0
    photo_in = np.zeros(ok.shape, dtype=bool)
    if frames is not None:
        fi = frames[i][gy[inside], gx[inside]].astype(np.float32)
        fj = frames[j][vi, ui].astype(np.float32)
        diff = np.abs(fi - fj).mean(axis=1) / 255.0
        photo_in = ok & (diff <= float(photo_tol))
        n_photo_in = int(np.count_nonzero(photo_in))

    n_ok = int(np.count_nonzero(ok))
    return {
        "i": int(i), "j": int(j),
        "n_src": int(gy.size), "n_valid": n_valid, "n_ok": n_ok,
        "n_depth_inlier": int(np.count_nonzero(depth_in)),
        "n_photo_inlier": n_photo_in,
        "resid": resid, "resid_full": resid_full, "ok": ok, "gy": gy, "gx": gx,
    }


# ---------------------------------------------------------------------------
# §10.1-1 跨视图 warp 内点率
# ---------------------------------------------------------------------------

def cross_view_warp_inliers(
    frames: Sequence[np.ndarray],
    depth_maps: np.ndarray,
    c2w_list: np.ndarray,
    intrinsics: np.ndarray,
    *,
    n_neighbors: int = N_NEIGHBOR_FRAMES,
    rel_depth_tol: float = TH_WARP_REL_DEPTH_TOL,
    photo_tol: float = TH_WARP_PHOTO_TOL,
) -> dict:
    """§10.1-1：每帧选约 `n_neighbors` 个近邻帧，报相对深度差内点率 + warp 光度内点率。

    参数
    ----
    frames : (N,H,W,3) RGB uint8（若分辨率与深度网格不同会先缩放到深度网格）
    depth_maps : (N,H,W) VGGT 归一化深度（相机系 z 深度；NaN/Inf 视为无效像素）
    c2w_list : (N,4,4) camera→world（世界系 = 首帧相机系，`c2w[0]=I`）
    intrinsics : (3,3) 或 (N,3,3)

    返回
    ----
    {"warp_inlier_ratio": float, "warp_photometric_inlier_ratio": float,
     "n_pairs": int, "per_pair": list[dict]}

    - 两个比率都是**按可比像素数池化**（pooled，不是配对平均）；分母 = 两帧都有有效深度
      的可比像素数，见 `_pair_warp_stats`。有效配对不足（全部被 `MIN_WARP_PAIR_SAMPLES`
      挡掉）→ 返回 **NaN**（"算不出来"≠0，硬要求 8）。
    - `per_pair` 是 JSON 友好的诊断列表（不含逐像素大数组）；每像素残差请用
      `warp_residual_map`（与这里同一个核心实现，口径唯一）。
    - 无光度信息（frames 传 None）时 `warp_photometric_inlier_ratio` 记 NaN：
      它**不参与主门**（§10.1 主门只由深度内点率 + 重叠率决定），只作诊断。
    """
    depth, c2w, k, hw, frames_arr = _validate_geometry(frames, depth_maps, c2w_list, intrinsics)
    n = int(depth.shape[0])
    gy, gx = _sample_grid(hw[0], hw[1], N_WARP_SAMPLES_PER_PAIR)

    per_pair: list[dict] = []
    n_pairs = 0
    d_tot = d_in = p_in = 0
    for i, j in _neighbor_pairs(n, int(n_neighbors)):
        st = _pair_warp_stats(i, j, gy, gx, depth, frames_arr, c2w, k, hw,
                              rel_depth_tol=rel_depth_tol, photo_tol=photo_tol)
        n_ok = st["n_ok"]
        enough = n_ok >= MIN_WARP_PAIR_SAMPLES
        if enough:
            n_pairs += 1
            d_tot += n_ok
            d_in += st["n_depth_inlier"]
            p_in += st["n_photo_inlier"]
        per_pair.append({
            "i": st["i"], "j": st["j"],
            "n_src": st["n_src"], "n_valid": st["n_valid"], "n_ok": n_ok,
            "n_depth_inlier": st["n_depth_inlier"], "n_photo_inlier": st["n_photo_inlier"],
            "depth_inlier_ratio": _finite_ratio(st["n_depth_inlier"], n_ok) if enough else NaN,
            "photo_inlier_ratio": (
                _finite_ratio(st["n_photo_inlier"], n_ok) if (enough and frames_arr is not None)
                else NaN),
            "skipped": "" if enough else "insufficient_comparable_pixels",
        })

    return {
        "warp_inlier_ratio": _finite_ratio(d_in, d_tot) if n_pairs else NaN,
        "warp_photometric_inlier_ratio": (
            _finite_ratio(p_in, d_tot) if (n_pairs and frames_arr is not None) else NaN),
        "n_pairs": int(n_pairs),
        "per_pair": per_pair,
    }


def warp_residual_map(
    depth_maps: np.ndarray,
    c2w_list: np.ndarray,
    intrinsics: np.ndarray,
    *,
    n_neighbors: int = N_NEIGHBOR_FRAMES,
    rel_depth_tol: float = TH_WARP_REL_DEPTH_TOL,
) -> np.ndarray:
    """每像素 warp 残差 `|z_j − d_j|/d_j`（(N,H,W) float32，无观测处 NaN）。

    这是 `conf_warp_monotonic` 需要的 `warp_residual` 输入，与 `cross_view_warp_inliers`
    共用同一核心实现（`_pair_warp_stats`），避免出现两套口径。同一像素在多对帧上都有
    观测时取**中位数**（对单个坏配对稳健）。
    """
    depth, c2w, k, hw, _ = _validate_geometry(None, depth_maps, c2w_list, intrinsics)
    n = int(depth.shape[0])
    gy, gx = _sample_grid(hw[0], hw[1], N_WARP_SAMPLES_PER_PAIR)
    out = np.full((n, int(hw[0]), int(hw[1])), NaN, dtype=np.float32)
    for i in range(n):
        rows: list[np.ndarray] = []
        for _, j in sorted((abs(j - i), j) for j in range(n) if j != i)[:int(n_neighbors)]:
            st = _pair_warp_stats(i, j, gy, gx, depth, None, c2w, k, hw,
                                  rel_depth_tol=rel_depth_tol, photo_tol=0.0)
            rows.append(st["resid_full"])
        if not rows:
            continue
        with warnings.catch_warnings():
            # 某像素在所有配对上都没观测 → 全 NaN 列，nanmedian 会发 "All-NaN slice"
            # 告警；结果本来就该是 NaN（"无观测"），这里局部静音，不改语义。
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.vstack(rows), axis=0)
        out[i][gy, gx] = med.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# §10.1-2 分组点云重叠率（相对阈值，报率不报绝对距离）
# ---------------------------------------------------------------------------

def _subsample_cloud(points: np.ndarray, max_points: int, *,
                     seed: int = _SUBSAMPLE_SEED) -> np.ndarray:
    """确定性下采样：先等步长粗抽（O(1) 内存），再固定种子均匀抽到上限。

    固定种子 → 同一输入两次调用得到同一子集（硬要求 7：禁止无种子随机/时间戳）。
    """
    flat = np.asarray(points).reshape(-1, 3)
    if flat.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    flat = flat[np.isfinite(flat).all(axis=1)]
    if flat.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    cap = max(1, int(max_points))
    if flat.shape[0] > cap:
        stride = int(np.ceil(flat.shape[0] / (4.0 * cap)))
        flat = flat[::stride]
        if flat.shape[0] > cap:
            rng = np.random.default_rng(seed)     # 固定种子（每次调用新建 → 可重复）
            sel = np.sort(rng.choice(flat.shape[0], size=cap, replace=False))
            flat = flat[sel]
    return np.ascontiguousarray(flat, dtype=np.float64)


def _robust_scene_scale(points: np.ndarray) -> float:
    """场景稳健尺度 = 全部点到**中位中心**距离的中位数（无量纲，VGGT 归一化单位）。

    这是 §10.1 "场景中位深度"的可复现代理口径：由点云自身算出，故与 VGGT 的任意
    归一化倍数无关（换尺度不改判据），**绝不**解释成米。
    """
    if points.shape[0] == 0:
        return NaN
    center = np.median(points, axis=0)
    rad = np.linalg.norm(points - center, axis=1)
    return float(np.median(rad))


def grouped_cloud_overlap(
    point_map: np.ndarray,
    *,
    rel_tol: float = TH_CLOUD_REL_TOL,
    max_points: int = MAX_CLOUD_POINTS,
) -> dict:
    """§10.1-2：前 16 vs 后 16 帧两个自举子云互 NN，报**相对阈值**双向内点率。

    判据（**没有米制含义**）：`最近邻距离 ≤ rel_tol × 场景稳健尺度` 记内点；
    阈值来自点云自身（`_robust_scene_scale`），故对 VGGT 的中位归一化倍数不变
    （尺度不变性有单测）。**永不**用绝对距离/米制阈值判分（§10.6）。

    参数
    ----
    point_map : (N,H,W,3) 世界系点图（VGGT 中位归一化，无米制尺度）
    rel_tol : TODO_CALIBRATE 相对阈值（场景稳健尺度的百分比）
    max_points : TODO_CALIBRATE 每个子云的下采样上限（**与 rel_tol 联标**：下采样越稀，
        最近邻距离越大、重叠率越低；两半必须用同一个 max_points 才能比阈值）

    返回
    ----
    {"cloud_overlap_ratio": float, "n_fwd": int, "n_bwd": int,
     "n_fwd_inliers": int, "n_bwd_inliers": int, "fwd_ratio": float, "bwd_ratio": float,
     "scene_scale": float, "rel_tol": float, "max_points": int, "reason": str}

    - `cloud_overlap_ratio` = 双向**池化**内点率 `(fwd_in + bwd_in)/(n_fwd + n_bwd)`，
      两条方向分别也可从 `fwd_ratio` / `bwd_ratio` 读到；`n_fwd`/`n_bwd` 是两个方向的
      **查询点数**（下采样后）。
    - 帧数 < 2 / 子云点数不足 / 尺度退化 → `cloud_overlap_ratio = NaN` 且 `reason`
      写明原因（**不返回 0**：0 是"确实不重叠"这个强判断，不能拿来表示"算不出来"）。
    - `scene_scale`/`rel_tol` 仅供审计与标定；它们是无量纲单位，**不得**当米制阈值使用。
    """
    pm = np.asarray(point_map)
    if pm.ndim != 4 or pm.shape[-1] != 3:
        raise ValueError(f"point_map 必须是 (N,H,W,3)，收到 {pm.shape}")
    n = int(pm.shape[0])
    rt = float(rel_tol)
    out = {
        "cloud_overlap_ratio": NaN, "n_fwd": 0, "n_bwd": 0,
        "n_fwd_inliers": 0, "n_bwd_inliers": 0, "fwd_ratio": NaN, "bwd_ratio": NaN,
        "scene_scale": NaN, "rel_tol": rt, "max_points": int(max_points),
        "reason": "",
    }
    if n < 2:
        out["reason"] = "n_frames_lt_2"
        return out
    if not np.isfinite(rt) or rt <= 0:
        out["reason"] = "invalid_rel_tol"
        return out

    split = n // 2                                  # 前 16 vs 后 16（N=32 时正好 16/16）
    fwd = _subsample_cloud(pm[:split], max_points)
    bwd = _subsample_cloud(pm[split:], max_points)
    if fwd.shape[0] < MIN_CLOUD_POINTS or bwd.shape[0] < MIN_CLOUD_POINTS:
        out["reason"] = "insufficient_points"
        return out

    scale = _robust_scene_scale(np.vstack([fwd, bwd]))
    out["scene_scale"] = scale
    if not np.isfinite(scale) or scale <= 0:
        out["reason"] = "degenerate_scene_scale"
        return out
    thr = rt * scale

    tree_bwd = cKDTree(bwd)
    tree_fwd = cKDTree(fwd)
    d_fwd, _ = tree_bwd.query(fwd, k=1)             # NN 距离（不建笛卡尔积）
    d_bwd, _ = tree_fwd.query(bwd, k=1)
    n_fwd_in = int(np.count_nonzero(d_fwd <= thr))
    n_bwd_in = int(np.count_nonzero(d_bwd <= thr))
    n_fwd, n_bwd = int(fwd.shape[0]), int(bwd.shape[0])
    out.update({
        "cloud_overlap_ratio": _finite_ratio(n_fwd_in + n_bwd_in, n_fwd + n_bwd),
        "n_fwd": n_fwd, "n_bwd": n_bwd,
        "n_fwd_inliers": n_fwd_in, "n_bwd_inliers": n_bwd_in,
        "fwd_ratio": _finite_ratio(n_fwd_in, n_fwd),
        "bwd_ratio": _finite_ratio(n_bwd_in, n_bwd),
        "reason": "",
    })
    return out


# ---------------------------------------------------------------------------
# §10.3 conf-warp 单调性自检（conf 只作软权重，不作硬门）
# ---------------------------------------------------------------------------

def conf_warp_monotonic(
    depth_conf: np.ndarray,
    warp_residual: np.ndarray,
    *,
    n_bins: int = 5,
) -> dict:
    """§10.3 自检：conf 分桶 vs warp 差**中位数**应单调（conf 越高 → 残差中位数越低）。

    参数
    ----
    depth_conf : (N,H,W) VGGT depth 置信度（`C = exp(Σ)+1` 口径；越高越自信）
    warp_residual : (N,H,W) 每像素 warp 残差（用 `warp_residual_map` 生成；无观测处 NaN）
    n_bins : 分桶数（分位分桶，等样本量）

    返回
    ----
    {"monotonic": bool|None, "spearman": float|None, "bin_medians": list[float],
     "n_samples": int, "n_bins_used": int, "bin_conf_centers": list[float],
     "bin_counts": list[int], "reason": str}

    - `monotonic=True` ⇔ 各桶残差中位数随 conf 升序**不增**；`spearman` 是桶心 conf 与
      桶残差中位数的秩相关（≈ −1 表示健康；符号约定：conf 越高残差越低）。
    - **数据不足/退化 → `monotonic=None`（不是 False）**，`reason` 写明：
      `insufficient_samples` / `conf_constant` / `insufficient_bins` /
      `degenerate_residuals`（残差全同一个值 → 无从判单调）。调用方据此"降权"而非"否决"
      （§10.3：conf 只作软权重）。真·不单调才返回 False。
    - **本函数不产生任何硬阈值**：`C>2` 之类只能作可选掩码（见 `conf_optional_mask`）。
    """
    c = np.asarray(depth_conf, dtype=np.float64)
    r = np.asarray(warp_residual, dtype=np.float64)
    if c.shape != r.shape:
        raise ValueError(f"depth_conf {c.shape} 与 warp_residual {r.shape} 形状必须一致")
    res = {
        "monotonic": None, "spearman": None, "bin_medians": [],
        "n_samples": 0, "n_bins_used": 0, "bin_conf_centers": [], "bin_counts": [],
        "reason": "",
    }
    if int(n_bins) < 2:
        res["reason"] = "invalid_n_bins"
        return res
    ok = np.isfinite(c.ravel()) & np.isfinite(r.ravel())
    n_ok = int(np.count_nonzero(ok))
    res["n_samples"] = n_ok
    if n_ok < MIN_CONF_WARP_SAMPLES:
        res["reason"] = "insufficient_samples"
        return res
    c = c.ravel()[ok]
    r = r.ravel()[ok]
    if np.unique(c).size < 2:
        res["reason"] = "conf_constant"
        return res

    edges = np.unique(np.quantile(c, np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]))
    if edges.size == 0:
        res["reason"] = "conf_constant"
        return res
    idx = np.searchsorted(edges, c, side="right")     # 0 … edges.size

    medians: list[float] = []
    centers: list[float] = []
    counts: list[int] = []
    for b in range(edges.size + 1):
        m = idx == b
        cnt = int(np.count_nonzero(m))
        if cnt < MIN_CONF_BIN_SAMPLES:
            continue                                  # 空桶/样本过少的桶直接弃（不掺噪声）
        medians.append(float(np.median(r[m])))
        centers.append(float(np.mean(c[m])))
        counts.append(cnt)
    res["bin_medians"] = medians
    res["bin_conf_centers"] = centers
    res["bin_counts"] = counts
    res["n_bins_used"] = len(medians)
    if len(medians) < MIN_CONF_WARP_BINS:
        res["reason"] = "insufficient_bins"
        return res
    if not np.isfinite(np.asarray(medians)).all():
        res["reason"] = "non_finite_bin_median"
        return res
    if max(medians) - min(medians) <= 0.0:
        res["reason"] = "degenerate_residuals"        # 残差处处相同 → 单调性无从谈起
        return res

    res["monotonic"] = bool(np.all(np.diff(np.asarray(medians)) <= 0.0))
    res["spearman"] = _spearman(centers, medians)
    res["reason"] = ""
    return res


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """秩相关（常量输入 → None；样本 < 2 → None）。确定性、无随机。"""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    rx = rankdata(np.asarray(x, dtype=np.float64))
    ry = rankdata(np.asarray(y, dtype=np.float64))
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return rho if np.isfinite(rho) else None


def conf_optional_mask(
    depth_conf: np.ndarray,
    *,
    c_floor: float = CONF_OPTIONAL_MASK_C,
) -> np.ndarray:
    """§10.3 的 `C>2` **可选掩码**（滤飞点用）：返回 `conf ≥ c_floor` 的 bool 掩码。

    纪律（硬要求 5）：本函数**不参与任何门**，`c_floor` **不是**硬阈值 ——
    DTU 审计显示 conf 过自信（point pUI≈61.2%、depth pAC≈48.5%，`C = exp(Σ)+1`），
    固定 `C>2` 当门会误杀。调用方若启用本掩码，必须在 trace 里记录"启用了可选掩码
    及其 c_floor"，且**不得**用启用/未启用改变 `main_gate_passed`。NaN/Inf → False。
    """
    c = np.asarray(depth_conf, dtype=np.float64)
    return np.isfinite(c) & (c >= float(c_floor))


# ---------------------------------------------------------------------------
# §10.1 主门（AND，多指标不得单挑）
# ---------------------------------------------------------------------------

def default_thresholds() -> dict[str, float]:
    """主门阈值快照（可写入 `QualityMetrics.gate_thresholds`，trace 可审计）。"""
    return {
        "warp_inlier_ratio": float(TH_WARP_INLIER),
        "cloud_overlap_ratio": float(TH_CLOUD_OVERLAP),
    }


def _merge_thresholds(base: dict[str, float],
                      override: Optional[dict]) -> tuple[dict[str, float], list[str]]:
    """阈值覆盖：未知键/非有限值一律**忽略并告警**（沿用起始参考值，不静默改门）。"""
    out = dict(base)
    warns: list[str] = []
    if not override:
        return out, warns
    if not isinstance(override, dict):
        warns.append(f"thresholds 不是 dict（收到 {type(override).__name__}）→ 全部沿用默认阈值")
        return out, warns
    for key, val in override.items():
        if key not in out:
            warns.append(f"thresholds 含未知键 {key!r} → 忽略（本门只用 {sorted(out)}）")
            continue
        fv = _to_float(val)
        if not np.isfinite(fv):
            warns.append(f"thresholds[{key!r}] 非有限（{val!r}）→ 忽略，沿用 {out[key]}")
            continue
        out[key] = float(fv)
    return out, warns


def main_gate(
    quality_inputs: dict,
    *,
    thresholds: dict | None = None,
) -> dict:
    """§10.1 主门 = warp 内点率 ≥ τ_warp **且** 分组点云重叠率 ≥ τ_cloud。

    **多指标不得单挑**（SysCON3D：前馈 backbone 会幻觉跨视图一致性）：两个子项都过
    `main_gate_passed` 才为 True，任一 NaN/非有限 → 该子项 False 且进 `warnings`
    （fail-closed）。本函数**不读真值、不读米制尺度**，也不把 conf 当门。

    `quality_inputs` 取值口径（同一 key 两种来源，**显式值优先**）：

    - 已算好的指标：`{"warp_inlier_ratio": f, "warp_photometric_inlier_ratio": f,
      "cloud_overlap_ratio": f, "conf_warp_monotonic": bool|None,
      "conf_warp_spearman": f|None, ...}`（把 `cross_view_warp_inliers` /
      `grouped_cloud_overlap` / `conf_warp_monotonic` 的返回合并进来即可）；
    - 或原始数组：`frames / depth_maps / c2w_list / intrinsics` → 现场算 warp；
      `point_map` → 现场算重叠率。原始数组路径内部**捕获异常**转 NaN + 告警
      （坏输入 fail-closed，不让门把 episode 打崩）。

    返回
    ----
    {"main_gate_passed": bool,           # 恒 = 两个子项的 AND
     "sub_results": {"warp_inlier_ratio": bool, "cloud_overlap_ratio": bool},
     "values": {"warp_inlier_ratio": f, "warp_photometric_inlier_ratio": f,
                "cloud_overlap_ratio": f},     # 算不出的记 NaN
     "thresholds": {...},                # 本次判定实际用的阈值快照（写 gate_thresholds）
     "warnings": [str, ...],             # 异常（NaN/缺失/非法）**与**未过的原因
     "conf_weight": float,               # §10.3 软权重（见下），**不参与** AND
     "conf_warp_monotonic": bool|None, "conf_warp_spearman": float|None,
     "n_pairs": int, "gate_version": str}

    `conf_weight`（§10.3/硬要求 4）：
    `conf_warp_monotonic=True → 1.0`（conf 可作软权重）；`False 或 None → TH_CONF_DOWNWEIGHT`
    （**降权，不否决**：不改变 `main_gate_passed`）；**未传该键 → 1.0 且不告警**
    （本次没做自检 ≠ 自检没过）。任何 conf 硬阈值都不存在于本模块。
    """
    warnings: list[str] = []
    if isinstance(quality_inputs, dict):
        qi: dict = dict(quality_inputs)
    else:
        qi = {}
        warnings.append(f"quality_inputs 不是 dict（收到 {type(quality_inputs).__name__}）"
                        "→ 全部子项 fail-closed")
    th, th_warns = _merge_thresholds(default_thresholds(), thresholds)
    warnings.extend(th_warns)

    # ---- 子项 1：跨视图 warp（§10.1-1）----
    warp_val = NaN
    photo_val = NaN
    n_pairs = 0
    per_pair: Optional[list] = None
    if _has_input(qi, "warp_inlier_ratio"):
        warp_val = _to_float(qi.get("warp_inlier_ratio"))
        photo_val = _to_float(qi.get("warp_photometric_inlier_ratio"))
        np_pairs = _to_float(qi.get("n_pairs", 0.0))
        n_pairs = int(np_pairs) if np.isfinite(np_pairs) else 0
        if isinstance(qi.get("per_pair"), list):
            per_pair = qi["per_pair"]
    elif _has_raw_arrays(qi, ("frames", "depth_maps", "c2w_list", "intrinsics")):
        try:
            w = cross_view_warp_inliers(qi["frames"], qi["depth_maps"],
                                        qi["c2w_list"], qi["intrinsics"])
            warp_val = float(w["warp_inlier_ratio"])
            photo_val = float(w["warp_photometric_inlier_ratio"])
            n_pairs = int(w["n_pairs"])
            per_pair = w["per_pair"]
        except Exception as exc:  # noqa: BLE001 - 坏几何输入 → fail-closed（不崩 episode）
            warnings.append(f"cross_view_warp_inliers 失败（{type(exc).__name__}: {exc}）"
                            "→ warp 子项 fail-closed")
    else:
        warnings.append("缺少 warp_inlier_ratio（也未给全 frames/depth_maps/c2w_list/intrinsics）"
                        "→ warp 子项 fail-closed（不得单挑放行）")

    # ---- 子项 2：分组点云重叠（§10.1-2）----
    cloud_val = NaN
    if _has_input(qi, "cloud_overlap_ratio"):
        cloud_val = _to_float(qi.get("cloud_overlap_ratio"))
    elif qi.get("point_map") is not None:
        try:
            cloud_val = float(grouped_cloud_overlap(qi["point_map"])["cloud_overlap_ratio"])
        except Exception as exc:  # noqa: BLE001 - 坏点图 → fail-closed（不崩 episode）
            warnings.append(f"grouped_cloud_overlap 失败（{type(exc).__name__}: {exc}）"
                            "→ 重叠率子项 fail-closed")
    else:
        warnings.append("缺少 cloud_overlap_ratio（也未给 point_map）"
                        "→ 重叠率子项 fail-closed（不得单挑放行）")

    sub: dict[str, bool] = {}
    for name, val in (("warp_inlier_ratio", warp_val), ("cloud_overlap_ratio", cloud_val)):
        if not np.isfinite(val):
            sub[name] = False
            warnings.append(f"{name} 非有限（NaN/Inf）或缺失 → 子项 False（fail-closed；"
                            "未知不等于通过）")
            continue
        sub[name] = bool(val >= th[name])
        if not sub[name]:
            warnings.append(f"{name}={val:.4f} < 阈值 {th[name]:.4f} → 子项未过")

    # ---- §10.3 conf 软权重（只降权，不否决）----
    mono_src = qi.get("conf_warp_monotonic", None)
    mono: Optional[bool]
    if "conf_warp_monotonic" not in qi:
        mono, conf_weight = None, 1.0          # 本次没做自检 → 不加罚、也不放大
    elif isinstance(mono_src, (bool, np.bool_)):
        mono = bool(mono_src)
        if mono:
            conf_weight = 1.0                  # 自检通过 → conf 可作软权重
        else:
            conf_weight = float(TH_CONF_DOWNWEIGHT)
            warnings.append("conf_warp_monotonic=False（conf 与 warp 残差不单调）→ conf 只降权"
                            f"（weight={conf_weight}），不得当硬门、不得否决主门（§10.3）")
    elif mono_src is None:
        mono, conf_weight = None, float(TH_CONF_DOWNWEIGHT)
        warnings.append("conf_warp_monotonic=None（自检证据不足/退化）→ 按降权处理"
                        f"（weight={conf_weight}），不否决主门（§10.3）")
    else:
        mono, conf_weight = None, float(TH_CONF_DOWNWEIGHT)
        warnings.append(f"conf_warp_monotonic 取值非法（{mono_src!r}）→ 按降权处理"
                        f"（weight={conf_weight}）")
    spearman = qi.get("conf_warp_spearman", None)
    spearman_f = _to_float(spearman)
    if spearman is not None and not np.isfinite(spearman_f):
        warnings.append(f"conf_warp_spearman 非有限（{spearman!r}）→ 记 NaN（不影响主门）")

    # ---- 证据覆盖面诊断：太多配对因"可比像素不足"被跳过 → warp 比率只代表部分帧对 ----
    if per_pair:
        n_skip = sum(1 for p in per_pair if isinstance(p, dict) and p.get("skipped"))
        if n_skip * 2 > len(per_pair):
            warnings.append(
                f"warp 配对覆盖率低：{n_skip}/{len(per_pair)} 对被跳过（可比像素 < "
                f"{MIN_WARP_PAIR_SAMPLES}）→ warp 内点率只代表剩余配对（多为互视图重叠的"
                "那些），不能当作全序列证据（§10.5 诚实边界）")

    # ---- §10.2 诊断：光度内点率只告警，不作主门条件 ----
    if np.isfinite(photo_val) and photo_val < TH_PHOTO_WARN:
        warnings.append(f"warp_photometric_inlier_ratio={photo_val:.4f} < 诊断告警线 "
                        f"{TH_PHOTO_WARN:.4f}（几何过、外观不过 → 疑似光照/动态/曝光变化；"
                        "仅告警，§10.2 不改变主门）")

    passed = bool(sub["warp_inlier_ratio"] and sub["cloud_overlap_ratio"])   # 逻辑与，不得单挑
    return {
        "main_gate_passed": passed,
        "sub_results": sub,
        "values": {
            "warp_inlier_ratio": float(warp_val),
            "warp_photometric_inlier_ratio": float(photo_val),
            "cloud_overlap_ratio": float(cloud_val),
        },
        "thresholds": {k: float(v) for k, v in th.items()},
        "warnings": warnings,
        "conf_weight": float(conf_weight),
        "conf_warp_monotonic": mono,
        "conf_warp_spearman": float(spearman_f) if np.isfinite(spearman_f) else None,
        "n_pairs": int(n_pairs),
        "gate_version": str(M4_GATE_VERSION),
    }


def _has_input(qi: dict, key: str) -> bool:
    """key 存在且非 None（显式 None = "没给"，走 fail-closed 分支）。"""
    return key in qi and qi.get(key) is not None


def _has_raw_arrays(qi: dict, keys: Sequence[str]) -> bool:
    """原始数组路径：所有 key 都在且非 None（否则视为没给，不半算）。"""
    return all(_has_input(qi, k) for k in keys)
