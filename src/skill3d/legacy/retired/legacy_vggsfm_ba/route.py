"""M3 BA 两条 route 的封装（§3 M3 / §10.1，A-8 [Conditional Go]）。

```
                      ┌──────────────────────────────┐
VGGT 32 帧 ──────────►│ route A（默认主线）           │ feed-forward：pose_enc → c2w/K
                      │  recon_method="vggt"          │ depth/point_map/point_conf
                      └──────────────────────────────┘
                      ┌──────────────────────────────┐
                      │ route B（[Conditional Go]）   │ + VGGSfM tracker（aliked+sp，
                      │  recon_method="vggt_ba"       │   DINO 选 query 帧）→ tracks
                      └──────────────────────────────┘   → batch_np_matrix_to_pycolmap
                                                          → pycolmap.bundle_adjustment
```

**PoC 门槛（§10.1，未通过前不得当作已验证事实）**：`hasattr(pycolmap,
"bundle_adjustment")` 且两参签名成立 → 正方形预处理跑通 → 32 帧不 OOM → BA 收敛
cost 下降 → `g5_reproj_err_median` 非 NaN。

**关键纪律**：

- 主线是 feed-forward；BA 是**可选 route**，失败/跳过一律回退 feed-forward，
  `g5_reproj_err_median/p95` 标 None（**不得**标成"BA 结果"），改用
  `depth_conf/point_conf` 作弱代理；
- BA **不能恢复度量尺度**（gauge 自由度）：G-11 度量锚定必须独立；
- `min_inlier_per_frame=64`（任一帧 inlier < 64 → **整单跳过 BA**）、
  `max_reproj_error=8.0`、`camera_type=SIMPLE_PINHOLE`（无径向畸变，官方 TODO）
  全部 `TODO_CALIBRATE`；
- 正方形预处理是硬前提：官方 `track_predict._forward_on_query` 里有
  `assert height == width`（`vggt/dependency/track_predict.py:182`），
  非正方形输入必失败，故 BA route 只在 `square_preprocessed=True` 时执行。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from skill3d.coords import (
    grid_transform_identity,
    grid_transform_square_padded,
)

# ---- 参数（全部 TODO_CALIBRATE，起始参考值见 §3 M3 / §10.1）----
MIN_INLIER_PER_FRAME: int = 64        # TODO_CALIBRATE: 任一帧 inlier < 该值 → 整单跳过 BA
MAX_REPROJ_ERROR: float = 8.0         # TODO_CALIBRATE: batch_np_matrix_to_pycolmap 的过滤阈值
CAMERA_TYPE: str = "SIMPLE_PINHOLE"   # 官方 TODO：不建模径向畸变
KEYPOINT_EXTRACTOR: str = "aliked+sp"  # VGGSfM tracker 主干（官方 demo 默认）
QUERY_FRAME_NUM: int = 5              # DINO 选 query 帧数量（官方默认）
MAX_QUERY_PTS: int = 2048             # 官方默认
VGGT_FIXED_RESOLUTION: int = 518      # VGGT 前向分辨率（正方形）
IMG_LOAD_RESOLUTION: int = 1024       # BA 用的图像装载分辨率（官方 demo）

# route 名（与 ReconstructionArtifact.recon_method 受控枚举对齐）
ROUTE_FEEDFORWARD = "vggt"
ROUTE_BA = "vggt_ba"


class BAUnavailable(RuntimeError):
    """BA route 不可用（pycolmap 缺失/签名不符/tracker 依赖缺失）。"""


@dataclass
class BAProbe:
    """PoC 门槛探测结果（§10.1 通过/否决标准的可判定部分）。"""

    available: bool
    reason: str
    pycolmap_version: str = ""
    has_bundle_adjustment: bool = False
    two_arg_signature: bool = False


@dataclass
class BARouteResult:
    """BA route 执行结果（失败/跳过时 `applied=False`，preds 保持 feed-forward）。"""

    preds: dict
    recon_method: str = ROUTE_FEEDFORWARD
    applied: bool = False
    skip_reason: str = ""
    g5_reproj_err_median: Optional[float] = None
    g5_reproj_err_p95: Optional[float] = None
    weak_proxy_g5: Optional[float] = None   # depth_conf/point_conf 弱代理（非 BA 残差）
    weak_proxy_name: str = ""
    n_points3D: int = 0                     # BA 后的稀疏点数（收敛性审计）
    ba_converged: bool = False              # 是否拿到可用的精化位姿读回
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.applied:
            return (f"BA route 生效 method={self.recon_method} "
                    f"g5_median={self.g5_reproj_err_median}")
        return (f"BA route 未生效（{self.skip_reason or '未启用'}）→ 回退 feed-forward；"
                f"G5 记 None（{'弱代理=' + self.weak_proxy_name if self.weak_proxy_name else '无弱代理'}）")


# ------------------------------------------------------------ PoC 门槛探测 ----

def probe_pycolmap() -> BAProbe:
    """`hasattr(pycolmap, "bundle_adjustment")` + 两参签名检查（§10.1 否决标准）。"""
    try:
        import pycolmap  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return BAProbe(False, f"pycolmap 不可用（{type(exc).__name__}: {exc}）")

    version = str(getattr(pycolmap, "__version__", "") or "unknown")
    has_fn = hasattr(pycolmap, "bundle_adjustment")
    two_arg = False
    if has_fn:
        try:
            sig = inspect.signature(pycolmap.bundle_adjustment)
            # 官方两参形式：bundle_adjustment(reconstruction, options)
            two_arg = len([p for p in sig.parameters.values()
                           if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= 2
        except (TypeError, ValueError):
            two_arg = True  # 无法内省（C 扩展）→ 以运行时调用为准
    if not has_fn:
        return BAProbe(False, f"pycolmap {version} 无 bundle_adjustment", version, False, False)
    if not two_arg:
        return BAProbe(False, f"pycolmap {version} bundle_adjustment 签名不是两参",
                       version, True, False)
    return BAProbe(True, f"pycolmap {version} bundle_adjustment 两参签名可用",
                   version, True, True)


def probe_tracker() -> tuple[bool, str]:
    """VGGSfM tracker / DINO / 特征提取器依赖与**权重**是否就位（§3 M3 [待实码核验]）。

    已核验（2026-09-20，本机）：`vggt.dependency` 随源码自带，但**权重需下载**——
    `vggsfm_utils.build_vggsfm_tracker` 从 HF `facebook/VGGSfM` 拉
    `vggsfm_v2_tracker.pt`，`generate_rank_by_dino` 用 `torch.hub` 拉 DINOv2，
    `initialize_feature_extractors` 拉 lightglue 的 ALIKED/SuperPoint。
    故本探测**逐个检查权重文件**，缺失即给出精确下载路径（PoC 前置）。
    """
    try:
        from vggt.dependency.track_predict import predict_tracks  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"vggt.dependency.track_predict 不可导入（{type(exc).__name__}: {exc}）"
    try:
        from vggt.dependency import vggsfm_tracker  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"VGGSfM tracker 不可导入（{type(exc).__name__}: {exc}）"

    missing = missing_tracker_weights()
    if missing:
        return False, ("tracker 权重缺失（需先下载，见 ba_route.TRACKER_WEIGHT_HINTS）："
                       + "；".join(missing))
    return True, "VGGSfM tracker 依赖 + 权重就位"


def _torch_hub_dir() -> Path:
    import os

    return Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch"))) \
        / "hub" / "checkpoints"


def missing_tracker_weights() -> list[str]:
    """列出 BA route 需要的、当前缺失的权重（空列表 = 齐备）。

    检查项（与官方 `vggsfm_utils` 的加载点一一对应）：

    1. VGGSfM tracker：`~/.cache/torch/hub/checkpoints/vggsfm_v2_tracker.pt`
       （官方从 HF `facebook/VGGSfM` 下载；本机可经 hf-mirror 取得）
    2. DINOv2 主干：`~/.cache/torch/hub/facebookresearch_dinov2_main` + 权重
       `dinov2_vitb14_reg4_pretrain.pth`（官方走 `torch.hub.load`）
    """
    hub = _torch_hub_dir()
    missing: list[str] = []
    if not (hub / "vggsfm_v2_tracker.pt").is_file():
        missing.append("vggsfm_v2_tracker.pt")
    repo = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
    if not repo.is_dir():
        missing.append("facebookresearch_dinov2_main（torch.hub 仓库）")
    dino = hub / "dinov2_vitb14_reg4_pretrain.pth"
    if not dino.is_file():
        dino_alt = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / \
            "dinov2_vitb14_reg4_pth"
        if not dino_alt.is_file():
            missing.append("dinov2_vitb14_reg4_pretrain.pth")
    return missing


# 权重下载指引（本机实测：hf-mirror.com 可达，huggingface.co 不可达）
TRACKER_WEIGHT_HINTS: tuple[str, ...] = (
    "① tracker：curl -L -o ~/.cache/torch/hub/checkpoints/vggsfm_v2_tracker.pt "
    "https://hf-mirror.com/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt",
    "② DINOv2 仓库：python -c \"import torch;"
    "torch.hub.load('facebookresearch/dinov2','dinov2_vitb14_reg')\"",
    "③ DINOv2 权重：curl -L -o "
    "~/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth "
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
)


def preflight_ba() -> dict:
    """§10.1 PoC 门槛的前置清单（可打印的 receipt：缺什么、怎么补）。"""
    pyc = probe_pycolmap()
    trk_ok, trk_why = probe_tracker()
    missing = missing_tracker_weights()
    return {
        "pycolmap_ok": pyc.available,
        "pycolmap_reason": pyc.reason,
        "pycolmap_version": pyc.pycolmap_version,
        "bundle_adjustment_2arg": pyc.two_arg_signature,
        "tracker_ok": trk_ok,
        "tracker_reason": trk_why,
        "missing_weights": missing,
        "download_hints": list(TRACKER_WEIGHT_HINTS) if missing else [],
        "ready": bool(pyc.available and trk_ok),
    }


def ba_route_available() -> BAProbe:
    """BA route 是否可用（pycolmap PoC + tracker 依赖与权重），不可用给出明确原因。"""
    probe = probe_pycolmap()
    if not probe.available:
        return probe
    ok, why = probe_tracker()
    if not ok:
        return BAProbe(False, why, probe.pycolmap_version, True, True)
    return probe


# ------------------------------------------------------------ 预处理与前置条件 ----

def square_preprocess_wh(height: int, width: int) -> tuple[int, int]:
    """正方形预处理后的 (H, W)：官方 `load_and_preprocess_images_square` 语义（取长边）。

    `track_predict._forward_on_query` 有 `assert height == width`，
    非正方形输入必失败——这是 BA route 的硬前提（§3 M3）。
    """
    side = int(max(int(height), int(width)))
    return side, side


def square_preprocess_frames(paths: Sequence[str], *,
                             resolution: int = VGGT_FIXED_RESOLUTION,
                             load_resolution: int = IMG_LOAD_RESOLUTION):
    """**真正执行**正方形预处理（v4/A-8 修复点）：返回 `(images, grid_transform)`。

    官方做法（`vggt/utils/load_fn.load_and_preprocess_images_square`）：
    中心 pad 到 `max(H,W)` 的黑边正方形 → 缩放到 `resolution`。**不是拉伸**，
    因此内参无需按轴各向异性缩放（几何不变形）。

    返回的 `grid_transform` 是 §9 需要的"原图→深度网格"仿射映射
    （`dst = (src + pad_offset) × scale`），随 artifact 落盘供 M5 对齐 mask。

    优先用官方实现（已核验、随源码自带）；不可导入时用几何逐字等价的 numpy 兜底。
    """
    paths = [str(p) for p in paths]
    try:
        import torch  # noqa: F401

        from vggt.utils.load_fn import load_and_preprocess_images_square

        images, original_coords = load_and_preprocess_images_square(
            paths, load_resolution)
        if int(images.shape[-1]) != int(resolution):
            import torch.nn.functional as F

            images = F.interpolate(images, size=(int(resolution), int(resolution)),
                                   mode="bilinear", align_corners=False)
        # original_coords: (N,6) = [x1,y1,x2,y2,width,height]（**已在 target 空间**）
        oc = np.asarray(original_coords.detach().cpu().numpy() if hasattr(
            original_coords, "detach") else original_coords, dtype=np.float64)
        src_hw = (int(oc[0][5]), int(oc[0][4]))       # (H, W) 原图
        tf = grid_transform_square_padded(src_hw, (int(resolution), int(resolution)))
        return images, tf
    except Exception as exc:  # noqa: BLE001 - 无 torch/PIL：走等价 numpy 实现
        images, tf, why = _square_preprocess_numpy(paths, resolution, load_resolution)
        _ = exc
        return images, {**tf, "fallback_reason": str(exc)[:200]}


def _square_preprocess_numpy(paths: Sequence[str], resolution: int,
                             load_resolution: int):
    """正方形预处理的 numpy/cv2 等价实现（与官方几何逐字一致；用于兜底与单测）。

    仅返回 **uint8 HWC RGB 列表**（不引入 torch），调用方按需转张量。
    """
    import cv2

    frames: list[np.ndarray] = []
    src_hw: Optional[tuple[int, int]] = None
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise BAUnavailable(f"无法解码帧: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if src_hw is None:
            src_hw = (h, w)
        max_dim = max(h, w)
        top = (max_dim - h) // 2
        left = (max_dim - w) // 2
        canvas = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)   # 黑边（官方语义）
        canvas[top:top + h, left:left + w] = img
        interp = cv2.INTER_AREA if max_dim > resolution else cv2.INTER_CUBIC
        frames.append(cv2.resize(canvas, (int(resolution), int(resolution)),
                                 interpolation=interp))
    tf = grid_transform_square_padded(src_hw or (resolution, resolution),
                                      (int(resolution), int(resolution)))
    return frames, tf, "numpy 兜底实现"


def is_square(shape: Sequence[int]) -> bool:
    """图像/置信图是否已是正方形（BA route 前提）。"""
    if len(shape) < 2:
        return False
    return int(shape[-2]) == int(shape[-1])


def run_inlier_policy(
    frame_inliers: Optional[Sequence[int]],
    *,
    min_inlier_per_frame: int = MIN_INLIER_PER_FRAME,
) -> Optional[str]:
    """`min_inlier_per_frame` 策略：任一帧 inlier < 64 → 返回跳过原因（§10.1）。

    返回 None 表示可继续 BA。`frame_inliers` 为 None（无 track 统计）→ 不判死。
    """
    if frame_inliers is None:
        return None
    bad = [i for i, n in enumerate(frame_inliers) if int(n) < int(min_inlier_per_frame)]
    if bad:
        return (f"帧内点不足：{len(bad)}/{len(frame_inliers)} 帧 inlier < "
                f"{min_inlier_per_frame}（帧号 {bad[:5]}{'…' if len(bad) > 5 else ''}）")
    return None


def weak_g5_proxy(depth_conf: Optional[np.ndarray],
                  point_conf: Optional[np.ndarray]) -> tuple[Optional[float], str]:
    """未跑 BA 时的弱代理指标（**不是**重投影残差，只在附录/归因里用）。

    取 `point_conf`（优先）或 `depth_conf` 的置信度中位数：越高越可信。
    返回 `(值, 名称)`。
    """
    for arr, name in ((point_conf, "point_conf_median"), (depth_conf, "depth_conf_median")):
        if arr is None:
            continue
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size:
            return float(np.median(a)), name
    return None, ""


# ------------------------------------------------------------ 主流程 ----

def run_ba_route(
    preds: dict,
    images: Any,
    out_dir: str | Path,
    scene_name: str,
    *,
    enabled: bool = False,
    square_preprocessed: bool = False,
    shared_camera: bool = False,
    max_query_pts: int = MAX_QUERY_PTS,
    query_frame_num: int = QUERY_FRAME_NUM,
) -> BARouteResult:
    """执行 BA route（[Conditional Go]）；任何前置条件不满足 → 明确跳过并回退。

    成功时的产物：`preds["reproj_errors"]`（逐观测重投影残差）+ artifact 的
    `g5_reproj_err_median/p95`。失败时 **不写** 这些字段（保持 None），
    改记 `weak_proxy_g5`。
    """
    res = BARouteResult(preds=dict(preds))
    depth_conf = preds.get("depth_conf")
    point_conf = preds.get("point_conf")
    res.weak_proxy_g5, res.weak_proxy_name = weak_g5_proxy(depth_conf, point_conf)

    if not enabled:
        res.skip_reason = "未启用（默认主线 = VGGT feed-forward）"
        return res

    # 正方形硬前提（**先于**环境探测：这是输入属性，报错更可操作且与环境无关）。
    # 非正方形输入在官方 tracker 内会 assert 失败，这里提前拦截。
    img_shape = tuple(getattr(images, "shape", ()) or ())
    if len(img_shape) >= 2 and not is_square(img_shape):
        h, w = img_shape[-2], img_shape[-1]
        if not square_preprocessed:
            res.skip_reason = (
                f"非正方形输入（H={h}, W={w}）需先做正方形预处理"
                f"（官方 track_predict 断言 height==width；目标 "
                f"{square_preprocess_wh(h, w)}）")
        else:
            # square_preprocessed=True 却拿到非正方形 → 上游预处理没真正执行（A-8 原缺陷）
            res.skip_reason = (f"已声明正方形预处理但输入仍非正方形（shape={img_shape}）"
                               "——预处理未真正执行")
        return res

    probe = ba_route_available()
    if not probe.available:
        res.skip_reason = probe.reason
        res.notes.append("§10.1 否决标准命中 → BA route 不执行"
                         "（preflight_ba() 给出缺失清单与下载指引）")
        return res

    try:
        import pycolmap  # type: ignore
        import torch  # type: ignore

        from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
        from vggt.dependency.track_predict import predict_tracks
    except Exception as exc:  # noqa: BLE001
        res.skip_reason = f"BA 依赖不可导入（{type(exc).__name__}: {exc}）"
        return res

    # 生产者：把 BA 需要的键补齐（w2c / 原始内参 / BA 帧系点云）
    inputs = build_ba_inputs(preds, images)
    points_3d = inputs.get("point_map_in_ba_frame")
    extri = inputs.get("extrinsic")
    intri_raw = inputs.get("intrinsic_raw")
    if points_3d is None or extri is None or intri_raw is None:
        res.skip_reason = ("缺 BA 输入（需 point_map_in_ba_frame / extrinsic / "
                           "intrinsic_raw；生产者见 build_ba_inputs）")
        return res

    try:
        with torch.no_grad():
            pred_tracks, pred_vis, pred_confs, points_3d, points_rgb = predict_tracks(
                images, conf=inputs.get("depth_conf"), points_3d=points_3d, masks=None,
                max_query_pts=max_query_pts, query_frame_num=query_frame_num,
                keypoint_extractor=KEYPOINT_EXTRACTOR,
            )
        track_mask = np.asarray(pred_vis) > 0.2
        frame_inliers = [int(np.sum(track_mask[i])) for i in range(track_mask.shape[0])]
        skip = run_inlier_policy(frame_inliers)
        if skip:
            res.skip_reason = skip
            return res

        image_size = np.array(images.shape[-2:])
        reconstruction, _valid = batch_np_matrix_to_pycolmap(
            np.asarray(points_3d, dtype=np.float64), np.asarray(extri, dtype=np.float64),
            np.asarray(intri_raw, dtype=np.float64), pred_tracks,
            image_size, masks=track_mask, max_reproj_error=MAX_REPROJ_ERROR,
            shared_camera=shared_camera, camera_type=CAMERA_TYPE,
            min_inlier_per_frame=MIN_INLIER_PER_FRAME,
            points_rgb=points_rgb,
        )
        if reconstruction is None:
            res.skip_reason = ("batch_np_matrix_to_pycolmap 未能构建 reconstruction"
                               f"（min_inlier_per_frame={MIN_INLIER_PER_FRAME} 未过）")
            return res

        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)  # 官方两参形式
    except Exception as exc:  # noqa: BLE001 - OOM/收敛/版本差异都走回退
        res.skip_reason = f"BA 执行失败（{type(exc).__name__}: {exc}）"
        return res

    errs = _reproj_errors_after_ba(reconstruction)
    if errs is None or errs.size == 0:
        res.skip_reason = "BA 后无有效观测残差（判为未收敛/模型空）"
        return res

    # 读回精化位姿/内参 + 用精化位姿重新反投影稠密点云
    refined = refined_arrays_from_reconstruction(
        reconstruction, preds.get("depth_map"), image_size)
    res.notes.extend(refined.notes)
    if refined.c2w is not None:
        res.preds["c2w_refined"] = refined.c2w
        res.preds["intrinsic_refined"] = refined.intrinsics
        if refined.point_map is not None:
            res.preds["point_map_refined"] = refined.point_map
        res.preds["reconstruction"] = reconstruction
        res.n_points3D = refined.n_points3D
        res.ba_converged = True
    else:
        # 读回失败：仍然算 BA 成功（残差可得），但位姿保持 feed-forward 并显式标注
        res.notes.append("BA 残差可得但精化位姿读回失败 → 位姿保持 feed-forward"
                         "（不冒充精化结果）")

    res.preds["reproj_errors"] = errs
    res.applied = True
    res.recon_method = ROUTE_BA
    res.g5_reproj_err_median = float(np.median(errs))
    res.g5_reproj_err_p95 = float(np.percentile(errs, 95))
    res.notes.append(f"BA 收敛后重投影残差 median={res.g5_reproj_err_median:.3f}px "
                     f"(n={errs.size}，points3D={res.n_points3D})；"
                     "BA 不恢复度量尺度（gauge 自由度）")
    return res


def build_ba_inputs(preds: dict, images: Any) -> dict:
    """生产 BA 所需的键（v4/A-8 修复点 2/3）：`extrinsic` / `intrinsic_raw` /
    `point_map_in_ba_frame`。

    官方 `batch_np_matrix_to_pycolmap` 要的是 **world→camera 的 (N,3,4)**
    （`extrinsics[fidx][:3, :3]` / `[:, 3]` 直接当旋转+平移用），而 artifact 侧统一
    暴露 c2w（硬约束/术语表：`extrinsic` 是 w2c，需取逆才是 c2w）。本函数把两者都备好。

    `point_map_in_ba_frame` 用**官方** `unproject_depth_map_to_point_map`
    （`depth × extrinsic × intrinsic`）重算：官方 demo 就是这么给 `predict_tracks`
    喂 `points_3d` 的；与 VGGT 自带的 `world_points` 数值等价，但保证与
    `extrinsic/intrinsic_raw` 同一套约定（自洽优先）。
    """
    out: dict = {}
    # ---- ① w2c (N,3,4)：由 artifact 的 c2w(4×4) 取逆得到，不依赖 VGGT 原始输出 ----
    c2w = preds.get("c2w")
    extri = preds.get("extrinsic")
    if extri is None and c2w is not None:
        w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
        extri = w2c[:, :3, :4]
    if extri is not None:
        out["extrinsic"] = np.asarray(extri, dtype=np.float64)[:, :3, :4]
    # ---- ② 内参（BA 用的"原始"内参，与深度网格同分辨率）----
    k = preds.get("intrinsic_raw", preds.get("intrinsic"))
    if k is not None:
        out["intrinsic_raw"] = np.asarray(k, dtype=np.float64)
    # ---- ③ BA 帧系下的 3D 点 ----
    pts = preds.get("point_map_in_ba_frame")
    if pts is None:
        depth = preds.get("depth_map")
        if depth is not None and out.get("extrinsic") is not None \
                and out.get("intrinsic_raw") is not None:
            try:
                from vggt.utils.geometry import unproject_depth_map_to_point_map

                d = np.asarray(depth, dtype=np.float64)
                pts = unproject_depth_map_to_point_map(
                    d[..., None] if d.ndim == 3 else d,
                    out["extrinsic"], out["intrinsic_raw"])
            except Exception:  # noqa: BLE001 - 兜底用 VGGT 自带 world_points
                pts = preds.get("point_map")
        else:
            pts = preds.get("point_map")
    if pts is not None:
        out["point_map_in_ba_frame"] = np.asarray(pts, dtype=np.float64)
    if "point_conf" in preds:
        out["point_conf"] = preds["point_conf"]
    if "depth_conf" in preds:
        out["depth_conf"] = preds["depth_conf"]
    _ = images
    return out


@dataclass
class RefinedBAArrays:
    """BA 后的精化产物（读回 + 稠密点云重投影）。"""

    c2w: Optional[np.ndarray] = None            # (N,4,4) 精化后的 camera→world
    intrinsics: Optional[np.ndarray] = None     # (N,3,3) 精化后的 K（与深度网格同分辨率）
    point_map: Optional[np.ndarray] = None      # (N,H,W,3) 用精化位姿重新反投影的稠密点云
    n_points3D: int = 0
    notes: list[str] = field(default_factory=list)


def refined_arrays_from_reconstruction(reconstruction: Any,
                                       depth_maps: Optional[np.ndarray],
                                       image_size: Sequence[int],
                                       ) -> RefinedBAArrays:
    """从 BA 后的 sparse 模型读回精化位姿/内参，并用精化位姿**重新反投影稠密点云**。

    为什么要重投影点云（而不是拿 BA 的稀疏点）：M5 对象绑定与几何 Tool 需要稠密点云；
    而 BA 只精化了相机与**被 track 到的**稀疏点。正确做法是"位姿用精化的、深度用 VGGT 的"，
    故用精化后的 `extrinsic/intrinsic` 对**原始深度图**做 `unproject`（官方函数），
    得到与精化位姿自洽的稠密世界点云 —— 这正是架构里 `point_map_in_ba_frame`
    "在正方形预处理后重算 unproject_depth_map_to_point_map"的语义。

    读回顺序按官方 `batch_np_matrix_to_pycolmap` 的命名约定（`image_{fidx+1}`、`camera_id=fidx+1`），
    与 M1 的帧序**一一对应**（不重排帧，硬约束 21）。
    """
    out = RefinedBAArrays()
    try:
        from vggt.dependency.np_to_pycolmap import pycolmap_to_batch_np_matrix

        pts3d, extri, intri, _extra = pycolmap_to_batch_np_matrix(
            reconstruction, device="cpu", camera_type=CAMERA_TYPE)
    except Exception as exc:  # noqa: BLE001 - 读回 API 差异 → 由调用方回退 feed-forward
        out.notes.append(f"BA 读回失败（{type(exc).__name__}: {exc}）")
        return out
    out.n_points3D = int(len(pts3d))
    if extri is None or intri is None or len(extri) == 0:
        out.notes.append("BA 读回为空（模型无注册影像）")
        return out

    w2c = np.asarray(extri, dtype=np.float64)
    if w2c.shape[-2:] == (3, 4):
        h = np.tile(np.eye(4), (len(w2c), 1, 1))
        h[:, :3, :4] = w2c
        w2c = h
    out.c2w = np.linalg.inv(w2c)
    out.intrinsics = np.asarray(intri, dtype=np.float64)

    if depth_maps is not None:
        try:
            from vggt.utils.geometry import unproject_depth_map_to_point_map

            d = np.asarray(depth_maps, dtype=np.float64)
            out.point_map = unproject_depth_map_to_point_map(
                d[..., None] if d.ndim == 3 else d,
                np.asarray(extri, dtype=np.float64)[:, :3, :4], out.intrinsics)
            out.notes.append(
                f"BA 后稠密点云已用精化位姿重投影（grid={tuple(int(v) for v in image_size)}）")
        except Exception as exc:  # noqa: BLE001 - 重投影失败：保留精化位姿，点云留空
            out.notes.append(f"BA 后点云重投影失败（{type(exc).__name__}: {exc}）→ 保留 feed-forward 点云")
    return out


def _reproj_errors_after_ba(reconstruction: Any) -> Optional[np.ndarray]:
    """BA 后逐观测重投影残差（px）；pycolmap API 差异时返回 None。"""
    try:
        from skill3d.reconstruction.ba import reproj_errors_from_model

        return reproj_errors_from_model(reconstruction)
    except Exception:  # noqa: BLE001
        return None
