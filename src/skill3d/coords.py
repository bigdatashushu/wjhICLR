"""§9 坐标适配层：四层坐标系统的**显式**映射（单一事实源）。

| 坐标系统 | 来源 | 说明 |
|---|---|---|
| `original` | 原始像素 | 640×480 原图像素坐标 |
| `VLM-normalized-1000` | 在线 VLM 输出的 0–1000 归一化 | 与分辨率解耦 |
| `SAM2-mask` | SAM2 输出 mask | 在**原始分辨率** |
| `VGGT-depth-grid` | VGGT 深度/点图网格 | **预处理分辨率**（如 518×392） |

跨模块契约（C-6/C-7/C-8 已修项固化为接口）：

- bbox：`VLM-normalized-1000 → original 像素`（模型在 640×480 图上返回 999 量级）；
- mask：`SAM2-mask → VGGT-depth-grid` **最近邻**缩放（布尔语义，禁用插值）；
- 光流：必须在 `VGGT-depth-grid` 上计算（C-8：在原始帧算光流再去深度网格取坐标会越界）；
- 位姿：VGGT extrinsic 是 **world→camera**，Tool 层统一暴露 **c2w**；
  世界系 = 首帧相机系（`c2w[0] = I`）。

本模块只做纯算术，不 import torch/vllm/SAM2（可 CPU 单测）。
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np

CoordFrame = Literal["original", "vlm_normalized_1000", "sam2_mask", "vggt_depth_grid"]

# 在线 VLM 的 bbox 归一化上限（C-6：Qwen3-VL 输出 0–1000 整数）
VLM_NORM_MAX: float = 1000.0


def clamp_box(box: Sequence[float], width: int, height: int) -> list[float]:
    """把 `[x0, y0, x1, y1]` 裁剪到图像范围内并保证 x0<x1、y0<y1。"""
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((max(0.0, min(x0, width - 1)), max(0.0, min(x1, width - 1))))
    y0, y1 = sorted((max(0.0, min(y0, height - 1)), max(0.0, min(y1, height - 1))))
    return [x0, y0, x1, y1]


def vlm_box_to_pixels(box: Sequence[float], width: int, height: int,
                      *, norm_max: float = VLM_NORM_MAX) -> list[float]:
    """`VLM-normalized-1000 → original 像素`（§9 / C-6）。

    判定规则（对实测行为鲁棒）：坐标全部落在 `[0, norm_max]` 且明显超出图像尺寸
    （或图像本身小于 norm_max）→ 按归一化换算；否则视为已经是像素坐标。
    换算后统一 clamp 到图像内。
    """
    vals = [float(v) for v in box]
    if len(vals) != 4:
        raise ValueError(f"bbox 必须为 4 元组, 实际 {vals}")
    looks_normalized = (max(vals) <= norm_max * 1.0005
                        and max(vals) > max(width, height))
    if looks_normalized:
        sx, sy = width / norm_max, height / norm_max
        vals = [vals[0] * sx, vals[1] * sy, vals[2] * sx, vals[3] * sy]
    return clamp_box(vals, width, height)


def resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """`SAM2-mask → VGGT-depth-grid`：**最近邻**缩放（§9 / C-7）。

    mask 是布尔语义，禁用双线性插值（会造出不存在的边界像素）。
    已同尺寸时原样返回（不复制）。
    """
    m = np.asarray(mask)
    h, w = int(shape[0]), int(shape[1])
    if m.shape[:2] == (h, w):
        return m
    import cv2

    kind = cv2.INTER_NEAREST
    out = cv2.resize(m.astype(np.uint8), (w, h), interpolation=kind)
    return out.astype(bool) if m.dtype == bool else out


def mask_to_depth_grid(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """把 mask 对齐到 depth 网格（常用组合：mask 在原始分辨率、depth 在预处理分辨率）。"""
    return resize_mask_nearest(mask, (int(depth.shape[0]), int(depth.shape[1])))


# ------------------------------------------------ 原图 → 深度网格 的仿射映射 ----
#
# 为什么需要它（v4/A-8）：BA route 必须先做**正方形预处理**（官方
# `track_predict` 断言 `height == width`），官方做法是**中心 pad 到正方形**再缩放
# （`vggt/utils/load_fn.load_and_preprocess_images_square`）。pad 引入的不是纯缩放，
# 而是 `dst = (src + pad_offset) × scale` —— 此时若仍用 `resize_mask_nearest`
# 做"纯缩放"，mask 与深度网格会错位最多约 `pad_offset` 个像素（C-7 类缺陷）。
# 故把映射显式记为数据（进 artifact），M5 按它做最近邻重采样。


def grid_transform_identity(source_hw: Sequence[int],
                            grid_hw: Sequence[int]) -> dict:
    """非 pad 路径（feed-forward 主线）的映射：纯等比缩放，pad=0。

    返回**可序列化 dict**（写进 `ReconstructionArtifact.grid_transform`）；
    不入库也能由本函数重建，是"映射即数据"的最小表示。
    """
    sh, sw = int(source_hw[0]), int(source_hw[1])
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    return {
        "source_hw": [sh, sw],
        "grid_hw": [gh, gw],
        "pad_side": int(max(sh, sw)),
        "pad_offset_xy": [0, 0],
        "scale_x": (gw / sw) if sw else 0.0,
        "scale_y": (gh / sh) if sh else 0.0,
        "padded_to_square": False,
        "method": "resize",
    }


def grid_transform_square_padded(source_hw: Sequence[int],
                                 grid_hw: Sequence[int]) -> dict:
    """正方形预处理（官方语义）的映射：中心 pad 到 `max(H,W)` 后等比缩放到网格。

    与官方 `load_and_preprocess_images_square` 逐字对齐：

    ```
    max_dim = max(w, h); left = (max_dim - w)//2; top = (max_dim - h)//2
    scale = target_size / max_dim
    dst = (src + (left, top)) * scale
    ```
    """
    sh, sw = int(source_hw[0]), int(source_hw[1])
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    max_dim = int(max(sh, sw))
    left = (max_dim - sw) // 2
    top = (max_dim - sh) // 2
    sx = (gw / max_dim) if max_dim else 0.0
    sy = (gh / max_dim) if max_dim else 0.0
    return {
        "source_hw": [sh, sw],
        "grid_hw": [gh, gw],
        "pad_side": max_dim,
        "pad_offset_xy": [left, top],
        "scale_x": sx,
        "scale_y": sy,
        "padded_to_square": True,
        "method": "center_pad_then_resize",
    }


def grid_transform_from_dict(data: Optional[dict],
                             fallback_source_hw: Optional[Sequence[int]] = None,
                             fallback_grid_hw: Optional[Sequence[int]] = None) -> dict:
    """取用 artifact 里的 `grid_transform`；缺失时退化为纯缩放映射（老 artifact 兼容）。"""
    if data:
        return {
            "source_hw": [int(data["source_hw"][0]), int(data["source_hw"][1])],
            "grid_hw": [int(data["grid_hw"][0]), int(data["grid_hw"][1])],
            "pad_side": int(data.get("pad_side", 0) or 0),
            "pad_offset_xy": [int(data.get("pad_offset_xy", [0, 0])[0]),
                              int(data.get("pad_offset_xy", [0, 0])[1])],
            "scale_x": float(data.get("scale_x", 0.0) or 0.0),
            "scale_y": float(data.get("scale_y", 0.0) or 0.0),
            "padded_to_square": bool(data.get("padded_to_square", False)),
            "method": str(data.get("method", "resize")),
        }
    if fallback_source_hw is not None and fallback_grid_hw is not None:
        return grid_transform_identity(fallback_source_hw, fallback_grid_hw)
    return grid_transform_identity((0, 0), (0, 0))


def mask_to_grid(mask: np.ndarray, transform: Optional[dict] = None,
                 *, shape: Optional[Sequence[int]] = None) -> np.ndarray:
    """按映射把 mask（原始分辨率）重采样到深度网格（最近邻，布尔语义）。

    `transform` 为 `None` 或 `padded_to_square=False` 时等价于旧的
    `resize_mask_nearest`（向后兼容，老 artifact 行为不变）；有 pad 时用
    `cv2.warpAffine` 做**精确仿射**（`dst = (src + pad) × scale`），避免错位。
    """
    t = grid_transform_from_dict(transform)
    m = np.asarray(mask)
    gh, gw = ((int(shape[0]), int(shape[1])) if shape is not None
              else (t["grid_hw"][0], t["grid_hw"][1]))
    if gh <= 0 or gw <= 0:
        return m
    if not t["padded_to_square"]:
        return resize_mask_nearest(m, (gh, gw))
    sx = t["scale_x"] or 1.0
    sy = t["scale_y"] or 1.0
    ox, oy = t["pad_offset_xy"]
    affine = np.array([[sx, 0.0, ox * sx],
                       [0.0, sy, oy * sy]], dtype=np.float64)
    import cv2

    out = cv2.warpAffine(m.astype(np.uint8), affine, (gw, gh),
                         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)
    return out.astype(bool) if m.dtype == bool else out


def box_to_grid(box: Sequence[float], transform: Optional[dict]) -> list[float]:
    """把原图像素 bbox 映射到深度网格（含 pad）。"""
    t = grid_transform_from_dict(transform)
    x0, y0, x1, y1 = (float(v) for v in box)
    if not t["padded_to_square"]:
        return [x0 * t["scale_x"], y0 * t["scale_y"],
                x1 * t["scale_x"], y1 * t["scale_y"]]
    ox, oy = t["pad_offset_xy"]
    sx, sy = t["scale_x"], t["scale_y"]
    return [(x0 + ox) * sx, (y0 + oy) * sy, (x1 + ox) * sx, (y1 + oy) * sy]


def depth_grid_shape(depth_maps: np.ndarray) -> tuple[int, int]:
    """VGGT 深度网格 (H, W)（支持 (N,H,W) / (H,W) 两种输入）。"""
    a = np.asarray(depth_maps)
    if a.ndim < 2:
        raise ValueError(f"depth_maps 维度异常: {a.shape}")
    return int(a.shape[-2]), int(a.shape[-1])


def world_to_camera(points_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """世界系 → 相机系（用 c2w 的逆；VGGT extrinsic 本身就是 world→camera）。"""
    p = np.asarray(points_world, dtype=np.float64)
    m = np.asarray(c2w, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"c2w 必须为 4x4, 实际 {m.shape}")
    w2c = np.linalg.inv(m)
    flat = p.reshape(-1, 3)
    homo = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    cam = (w2c @ homo.T).T
    return cam[:, :3].reshape(p.shape)


def camera_to_world(points_cam: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """相机系 → 世界系（c2w 直乘）。"""
    p = np.asarray(points_cam, dtype=np.float64)
    m = np.asarray(c2w, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"c2w 必须为 4x4, 实际 {m.shape}")
    flat = p.reshape(-1, 3)
    homo = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    world = (m @ homo.T).T
    return world[:, :3].reshape(p.shape)


def assert_first_frame_is_world_origin(c2w_list: np.ndarray, atol: float = 1e-5) -> None:
    """世界系约定断言：`c2w[0] == I`（首帧相机系 = 世界系，C-10 实测印证）。"""
    m = np.asarray(c2w_list, dtype=np.float64)
    if m.ndim != 3 or m.shape[-2:] != (4, 4) or len(m) < 1:
        raise ValueError(f"c2w_list 形状异常: {m.shape}")
    if not np.allclose(m[0], np.eye(4), atol=atol):
        raise ValueError(
            "c2w[0] 不是单位阵：世界系与首帧相机系不一致（§9 约定 / C-10）")


def optical_flow_on_depth_grid(frames: Sequence[np.ndarray], depth_shape: tuple[int, int],
                               *, index: int = 0) -> np.ndarray:
    """在 **VGGT-depth-grid** 上算光流（§9 / C-8）。

    C-8 的教训：在原始帧上算光流、再拿深度网格坐标去取值会越界，
    故这里先把两帧缩放到深度网格再算光流（返回 HxWx2 的 (dx, dy)）。
    """
    import cv2

    if len(frames) < 2:
        raise ValueError("光流至少需要 2 帧")
    idx = int(np.clip(index, 0, len(frames) - 2))
    h, w = int(depth_shape[0]), int(depth_shape[1])

    def _gray(img: np.ndarray) -> np.ndarray:
        a = np.asarray(img)
        if a.shape[:2] != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        return a

    g0, g1 = _gray(frames[idx]), _gray(frames[idx + 1])
    return cv2.calcOpticalFlowFarneback(
        g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
