"""M5 对象绑定/分割：SAM2 视频 mask 传播 + mask×深度反投影绑定世界点云（§4 M5）。

在线模块，严禁任何 GPT-6 相关依赖（硬约束 1）。
sam2/torch 未安装，lazy import；bind_masks_to_world 为 numpy 真实实现。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from skill3d.schemas.reconstruction import ObjectInstance

SAM2_CHECKPOINT: Optional[str] = None    # TODO_USER_INPUT: Hiera-Tiny/Base+/Large 待选
SAM2_CONFIG: Optional[str] = None        # TODO_USER_INPUT: 模型配置名


def build_video_predictor(checkpoint: Optional[str] = SAM2_CHECKPOINT,
                          config: Optional[str] = SAM2_CONFIG):
    """构建 SAM2 视频预测器（lazy import）。

    TODO: build_sam2_video_predictor 具体签名以官方 repo 为准（§4 M5 字段 5）。
    """
    from sam2.build_sam import build_sam2_video_predictor  # lazy import

    if checkpoint is None or config is None:
        raise RuntimeError("SAM2 checkpoint/config 未配置（TODO_USER_INPUT）")
    return build_sam2_video_predictor(config, checkpoint)


def track_objects(
    frames: Sequence[np.ndarray],
    box_prompts: Sequence[Sequence[float]],
    predictor=None,
) -> list[dict[int, np.ndarray]]:
    """SAM2 视频 mask 传播（§4 M5 伪代码）。

    box_prompts: 每个对象一个 [x0,y0,x1,y1] 提示框（开放词汇对象名提示
    由在线 Qwen3-VL-8B 给出，非 GPT-6）。
    返回 per-object {frame_idx: mask(H,W) bool}。
    """
    if predictor is None:
        predictor = build_video_predictor()
    state = predictor.init_state(frames)
    for obj_id, box in enumerate(box_prompts):
        predictor.add_new_points_or_box(state, box=np.asarray(box), obj_id=obj_id)
    # propagate_in_video 产出 (frame_idx, obj_ids, mask_logits)
    per_object: list[dict[int, np.ndarray]] = [dict() for _ in box_prompts]
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        for oid, logits in zip(obj_ids, mask_logits):
            mask = (np.asarray(logits) > 0.0)
            per_object[int(oid)][int(frame_idx)] = mask
    return per_object


def bind_masks_to_world(
    masks_per_object: Sequence[dict[int, np.ndarray]],
    depth_maps: np.ndarray,
    c2w_list: np.ndarray,
    intrinsics: np.ndarray,
    class_hints: Optional[Sequence[str]] = None,
    confidences: Optional[Sequence[float]] = None,
    max_points: int = 20000,
) -> list[ObjectInstance]:
    """mask × 深度反投影 → 世界坐标点云绑定（numpy 真实实现，§4 M5）。

    - masks_per_object: per-object {frame_idx: mask(H,W)}
    - depth_maps: (N,H,W) 逐帧深度（与相机坐标系对齐，正值）
    - c2w_list: (N,4,4) 世界坐标 SE(3)
    - intrinsics: (N,3,3) 或 (3,3) 相机内参
    """
    depth_maps = np.asarray(depth_maps, dtype=np.float64)
    c2w_list = np.asarray(c2w_list, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.ndim == 2:
        intrinsics = np.broadcast_to(intrinsics, (len(c2w_list), 3, 3))

    instances: list[ObjectInstance] = []
    for obj_id, masks in enumerate(masks_per_object):
        pts_world_all = []
        for frame_idx, mask in sorted(masks.items()):
            if frame_idx >= len(depth_maps):
                continue
            depth = depth_maps[frame_idx]
            K = intrinsics[min(frame_idx, len(intrinsics) - 1)]
            c2w = c2w_list[min(frame_idx, len(c2w_list) - 1)]
            m = np.asarray(mask, dtype=bool)
            if m.shape != depth.shape:
                continue  # mask/深度分辨率不一致，跳过该帧（TODO: 上采样对齐）
            vs, us = np.nonzero(m)
            z = depth[vs, us]
            valid = np.isfinite(z) & (z > 0)
            vs, us, z = vs[valid], us[valid], z[valid]
            if z.size == 0:
                continue
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            # 相机坐标系反投影
            x = (us - cx) * z / fx
            y = (vs - cy) * z / fy
            pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)  # (4,M)
            pts_world = (c2w @ pts_cam)[:3].T                        # (M,3)
            pts_world_all.append(pts_world)

        if not pts_world_all:
            # 分割失败 → 该对象标记 unverified（§4 M5 字段 9）
            instances.append(ObjectInstance(
                instance_id=f"obj_{obj_id}",
                class_hint=class_hints[obj_id] if class_hints else "unverified",
                mask_per_frame="",
                pointcloud_world="",
                centroid_world=[0.0, 0.0, 0.0],
                bbox=[0.0] * 6,
                confidence=0.0,
            ))
            continue

        pts = np.concatenate(pts_world_all, axis=0)
        if len(pts) > max_points:
            sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
            pts = pts[sel]
        centroid = pts.mean(axis=0)
        pmin, pmax = pts.min(axis=0), pts.max(axis=0)
        # 置信度：优先用外部（SAM2 打分），否则按有效帧占比估计
        if confidences is not None and obj_id < len(confidences):
            conf = float(confidences[obj_id])
        else:
            conf = min(1.0, len(masks) / max(1, len(depth_maps)))

        instances.append(ObjectInstance(
            instance_id=f"obj_{obj_id}",
            class_hint=class_hints[obj_id] if class_hints else "unknown",
            mask_per_frame="",       # TODO: mask 数组落盘后填 ref
            pointcloud_world="",     # TODO: 点云落盘后填 ref
            centroid_world=centroid.tolist(),
            bbox=(list(pmin) + list(pmax)),
            confidence=conf,
        ))
    return instances
