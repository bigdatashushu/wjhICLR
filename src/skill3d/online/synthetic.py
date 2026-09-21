"""mock_light 合成输入（§9.2 Tool 三档 Mock 的 mock_light 语义）。

**用途限定**：M0/M1 验收与集成测试的管道验证（`online/runner.py` 在
`mode="mock_light"` 时调用）。合成数据**绝不可用于论文实验结论**；
真实实验一律 `mode="real"`（准入门强制 real，§5.6b）。

合成内容全部由 seed 决定 → 同 seed 字节级一致（§4 M8/M17 验收）。
几何是真算的，且**与 M4 主门同一套约定**（v6 §10，见下），因此下游 M4 / M7 /
M9 / M10 / M11 在合成场景上执行的是**真实计算**（M4 主门、MetricEvidenceGate、
Tool 契约与执行都在真跑），只是数据是合成的。

## 合成 bundle 的约定（v6：必须与 `reconstruction_gate/m4_main_gate.py` 逐字一致）

`m4_main_gate` 的口径（不得改门，只能让合成数据满足它）：

- 深度网格像素 `(u, v)`（**整数索引**，不是像素中心 +0.5）反投影为相机系点
  `p_cam = ((u−cx)/fx·z, (v−cy)/fy·z, z)`，其中 `z` 是该像素的**相机系 z 深度**；
- 世界系 = 首帧相机系：`p_world = R_i·p_cam + t_i`（`c2w_i = [R_i|t_i]`）；
- 投回第 j 帧：`p_cam_j = R_jᵀ·(p_world − t_j)`，`u_j = fx_j·x/z_j + cx_j`（再四舍五入）；
- 分组点云重叠率用**相对**阈值（场景稳健尺度的百分比），与绝对尺度无关。

本模块按同一套公式**正着构造**（`_depth_grid` 射线-盒求交 → `_point_map` 反投影），
所以 `point_map` 与 `depth_maps`/`c2w`/`K` 逐像素自洽：干净合成场景的主门必然通过
（`warp_inlier_ratio ≈ 1`、重叠率远高于阈），而**任何一处被破坏**（位姿被洗牌、
深度被扰动、K 换了网格口径）都会让主门 fail —— 这正是 mock 路径作为
"真实烟囱测试"而非"绕过门"的意义（见 `tests/unit/test_synthetic_geometry_gate.py`）。

深度网格 `96×128`、内参 `fx=fy=100`（等效全分辨率 480×640 @ f=500）与
`K` 的像素索引口径：`cx=(W−1)/2`、`cy=(H−1)/2` —— 与门里
`u = fx·x/z + cx` 的取整口径同源，两者不允许各自"再解释"一遍。

合成 GT 与合成场景同源构造。`route_planning` 的 GT 仍不携带真实语义（占位字母，
见 `_question_and_gt`）—— 故合成集上的 accuracy/MRA **不构成任何精度结论**。
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from skill3d.gates import iqa
from skill3d.reconstruction.metric_fusion import write_per_frame_receipt
from skill3d.reconstruction_gate.confidence_map import fuse_confidence, coverage_ratio
from skill3d.reconstruction_gate.evidence_profile import (
    M5EvidenceSummary,
    build_evidence_profile,
    evaluate_metric_gate,
)
from skill3d.reconstruction_gate.quality_metrics import compute_quality
from skill3d.reconstruction_gate.scene_state import scene_route_from_quality
from skill3d.reconstruction_gate.world_frame import estimate_world_frame
from skill3d.routing.task_classifier import canonical_task
from skill3d.schemas import (
    GATE_VERSION,
    InputFrame,
    MetricEvidenceGateResult,
    ObjectInstance,
    QualityMetrics,
    SceneState,
    VSIBenchEpisode,
)
from skill3d.tools.contract import available_artifacts_for, question_tool_scope_of
from skill3d.tools.scene_handle import SceneHandle

# 与 VSI-Bench 视频规格对齐（§1.2：640×480）
FRAME_H, FRAME_W = 480, 640
N_FRAMES = 32

# 合成房间盒（米，z 向上）：5.0 m × 6.0 m × 2.6 m
ROOM_MIN = np.array([-2.5, -3.0, 0.0])
ROOM_MAX = np.array([2.5, 3.0, 2.6])

# 相机内参（**全分辨率帧像素**；深度网格内参由 `_intrinsics()` 按网格等比缩放）
FOCAL = 500.0
# 深度图网格（VGGT 深度是预处理分辨率下的网格；这里取 1/5 边长，省内存且与门兼容）
DEPTH_H, DEPTH_W = 96, 128

# 相机轨迹：缓慢前进 + 左右扫视（手持室内巡游；见 `_camera_trajectory`）
CAMERA_HEIGHT = 1.5
CAM_YAW_SWEEP_DEG = 28.0
CAM_TRAVEL_Y = 0.8
CAM_BOB = 0.03

@dataclass
class SyntheticGeometry:
    """合成场景几何（可注入 SceneHandle 的世界系数据）。

    所有数组都是**按 §10 门口径真算**出来的，互相自洽：

    - `c2w`（N,4,4）：世界系 = 首帧相机系（`c2w[0] = I`）；
    - `intrinsics`（3,3）：**深度网格**像素单位（与 `depth_maps` 同网格）；
    - `depth_maps`（N,h,w）：相机系 z 深度（米）；
    - `point_map`（N,h,w,3）：由 `depth`/`K`/`c2w` 反投影得到的精确世界点；
    - `depth_conf`（N,h,w）：VGGT 口径置信度（有限且 > 0；§10.3 只作软权重）；
    - `point_conf` / `coverage_count`：逐点软权重与观测帧数。
    """

    scene_name: str
    c2w: np.ndarray                      # (N,4,4) SE(3)
    intrinsics: np.ndarray               # (3,3) 深度网格内参
    depth_maps: np.ndarray               # (N,h,w) 沿相机 z 的深度（米）
    point_map: np.ndarray                # (N,h,w,3) 世界系点（由 depth 精确反投影）
    depth_conf: np.ndarray               # (N,h,w) 置信度（有限、正）
    point_conf: np.ndarray               # (N,h,w) 逐点置信度
    coverage_count: np.ndarray           # (N,h,w) 逐点观测帧数
    objects: list[ObjectInstance] = field(default_factory=list)
    # 合成对象的世界系点云（obj_id → (n,3)），直接注入 SceneHandle（不依赖磁盘 ref）
    object_points: dict[str, np.ndarray] = field(default_factory=dict)
    metric_scale: float = 1.0
    world_up: Optional[list[float]] = None
    handedness: Optional[str] = None
    world_frame_status: str = "unavailable"
    per_frame_scale_ref: Optional[str] = None


@dataclass
class SyntheticEpisode:
    """合成 episode：帧元数据 + 真实像素 + 场景几何 + 合成 GT。"""

    episode: VSIBenchEpisode
    frames: list[np.ndarray]              # (N,H,W,3) uint8 RGB
    geometry: SyntheticGeometry


@dataclass
class SyntheticArtifactRef:
    """合成 bundle 的**只读证据载体**（duck-typed，供 EvidenceProfile / SceneHandle 读取）。

    **它不是 `ReconstructionArtifact`**（§5.2：`recon_method` 是受控枚举，合成产物
    不得冒用 `vggt` 名义），也绝不落盘成 artifact JSON：只承载"合成场景按构造已知"
    的事实（世界系契约、米制尺度、质量），使 mock 路径的 EvidenceProfile 与 Tool
    可见性走**同一套生产判定**（而不是"没有画像 → 全部放行/全部收回"）。
    """

    scene_name: str
    quality: Optional[QualityMetrics]
    quality_status: str
    recon_method: str = "mock_light_synthetic"
    world_up: Optional[list[float]] = None
    handedness: Optional[str] = None
    world_frame_status: str = "unavailable"
    metric_scale: Optional[float] = None
    scale_self_consistency: Optional[float] = None
    scale_fusion_status: str = "not_run"
    per_frame_scale_ref: Optional[str] = None
    metric_model: str = "none"
    metric_fusion_version: str = ""
    frame_set_hash: str = ""


# ---------------------------------------------------------------- 几何合成 ----

def _intrinsics() -> np.ndarray:
    """深度网格内参（**像素索引口径**：`u = fx·x/z + cx`，与 M4 门的投影同源）。

    主点取 `(W−1)/2`（像素索引的中心），焦距按网格与全分辨率帧的边长比缩放：
    `fx = FOCAL · W/FRAME_W`、`fy = FOCAL · H/FRAME_H`（本设置下都等于 100）。
    """
    fx = FOCAL * (DEPTH_W / FRAME_W)
    fy = FOCAL * (DEPTH_H / FRAME_H)
    return np.array([[fx, 0.0, (DEPTH_W - 1) / 2.0],
                     [0.0, fy, (DEPTH_H - 1) / 2.0],
                     [0.0, 0.0, 1.0]])


def _pixel_rays() -> np.ndarray:
    """深度网格逐像素的相机系射线方向 `(h,w,3)`，z 分量为 1（口径唯一事实源）。

    **与 `m4_main_gate._pair_warp_stats` 的反投影逐字一致**：
    `x = (u−cx)/fx·z`、`y = (v−cy)/fy·z`（`u,v` 为整数像素索引）。
    """
    K = _intrinsics()
    u = np.arange(DEPTH_W, dtype=np.float64)
    v = np.arange(DEPTH_H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    return np.stack([(uu - K[0, 2]) / K[0, 0],
                     (vv - K[1, 2]) / K[1, 1],
                     np.ones_like(uu)], axis=-1)


def _rotation_yaw(yaw: float) -> np.ndarray:
    """房间系 z 向上的偏航旋转矩阵（列 = 相机轴在世界系的表示）。

    相机约定为 OpenCV（`x=右, y=下, z=前`，右手系 det=+1）：
    `forward = (sin yaw, cos yaw, 0)`、`down = (0,0,−1)`、`right = down × forward`。
    于是 `−R[:,1] = (0,0,1)` 就是"图像上方"——与 `world_frame.estimate_world_frame`
    的口径一致（它用 `-R_f[:,1]` 推 `world_up`）。
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0.0, s],
                     [-s, 0.0, c],
                     [0.0, -1.0, 0.0]])


def _room_frame_poses(n_frames: int, seed: int) -> np.ndarray:
    """房间系里的手持巡游轨迹（慢速前进 + 左右扫视 + 轻微起伏），(N,4,4) c2w。"""
    t = np.linspace(0.0, 1.0, n_frames)
    jitter = 0.2 * np.sin(2.0 * np.pi * t + 0.3 * seed)
    y_room = (-1.9 + CAM_TRAVEL_Y * t) + 0.1 * jitter
    x_room = 0.15 * np.sin(np.pi * t + 0.7 * seed)
    z_room = CAMERA_HEIGHT + CAM_BOB * np.sin(4.0 * np.pi * t)
    yaw = np.radians(CAM_YAW_SWEEP_DEG) * (2.0 * t - 1.0) + np.radians(2.0) * jitter
    # pitch 只做小抖动（大俯仰会把"竖直"约束压薄：world_frame 估计会如实降级）
    pitch = np.radians(3.0) * np.sin(2.0 * np.pi * t + seed)
    poses = []
    for i in range(n_frames):
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(pitch[i]), -np.sin(pitch[i])],
                       [0.0, np.sin(pitch[i]), np.cos(pitch[i])]])
        R = _rotation_yaw(float(yaw[i])) @ Rx
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = np.array([x_room[i], y_room[i], z_room[i]])
        poses.append(c2w)
    return np.stack(poses)


def _camera_trajectory(n_frames: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """轨迹搬到 **首帧相机系**（硬要求：`c2w[0] = I`，与 VGGT 主线口径一致）。

    返回 `(c2w_world, M)`，`M = inv(pose_room[0])` 是"房间系 → 世界系"的刚体变换
    （合成对象也用它搬到同一世界系）。
    """
    poses_room = _room_frame_poses(n_frames, seed)
    M = np.linalg.inv(poses_room[0])
    return np.stack([M @ p for p in poses_room]), M


def _box_hit_t(o: np.ndarray, d_unit: np.ndarray, lo: np.ndarray,
               hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """射线-轴对齐盒 slab 求交：返回 `(t_min, t_max, inside)`（相机在盒内 → inside）。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o) / d_unit
        t2 = (hi - o) / d_unit
    t_min = np.nanmax(np.minimum(t1, t2), axis=-1)
    t_max = np.nanmin(np.maximum(t1, t2), axis=-1)
    inside = t_min <= 0.0
    return t_min, t_max, inside


def _depth_grid(poses_room: np.ndarray,
                boxes: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """逐帧**相机系 z 深度**（米）：逐像素取与房间盒/对象盒的最近交点。

    相机在房间盒内 → 取**出射**交点；相机在对象盒外 → 取出射前的入口交点。
    深度 = `t · (R[:,2]·d_world)`（把世界距离投影到相机前向轴），与 `_point_map`
    的反投影公式互为逆运算（VGGT 口径的 z 深度，不是斜距）。
    """
    d_cam = _pixel_rays()
    out = np.empty((len(poses_room), DEPTH_H, DEPTH_W), dtype=np.float32)
    room_lo, room_hi = boxes[0]
    for i, pose in enumerate(poses_room):
        R, o = pose[:3, :3], pose[:3, 3]
        d_world = d_cam @ R.T
        d_world = d_world / np.linalg.norm(d_world, axis=-1, keepdims=True)
        # 房间盒：相机在盒内 → 取出射交点
        t_min, t_max, inside = _box_hit_t(o, d_world, room_lo, room_hi)
        t_hit = np.where(inside, t_max, t_min)
        valid = (t_max >= np.maximum(t_min, 0.0)) & (t_hit > 0.0)
        # 对象盒：相机恒在盒外 → 取入口交点（更近者胜）
        for lo, hi in boxes[1:]:
            o_min, o_max, _ = _box_hit_t(o, d_world, lo, hi)
            o_hit = o_min
            o_ok = (o_max >= np.maximum(o_min, 0.0)) & (o_hit > 0.0)
            closer = o_ok & ((~valid) | (o_hit < t_hit))
            t_hit = np.where(closer, o_hit, t_hit)
            valid = valid | o_ok
        depth = np.where(valid, t_hit, np.nan) * (d_world @ R[:, 2])
        out[i] = depth.astype(np.float32)
    return out


def _point_map(depth: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """世界系点图 = `depth` 经 `K` 反投影再经 `c2w` 变换（**与门同一公式**）。

    `p_cam = ((u−cx)/fx·z, (v−cy)/fy·z, z)`；`p_world = R_i·p_cam + t_i`。
    """
    K = _intrinsics()
    u = np.arange(DEPTH_W, dtype=np.float64)
    v = np.arange(DEPTH_H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    out = np.empty((len(c2w), DEPTH_H, DEPTH_W, 3), dtype=np.float32)
    for i in range(len(c2w)):
        z = depth[i].astype(np.float64)
        x = (uu - K[0, 2]) / K[0, 0] * z
        y = (vv - K[1, 2]) / K[1, 1] * z
        p_flat = np.stack([x, y, z], axis=-1)
        out[i] = (p_flat @ c2w[i, :3, :3].T + c2w[i, :3, 3]).astype(np.float32)
    return out


def _depth_confidence(depth: np.ndarray) -> np.ndarray:
    """VGGT 口径的 depth 置信度（有限且 **> 0**；§10.3：只作软权重，不作硬门）。

    构造：`C = exp(Σ)+1 ≥ 1`（与 VGGT depth 头同量纲），近处/画面中心（更接近正视
    的像素）更高、掠射边缘更低 —— 这样如果真跑 conf-warp 单调性自检，conf 与 warp
    残差的关系也是单调的（不是常数伪证）。
    """
    us = (np.arange(DEPTH_W) + 0.5) / DEPTH_W
    vs = (np.arange(DEPTH_H) + 0.5) / DEPTH_H
    uu, vv = np.meshgrid(us, vs)
    rr = np.sqrt((uu - 0.5) ** 2 + (vv - 0.5) ** 2) / np.sqrt(0.5)
    with np.errstate(invalid="ignore"):
        z = np.asarray(depth, dtype=np.float64)
        z_norm = np.clip(np.nan_to_num(z, nan=1.0) / np.nanmedian(z), 0.2, 4.0)
    conf1 = 2.0 + 6.0 * np.clip(1.0 - rr, 0.0, 1.0) - 0.8 * (z_norm - 1.0)
    conf = np.clip(conf1, 1.0001, 12.0)
    return conf.astype(np.float32)


def _per_point_stats(n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """逐点置信度与观测帧数：中心高、边缘低（用于触发置信度门控的真实融合）。"""
    us = (np.arange(DEPTH_W) + 0.5) / DEPTH_W
    vs = (np.arange(DEPTH_H) + 0.5) / DEPTH_H
    uu, vv = np.meshgrid(us, vs)
    rr = np.sqrt((uu - 0.5) ** 2 + (vv - 0.5) ** 2) / np.sqrt(0.5)
    conf = np.clip(1.1 - 0.7 * rr, 0.05, 1.0).astype(np.float32)
    cov = np.where(rr < 0.85, 3.0, 1.0).astype(np.float32)
    conf = np.repeat(conf[None, ...], n_frames, axis=0)
    cov = np.repeat(cov[None, ...], n_frames, axis=0)
    return conf, cov


def _stable_seed(text: str, mod: int) -> int:
    """跨进程稳定的确定性整数（不用内置 hash：字符串 hash 有 PYTHONHASHSEED 随机化）。"""
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) % mod


# 合成对象（靠墙/贴地摆放；全部落在轨迹的可见锥内，见模块 docstring）
_SYNTH_OBJECTS: list[tuple[str, list[float], list[float]]] = [
    # class_hint, center, size(xyz)
    ("sofa", [1.70, 1.70, 0.375], [1.60, 0.80, 0.75]),
    ("table", [-1.77, 1.77, 0.35], [1.30, 0.80, 0.70]),
    ("chair", [-0.57, 2.13, 0.45], [0.60, 0.60, 0.90]),
    ("cabinet", [0.41, 1.55, 0.90], [0.80, 0.50, 1.80]),
]


def _box_of(center: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(size, dtype=np.float64) / 2.0
    c = np.asarray(center, dtype=np.float64)
    return c - half, c + half


def _bbox_of(lo: np.ndarray, hi: np.ndarray) -> list[float]:
    return [float(lo[i]) for i in range(3)] + [float(hi[i]) for i in range(3)]


def _world_bbox(corners: np.ndarray) -> list[float]:
    """世界系**轴对齐**包围盒（盒在世界系里一般不轴对齐，必须由 8 角点重算）。"""
    return _bbox_of(np.min(corners, axis=0), np.max(corners, axis=0))


def _box_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array([[lo[0] if not (i & 1) else hi[0],
                      lo[1] if not (i & 2) else hi[1],
                      lo[2] if not (i & 4) else hi[2]] for i in range(8)],
                    dtype=np.float64)


def _surface_points(lo: np.ndarray, hi: np.ndarray, n: int,
                    rng: np.random.Generator) -> np.ndarray:
    """在盒**表面**均匀采样（面积加权）：点集即"表面样本"，符合 §12.2 的语义。

    比"体积内均匀采样"更接近真实 mask 内点云（低分位距离量的是表面）。
    """
    size = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
    areas = np.array([size[1] * size[2], size[1] * size[2],
                      size[0] * size[2], size[0] * size[2],
                      size[0] * size[1], size[0] * size[1]], dtype=np.float64)
    faces = rng.choice(6, size=int(n), p=areas / areas.sum())
    uv = rng.uniform(-0.5, 0.5, size=(int(n), 2))
    pts = np.empty((int(n), 3), dtype=np.float64)
    for f in range(6):
        m = faces == f
        if not np.any(m):
            continue
        ax = f // 2                       # 0 → x 面，1 → y 面，2 → z 面
        sgn = 0.5 if f % 2 == 0 else -0.5
        other = [a for a in range(3) if a != ax]
        local = np.empty((int(np.count_nonzero(m)), 3), dtype=np.float64)
        local[:, ax] = sgn
        local[:, other[0]] = uv[m, 0]
        local[:, other[1]] = uv[m, 1]
        pts[m] = local * size
    return pts + (np.asarray(lo, dtype=np.float64) + size / 2.0)


def _transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """刚体变换：`p_world = R·p + t`（`M = [R|t]`）。"""
    return np.asarray(pts, dtype=np.float64) @ M[:3, :3].T + M[:3, 3]


def _project_objects(objects: list[ObjectInstance], c2w: np.ndarray,
                     depth: np.ndarray, out_dir: Optional[str],
                     point_sets: dict[str, np.ndarray]) -> None:
    """把（世界系）对象点云投影进各帧，写 `visible_frames` + 逐帧 mask（有 out_dir 时落盘）。

    投影用**与门同一套** K/c2w 口径；可见 = 投影落在网格内 ∧ 该像素处相机看到的就是
    该对象（`|z_obj − depth| ≤ 容差`，即未被前景/背面遮挡）。
    """
    K = _intrinsics()
    fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]),
                      float(K[0, 2]), float(K[1, 2]))
    for obj in objects:
        pts = point_sets[obj.obj_id]
        visible: list[int] = []
        masks = np.zeros((len(c2w), DEPTH_H, DEPTH_W), dtype=np.uint8)
        tol = max(0.02 * float(np.max(np.asarray(obj.bbox[3:], dtype=np.float64)
                                      - np.asarray(obj.bbox[:3], dtype=np.float64))),
                  1e-3)
        for i in range(len(c2w)):
            R, t = c2w[i, :3, :3], c2w[i, :3, 3]
            p_cam = (pts - t) @ R
            z = p_cam[:, 2]
            ok = z > 1e-6
            if not np.any(ok):
                continue
            zz = np.where(ok, z, 1.0)
            u = fx * p_cam[:, 0] / zz + cx
            v = fy * p_cam[:, 1] / zz + cy
            inside = ok & (u >= 0) & (u <= DEPTH_W - 1) & (v >= 0) & (v <= DEPTH_H - 1)
            if not np.any(inside):
                continue
            ui = np.rint(u[inside]).astype(np.int64)
            vi = np.rint(v[inside]).astype(np.int64)
            z_in = z[inside]
            seen = np.abs(np.asarray(depth[i], dtype=np.float64)[vi, ui] - z_in) <= tol
            if np.any(seen):
                masks[i, vi[seen], ui[seen]] = 1
                visible.append(i)
        obj.visible_frames = visible
        if out_dir is not None:
            np.save(Path(out_dir) / f"{obj.category_name}_mask.npy", masks)


def _make_objects(out_dir: Optional[str], c2w: np.ndarray, depth: np.ndarray,
                  M: np.ndarray) -> tuple[list[ObjectInstance], dict[str, np.ndarray]]:
    """构造合成对象清单（自描述 ObjectRecord）：点云 + bbox + 真实可见帧 + mask。

    对象在**房间系**里定义（贴地/靠墙摆位），再用 `M`（房间系 → 世界系）搬到
    世界系 —— 世界系 = 首帧相机系，与深度图/点图同一坐标系。
    返回 `(objects, {obj_id: 世界系点云})`：点云**常驻内存**（mock 路径不必依赖
    磁盘 ref；有 out_dir 时同时落盘作为审计产物）。
    """
    objs: list[ObjectInstance] = []
    point_sets: dict[str, np.ndarray] = {}
    for hint, center, size in _SYNTH_OBJECTS:
        c_room = np.asarray(center, dtype=np.float64)
        s = np.asarray(size, dtype=np.float64)
        lo_room, hi_room = _box_of(c_room, s)
        rng = np.random.default_rng(_stable_seed(hint, 2**31))
        pts_world = _transform_points(M, _surface_points(lo_room, hi_room, 600, rng))
        world_bbox = _world_bbox(_transform_points(M, _box_corners(lo_room, hi_room)))
        centroid_world = _transform_points(M, c_room[None, :])[0]
        obj_id = f"{hint}-0"
        point_sets[obj_id] = pts_world
        pc_ref, mask_ref = "", ""
        if out_dir is not None:
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{hint}_points.npy"
            np.save(p, pts_world)
            pc_ref = str(p)
            mask_ref = str(d / f"{hint}_mask.npy")
        objs.append(
            ObjectInstance(          # v6 §5.6：`ObjectRecord`（自描述 category_name）
                obj_id=obj_id,
                category_name=hint,
                mask_per_frame=mask_ref,
                pointcloud_world=pc_ref,
                centroid_world=[float(x) for x in centroid_world],
                bbox=world_bbox,
                track_id=f"track-{hint}-0",   # 合成 track：每个对象一条独立 track
                det_conf=0.9,
                visible_frames=[],
            )
        )
    _project_objects(objs, c2w, depth, out_dir, point_sets)
    return objs, point_sets


# ---------------------------------------------------------------- 帧合成 ----

_DEGRADE_MODES = ("blur_all", "blur_some", "overexposed_all", "few_frames")


def _synth_frame(idx: int, n_frames: int, rng: np.random.Generator,
                 h: int, w: int) -> np.ndarray:
    """单帧合成 RGB：平滑渐变底 + 若干锐利矩形（提供可计算的清晰度/曝光）。"""
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    img = 70.0 + 55.0 * yy + np.zeros((h, w, 3), dtype=np.float32)
    # 随相机前进缓慢平移的矩形（帧间小位移 → 低运动量）
    shift = int(round((idx / max(n_frames - 1, 1)) * w * 0.06))
    rects = [
        (0.10 * w + shift, 0.55 * h, 0.18 * w, 0.30 * h, 175.0),
        (0.45 * w + shift, 0.35 * h, 0.12 * w, 0.45 * h, 120.0),
        (0.70 * w + shift, 0.60 * h, 0.16 * w, 0.25 * h, 200.0),
    ]
    for cx, cy, rw, rh, val in rects:
        x0 = int(np.clip(cx, 0, w - 2))
        x1 = int(np.clip(cx + rw, x0 + 1, w))
        y0 = int(np.clip(cy, 0, h - 2))
        y1 = int(np.clip(cy + rh, y0 + 1, h))
        img[y0:y1, x0:x1, :] = val
    # 细纹理块（4px 棋盘）：提供足够的清晰度（G1 通过），且在重度模糊后消失
    cell = 4
    ty0, ty1 = int(0.12 * h), int(0.36 * h)
    tx0, tx1 = int(0.05 * w), int(0.35 * w)
    ys, xs = np.mgrid[ty0:ty1, tx0:tx1]
    checker = (((ys // cell) + (xs // cell)) % 2) * 2.0 - 1.0
    img[ty0:ty1, tx0:tx1, :] += (checker * 45.0)[..., None]
    img += rng.normal(0.0, 2.0, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def _apply_degrade(frames: list[np.ndarray], mode: str) -> list[np.ndarray]:
    import cv2

    if mode == "blur_all":
        return [cv2.GaussianBlur(f, (31, 31), 0) for f in frames]
    if mode == "blur_some":
        n = len(frames)
        out, k = [], (31, 31)
        for i, f in enumerate(frames):
            # 前 40% 帧模糊（> TH_DEGRADED_RATIO=0.25 → locally_degraded）
            out.append(cv2.GaussianBlur(f, k, 0) if i < int(0.4 * n) else f)
        return out
    if mode == "overexposed_all":
        out = []
        for f in frames:
            g = f.copy()
            g[: int(0.6 * g.shape[0]), :, :] = 255  # 60% 像素过曝
            out.append(g)
        return out
    raise ValueError(f"未知 degradation: {mode}")


# ---------------------------------------------------------------- episode ----

def _object_names(geometry: SyntheticGeometry) -> list[str]:
    return [o.category_name for o in geometry.objects]


def _first_visible_frame(obj: ObjectInstance) -> Optional[int]:
    """对象最早出现的帧（无可见帧 → None，不伪造帧号）。"""
    return int(min(obj.visible_frames)) if obj.visible_frames else None


def _independent_direction(observer: np.ndarray, facing_at: np.ndarray,
                           target: np.ndarray, world_up=(0.0, 0.0, 1.0),
                           handedness: str = "right") -> str:
    """独立复算方位（**不调用 Tool**），规则与 §9.8 一致（45° 半角 + 叉积符号）：

    `f = facing_at − observer`、`d = target − observer`，先投影到垂直于 `world_up`
    的水平面；`|cos| ≥ cos45°` → front/behind；否则 `right ⟺ (f×d)·u < 0`。
    """
    u = np.asarray(world_up, dtype=np.float64)
    u = u / np.linalg.norm(u)
    f = np.asarray(facing_at, dtype=np.float64) - np.asarray(observer, dtype=np.float64)
    d = np.asarray(target, dtype=np.float64) - np.asarray(observer, dtype=np.float64)
    f_h = f - float(np.dot(f, u)) * u
    d_h = d - float(np.dot(d, u)) * u
    f_h = f_h / np.linalg.norm(f_h)
    d_h = d_h / np.linalg.norm(d_h)
    cos_t = float(np.dot(f_h, d_h))
    if cos_t >= np.cos(np.radians(45.0)):
        return "front"
    if cos_t <= -np.cos(np.radians(45.0)):
        return "behind"
    sign = float(np.dot(np.cross(f_h, d_h), u))
    if handedness != "right":
        sign = -sign
    return "right" if sign < 0.0 else "left"


def _question_and_gt(question_type: str, geometry: SyntheticGeometry,
                     rng: np.random.Generator) -> tuple[str, list[str] | None, str]:
    """合成 question / options / ground_truth（题型为 §4 M7 的 8 个规范题型）。

    MCA/NA 分流与官方 meta 一致（§1.2：4 MCA + 4 NA）：
    `object_counting`/`object_abs_distance`/`object_size_estimation`/
    `room_size_estimation` 为数值题（无 options）；其余四类为选择题。

    GT 与工具口径对齐（否则"mock 答错"无法区分是管道坏了还是口径不同）：

    - `object_abs_distance`：§12.3 的官方口径是**相机→对象**（不是对象↔对象），
      GT 用相机中心到对象表面点集的低分位距离；
    - `object_rel_distance`：观察点（相机）到各候选对象的稳健距离比较；
    - `object_rel_direction`：观察者 = cabinet、面向 chair、指向 sofa（三对象口径，
      与 `relative_direction_of(observer, facing_at, target)` 同构）；
    - `obj_appearance_order`：按对象**最早可见帧**排序（来自真实投影可见性）。

    `route_planning` 仍是**非信息性占位**（合成集没有可走的路线语义，GT 为占位字母）——
    故合成集上的 accuracy/MRA **不构成任何精度结论**。

    另外：合成相机不做俯视（pitch 只在 0 附近抖动），地面只在中距离之外可见 →
    `room_size_estimation` 的合成 MRA 只反映"可见地面那一块"，**不是**房间拟合能力的
    结论（真实测评一律走 `mode="real"`）。
    """
    names = _object_names(geometry)
    if question_type == "object_counting":
        n = len(names)
        return ("How many objects are in the scene? Count them.", None, str(n))
    if question_type == "object_rel_direction":
        options = ["left", "right", "front", "behind"]
        direction = _independent_direction(
            geometry.objects[3].centroid_world, geometry.objects[2].centroid_world,
            geometry.objects[0].centroid_world,
            world_up=geometry.world_up or (0.0, 0.0, 1.0),
            handedness=geometry.handedness or "right")
        gt = "ABCD"[options.index(direction)]
        return (f"Standing at the {names[3]} and facing the {names[2]}, in which "
                f"direction is the {names[0]} (left / right / front / behind)?",
                options, gt)
    if question_type == "route_planning":
        # 非信息性占位（见 docstring）：合成集没有路线语义 → 不参与任何精度结论
        options = ["A", "B", "C", "D"]
        return ("Which route leads to the sofa?", options,
                "ABCD"[int(rng.integers(0, 4))])
    if question_type == "obj_appearance_order":
        pair = (geometry.objects[0], geometry.objects[1])
        firsts = [_first_visible_frame(o) for o in pair]
        options = [pair[0].category_name, pair[1].category_name]
        if firsts[0] is None and firsts[1] is None:
            gt = "A"
        elif firsts[0] is None:
            gt = "B"
        elif firsts[1] is None:
            gt = "A"
        else:
            gt = "A" if firsts[0] <= firsts[1] else "B"
        return (f"As the camera moves through the room, which object appears first "
                f"in the video: the {options[0]} or the {options[1]}?", options, gt)
    if question_type == "room_size_estimation":
        gt = float(np.prod(ROOM_MAX[:2] - ROOM_MIN[:2]))
        return ("What is the area of the room in square meters? (floor plan area)",
                None, f"{gt:.2f}")
    if question_type == "object_size_estimation":
        gt = float(max(_SYNTH_OBJECTS[0][2]))
        return (f"What is the longest dimension of the {names[0]} in meters?",
                None, f"{gt:.2f}")
    if question_type == "object_abs_distance":
        pts = _load_object_points(geometry, geometry.objects[0])
        cam = np.asarray(geometry.c2w[0][:3, 3], dtype=np.float64)
        gt = float(np.quantile(np.linalg.norm(pts - cam, axis=1), 0.01))
        return (f"In the first frame, how far is the camera from the {names[0]}? "
                f"Answer in meters.", None, f"{gt:.2f}")
    if question_type == "object_rel_distance":
        # MCA：观察点（相机）比较 objects[0] / objects[2] 谁更近（选项 = 对象名）
        cam = np.asarray(geometry.c2w[0][:3, 3], dtype=np.float64)
        d0 = float(np.quantile(np.linalg.norm(
            _load_object_points(geometry, geometry.objects[0]) - cam, axis=1), 0.01))
        d2 = float(np.quantile(np.linalg.norm(
            _load_object_points(geometry, geometry.objects[2]) - cam, axis=1), 0.01))
        options = [names[0], names[2]]
        gt = "A" if d0 <= d2 else "B"
        return (f"Between the {names[0]} and the {names[2]}, which is closer to the "
                f"camera (the observer) in the first frame?", options, gt)
    raise ValueError(f"未支持的合成题型: {question_type}")


def _load_object_points(geometry: SyntheticGeometry, obj: ObjectInstance) -> np.ndarray:
    """取对象的世界系点云（常驻内存）；缺失时退回 bbox 角点，绝不编造点数。"""
    pts = geometry.object_points.get(obj.obj_id)
    if pts is not None and np.asarray(pts).size:
        return np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    bbox = np.asarray(obj.bbox, dtype=np.float64).reshape(2, 3)
    return np.array([bbox[0], bbox[1],
                     [bbox[0, 0], bbox[1, 1], bbox[0, 2]],
                     [bbox[1, 0], bbox[0, 1], bbox[1, 2]],
                     np.asarray(obj.centroid_world, dtype=np.float64)])


def make_synthetic_episode(
    question_type: str,
    *,
    scene_name: str = "synthetic-scene",
    qa_id: str = "synth-0001",
    split: str = "inner_validation",
    seed: int = 0,
    n_frames: int = N_FRAMES,
    frame_size: tuple[int, int] = (FRAME_H, FRAME_W),
    degrade: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> SyntheticEpisode:
    """构造一个确定性的合成 episode（帧 + 几何 + 合成 GT）。"""
    question_type = canonical_task(question_type)  # 官方取值/别名 → 规范题型
    if degrade is not None and degrade not in _DEGRADE_MODES:
        raise ValueError(f"未知 degradation: {degrade}（可选 {_DEGRADE_MODES}）")

    rng = np.random.default_rng(seed)
    h, w = frame_size
    poses_room = _room_frame_poses(n_frames, seed)
    c2w, M = _camera_trajectory(n_frames, seed)
    # 场景几何在**房间系**里做射线-盒求交（房间盒 + 对象盒 → 深度图含家具，
    # 遮挡关系真实），深度是相机系 z 深度，与所选世界系无关
    boxes = [(np.asarray(ROOM_MIN, dtype=np.float64), np.asarray(ROOM_MAX, dtype=np.float64))]
    boxes += [_box_of(np.asarray(c, dtype=np.float64), np.asarray(s, dtype=np.float64))
              for _hint, c, s in _SYNTH_OBJECTS]
    depth = _depth_grid(poses_room, boxes)
    objects, object_points = _make_objects(out_dir, c2w, depth, M)
    wf = estimate_world_frame(c2w)      # M3 生产估计器真跑（不猜竖直方向）
    geometry = SyntheticGeometry(
        scene_name=scene_name,
        c2w=c2w,
        intrinsics=_intrinsics(),
        depth_maps=depth,
        point_map=_point_map(depth, c2w),
        depth_conf=_depth_confidence(depth),
        point_conf=np.zeros((n_frames, DEPTH_H, DEPTH_W), dtype=np.float32),
        coverage_count=np.zeros((n_frames, DEPTH_H, DEPTH_W), dtype=np.float32),
        objects=objects,
        object_points=object_points,
        world_up=wf.world_up,
        handedness=wf.handedness,
        world_frame_status=wf.status,
    )
    geometry.point_conf, geometry.coverage_count = _per_point_stats(n_frames)
    # 逐帧尺度 receipt（生产核 `fuse_metric_scale` 真跑的产物）：M4 之后的
    # `scene_state.make_metric_gate` 会按 §13.1 子条件 3 读它取 `valid_frame_ratio`，
    # 缺 receipt 就等于"融合覆盖不可证明" → 米制 Tool 被收回（与真实路径同一判据）。
    receipt_dir = (Path(out_dir) if out_dir is not None else
                   Path(tempfile.gettempdir()) / "skill3d_mock_light" / scene_name)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = write_per_frame_receipt(
        _synthetic_metric_fusion(geometry),
        receipt_dir / f"{scene_name}_per_frame_scale.json")
    geometry.per_frame_scale_ref = str(receipt_path)

    pixels = [_synth_frame(i, n_frames, rng, h, w) for i in range(n_frames)]
    if degrade == "few_frames":
        pixels = pixels[: n_frames // 2]
    elif degrade is not None:
        pixels = _apply_degrade(pixels, degrade)

    question, options, gt = _question_and_gt(question_type, geometry, rng)

    frames = []
    for i, img in enumerate(pixels):
        blur = iqa.laplacian_var(img)
        p_over, p_under = iqa.exposure_ratios(img)
        frames.append(
            InputFrame(
                frame_idx=i,
                timestamp=i / 30.0,
                width=w,
                height=h,
                blur_var=blur,
                overexposed_ratio=p_over,
                underexposed_ratio=p_under,
                quality_ok=not (blur < 100.0 or p_over > 0.05 or p_under > 0.05),
            )
        )

    # 统一固定 FrameSet（硬约束 21）：mock_light 也走同一条帧集契约，
    # 帧集哈希随 episode 落盘，保证消融/重放可对齐
    from skill3d.adapters.frame_set import frame_set_hash as _fsh

    ids = list(range(len(pixels)))
    fset = {
        "frame_ids": ids,
        "source_frame_indices": ids,
        "timestamps": [i / 30.0 for i in ids],
        "frame_set_hash": _fsh(ids),
        "n_frames": len(pixels),
        "n_total_frames": n_frames,
        "fps": 30.0,
    }
    episode = VSIBenchEpisode(
        qa_id=qa_id,
        scene_name=scene_name,
        dataset="synthetic",  # 合成数据自我标注，绝不与 scannet/arkitscenes 混淆
        question_type=question_type,
        question=question,
        options=options,
        ground_truth=gt,
        frames=frames,
        split=split,  # type: ignore[arg-type]
        frame_set=fset,
    )
    return SyntheticEpisode(episode=episode, frames=pixels, geometry=geometry)


# ------------------------------------------------------- SceneState / 句柄 ----

def synthetic_scale_known() -> bool:
    """合成场景的尺度**按构造即为真值**（房间/物体尺寸与相机高都是米制设计值）。

    v6 §20：v5 的"相对 CI 半宽 ≤ TH_SCALE_CI_REL"判据随校准路线整体废止（v6 无 CI
    概念）；mock_light 改由 `synthetic_metric_gate` / `_synthetic_metric_fusion`
    表达米制授权（**真跑**生产融合核与 §13.1 六项子条件），且只用于管道验证。
    """
    return True


def synthetic_metric_tasks() -> set[str]:
    """mock_light 合成场景的米制题型授权（v6：由合成 MetricEvidenceGate 表达）。

    **为什么合成场景可以授权**：合成几何是**按米制构造**的（房间与物体尺寸已知），
    逐帧尺度 s_k ≡ 1.0 来自"构造真值"而非推理期融合（真实路径必须由 MoGe-2 +
    MetricEvidenceGate 决定，§13.1 / D3）。这条通道只用于**管道验证**
    （`synthesis_source=mock_stub`、`scale_source=synthetic_mock_light`），
    **绝不允许进入论文主表或真实结果**（HC24：mock 仅开发用；mock_switch 负责
    阻断 real 模式下的 mock 污染）。
    """
    return {"object_abs_distance", "object_size_estimation", "room_size_estimation"}


def _synthetic_metric_fusion(geometry: SyntheticGeometry, *,
                             conf_warp_monotonic: Optional[bool] = None):
    """真跑 `reconstruction.metric_fusion.fuse_metric_scale`（合成 bundle 版）。

    合成 bundle 的 `depth_maps` 本身就是**米制**的，且世界系按中位距离归一化因子
    取 1（`metric_scale = 1.0`）—— 于是 `s_k = median(d_metric / d_VGGT_norm) ≡ 1.0`、
    离散度 0。这不是"绕过融合"，而是把**无噪声的合成输入**喂给生产融合核：
    融合核、receipt、§13.1 子条件判定全部真跑（真实数据上才换成 MoGe-2 的输出）。
    """
    from skill3d.reconstruction.metric_fusion import (
        METRIC_MODEL_NONE,
        fuse_metric_scale,
    )

    n = len(geometry.depth_maps)
    depth_metric = [geometry.depth_maps[i] for i in range(n)]
    depth_vggt_norm = [geometry.depth_maps[i] for i in range(n)]
    # valid mask = 有限 ∧ 正深度 ∧ 置信度有限（合成场景没有黑边；不设 conf 硬门）
    valid = [np.isfinite(geometry.depth_maps[i])
             & (geometry.depth_maps[i] > 0)
             & np.isfinite(geometry.depth_conf[i]) for i in range(n)]
    return fuse_metric_scale(
        depth_metric, depth_vggt_norm, valid,
        [geometry.depth_conf[i] for i in range(n)],
        model=METRIC_MODEL_NONE,               # 合成通道没有度量模型 → 如实标 none
        conf_warp_monotonic=conf_warp_monotonic,
    )


def synthetic_metric_gate(allowed: set[str], *,
                          scene_route: str, main_gate_passed: bool,
                          fusion=None, inputs_finite: bool = True
                          ) -> MetricEvidenceGateResult:
    """mock_light 的合成 MetricEvidenceGate（走 §13.1 的**生产判定函数**）。

    六项子条件逐项有实值来源（不再是"全部填 True"）：

    1. `scene_route` —— 由 **M4 主门**真算得出（本函数只透传）；
    2. M4 主门 —— 同上（真算）；
    3. 尺度融合成功 —— `fuse_metric_scale` 在生产融合核上真跑（合成输入：逐帧
       `d_metric/d_VGGT_norm ≡ 1.0`，故 `status="success"`、`valid_frame_ratio=1.0`）；
    4. 尺度自洽 —— 同一融合结果的 `scale_self_consistency`（合成场景 = 0）；
    5. 输入有限 —— 调用方对 c2w/K/depth/point_map 做有限性检查后传入；
    6. 题型是米制 —— `allowed`（合成场景的米制题型集合，见 `synthetic_metric_tasks`）。

    `allowed` 为空 → 第 6 项不过 → `gate_passed=False`（用于验证"米制不可用 →
    米制 Tool 全收回"的路径）。合成 gate **不影响**真实链路：真实路径的
    `metric_scale` 一律由 `reconstruction_gate.evidence_profile.evaluate_metric_gate`
    按 artifact 实算（§13.8）。
    """
    if fusion is None:
        # 没有融合结果 = 没有尺度证据 → fail-closed（不猜、不假设"合成即通过"）
        return MetricEvidenceGateResult(
            gate_passed=False, gate_version=GATE_VERSION,
            sub_results={}, values={},
            missing_subconditions=["scale_fusion_success"])
    result = evaluate_metric_gate(
        scene_route=scene_route,
        main_gate_passed=bool(main_gate_passed),
        fusion_status=str(getattr(fusion, "status", "failed")),
        scale_self_consistency=getattr(fusion, "scale_self_consistency", None),
        valid_frame_ratio=float(getattr(fusion, "valid_frame_ratio", 0.0) or 0.0),
        metric_scale=getattr(fusion, "metric_scale", None),
        inputs_finite=bool(inputs_finite),
        question_type=sorted(allowed)[0] if allowed else "",
        gate_version=GATE_VERSION,
    )
    # 第 6 项子条件（题型是米制）由 `allowed` 表达：合成通道的题型是构造给定的
    if result.sub_results.get("metric_question") is False or not allowed:
        subs = dict(result.sub_results)
        subs["metric_question"] = False
        return MetricEvidenceGateResult(
            gate_passed=False, gate_version=result.gate_version,
            sub_results=subs, values=result.values,
            missing_subconditions=sorted(k for k, v in subs.items() if not v))
    return result


def _inputs_finite(geometry: SyntheticGeometry, frames: Sequence[np.ndarray]) -> bool:
    """§13.1 子条件 5：内参/位姿/深度/点图全部有限（真检查，不假定）。"""
    try:
        return bool(
            np.all(np.isfinite(geometry.c2w))
            and np.all(np.isfinite(geometry.intrinsics))
            and np.all(np.isfinite(geometry.depth_maps))
            and np.all(np.isfinite(geometry.point_map))
            and len(frames) == int(geometry.depth_maps.shape[0]))
    except Exception:  # noqa: BLE001 - 形状异常 → 判不有限（fail-closed）
        return False


def build_scene_state(
    geometry: SyntheticGeometry,
    frames: Sequence[np.ndarray],
    artifact_ref: Optional[str] = None,
    *,
    input_quality_weight: float = 1.0,
    input_degradation_flags: Optional[Sequence[str]] = None,
    metric_tasks: Optional[set[str]] = None,
) -> tuple[SceneState, SceneHandle, QualityMetrics]:
    """按 §10 主门口径在合成数据上**真实计算**质量并构造 v6 SceneState。

    数据源与真实路径同形（`compute_quality` 吃 frames/depth/c2w/K/point_map/conf），
    **不接受**任何"mock 专用放行"：主门、证据画像、米制门、Tool 可见性都走生产判定。
    `SyntheticArtifactRef` 只承载"合成场景按构造已知"的事实（见其 docstring），
    不是 `ReconstructionArtifact`、不落盘。

    `metric_tasks` 默认 `synthetic_metric_tasks()`（合成 GT 尺度，仅管道
    验证）；要验证"尺度不可用 → 米制 Tool 收回"的路径时显式传空集。
    """
    depth = geometry.depth_maps
    # 合成场景无对应关系 → G5 语义项留空（NaN）；G7/G9 需 M5 产物 → 交真实路径
    q = compute_quality(
        None,
        frames=frames,
        depth_maps=depth,
        c2w_list=geometry.c2w,
        intrinsics=geometry.intrinsics,
        point_map=geometry.point_map,
        depth_conf=geometry.depth_conf,
    )
    fused = fuse_confidence(geometry.point_conf, geometry.coverage_count)
    coverage = coverage_ratio(fused)
    # M2 被动观测（flag/weight）参与 route 判定（§4 M2 字段 6）：
    # 有效质量 = 质量诊断综合分 × 输入权重；权重不改变帧集，只影响分流
    effective_quality = q.overall_quality * float(input_quality_weight)
    q_eff = q.model_copy(update={"overall_quality": effective_quality})
    # route 只由质量决定（§6.2）；米制能力走逐题通道（v6 D3/D4）
    route = scene_route_from_quality(q_eff)
    allowed = set(metric_tasks) if metric_tasks is not None else synthetic_metric_tasks()
    fusion = _synthetic_metric_fusion(geometry)
    gate = synthetic_metric_gate(
        allowed if route == "full_3d" else set(),
        scene_route=route, main_gate_passed=bool(q.main_gate_passed),
        fusion=fusion, inputs_finite=_inputs_finite(geometry, frames))
    metric_question = bool(allowed)
    scope = question_tool_scope_of(
        route, metric_question=metric_question, gate_passed=gate.gate_passed)

    art_ref = SyntheticArtifactRef(
        scene_name=geometry.scene_name,
        quality=q_eff,
        quality_status="computed",
        world_up=geometry.world_up,
        handedness=geometry.handedness,
        world_frame_status=geometry.world_frame_status,
        metric_scale=(float(fusion.metric_scale)
                      if fusion.metric_scale is not None else geometry.metric_scale),
        scale_self_consistency=fusion.scale_self_consistency,
        scale_fusion_status=str(fusion.status),
        per_frame_scale_ref=geometry.per_frame_scale_ref,
        metric_model=str(getattr(fusion, "model", "none") or "none"),
        metric_fusion_version=str(getattr(fusion, "version", "") or ""),
    )
    profile = build_evidence_profile(
        artifact=art_ref, scene_route=route, question_type="",
        gate=gate, m5=synthetic_m5_summary(geometry),
        metric_scale_override=None)

    summary = (
        f"scene={geometry.scene_name} (mock_light 合成) scene_route={route} "
        f"question_tool_scope={scope} "
        f"main_gate_passed={bool(q.main_gate_passed)} "
        f"warp_inlier={float(q.warp_inlier_ratio):.3f} "
        f"cloud_overlap={float(q.cloud_overlap_ratio):.3f} "
        f"overall_quality={effective_quality:.3f} "
        f"(诊断综合={q.overall_quality:.3f} × M2 权重={float(input_quality_weight):.2f}) "
        f"input_flags={sorted(set(input_degradation_flags or [])) or '无'} "
        f"world_frame={geometry.world_frame_status} "
        f"metric_gate_passed={gate.gate_passed} "
        f"metric_tasks={sorted(allowed)}（合成 GT 尺度，仅管道验证，不得当结果）"
        f"coverage={coverage:.3f} objects={len(geometry.objects)}"
    )
    scene = SceneState(
        artifact_ref=artifact_ref or f"mock_light:{geometry.scene_name}",
        artifact=art_ref,
        scene_route=route,  # type: ignore[arg-type]
        question_tool_scope=scope,  # type: ignore[arg-type]
        available_artifacts=set(available_artifacts_for(
            route, gate_passed=gate.gate_passed, metric_question=metric_question)),
        evidence_profile=profile,
        metric_evidence_gate_result=gate,
        answer_source="abstain",
        objects=[o.obj_id for o in geometry.objects],
        summary=summary,
        question_type="",
        quality=q_eff,
    )
    handle = SceneHandle(
        scene,
        objects=geometry.objects,
        c2w_list=geometry.c2w,
        intrinsics=geometry.intrinsics,
        quality_overall=effective_quality,
        # 合成几何自述 depth 单位为米 → metric_scale=1.0（米制 Tool 的换算系数）
        metric_scale=geometry.metric_scale,
    )
    # 世界系点图注入句柄（平面拟合 / 连通性图的唯一数据入口，硬约束 17）
    handle.set_point_map(geometry.point_map)
    # 合成对象点云常驻内存：Tool 仍只经句柄的只读访问器取数（不接触路径）
    for oid, pts in (geometry.object_points or {}).items():
        handle.set_object_points(oid, pts)
    return scene, handle, q_eff


def synthetic_m5_summary(geometry: SyntheticGeometry) -> M5EvidenceSummary:
    """合成场景的 M5 证据输入（对象清单是**构造真值**，不是检测器产出）。

    如实标注：基础清单直接来自合成对象的构造（4 个对象、各自一条 track、覆盖全部
    可见帧），故 `object_detection` / `track_consensus` 在合成通道按 available 计入；
    它只影响 mock 管道的 Tool 可见性，**不构成任何检测/跟踪能力的证据**。
    """
    return M5EvidenceSummary(
        detection_fault=False,
        n_objects=len(geometry.objects),
        n_tracks=len(geometry.objects),
        track_stable_ratio=1.0,
        grounding_pointed_hit=None,
        grounding_conf=None,
        grounding_miss=False,
        grounding_filled=False,
        notes=["mock_light：对象清单为合成构造真值（非检测器产出），仅管道验证"],
    )


# --------------------------------------------------------- 确定性 stub 程序 ----

_STUB_BANNER = "# mock_light 确定性 stub（非 Qwen3-VL-8B 输出，仅管道验证）"


def stub_program(question_type: str, episode: VSIBenchEpisode,
                 geometry: Optional[SyntheticGeometry] = None,
                 *, object_names: Optional[Sequence[str]] = None) -> str:
    """按题型给出确定性 stub program（仅 mock_light；程序仍走 M9 AST + M10 沙箱）。

    只用**注册表里的真名**（v5 的 `room_size_m2` / `relative_direction` /
    `object_size_longest_dim` 等别名已不存在；写别名会被 M9 AST 拒绝 → episode
    白丢），并且用与合成 GT 同口径的官方原语。

    `geometry` 为 None 时用 `object_names`（来自真实 SceneHandle 的对象名）——
    使 mock_light 的确定性 stub 也能跑在**冻结的真实 artifact** 上（golden 重放、
    A/B 对照用），不必造合成几何。
    """
    if geometry is None and object_names is None:
        raise ValueError("stub_program 需要 geometry 或 object_names 之一")
    names = (_object_names(geometry) if geometry is not None
             else list(object_names or []))
    anchor_a = names[0] if names else "sofa"
    anchor_b = names[2] if len(names) > 2 else anchor_a
    anchor_c = names[3] if len(names) > 3 else anchor_b
    opts = list(episode.options or [])

    if question_type == "object_counting":
        # NA 数值题：答案是被查询类别的对象数（合成场景中即基础清单长度）
        return (
            f"{_STUB_BANNER}\n"
            f"n = len(list_objects())\n"
            f"ReturnAnswer(str(n))\n"
        )
    if question_type == "object_rel_direction":
        pairs = ", ".join(f'("{o}", "{"ABCD"[i]}")' for i, o in enumerate(opts))
        return (
            f"{_STUB_BANNER}\n"
            f"pairs = [{pairs}]\n"
            f'res = relative_direction_of("{anchor_c}", "{anchor_b}", "{anchor_a}")\n'
            f'answer = pairs[0][1]\n'
            f"for text, letter in pairs:\n"
            f'    if text == res["direction"]:\n'
            f"        answer = letter\n"
            f"ReturnAnswer(answer)\n"
        )
    if question_type == "object_rel_distance":
        # MCA：按 §12.3 官方口径比较"观察点（相机）→ 各候选对象"的稳健距离
        pairs = ", ".join(f'("{o}", "{"ABCD"[i]}")' for i, o in enumerate(opts))
        return (
            f"{_STUB_BANNER}\n"
            f"pairs = [{pairs}]\n"
            f"dists = {{}}\n"
            f"for text, letter in pairs:\n"
            f'    dists[text] = robust_distance("camera", text)["distance_normalized"]\n'
            f'nearest = min(dists, key=dists.get)\n'
            f'answer = pairs[0][1]\n'
            f"for text, letter in pairs:\n"
            f"    if text == nearest:\n"
            f"        answer = letter\n"
            f"ReturnAnswer(answer)\n"
        )
    if question_type == "route_planning":
        # 非信息性 stub：该题型合成 GT 也无真实语义（见 `_question_and_gt`）
        letter = "ABCD"[_stable_seed(episode.qa_id + question_type, 4)]
        return f"{_STUB_BANNER}\nReturnAnswer(\"{letter}\")\n"
    if question_type == "obj_appearance_order":
        # 外观顺序：按"最早可见帧"排序（`object_visible_frames` 是官方原语）
        pairs = ", ".join(f'("{o}", "{"ABCD"[i]}")' for i, o in enumerate(opts))
        return (
            f"{_STUB_BANNER}\n"
            f"pairs = [{pairs}]\n"
            f"best, answer = None, pairs[0][1]\n"
            f"for text, letter in pairs:\n"
            f"    vis = object_visible_frames(text)\n"
            f"    if not vis:\n"
            f"        continue\n"
            f"    first = min(vis)\n"
            f"    if best is None or first < best:\n"
            f"        best, answer = first, letter\n"
            f"ReturnAnswer(answer)\n"
        )
    if question_type == "room_size_estimation":
        return (
            f"{_STUB_BANNER}\n"
            f'ReturnAnswer(str(round(plane_fit_room_size()["room_area_m2"], 2)))\n'
        )
    if question_type == "object_size_estimation":
        return (
            f"{_STUB_BANNER}\n"
            f'ext = object_3d_extent("{anchor_a}")\n'
            f'ReturnAnswer(str(round(max(ext["extent_metric"]), 2)))\n'
        )
    if question_type == "object_abs_distance":
        # §12.3 官方口径：相机（观察点）→ 对象表面的稳健低分位距离
        return (
            f"{_STUB_BANNER}\n"
            f'd = camera_object_distance("{anchor_a}")\n'
            f'ReturnAnswer(str(round(d["distance_metric"], 2)))\n'
        )
    raise ValueError(f"未支持的合成题型: {question_type}")


def stub_direct_answer(episode: VSIBenchEpisode) -> str:
    """C0 direct-VLM 基线的 mock_light stub（仅管道验证，非模型输出）。"""
    if episode.options:
        return "ABCD"[_stable_seed(episode.qa_id, len(episode.options))]
    return "0.0"
