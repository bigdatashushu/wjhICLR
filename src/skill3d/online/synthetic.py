"""mock_light 合成输入（§9.2 Tool 三档 Mock 的 mock_light 语义）。

**用途限定**：M0/M1 验收与集成测试的管道验证（`online/runner.py` 在
`mode="mock_light"` 时调用）。合成数据**绝不可用于论文实验结论**；
真实实验一律 `mode="real"`（准入门强制 real，§5.6b）。

合成内容全部由 seed 决定 → 同 seed 字节级一致（§4 M8/M17 验收）。
几何是真算的（射线-房间盒求交得深度、c2w 由轨迹给出），因此下游
G1–G11 / Tool / Verifier 在合成场景上执行的是**真实计算**，只是数据是合成的。

合成 GT 与合成场景同源构造，且 route_plan / appearance_order 两类的 GT 不携带
真实语义（仅为占位字母）——故合成集上的 accuracy/MRA **不构成任何精度结论**。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from skill3d.gates import iqa
from skill3d.reconstruction_gate.confidence_map import fuse_confidence, coverage_ratio
from skill3d.reconstruction_gate.quality_metrics import compute_g1_g11
from skill3d.reconstruction_gate.scene_state import TH_SCALE_CI, route_from_quality
from skill3d.routing.task_classifier import MCA_TYPES, NA_TYPES
from skill3d.schemas import (
    InputFrame,
    ObjectInstance,
    QualityMetrics,
    ReconstructionArtifact,
    SceneState,
    VSIBenchEpisode,
)
from skill3d.tools.scene_handle import SceneHandle

# 与 VSI-Bench 视频规格对齐（§1.2：640×480）
FRAME_H, FRAME_W = 480, 640
N_FRAMES = 32

# 合成房间盒（米，z 向上）：6m × 8m × 2.8m
ROOM_MIN = np.array([-3.0, -4.0, 0.0])
ROOM_MAX = np.array([3.0, 4.0, 2.8])
# 合成相机内参
FOCAL = 500.0
# 深度图子采样网格（省内存：32×60×80）
DEPTH_H, DEPTH_W = 60, 80
# 合成尺度 CI（米）：小于 TH_SCALE_CI 记 scale_known=True
SYNTH_SCALE_CI = 0.02


@dataclass
class SyntheticGeometry:
    """合成场景几何（可注入 SceneHandle 的世界系数据）。"""

    scene_name: str
    c2w: np.ndarray                      # (N,4,4) SE(3)
    intrinsics: np.ndarray               # (3,3)
    depth_maps: np.ndarray               # (N,h,w) 沿相机 z 的深度（米）
    point_conf: np.ndarray               # (N,h,w) 逐点置信度
    coverage_count: np.ndarray           # (N,h,w) 逐点观测帧数
    objects: list[ObjectInstance] = field(default_factory=list)
    metric_scale: float = 1.0


@dataclass
class SyntheticEpisode:
    """合成 episode：帧元数据 + 真实像素 + 场景几何 + 合成 GT。"""

    episode: VSIBenchEpisode
    frames: list[np.ndarray]              # (N,H,W,3) uint8 RGB
    geometry: SyntheticGeometry


# ---------------------------------------------------------------- 几何合成 ----

def _rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _camera_pose_base() -> np.ndarray:
    """相机基姿态：cam z（前向）→ 世界 +y，cam y（下）→ 世界 -z，cam x → 世界 +x。"""
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [0.0, -1.0, 0.0]])


def _intrinsics() -> np.ndarray:
    return np.array([[FOCAL, 0.0, FRAME_W / 2.0],
                     [0.0, FOCAL, FRAME_H / 2.0],
                     [0.0, 0.0, 1.0]])


def _camera_trajectory(n_frames: int, seed: int) -> np.ndarray:
    """沿房间对角走一条直线轨迹（非退化基线），带轻微偏航摆动。"""
    t = np.linspace(0.0, 1.0, n_frames)
    start = np.array([-1.6, -3.0, 1.55])
    end = np.array([1.6, 3.0, 1.55])
    base_R = _camera_pose_base()
    poses = []
    for i, s in enumerate(t):
        pos = start + s * (end - start)
        yaw = 0.15 * np.sin(2.0 * np.pi * (i / max(n_frames - 1, 1)) + 0.1 * seed)
        R = _rotation_z(yaw) @ base_R
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos
        poses.append(c2w)
    return np.stack(poses)


def _depth_grid(c2w: np.ndarray) -> np.ndarray:
    """逐帧深度：对每个像素做相机射线与房间盒的 slab 求交（真实几何）。"""
    K = _intrinsics()
    us = (np.arange(DEPTH_W) + 0.5) * (FRAME_W / DEPTH_W)
    vs = (np.arange(DEPTH_H) + 0.5) * (FRAME_H / DEPTH_H)
    uu, vv = np.meshgrid(us, vs)
    pix = np.stack([uu, vv, np.ones_like(uu)], axis=-1)          # (h,w,3)
    d_cam = pix @ np.linalg.inv(K).T                              # 相机系射线
    d_cam_unit = d_cam / np.linalg.norm(d_cam, axis=-1, keepdims=True)

    out = np.empty((len(c2w), DEPTH_H, DEPTH_W), dtype=np.float32)
    for i, pose in enumerate(c2w):
        R = pose[:3, :3]
        o = pose[:3, 3]
        d_world = d_cam_unit @ R.T                                # 世界系单位射线
        # slab 法求交（房间盒）
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (ROOM_MIN - o) / d_world
            t2 = (ROOM_MAX - o) / d_world
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = np.where((tmax >= np.maximum(tmin, 0.0)), tmin, np.nan)
        hit = np.clip(np.nan_to_num(hit, nan=0.0), 0.2, 50.0)
        # 深度取沿相机 z 的分量
        out[i] = (hit * d_cam_unit[..., 2]).astype(np.float32)
    return out


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


# 合成对象：靠墙摆放，使点云并集 bbox 接近房间footprint
_SYNTH_OBJECTS: list[tuple[str, list[float], list[float]]] = [
    # class_hint, center, size(xyz)
    ("sofa", [2.0, 0.0, 0.40], [1.9, 0.9, 0.80]),
    ("table", [-2.2, 1.5, 0.40], [1.5, 0.9, 0.75]),
    ("chair", [0.0, 3.3, 0.45], [0.6, 0.6, 0.90]),
    ("cabinet", [0.0, -3.3, 0.90], [1.2, 0.6, 1.80]),
]


def _make_objects(out_dir: Optional[str]) -> list[ObjectInstance]:
    from pathlib import Path

    objs: list[ObjectInstance] = []
    for hint, center, size in _SYNTH_OBJECTS:
        c = np.asarray(center, dtype=np.float64)
        s = np.asarray(size, dtype=np.float64)
        bbox = [float(c[i] - s[i] / 2) for i in range(3)] + \
               [float(c[i] + s[i] / 2) for i in range(3)]
        pc_ref, mask_ref = "", ""
        if out_dir is not None:
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            # 8 角点 + 面上采样，作为对象点云（合成）
            corners = np.array([[bbox[0] if i & 1 else bbox[3] for i in range(3)],
                                [bbox[3] if i & 2 else bbox[0] for i in range(3)],
                                [bbox[3] if i & 4 else bbox[2] for i in range(3)]])
            rng = np.random.default_rng(_stable_seed(hint, 2**31))
            surf = rng.uniform(c - s / 2, c + s / 2, size=(64, 3))
            pts = np.vstack([corners.T.reshape(-1, 3), surf])
            p = d / f"{hint}_points.npy"
            np.save(p, pts)
            pc_ref = str(p)
            m = d / f"{hint}_mask.npy"
            np.save(m, np.ones((N_FRAMES, DEPTH_H, DEPTH_W), dtype=np.uint8))
            mask_ref = str(m)
        objs.append(
            ObjectInstance(
                instance_id=f"{hint}-0",
                class_hint=hint,
                mask_per_frame=mask_ref,
                pointcloud_world=pc_ref,
                centroid_world=[float(x) for x in c],
                bbox=[float(x) for x in bbox],
                confidence=0.9,
            )
        )
    return objs


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

def _question_and_gt(question_type: str, geometry: SyntheticGeometry,
                     rng: np.random.Generator) -> tuple[str, list[str] | None, str]:
    """合成 question / options / ground_truth。

    route_plan / appearance_order 的 GT 无真实语义（固定字母占位），
    见模块 docstring：合成集精度不构成结论。
    """
    names = [o.class_hint for o in geometry.objects]
    if question_type == "object_counting":
        n = len(names)
        options = ["2", "3", "4", "5"]
        gt_letter = "ABCD"[options.index(str(n))] if str(n) in options else "A"
        return ("How many objects are in the scene? Count them.", options, gt_letter)
    if question_type == "relative_direction":
        options = ["left", "right", "front", "behind"]
        direction = _independent_direction(geometry)
        gt = "ABCD"[options.index(direction)] if direction in options else "A"
        return ("What is the relative direction of the sofa from the chair "
                "(facing +y)?", options, gt)
    if question_type == "route_plan":
        options = ["A", "B", "C", "D"]
        return ("Which route leads to the sofa?", options, "ABCD"[int(rng.integers(0, 4))])
    if question_type == "appearance_order":
        options = ["A", "B", "C", "D"]
        return ("In what order do the objects appear?", options,
                "ABCD"[int(rng.integers(0, 4))])
    if question_type == "room_size":
        gt = float(np.prod(ROOM_MAX[:2] - ROOM_MIN[:2]))
        return ("What is the area of the room in square meters?", None, f"{gt:.2f}")
    if question_type == "object_size":
        gt = float(max(_SYNTH_OBJECTS[0][2]))
        return ("What is the longest dimension of the sofa in meters?", None, f"{gt:.2f}")
    if question_type in ("absolute_distance", "relative_distance"):
        a = np.asarray(geometry.objects[0].centroid_world)
        b = np.asarray(geometry.objects[2].centroid_world)
        gt = float(np.linalg.norm(a - b))
        return ("What is the distance between the sofa and the chair in meters?",
                None, f"{gt:.2f}")
    raise ValueError(f"未支持的合成题型: {question_type}")


def _independent_direction(geometry: SyntheticGeometry) -> str:
    """独立复算方向（不调用 Tool），用于合成 GT；规则与 M6 Tool 一致（45° 半角）。"""
    chair = np.asarray(geometry.objects[2].centroid_world)
    sofa = np.asarray(geometry.objects[0].centroid_world)
    facing = np.array([0.0, 1.0])          # 世界 +y（地面平面 x-y）
    d = (sofa - chair)[:2]
    d = d / np.linalg.norm(d)
    cos_t = float(np.dot(facing, d))
    if cos_t >= 0.7071:
        return "front"
    if cos_t <= -0.7071:
        return "behind"
    cross = float(facing[0] * d[1] - facing[1] * d[0])
    return "left" if cross > 0 else "right"


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
    if question_type not in (MCA_TYPES | NA_TYPES):
        raise ValueError(f"未知题型: {question_type}")
    if degrade is not None and degrade not in _DEGRADE_MODES:
        raise ValueError(f"未知 degradation: {degrade}（可选 {_DEGRADE_MODES}）")

    rng = np.random.default_rng(seed)
    h, w = frame_size
    c2w = _camera_trajectory(n_frames, seed)
    geometry = SyntheticGeometry(
        scene_name=scene_name,
        c2w=c2w,
        intrinsics=_intrinsics(),
        depth_maps=_depth_grid(c2w),
        point_conf=np.zeros((n_frames, DEPTH_H, DEPTH_W), dtype=np.float32),
        coverage_count=np.zeros((n_frames, DEPTH_H, DEPTH_W), dtype=np.float32),
        objects=_make_objects(out_dir),
    )
    geometry.point_conf, geometry.coverage_count = _per_point_stats(n_frames)

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
    )
    return SyntheticEpisode(episode=episode, frames=pixels, geometry=geometry)


# ------------------------------------------------------- SceneState / 句柄 ----

def synthetic_scale_known() -> bool:
    """合成场景量在米制下定义，CI 取 SYNTH_SCALE_CI ≤ TH_SCALE_CI → 尺度已知。"""
    return SYNTH_SCALE_CI <= TH_SCALE_CI


def build_scene_state(
    geometry: SyntheticGeometry,
    frames: Sequence[np.ndarray],
    artifact_ref: Optional[str] = None,
) -> tuple[SceneState, SceneHandle, QualityMetrics]:
    """按 §4 M4 口径在合成数据上**真实计算** G1–G11 并构造 SceneState。

    不构造 ReconstructionArtifact（其 recon_method 为 §5.2 受控枚举，
    合成产物不得冒用 vggt/dust3r_mastr/colmap 名义）。
    """
    import numpy as np

    depth = geometry.depth_maps
    # 合成场景无对应关系 → G5 重投影残差留空（NaN）；G7/G8/G9 需 M5 产物 → 交真实路径
    q = compute_g1_g11(
        None,
        frames=frames,
        depth_maps=depth,
        c2w_list=geometry.c2w,
        scale_ci=SYNTH_SCALE_CI,
    )
    fused = fuse_confidence(geometry.point_conf, geometry.coverage_count)
    coverage = coverage_ratio(fused)
    scale_known = synthetic_scale_known()
    route = route_from_quality(q, scale_known)

    scene = SceneState(
        artifact_ref=artifact_ref or f"mock_light:{geometry.scene_name}",
        route=route,  # type: ignore[arg-type]
        frame="world",
        scale_known=scale_known,
        objects=[o.instance_id for o in geometry.objects],
        summary=(
            f"scene={geometry.scene_name} (mock_light 合成) route={route} "
            f"overall_quality={q.overall_quality:.3f} scale_known={scale_known} "
            f"coverage={coverage:.3f} objects={len(geometry.objects)}"
        ),
    )
    handle = SceneHandle(
        scene,
        objects=geometry.objects,
        c2w_list=geometry.c2w,
        intrinsics=geometry.intrinsics,
        quality_overall=q.overall_quality,
    )
    return scene, handle, q


# --------------------------------------------------------- 确定性 stub 程序 ----

_STUB_BANNER = "# mock_light 确定性 stub（非 Qwen3-VL-8B 输出，仅管道验证）"


def stub_program(question_type: str, episode: VSIBenchEpisode,
                 geometry: SyntheticGeometry) -> str:
    """按题型给出确定性 stub program（仅 mock_light；程序仍走 M9 AST + M10 沙箱）。"""
    names = [o.class_hint for o in geometry.objects]
    anchor_a = names[0] if names else "sofa"
    anchor_b = names[2] if len(names) > 2 else anchor_a
    opts = list(episode.options or [])

    if question_type == "object_counting":
        pairs = ", ".join(f'("{o}", "{"ABCD"[i]}")' for i, o in enumerate(opts))
        return (
            f"{_STUB_BANNER}\n"
            f"pairs = [{pairs}]\n"
            f"n = len(scene.list_objects())\n"
            f'answer = pairs[0][1]\n'
            f"for text, letter in pairs:\n"
            f"    if text.strip() == str(n):\n"
            f"        answer = letter\n"
            f"ReturnAnswer(answer)\n"
        )
    if question_type == "relative_direction":
        pairs = ", ".join(f'("{o}", "{"ABCD"[i]}")' for i, o in enumerate(opts))
        return (
            f"{_STUB_BANNER}\n"
            f"pairs = [{pairs}]\n"
            f'obs = object_centroid("{anchor_b}")\n'
            f'obj = object_centroid("{anchor_a}")\n'
            f"d = relative_direction(obs, [0.0, 1.0, 0.0], obj)\n"
            f'answer = pairs[0][1]\n'
            f"for text, letter in pairs:\n"
            f"    if text == d:\n"
            f"        answer = letter\n"
            f"ReturnAnswer(answer)\n"
        )
    if question_type in ("route_plan", "appearance_order"):
        # 非信息性 stub：该两类合成 GT 也无真实语义（见模块 docstring）
        letter = "ABCD"[_stable_seed(episode.qa_id + question_type, 4)]
        return f"{_STUB_BANNER}\nReturnAnswer(\"{letter}\")\n"
    if question_type == "room_size":
        return f"{_STUB_BANNER}\nReturnAnswer(str(round(room_size_m2(), 2)))\n"
    if question_type == "object_size":
        return (
            f"{_STUB_BANNER}\n"
            f'ReturnAnswer(str(round(object_size_longest_dim("{anchor_a}"), 2)))\n'
        )
    if question_type in ("absolute_distance", "relative_distance"):
        return (
            f"{_STUB_BANNER}\n"
            f'a = object_centroid("{anchor_a}")\n'
            f'b = object_centroid("{anchor_b}")\n'
            f"ReturnAnswer(str(round(euclidean_distance(a, b), 2)))\n"
        )
    raise ValueError(f"未支持的合成题型: {question_type}")


def stub_direct_answer(episode: VSIBenchEpisode) -> str:
    """C0 direct-VLM 基线的 mock_light stub（仅管道验证，非模型输出）。"""
    if episode.options:
        return "ABCD"[_stable_seed(episode.qa_id, len(episode.options))]
    return "0.0"
