"""G-19 与 G7/G8/G9 数据源单测（§4 M5、§10 G7/G8/G9）。

SAM2/torch 不可用，故：提示词抽取、动态 mask（刚性残差）、mask→世界绑定、
track IoU、产物 ref 落盘全部走真实 numpy 代码路径；predictor 用 fake 注入。
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.segmentation.sam2_tracker import (
    TH_RIGIDITY_RESIDUAL_PX,
    bind_masks_to_world,
    bind_objects_for_scene,
    build_video_predictor,
    dynamic_mask_from_rigidity,
    ensure_checkpoint,
    object_prompts_from_handle,
    object_prompts_from_question,
    resolve_sam2_config,
    track_objects,
)

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
H, W = 480, 640


# ------------------------------------------------------------------ 配置解析 ----

def test_resolve_config_precedence(monkeypatch):
    monkeypatch.delenv("SKILL3D_SAM2_CHECKPOINT", raising=False)
    monkeypatch.delenv("SKILL3D_SAM2_CONFIG", raising=False)
    ckpt, cfg = resolve_sam2_config()
    # G-19 默认档 = 本机实测可用的 SAM2.1 Hiera-Large（Tiny 未下载）
    assert ckpt and ckpt.endswith("sam2.1_hiera_large.pt"), ckpt
    assert cfg == "configs/sam2.1/sam2.1_hiera_l.yaml"   # hydra 需相对形式
    monkeypatch.setenv("SKILL3D_SAM2_CHECKPOINT", "/tmp/x.pt")
    assert resolve_sam2_config()[0] == "/tmp/x.pt"
    assert resolve_sam2_config(checkpoint="explicit.pt")[0] == "explicit.pt"


def test_build_video_predictor_reports_missing_checkpoint(monkeypatch):
    """checkpoint/config 缺失 → 明确 RuntimeError（由 M5 降级分支接住）。

    注意：本机权重已就位（见 本机环境实测.md），故需把模块默认显式置空才能测该路径。
    """
    monkeypatch.setattr("skill3d.segmentation.sam2_tracker.SAM2_CHECKPOINT", None)
    monkeypatch.setattr("skill3d.segmentation.sam2_tracker.SAM2_CONFIG", None)
    monkeypatch.delenv("SKILL3D_SAM2_CHECKPOINT", raising=False)
    monkeypatch.delenv("SKILL3D_SAM2_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="未配置|权重不可得"):
        build_video_predictor(checkpoint=None, config=None)


def test_ensure_checkpoint_returns_local_file(tmp_path):
    p = tmp_path / "sam2.pt"
    p.write_bytes(b"w")
    assert ensure_checkpoint(str(p)) == str(p)


def test_ensure_checkpoint_returns_none_when_unfetchable():
    assert ensure_checkpoint("some/nonexistent-repo-xyz") is None


# ------------------------------------------------------------------ 提示词 ----

def test_object_prompts_from_question_word_boundary():
    assert object_prompts_from_question("How many table(s) are in this room?") == ["table"]
    got = object_prompts_from_question("What is the distance between the sofa and the chair?")
    assert got == ["sofa", "chair"]
    # 词边界：chair 不应命中 chairman
    assert "chair" not in object_prompts_from_question("Who is the chairman here?")
    assert object_prompts_from_question("nothing relevant") == []


def test_object_prompts_from_handle():
    class _O:
        def __init__(self, iid, hint, bbox):
            self.instance_id, self.class_hint, self.bbox = iid, hint, bbox

    class _H:
        _objs = {"obj_0": _O("obj_0", "sofa", [0, 0, 0, 1, 1, 1]),
                 "obj_1": _O("obj_1", "table", [0, 0, 0, 1, 1, 1])}

        def list_objects(self):
            return sorted(self._objs)

        def get_object(self, i):
            return self._objs[i]

    hints, boxes = object_prompts_from_handle(_H())
    assert hints == ["sofa", "table"] and len(boxes) == 2
    assert object_prompts_from_handle(None) == ([], [])


def _fake_predictor(n_frames=3):
    class _P:
        def init_state(self, video_path, **kw):
            # 真实 API：init_state 收 MP4/JPEG 目录路径；此处按目录内 JPEG 数计帧数
            import glob as _g
            n = len(_g.glob(str(video_path) + "/*.jpg"))
            return {"n": n}

        def add_new_points_or_box(self, state, frame_idx, obj_id, box=None, **kw):
            assert frame_idx == 0, "首帧给框提示"
            state.setdefault("objs", []).append(obj_id)

        def propagate_in_video(self, state):
            m = np.zeros((H, W), dtype=bool)
            m[100:140, 100:160] = True
            for t in range(state["n"]):
                yield t, list(range(len(state["objs"]))), [m.astype(np.float32)] * len(state["objs"])

    return _P()


def test_track_objects_with_box_prompts():
    frames = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(3)]
    masks = track_objects(frames, [[0, 0, 10, 10], [5, 5, 20, 20]], _fake_predictor(3))
    assert len(masks) == 2 and all(len(m) == 3 for m in masks)


def test_track_objects_requires_prompts():
    frames = [np.zeros((H, W, 3), dtype=np.uint8)]
    with pytest.raises(RuntimeError, match="需要 box_prompts"):
        track_objects(frames, None, _fake_predictor(1))
    with pytest.raises(RuntimeError, match="提示不可得"):
        track_objects(frames, None, _fake_predictor(1), question="irrelevant text")


# ------------------------------------------------------------------ G7 动态 mask ----

def _depth_and_pose(n=3, z=4.0, dx_px=1.0, cam_h=1.5):
    """恒定深度 z 的平面 + 沿 x 平移的相机。

    平移量按 `dx_world = dx_px * z / f` 取值，使真值光流恰为 `dx_px` 像素。
    """
    depth = np.full((n, H, W), z)
    c2w = np.zeros((n, 4, 4))
    step = dx_px * z / K[0, 0]
    for t in range(n):
        c2w[t, :3, :3] = np.eye(3)
        c2w[t, :3, 3] = [step * t, cam_h, 0.0]
        c2w[t, 3, 3] = 1.0
    return depth, c2w


def _textured_frames(n=3, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(n)]


def _smooth_texture(seed: int = 0) -> np.ndarray:
    """低频平滑纹理：Farneback 依赖局部亮度恒定，白噪声不可跟踪。"""
    import cv2

    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, (H // 16, W // 16, 3), dtype=np.uint8)
    base = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(base, (0, 0), 2)


def _consistent_frames(n=3, shift_px=1.0, moving_block: bool = False, seed=1):
    """构造与"恒定深度 + 相机平移"**几何一致**的帧序列。

    背景整体右移 `shift_px` 像素（np.roll 为其整数近似）；`moving_block=True` 时
    再让中央区块额外多移 14 像素，模拟独立运动的动态物体。

    区块内叠加高频斑纹：光流估计器需要可跟踪纹理才能测出大位移
    （否则测到的是"估计器失效"而不是"检测器失效"）。
    """
    base = _smooth_texture(seed)
    block = base[300:340, 100:160].copy()
    rng = np.random.default_rng(seed + 99)
    block = np.clip(block.astype(int)
                    + rng.integers(-60, 61, block.shape), 0, 255).astype(np.uint8)
    frames = [base]
    for t in range(1, n):
        f = np.roll(base, int(round(shift_px * t)), axis=1)
        if moving_block:
            f[300:340, 100:160] = np.roll(block, int(round(14 * t)), axis=1)
        frames.append(f)
    return frames


def test_dynamic_mask_pure_ego_motion_has_no_dynamic():
    """几何一致的自运动序列 → 观测光流 = 自运动预测光流 → 动态占比≈0。"""
    frames = _consistent_frames(3, shift_px=2.0)
    depth, c2w = _depth_and_pose(3, z=4.0, dx_px=2.0)
    dyn, notes = dynamic_mask_from_rigidity(frames, depth, c2w, K)
    assert dyn.shape == (3, H, W)
    assert float(dyn.mean()) < 0.05, float(dyn.mean())      # 无真实运动 → 不误报
    assert any("G7" in n for n in notes)


def test_dynamic_mask_requires_two_frames():
    dyn, notes = dynamic_mask_from_rigidity([np.zeros((H, W, 3), dtype=np.uint8)],
                                            np.zeros((1, H, W)), np.zeros((1, 4, 4)), K)
    assert dyn.size == 0 and any("帧数不足" in n for n in notes)


def test_dynamic_mask_flags_independently_moving_block():
    """背景按自运动模型移动、中央区块另按 14px 移动 → 该区块被检出且背景不误报。"""
    frames = _consistent_frames(3, shift_px=2.0, moving_block=True)
    depth, c2w = _depth_and_pose(3, z=4.0, dx_px=2.0)
    dyn, _ = dynamic_mask_from_rigidity(frames, depth, c2w, K, th_residual_px=3.0)
    in_block = float(dyn[0, 300:340, 100:160].mean())
    assert in_block > 0.8, in_block                    # 动态块被检出
    outside = dyn[0].copy()
    outside[290:350, 90:170] = False                   # 排除区块及其边缘
    assert float(outside.mean()) < 0.02, float(outside.mean())
    # 检出率显著高于背景（分离度）
    assert in_block > 20 * float(outside.mean())


def test_dynamic_mask_static_scene_stays_quiet():
    """相机平移与深度一致时（纯自运动）不应大面积误报。"""
    frames = _consistent_frames(3, shift_px=2.0, moving_block=False)
    depth, c2w = _depth_and_pose(3, z=4.0, dx_px=2.0)
    dyn, _ = dynamic_mask_from_rigidity(frames, depth, c2w, K)
    assert float(dyn.mean()) < 0.02, float(dyn.mean())


def test_threshold_is_configurable():
    frames = _consistent_frames(3, shift_px=2.0)
    depth, c2w = _depth_and_pose(3, z=4.0, dx_px=2.0)
    loose, _ = dynamic_mask_from_rigidity(frames, depth, c2w, K, th_residual_px=1e9)
    assert loose.sum() == 0                 # 阈值极大 → 全判静止
    assert TH_RIGIDITY_RESIDUAL_PX > 0


# ------------------------------------------------------------------ 世界绑定 ----

def test_bind_masks_to_world_centroid_in_front_of_camera(tmp_path):
    masks = [{t: _square_mask() for t in range(2)}]
    depth = np.full((2, H, W), 3.0)
    c2w = np.zeros((2, 4, 4))
    for t in range(2):
        c2w[t] = np.eye(4)
    objs = bind_masks_to_world(masks, depth, c2w, K, class_hints=["sofa"],
                              out_dir=tmp_path, scene_name="sc")
    assert len(objs) == 1
    o = objs[0]
    assert o.class_hint == "sofa" and o.confidence > 0
    # 相机在原点朝 +z 看，物体深度 3m → 世界 z ≈ +3
    assert o.centroid_world[2] == pytest.approx(3.0, abs=0.05)
    assert len(o.bbox) == 6 and o.bbox[3] >= o.bbox[0]
    # 产物 ref 已落盘（G-19：mask/点云 ref 不再是空串）
    assert np.load(o.mask_per_frame).shape == (2, H, W)
    assert np.load(o.pointcloud_world).shape[1] == 3


def test_bind_masks_to_world_unverified_when_no_valid_depth():
    """无有效深度 → 该对象标 unverified、confidence=0（§4 M5 字段 9）。"""
    masks = [{0: _square_mask()}]
    depth = np.zeros((1, H, W))                    # 全 0 → 无有效深度
    objs = bind_masks_to_world(masks, depth, np.eye(4)[None], K)
    assert objs[0].confidence == 0.0 and objs[0].class_hint == "unverified"
    assert objs[0].pointcloud_world == ""


def test_bind_masks_to_world_skips_resolution_mismatch():
    masks = [{0: np.zeros((10, 10), dtype=bool)}]   # 与深度分辨率不一致
    objs = bind_masks_to_world(masks, np.ones((1, H, W)), np.eye(4)[None], K)
    assert objs[0].confidence == 0.0


def _square_mask():
    m = np.zeros((H, W), dtype=bool)
    m[200:260, 300:380] = True
    return m


# ------------------------------------------------------------------ M5 端到端 ----

def test_bind_objects_for_scene_returns_g7_g9_stats(tmp_path):
    """M5 端到端：objects + stats(track_ious/dynamic_ratio) 供 M4 用。

    G8（bbox 覆盖）已按附录 A 删除 → stats 里不再有 `bbox_coverage`。
    """
    frames = _textured_frames(3, seed=4)
    depth = np.full((3, H, W), 3.0)
    c2w = np.zeros((3, 4, 4))
    for t in range(3):
        c2w[t] = np.eye(4)
    objs, stats, notes = bind_objects_for_scene(
        frames, None, "How many chair(s) are in this room?",
        depth_maps=depth, c2w_list=c2w, intrinsics=K,
        out_dir=tmp_path, scene_name="sc", predictor=_fake_predictor(3),
    )
    assert objs and len(objs) == 1
    assert "track_ious" in stats and stats["track_ious"][0] == pytest.approx(1.0)
    assert "dynamic_ratio" in stats and 0.0 <= stats["dynamic_ratio"] <= 1.0
    assert "dynamic_masks" in stats
    assert "bbox_coverage" not in stats          # G8 已删除，不再产出
    assert any("G9" in n for n in notes)


def test_bind_objects_for_scene_without_depth_degrades():
    objs, stats, notes = bind_objects_for_scene([np.zeros((H, W, 3), dtype=np.uint8)],
                                                None, "q", depth_maps=None, c2w_list=None)
    assert objs == [] and stats == {} and any("跳过" in n for n in notes)


# ------------------------------------------- §3 M5 字段 5：3D 去重三判据 ----

def _inst(iid, hint, centroid, conf=0.9):
    from skill3d.schemas import ObjectInstance

    return ObjectInstance(instance_id=iid, class_hint=hint, mask_per_frame="",
                          pointcloud_world="", centroid_world=list(centroid),
                          bbox=[0.0] * 6, confidence=conf)


def test_dedupe_merges_same_class_close_and_temporally_overlapping():
    """三判据同时满足才合并：同类 + 质心近 + 支撑帧有交集。"""
    from skill3d.segmentation.sam2_tracker import _dedupe_by_world_centroid

    depth = np.full((4, 8, 8), 2.0)                 # 深度中位 2.0 → tol = 0.3
    objs = [_inst("a", "chair", [0.0, 0.0, 0.0], 0.9),
            _inst("b", "chair", [0.05, 0.0, 0.0], 0.5)]      # 近且同类
    masks = [{0: np.zeros((8, 8), bool), 1: np.zeros((8, 8), bool)},   # a: 帧 0,1
             {1: np.zeros((8, 8), bool), 2: np.zeros((8, 8), bool)}]   # b: 帧 1,2
    kept, n_dup, groups = _dedupe_by_world_centroid(objs, depth, masks_per_object=masks)
    assert n_dup == 1 and len(kept) == 1 and groups == [[0, 1]]
    assert kept[0].instance_id == "a"               # 保留置信度更高的


def test_dedupe_keeps_distinct_objects():
    """不同类别 / 质心过远 / 时序不重叠 → 一律不合并（防止误并）。"""
    from skill3d.segmentation.sam2_tracker import _dedupe_by_world_centroid

    depth = np.full((4, 8, 8), 2.0)                 # tol = 0.3
    masks_ov = [{0: np.ones((8, 8), bool)}, {0: np.ones((8, 8), bool)}]

    # 1) 同类、近、时序重叠 → 合并（基线）
    kept, n, _ = _dedupe_by_world_centroid(
        [_inst("a", "chair", [0, 0, 0]), _inst("b", "chair", [0.1, 0, 0])],
        depth, masks_per_object=masks_ov)
    assert n == 1

    # 2) 类别不同（桌 vs 椅）→ 不合并
    kept, n, _ = _dedupe_by_world_centroid(
        [_inst("a", "table", [0, 0, 0]), _inst("b", "chair", [0.1, 0, 0])],
        depth, masks_per_object=masks_ov)
    assert n == 0 and len(kept) == 2

    # 3) 质心超过 tol → 不合并
    kept, n, _ = _dedupe_by_world_centroid(
        [_inst("a", "chair", [0, 0, 0]), _inst("b", "chair", [5.0, 0, 0])],
        depth, masks_per_object=masks_ov)
    assert n == 0 and len(kept) == 2

    # 4) 时序不重叠（先后出现在同一位置）→ 不合并
    masks_seq = [{0: np.ones((8, 8), bool)}, {3: np.ones((8, 8), bool)}]
    kept, n, _ = _dedupe_by_world_centroid(
        [_inst("a", "chair", [0, 0, 0]), _inst("b", "chair", [0.1, 0, 0])],
        depth, masks_per_object=masks_seq)
    assert n == 0 and len(kept) == 2


def test_merge_mask_groups_unions_frames():
    """去重分组 → 每组一份掩码字典（帧级并集），供去重后的 G8/G9 使用。"""
    from skill3d.segmentation.sam2_tracker import _merge_mask_groups

    m1 = {0: np.array([[True, False]]), 1: np.array([[False, False]])}
    m2 = {1: np.array([[False, True]]), 2: np.array([[True, True]])}
    merged = _merge_mask_groups([m1, m2], [[0, 1]])
    assert set(merged[0]) == {0, 1, 2}
    assert merged[0][1].tolist() == [[False, True]]      # 并集
    assert merged[0][0].tolist() == [[True, False]]
