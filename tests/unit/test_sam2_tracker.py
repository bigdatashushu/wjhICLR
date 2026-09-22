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
        def __init__(self, oid, name, bbox):
            self.obj_id, self.category_name, self.bbox = oid, name, bbox

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
    assert o.category_name == "sofa" and o.det_conf > 0
    # visible_frames 语义不变：帧槽位序号、升序
    assert o.visible_frames == [0, 1]
    # 相机在原点朝 +z 看，物体深度 3m → 世界 z ≈ +3
    assert o.centroid_world[2] == pytest.approx(3.0, abs=0.05)
    assert len(o.bbox) == 6 and o.bbox[3] >= o.bbox[0]
    # 产物 ref 已落盘（G-19：mask/点云 ref 不再是空串）
    assert np.load(o.mask_per_frame).shape == (2, H, W)
    assert np.load(o.pointcloud_world).shape[1] == 3
    # v6 §5.6：未给逐点 conf 时写空 ref（不伪造），质心不写第二份文件
    assert o.pointconf_world == "" and o.centroid_ref == ""
    assert o.grounding_status == "base_list" and o.duplicate_suspect is False


def test_bind_masks_to_world_unverified_when_no_valid_depth():
    """无有效深度 → 该对象标 unverified、det_conf=0（§4 M5 字段 9）。"""
    masks = [{0: _square_mask()}]
    depth = np.zeros((1, H, W))                    # 全 0 → 无有效深度
    objs = bind_masks_to_world(masks, depth, np.eye(4)[None], K)
    assert objs[0].det_conf == 0.0 and objs[0].category_name == "unverified"
    assert objs[0].pointcloud_world == ""
    assert objs[0].pointconf_world == ""


def test_bind_masks_to_world_skips_resolution_mismatch():
    masks = [{0: np.zeros((10, 10), dtype=bool)}]   # 与深度分辨率不一致
    objs = bind_masks_to_world(masks, np.ones((1, H, W)), np.eye(4)[None], K)
    assert objs[0].det_conf == 0.0


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
    # v6 §7.1：track_consensus 的占比输入（对象 3 帧全可见 → 1.0）
    assert stats["track_stable_ratio"] == pytest.approx(1.0)


def test_bind_objects_for_scene_without_depth_degrades():
    objs, stats, notes = bind_objects_for_scene([np.zeros((H, W, 3), dtype=np.uint8)],
                                                None, "q", depth_maps=None, c2w_list=None)
    assert objs == [] and stats == {} and any("跳过" in n for n in notes)


def test_bind_objects_for_scene_threads_point_conf_and_track_ids(tmp_path):
    """v6 §5.6：逐点 conf 落盘与点云 1:1；每个对象都有确定性 track_id。

    `point_conf` / `depth_conf` 两条命名都能传（后者是别名），缺省时不落 conf。
    """
    frames = _textured_frames(3, seed=4)
    depth = np.full((3, H, W), 3.0)
    c2w = np.stack([np.eye(4)] * 3)
    conf = np.full((3, H, W), 2.5)
    objs, _, _ = bind_objects_for_scene(
        frames, None, "How many chair(s) are in this room?",
        depth_maps=depth, c2w_list=c2w, intrinsics=K, out_dir=tmp_path,
        scene_name="sc", predictor=_fake_predictor(3), point_conf=conf,
    )
    assert objs and all(o.track_id for o in objs)
    assert [o.track_id for o in objs] == [f"trk{i}" for i in range(len(objs))]
    assert all(o.grounding_status == "base_list" for o in objs)
    for o in objs:
        assert o.pointconf_world
        assert len(np.load(o.pointconf_world)) == len(np.load(o.pointcloud_world))

    # 别名 `depth_conf` 等价；不给 conf 时 ref 为空（不伪造）
    objs2, _, _ = bind_objects_for_scene(
        frames, None, "How many chair(s) are in this room?",
        depth_maps=depth, c2w_list=c2w, intrinsics=K, out_dir=tmp_path,
        scene_name="sc2", predictor=_fake_predictor(3), depth_conf=conf,
        use_inventory_cache=False,
    )
    assert objs2 and all(o.pointconf_world for o in objs2)
    objs3, _, _ = bind_objects_for_scene(
        frames, None, "How many chair(s) are in this room?",
        depth_maps=depth, c2w_list=c2w, intrinsics=K, out_dir=tmp_path,
        scene_name="sc3", predictor=_fake_predictor(3),
        use_inventory_cache=False,
    )
    assert objs3 and all(o.pointconf_world == "" for o in objs3)


# ------------------------------------------- §3 M5 字段 5：3D 去重三判据 ----

def _inst(iid, hint, centroid, conf=0.9):
    from skill3d.schemas import ObjectRecord

    return ObjectRecord(obj_id=iid, category_name=hint, mask_per_frame="",
                        pointcloud_world="", centroid_world=list(centroid),
                        bbox=[0.0] * 6, det_conf=conf)


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
    assert kept[0].obj_id == "a"                    # 保留置信度更高的
    # v6：合并组只有一条 track（幸存代表的身份），且该对象被标重复嫌疑
    assert kept[0].track_id == "trk0"
    assert kept[0].duplicate_suspect is True


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


def test_track_stable_ratio_is_visibility_fraction_mean():
    """§7.1 track_consensus 输入：每条 track 的可见帧占比均值；无数据 → None（不伪造）。"""
    from skill3d.segmentation.sam2_tracker import track_stable_ratio

    full = _inst("a", "chair", [0, 0, 0])
    full.visible_frames = [0, 1, 2, 3]
    half = _inst("b", "table", [5, 0, 0])
    half.visible_frames = [0, 1]
    assert track_stable_ratio([full], 4) == pytest.approx(1.0)
    assert track_stable_ratio([full, half], 4) == pytest.approx(0.75)
    assert track_stable_ratio([], 4) is None            # 无对象 → None
    assert track_stable_ratio([full], 0) is None        # 无帧集 → None


def test_merge_mask_groups_unions_frames():
    """去重分组 → 每组一份掩码字典（帧级并集），供去重后的 G8/G9 使用。"""
    from skill3d.segmentation.sam2_tracker import _merge_mask_groups

    m1 = {0: np.array([[True, False]]), 1: np.array([[False, False]])}
    m2 = {1: np.array([[False, True]]), 2: np.array([[True, True]])}
    merged = _merge_mask_groups([m1, m2], [[0, 1]])
    assert set(merged[0]) == {0, 1, 2}
    assert merged[0][1].tolist() == [[False, True]]      # 并集
    assert merged[0][0].tolist() == [[True, False]]


# =================== v6 §5.6 / §7.1：track_id / duplicate_suspect ===================
#
# 这三条字段是 `count_objects`（§9.2 "track 共识计数，不数清单长度"）与
# `track_consensus` 证据（§7.1）的唯一数据来源，必须有单测盯着。

def _tracks(objs) -> set:
    return {o.track_id for o in objs if o.track_id}


def test_bind_masks_to_world_track_id_is_deterministic_propagation_index():
    """track_id = 确定性传播序号（同名空间前缀），重跑逐位一致 —— 可复现计数。"""
    depth = np.full((2, H, W), 3.0)
    c2w = np.stack([np.eye(4)] * 2)
    masks = [{0: _square_mask(), 1: _square_mask()},
             {0: _square_mask(), 1: _square_mask()}]
    a = bind_masks_to_world(masks, depth, c2w, K, class_hints=["chair", "table"])
    b = bind_masks_to_world(masks, depth, c2w, K, class_hints=["chair", "table"])
    assert [o.track_id for o in a] == ["trk0", "trk1"]
    assert [o.track_id for o in a] == [o.track_id for o in b]      # 稳定可复现
    # 逐题补漏用另一命名空间：不同 pass 的传播序号会重名，必须隔开
    q = bind_masks_to_world(masks[:1], depth, c2w, K, class_hints=["telephone"],
                            track_prefix="qtrk", grounding_status="question_targeted_fill")
    assert q[0].track_id == "qtrk0" and q[0].grounding_status == "question_targeted_fill"
    assert not (_tracks(a) & _tracks(q))                            # 不与基础清单撞名


def test_dedupe_merged_group_shares_representative_track():
    """合并组只算一条 track（幸存代表的身份）→ 3 条候选计数为 1。"""
    from skill3d.segmentation.sam2_tracker import _dedupe_by_world_centroid

    depth = np.full((4, 8, 8), 2.0)                 # 深度中位 2.0 → tol = 0.3
    objs = [_inst("a", "chair", [0.0, 0.0, 0.0], 0.9),
            _inst("b", "chair", [0.05, 0.0, 0.0], 0.5),
            _inst("c", "chair", [0.08, 0.0, 0.0], 0.4)]
    for i, o in enumerate(objs):
        o.track_id = f"trk{i}"
    frames = {0: np.ones((8, 8), bool), 1: np.ones((8, 8), bool)}
    masks = [frames, frames, frames]                # 三份候选时序全重叠
    kept, n_dup, groups = _dedupe_by_world_centroid(objs, depth, masks_per_object=masks)
    assert n_dup == 2 and len(kept) == 1 and groups == [[0, 1, 2]]
    # 组内所有成员统一为幸存代表（trk0）的 track；清单长度 3 ≠ track 数 1
    assert kept[0].track_id == "trk0"
    assert _tracks(kept) == {"trk0"}
    assert kept[0].duplicate_suspect is True
    # 原对象不被就地改写（去重结论写在返回的保留对象上）
    assert objs[1].duplicate_suspect is False and objs[1].track_id == "trk1"


def test_duplicate_suspect_true_exactly_for_merged_groups():
    """① 合并组（≥2 条候选）→ True；未合并的独立对象 → False。"""
    from skill3d.segmentation.sam2_tracker import _dedupe_by_world_centroid

    depth = np.full((4, 8, 8), 2.0)                 # tol = 0.3
    ov = {0: np.ones((8, 8), bool), 1: np.ones((8, 8), bool)}
    # a/b 同类近且时序重叠（合并）；c 同类但质心远（独立）；d 异物近（独立）
    objs = [_inst("a", "chair", [0.0, 0.0, 0.0], 0.9),
            _inst("b", "chair", [0.05, 0.0, 0.0], 0.5),
            _inst("c", "chair", [5.0, 0.0, 0.0], 0.8),
            _inst("d", "table", [0.05, 0.0, 0.0], 0.7)]
    kept, n_dup, groups = _dedupe_by_world_centroid(objs, depth,
                                                    masks_per_object=[ov] * 4)
    assert n_dup == 1 and groups == [[0, 1], [2], [3]]
    flags = {o.obj_id: o.duplicate_suspect for o in kept}
    assert flags == {"a": True, "c": False, "d": False}
    assert sum(1 for o in kept if o.duplicate_suspect) == 1


def test_duplicate_suspect_flags_close_same_class_without_temporal_overlap():
    """② 去重后仍"同类 + 质心过近"（时序不重叠故不合并）→ 两条都标嫌疑。

    这正是"先后出现在同一位置"的重复绑定风险：按 C-9 纪律**不合并**（可能是两个
    物体），但计数必须降级输出（`count_objects` 的 `duplicate_suspect`）。
    """
    from skill3d.segmentation.sam2_tracker import _dedupe_by_world_centroid

    depth = np.full((4, 8, 8), 2.0)                 # tol = 0.3
    seq = [{0: np.ones((8, 8), bool)}, {3: np.ones((8, 8), bool)}]
    kept, n_dup, _ = _dedupe_by_world_centroid(
        [_inst("a", "chair", [0.0, 0.0, 0.0]), _inst("b", "chair", [0.1, 0.0, 0.0])],
        depth, masks_per_object=seq)
    assert n_dup == 0 and len(kept) == 2            # 不合并（时序不重叠）
    assert all(o.duplicate_suspect for o in kept)   # 但都标重复嫌疑

    # 类别不同 / 质心过远 → 不标嫌疑（避免把不同物体误报成重复）
    ov = [{0: np.ones((8, 8), bool)}, {0: np.ones((8, 8), bool)}]
    kept, _, _ = _dedupe_by_world_centroid(
        [_inst("a", "table", [0, 0, 0]), _inst("b", "chair", [0.1, 0, 0])],
        depth, masks_per_object=ov)
    assert not any(o.duplicate_suspect for o in kept)
    kept, _, _ = _dedupe_by_world_centroid(
        [_inst("a", "chair", [0, 0, 0]), _inst("b", "chair", [5.0, 0, 0])],
        depth, masks_per_object=ov)
    assert not any(o.duplicate_suspect for o in kept)

    # unverified（类别空、质心是占位 [0,0,0]）不成堆误报
    unver = [_inst("a", "unverified", [0, 0, 0], 0.0), _inst("b", "unverified", [0, 0, 0], 0.0)]
    kept, _, _ = _dedupe_by_world_centroid(unver, depth, masks_per_object=seq)
    assert not any(o.duplicate_suspect for o in kept)


# =================== v6 §12.2：pointconf_world（逐点 1:1） ===================

def test_bind_masks_to_world_pointconf_saved_and_aligned_1to1(tmp_path):
    """落盘的 conf 与点云逐点对齐（连下采样后也保持），且不给就写空 ref。"""
    masks = [{0: _square_mask(), 1: _square_mask()}]
    depth = np.full((2, H, W), 3.0)
    c2w = np.stack([np.eye(4)] * 2)
    # 每个像素一个可区分的 conf 值 → 能反查"这条点的 conf 是不是它自己的像素"
    conf = np.arange(H * W, dtype=np.float64).reshape(H, W)
    objs = bind_masks_to_world(masks, depth, c2w, K, class_hints=["sofa"],
                               out_dir=tmp_path, scene_name="sc", point_conf=conf)
    o = objs[0]
    assert o.pointcloud_world and o.pointconf_world
    pts, saved = np.load(o.pointcloud_world), np.load(o.pointconf_world)
    assert saved.shape == (pts.shape[0],) and saved.shape[0] > 0
    # 由 3D 坐标反推像素（c2w=I、恒定深度）→ 逐点核对 conf
    u = np.round(pts[:, 0] * K[0, 0] / pts[:, 2] + K[0, 2]).astype(int)
    v = np.round(pts[:, 1] * K[1, 1] / pts[:, 2] + K[1, 2]).astype(int)
    assert np.array_equal(saved, conf[v, u])

    # 下采样后仍逐点对齐（同一套下标）
    small = bind_masks_to_world(masks, depth, c2w, K, class_hints=["sofa"],
                                out_dir=tmp_path, scene_name="sc_small", max_points=100,
                                point_conf=conf)
    pts_s = np.load(small[0].pointcloud_world)
    saved_s = np.load(small[0].pointconf_world)
    assert pts_s.shape[0] == saved_s.shape[0] == 100
    u = np.round(pts_s[:, 0] * K[0, 0] / pts_s[:, 2] + K[0, 2]).astype(int)
    v = np.round(pts_s[:, 1] * K[1, 1] / pts_s[:, 2] + K[1, 2]).astype(int)
    assert np.array_equal(saved_s, conf[v, u])

    # (H,W) 广播与 (N,H,W) 等价（同一 conf 逐帧复用）
    three_d = bind_masks_to_world([{0: _square_mask()}], depth, c2w, K,
                                  out_dir=tmp_path, scene_name="sc3",
                                  point_conf=np.broadcast_to(conf, (2, H, W)))
    assert len(np.load(three_d[0].pointconf_world)) == \
        len(np.load(three_d[0].pointcloud_world))

    # 不给 conf → 空 ref 且不落任何 conf 文件；形状不符 → 同样不伪造
    plain = bind_masks_to_world(masks, depth, c2w, K, out_dir=tmp_path,
                                scene_name="sc_plain")
    assert plain[0].pointconf_world == ""
    assert not list(tmp_path.glob("sc_plain*pointconf*"))
    bad = bind_masks_to_world(masks, depth, c2w, K, out_dir=tmp_path,
                              scene_name="sc_bad", point_conf=np.ones((7, 9)))
    assert bad[0].pointconf_world == ""
    assert not list(tmp_path.glob("sc_bad*pointconf*"))


# ============== v6 §20：BA / 正方形 pad 已废止 → 只做纯等比缩放 ==============

def test_bind_masks_to_world_ignores_retired_square_pad_transform():
    """`grid_transform` 形参保留但不再应用 pad 仿射（v6 §20：唯一正确映射 = 纯缩放）。

    旧 v5 行为会把 pad 仿射（`dst = (src + pad) × scale`）真的应用上；把 pad 变换传进来
    与传 `None` 得到的必须是**同一份世界点云**，否则说明 pad 路径又回来了。
    """
    from skill3d.coords import grid_transform_square_padded

    masks = [{0: _square_mask(), 1: _square_mask()}]
    depth = np.full((2, H, W), 3.0)
    c2w = np.stack([np.eye(4)] * 2)
    pad = grid_transform_square_padded((H, W), (H, W))     # 老 artifact 形态
    assert pad["padded_to_square"] is True

    plain = bind_masks_to_world(masks, depth, c2w, K, class_hints=["sofa"])
    same = bind_masks_to_world(masks, depth, c2w, K, class_hints=["sofa"],
                               grid_transform=pad)
    assert same[0].centroid_world == pytest.approx(plain[0].centroid_world)
    assert same[0].visible_frames == plain[0].visible_frames == [0, 1]


def test_bind_masks_to_world_resizes_mask_to_depth_grid():
    """mask 在原始分辨率、深度在预处理网格 → 纯最近邻缩放到深度网格后绑定。"""
    half = np.zeros((H // 2, W // 2), dtype=bool)
    half[100:130, 150:190] = True                  # ×2 缩放后 = _square_mask 的位置
    full = bind_masks_to_world([{0: _square_mask()}], np.full((1, H, W), 3.0),
                               np.eye(4)[None], K)
    resized = bind_masks_to_world([{0: half}], np.full((1, H, W), 3.0),
                                  np.eye(4)[None], K)
    assert resized[0].centroid_world == pytest.approx(full[0].centroid_world)
    assert resized[0].det_conf == full[0].det_conf > 0


# ============ v6 §5.6：question_targeted_fill（逐题补漏的对象来源） ============

class _FakeVLM:
    """假在线 VLM：抽名字问法 → JSON 名表；带图定位问法 → JSON 框表（0-1000 归一）。"""

    def __init__(self, names, boxes_norm=None):
        self.names = list(names)
        self.boxes = list(boxes_norm or [])
        self.calls = 0

    def chat(self, messages, max_tokens=None, seed=None):
        self.calls += 1
        content = messages[0]["content"]
        if isinstance(content, list):               # 带图定位
            import json as _json

            return _json.dumps([{"name": n, "bbox": b}
                                for n, b in zip(self.names, self.boxes)])
        import json as _json

        return _json.dumps(self.names)


def _fake_predictor_any_frame(n_frames=3):
    """允许多帧给框的替身（SAM2 真实 API 支持逐帧框提示；`_fake_predictor` 只允许首帧）。

    每个 obj_id 拿到同一块掩码 → 逐探测帧给出的候选彼此完全重合（由 3D 去重合并）。
    """

    class _P:
        def init_state(self, video_path, **kw):
            import glob as _g
            return {"n": len(_g.glob(str(video_path) + "/*.jpg"))}

        def add_new_points_or_box(self, state, frame_idx, obj_id, box=None, **kw):
            state.setdefault("objs", []).append(obj_id)

        def propagate_in_video(self, state):
            m = np.zeros((H, W), dtype=bool)
            m[30:70, 40:90] = True
            for t in range(state["n"]):
                yield t, list(range(len(state["objs"]))), \
                    [m.astype(np.float32)] * len(state["objs"])

    return _P()


def _supplement_case(tmp_path, existing, point_conf=None):
    from skill3d.segmentation.sam2_tracker import _bind_question_supplement

    frames = _textured_frames(3, seed=7)
    depth = np.full((3, H, W), 3.0)
    c2w = np.stack([np.eye(4)] * 3)
    client = _FakeVLM(["telephone"], [[100, 100, 200, 200]])
    return _bind_question_supplement(
        frames, [0, 1, 2], "What is the distance to the telephone?", existing,
        depth, c2w, K, grid_transform=None, out_dir=tmp_path, scene_name="sup",
        predictor=_fake_predictor_any_frame(3), vlm_client=client, handle=None,
        point_conf=point_conf)


def test_question_supplement_sets_grounding_status_and_track_namespace(tmp_path):
    """逐题补漏的对象 = `question_targeted_fill` + `qtrk*` track（不与清单撞名）。"""
    base = [_inst("obj_0", "chair", [10.0, 10.0, 10.0]),
            _inst("obj_1", "table", [10.0, 10.0, 10.0])]
    conf = np.full((3, H, W), 4.0)
    out, stats, note = _supplement_case(tmp_path, base, point_conf=conf)
    assert len(out) == 1 and "逐题补漏" in note
    o = out[0]
    assert o.grounding_status == "question_targeted_fill"
    assert o.category_name == "telephone"
    assert o.obj_id == "obj_2"                          # scene 内唯一（不覆盖已有 id）
    # 三个探测帧给出 3 条候选、掩码完全重合 → 合并成 1 条 track（qtrk0）
    assert o.track_id == "qtrk0"
    assert o.duplicate_suspect is True                  # 清单曾被高估 → 计数降级信号
    assert o.pointconf_world                            # 逐点 conf 随补漏对象一起落盘
    assert len(np.load(o.pointconf_world)) == len(np.load(o.pointcloud_world))
    assert stats["track_ious"] or stats == {}


def test_question_supplement_skips_targets_already_in_inventory(tmp_path):
    """题面目标物已在清单中 → 不重复绑定（只复用，不新增）。"""
    base = [_inst("obj_0", "chair", [10.0, 10.0, 10.0])]
    first, _, _ = _supplement_case(tmp_path, base)
    assert len(first) == 1
    again, stats, note = _supplement_case(tmp_path, base + first)
    # v6：即使没有新绑对象，也要回传 grounding 证据（题面点名物已确认在清单中）
    assert again == []
    assert stats["grounding"]["attempted"] is True
    assert stats["grounding"]["all_present"] is True
    assert "均已在场景清单中" in note
