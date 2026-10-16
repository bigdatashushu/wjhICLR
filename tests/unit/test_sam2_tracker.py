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
    resolve_sam2_config,
    track_objects,
)

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
H, W = 480, 640


class _InventoryVLM:
    model = "fixture-vlm"
    model_weights_sha256 = "c" * 64
    temperature = 0.0
    max_output_tokens = 512

    def chat(self, messages, **kwargs):
        if isinstance(messages[0]["content"], str):
            return '["chair"]'
        return '[{"name":"chair","bbox":[250,250,650,650]}]'


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

def _fake_predictor(n_frames=3):
    class _P:
        def init_state(self, video_path, **kw):
            # 真实 API：init_state 收 MP4/JPEG 目录路径；此处按目录内 JPEG 数计帧数
            import glob as _g
            n = len(_g.glob(str(video_path) + "/*.jpg"))
            return {"n": n}

        def add_new_points_or_box(self, state, frame_idx, obj_id, box=None, **kw):
            assert 0 <= frame_idx < state["n"], "冻结帧内给框提示"
            state.setdefault("objs", []).append(obj_id)

        def propagate_in_video(self, state, reverse=False):
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
    with pytest.raises(RuntimeError, match="detector box prompts"):
        track_objects(frames, None, _fake_predictor(1))
    with pytest.raises(RuntimeError, match="detector box prompts"):
        track_objects(frames, None, _fake_predictor(1), question="irrelevant text")


def test_image_predictor_encodes_frame_once_for_all_boxes():
    from skill3d.segmentation.sam2_tracker import segment_frame_boxes

    class Predictor:
        def __init__(self):
            self.set_calls = 0
            self.boxes = []

        def set_image(self, frame):
            self.set_calls += 1
            self.shape = frame.shape[:2]

        def predict(self, *, box, multimask_output):
            assert not multimask_output
            self.boxes.append(np.asarray(box).tolist())
            mask = np.zeros((1, *self.shape), dtype=bool)
            x0, y0, x1, y1 = map(int, box)
            mask[0, y0:y1, x0:x1] = True
            return mask, np.array([1.0]), np.zeros_like(mask, dtype=float)

    predictor = Predictor()
    masks = segment_frame_boxes(
        np.zeros((32, 48, 3), dtype=np.uint8),
        [[1, 2, 8, 9], [10, 11, 20, 24]], predictor)
    assert predictor.set_calls == 1
    assert len(predictor.boxes) == len(masks) == 2
    assert masks[0][2:9, 1:8].all() and masks[1][11:24, 10:20].all()


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
    assert o.category_name == "sofa" and o.det_conf == 0.0
    assert o.det_conf_source == "unknown"
    assert o.mask_support_ratio == pytest.approx(1.0)
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


def test_bind_masks_to_world_separates_detector_confidence_from_mask_support():
    empty = np.zeros((H, W), dtype=bool)
    masks = [{0: _square_mask(), 1: empty}]
    depth = np.full((2, H, W), 3.0)
    objs = bind_masks_to_world(
        masks, depth, np.stack([np.eye(4)] * 2), K,
        class_hints=["telephone"], confidences=[0.87],
        det_conf_sources=["groundingdino"])
    o = objs[0]
    assert o.det_conf == pytest.approx(0.87)
    assert o.det_conf_source == "groundingdino"
    assert o.visible_frames == [0]
    assert o.mask_support_ratio == pytest.approx(0.5)


def test_bind_masks_to_world_vlm_box_has_zero_detector_confidence():
    masks = [{0: _square_mask()}]
    objs = bind_masks_to_world(
        masks, np.full((1, H, W), 3.0), np.eye(4)[None], K,
        class_hints=["telephone"], confidences=[0.0],
        det_conf_sources=["vlm_box"])
    assert objs[0].det_conf == 0.0
    assert objs[0].det_conf_source == "vlm_box"
    assert objs[0].mask_support_ratio == pytest.approx(1.0)


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


def test_bind_masks_to_world_persists_mask_at_video_resolution(tmp_path):
    """回归：SAM2 掩码在**视频分辨率**、深度在**VGGT 网格**时，落盘 mask 不得为全 0。

    2026-09-22 实测缺陷：`_stack_masks` 用 `np.asarray(m).shape == out.shape[1:]`
    做守卫，尺寸不等就**静默跳过** → 每个对象的 `*_mask.npy` 落盘成整片 0
    （真实 corpus 7 个 scene、全部 54 个对象的 mask 数组 sum 均为 0），
    而 `visible_frames` 由未缩放的 mask 算出、依然非空 → 落盘产物与清单自相矛盾。
    旧测试没抓到是因为它让 mask 与 depth **同尺寸**（都 (H,W)），
    恰好绕过缩放路径；真实链路里两者从不同尺寸。本测试固定视频分辨率 ≠ 深度网格。
    """
    # 掩码 960×1280（视频分辨率），深度网格 480×640 → 必须走等比缩放
    vid_h, vid_w = 2 * H, 2 * W
    vid_mask = np.zeros((vid_h, vid_w), dtype=bool)
    vid_mask[400:520, 600:760] = True          # 对应 _square_mask 的区域
    masks = [{t: vid_mask.copy() for t in range(2)}]
    depth = np.full((2, H, W), 3.0)
    objs = bind_masks_to_world(masks, depth, np.broadcast_to(np.eye(4), (2, 4, 4)), K,
                               class_hints=["sofa"], out_dir=tmp_path, scene_name="sc")
    o = objs[0]
    saved = np.load(o.mask_per_frame)
    assert saved.shape == (2, H, W)
    # 缺陷本体：非空掩码被静默丢成 0
    assert saved.sum() > 0, "落盘 mask 全 0：掩码未按 _resize_mask_to 映射到深度网格"
    # 与清单自洽：visible_frames 说可见的帧，落盘 mask 必须真的非空
    assert o.visible_frames == [0, 1]
    for f in o.visible_frames:
        assert saved[f].any(), f"frame {f} 在 visible_frames 里但落盘 mask 为空"


# ------------------------------------------------------------------ M5 端到端 ----

def _inst(iid, hint, centroid, conf=0.9):
    from skill3d.schemas import ObjectRecord

    return ObjectRecord(obj_id=iid, category_name=hint, mask_per_frame="",
                        pointcloud_world="", centroid_world=list(centroid),
                        bbox=[0.0] * 6, det_conf=conf,
                        visible_frames=[0], mask_support_ratio=1.0)


def _targeted_with_mask(tmp_path, qa_id, category, mask, obj_id="obj_0"):
    obj = _inst(obj_id, category, [0.0, 0.0, 2.0])
    path = tmp_path / f"{qa_id}_{obj_id}_{category}.npy"
    np.save(path, np.asarray(mask, dtype=np.uint8))
    return obj.model_copy(update={
        "grounding_status": "question_targeted_fill",
        "mask_per_frame": str(path),
    })


def test_semantic_mask_conflict_marks_both_cross_question_objects(tmp_path):
    from skill3d.segmentation.sam2_tracker import audit_semantic_mask_conflicts

    mask = np.zeros((3, 8, 8), dtype=bool)
    mask[:, 2:6, 2:6] = True
    tv = _targeted_with_mask(tmp_path, "2241", "tv", mask)
    phone = _targeted_with_mask(tmp_path, "2242", "telephone", mask)
    audited, audits = audit_semantic_mask_conflicts(
        [("2241", [tv]), ("2242", [phone])], scene_identity="scene-frozen")
    assert len(audits) == 1
    audit = audits[0]
    assert audit["stacked_iou"] == pytest.approx(1.0)
    assert audit["common_nonempty_frames"] == [0, 1, 2]
    assert set(audit["qa_ids"]) == {"2241", "2242"}
    left = audited["2241"][0].binding_flags
    right = audited["2242"][0].binding_flags
    assert left[0]["audit_sha256"] == right[0]["audit_sha256"]
    assert left[0]["code"] == "semantic_label_mask_conflict"


def test_semantic_mask_conflict_excludes_synonyms_and_requires_two_frames(tmp_path):
    from skill3d.segmentation.sam2_tracker import audit_semantic_mask_conflicts

    two = np.zeros((2, 8, 8), dtype=bool)
    two[:, 1:5, 1:5] = True
    tv = _targeted_with_mask(tmp_path, "a", "tv", two)
    television = _targeted_with_mask(tmp_path, "b", "television", two)
    audited, audits = audit_semantic_mask_conflicts(
        [("a", [tv]), ("b", [television])], scene_identity="scene")
    assert audits == []
    assert not audited["a"][0].binding_flags

    one = two.copy()
    one[1] = False
    phone = _targeted_with_mask(tmp_path, "c", "telephone", one)
    _, audits = audit_semantic_mask_conflicts(
        [("a", [tv]), ("c", [phone])], scene_identity="scene")
    assert len(audits) == 1
    assert audits[0]["status"] == "skipped"
    assert audits[0]["skip_reason"] == "insufficient_common_nonempty_frames"


@pytest.mark.parametrize("left,right,relation,hard", [
    ("door", "chair", "incompatible", True),
    ("door", "drawer", "incompatible", True),
    ("desk", "table", "compatible", False),
    ("water bottle", "bottle", "compatible", False),
    ("computer", "tv monitor", "compatible", False),
    ("desk counter", "dresser", "compatible", False),
    ("cabinet door", "chair", "unknown", False),
    ("lamp", "chair", "unknown", False),
])
def test_mask_audit_covers_base_and_targeted_without_blanket_block(
        tmp_path, left, right, relation, hard):
    from skill3d.segmentation.sam2_tracker import audit_semantic_mask_conflicts

    mask = np.ones((2, 8, 8), dtype=np.uint8)
    base = _targeted_with_mask(tmp_path, "base", left, mask).model_copy(
        update={"grounding_status": "base_list"})
    target = _targeted_with_mask(tmp_path, "target", right, mask, "obj_1")
    audited, audits = audit_semantic_mask_conflicts(
        [("q", [base, target])], scene_identity="scene")
    assert len(audits) == 1
    assert audits[0]["relation"] == relation
    assert audits[0]["hard_block"] == hard
    assert all(bool(o.binding_flags) == hard for o in audited["q"])
    assert len(audited["q"]) == 2


def test_base_aliases_receive_conflict_flags_once_and_foreign_flags_survive(tmp_path):
    from skill3d.segmentation.sam2_tracker import audit_semantic_mask_conflicts

    mask = np.ones((2, 8, 8), dtype=np.uint8)
    base = _targeted_with_mask(tmp_path, "base", "door", mask).model_copy(
        update={"grounding_status": "base_list", "binding_flags": [{"code": "other"}]})
    copied = base.model_copy(update={"obj_id": "copy"})
    target = _targeted_with_mask(tmp_path, "target", "chair", mask)
    audited, audits = audit_semantic_mask_conflicts(
        [("a", [base]), ("b", [copied, target])], scene_identity="scene")
    assert len(audits) == 1
    assert len(audits[0]["endpoints"][0]["aliases"]) == 2
    assert audited["a"][0].binding_flags[0] == {"code": "other"}
    assert audited["b"][0].binding_flags[-1]["code"] == "semantic_label_mask_conflict"
    again, repeated = audit_semantic_mask_conflicts(
        list(audited.items()), scene_identity="scene")
    assert repeated == audits
    assert again["a"][0].binding_flags == audited["a"][0].binding_flags
    _, base_only = audit_semantic_mask_conflicts(
        [("q", [base, target.model_copy(update={"grounding_status": "base_list"})])],
        scene_identity="scene")
    assert base_only[0]["hard_block"] is False


def test_unreadable_and_mismatched_masks_remain_skipped_diagnostics(tmp_path):
    from skill3d.segmentation.sam2_tracker import audit_semantic_mask_conflicts

    left = _targeted_with_mask(tmp_path, "a", "door", np.ones((2, 8, 8)))
    right = _targeted_with_mask(tmp_path, "b", "chair", np.ones((2, 9, 8)))
    missing = _inst("missing", "lamp", [0, 0, 0])
    _, audits = audit_semantic_mask_conflicts(
        [("q", [left, right, missing])], scene_identity="scene")
    assert {a["skip_reason"] for a in audits} == {
        "mask_shape_mismatch", "unreadable_or_invalid_mask"}


def test_semantic_conflict_audit_identity_binds_scene_mask_and_contract(tmp_path):
    from skill3d.segmentation.sam2_tracker import audit_semantic_mask_conflicts

    mask = np.ones((2, 8, 8), dtype=bool)
    tv = _targeted_with_mask(tmp_path, "a", "tv", mask)
    phone = _targeted_with_mask(tmp_path, "b", "telephone", mask)

    def audit_id(scene, left=tv, version="v1"):
        _, rows = audit_semantic_mask_conflicts(
            [("a", [left]), ("b", [phone])],
            scene_identity=scene, contract_version=version)
        return rows[0]["audit_sha256"]

    original = audit_id("scene-1")
    assert audit_id("scene-2") != original
    assert audit_id("scene-1", version="v2") != original
    changed = mask.copy()
    changed[:, 0, 0] = False
    changed_tv = _targeted_with_mask(tmp_path, "changed", "tv", changed)
    assert audit_id("scene-1", left=changed_tv) != original


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
    assert resized[0].det_conf == full[0].det_conf == 0.0
    assert resized[0].mask_support_ratio == full[0].mask_support_ratio == 1.0


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
