"""M5 场景清单：缓存复用 + 检测器降级出声（2026-09-21 真实缺陷回归）。

背景（outer_holdout 实测）：
- 检测器服务偶发不可用时，`ovd.detect` 静默返回 `[]`，调用方只在"新增了框"时记
  note → 对象清单悄悄塌缩成"只有 VLM 框"（同一 scene 一题只剩 2 个对象），
  日志里毫无痕迹，程序路径那几题直接不可答；
- 对象清单本是 **scene 级产物**，旧实现每个 episode 重跑一遍 VLM+检测器+SAM2
  （~1.5 min/题），既是扩大样本的瓶颈，也让同一 scene 的不同 episode 看到不同的
  对象清单（id 不稳定）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from skill3d.schemas import ObjectInstance
from skill3d.segmentation.sam2_tracker import (
    _detector_boxes,
    _inventory_key,
    _load_inventory,
    _save_inventory,
    bind_objects_for_scene,
)

H, W = 120, 160
K = np.array([[100.0, 0.0, W / 2], [0.0, 100.0, H / 2], [0.0, 0.0, 1.0]])


def _frames(n: int):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(n)]


def _inst(iid="obj_0", hint="chair", centroid=(0.0, 0.0, 3.0)):
    return ObjectInstance(instance_id=iid, class_hint=hint, mask_per_frame="",
                          pointcloud_world="", centroid_world=list(centroid),
                          bbox=[0.0] * 6, confidence=1.0)


# ----------------------------------------------------------------- 缓存 ----

def test_inventory_cache_roundtrip(tmp_path):
    objs = [_inst()]
    key = _inventory_key("sc", "fsh1", np.zeros((2, H, W)), 2)
    _save_inventory(tmp_path, "sc", key, objs, {"track_ious": [0.5], "dynamic_ratio": 0.1},
                    None)
    loaded = _load_inventory(tmp_path, "sc", key)
    assert loaded is not None
    got, stats, g7 = loaded
    assert [o.instance_id for o in got] == ["obj_0"]
    assert stats["track_ious"] == [0.5] and stats["dynamic_ratio"] == 0.1
    assert g7 == ""


def test_inventory_cache_key_depends_on_frame_set_and_scene():
    a = _inventory_key("s1", "h1", np.zeros((2, H, W)), 2)
    b = _inventory_key("s1", "h2", np.zeros((2, H, W)), 2)
    c = _inventory_key("s2", "h1", np.zeros((2, H, W)), 2)
    assert len({a, b, c}) == 3


def test_inventory_cache_miss_when_mask_ref_missing(tmp_path):
    """缓存里的 mask ref 不存在（目录被清理/半写入）→ 视为未命中，不得半用。"""
    objs = [_inst()]
    key = _inventory_key("sc", "fsh", np.zeros((2, H, W)), 2)
    _save_inventory(tmp_path, "sc", key, objs, {}, None)
    p = tmp_path / f"sc_inventory_{key}.json"
    payload = json.loads(p.read_text())
    payload["objects"][0]["mask_per_frame"] = str(tmp_path / "gone.npy")
    p.write_text(json.dumps(payload))
    assert _load_inventory(tmp_path, "sc", key) is None


def test_inventory_cache_corrupt_json_is_a_miss(tmp_path):
    key = _inventory_key("sc", "fsh", np.zeros((2, H, W)), 2)
    (tmp_path / f"sc_inventory_{key}.json").write_text("{not json")
    assert _load_inventory(tmp_path, "sc", key) is None


# ------------------------------------------------------ 检测器降级要出声 ----

def test_detector_failure_is_reported_not_silent(monkeypatch):
    """检测器整轮零检出 + last_error → 必须记"检测器侧降级"，不得静默。"""
    from skill3d.segmentation import open_vocab_detector as ovd

    def fake_detect(frame, prompt, **kw):
        ovd.detect.last_error = "ConnectionError: refused"
        return []

    monkeypatch.setattr(ovd, "available", lambda: True)
    monkeypatch.setattr(ovd, "detector_endpoint", lambda: "http://x:1")
    monkeypatch.setattr(ovd, "detect", fake_detect)
    notes: list[str] = []
    assert _detector_boxes(_frames(3), [0, 1, 2], "chair", notes) == []
    assert any("检测器侧降级" in n and "ConnectionError" in n for n in notes)


def test_detector_healthy_but_no_detection_is_not_blamed(monkeypatch):
    """服务正常但确实没检出 → note 不得误判为服务故障。"""
    from skill3d.segmentation import open_vocab_detector as ovd

    def fake_detect(frame, prompt, **kw):
        ovd.detect.last_error = ""
        return []

    monkeypatch.setattr(ovd, "available", lambda: True)
    monkeypatch.setattr(ovd, "detector_endpoint", lambda: "http://x:1")
    monkeypatch.setattr(ovd, "detect", fake_detect)
    notes: list[str] = []
    assert _detector_boxes(_frames(3), [0, 1, 2], "chair", notes) == []
    assert any("服务正常" in n for n in notes)
    assert not any("降级" in n for n in notes)


def test_detector_retry_recovers(monkeypatch):
    from skill3d.segmentation import open_vocab_detector as ovd

    calls = {"n": 0}

    class _D:
        label = "chair"
        bbox_xyxy = (10.0, 10.0, 50.0, 50.0)

    def fake_detect(frame, prompt, **kw):
        calls["n"] += 1
        return [] if calls["n"] <= 3 else [_D()]

    monkeypatch.setattr(ovd, "available", lambda: True)
    monkeypatch.setattr(ovd, "detector_endpoint", lambda: "http://x:1")
    monkeypatch.setattr(ovd, "detect", fake_detect)
    notes: list[str] = []
    out = _detector_boxes(_frames(3), [0, 1, 2], "chair", notes)
    assert out and out[0][1] == "chair"
    assert any("重试后恢复" in n for n in notes)


# ----------------------------------------------------- 端到端：清单复用 ----

def _fake_predictor(n_frames: int):
    class _P:
        def init_state(self, path, **kw):
            return {"n": n_frames}

        def add_new_points_or_box(self, state, *, frame_idx, obj_id, box):
            state.setdefault("objs", set()).add(obj_id)

        def propagate_in_video(self, state):
            h, w = H, W
            for t in range(n_frames):
                for oid in sorted(state.get("objs", ())):
                    m = np.zeros((h, w), dtype=bool)
                    m[30:70, 40:90] = True
                    yield t, [oid], m[None].astype(np.float32)

    return _P()


def test_bind_objects_reuses_scene_inventory_cache(tmp_path):
    """同一 (scene, frame_set_hash) 第二次调用 → 命中缓存、不重跑传播。"""
    frames, depth = _frames(3), np.full((3, H, W), 3.0)
    c2w = np.stack([np.eye(4)] * 3)
    calls = {"n": 0}
    import skill3d.segmentation.sam2_tracker as st

    orig = st.track_objects

    def counting_track(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)
    st.track_objects = counting_track
    try:
        objs1, stats1, notes1 = bind_objects_for_scene(
            frames, None, "How many chair(s)?", depth_maps=depth, c2w_list=c2w,
            intrinsics=K, out_dir=tmp_path, scene_name="sc",
            predictor=_fake_predictor(3), frame_set_hash="fsh-1")
        first_calls = calls["n"]
        objs2, stats2, notes2 = bind_objects_for_scene(
            frames, None, "How many chair(s)?", depth_maps=depth, c2w_list=c2w,
            intrinsics=K, out_dir=tmp_path, scene_name="sc",
            predictor=_fake_predictor(3), frame_set_hash="fsh-1")
    finally:
        st.track_objects = orig
    assert first_calls == 1
    assert calls["n"] == first_calls          # 第二次没有重跑传播
    assert any("缓存命中" in n for n in notes2)
    assert [o.instance_id for o in objs1] == [o.instance_id for o in objs2]
    assert stats2.get("dynamic_ratio") == pytest.approx(stats1.get("dynamic_ratio"))


def test_bind_objects_different_frame_set_does_not_reuse(tmp_path):
    frames, depth = _frames(3), np.full((3, H, W), 3.0)
    c2w = np.stack([np.eye(4)] * 3)
    calls = {"n": 0}
    import skill3d.segmentation.sam2_tracker as st

    orig = st.track_objects

    def counting_track(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    st.track_objects = counting_track
    try:
        for fsh in ("fsh-1", "fsh-2"):
            bind_objects_for_scene(frames, None, "How many chair(s) are here?", depth_maps=depth, c2w_list=c2w,
                                   intrinsics=K, out_dir=tmp_path, scene_name="sc",
                                   predictor=_fake_predictor(3), frame_set_hash=fsh)
    finally:
        st.track_objects = orig
    assert calls["n"] == 2                    # 换 FrameSet → 必须重算（不得混用）


def test_bind_objects_cache_can_be_disabled(tmp_path):
    frames, depth = _frames(3), np.full((3, H, W), 3.0)
    c2w = np.stack([np.eye(4)] * 3)
    calls = {"n": 0}
    import skill3d.segmentation.sam2_tracker as st

    orig = st.track_objects

    def counting_track(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    st.track_objects = counting_track
    try:
        for _ in range(2):
            bind_objects_for_scene(frames, None, "How many chair(s) are here?", depth_maps=depth, c2w_list=c2w,
                                   intrinsics=K, out_dir=tmp_path, scene_name="sc",
                                   predictor=_fake_predictor(3), frame_set_hash="f",
                                   use_inventory_cache=False)
    finally:
        st.track_objects = orig
    assert calls["n"] == 2
