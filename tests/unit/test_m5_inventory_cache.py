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
from pathlib import Path

import numpy as np
import pytest

from skill3d.schemas import ObjectRecord
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


def _inst(iid="obj_0", name="chair", centroid=(0.0, 0.0, 3.0), track_id="trk0"):
    return ObjectRecord(obj_id=iid, category_name=name, mask_per_frame="",
                        pointcloud_world="", centroid_world=list(centroid),
                        bbox=[0.0] * 6, det_conf=0.87,
                        det_conf_source="groundingdino",
                        mask_support_ratio=0.5,
                        binding_flags=[{"code": "test_flag"}],
                        track_id=track_id)


# ----------------------------------------------------------------- 缓存 ----

def _generation(*, fsh="fsh", runtime="a"):
    from skill3d.infra.m5_identity import generation_identity
    return generation_identity(
        scene_name="sc", frame_set_hash=fsh, frames=_frames(2),
        depth_maps=np.ones((2, H, W)), c2w_list=np.stack([np.eye(4)] * 2),
        sam2={"checkpoint_sha256": runtime}, model={"enabled": False},
        detector={"enabled": False}, sources={"test": {"sha256": "a" * 64}})


def _key(generation):
    return _inventory_key("sc", "fsh", None, 2, generation=generation)


def test_inventory_cache_roundtrip(tmp_path):
    objs = [_inst()]
    generation = _generation()
    key = _key(generation)
    _save_inventory(tmp_path, "sc", key, objs, {"track_ious": [0.5], "dynamic_ratio": 0.1},
                    None, generation=generation)
    loaded = _load_inventory(tmp_path, "sc", key, generation=generation)
    assert loaded is not None
    got, stats, g7 = loaded
    assert [o.obj_id for o in got] == ["obj_0"]
    assert got[0].track_id == "trk0" and got[0].category_name == "chair"
    assert got[0].det_conf == pytest.approx(0.87)
    assert got[0].det_conf_source == "groundingdino"
    assert got[0].mask_support_ratio == pytest.approx(0.5)
    assert got[0].binding_flags == [{"code": "test_flag"}]
    assert stats["track_ious"] == [0.5] and stats["dynamic_ratio"] == 0.1
    assert g7 == ""


def test_inventory_cache_key_depends_on_frame_set_and_scene():
    a = _inventory_key("s1", "h1", np.zeros((2, H, W)), 2)
    b = _inventory_key("s1", "h2", np.zeros((2, H, W)), 2)
    c = _inventory_key("s2", "h1", np.zeros((2, H, W)), 2)
    assert len({a, b, c}) == 3


def test_inventory_cache_invalidated_when_sam2_runtime_changes(tmp_path):
    original = _generation(runtime="extension-absent")
    before = _key(original)
    _save_inventory(tmp_path, "sc", before, [_inst()], {}, None, generation=original)
    assert _load_inventory(tmp_path, "sc", before, generation=original) is not None
    for runtime in ("extension-a", "extension-b", "new-package"):
        changed = _generation(runtime=runtime)
        key = _key(changed)
        assert key != before
        assert _load_inventory(tmp_path, "sc", key, generation=changed) is None
    assert _load_inventory(tmp_path, "sc", before, generation=original) is not None


def test_inventory_cache_miss_when_mask_ref_missing(tmp_path):
    """缓存里的 mask ref 不存在（目录被清理/半写入）→ 视为未命中，不得半用。"""
    objs = [_inst()]
    generation = _generation()
    key = _key(generation)
    _save_inventory(tmp_path, "sc", key, objs, {}, None, generation=generation)
    p = tmp_path / f"sc_inventory_{key}.json"
    payload = json.loads(p.read_text())
    payload["objects"][0]["mask_per_frame"] = str(tmp_path / "gone.npy")
    p.write_text(json.dumps(payload))
    assert _load_inventory(tmp_path, "sc", key, generation=generation) is None


def test_inventory_cache_miss_when_pointconf_ref_missing(tmp_path):
    """逐点 conf ref 丢失 → 同样视为未命中（conf 与点云必须成对可用，§12.2）。"""
    o = _inst()
    o.pointconf_world = str(tmp_path / "conf.npy")
    np.save(o.pointconf_world, np.ones(8))
    generation = _generation()
    key = _key(generation)
    _save_inventory(tmp_path, "sc", key, [o], {}, None, generation=generation)
    assert _load_inventory(tmp_path, "sc", key, generation=generation) is not None
    (tmp_path / "conf.npy").unlink()
    assert _load_inventory(tmp_path, "sc", key, generation=generation) is None


def test_inventory_cache_corrupt_json_is_a_miss(tmp_path):
    generation = _generation()
    key = _key(generation)
    (tmp_path / f"sc_inventory_{key}.json").write_text("{not json")
    assert _load_inventory(tmp_path, "sc", key, generation=generation) is None


def test_inventory_cache_old_version_is_a_miss(tmp_path):
    generation = _generation()
    key = _key(generation)
    _save_inventory(tmp_path, "sc", key, [_inst()], {}, None, generation=generation)
    p = tmp_path / f"sc_inventory_{key}.json"
    payload = json.loads(p.read_text())
    payload["inventory_version"] = "m5-inventory-7-en"
    p.write_text(json.dumps(payload))
    assert _load_inventory(tmp_path, "sc", key, generation=generation) is None


def test_missing_generation_disables_persistent_cache(tmp_path):
    key = _inventory_key("sc", "fsh", np.ones((2, H, W)), 2)
    assert _save_inventory(tmp_path, "sc", key, [_inst()], {}, None) == ""
    assert not list(tmp_path.glob("*inventory*.json"))
    assert _load_inventory(tmp_path, "sc", key) is None


def test_same_shape_corrupted_binary_is_cache_miss(tmp_path):
    generation = _generation()
    key = _key(generation)
    obj = _inst()
    obj.mask_per_frame = str(tmp_path / "mask.npy")
    np.save(obj.mask_per_frame, np.ones((2, H, W), dtype=bool))
    _save_inventory(tmp_path, "sc", key, [obj], {}, None, generation=generation)
    assert _load_inventory(tmp_path, "sc", key, generation=generation) is not None
    np.save(obj.mask_per_frame, np.zeros((2, H, W), dtype=bool))
    assert _load_inventory(tmp_path, "sc", key, generation=generation) is None


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


def test_inventory_does_not_repeat_detector_requests_for_empty_results(monkeypatch):
    from skill3d.segmentation import open_vocab_detector as ovd

    calls = {"n": 0}

    class _D:
        label = "chair"
        bbox_xyxy = (10.0, 10.0, 50.0, 50.0)

    def fake_detect(frame, prompt, **kw):
        calls["n"] += 1
        ovd.detect.last_error = ""
        return [] if calls["n"] <= 3 else [_D()]

    monkeypatch.setattr(ovd, "available", lambda: True)
    monkeypatch.setattr(ovd, "detector_endpoint", lambda: "http://x:1")
    monkeypatch.setattr(ovd, "detect", fake_detect)
    notes: list[str] = []
    out = _detector_boxes(_frames(3), [0, 1, 2], "chair", notes)
    assert not out and calls["n"] == 3
    assert any("服务正常" in n for n in notes)


def test_inventory_preserves_earlier_frame_failure_after_healthy_response(monkeypatch):
    from skill3d.segmentation import open_vocab_detector as ovd
    calls = []
    def fake_detect(frame, prompt, **kw):
        ovd.detect.last_error = "" if calls else "ConnectionError: refused"
        calls.append(1)
        return []
    monkeypatch.setattr(ovd, "available", lambda: True)
    monkeypatch.setattr(ovd, "detect", fake_detect)
    notes, stats = [], {}
    _detector_boxes(_frames(2), [0, 1], "chair", notes, stats=stats)
    assert len(calls) == 2
    assert stats["detector_fault"] is True
    assert any("frame=0" in n and "检测器侧降级" in n for n in notes)


# ----------------------------------------------------- 端到端：清单复用 ----

def _fake_predictor(n_frames: int):
    class _P:
        _skill3d_m5_identity = {"fixture_predictor_sha256": "b" * 64}
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


class _InventoryVLM:
    model = "fixture-vlm"
    model_weights_sha256 = "c" * 64
    temperature = 0.0
    max_output_tokens = 512

    def chat(self, messages, **kwargs):
        if isinstance(messages[0]["content"], str):
            return '["chair"]'
        return '[{"name":"chair","bbox":[250,250,650,650]}]'


def test_bind_objects_preserves_groundingdino_confidence(tmp_path, monkeypatch):
    from skill3d.segmentation import open_vocab_detector as ovd

    class _D:
        label = "chair"
        bbox_xyxy = (10.0, 10.0, 50.0, 50.0)
        confidence = 0.87

    def fake_detect(frame, prompt, **kw):
        ovd.detect.last_error = ""
        return [_D()]

    monkeypatch.setattr(ovd, "available", lambda: True)
    monkeypatch.setattr(ovd, "detect", fake_detect)
    frames = _frames(2)
    objects, _, _ = bind_objects_for_scene(
        frames, None, "How many chairs are visible?",
        depth_maps=np.full((2, H, W), 3.0),
        c2w_list=np.stack([np.eye(4)] * 2),
        intrinsics=K, out_dir=tmp_path, scene_name="confidence",
        predictor=_fake_predictor(2), frame_set_hash="fsh",
        use_inventory_cache=False)
    assert objects
    assert all(o.det_conf == pytest.approx(0.87) for o in objects)
    assert all(o.det_conf_source == "groundingdino" for o in objects)
    assert all(o.mask_support_ratio == pytest.approx(1.0) for o in objects)


def test_bind_objects_reuses_scene_inventory_cache(tmp_path, monkeypatch):
    """同一 (scene, frame_set_hash) 第二次调用 → 命中缓存、不重跑传播。"""
    frames, depth = _frames(3), np.full((3, H, W), 3.0)
    c2w = np.stack([np.eye(4)] * 3)
    calls = {"n": 0}
    import skill3d.segmentation.sam2_tracker as st
    from skill3d.segmentation import open_vocab_detector as ovd
    monkeypatch.setattr(ovd, "detector_endpoint", lambda: "")

    orig = st.track_objects

    def counting_track(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)
    st.track_objects = counting_track
    try:
        objs1, stats1, notes1 = bind_objects_for_scene(
            frames, None, "How many chair(s)?", depth_maps=depth, c2w_list=c2w,
            intrinsics=K, out_dir=tmp_path, scene_name="sc",
            predictor=_fake_predictor(3), frame_set_hash="fsh-1",
            vlm_client=_InventoryVLM())
        first_calls = calls["n"]
        objs2, stats2, notes2 = bind_objects_for_scene(
            frames, None, "How many chair(s)?", depth_maps=depth, c2w_list=c2w,
            intrinsics=K, out_dir=tmp_path, scene_name="sc",
            predictor=_fake_predictor(3), frame_set_hash="fsh-1",
            vlm_client=_InventoryVLM())
    finally:
        st.track_objects = orig
    assert first_calls == 1
    assert calls["n"] == first_calls          # 第二次没有重跑传播
    assert any("缓存命中" in n for n in notes2)
    assert [o.obj_id for o in objs1] == [o.obj_id for o in objs2]
    # v6 §5.6：缓存往返后 track 身份与重复嫌疑原样保留（计数口径跨 episode 稳定）
    assert [o.track_id for o in objs1] == [o.track_id for o in objs2]
    assert [o.category_name for o in objs1] == [o.category_name for o in objs2]
    assert all(o.grounding_status == "base_list" for o in objs2)
    assert stats2.get("dynamic_ratio") == pytest.approx(stats1.get("dynamic_ratio"))
    # §7.1 track_consensus 的占比输入跨 episode 一致（同一份清单 → 同一证据）
    assert stats2.get("track_stable_ratio") == pytest.approx(
        stats1.get("track_stable_ratio"))


def test_no_base_boxes_never_borrow_question_into_shared_inventory(tmp_path, monkeypatch):
    from skill3d.segmentation import open_vocab_detector as ovd
    monkeypatch.setattr(ovd, "detector_endpoint", lambda: "")
    objects, stats, _ = bind_objects_for_scene(
        _frames(2), None, "How many chair(s)?",
        depth_maps=np.ones((2, H, W)), c2w_list=np.stack([np.eye(4)] * 2),
        intrinsics=K, out_dir=tmp_path, predictor=_fake_predictor(2))
    assert objects == []
    assert stats["base_inventory"]["n_propagations"] == 0
    assert stats["grounding"]["unresolved_names"] == ["chair"]


def test_bind_objects_different_frame_set_does_not_reuse(tmp_path, monkeypatch):
    frames, depth = _frames(3), np.full((3, H, W), 3.0)
    c2w = np.stack([np.eye(4)] * 3)
    calls = {"n": 0}
    import skill3d.segmentation.sam2_tracker as st
    from skill3d.segmentation import open_vocab_detector as ovd
    monkeypatch.setattr(ovd, "detector_endpoint", lambda: "")

    orig = st.track_objects
    saved_objects = []

    def counting_track(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    st.track_objects = counting_track
    try:
        for fsh in ("fsh-1", "fsh-2"):
            objects, _, _ = bind_objects_for_scene(
                frames, None, "How many chair(s) are here?", depth_maps=depth, c2w_list=c2w,
                intrinsics=K, out_dir=tmp_path, scene_name="sc",
                predictor=_fake_predictor(3), frame_set_hash=fsh,
                vlm_client=_InventoryVLM())
            saved_objects.append([(o.mask_per_frame, Path(o.mask_per_frame).read_bytes())
                                  for o in objects])
    finally:
        st.track_objects = orig
    assert calls["n"] == 2                    # 换 FrameSet → 必须重算（不得混用）
    assert {p for p, _ in saved_objects[0]}.isdisjoint(p for p, _ in saved_objects[1])
    assert all(Path(p).read_bytes() == content for rows in saved_objects for p, content in rows)


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
                                   use_inventory_cache=False, vlm_client=_InventoryVLM())
    finally:
        st.track_objects = orig
    assert calls["n"] == 2
