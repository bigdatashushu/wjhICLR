"""v9 P4：§9.4 主动图像与补检的三态、布局与失败语义。

规范原文（§9.4）：

    「`inspect_frames` 和 `detect_objects` 必须验证真实生产调用、真实结果与后续模型
    输入；注册表中存在其他几何工具不能代替这两项。」
    「裁剪必须来自**同一冻结 FrameSet**，并记录源帧、像素范围和变换。」
    「观察链分开记录 `produced / delivered / observed`：产物已生成不代表进入请求；
    模型收到实际内容并完成本轮响应后，才可记为 observed…仅路径、ID 或成功标记不能算
    看过图。超出模型图像／token 服务上限时，使用事先声明的布局或分批观察方案；
    **未交付的材料保持 unobserved**，不静默丢原图或派生图。」

守四件事：工具真的从冻结帧集裁剪（不新采样）、三态不许互相顶替、布局不超服务上限且
记录被省略的图、服务故障与有效空检出分开。
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.tools import REGISTRY
from skill3d.tools.image_ledger import ImageLedger
from skill3d.tools.vision_tools import MAX_DERIVED_SIDE, detect_objects, inspect_frames


def _frames(n: int = 4, size=(40, 60)):
    out = []
    for i in range(n):
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        img[:, :] = (i * 40) % 255
        out.append(img)
    return out


def _handle(n_frames: int = 4, *, max_images: int = 32, max_derived: int = 8):
    """最小句柄桩：只实现 inspect_frames/detect_objects 真正用到的东西。"""
    from skill3d.schemas import SceneState
    from skill3d.schemas.evidence import EvidenceProfile
    from skill3d.tools.scene_handle import SceneHandle

    profile = EvidenceProfile(geometry_3d="unavailable", world_frame="unavailable",
                              metric_scale="unavailable", object_detection="unavailable",
                              track_consensus="unavailable", object_grounding="unavailable")
    state = SceneState(artifact_ref="a", scene_route="fallback_2d_only",
                       evidence_profile=profile, objects=[], summary="s")
    handle = SceneHandle(state)
    handle._set_image_ledger(ImageLedger(episode_id="q1", max_images=max_images,
                                         max_derived_images=max_derived))
    handle._set_frames(list(range(n_frames)), _frames(n_frames),
                       source_frame_indices=[i * 100 for i in range(n_frames)])
    return handle


# ---------------------------------------------------- 工具在册与暴露面 ----

def test_both_active_vision_tools_are_registered():
    """§9.4 明文：注册表里存在其他几何工具**不能代替**这两项。"""
    assert "inspect_frames" in REGISTRY and "detect_objects" in REGISTRY
    for name in ("inspect_frames", "detect_objects"):
        spec = REGISTRY.spec(name)
        assert spec.requires_artifacts == ["frames"]
        assert "image_2d" in spec.requires_evidence
    # §9.2：重建失败（2D-only）时这两个工具必须**仍然**可用（视觉路径不能被收回）
    fallback = set(REGISTRY.names_for_scope("fallback_2d_only"))
    assert {"inspect_frames", "detect_objects"} <= fallback


# ---------------------------------------------------- inspect_frames ----

def test_inspect_frames_crops_from_the_frozen_frame_set():
    """裁剪来自同一冻结 FrameSet，且记录源帧、像素范围、尺寸与变换。"""
    handle = _handle()
    out = inspect_frames(handle, [2], [[5, 6, 35, 36]])
    assert out["n_images"] == 1
    item = out["images"][0]
    assert item["frame_id"] == 2
    assert item["box_xyxy"] == [5.0, 6.0, 35.0, 36.0]
    assert item["source_hw"] == [30, 30]          # 裁剪后的像素尺寸
    assert item["scale"] == 1.0
    assert item["content_sha256"].startswith("sha256:")
    # 槽位 → 物理源帧号（审计"来自视频哪一帧"）
    assert item["source_frame_index"] == 200
    # 真实像素：确认裁的是源帧那一块（帧 2 的像素值是 80）
    ledger = handle._ledger
    assert int(ledger.pixels_of(item["image_id"])[0, 0, 0]) == 80


def test_inspect_frames_refuses_unknown_or_degenerate_boxes():
    """越界/过小的框与不存在的帧号都必须报错，不得静默给一张空图或整帧。

    报的是**域值错误**（`domain_value`）：模型把帧号写错时，恢复层能把"帧槽位 99
    不存在"这条原因回灌给它 —— 而不是崩成 `violation_run` 让模型猜（真实链路上
    踩过：模型写 `inspect_frames([8])` 却拿到一个看不懂的 KeyError）。
    """
    from skill3d.tools.contract import DomainValueError

    handle = _handle()
    with pytest.raises(DomainValueError):
        inspect_frames(handle, [2], [[0, 0, 3, 3]])            # 太小
    with pytest.raises(DomainValueError):
        inspect_frames(handle, [99])                            # 帧不在冻结集里
    with pytest.raises(DomainValueError):
        inspect_frames(handle, [1, 2], [[0, 0, 20, 20]])        # 数量不匹配
    with pytest.raises(DomainValueError):
        inspect_frames(handle, [])


def test_inspect_frames_does_not_resample_the_video():
    """§9.4：不能新采样视频帧 —— 只能读账本里已登记的冻结帧。"""
    handle = _handle(n_frames=3)
    ledger = handle._ledger
    before = list(ledger.frame_ids)
    inspect_frames(handle, [0, 1, 2])
    assert ledger.frame_ids == before                  # 帧集没变
    assert len(ledger.all_images()) == 3 + 3           # 3 原帧记录 + 3 张裁剪


def test_large_crops_are_scaled_and_the_factor_is_recorded():
    """派生图缩放要记录（§9.4"记录缩放"），且不超单图上限。"""
    handle = _handle(n_frames=1)
    handle._ledger.frames[0] = np.zeros((900, 1200, 3), dtype=np.uint8)
    handle._ledger.ensure_frame_records()
    out = inspect_frames(handle, [0])
    item = out["images"][0]
    assert max(item["sent_hw"]) <= MAX_DERIVED_SIDE
    assert 0 < item["scale"] < 1.0
    assert item["sent_hw"] != item["source_hw"]


# ---------------------------------------------------- detect_objects ----

def test_detect_objects_reports_fault_instead_of_empty(monkeypatch):
    """§9.1：服务故障 ≠ 有效空检出；故障必须返回失败状态，不能借 last_error 掩盖。"""
    from skill3d.segmentation import open_vocab_detector as ovd

    def _fail(frame, prompt, **kw):
        _fail.last_error = "ConnectionError: refused"
        return []

    _fail.last_error = ""
    monkeypatch.setattr(ovd, "detect", _fail)
    handle = _handle()
    out = detect_objects(handle, [0], ["chair"])
    assert out["status"] == "fault" and out["service_healthy"] is False
    assert out["n_detections"] == 0 and "ConnectionError" in out["error"]


def test_detect_objects_reports_empty_when_service_is_healthy(monkeypatch):
    """服务正常但没有该类物体 → `empty`（有效空检出），与故障区分。"""
    from skill3d.segmentation import open_vocab_detector as ovd

    def _ok(frame, prompt, **kw):
        _ok.last_error = ""
        return []

    _ok.last_error = ""
    monkeypatch.setattr(ovd, "detect", _ok)
    handle = _handle()
    out = detect_objects(handle, [0, 1], ["chair"])
    assert out["status"] == "empty" and out["service_healthy"] is True
    assert out["n_detections"] == 0


def test_detected_objects_are_merged_by_the_framework(monkeypatch):
    """§9.4：新检出由**框架**并入 episode 对象记录（agent 不直接改共享状态）。"""
    from skill3d.segmentation import open_vocab_detector as ovd

    class _D:
        def __init__(self, label, box, conf):
            self.label, self.bbox_xyxy, self.confidence = label, box, conf

    def _det(frame, prompt, **kw):
        _det.last_error = ""
        return [_D("chair", (1.0, 2.0, 30.0, 40.0), 0.87)]

    _det.last_error = ""
    monkeypatch.setattr(ovd, "detect", _det)
    handle = _handle()
    out = detect_objects(handle, [1], ["chair"])
    assert out["status"] == "ok" and out["n_new_objects"] == 1
    oid = out["new_object_ids"][0]
    rec = handle.get_object(oid)
    assert rec.category_name == "chair"
    assert rec.grounding_status == "tool_detection"
    assert rec.visible_frames == [1]
    # 只记 2D 检出：**没有** 3D 质心/包围盒（不得伪 3D 主张）
    assert rec.centroid_world == [] and rec.bbox == []
    # 后续工具看得见它（工具可用性随框架侧并入而变）
    assert oid in handle.list_objects()


# ---------------------------------------------------- 布局与三态 ----

def test_layout_keeps_the_request_within_the_service_limit():
    """§9.4：声明布局必须把请求压在服务图像上限内，并记录被省略的图。"""
    handle = _handle(n_frames=32, max_images=32, max_derived=8)
    ledger = handle._ledger
    crops = [inspect_frames(handle, [i], [[0, 0, 20, 20]])["images"][0]["image_id"]
             for i in range(10)]
    rnd = ledger.plan_round(round_index=2, trigger="observation",
                            derived_image_ids=crops,
                            original_frame_ids=ledger.frame_ids)
    assert rnd.n_images == 32                                   # = 服务上限
    assert len(rnd.derived_image_ids) == 8                      # 派生图上限
    assert len(rnd.original_image_ids) == 24
    assert len(rnd.omitted_originals) == 8
    assert len(rnd.omitted_derived) == 2                        # 第 9/10 张裁剪没进
    for i in rnd.omitted_derived:
        assert ledger.get(i).unobserved_reason                # 未交付要写明原因
    assert rnd.layout == "derived_plus_originals_v1"
    assert rnd.token_estimate_total > 0


def test_round_plan_is_idempotent_within_the_same_round():
    """同一轮内第二次请求（M9 重生成）并入既有轮记录，不产生两轮同号。"""
    handle = _handle(n_frames=8, max_images=32)
    ledger = handle._ledger
    ledger.plan_round(round_index=1, trigger="initial", derived_image_ids=[],
                      original_frame_ids=ledger.frame_ids)
    ledger.plan_round(round_index=1, trigger="initial", derived_image_ids=[],
                      original_frame_ids=ledger.frame_ids)
    assert len(ledger.rounds) == 1
    assert ledger.rounds[0].layout == "originals_only"


def test_delivery_and_observation_are_recorded_separately():
    """§9.4：produced ≠ delivered ≠ observed；未发出的请求一条都不算交付。"""
    handle = _handle(n_frames=4)
    ledger = handle._ledger
    iid = inspect_frames(handle, [1])["images"][0]["image_id"]
    ledger.plan_round(round_index=1, trigger="observation", derived_image_ids=[iid],
                      original_frame_ids=ledger.frame_ids)
    rec = ledger.get(iid)
    assert rec.delivered_rounds == [] and rec.observed_rounds == []   # 只产出，未交付

    ledger.mark_round_sent(1)
    ledger.mark_delivered(1, ledger.rounds[0].image_ids)
    rec = ledger.get(iid)
    assert rec.delivered_rounds == [1] and rec.observed_rounds == []  # 进了请求，未观察

    ledger.mark_round_observed(1, prompt_tokens=1234)
    rec = ledger.get(iid)
    assert rec.observed_rounds == [1]                                 # 收到响应才算看过
    assert ledger.rounds[0].prompt_tokens == 1234
    assert rec.unobserved_reason == ""


def test_failed_request_leaves_images_unobserved_with_reason():
    """请求失败 → 已标记交付的图保持 unobserved 并写明原因（不静默）。"""
    handle = _handle(n_frames=4)
    ledger = handle._ledger
    iid = inspect_frames(handle, [0])["images"][0]["image_id"]
    ledger.plan_round(round_index=1, trigger="observation", derived_image_ids=[iid],
                      original_frame_ids=ledger.frame_ids)
    ledger.mark_round_sent(1)
    ledger.mark_delivered(1, ledger.rounds[0].image_ids)
    ledger.mark_round_failed(1, "ServiceUnavailable: refused")
    rec = ledger.get(iid)
    assert rec.delivered_rounds == [1] and rec.observed_rounds == []
    assert "请求失败" in rec.unobserved_reason
    assert ledger.rounds[0].delivered is True and ledger.rounds[0].observed is False


def test_ledger_dump_carries_tri_state_and_stats():
    """落盘形态必须含三态、每轮清单与统计（trace 直接读它）。"""
    handle = _handle(n_frames=4)
    ledger = handle._ledger
    iid = inspect_frames(handle, [2], [[0, 0, 20, 20]])["images"][0]["image_id"]
    ledger.plan_round(round_index=1, trigger="observation", derived_image_ids=[iid],
                      original_frame_ids=ledger.frame_ids)
    ledger.mark_round_sent(1)
    ledger.mark_delivered(1, ledger.rounds[0].image_ids)
    ledger.mark_round_observed(1)
    dump = ledger.to_dict()
    assert dump["schema_version"] and dump["frame_ids"] == [0, 1, 2, 3]
    assert dump["stats"]["n_derived_produced"] == 1
    assert dump["stats"]["n_derived_observed"] == 1
    assert dump["rounds"][0]["image_ids"]
    # 原帧也在账本里（身份 = 内容哈希），"仅 ID 不算看过图"
    frames = [i for i in dump["images"] if i["kind"] == "frame"]
    assert len(frames) == 4 and all(i["content_sha256"] for i in frames)


def test_record_validators_reject_inconsistent_tri_state():
    """自洽性 fail-closed：observed ⊆ delivered；派生图必须带源帧与框。"""
    from skill3d.schemas.image_ledger import ProducedImage

    with pytest.raises(Exception):
        ProducedImage(image_id="i", kind="crop", content_sha256="sha256:x",
                      source_frame_id=1, box_xyxy=[0, 0, 5, 5],
                      delivered_rounds=[], observed_rounds=[1])
    with pytest.raises(Exception):
        ProducedImage(image_id="i", kind="crop", content_sha256="sha256:x")  # 缺源帧/框
    with pytest.raises(Exception):
        ProducedImage(image_id="i", kind="frame", content_sha256="")          # 缺内容哈希
