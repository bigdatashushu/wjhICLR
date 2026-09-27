"""§9.2/§9.4 主动图像与补检工具：`inspect_frames` / `detect_objects`。

规范原文（§9.2 最小工具面）：

    | `inspect_frames(frame_ids, boxes=None)` | image_2d；允许 degraded |
      从冻结帧集中查看/裁剪真实图片；返回图像引用和变换，**不能新采样视频帧** |
    | `detect_objects(frame_ids, categories)` | image_2d 及健康检测服务 |
      返回候选、置信度和执行状态；**服务健康属于执行前置条件**，不要求已有检测成功 |

规范原文（§9.4）：

    「`inspect_frames` 和 `detect_objects` 必须验证真实生产调用、真实结果与后续模型
    输入；注册表中存在其他几何工具不能代替这两项。新检出结果由框架更新当前 episode
    对象记录、证据版本和工具可用性；**agent 不能直接修改共享状态**。」
    「裁剪必须来自**同一冻结 FrameSet**，并记录源帧、像素范围和变换。…仅输出服务器
    文件路径不算模型已查看图片。」

因此这两个工具只做三件事：读**本 episode 的冻结帧**、产出真实像素并登记进图像账本
（`produced`）、把新检出并入 episode 对象记录。**交付与观察由 runner 记录**（图像真的
进了哪次请求、模型是否返回了响应），工具无权标记 —— 否则"看过图"就成了工具自称。
"""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import numpy as np

from skill3d.schemas import ObjectRecord
from skill3d.schemas.evidence import CAPABILITIES  # noqa: F401 - 词汇表可达性（证据声明校验）
from skill3d.schemas.tool import ToolSpec

from .contract import ArtifactUnavailableError, DomainValueError
from .registry import REGISTRY
from .scene_handle import SceneHandle

# 证据能力名（与 schemas.evidence.CAPABILITIES 同源）
EV_IMAGE_2D = "image_2d"

# 单次调用允许的最大裁剪数与单图最小边长（防止模型用 1×1 图刷轮次）
MAX_BOXES_PER_CALL = 8
MIN_CROP_SIDE = 8
# 送进模型前的派生图最长边上限（§9.4"模型服务的图像数量／像素上限必须涵盖…"）
MAX_DERIVED_SIDE = 640


def _clamp_box(box: Sequence[float], h: int, w: int) -> tuple[int, int, int, int]:
    """裁剪框裁到帧内并取整（越界框不得静默变成空图）。"""
    if len(box) != 4:
        raise ValueError(f"box 必须是 [x0, y0, x1, y1]，收到 {list(box)!r}")
    x0, y0, x1, y1 = (float(v) for v in box)
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    xa = int(max(0, min(round(xa), w - 1)))
    xb = int(max(0, min(round(xb), w - 1)))
    ya = int(max(0, min(round(ya), h - 1)))
    yb = int(max(0, min(round(yb), h - 1)))
    if xb - xa < MIN_CROP_SIDE or yb - ya < MIN_CROP_SIDE:
        raise ValueError(
            f"裁剪框太小或越界：{[xa, ya, xb, yb]}（要求每边 ≥ {MIN_CROP_SIDE} 像素）")
    return xa, ya, xb, yb


def _resize_for_model(img: np.ndarray, max_side: int = MAX_DERIVED_SIDE
                      ) -> tuple[np.ndarray, float]:
    """派生图等比缩放到最长边 ≤ `max_side`（记录缩放系数，§9.4"记录缩放"）。"""
    import cv2

    h, w = int(img.shape[0]), int(img.shape[1])
    longest = max(h, w)
    if longest <= max_side:
        return img, 1.0
    scale = float(max_side) / float(longest)
    out = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                     interpolation=cv2.INTER_AREA)
    return out, scale


@REGISTRY.register(
    ToolSpec(
        name="inspect_frames",
        description=(
            "查看/裁剪**本 episode 冻结帧集**里的真实图片。参数："
            "`frame_ids=[3,7]`（**帧槽位序号 0..31**，即统一 FrameSet 里的第几张；"
            "与其它按帧取值的工具同一口径，**不是**视频物理帧号）；"
            "`boxes=[[x0,y0,x1,y1], ...]`（可选，**像素坐标**，左上原点；"
            "与 frame_ids 一一对应；不传则返回整帧）。"
            "返回 dict：{'images': [{'image_id','frame_id','box_xyxy','source_hw',"
            "'sent_hw','scale'}], 'n_images': k, 'note': ...}。"
            "**裁出来的图会在你让出后的下一轮直接送进模型**（用 "
            "`return YieldObservations([...], '理由')` 让出即可看到），"
            "因此不要用本工具去'猜'像素 —— 图会真的出现在你眼前。"
            "本工具**只读冻结帧集，不会重新采样视频**；越界/过小的框会被拒绝"),
        args_schema_ref="frame_ids:list[int], boxes:list[list[float]]|None=None",
        returns_schema_ref="dict",
        cost_estimate_ms=5.0,
        source_default="real",
        requires_artifacts=["frames"],
        requires_evidence=[EV_IMAGE_2D],
        # §9.2：允许 degraded —— 有图必答的前提就是 image_2d 可用（v6 §5.4 恒 available）
        tolerates_degraded=[EV_IMAGE_2D],
    )
)
def inspect_frames(handle: SceneHandle, frame_ids: Sequence[int],
                   boxes: Optional[Sequence[Sequence[float]]] = None) -> dict:
    """从冻结 FrameSet 取真实像素（整帧或裁剪），登记进图像账本（§9.4 produced）。

    返回的图像**只是产出**：`image_id` 本身不代表模型看过（§9.4 明文"仅路径、ID 或
    成功标记不能算看过图"）。真正的交付/观察由 runner 在发请求与收到响应时记录。
    """
    if not handle.has_frames():
        # 没有可读帧 = frames 产物事实上不可用（不是"这一帧没内容"那种空状态）
        raise ArtifactUnavailableError("inspect_frames", ["frames"],
                                       route=handle.scene_route,
                                       available=handle.available_artifacts)

    ids = [int(x) for x in (frame_ids or [])]
    if not ids:
        raise DomainValueError("inspect_frames", "需要至少一个 frame_id（帧槽位序号）",
                               args={"frame_ids": list(frame_ids or [])})
    if boxes is not None and len(boxes) not in (0, len(ids)):
        raise DomainValueError(
            "inspect_frames",
            f"boxes 数量（{len(boxes)}）必须与 frame_ids（{len(ids)}）一致，"
            "或整体不传（返回整帧）",
            args={"frame_ids": ids, "n_boxes": len(boxes)})
    if len(ids) > MAX_BOXES_PER_CALL * 4:
        raise DomainValueError(
            "inspect_frames", f"单次最多查看 {MAX_BOXES_PER_CALL * 4} 帧，收到 {len(ids)}",
            args={"frame_ids": ids})

    ledger = handle._ledger  # noqa: SLF001 - 框架侧访问（模型撞下划线禁令）
    out_images: list[dict] = []
    for k, fid in enumerate(ids):
        try:
            frame = ledger.frame_pixels(fid)             # 只读冻结帧集
        except KeyError:
            # 越界帧号 = 参数错误（交给恢复层回灌），不是"没有图片"
            raise DomainValueError(
                "inspect_frames",
                f"帧槽位 {fid} 不在本 episode 的冻结帧集里"
                f"（可用槽位 0..{len(ledger.frame_ids) - 1}）",
                args={"frame_ids": ids}) from None
        src_h, src_w = int(frame.shape[0]), int(frame.shape[1])
        box = boxes[k] if (boxes and len(boxes) == len(ids)) else None
        if box is None:
            crop, scale = _resize_for_model(np.asarray(frame))
            rec = ledger.produce(
                crop, kind="crop", produced_by="inspect_frames",
                source_frame_id=fid, box_xyxy=[0.0, 0.0, float(src_w), float(src_h)],
                source_hw=[src_h, src_w], sent_hw=[int(crop.shape[0]), int(crop.shape[1])],
                scale=scale)
        else:
            try:
                x0, y0, x1, y1 = _clamp_box(box, src_h, src_w)
            except ValueError as exc:
                raise DomainValueError(
                    "inspect_frames", f"frame={fid} 的裁剪框非法：{exc}",
                    args={"frame_id": fid, "box": list(box)}) from None
            sub = np.asarray(frame)[y0:y1, x0:x1]
            if sub.size == 0:
                raise DomainValueError(
                    "inspect_frames", f"裁剪结果为空：frame={fid} box={[x0, y0, x1, y1]}",
                    args={"frame_id": fid, "box": [x0, y0, x1, y1]})
            crop, scale = _resize_for_model(sub)
            rec = ledger.produce(
                crop, kind="crop", produced_by="inspect_frames",
                source_frame_id=fid, box_xyxy=[float(x0), float(y0), float(x1), float(y1)],
                source_hw=[int(sub.shape[0]), int(sub.shape[1])],
                sent_hw=[int(crop.shape[0]), int(crop.shape[1])], scale=scale)
        out_images.append({
            "image_id": rec.image_id,
            "frame_id": int(fid),                      # 槽位序号（0..n-1）
            "source_frame_index": rec.source_frame_index,   # 视频物理帧号（审计）
            "box_xyxy": list(rec.box_xyxy or []),
            "source_hw": list(rec.source_hw),
            "sent_hw": list(rec.sent_hw),
            "scale": rec.scale,
            "content_sha256": rec.content_sha256,
            "transform_version": rec.transform_version,
        })
    return {
        "images": out_images,
        "n_images": len(out_images),
        "frame_ids": ids,
        "note": ("这些图已产出（produced）。把它们连同其它结果交给 "
                 "`return YieldObservations([...], '理由')` 让出，下一轮模型请求会"
                 "**真的带上这些图**（受服务图像上限约束，未交付的图会保持 unobserved）"),
    }


@REGISTRY.register(
    ToolSpec(
        name="detect_objects",
        description=(
            "用本地开放词表检测服务在**指定帧**上补检对象（确定性工具，非 LLM）。"
            "参数：`frame_ids=[0,8,16]`（帧槽位序号 0..31），"
            "`categories=['chair','table']`。"
            "返回 dict：{'status': 'ok'|'fault'|'empty', 'detections': "
            "[{'category_name','frame_id','bbox_xyxy','confidence',"
            "'bbox_norm1000'}], 'n_detections': k, 'n_new_objects': m, "
            "'new_object_ids': [...], 'service_healthy': bool, 'error': ''}。"
            "**status 必须看**：'fault' 表示检测服务不可用（此时 detections 为空是"
            "执行故障，不是'这一帧没有该物体'）；'empty' 才是**有效空检出**。"
            "新检出的对象会并入 episode 对象记录，后续的对象清单工具都能看到它们。"
            "检测只按你给的 categories 名词，不会自动扩词"),
        args_schema_ref="frame_ids:list[int], categories:list[str]",
        returns_schema_ref="dict",
        cost_estimate_ms=800.0,
        source_default="real",
        requires_artifacts=["frames"],
        # §9.2：证据上只要 image_2d（**不要求已有检测成功** —— 补检的意义正是补上它）；
        # "服务健康"是执行前置条件，由工具在调用时判定并**如实返回 fault**。
        requires_evidence=[EV_IMAGE_2D],
        tolerates_degraded=[EV_IMAGE_2D],
    )
)
def detect_objects(handle: SceneHandle, frame_ids: Sequence[int],
                   categories: Sequence[str]) -> dict:
    """在指定帧上做开放词表补检，新检出并入 episode 对象记录（§9.1/§9.4）。

    失败语义（§9.1："服务故障、有效空检出…分别记录…故障返回失败，不能以 last_error
    旁路掩盖主结果的成功空列表"）：服务不可用 → `status="fault"` + `error`，
    **不**返回"成功但空"；服务正常但没有该类物体 → `status="empty"`。
    """
    from skill3d.segmentation import open_vocab_detector as ovd

    if not handle.has_frames():
        raise ArtifactUnavailableError("detect_objects", ["frames"],
                                       route=handle.scene_route,
                                       available=handle.available_artifacts)
    ids = [int(x) for x in (frame_ids or [])]
    cats = [str(c).strip() for c in (categories or []) if str(c).strip()]
    if not ids:
        raise DomainValueError("detect_objects",
                               "需要至少一个 frame_id（帧槽位序号）",
                               args={"frame_ids": list(frame_ids or [])})
    if not cats:
        raise DomainValueError("detect_objects",
                               "需要至少一个类别名（categories）",
                               args={"categories": list(categories or [])})

    ledger = handle._ledger  # noqa: SLF001 - 框架侧访问
    prompt = ovd.prompt_from_nouns(cats)
    detections: list[dict] = []
    records: list[ObjectRecord] = []
    errors: list[str] = []
    for fid in ids:
        try:
            frame = ledger.frame_pixels(fid)
        except KeyError:
            raise DomainValueError(
                "detect_objects",
                f"帧槽位 {fid} 不在本 episode 的冻结帧集里"
                f"（可用槽位 0..{len(ledger.frame_ids) - 1}）",
                args={"frame_ids": ids}) from None
        dets = ovd.detect(np.asarray(frame), prompt)
        if not dets and getattr(ovd.detect, "last_error", ""):
            errors.append(f"frame={fid}: {ovd.detect.last_error}")
        h, w = int(frame.shape[0]), int(frame.shape[1])
        for j, d in enumerate(dets):
            cat = str(d.label or cats[0]).strip() or cats[0]
            x0, y0, x1, y1 = (float(v) for v in d.bbox_xyxy)
            detections.append({
                "category_name": cat,
                "frame_id": int(fid),
                "bbox_xyxy": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                "confidence": round(float(d.confidence), 4),
                "bbox_norm1000": [round(x0 / max(w, 1) * 1000, 1),
                                  round(y0 / max(h, 1) * 1000, 1),
                                  round(x1 / max(w, 1) * 1000, 1),
                                  round(y1 / max(h, 1) * 1000, 1)],
            })
            oid = "obj-d" + hashlib.sha256(
                f"{fid}:{cat}:{j}:{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}".encode()
            ).hexdigest()[:10]
            # 只记 2D 检出：**不**把像素框写进 3D `bbox`/质心（那会是伪 3D 主张）；
            # 没有 3D 数据的对象在几何工具里按空点集/维度异常 fail-closed。
            records.append(ObjectRecord(
                obj_id=oid, category_name=cat, visible_frames=[int(fid)],
                track_id="", det_conf=float(d.confidence),
                grounding_status="tool_detection", duplicate_suspect=False,
                centroid_world=[], bbox=[],
                mask_per_frame="",
                pointcloud_world="", pointconf_world=""))

    new_ids = handle._add_detected_objects(records)  # noqa: SLF001 - 框架侧并入
    if errors and not detections:
        status = "fault"
    elif detections:
        status = "ok"
    else:
        status = "empty"
    return {
        "status": status,
        "detections": detections,
        "n_detections": len(detections),
        "n_new_objects": len(new_ids),
        "new_object_ids": new_ids,
        "service_healthy": status != "fault",
        "frame_ids": ids,
        "categories": cats,
        "error": "; ".join(errors)[:400],
        "detector_endpoint_hash": hashlib.sha256(
            str(ovd.detector_endpoint()).encode()).hexdigest()[:12],
        "note": ("detections 为实际检出；status='fault' 表示检测服务不可用（不是空检出）；"
                 "新检出已并入本 episode 对象记录（框架侧完成，agent 不直接改状态）"),
    }


__all__ = ["detect_objects", "inspect_frames"]
