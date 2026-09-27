"""episode 级图像账本（§9.4 主动图像的 `produced / delivered / observed` 三态）。

为什么需要它：`ToolResult.payload` 是 JSON，装不下像素；而"模型到底看没看过这张裁剪图"
又不能靠 `result_id` 或文件路径糊过去（§9.4 明文）。于是像素留在本账本里，payload 只带
`image_id`，身份由内容哈希给出。

账本由**框架**持有（`SceneHandle` 内部），模型程序只能经工具间接触发产出：
`SceneHandle` 上的写入口全是下划线私有方法，`ast_guard` 禁止模型访问下划线属性
（"agent 不能直接修改共享状态"，§9.4）。

生命周期：episode 内唯一，随 `_retarget_handle` 一起搬（逐题 scope 重派生不丢图）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from skill3d.schemas.image_ledger import (
    IMAGE_LEDGER_SCHEMA_VERSION,
    LAYOUT_DERIVED_PLUS_ORIGINALS,
    LAYOUT_ORIGINALS_ONLY,
    ProducedImage,
    ImageRound,
    TOKEN_ESTIMATE_SOURCE,
)

# 送入模型的单图提示：模型需要知道"这张图是什么"，否则裁剪图与 ID 无异
DEFAULT_MAX_DERIVED_IMAGES = 8


def estimate_image_tokens(hw: Sequence[int]) -> int:
    """图像 token 估算（Qwen-VL 家族 28×28 patch 口径）。

    真实值以服务返回的 `usage.prompt_tokens` 为准（§9.4"token 成本完整记录"）；
    这里只是**显式标注口径**的估算，便于逐图分摊。
    """
    h, w = (int(hw[0]), int(hw[1])) if hw and len(hw) == 2 else (0, 0)
    if h <= 0 or w <= 0:
        return 0
    return int(np.ceil(h / 28.0) * np.ceil(w / 28.0))


def _digest(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.tobytes()).hexdigest()


@dataclass
class _Entry:
    """账本条目：记录 + 像素（像素不进 JSON）。"""

    record: ProducedImage
    pixels: Optional[np.ndarray] = None


@dataclass
class ImageLedger:
    """episode 级图像账本（产出 / 交付 / 观察三态 + 每轮清单）。"""

    episode_id: str = ""
    max_images: int = 32
    max_derived_images: int = DEFAULT_MAX_DERIVED_IMAGES
    frames: dict[int, np.ndarray] = field(default_factory=dict)
    # 槽位 → 物理源帧号（FrameSet.source_frame_indices）；缺省与槽位相同（合成/图片源）
    source_frame_indices: dict[int, int] = field(default_factory=dict)
    schema_version: str = IMAGE_LEDGER_SCHEMA_VERSION
    _entries: dict[str, _Entry] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    rounds: list[ImageRound] = field(default_factory=list)
    # 当前正在执行的求解轮（runner 每轮设置）：工具产出图像时据此记账
    current_round: int = 0
    _seq: int = 0

    # ---- 冻结帧集（同一 FrameSet，不新采样视频帧；§9.4）----
    def set_frames(self, pixel_ids: Sequence[int], pixels: Sequence[np.ndarray],
                   source_frame_indices: Optional[Sequence[int]] = None) -> None:
        """登记本 episode 的**实际可读**帧（槽位 → 像素 + 槽位 → 物理源帧号）。

        `pixel_ids` 是**帧槽位序号**（0..n-1）：这是模型与其它工具统一使用的指帧方式
        （`reproject(..., frame_idx)` / `object_visible_frames` 都用槽位）。物理源帧号
        （真实视频里是 0/116/233… 这些采样点）单独记，供审计"图来自视频哪一帧"。
        """
        slots = [int(i) for i in pixel_ids]
        self.frames = {i: np.asarray(p) for i, p in zip(slots, pixels)}
        phys = [int(i) for i in (source_frame_indices or [])]
        self.source_frame_indices = ({slot: phys[k] for k, slot in enumerate(slots)}
                                     if len(phys) == len(slots)
                                     else {slot: slot for slot in slots})

    def physical_index_of(self, slot: int) -> Optional[int]:
        """槽位 → 物理源帧号（缺映射时返回槽位本身，不伪造）。"""
        return self.source_frame_indices.get(int(slot), int(slot))

    @property
    def frame_ids(self) -> list[int]:
        return sorted(self.frames)

    def has_frames(self) -> bool:
        return bool(self.frames)

    def frame_pixels(self, frame_id: int) -> np.ndarray:
        if int(frame_id) not in self.frames:
            raise KeyError(f"帧 {frame_id} 不在本 episode 的冻结 FrameSet 中"
                           f"（可读帧={self.frame_ids[:8]}…）")
        return self.frames[int(frame_id)]

    # ---- 产出（§9.4 produced）----
    def produce(self, pixels: np.ndarray, *, kind: str, produced_by: str,
                result_id: str = "", round_index: Optional[int] = None,
                source_frame_id: Optional[int] = None,
                source_frame_index: Optional[int] = None,
                box_xyxy: Optional[Sequence[float]] = None,
                source_hw: Optional[Sequence[int]] = None,
                sent_hw: Optional[Sequence[int]] = None,
                scale: Optional[float] = None) -> ProducedImage:
        """登记一张**真实产出**的图像（像素必须存在）。"""
        arr = np.asarray(pixels)
        if arr.size == 0:
            raise ValueError("produce() 收到空像素：空图不得进入观察链")
        self._seq += 1
        digest = _digest(arr)
        image_id = f"img-{kind}-{self._seq:03d}-{digest[:8]}"
        h, w = int(arr.shape[0]), int(arr.shape[1])
        rec = ProducedImage(
            image_id=image_id, kind=kind, produced_by=str(produced_by),
            result_id=str(result_id),
            produced_round=int(self.current_round if round_index is None else round_index),
            source_frame_id=(None if source_frame_id is None else int(source_frame_id)),
            source_frame_index=(self.physical_index_of(int(source_frame_id))
                                if source_frame_id is not None
                                else (None if source_frame_index is None
                                      else int(source_frame_index))),
            box_xyxy=[float(x) for x in box_xyxy] if box_xyxy else None,
            source_hw=[int(x) for x in (source_hw or (h, w))],
            sent_hw=[int(x) for x in (sent_hw or (h, w))],
            scale=(None if scale is None else float(scale)),
            content_sha256=f"sha256:{digest}",
            n_pixels_sent=h * w,
            token_estimate=estimate_image_tokens((h, w)),
            token_estimate_source=TOKEN_ESTIMATE_SOURCE,
        )
        self._entries[image_id] = _Entry(record=rec, pixels=arr)
        self._order.append(image_id)
        return rec

    # ---- 查询 ----
    def get(self, image_id: str) -> ProducedImage:
        if image_id not in self._entries:
            raise KeyError(f"未知 image_id: {image_id}")
        return self._entries[image_id].record

    def pixels_of(self, image_id: str) -> np.ndarray:
        entry = self._entries.get(image_id)
        if entry is None or entry.pixels is None:
            raise KeyError(f"未知 image_id（或像素已释放）: {image_id}")
        return entry.pixels

    def all_images(self) -> list[ProducedImage]:
        return [self._entries[i].record for i in self._order]

    def images_produced_in_round(self, round_index: int) -> list[ProducedImage]:
        return [r for r in self.all_images() if r.produced_round == int(round_index)]

    def image_ids_for_results(self, result_ids: Iterable[str]) -> list[str]:
        """点名结果 → 它们产出的图像 id（§9.4：让出的结果里带的图才回灌）。

        模型可能点名 **result_id**（prompt 里教的写法）也可能点名它刚在 payload 里
        看到的 **image_id** —— 两种都认，否则"我点名要看这张图"会静默什么都看不到。
        """
        wanted = {str(r) for r in result_ids}
        return [r.image_id for r in self.all_images()
                if r.result_id in wanted or r.image_id in wanted]

    def pending_derived_images(self) -> list[str]:
        """当前轮已产出但**从未交付**的派生图（收口/恢复轮要一并带上）。"""
        return self.undelivered_images_produced_in_round(int(self.current_round))

    def note_undelivered(self, reason: str) -> list[str]:
        """episode 结束时给"产出过但从未交付"的图盖章写明原因（§9.4 不静默丢图）。

        返回被盖章的 image_id。已经交付过（哪怕是更早轮次）的图不动 —— 它们不算丢。
        """
        stamped: list[str] = []
        for rec in self.all_images():
            if rec.delivered_rounds or rec.unobserved_reason:
                continue
            rec.unobserved_reason = str(reason)[:200]
            stamped.append(rec.image_id)
        return stamped

    def undelivered_images_produced_in_round(self, round_index: int) -> list[str]:
        """本轮新产出、**从未交付过**的派生图（让出时点名没写清的兜底，仍然有界）。"""
        return [r.image_id for r in self.all_images()
                if r.kind != "frame" and r.produced_round == int(round_index)
                and not r.delivered_rounds]

    # ---- 每轮清单的布局（§9.4"事先声明的布局或分批观察方案"）----
    def plan_round(self, *, round_index: int, trigger: str,
                   derived_image_ids: Sequence[str],
                   original_frame_ids: Sequence[int]) -> ImageRound:
        """按**声明布局**排本轮实际进请求的图像，并如记录被省略的图。

        `derived_plus_originals_v1`：派生图优先占位（上限 `max_derived_images`），
        其余名额按帧序补原帧；总数不超过 `max_images`（服务上限，实测 8100 为 32）。
        被省略的原帧**已经在本轮之前的请求里交付过**（否则它们不可能不是首次出现），
        被省略的派生图保持未交付并记 `unobserved_reason` —— 两者都不静默消失。
        """
        limit = max(1, int(self.max_images))
        derived = [i for i in derived_image_ids if i in self._entries][
            : max(0, int(self.max_derived_images))]
        originals = [f"frame-{int(fid)}" for fid in original_frame_ids]
        chosen, omitted_derived = list(derived), []
        remaining = max(0, limit - len(chosen))
        chosen_originals = originals[:remaining]
        chosen += chosen_originals
        omitted_originals = originals[remaining:]
        chosen_set = set(chosen)
        omitted_derived = [i for i in derived_image_ids if i not in chosen_set]
        for i in omitted_derived:
            rec = self._entries[i].record
            if not rec.delivered_rounds:
                rec.unobserved_reason = (
                    "本轮图像名额已满（layout=derived_plus_originals_v1），保持 unobserved")
        n_pixels = 0
        tokens = 0
        for i in chosen:
            rec = self._entries[i].record
            n_pixels += int(rec.n_pixels_sent)
            tokens += int(rec.token_estimate)
        existing = next((r for r in self.rounds
                         if r.round_index == int(round_index)), None)
        merged_ids = list(chosen)
        if existing is not None:
            # 同一轮内的**第二次**请求（M9 重生成 / 退化重生成）：并入既有轮记录，
            # 不新开一轮 —— 否则 trace 里会出现两轮同号、图像集合互相矛盾。
            merged_ids = list(dict.fromkeys(list(existing.image_ids) + merged_ids))
        rnd = ImageRound(
            round_index=int(round_index), trigger=str(trigger),
            layout=(LAYOUT_DERIVED_PLUS_ORIGINALS
                    if any(not i.startswith("frame-") for i in merged_ids)
                    else LAYOUT_ORIGINALS_ONLY),
            image_ids=merged_ids,
            original_image_ids=[i for i in merged_ids if i.startswith("frame-")],
            derived_image_ids=[i for i in merged_ids if not i.startswith("frame-")],
            n_images=len(merged_ids), max_images=limit,
            omitted_originals=sorted(set(omitted_originals)
                                     | set(existing.omitted_originals if existing else [])),
            omitted_derived=sorted(set(omitted_derived)
                                   | set(existing.omitted_derived if existing else [])),
            n_pixels_total=n_pixels, token_estimate_total=tokens,
            delivered=bool(existing.delivered) if existing else False,
            observed=bool(existing.observed) if existing else False,
            prompt_tokens=(existing.prompt_tokens if existing else None))
        if existing is not None:
            self.rounds[self.rounds.index(existing)] = rnd
        else:
            self.rounds.append(rnd)
        return rnd

    def frame_image_ids(self, frame_ids: Sequence[int]) -> list[str]:
        """原帧在账本里的 id（原帧也在账本里登记，身份是内容哈希）。"""
        return [f"frame-{int(f)}" for f in frame_ids]

    def ensure_frame_records(self, *, produced_by: str = "frame_set",
                             round_index: int = 0) -> None:
        """为每个可读帧建一条 `kind="frame"` 记录（只建一次；不复制像素）。"""
        for fid in self.frame_ids:
            key = f"frame-{fid}"
            if key in self._entries:
                continue
            arr = self.frames[fid]
            digest = _digest(np.asarray(arr))
            rec = ProducedImage(
                image_id=key, kind="frame", produced_by=produced_by,
                result_id="", produced_round=int(round_index),
                source_frame_id=int(fid),
                source_frame_index=self.physical_index_of(int(fid)),
                source_hw=[int(arr.shape[0]), int(arr.shape[1])],
                sent_hw=[int(arr.shape[0]), int(arr.shape[1])],
                content_sha256=f"sha256:{digest}",
                n_pixels_sent=int(arr.shape[0]) * int(arr.shape[1]),
                token_estimate=estimate_image_tokens((arr.shape[0], arr.shape[1])),
            )
            self._entries[key] = _Entry(record=rec, pixels=np.asarray(arr))
            self._order.append(key)

    # ---- 三态迁移 ----
    def mark_delivered(self, round_index: int, image_ids: Iterable[str]) -> None:
        """请求**发出前**标记：这些图进了本次请求（§9.4 delivered = 进入请求）。"""
        for i in image_ids:
            entry = self._entries.get(i)
            if entry is None:
                continue
            if int(round_index) not in entry.record.delivered_rounds:
                entry.record.delivered_rounds.append(int(round_index))

    def mark_observed(self, round_index: int, image_ids: Iterable[str]) -> None:
        """请求**返回响应后**标记：模型收到了实际内容并完成了本轮响应。"""
        for i in image_ids:
            entry = self._entries.get(i)
            if entry is None:
                continue
            rec = entry.record
            if int(round_index) not in rec.delivered_rounds:
                # 没进请求就不可能被看到 → 不标记（validator 也会拦）
                continue
            if int(round_index) not in rec.observed_rounds:
                rec.observed_rounds.append(int(round_index))
            rec.unobserved_reason = ""

    def mark_round_failed(self, round_index: int, note: str) -> None:
        """请求失败：本轮已标记 delivered 的图**保持未观察**，并写明原因。"""
        for rnd in self.rounds:
            if rnd.round_index != int(round_index):
                continue
            rnd.delivered = True
            rnd.observed = False
            for i in rnd.image_ids:
                rec = self._entries[i].record
                if rec.delivered_rounds and not rec.observed_rounds:
                    rec.unobserved_reason = f"请求失败：{note}"[:200]

    def mark_round_sent(self, round_index: int, *, prompt_tokens: Optional[int] = None
                        ) -> None:
        for rnd in self.rounds:
            if rnd.round_index == int(round_index):
                rnd.delivered = True
                if prompt_tokens is not None:
                    rnd.prompt_tokens = int(prompt_tokens)

    def mark_round_observed(self, round_index: int, *,
                            prompt_tokens: Optional[int] = None) -> None:
        self.mark_observed(round_index, self._round_images(round_index))
        for rnd in self.rounds:
            if rnd.round_index == int(round_index):
                rnd.delivered = True
                rnd.observed = True
                if prompt_tokens is not None:
                    rnd.prompt_tokens = int(prompt_tokens)

    def _round_images(self, round_index: int) -> list[str]:
        for rnd in self.rounds:
            if rnd.round_index == int(round_index):
                return list(rnd.image_ids)
        return []

    # ---- 落盘 ----
    def stats(self) -> dict[str, Any]:
        imgs = self.all_images()
        return {
            "n_images_produced": len(imgs),
            "n_images_delivered": sum(1 for r in imgs if r.delivered_rounds),
            "n_images_observed": sum(1 for r in imgs if r.observed_rounds),
            "n_images_unobserved": sum(1 for r in imgs if not r.observed_rounds),
            "n_derived_produced": sum(1 for r in imgs if r.kind != "frame"),
            "n_derived_observed": sum(1 for r in imgs
                                      if r.kind != "frame" and r.observed_rounds),
        }

    def to_dict(self) -> dict[str, Any]:
        """落盘形态：三态 + 每轮清单 + 统计（像素与路径都不进 trace）。"""
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "max_images": int(self.max_images),
            "max_derived_images": int(self.max_derived_images),
            "frame_ids": self.frame_ids,
            "images": [r.model_dump() for r in self.all_images()],
            "rounds": [r.model_dump() for r in self.rounds],
            "stats": self.stats(),
        }


__all__ = [
    "DEFAULT_MAX_DERIVED_IMAGES",
    "ImageLedger",
    "estimate_image_tokens",
]
