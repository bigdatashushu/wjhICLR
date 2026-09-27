"""§9.4 主动图像记录 Schema：`produced / delivered / observed` 三态。

规范原文（§9.4）：

    「`inspect_frames` 和 `detect_objects` 必须验证真实生产调用、真实结果与后续模型
    输入…裁剪必须来自同一冻结 FrameSet，并记录源帧、像素范围和变换。模型服务的图像
    数量／像素上限必须涵盖原图与派生图；**不得静默丢图**。可使用带明确映射的图像布局
    或分次观察，**实际传入图像、缩放和 token 成本完整记录**。仅输出服务器文件路径
    不算模型已查看图片。」
    「观察链分开记录 `produced / delivered / observed`：产物已生成不代表进入请求；
    模型收到实际内容并完成本轮响应后，才可记为 observed，且这仍不证明模型正确理解。
    仅路径、ID 或成功标记不能算看过图。超出模型图像／token 服务上限时，使用事先声明
    的布局或分批观察方案；**未交付的材料保持 unobserved**，不静默丢原图或派生图。」

三个状态的定义（本模块是唯一事实源）：

| 状态 | 判据 |
|---|---|
| `produced` | 工具真的产出了像素（`inspect_frames` 裁剪 / 原帧复制）；只有路径或 ID 不算 |
| `delivered` | 该图像被装进了**实际发出**的模型请求（请求发出前的最后一刻标记） |
| `observed` | 该请求**返回了响应**（模型收到了实际内容并完成本轮响应）——仍不证明它理解正确 |

自洽性由 validator 强制：`observed ⊆ delivered`、`delivered/observed` 轮次非空、
`sent_hw` 与 `source_hw` 均为正、派生图必须带源帧与裁剪框、`content_sha256` 必填
（"仅 ID 不算看过图"，身份必须是内容哈希）。
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import field_validator

from . import Spec

IMAGE_LEDGER_SCHEMA_VERSION = "1.0"

# 图像种类：原帧 / 裁剪 / 拼版（多张派生图合并成一张，用于服务图像上限内交付）
ImageKind = Literal["frame", "crop", "contact_sheet"]

# 本轮图像的布局策略（§9.4"事先声明的布局或分批观察方案"）
# - `originals_only`：只有冻结 FrameSet 的原帧（无派生图时的常态）
# - `derived_plus_originals_v1`：派生图优先占位，其后按帧序补原帧，总数不超过服务上限
LAYOUT_ORIGINALS_ONLY = "originals_only"
LAYOUT_DERIVED_PLUS_ORIGINALS = "derived_plus_originals_v1"
IMAGE_LAYOUTS: tuple[str, ...] = (
    LAYOUT_ORIGINALS_ONLY, LAYOUT_DERIVED_PLUS_ORIGINALS,
)

# 裁剪/缩放变换的版本（§9.4"记录源帧、像素范围和变换"）
IMAGE_TRANSFORM_VERSION = "inspect_frames_v1"

# token 估算口径（真实值以服务返回的 usage.prompt_tokens 为准）
TOKEN_ESTIMATE_SOURCE = "qwen_vl_patch_28"


class ProducedImage(Spec):
    """一张进入观察链的图像的完整身份与三态（§9.4）。"""

    image_id: str
    kind: ImageKind = "crop"
    # 产出它的工具与结果（可回溯到 ToolResult）
    produced_by: str = ""
    result_id: str = ""
    produced_round: int = 0
    # 源帧与裁剪（§9.4"记录源帧、像素范围和变换"）
    # `source_frame_id` = **帧槽位序号**（0..n-1，模型与其它工具统一用它指帧）
    source_frame_id: Optional[int] = None
    # 物理源帧号（FrameSet.source_frame_indices[槽位]；真实视频里 32 个均匀采样点，
    # 例如 0/116/233…）—— 审计"这张图来自视频的哪一帧"用它，不用槽位。
    source_frame_index: Optional[int] = None
    box_xyxy: Optional[list[float]] = None      # 像素坐标（左上原点），全帧时为空
    source_hw: list[int] = []                   # 源帧尺寸 [H, W]
    sent_hw: list[int] = []                     # 实际送进请求的尺寸 [H, W]（含缩放）
    scale: Optional[float] = None               # 相对源帧的线性缩放（派生图用）
    transform_version: str = IMAGE_TRANSFORM_VERSION
    content_sha256: str = ""                    # 内容身份（"仅 ID 不算看过图"）
    # ---- 三态 ----
    delivered_rounds: list[int] = []
    observed_rounds: list[int] = []
    unobserved_reason: str = ""                 # 未交付/未观察的原因（不静默）
    n_pixels_sent: int = 0
    token_estimate: int = 0
    token_estimate_source: str = TOKEN_ESTIMATE_SOURCE

    @field_validator("content_sha256")
    @classmethod
    def _needs_content_hash(cls, v: str) -> str:
        if not str(v or "").strip():
            raise ValueError(
                "图像必须有内容哈希（§9.4：仅路径、ID 或成功标记不能算看过图）")
        return v

    @field_validator("observed_rounds")
    @classmethod
    def _observed_implies_delivered(cls, v: list[int], info) -> list[int]:
        delivered = set(info.data.get("delivered_rounds") or [])
        if not set(v) <= delivered:
            raise ValueError(
                f"observed_rounds={sorted(v)} 不在 delivered_rounds={sorted(delivered)} 内"
                "（§9.4：进了请求才可能被观察到）")
        return v

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, v: str) -> str:
        if v not in ("frame", "crop", "contact_sheet"):
            raise ValueError(f"未知图像种类 {v!r}")
        return v

    def model_post_init(self, __context) -> None:  # noqa: D105 - pydantic v2 hook
        if self.kind in ("crop", "contact_sheet"):
            if self.source_frame_id is None or not self.box_xyxy:
                raise ValueError(
                    f"派生图（{self.kind}）必须记录源帧与像素范围（§9.4）")
        for hw in (self.source_hw, self.sent_hw):
            if hw and (len(hw) != 2 or min(int(x) for x in hw) <= 0):
                raise ValueError(f"尺寸 {hw!r} 非法（应为正的 [H, W]）")
        if self.box_xyxy and len(self.box_xyxy) != 4:
            raise ValueError(f"box_xyxy 必须是 [x0, y0, x1, y1]，收到 {self.box_xyxy!r}")

    def observed(self) -> bool:
        """是否已被模型实际看到（收到响应才算）。"""
        return bool(self.observed_rounds)


class ImageRound(Spec):
    """一次模型请求的图像清单（§9.4"实际传入图像、缩放和 token 成本完整记录"）。"""

    round_index: int = 0
    layout: str = LAYOUT_ORIGINALS_ONLY
    trigger: str = ""                      # initial / observation / ...
    image_ids: list[str] = []              # 实际装进请求的顺序（与请求里逐一对齐）
    original_image_ids: list[str] = []
    derived_image_ids: list[str] = []
    n_images: int = 0
    max_images: int = 0                    # 服务图像上限（本 run 的声明值）
    omitted_originals: list[str] = []      # 本轮没进请求的原帧（已在更早轮次交付过）
    omitted_derived: list[str] = []        # 本轮没进请求的派生图（保持 unobserved，非静默）
    n_pixels_total: int = 0
    token_estimate_total: int = 0
    prompt_tokens: Optional[int] = None    # 服务返回的真实 prompt_tokens（有则记）
    delivered: bool = False                # 请求是否真的发出
    observed: bool = False                 # 请求是否返回了响应

    @field_validator("layout")
    @classmethod
    def _known_layout(cls, v: str) -> str:
        if v not in IMAGE_LAYOUTS:
            raise ValueError(f"未知图像布局 {v!r}；已声明布局={list(IMAGE_LAYOUTS)}")
        return v

    def model_post_init(self, __context) -> None:  # noqa: D105
        if self.observed and not self.delivered:
            raise ValueError("observed=True 但 delivered=False（§9.4：没进请求就不算看到）")
        if self.n_images and self.n_images != len(self.image_ids):
            raise ValueError(
                f"n_images={self.n_images} 与 image_ids={len(self.image_ids)} 不一致")


__all__ = [
    "IMAGE_LEDGER_SCHEMA_VERSION",
    "IMAGE_LAYOUTS",
    "IMAGE_TRANSFORM_VERSION",
    "LAYOUT_DERIVED_PLUS_ORIGINALS",
    "LAYOUT_ORIGINALS_ONLY",
    "TOKEN_ESTIMATE_SOURCE",
    "ImageKind",
    "ImageRound",
    "ProducedImage",
]
