"""§13.5/§13.6 方法交付：完整正文、交付身份与上下文上限。

规范原文（§13.5）：

    「…选取并交付**完整方法正文**。」
    「上下文放不下时**先减少完整条目**，不能截掉检查、局部条件或来源后仍称
    "完整 Skill 已交付"。」
    「**超过服务限制的候选在静态检查中拒绝或修订**。」

规范原文（§13.6）：

    「区分"检索选中但未送达模型"与"已交付"。」

因此本模块是**交付文本的唯一事实源**：prompt 里渲染的方法条目与 trace 里记的
正文 hash 由同一个函数产出。否则 trace 里的 `delivered_content_sha256` 只是
"某个看起来像正文的字符串"的 hash，无法证明模型真的收到了那一段文本。

三种状态必须分清（§13.6）：

- `selected`（检索选中）：排序后进入 top-k，**不等于**模型见过；
- `delivered`（已交付）：条目进了**实际发出**的模型请求；
- `dropped`：因上下文上限被整条丢弃（不许截断后仍记"已交付"）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

# 上下文上限的默认值（可被 `configs/config.yaml` 的 `retrieval.method_context_max_chars`
# 覆盖）。它不是性能结论，只是"服务限制内可放多少完整方法"的起始配置：
# 32 帧 @max_pixels=131072 ≈ 9.7k tokens，`max_model_len=32768`，留出推理与程序输出
# 余量后给方法上下文的预算约为 8k 字符。
DEFAULT_METHOD_CONTEXT_MAX_CHARS = 8000

# 交付渠道（§13.6：只有真正发出的模型请求才算"已交付"）
DELIVERY_CHANNEL_MODEL_REQUEST = "model_request"   # 请求已发出（模型收到了正文）
DELIVERY_CHANNEL_REQUEST_FAILED = "request_failed"  # 请求发出但失败
DELIVERY_CHANNEL_NOT_SENT = "not_sent"             # 没有发出任何请求

# 未交付原因（"选中但未送达"的解释；受词表校验，避免自由字符串漂移）
DELIVERY_REASON_DELIVERED = "delivered"
DELIVERY_REASON_CODES: frozenset[str] = frozenset({
    DELIVERY_REASON_DELIVERED,
    "not_selected",          # 没被选中（硬条件未过 / 排序未进 top-k）→ 谈不上交付
    "context_cap_exceeded",  # §13.5：整条放不下 → 丢弃整条（不截断）
    "no_model_request",      # C0/mock/未配置客户端：没有发出请求
    "request_failed",        # 请求失败（服务不可用），不假装模型见过
})


def render_skill_entry(skill: Any) -> str:
    """单条方法的**完整**正文（进 prompt 的唯一形态；不做任何截断）。

    直接返回完整 ``skill_md``；这个字符串同时用于模型请求与 ``content_sha256``。
    """
    from skill3d.schemas.skill import SkillSpecV11

    if not isinstance(skill, SkillSpecV11):
        raise ValueError("当前交付只接受 SkillSpecV11")
    return skill.skill_md


def skill_content_sha256(skill: Any) -> str:
    """方法正文 hash（交付身份；§13.6"正文 hash"）。"""
    return hashlib.sha256(render_skill_entry(skill).encode("utf-8")).hexdigest()


def skill_version_key(skill: Any) -> str:
    """`skill_id@version`（检索、交付、程序三处共用的键）。"""
    return f"{skill.skill_id}@{skill.version}"


@dataclass(frozen=True)
class DeliveredSkill:
    skill_version: str
    content_sha256: str
    chars: int
    # 最终交付文本（prompt 渲染直接用它）。`to_dict()` **不**落盘正文，只落 hash：
    # trace 里存正文会与 prompt 重复且放大体积，而身份由 hash 唯一确定。
    text: str = ""

    def to_dict(self) -> dict:
        return {"skill_version": self.skill_version,
                "content_sha256": self.content_sha256,
                "chars": int(self.chars)}


@dataclass(frozen=True)
class DroppedSkill:
    skill_version: str
    reason: str
    chars: int

    def to_dict(self) -> dict:
        return {"skill_version": self.skill_version, "reason": self.reason,
                "chars": int(self.chars)}


@dataclass
class SkillDeliveryPlan:
    """一次合成的交付计划：**整条**保留 / **整条**丢弃，没有中间态。"""

    entries: list[DeliveredSkill] = field(default_factory=list)
    dropped: list[DroppedSkill] = field(default_factory=list)
    max_chars: int = DEFAULT_METHOD_CONTEXT_MAX_CHARS
    used_chars: int = 0
    channel: str = DELIVERY_CHANNEL_NOT_SENT
    channel_note: str = ""

    @property
    def delivered_skill_versions(self) -> list[str]:
        """**已交付**版本：只有请求真正发出（channel=model_request）时才非空。"""
        if self.channel != DELIVERY_CHANNEL_MODEL_REQUEST:
            return []
        return [e.skill_version for e in self.entries]

    @property
    def selected_skill_versions(self) -> list[str]:
        """**检索选中**版本（进了报告/prompt 计划，不一定送达）。"""
        return [e.skill_version for e in self.entries]

    @property
    def delivered_content_sha256(self) -> dict[str, str]:
        if self.channel != DELIVERY_CHANNEL_MODEL_REQUEST:
            return {}
        return {e.skill_version: e.content_sha256 for e in self.entries}

    def mark_delivered(self) -> None:
        self.channel = DELIVERY_CHANNEL_MODEL_REQUEST

    def mark_request_failed(self, note: str = "") -> None:
        self.channel = DELIVERY_CHANNEL_REQUEST_FAILED
        self.channel_note = str(note or "")

    def to_dict(self) -> dict:
        return {
            "max_chars": int(self.max_chars),
            "used_chars": int(self.used_chars),
            "channel": self.channel,
            "channel_note": self.channel_note,
            "entries": [e.to_dict() for e in self.entries],
            "dropped": [d.to_dict() for d in self.dropped],
            "delivered_skill_versions": self.delivered_skill_versions,
        }


def plan_delivery(skills: Optional[Sequence[Any]], *,
                  max_chars: Optional[int] = None) -> SkillDeliveryPlan:
    """把候选方法排成"整条交付"的计划（§13.5 上下文上限）。

    规则（按规范原文）：

    - 按**传入顺序**（= 检索排序）贪心取完整条目，累计长度含分隔符不得超上限；
    - v11 每题只有一条完整方法，单条放不下表示候选静态准入失效，直接报错；
    - 上限必须为正：`None` 用默认值，非正数 raise（"0 字符预算"与"无上限"是两回事，
      静默当成无上限会让上限形同虚设）。
    """
    limit = int(DEFAULT_METHOD_CONTEXT_MAX_CHARS if max_chars is None else max_chars)
    if limit <= 0:
        raise ValueError(f"method_context_max_chars 必须为正整数，收到 {limit!r}")
    plan = SkillDeliveryPlan(max_chars=limit)
    used = 0
    for skill in (skills or []):
        text = render_skill_entry(skill)
        n = len(text)
        sep = 1 if plan.entries else 0        # 条目间的换行分隔符也算进预算
        if used + sep + n > limit:
            raise ValueError(
                f"v11 Skill {skill_version_key(skill)} 完整正文 {n} 字符超过"
                f"方法上下文上限 {limit}；必须压缩候选，不能运行时丢弃")
        used += sep + n
        plan.entries.append(DeliveredSkill(
            skill_version=skill_version_key(skill),
            content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            chars=n,
            text=text))
    plan.used_chars = used
    return plan


def skill_body_length(skill: Any) -> int:
    """单条方法的完整正文字符数（静态检查用：见 §13.5"超过服务限制的候选"）。"""
    return len(render_skill_entry(skill))


def check_service_limit(skill: Any, *, max_chars: int) -> list[str]:
    """§13.5 静态检查：**单条正文**超过方法上下文上限 → 该候选永远无法完整交付。

    只拒绝"连单独一条都放不下"的候选。多个候选之间的竞争由运行期排序 + 上限决定
    （那时是"减少完整条目"，不是拒绝候选）。
    """
    n = skill_body_length(skill)
    if int(max_chars) > 0 and n > int(max_chars):
        return [f"方法正文 {n} 字符 > 服务限制 {int(max_chars)} 字符"
                f"（§13.5：超过服务限制的候选在静态检查中拒绝或修订）"]
    return []


__all__ = [
    "DEFAULT_METHOD_CONTEXT_MAX_CHARS",
    "DELIVERY_CHANNEL_MODEL_REQUEST",
    "DELIVERY_CHANNEL_NOT_SENT",
    "DELIVERY_CHANNEL_REQUEST_FAILED",
    "DELIVERY_REASON_CODES",
    "DELIVERY_REASON_DELIVERED",
    "DeliveredSkill",
    "DroppedSkill",
    "SkillDeliveryPlan",
    "check_service_limit",
    "plan_delivery",
    "render_skill_entry",
    "skill_body_length",
    "skill_content_sha256",
    "skill_version_key",
]
