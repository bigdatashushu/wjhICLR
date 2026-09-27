"""§13.5 检索策略：top-k／排序规则／方法上下文上限的**唯一冻结来源**。

规范原文（§13.5）：

    「top-k、排序规则和方法上下文上限在演化开始前写入配置并冻结。开发起点可取
    top-k=1；该值是起始配置，不是性能结论。」
    「上下文放不下时先减少完整条目，不能截掉检查、局部条件或来源后仍称
    "完整 Skill 已交付"。」
    「超过服务限制的候选在静态检查中拒绝或修订。」

`configs/config.yaml` 的 `retrieval:` 段是这批参数的落盘位置；本模块把它读成
不可变对象，并给出**可复算的内容摘要**（`sha256`）——§13.6 要求每次检索记录
"配置版本"，而版本号本身不可抵赖：它由策略内容算出，改一个字符摘要就变。

`source` 字段区分"配置里明确写了"与"缺键用了默认值"：两者都可用，但 trace 里
必须看得出这次跑的是哪一种（默认值被静默当成"已冻结配置"是另一种形式的谎报）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from skill3d.skills.delivery import DEFAULT_METHOD_CONTEXT_MAX_CHARS

# 排序权重（§16.1：排序权重在 inner 上选定后**冻结**，outer 只验一次、final 不改）。
# 这三个键是排序规则的完整参数集；出现未登记键 = 配置写错了，直接拒绝（不静默忽略）。
DEFAULT_RANK_WEIGHTS: dict[str, float] = {
    "semantic": 1.0,              # 语义排序主分权重（rerank=True）
    "keyword": 1.0,               # 关键词主分权重（rerank=False 的消融口径）
    "semantic_mix_keyword": 0.0,  # 语义路径里混入的关键词分权重（0=纯语义）
}
RANK_WEIGHT_KEYS: tuple[str, ...] = tuple(DEFAULT_RANK_WEIGHTS)

DEFAULT_TOP_K = 3        # 每题进 prompt 的 Skill 条数（§13.5 开发起点可改用 1）
DEFAULT_CANDIDATES = 50  # 向量候选池大小

# §13.5 的顺序是**固定**的：先题型分区，再硬条件，再排序，最后选取。
RANKING_RULE = "partition_then_hard_requirements_then_relevance"


@dataclass(frozen=True)
class RetrievalPolicy:
    """冻结的检索策略（不可变；运行期不得按题/按场景临时改）。"""

    top_k: int = DEFAULT_TOP_K
    rerank: bool = True
    candidates: int = DEFAULT_CANDIDATES
    rank_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RANK_WEIGHTS))
    method_context_max_chars: int = DEFAULT_METHOD_CONTEXT_MAX_CHARS
    ranking_rule: str = RANKING_RULE
    label: str = ""          # 人类可读标签（yaml: retrieval.config_version）
    source: str = "default"  # config | default

    def __post_init__(self) -> None:
        if int(self.top_k) < 1:
            raise ValueError(f"retrieval.top_k 必须 >= 1，收到 {self.top_k!r}")
        if int(self.candidates) < 1:
            raise ValueError(
                f"retrieval.candidates 必须 >= 1，收到 {self.candidates!r}")
        if int(self.method_context_max_chars) <= 0:
            raise ValueError("retrieval.method_context_max_chars 必须为正整数，"
                             f"收到 {self.method_context_max_chars!r}")
        unknown = sorted(set(self.rank_weights) - set(RANK_WEIGHT_KEYS))
        if unknown:
            raise ValueError(
                f"retrieval.rank_weights 含未登记键 {unknown}；"
                f"排序规则只有 {list(RANK_WEIGHT_KEYS)}（写错的权重不许被静默忽略）")
        object.__setattr__(self, "rank_weights",
                           {k: float(self.rank_weights.get(k, DEFAULT_RANK_WEIGHTS[k]))
                            for k in RANK_WEIGHT_KEYS})

    # ---- 身份 ----
    def canonical(self) -> dict[str, Any]:
        """参与摘要计算的规范内容（键序固定，可复算）。"""
        return {
            "top_k": int(self.top_k),
            "rerank": bool(self.rerank),
            "candidates": int(self.candidates),
            "rank_weights": {k: float(self.rank_weights[k]) for k in RANK_WEIGHT_KEYS},
            "method_context_max_chars": int(self.method_context_max_chars),
            "ranking_rule": str(self.ranking_rule),
        }

    def sha256(self) -> str:
        blob = json.dumps(self.canonical(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def version(self) -> str:
        """§13.6"配置版本"：有人类标签就用标签，否则用内容摘要前缀。"""
        return str(self.label) or f"ret-{self.sha256()[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {**self.canonical(), "version": self.version(),
                "sha256": self.sha256(), "source": str(self.source),
                "label": str(self.label)}

    def weight(self, name: str) -> float:
        if name not in RANK_WEIGHT_KEYS:
            raise KeyError(f"未知排序权重 {name!r}；只有 {list(RANK_WEIGHT_KEYS)}")
        return float(self.rank_weights[name])


def retrieval_policy_from_config(cfg: Optional[Mapping[str, Any]]) -> RetrievalPolicy:
    """从主配置读检索策略（缺段/缺键 → 默认值，并记 `source`）。"""
    section = (cfg or {}).get("retrieval")
    if not isinstance(section, Mapping) or not section:
        return RetrievalPolicy(source="default")
    weights = section.get("rank_weights")
    weights = dict(weights) if isinstance(weights, Mapping) else dict(DEFAULT_RANK_WEIGHTS)
    return RetrievalPolicy(
        top_k=int(section.get("top_k", DEFAULT_TOP_K)),
        rerank=bool(section.get("rerank", True)),
        candidates=int(section.get("candidates", DEFAULT_CANDIDATES)),
        rank_weights=weights,
        method_context_max_chars=int(section.get(
            "method_context_max_chars", DEFAULT_METHOD_CONTEXT_MAX_CHARS)),
        label=str(section.get("config_version", "") or ""),
        source="config",
    )


__all__ = [
    "DEFAULT_CANDIDATES",
    "DEFAULT_RANK_WEIGHTS",
    "DEFAULT_TOP_K",
    "RANKING_RULE",
    "RANK_WEIGHT_KEYS",
    "RetrievalPolicy",
    "retrieval_policy_from_config",
]
