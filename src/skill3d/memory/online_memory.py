"""M14 在线 episodic 记忆：写入 + 三因子检索（G-26）。

三层记忆的在线部分（§4 M14、§7.1 G-26）：

- **working**：单 episode 内自然存在，episode 结束即弃（`ThreeLayerMemory.end_episode`），不落盘；
- **episodic**：本模块在线写入，按 `scene_id` 聚合 —— 每 episode 一条，含
  provenance（trace ref）与证据强度；
- **semantic**：由离线归纳（G-28）产出，**在线绝不写**（硬约束 1/2）。

检索复用 Generative Agents（arXiv:2304.03442）的三因子公式：
`score = w_rec·recency + w_imp·importance + w_rel·relevance`，
其中 recency 为指数衰减，importance 取条目 `evidence_strength`，
relevance 为查询与内容的确定性文本相似度（无需 embedding，可离线复现）。

纪律：
- final_test 的 episode **不得写 Memory**（硬约束 9：final test 不写 Memory/Skill）；
- provenance 必填（硬约束 19）：缺失即拒写；
- 写入前做 LEAKAGE_CHECK（含 sample 答案/ID 即拒）。

持久化：默认 JSONL 追加（可读可审计）；LanceDB 可用时由
`memory/lancedb_backend` 承担向量检索，本模块保持无第三方依赖。
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Sequence

from skill3d.memory.consolidation import leakage_scan_text
from skill3d.schemas import MemoryEntry

# 三因子权重（TODO_CALIBRATE，起始等权 + relevance 略高）
W_RECENCY = 1.0
W_IMPORTANCE = 1.0
W_RELEVANCE = 1.5
# recency 指数衰减半衰期（天，TODO_CALIBRATE）
RECENCY_HALFLIFE_DAYS = 7.0
# 检索默认 top-k（TODO_CALIBRATE）
DEFAULT_TOP_K = 5

_FINAL_TEST_SPLITS = ("final_test",)


class MemoryWriteForbiddenError(RuntimeError):
    """final_test 或缺少 provenance 时的写入门禁（硬约束 9/19）。"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_days(created_at: str, now: Optional[datetime] = None) -> float:
    """条目年龄（天）；时间戳不可解析时按 0（最新）处理，不因脏数据丢条目。"""
    try:
        ts = datetime.fromisoformat(str(created_at))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return 0.0
    ref = now or datetime.now(timezone.utc)
    return max((ref - ts).total_seconds() / 86400.0, 0.0)


def recency_score(created_at: str, *, now: Optional[datetime] = None,
                  halflife_days: float = RECENCY_HALFLIFE_DAYS) -> float:
    """指数衰减 recency ∈ (0,1]（Generative Agents 的 exponential decay）。"""
    return float(math.exp(-math.log(2) * _age_days(created_at, now) / max(halflife_days, 1e-6)))


def relevance_score(query: str, content: str) -> float:
    """确定性文本相关性 ∈ [0,1]（SequenceMatcher；无 embedding 依赖，可复现）。"""
    if not query or not content:
        return 0.0
    return float(SequenceMatcher(None, query.lower(), content.lower()).ratio())


def three_factor_score(entry: MemoryEntry, query: str, *, now: Optional[datetime] = None,
                       w_recency: float = W_RECENCY, w_importance: float = W_IMPORTANCE,
                       w_relevance: float = W_RELEVANCE) -> float:
    """三因子检索打分（G-26 验收：top-k 合理）。"""
    return (w_recency * recency_score(entry.created_at, now=now)
            + w_importance * float(entry.evidence_strength)
            + w_relevance * relevance_score(query, entry.content))


@dataclass
class EpisodicMemory:
    """在线 episodic 记忆（JSONL 追加 + 内存索引，按 scene 聚合）。

    `path` 指向不可写位置时**不抛异常**：降级为纯内存 + `degraded_reason`，
    记忆是增强项，绝不因存储问题阻断在线链。
    """

    path: Optional[str] = None
    entries: list[MemoryEntry] = field(default_factory=list)
    degraded_reason: str = ""

    def __post_init__(self) -> None:
        if not self.path:
            return
        p = Path(self.path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.is_file():
                self.entries = [MemoryEntry.model_validate_json(line)
                                for line in p.read_text(encoding="utf-8").splitlines()
                                if line.strip()]
        except OSError as exc:
            self.degraded_reason = f"{type(exc).__name__}: {exc}（降级为纯内存，不落盘）"
            self.path = None

    # ---- 写入 ----
    def record(
        self,
        *,
        scene_name: str,
        content: str,
        provenance: Sequence[str],
        evidence_strength: float = 0.0,
        split: str = "",
        forbidden_sample_ids: Optional[set[str]] = None,
    ) -> MemoryEntry:
        """写一条 episodic 记忆（在线；final_test 与无 provenance 一律拒写）。"""
        if split in _FINAL_TEST_SPLITS:
            raise MemoryWriteForbiddenError(
                f"split={split} 不得写 Memory（硬约束 9：final test 不写 Memory/Skill）"
            )
        if not provenance:
            raise MemoryWriteForbiddenError(
                "episodic 记忆必须携带 provenance（trace ref），硬约束 19")
        hits = leakage_scan_text(content, forbidden_sample_ids=forbidden_sample_ids)
        if hits:
            raise MemoryWriteForbiddenError(f"episodic 记忆未过 LEAKAGE_CHECK: {hits}")

        entry = MemoryEntry(
            memory_id=f"mem-{uuid.uuid4().hex[:12]}",
            layer="episodic",
            content=f"[scene={scene_name}] {content}",
            provenance=list(provenance),
            evidence_strength=float(evidence_strength),
            contradiction_group_id=None,
            created_at=_now_iso(),
        )
        self.entries.append(entry)
        self._append_jsonl(entry)
        return entry

    def record_episode(
        self,
        *,
        qa_id: str,
        scene_name: str,
        task: str,
        final_state: str,
        correct: Optional[bool],
        split: str = "",
        route: str = "",
        flags: Sequence[str] = (),
        forbidden_sample_ids: Optional[set[str]] = None,
    ) -> Optional[MemoryEntry]:
        """把一个 episode 的**结果摘要**写成一条 episodic 记忆。

        内容为"题型/终态/路由/失败标志"的聚合摘要——**不含问题文本、不含答案**
        （硬约束 19）；证据强度按结果给分（成功 1.0 / 兜底 0.5 / 未答 0.2）。
        """
        if split in _FINAL_TEST_SPLITS:
            return None  # 硬约束 9：final test 静默不写（调用方无需分支）
        if correct is True:
            strength, verdict = 1.0, "answer_correct"
        elif final_state == "answer":
            strength, verdict = 0.5, "answer_incorrect"
        elif final_state == "answer_best_effort":
            strength, verdict = 0.3, "answer_best_effort"
        else:
            strength, verdict = 0.2, str(final_state)
        flag_txt = f" flags={list(flags)}" if flags else ""
        content = (f"task={task} final_state={final_state} verdict={verdict}"
                   f" route={route}{flag_txt}")
        return self.record(scene_name=scene_name, content=content,
                           provenance=[f"episode_trace:{qa_id}"],
                           evidence_strength=strength, split=split,
                           forbidden_sample_ids=forbidden_sample_ids)

    def _append_jsonl(self, entry: MemoryEntry) -> None:
        if not self.path:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    # ---- 检索 ----
    def list_scene(self, scene_name: str) -> list[MemoryEntry]:
        """按 scene 聚合检索（G-26 验收：episodic 写入后可按 scene 检索）。"""
        needle = f"[scene={scene_name}]"
        return [e for e in self.entries if e.content.startswith(needle)]

    def retrieve(self, query: str, *, scene_name: str = "", top_k: int = DEFAULT_TOP_K,
                 now: Optional[datetime] = None) -> list[tuple[float, MemoryEntry]]:
        """三因子检索 top-k（可选限定 scene）。返回 `[(score, entry)]` 降序。"""
        pool = self.list_scene(scene_name) if scene_name else list(self.entries)
        scored = [(three_factor_score(e, query, now=now), e) for e in pool]
        scored.sort(key=lambda t: (-t[0], t[1].memory_id))
        return scored[: max(top_k, 0)]

    def stats(self) -> dict:
        scenes = {e.content.split("]")[0] + "]" for e in self.entries}
        strengths = [e.evidence_strength for e in self.entries]
        return {
            "n_entries": len(self.entries),
            "n_scenes": len(scenes),
            "mean_evidence_strength": (sum(strengths) / len(strengths)) if strengths else 0.0,
        }
