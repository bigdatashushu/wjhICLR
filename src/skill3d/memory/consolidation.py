"""M14 记忆巩固 / 遗忘 / 矛盾检测（离线）。

- LEAKAGE_CHECK：条目内容含 sample 答案 / sample ID 即拒（硬约束 19），真实实现正则 + 字段扫描。
- 近重复检测：相似度 > 0.85（TODO_CALIBRATE）→ 合并或拒写。
- 矛盾检测：检测到矛盾时分配 contradiction_group_id 隔离，不直接删除。
- 巩固：episodic 跨 scene 聚类 → ≥ N_min（TODO_CALIBRATE）跨 scene 样本 → 产 semantic 条目。
"""

from __future__ import annotations

import re
import uuid
from difflib import SequenceMatcher

from skill3d.schemas import MemoryEntry

# TODO_CALIBRATE：近重复相似度阈值（起始参考值）
DUP_SIMILARITY_THRESHOLD = 0.85
# TODO_CALIBRATE：巩固所需跨 scene 最小样本数
N_MIN_CROSS_SCENE = 3

# 泄漏扫描模式：答案字样 + 官方 qa_id 样式（真实实现：正则/字段扫描）
_ANSWER_PATTERNS = [
    re.compile(r"ground[_ ]?truth", re.IGNORECASE),
    re.compile(r"\banswer\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"correct\s+option", re.IGNORECASE),
]
_QA_ID_PATTERN = re.compile(r"\bqa_[0-9a-f]{8,}\b", re.IGNORECASE)


class LeakageCheckError(ValueError):
    """条目含 sample 答案/ID，被 LEAKAGE_CHECK 拒绝。"""


def leakage_scan_text(text: str, forbidden_sample_ids: set[str] | None = None) -> list[str]:
    """扫描文本，返回命中的泄漏模式列表（空 = 通过）。"""
    hits: list[str] = []
    for pat in _ANSWER_PATTERNS:
        if pat.search(text):
            hits.append(f"answer_pattern:{pat.pattern}")
    if _QA_ID_PATTERN.search(text):
        hits.append("qa_id_pattern")
    for sid in (forbidden_sample_ids or set()):
        if sid and sid in text:
            hits.append(f"sample_id:{sid}")
    return hits


def leakage_check_entry(entry: MemoryEntry, forbidden_sample_ids: set[str] | None = None) -> None:
    """硬门：含 sample 答案/ID 即拒（硬约束 13/19，任何建议不得覆盖）。"""
    hits = leakage_scan_text(entry.content, forbidden_sample_ids)
    # 字段扫描：provenance 引用本身也不得是 sample 答案引用
    for ref in entry.provenance:
        for sid in (forbidden_sample_ids or set()):
            if sid and sid == ref:
                hits.append(f"provenance_sample_id:{sid}")
    if hits:
        raise LeakageCheckError(f"条目 {entry.memory_id} 未通过 LEAKAGE_CHECK: {hits}")


def text_similarity(a: str, b: str) -> float:
    """确定性文本相似度（SequenceMatcher，0~1）。"""
    return SequenceMatcher(None, a, b).ratio()


def is_near_duplicate(new_content: str, existing_contents: list[str],
                      threshold: float = DUP_SIMILARITY_THRESHOLD) -> bool:
    """近重复检测：与任一现存条目相似度 > 阈值即视为重复。"""
    return any(text_similarity(new_content, c) > threshold for c in existing_contents)


def detect_contradictions(entries: list[MemoryEntry],
                          contradict_fn=None) -> dict[str, str]:
    """矛盾检测：返回 {memory_id: contradiction_group_id}，矛盾条目同组隔离。

    contradict_fn(a, b) -> bool：外部判矛盾（默认按 group 启发式：内容高度相似但
    含否定词差异视为矛盾；MVP 简化实现，TODO_CALIBRATE）。
    """
    groups: dict[str, str] = {}
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            if a.memory_id in groups and groups[a.memory_id] == groups.get(b.memory_id):
                continue
            if contradict_fn is not None:
                contra = contradict_fn(a, b)
            else:
                contra = _heuristic_contradiction(a.content, b.content)
            if contra:
                gid = groups.get(a.memory_id) or groups.get(b.memory_id) \
                    or f"contra-{uuid.uuid4().hex[:8]}"
                groups[a.memory_id] = gid
                groups[b.memory_id] = gid
    return groups


_NEG_WORDS = ("not", "never", "no ", "不", "无")


def _heuristic_contradiction(a: str, b: str) -> bool:
    """MVP 启发式：高相似但一方含否定词 → 疑似矛盾。TODO_CALIBRATE。"""
    sim = text_similarity(a, b)
    if sim < 0.6:
        return False
    a_neg = any(w in a.lower() for w in _NEG_WORDS)
    b_neg = any(w in b.lower() for w in _NEG_WORDS)
    return a_neg != b_neg


def consolidate(episodic_by_scene: dict[str, list[MemoryEntry]],
                existing_semantic: list[MemoryEntry],
                forbidden_sample_ids: set[str] | None = None,
                n_min: int = N_MIN_CROSS_SCENE) -> list[MemoryEntry]:
    """巩固：跨 scene ≥ n_min 的同类 episodic 经验 → semantic 条目。

    - 硬门：LEAKAGE_CHECK 不过即拒（硬约束 19）。
    - 近重复：与现有 semantic 重复则跳过（合并不重复入库）。
    - 矛盾：分配 contradiction_group_id 隔离而非删除。
    """
    scene_ids = [s for s, entries in episodic_by_scene.items() if entries]
    if len(set(scene_ids)) < n_min:
        return []  # 跨 scene 样本不足，不巩固（TODO_CALIBRATE）

    # 汇总全部 episodic 内容做聚类（MVP：按内容相似度贪心聚类）
    all_entries = [e for entries in episodic_by_scene.values() for e in entries]
    clusters: list[list[MemoryEntry]] = []
    for e in all_entries:
        placed = False
        for c in clusters:
            if text_similarity(e.content, c[0].content) > DUP_SIMILARITY_THRESHOLD:
                c.append(e)
                placed = True
                break
        if not placed:
            clusters.append([e])

    new_semantic: list[MemoryEntry] = []
    existing_contents = [s.content for s in existing_semantic]
    for cluster in clusters:
        # 跨 scene 覆盖检查：簇内条目须来自 ≥ n_min 个不同 scene
        covered = {s for s, entries in episodic_by_scene.items()
                   if any(e in entries for e in cluster)}
        if len(covered) < n_min:
            continue
        rep = max(cluster, key=lambda e: e.evidence_strength)
        if is_near_duplicate(rep.content, existing_contents):
            continue  # 近重复合并：不重复入库
        semantic_entry = MemoryEntry(
            memory_id=f"sem-{uuid.uuid4().hex[:12]}",
            layer="semantic",
            content=rep.content,
            provenance=[ref for e in cluster for ref in e.provenance],
            evidence_strength=sum(e.evidence_strength for e in cluster) / len(cluster),
            contradiction_group_id=None,
            created_at=rep.created_at,
        )
        leakage_check_entry(semantic_entry, forbidden_sample_ids)  # 硬门
        new_semantic.append(semantic_entry)
        existing_contents.append(rep.content)
    return new_semantic
