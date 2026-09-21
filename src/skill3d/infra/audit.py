"""G-40 审计回溯（简化）：任一 revision_id 回溯完整证据链。

证据链（§8 审计轨迹）：`induction 轨迹 → 反例包 → patch → 实验结果 → 准入决策`。
数据源是离线链落的 JSONL：`candidate_revision` / `counterexample_bundle` /
`paired_outcome` / `optimization_run` / `admission_decision` / `promotion`
/ `leakage_check` / `skill_governance_decision`（topics 见各自模块）。

本模块不做完整审计工具，只提供**可复现的查询视图**：按 revision_id 或
candidate_id 拉出该候选的全链记录，缺环显式标 `missing`（不假装完整）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# 证据链的期望环节（顺序即论文叙述顺序）
CHAIN_TOPICS = (
    "candidate_revision",
    "leakage_check",
    "counterexample_bundle",
    "paired_outcome",
    "optimization_run",
    "admission_decision",
    "skill_governance_decision",
    "promotion",
)


@dataclass
class AuditTrail:
    """单个候选的审计链。"""

    revision_id: str = ""
    candidate_id: str = ""
    steps: dict = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    n_records: int = 0

    @property
    def complete(self) -> bool:
        return not self.missing

    def summary(self) -> str:
        got = [t for t in CHAIN_TOPICS if self.steps.get(t)]
        return (f"revision={self.revision_id or '?'} candidate={self.candidate_id or '?'} "
                f"records={self.n_records} 环节={len(got)}/{len(CHAIN_TOPICS)} "
                f"缺={self.missing or '∅'}")

    def to_dict(self) -> dict:
        return {"revision_id": self.revision_id, "candidate_id": self.candidate_id,
                "steps": self.steps, "missing": self.missing,
                "n_records": self.n_records, "complete": self.complete}


def _iter_trace_records(trace_dir: str | Path) -> Iterable[tuple[str, dict]]:
    """遍历 trace 目录下所有 topic JSONL（每行即 payload）。"""
    root = Path(trace_dir)
    if not root.is_dir():
        return
    for f in sorted(root.glob("*.jsonl")):
        topic = f.stem   # 与 TraceStore 的 topic 名一致（下划线保留）
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield topic, json.loads(line)
            except Exception:  # noqa: BLE001 - 单行损坏不影响其余记录
                continue


def _ids_of(record: dict) -> set[str]:
    """一条记录关联的 revision/candidate id 集合（兼容各 schema 的字段命名）。"""
    ids: set[str] = set()
    for key in ("revision_id", "root_candidate_id", "candidate_id",
                "source_revision_id", "target_revision_id", "current_revision_id"):
        v = record.get(key)
        if isinstance(v, str) and v:
            ids.add(v)
    for key in ("revision_history", "paired_outcomes"):
        v = record.get(key)
        if isinstance(v, list):
            ids.update(str(x) for x in v if isinstance(x, (str, int)))
    return ids


def build_audit_trail(trace_dir: str | Path, *, revision_id: str = "",
                      candidate_id: str = "") -> AuditTrail:
    """按 revision_id / candidate_id 汇总证据链；缺环记录在 `missing`。

    匹配用**共现 id 的连通分量**：一条记录里出现的所有 id 视为同一候选的证据，
    因此只给 revision_id 也能拉到该候选的候选级记录（反之亦然）。
    注意 `SkillGovernanceDecision.candidate_id` 按其调用方存的是 revision id，
    连通分量法可自动处理这类字段口径差异。
    """
    if not revision_id and not candidate_id:
        raise ValueError("需要 revision_id 或 candidate_id 之一")
    records = list(_iter_trace_records(trace_dir))
    seed = revision_id or candidate_id

    # 连通分量（并查集）：同一记录内的 id 互相连通
    parent: dict[str, str] = {}

    def _find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for _topic, rec in records:
        ids = sorted(_ids_of(rec))
        for other in ids[1:]:
            _union(ids[0], other)

    trail = AuditTrail(revision_id=revision_id, candidate_id=candidate_id)
    if seed not in parent:
        # 未知 id：显式返回空链（缺环列全），绝不退化为"返回全部记录"
        trail.missing = list(CHAIN_TOPICS)
        return trail
    target = _find(seed)
    for topic, rec in records:
        if not any(_find(i) == target for i in _ids_of(rec) if i in parent):
            continue
        trail.n_records += 1
        trail.steps.setdefault(topic, []).append(rec)
        if not trail.revision_id and rec.get("revision_id"):
            trail.revision_id = str(rec["revision_id"])
        if not trail.candidate_id and rec.get("root_candidate_id"):
            trail.candidate_id = str(rec["root_candidate_id"])
    trail.missing = [t for t in CHAIN_TOPICS if not trail.steps.get(t)]
    return trail


# 真正承载"候选级 id"的 topic（其余 topic 的 candidate_id 可能是 revision id）
_CANDIDATE_ID_TOPICS = ("candidate_revision", "optimization_run", "admission_decision")


def list_candidates(trace_dir: str | Path) -> list[dict]:
    """列出 trace 中所有可审计的候选（id + 已具备的环节数）。"""
    seen: dict[str, set[str]] = {}
    for topic, rec in _iter_trace_records(trace_dir):
        cid = rec.get("root_candidate_id")
        if not cid and topic in _CANDIDATE_ID_TOPICS:
            cid = rec.get("candidate_id")
        if not cid:
            continue
        seen.setdefault(str(cid), set()).add(topic)
    return [{"candidate_id": cid, "topics": sorted(ts),
             "n_chain_steps": len([t for t in CHAIN_TOPICS if t in ts])}
            for cid, ts in sorted(seen.items())]


def write_audit_trail(trail: AuditTrail, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trail.to_dict(), ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    return out
