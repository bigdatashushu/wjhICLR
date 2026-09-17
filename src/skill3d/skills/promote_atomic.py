"""M15 promote 原子切换（硬约束 12）：

1. 读 active snapshot；
2. apply_candidate 产出新 snapshot（物理隔离，绝不动 active）；
3. validate_references 校验新 snapshot 完整性，失败则不切换；
4. 原子写指针（写临时文件 + os.replace）；
5. 旧 snapshot 文件保留，记录 snapshot_before，可 rollback。
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from skill3d.schemas import CandidateRevision


class SnapshotValidationError(ValueError):
    """新 snapshot 引用校验失败：不得切换。"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snapshot_path(store_dir: Path, snapshot_id: str) -> Path:
    return store_dir / f"snapshot_{snapshot_id}.json"


def read_active_snapshot(store_dir: str | Path) -> dict:
    """读 active 指针指向的 snapshot；无 active 时返回空初始 snapshot。"""
    store_dir = Path(store_dir)
    pointer = store_dir / "active_snapshot.json"
    if not pointer.exists():
        return {"snapshot_id": "genesis", "entries": {}, "parent_snapshot_id": None,
                "created_at": _now_iso()}
    ref = json.loads(pointer.read_text(encoding="utf-8"))
    snap = json.loads(_snapshot_path(store_dir, ref["snapshot_id"]).read_text(encoding="utf-8"))
    return snap


def apply_candidate(active: dict, candidate: CandidateRevision) -> dict:
    """在 active 的副本上应用候选，产出新 snapshot（不修改 active 本体）。"""
    new = json.loads(json.dumps(active))  # 深拷贝，物理隔离
    new["parent_snapshot_id"] = active["snapshot_id"]
    new["snapshot_id"] = f"snap-{uuid.uuid4().hex[:12]}"
    new["created_at"] = _now_iso()
    new.setdefault("entries", {})[candidate.revision_id] = {
        "root_candidate_id": candidate.root_candidate_id,
        "candidate_type": candidate.candidate_type,
        "spec_content": candidate.spec_content,
        "parent_version": candidate.parent_version,
        "created_by": candidate.created_by,
    }
    return new


def validate_references(snapshot: dict, known_roots: set[str] | None = None) -> None:
    """校验新 snapshot 完整性：结构合法 + 引用可解析（parent 链非空）。"""
    if not snapshot.get("snapshot_id"):
        raise SnapshotValidationError("缺少 snapshot_id")
    entries = snapshot.get("entries")
    if not isinstance(entries, dict):
        raise SnapshotValidationError("entries 必须为 dict")
    for rid, e in entries.items():
        if not e.get("spec_content"):
            raise SnapshotValidationError(f"条目 {rid} 缺少 spec_content")
        if e.get("root_candidate_id") is None:
            raise SnapshotValidationError(f"条目 {rid} 缺少 root_candidate_id 引用")
        if known_roots is not None and e["root_candidate_id"] not in known_roots:
            raise SnapshotValidationError(
                f"条目 {rid} 引用未知 root_candidate_id={e['root_candidate_id']}"
            )


def _write_pointer_atomic(store_dir: Path, snapshot_id: str) -> None:
    """原子写 active 指针：临时文件 + os.replace（硬约束 12）。"""
    pointer = store_dir / "active_snapshot.json"
    tmp = store_dir / f".active_snapshot.{uuid.uuid4().hex[:8]}.tmp"
    tmp.write_text(json.dumps({"snapshot_id": snapshot_id}, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, pointer)  # 同目录原子替换


def promote(store_dir: str | Path, candidate: CandidateRevision,
            known_roots: set[str] | None = None,
            promotion_log: list | None = None) -> dict:
    """promote(candidate)：原子切换 active snapshot，返回新 snapshot。

    validate 失败 → 不切换（抛 SnapshotValidationError，active 保持不变）。
    """
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    before = read_active_snapshot(store_dir)
    new = apply_candidate(before, candidate)
    validate_references(new, known_roots=known_roots)  # 失败即不切换
    _snapshot_path(store_dir, new["snapshot_id"]).write_text(
        json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_pointer_atomic(store_dir, new["snapshot_id"])
    if promotion_log is not None:
        promotion_log.append({
            "revision_id": candidate.revision_id,
            "snapshot_before": before["snapshot_id"],  # 记录 snapshot_before 供回滚
            "snapshot_after": new["snapshot_id"],
            "timestamp": _now_iso(),
        })
    return new


def rollback(store_dir: str | Path, snapshot_before: str,
             promotion_log: list | None = None) -> dict:
    """一键回滚到旧 snapshot（旧 snapshot 文件必须仍在）。"""
    store_dir = Path(store_dir)
    path = _snapshot_path(store_dir, snapshot_before)
    if not path.exists():
        raise FileNotFoundError(f"回滚目标 snapshot 不存在: {snapshot_before}")
    snap = json.loads(path.read_text(encoding="utf-8"))
    _write_pointer_atomic(store_dir, snapshot_before)
    if promotion_log is not None:
        promotion_log.append({"rollback_to": snapshot_before, "timestamp": _now_iso()})
    return snap
