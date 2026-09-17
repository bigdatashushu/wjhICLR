"""M14 三层记忆：working（单 episode 即弃）/ episodic（按 scene 聚合）/ semantic（离线巩固）。

- working：仅当前 episode 生命周期内有效，episode 结束必须清空（不写盘）。
- episodic：按 scene_id 聚合的轨迹经验，可持久化。
- semantic：离线巩固产物（见 consolidation.py），跨场景泛化条目。
- provenance 必填：任何条目写入时必须携带来源 trace ref，否则拒绝（硬约束 19 防污染）。
"""

from __future__ import annotations

from datetime import datetime, timezone

from skill3d.schemas import MemoryEntry


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProvenanceRequiredError(ValueError):
    """provenance 为空时抛出（硬约束 19：无来源条目不得入库）。"""


class ThreeLayerMemory:
    """内存态三层记忆存储（持久化交给 backend，如 lancedb_backend）。"""

    def __init__(self) -> None:
        self._working: dict[str, MemoryEntry] = {}
        self._episodic: dict[str, MemoryEntry] = {}
        self._semantic: dict[str, MemoryEntry] = {}
        # episodic 按 scene 聚合的索引：scene_id -> [memory_id]
        self._episodic_by_scene: dict[str, list[str]] = {}

    # ---------- 写入 ----------
    def add(self, entry: MemoryEntry, scene_id: str | None = None) -> MemoryEntry:
        """写入条目。provenance 必须非空，否则拒写。"""
        if not entry.provenance:
            raise ProvenanceRequiredError(
                f"MemoryEntry {entry.memory_id} 缺少 provenance，拒绝写入（硬约束 19）"
            )
        if entry.layer == "working":
            self._working[entry.memory_id] = entry
        elif entry.layer == "episodic":
            self._episodic[entry.memory_id] = entry
            if scene_id is not None:
                self._episodic_by_scene.setdefault(scene_id, []).append(entry.memory_id)
        elif entry.layer == "semantic":
            self._semantic[entry.memory_id] = entry
        else:  # pragma: no cover - Literal 已约束
            raise ValueError(f"未知 layer: {entry.layer}")
        return entry

    def make_entry(
        self,
        memory_id: str,
        layer: str,
        content: str,
        provenance: list[str],
        evidence_strength: float = 0.0,
        contradiction_group_id: str | None = None,
    ) -> MemoryEntry:
        """构造并写入条目的便捷接口。"""
        entry = MemoryEntry(
            memory_id=memory_id,
            layer=layer,  # type: ignore[arg-type]
            content=content,
            provenance=provenance,
            evidence_strength=evidence_strength,
            contradiction_group_id=contradiction_group_id,
            created_at=_now_iso(),
        )
        return self.add(entry)

    # ---------- 查询 ----------
    def get(self, memory_id: str) -> MemoryEntry | None:
        for store in (self._working, self._episodic, self._semantic):
            if memory_id in store:
                return store[memory_id]
        return None

    def list_layer(self, layer: str) -> list[MemoryEntry]:
        store = {"working": self._working, "episodic": self._episodic,
                 "semantic": self._semantic}[layer]
        return list(store.values())

    def list_episodic_by_scene(self, scene_id: str) -> list[MemoryEntry]:
        return [self._episodic[mid] for mid in self._episodic_by_scene.get(scene_id, [])]

    # ---------- 删除 / 生命周期 ----------
    def delete(self, memory_id: str) -> bool:
        for store in (self._working, self._episodic, self._semantic):
            if memory_id in store:
                del store[memory_id]
                return True
        return False

    def end_episode(self) -> None:
        """episode 结束：working 层即弃（硬约束：working 单 episode 即弃）。"""
        self._working.clear()
