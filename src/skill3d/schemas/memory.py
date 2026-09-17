"""§5.7 记忆 Schema。"""

from typing import Literal, Optional

from . import Spec


class MemoryEntry(Spec):
    memory_id: str
    layer: Literal["working", "episodic", "semantic"]
    content: str
    provenance: list[str]  # trace ref
    evidence_strength: float
    contradiction_group_id: Optional[str]
    created_at: str


class MemorySnapshot(Spec):
    snapshot_id: str
    entry_ids: list[str]
    hash: str
    parent_hash: Optional[str]
