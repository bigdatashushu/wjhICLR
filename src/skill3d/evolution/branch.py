"""M17 ExperimentBranch：从同一 EnvironmentSnapshot 派生多分支。

分支角色：baseline / memory_only / skill_only / candidate / old_version / error_adversarial。
所有分支共享同一 snapshot（含同一 reconstruction_artifact_ref，硬约束 18）。
"""

from __future__ import annotations

import uuid

from skill3d.schemas import EnvironmentSnapshot, ExperimentBranch
from skill3d.schemas.evolution import BranchRole

ALL_ROLES: list[BranchRole] = [
    "baseline", "memory_only", "skill_only", "candidate", "old_version", "error_adversarial",
]


def derive_branches(snapshot: EnvironmentSnapshot,
                    roles: list[BranchRole] | None = None,
                    gpu_rank: int = 0,
                    container_image_digest: str = "unknown",
                    patched_memory_entry_ids: dict[str, list[str]] | None = None,
                    patched_skill_semvers: dict[str, list[str]] | None = None) -> list[ExperimentBranch]:
    """从同一 snapshot 派生分支；未冻结校验通过（frozen_policy_ok=False）→ 拒跑。"""
    if not snapshot.frozen_policy_ok:
        raise ValueError(
            f"snapshot {snapshot.snapshot_id} 冻结差分校验未通过（frozen_policy_ok=False），拒跑"
        )
    branches = []
    for role in (roles or ["baseline", "candidate"]):
        branches.append(ExperimentBranch(
            branch_id=f"br-{uuid.uuid4().hex[:12]}",
            snapshot_id=snapshot.snapshot_id,
            role=role,
            patched_memory_entry_ids=(patched_memory_entry_ids or {}).get(role, []),
            patched_skill_semvers=(patched_skill_semvers or {}).get(role, []),
            gpu_rank=gpu_rank,
            container_image_digest=container_image_digest,
            status="pending",
        ))
    return branches
