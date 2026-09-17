"""M17 EnvironmentSnapshot：实验环境不可变起点 + 冻结差分校验。

- build_snapshot 构造 snapshot；
- check_frozen_diff 对 frozen_fields 逐字段比对：不一致 → frozen_policy_ok=False，拒跑。
"""

from __future__ import annotations

import hashlib
import json
import uuid

from skill3d.schemas import EnvironmentSnapshot


def build_snapshot(
    tool_registry_digest: str,
    reconstruction_artifact_ref: str,
    memory_snapshot_ref: str,
    skill_registry_snapshot_ref: str,
    prompt_version: str,
    code_commit: str,
    split_pointer: str,
    episode_set: list[str],
    seed: int,
    base_model_fingerprint: str,
    frozen_fields: list[str] | None = None,
) -> EnvironmentSnapshot:
    """构造 EnvironmentSnapshot（episode_set_hash 内容寻址）。"""
    episode_set_hash = hashlib.sha256(
        json.dumps(sorted(episode_set)).encode("utf-8")).hexdigest()
    return EnvironmentSnapshot(
        snapshot_id=f"env-{uuid.uuid4().hex[:12]}",
        tool_registry_digest=tool_registry_digest,
        reconstruction_artifact_ref=reconstruction_artifact_ref,
        memory_snapshot_ref=memory_snapshot_ref,
        skill_registry_snapshot_ref=skill_registry_snapshot_ref,
        prompt_version=prompt_version,
        code_commit=code_commit,
        split_pointer=split_pointer,  # type: ignore[arg-type]
        episode_set_hash=episode_set_hash,
        seed=seed,
        base_model_fingerprint=base_model_fingerprint,
        frozen_fields=frozen_fields
        or ["tool_registry_digest", "reconstruction_artifact_ref", "prompt_version",
            "code_commit", "seed", "base_model_fingerprint"],
        frozen_policy_ok=True,
    )


def check_frozen_diff(snapshot: EnvironmentSnapshot, reference: dict) -> EnvironmentSnapshot:
    """冻结差分校验：frozen_fields 逐字段与 reference 比对。

    任一字段不一致 → 返回 frozen_policy_ok=False 的副本（调用方拒跑，M17 异常降级）。
    """
    ok = True
    for field in snapshot.frozen_fields:
        if field not in reference:
            ok = False
            break
        if getattr(snapshot, field) != reference[field]:
            ok = False
            break
    if ok:
        return snapshot
    return snapshot.model_copy(update={"frozen_policy_ok": False})
