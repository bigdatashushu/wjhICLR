"""M17 EnvironmentSnapshot：实验环境不可变起点 + 冻结差分校验。

- build_snapshot 构造 snapshot；
- check_frozen_diff 对 frozen_fields 逐字段比对：不一致 → frozen_policy_ok=False，拒跑。

v6（§3.4/§16.4/§19.2）：**离线治理模型的身份也是实验环境的一部分** —— 同一代候选若在中途
换了离线强模型（或换了它的 prompt 模板），"promote 依据"就不可比。故 `build_snapshot`
把三者折进 `prompt_version` 这一冻结字段：

    <在线合成模板版本>|tool_face=<tool-face 版本>|offline=<model_id>@<endpoint_hash 前12>

`EnvironmentSnapshot` 的字段集由 `schemas/evolution.py` 拥有（本模块不拥有，且只有一个
`prompt_version` 字段），所以用复合串承载"环境指纹"；任何一项变化都会让
`check_frozen_diff` 判 `frozen_policy_ok=False` 从而拒跑，而不是静默混比。
"""

from __future__ import annotations

import hashlib
import json
import uuid

from skill3d.schemas import EnvironmentSnapshot

# 复合指纹分隔符与离线段标记（进 frozen_fields 比对，改动即破坏兼容性）
PROMPT_VERSION_SEP = "|"


def offline_model_fingerprint(offline_client=None) -> str:
    """离线治理模型指纹（§3.4：只含 provider/model_id/endpoint_hash，不含密钥）。

    无客户端时返回 `""`（本环境未使用离线模型）——**不虚构**模型名。
    """
    if offline_client is None:
        return ""
    fields = {}
    getter = getattr(offline_client, "manifest_fields", None)
    if callable(getter):
        fields = dict(getter() or {})
    model_id = str(fields.get("model_id") or getattr(offline_client, "model_id", "") or "")
    endpoint = str(fields.get("endpoint_hash") or
                   getattr(offline_client, "endpoint_hash", "") or "")
    if not model_id:
        return ""
    return f"{model_id}@{endpoint[:12]}" if endpoint else model_id


def compose_prompt_version(online_template: str, *,
                           tool_face_version: str = "",
                           offline_client=None) -> str:
    """在线模板 + tool-face + 离线模型身份 → 冻结用的环境指纹串。"""
    parts = [str(online_template)]
    if tool_face_version:
        parts.append(f"tool_face={tool_face_version}")
    fingerprint = offline_model_fingerprint(offline_client)
    if fingerprint:
        parts.append(f"offline={fingerprint}")
    return PROMPT_VERSION_SEP.join(parts)


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
    *,
    offline_client=None,
) -> EnvironmentSnapshot:
    """构造 EnvironmentSnapshot（episode_set_hash 内容寻址）。

    `prompt_version` 为空串时取在线合成模板版本，并与 tool-face 版本、离线模型身份
    （`offline_client.manifest_fields()`）一起合成环境指纹（见模块 docstring）。
    """
    from skill3d.synthesis.prompt_builder import TEMPLATE_VERSION
    from skill3d.tools.registry import TOOL_FACE_VERSION

    episode_set_hash = hashlib.sha256(
        json.dumps(sorted(episode_set)).encode("utf-8")).hexdigest()
    return EnvironmentSnapshot(
        snapshot_id=f"env-{uuid.uuid4().hex[:12]}",
        tool_registry_digest=tool_registry_digest,
        reconstruction_artifact_ref=reconstruction_artifact_ref,
        memory_snapshot_ref=memory_snapshot_ref,
        skill_registry_snapshot_ref=skill_registry_snapshot_ref,
        prompt_version=compose_prompt_version(
            prompt_version or TEMPLATE_VERSION,
            tool_face_version=TOOL_FACE_VERSION, offline_client=offline_client),
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
    `prompt_version` 是复合指纹，因此"换了 tool-face 或换了离线模型"同样会被判不一致。
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


__all__ = [
    "PROMPT_VERSION_SEP",
    "build_snapshot",
    "check_frozen_diff",
    "compose_prompt_version",
    "offline_model_fingerprint",
]
