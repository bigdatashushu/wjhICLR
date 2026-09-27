"""M16 基于反例的离线修订：CounterexampleBundle → GPT6Patch → 新版本 CandidateRevision。

v6 §3.4：执行修订的离线模型是 **DeepSeek-V4.1-Flash**（`governance/deepseek_client.py`）。

红线：

- **候选不可变（硬约束 11）**：绝不原地修改候选，修订必须产新版本（parent_version 链）；
- **只见可见摘要（硬约束 19）**：离线模型只拿 `bundle.gpt6_visible_summary`
  （不含答案、不含 sample id）；
- **advisory（§3.3）**：修订产物是 draft 候选，**不**构成准入决定；
- patch 必须声明 `affected_task_types`，否则静态检查失败（§8）。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from skill3d.schemas import CandidateRevision, CounterexampleBundle, GPT6Patch
from skill3d.evolution.firewall import scan_prompt_for_leakage

# prompt 模板版本（冻结配置项，进 RunManifest 的 prompt_version；§3.4）
REVISE_PROMPT_VERSION = "revise-v1"
# v5 名称别名（过渡期只读）
GPT6_REVISE_PROMPT_VERSION = REVISE_PROMPT_VERSION

# `CandidateRevision.created_by` 受控枚举定义在 `schemas/evolution.py`（本模块不拥有；
# 其字面量仍是 v5 取值）。集中在此，schema 改名后只需改这一行。
CREATED_BY_OFFLINE_REVISION = "gpt6_revision"


class PatchStaticCheckError(ValueError):
    """patch 静态检查失败（如未声明 affected_task_types）。"""


def build_revision_prompt(current_spec_content: str,
                          gpt6_visible_summary: str,
                          rejected_similar: list[str] | None = None) -> str:
    """构造修订 prompt：只含当前 spec + 反例摘要（不含答案）+ 已拒相似候选。

    `gpt6_visible_summary` 形参名沿用 `CounterexampleBundle` 的字段名（该字段名定义在
    `schemas/evolution.py`，本模块不拥有）；内容是**可见摘要**，绝不含答案。
    """
    lines = [
        "你是 harness3D 的离线 Skill 修订器（DeepSeek-V4.1-Flash）。"
        "基于反例摘要给出结构化 patch（JSON）。",
        "禁止包含任何具体题目答案或 sample ID。",
        "你**不**执行实验、不评分、不决定 promote/reject —— 准入由确定性门决定。",
        "## 当前候选 spec_content",
        current_spec_content,
        "## 反例摘要（不含答案）",
        gpt6_visible_summary,
        "## 已被拒绝的相似候选（避免重复）",
        *[f"- {s}" for s in (rejected_similar or [])],
        "## 输出 JSON 字段",
        "patch_type / affected_task_types(必填) / affected_skills / patch_content / "
        "rationale / expected_improvement / risk_notes",
    ]
    prompt = "\n".join(lines)
    scan_prompt_for_leakage(prompt)  # 硬门：prompt 不得含答案/ID（硬约束 19）
    return prompt


def parse_patch(response_text: str, target_revision_id: str,
                prompt_version: str = REVISE_PROMPT_VERSION,
                model_id: str = "") -> GPT6Patch:
    """解析离线模型输出为 patch，并做静态检查：affected_task_types 必填。

    `model_id` 落进 patch（§3.4：离线模型身份可审计；调用方未提供时保持 Schema 默认值，
    **不虚构**一个模型名）。
    """
    data = json.loads(response_text)
    patch = GPT6Patch(
        target_revision_id=target_revision_id,
        patch_type=data["patch_type"],
        affected_task_types=list(data.get("affected_task_types") or []),
        affected_skills=list(data.get("affected_skills") or []),
        patch_content=data["patch_content"],
        rationale=data.get("rationale", ""),
        expected_improvement=data.get("expected_improvement", ""),
        risk_notes=data.get("risk_notes", ""),
        model_id=(model_id or GPT6Patch.model_fields["model_id"].default),
        prompt_version=prompt_version,
    )
    if not patch.affected_task_types:
        raise PatchStaticCheckError(
            "patch 未声明 affected_task_types，静态检查失败（§8）"
        )
    return patch


def new_revision_from_patch(current: CandidateRevision, patch: GPT6Patch) -> CandidateRevision:
    """由 patch 产出新版本候选（绝不原地修改，硬约束 11；parent_version 链）。"""
    if patch.target_revision_id != current.revision_id:
        raise PatchStaticCheckError(
            f"patch 目标 {patch.target_revision_id} 与当前版本 {current.revision_id} 不一致"
        )
    return CandidateRevision(
        revision_id=f"rev-{uuid.uuid4().hex[:12]}",
        root_candidate_id=current.root_candidate_id,
        parent_version=current.revision_id,  # 版本 DAG：父链
        candidate_type=current.candidate_type,
        spec_content=current.spec_content + "\n\n# PATCH\n" + patch.patch_content,
        status="draft",
        induction_trace_refs=list(current.induction_trace_refs),
        evidence_lineage_ref=current.evidence_lineage_ref,
        created_by=CREATED_BY_OFFLINE_REVISION,  # type: ignore[arg-type]
        created_at=datetime.now(timezone.utc).isoformat(),
        source_split=current.source_split,
        experience_relation=current.experience_relation,
        source_path=current.source_path,
        source_sha256=current.source_sha256,
        generated_spec_path=current.generated_spec_path,
        generated_sha256=current.generated_sha256,
        manifest_ref=current.manifest_ref,
        candidate_record_ref=current.candidate_record_ref,
    )


def revise_from_bundle(current: CandidateRevision,
                       bundle: CounterexampleBundle,
                       offline_client,
                       rejected_similar: list[str] | None = None) -> CandidateRevision:
    """完整修订链：可见摘要（不含答案）→ prompt → 离线模型 → patch → 新版本。

    异常按 §3.4 归族（`OfflineAuthError` / `OfflineServiceUnavailable` / ...）由调用方
    （`evolution/offline_driver.py`）映射为"暂停/quarantine"，本函数不吞异常、不返回空串。
    """
    prompt = build_revision_prompt(
        current.spec_content, bundle.gpt6_visible_summary, rejected_similar)
    resp = offline_client.chat(prompt)
    patch = parse_patch(resp, target_revision_id=current.revision_id,
                        model_id=str(getattr(offline_client, "model_id", "") or ""))
    return new_revision_from_patch(current, patch)


__all__ = [
    "CREATED_BY_OFFLINE_REVISION",
    "GPT6_REVISE_PROMPT_VERSION",
    "REVISE_PROMPT_VERSION",
    "PatchStaticCheckError",
    "build_revision_prompt",
    "new_revision_from_patch",
    "parse_patch",
    "revise_from_bundle",
]
