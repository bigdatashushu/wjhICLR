"""M16 基于反例的 GPT-6 修订：CounterexampleBundle → GPT6Patch → 新版本 CandidateRevision。

红线（硬约束 11）：绝不原地修改候选，修订必须产新版本（parent_version 链）。
GPT-6 只见 gpt6_visible_summary（不含答案，硬约束 19）。
patch 必须声明 affected_task_types，否则静态检查失败（§8）。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from skill3d.schemas import CandidateRevision, CounterexampleBundle, GPT6Patch
from skill3d.evolution.firewall import scan_prompt_for_leakage

GPT6_REVISE_PROMPT_VERSION = "revise-v1"  # prompt 模板版本化


class PatchStaticCheckError(ValueError):
    """patch 静态检查失败（如未声明 affected_task_types）。"""


def build_revision_prompt(current_spec_content: str,
                          gpt6_visible_summary: str,
                          rejected_similar: list[str] | None = None) -> str:
    """构造修订 prompt：只含当前 spec + 反例摘要（不含答案）+ 已拒相似候选。"""
    lines = [
        "你是离线 Skill 修订器。基于反例摘要给出结构化 patch（JSON）。",
        "禁止包含任何具体题目答案或 sample ID。",
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
                prompt_version: str = GPT6_REVISE_PROMPT_VERSION) -> GPT6Patch:
    """解析 GPT-6 输出为 GPT6Patch，并做静态检查：affected_task_types 必填。"""
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
        created_by="gpt6_revision",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def gpt6_revise(current: CandidateRevision,
                bundle: CounterexampleBundle,
                gpt6_client,
                rejected_similar: list[str] | None = None) -> CandidateRevision:
    """完整修订链：bundle.gpt6_visible_summary（不含答案）→ prompt → GPT-6 → patch → 新版本。"""
    prompt = build_revision_prompt(
        current.spec_content, bundle.gpt6_visible_summary, rejected_similar)
    resp = gpt6_client.chat(prompt)
    patch = parse_patch(resp, target_revision_id=current.revision_id)
    return new_revision_from_patch(current, patch)
