"""M16 语义审查：GPT-6 写 SkillGovernanceDecision。

硬约束 2/13：GPT-6 只审查、不决定 promote/reject；promote 由 evolution/admission.py
的确定性硬门一票否决。本模块输出仅作为 AdmissionDecision.gpt6_review 的事后引用。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from skill3d.schemas import CandidateRevision, SkillGovernanceDecision
from skill3d.evolution.firewall import scan_prompt_for_leakage

GOVERNANCE_PROMPT_VERSION = "governance-v1"  # prompt 模板版本化


def build_review_prompt(candidate: CandidateRevision) -> str:
    """构造审查 prompt：只含候选 spec（不得含答案）。"""
    prompt = (
        "你是离线语义审查器。审查以下候选模板的语义风险与可泛化性，"
        "输出 JSON：review_summary / semantic_risk / generalization_notes。"
        "你不决定 promote/reject。\n## 候选 spec_content\n"
        + candidate.spec_content
    )
    scan_prompt_for_leakage(prompt)
    return prompt


def semantic_review(candidate: CandidateRevision, gpt6_client) -> SkillGovernanceDecision:
    """写 SkillGovernanceDecision（只审查不决定；model_id 默认 TODO_USER_INPUT）。"""
    prompt = build_review_prompt(candidate)
    resp = gpt6_client.chat(prompt)
    data = json.loads(resp)
    return SkillGovernanceDecision(
        decision_id=f"gov-{uuid.uuid4().hex[:12]}",
        candidate_id=candidate.revision_id,
        review_summary=data.get("review_summary", ""),
        semantic_risk=data.get("semantic_risk", ""),
        generalization_notes=data.get("generalization_notes", ""),
        gpt6_model_id=getattr(gpt6_client, "model_id", "TODO_USER_INPUT"),
        gpt6_prompt_version=GOVERNANCE_PROMPT_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
