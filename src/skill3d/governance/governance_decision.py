"""M16 语义审查：离线模型写 `DeepSeekGovernanceDecision`（v6 §3.3/§3.4）。

硬约束 2/13 + §3.3：离线强模型（DeepSeek-V4.1-Flash）**只审查**，**不决定**
promote/reject；promote 由 `evolution/admission.py` 的确定性硬门 + 预注册规则决定，
一票否决。本模块产物带 `advisory_only: Literal[True] = True`（Schema 层就写不进
"这是一个决定"），且**不 import** skills / memory——模型文本无法直接修改 active
Skill/Memory。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from skill3d.schemas import CandidateRevision, DeepSeekGovernanceDecision
from skill3d.evolution.firewall import scan_prompt_for_leakage

# prompt 模板版本（冻结配置项，进 RunManifest 的 prompt_version；§3.4）
GOVERNANCE_PROMPT_VERSION = "governance-v1"
# v5 名称别名（过渡期只读）
GPT6_GOVERNANCE_PROMPT_VERSION = GOVERNANCE_PROMPT_VERSION

# 离线模型**不得**输出的字段名：模型若在 JSON 里塞"准入结论"，一律丢弃（只留在
# review_summary 文本里供人读），保证"文本不构成决定"（§3.3）。
_FORBIDDEN_DECISION_KEYS = frozenset({
    "promote", "promotes", "promoted", "decision", "admission_decision",
    "reject", "rejected", "quarantine", "quarantined", "status", "verdict",
})


def build_review_prompt(candidate: CandidateRevision) -> str:
    """构造审查 prompt：只含候选 spec（不得含答案）。"""
    prompt = (
        "你是 harness3D 的离线语义审查器（DeepSeek-V4.1-Flash）。"
        "审查以下候选模板的语义风险与可泛化性，"
        "输出 JSON：review_summary / semantic_risk / generalization_notes。"
        "你**不**决定 promote/reject，也不执行实验或评分；"
        "不要输出任何 promote/reject/status 字段。\n## 候选 spec_content\n"
        + candidate.spec_content
    )
    scan_prompt_for_leakage(prompt)
    return prompt


def _sanitize_review_payload(data: dict) -> dict:
    """丢弃任何"准入结论"字段（§3.3：离线文本不构成 promote/reject 决定）。"""
    clean = {k: v for k, v in data.items()
             if str(k).lower() not in _FORBIDDEN_DECISION_KEYS}
    dropped = sorted(set(data) - set(clean))
    if dropped:
        note = f"（已丢弃离线模型输出的准入类字段: {dropped}）"
        clean["review_summary"] = f"{clean.get('review_summary', '')}{note}"
    return clean


def semantic_review(candidate: CandidateRevision,
                    offline_client) -> DeepSeekGovernanceDecision:
    """写 `DeepSeekGovernanceDecision`（只审查不决定；`advisory_only` 恒 True）。

    离线模型身份（`offline_model` / `provider` / `model_id` / `prompt_version`）如实落盘，
    认证值与 endpoint 本体**绝不出现**在该 Schema 里（§3.4）。
    """
    prompt = build_review_prompt(candidate)
    resp = offline_client.chat(prompt)
    data = _sanitize_review_payload(json.loads(resp))
    return DeepSeekGovernanceDecision(
        decision_id=f"gov-{uuid.uuid4().hex[:12]}",
        candidate_id=candidate.revision_id,
        review_summary=data.get("review_summary", ""),
        semantic_risk=data.get("semantic_risk", ""),
        generalization_notes=data.get("generalization_notes", ""),
        provider=str(getattr(offline_client, "provider", "") or
                     DeepSeekGovernanceDecision.model_fields["provider"].default),
        model_id=str(getattr(offline_client, "model_id", "") or
                     DeepSeekGovernanceDecision.model_fields["model_id"].default),
        prompt_version=GOVERNANCE_PROMPT_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


__all__ = [
    "GOVERNANCE_PROMPT_VERSION",
    "GPT6_GOVERNANCE_PROMPT_VERSION",
    "build_review_prompt",
    "semantic_review",
]
