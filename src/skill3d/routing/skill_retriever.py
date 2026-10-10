"""v11: canonical question type selects its unique active method."""
from __future__ import annotations

import logging
from typing import NamedTuple, Optional, Sequence
from skill3d.schemas import RetrievedSkill, SceneState, SkillSpecV11
from skill3d.schemas.retrieval import SkillCandidateRecord, SkillRetrievalRecord
from skill3d.skills.delivery import skill_body_length, skill_content_sha256, skill_version_key
from .retrieval_policy import RetrievalPolicy
from .task_classifier import canonical_task

logger = logging.getLogger(__name__)


class RetrievalDecision(NamedTuple):
    """单条 Skill 的检索判定结果（可审计：为什么检索/为什么被拦下）。

    v9（§13.6"候选及过滤原因"）：除人类可读的 `reason` 外，新增机器可读的
    `reason_code`（词表见 `schemas.retrieval.CANDIDATE_REASON_CODES`）。此前原因
    只写进日志，trace 里看不到"这条候选为什么没被选中"。
    """

    ok: bool
    matched_signature: dict[str, str]
    gate_version_matched: Optional[bool]
    reason: str
    reason_code: str = ""

    def code(self) -> str:
        """原因码（缺省回落到 hit / 未登记，绝不猜一个具体原因）。"""
        return str(self.reason_code or ("hit" if self.ok else "unregistered"))


def canonical_question_type(value: Optional[str]) -> Optional[str]:
    """原始 question_type / 规范题型 → 规范题型（8 类之一）；空或未知 → None。

    None = 题型不可知 → 调用方按 fail-closed 处理（不检索任何 Skill，§17.1）。
    未知题型只告警不抛错：上游数据的题型错字不应让整条在线链崩掉。
    """
    if not value:
        return None
    try:
        return canonical_task(str(value))
    except Exception as exc:  # noqa: BLE001 - 未知题型 → 走 fail-closed
        logger.warning("M7 检索：未知 question_type=%r（%s）→ 不检索任何 Skill",
                       value, type(exc).__name__)
        return None


def retrieval_decision(
    skill: SkillSpecV11, scene: SceneState, task_type: str,
) -> RetrievalDecision:
    """Method selection depends on question type; tools enforce evidence requirements."""
    if not isinstance(skill, SkillSpecV11):
        raise ValueError("当前检索只接受 SkillSpecV11")
    expected = canonical_question_type(skill.question_type)
    if not expected or task_type != expected:
        return RetrievalDecision(
            False, {}, None,
            f"题型不匹配（题目={task_type}，Skill={skill.question_type}）",
            "question_type_mismatch")
    return RetrievalDecision(True, {}, None, "v11 题型唯一 active Skill", "hit")


def retrieve_ex(
    question: str,
    scene: SceneState,
    skills: Sequence[SkillSpecV11],
    question_type: Optional[str] = None,
    *,
    policy: Optional[RetrievalPolicy] = None,
    trigger: str = "initial",
    retrieval_index: int = 1,
    evidence_version: str = "",
    snapshot_ref: str = "",
    snapshot_manifest_sha256: str = "",
) -> tuple[list[RetrievedSkill], SkillRetrievalRecord]:
    """Record the same v11 lookup for empty and populated Skill collections."""
    skills = list(skills)
    if any(not isinstance(skill, SkillSpecV11) for skill in skills):
        raise ValueError("当前检索只接受 SkillSpecV11")
    return _retrieve_v11_ex(
        scene, skills, question_type=question_type,
        policy=policy or RetrievalPolicy(), trigger=trigger,
        retrieval_index=retrieval_index, evidence_version=evidence_version,
        snapshot_ref=snapshot_ref,
        snapshot_manifest_sha256=snapshot_manifest_sha256)


def _retrieve_v11_ex(
    scene: SceneState,
    skills: Sequence[SkillSpecV11],
    *,
    question_type: Optional[str],
    trigger: str,
    retrieval_index: int,
    evidence_version: str,
    snapshot_ref: str,
    snapshot_manifest_sha256: str,
    policy: RetrievalPolicy,
) -> tuple[list[RetrievedSkill], SkillRetrievalRecord]:
    """v11 deterministic lookup: canonical question type -> one active method."""
    task_type = canonical_question_type(question_type)
    record = SkillRetrievalRecord(
        retrieval_index=int(retrieval_index),
        trigger=str(trigger),
        canonical_question_type=str(task_type or ""),
        question_type_raw=str(question_type or ""),
        question_type_known=bool(task_type),
        partition_policy="v11_unique_active_by_question_type",
        evidence_version=str(evidence_version or ""),
        config_version=policy.version(),
        config_sha256=policy.sha256(),
        config_source=policy.source,
        policy=policy.to_dict(),
        active_snapshot_ref=str(snapshot_ref or ""),
        active_snapshot_manifest_sha256=str(snapshot_manifest_sha256 or ""),
        n_skills_offered=len(skills),
    )
    rows: list[SkillCandidateRecord] = []
    selected: list[SkillSpecV11] = []
    for skill in skills:
        decision = retrieval_decision(skill, scene, str(task_type or ""))
        is_selected = bool(task_type and decision.ok)
        row = SkillCandidateRecord(
            skill_id=skill.skill_id,
            version=skill.version,
            skill_version=skill_version_key(skill),
            canonical_question_type=skill.question_type,
            hard_filter_passed=bool(decision.ok),
            reason_code=(
                "question_type_unknown" if not task_type else decision.reason_code
            ),
            reason=(
                "题型不可知，不加载 v11 Skill" if not task_type else decision.reason
            ),
            score=(1.0 if is_selected else None),
            rank=(1 if is_selected else None),
            selected=is_selected,
            content_sha256=skill_content_sha256(skill),
            content_chars=skill_body_length(skill),
            matched_evidence_signature={},
            gate_version_matched=None,
            delivered=False,
            delivery_reason=("no_model_request" if is_selected else "not_selected"),
        )
        rows.append(row)
        if is_selected:
            selected.append(skill)
    if len(selected) > 1:
        raise ValueError(
            f"v11 题型 {task_type} 存在多个 active Skill: "
            f"{[skill_version_key(skill) for skill in selected]}")
    record.candidates = rows
    record.eligible_skill_versions = [
        row.skill_version for row in rows if row.hard_filter_passed
    ]
    record.retrieved_skill_versions = [
        skill_version_key(skill) for skill in selected
    ]
    record.delivery_channel = "not_sent"
    record.delivery_note = (
        "v11 题型唯一 Skill 已确定，尚未发出模型请求"
        if selected else
        "v11 当前题型没有 active Skill，未发出 Skill 请求"
    )
    hits = [
        RetrievedSkill(
            skill_id=skill.skill_id,
            skill_version=skill_version_key(skill),
            score=1.0,
            hard_filter_passed=True,
            matched_evidence_signature={},
            gate_version_matched=None,
        )
        for skill in selected
    ]
    return hits, record
