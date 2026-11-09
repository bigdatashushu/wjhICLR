"""v11 complete-Skill evolution receipts and resumable campaign state."""

from __future__ import annotations

import hashlib
import re
from typing import Literal, Optional

from pydantic import Field, field_validator, model_serializer, model_validator

from . import Spec
from .data_access import DataAccessRecord, validate_learning_authorization
from .skill import SkillCandidateV11
from .evidence import SkillEvidenceState, skill_evidence_state

V11_CAMPAIGN_SCHEMA_VERSION = "skill-evolution-campaign-v11/2.0"
V11_EXPERIENCE_SCHEMA_VERSION = "skill-experience-bundle-v11/6.0"
V11_EVALUATION_SCHEMA_VERSION = "skill-paired-evaluation-v11/3.0"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: str, *, field_name: str) -> str:
    value = str(value or "")
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} 必须是 64 位小写 sha256")
    return value


class V11ExperienceCase(Spec):
    """One parent-learning case with the actual delivered method and outcome."""

    case_id: str
    qa_id: str
    scene_id: str
    question_type: str
    evidence_state: SkillEvidenceState
    question_text: str
    options: list[str] = Field(default_factory=list)
    source_split: Literal["induction", "inner_validation"]
    learning_authorization: dict = Field(default_factory=dict)
    split: Literal["learning"]
    label_access: Literal[True]
    outcome: Literal["success", "failure"]
    episode_status: str
    delivered_skill_version: str
    delivered_content_sha256: str
    delivered_skill_md: str
    response_text: str
    response_texts: list[str]
    program_source: str
    tool_observations: list[dict] = Field(default_factory=list)
    initial_evidence_states: dict[str, str]
    evidence_states: dict[str, str]
    evidence_history: list[dict]
    candidate_objects: list[dict] = Field(default_factory=list)
    selected_object_ids: list[str] = Field(default_factory=list)
    image_observations: dict = Field(default_factory=dict)
    rounds: list[dict] = Field(default_factory=list)
    image_refs: list[dict] = Field(default_factory=list)
    frame_set_hash: str = ""
    model_answer: Optional[str] = None
    reference_answer: str
    score: float = Field(ge=0.0, le=1.0)
    failure_categories: list[str] = Field(default_factory=list)
    error_direction: str = ""
    trace_ref: str

    @field_validator(
        "case_id",
        "qa_id",
        "scene_id",
        "question_type",
        "question_text",
        "episode_status",
        "delivered_skill_version",
        "reference_answer",
        "trace_ref",
    )
    @classmethod
    def _required_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("经验身份字段不能为空")
        return value

    @field_validator("delivered_content_sha256")
    @classmethod
    def _valid_delivery_sha(cls, value: str) -> str:
        return _require_sha256(value, field_name="delivered_content_sha256")

    @field_validator("frame_set_hash")
    @classmethod
    def _valid_optional_frame_sha(cls, value: str) -> str:
        return _require_sha256(value, field_name="frame_set_hash") if value else ""

    @model_validator(mode="after")
    def _delivery_body_matches_hash(self):
        from .evidence import EvidenceProfile

        if self.learning_authorization:
            validate_learning_authorization(self.learning_authorization, [self.qa_id])
            access = ((self.rounds[0].get("experience_provenance") or {}).get(
                "offline_label_access_record") if self.rounds else None)
            if not access:
                raise ValueError("authorized learning case requires its actual label access")
            access = DataAccessRecord.model_validate(access)
            if (access.split != self.source_split or access.n_items != 1
                    or access.learning_authorization != self.learning_authorization
                    or not access.at or not access.input_manifest_sha256):
                raise ValueError("case learning authorization/label access mismatch")
        elif self.source_split != "induction":
            raise ValueError("non-induction case requires explicit learning authorization")
        for states in (self.initial_evidence_states, self.evidence_states):
            EvidenceProfile.model_validate(states)
        if skill_evidence_state(self.initial_evidence_states) != self.evidence_state:
            raise ValueError("经验初始证据状态与所属槽位不一致")
        if not self.evidence_history:
            raise ValueError("经验必须记录首次证据状态及后续变化")
        for row in self.evidence_history:
            EvidenceProfile.model_validate(row.get("evidence_states", {}))
        if (self.initial_evidence_states != self.evidence_history[0]["evidence_states"]
                or self.evidence_states != self.evidence_history[-1]["evidence_states"]):
            raise ValueError("经验首末证据状态与历史快照不一致")
        if not self.delivered_skill_md:
            raise ValueError("经验必须记录实际交付的完整 SKILL.md")
        actual = hashlib.sha256(self.delivered_skill_md.encode("utf-8")).hexdigest()
        if actual != self.delivered_content_sha256:
            raise ValueError("经验中的实际交付正文与 delivered_content_sha256 不一致")
        if not self.response_texts or self.response_text != self.response_texts[-1]:
            raise ValueError("response_text 必须等于非空 response_texts 的最后一轮")
        if self.outcome == "success" and self.score != 1.0:
            raise ValueError("outcome=success 要求 score=1.0")
        if self.outcome == "failure" and self.score >= 1.0:
            raise ValueError("outcome=failure 要求 score<1.0")
        if self.outcome == "failure" and not (
            self.failure_categories or self.error_direction
        ):
            raise ValueError("失败经验必须记录 failure_categories 或 error_direction")
        return self


class V11ExperienceBundle(Spec):
    """Parent-version learning evidence consumed by one offline revision."""

    schema_version: Literal[
        "skill-experience-bundle-v11/6.0",
        "skill-experience-bundle-v11/5.0",
    ] = (
        V11_EXPERIENCE_SCHEMA_VERSION
    )
    campaign_id: str
    parent_snapshot_id: str
    parent_skill_version: str
    parent_content_sha256: str
    question_type: str
    evidence_state: SkillEvidenceState
    source_split: Literal["induction", "inner_validation"] | list[
        Literal["induction", "inner_validation"]
    ]
    learning_authorization: dict = Field(default_factory=dict)
    split: Literal["learning"]
    label_access: Literal[True]
    source_run_ref: str
    label_access_record: DataAccessRecord
    label_access_records: list[DataAccessRecord] = Field(default_factory=list)
    cases: list[V11ExperienceCase]

    @field_validator(
        "campaign_id",
        "parent_snapshot_id",
        "parent_skill_version",
        "question_type",
        "source_run_ref",
    )
    @classmethod
    def _required_identity(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("经验包身份字段不能为空")
        return value

    @field_validator("parent_content_sha256")
    @classmethod
    def _valid_parent_sha(cls, value: str) -> str:
        return _require_sha256(value, field_name="parent_content_sha256")

    @model_validator(mode="after")
    def _cases_belong_to_parent(self):
        if not self.cases:
            raise ValueError("经验包至少需要一个父版本 learning case")
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("经验包 case_id 重复")
        sources = sorted({case.source_split for case in self.cases})
        expected_source = sources[0] if len(sources) == 1 else sources
        if self.source_split != expected_source:
            raise ValueError("bundle source_split must preserve actual case sources")
        if self.learning_authorization:
            validate_learning_authorization(
                self.learning_authorization, [case.qa_id for case in self.cases])
        elif any(case.learning_authorization for case in self.cases) or sources != ["induction"]:
            raise ValueError("experiment bundle requires explicit learning authorization")
        access = self.label_access_record
        accesses = self.label_access_records or [access]
        if access != accesses[0] or sorted(row.split for row in accesses) != sources:
            raise ValueError("bundle label access must cover every real source split")
        for row in accesses:
            cases = [case for case in self.cases if case.source_split == row.split]
            if (
                row.purpose != "skill_induction" or row.component_role != "inducer"
                or not row.label_access or row.n_items != len(cases)
                or not row.at or not row.input_manifest_sha256
            ):
                raise ValueError("经验包缺少与 cases 一致的真实 induction 标签访问记录")
            if self.learning_authorization:
                validate_learning_authorization(row.learning_authorization, [
                    case.qa_id for case in cases])
                if any(row.learning_authorization[key] != self.learning_authorization[key]
                       for key in self.learning_authorization if key != "qa_ids"):
                    raise ValueError("bundle label access authorization identity mismatch")
            elif row.learning_authorization:
                raise ValueError("experiment label access requires bundle learning authorization")
        for case in self.cases:
            if case.question_type != self.question_type:
                raise ValueError("经验 case 与经验包 question_type 不一致")
            if case.evidence_state != self.evidence_state:
                raise ValueError("经验 case 与经验包 evidence_state 不一致")
            if (
                case.split != self.split
                or case.label_access != self.label_access
            ):
                raise ValueError("经验 case 与经验包 split/label_access 不一致")
            if self.learning_authorization:
                validate_learning_authorization(case.learning_authorization, [case.qa_id])
                if any(case.learning_authorization[key] != self.learning_authorization[key]
                       for key in self.learning_authorization if key != "qa_ids"):
                    raise ValueError("case/bundle learning authorization identity mismatch")
            if case.delivered_skill_version != self.parent_skill_version:
                raise ValueError("经验 case 未实际交付目标父版本")
            if case.delivered_content_sha256 != self.parent_content_sha256:
                raise ValueError("经验 case 交付正文 hash 与父版本不一致")
        return self


class V11RevisionProposal(Spec):
    """Offline reviser output; framework identity and version are deliberately absent."""

    action: Literal["revise", "no_change"] = "revise"
    full_skill_md: Optional[str] = None
    modification_reason: str
    source_run_ref: str
    change_hypothesis: Optional[str] = None
    changed_sections: list[str] = Field(default_factory=list)
    expected_process_effect: Optional[str] = None
    evidence_signature_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _no_change_omits_source(cls, value):
        if isinstance(value, dict) and value.get("action") == "no_change" and (
            "full_skill_md" in value
        ):
            raise ValueError("action=no_change must omit full_skill_md")
        return value

    @model_serializer(mode="wrap")
    def _serialize_proposal(self, handler):
        value = handler(self)
        if self.action == "no_change":
            value.pop("full_skill_md", None)
        return value

    @field_validator("modification_reason", "source_run_ref")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not str(value or "").strip():
            raise ValueError("修订提案字段不能为空")
        return str(value)

    @field_validator("change_hypothesis", "expected_process_effect")
    @classmethod
    def _optional_nonempty(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("declared revision metadata must be nonempty")
        return value

    @field_validator("changed_sections", "evidence_signature_refs")
    @classmethod
    def _valid_declared_refs(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("declared revision references must be nonempty and unique")
        return values

    @model_validator(mode="after")
    def _revision_requires_complete_source(self):
        if self.action == "revise" and not (self.full_skill_md or "").strip():
            raise ValueError("action=revise requires nonempty full_skill_md")
        if self.action == "no_change" and self.full_skill_md is not None:
            raise ValueError("action=no_change must omit full_skill_md")
        return self


class V11RevisionAttemptReceipt(Spec):
    campaign_id: str
    attempt: int = Field(ge=1)
    parent_skill_version: str
    experience_sha256: str
    proposal: V11RevisionProposal
    format_problems: list[str] = Field(default_factory=list)
    format_candidate_text: Optional[str] = None

    @field_validator("experience_sha256")
    @classmethod
    def _valid_experience_sha(cls, value: str) -> str:
        return _require_sha256(value, field_name="experience_sha256")


class V11StaticValidationReceipt(Spec):
    campaign_id: str
    attempt: int = Field(ge=1)
    generated_version: str
    candidate_id: str
    raw_skill_md_sha256: str
    content_sha256: str
    source_diff: str
    diff_sha256: str
    checks: dict[str, bool]
    problems: list[str] = Field(default_factory=list)
    passed: bool
    candidate: Optional[SkillCandidateV11] = None

    @field_validator("raw_skill_md_sha256", "content_sha256", "diff_sha256")
    @classmethod
    def _valid_hashes(cls, value: str, info) -> str:
        return _require_sha256(value, field_name=info.field_name)

    @model_validator(mode="after")
    def _result_consistent(self):
        expected = not self.problems and bool(self.checks) and all(self.checks.values())
        if self.passed != expected:
            raise ValueError("静态校验 passed 与 checks/problems 不一致")
        if self.passed and self.candidate is None:
            raise ValueError("通过静态校验的收据必须包含完整候选")
        if self.candidate is not None:
            if self.candidate.candidate_id != self.candidate_id:
                raise ValueError("静态收据 candidate_id 与候选不一致")
            if self.candidate.full_skill_spec.content_sha256 != self.content_sha256:
                raise ValueError("静态收据 content_sha256 与候选不一致")
        return self


class V11EvaluationArm(Spec):
    skill_version: str
    content_sha256: str
    delivered_skill_version: str
    delivered_content_sha256: str
    delivery_observed: bool
    n_scored: int = Field(ge=0)
    mean_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    runtime_error_count: int = Field(ge=0)
    program_error_count: int = Field(ge=0)
    untrusted_geometry_use_count: int = Field(ge=0)
    legal_answer_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("content_sha256", "delivered_content_sha256")
    @classmethod
    def _valid_hashes(cls, value: str, info) -> str:
        return _require_sha256(value, field_name=info.field_name)


class V11PairedEvaluationReceipt(Spec):
    """Frozen same-question-type parent/candidate evaluation result."""

    schema_version: Literal["skill-paired-evaluation-v11/3.0"] = (
        V11_EVALUATION_SCHEMA_VERSION
    )
    campaign_id: str
    evaluation_id: str
    panel_id: str
    panel_sha256: str
    question_type: str
    evidence_state: SkillEvidenceState
    seed: int
    request_seed_observed: bool
    model_id: str
    model_config_sha256: str
    quality_contract_sha256: str
    solver_config_sha256: str
    template_version: str
    execution_protocol_version: str
    tool_docs_version: str
    tool_face_version: str
    status: Literal["completed", "incomplete"]
    formal_result_eligible: bool
    frozen_panel: bool
    independent_arms: bool
    n_pairs: int = Field(ge=0)
    result_refs: list[str]
    parent: V11EvaluationArm
    candidate: V11EvaluationArm

    @field_validator(
        "panel_sha256",
        "model_config_sha256",
        "quality_contract_sha256",
        "solver_config_sha256",
    )
    @classmethod
    def _valid_evaluation_sha(cls, value: str, info) -> str:
        return _require_sha256(value, field_name=info.field_name)

    @field_validator(
        "campaign_id",
        "evaluation_id",
        "panel_id",
        "question_type",
        "model_id",
        "template_version",
        "execution_protocol_version",
        "tool_docs_version",
        "tool_face_version",
    )
    @classmethod
    def _required_evaluation_identity(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("E11 评测身份字段不能为空")
        return value

    @model_validator(mode="after")
    def _counts_fit_panel(self):
        if len(self.result_refs) != self.n_pairs or len(set(self.result_refs)) != len(
            self.result_refs
        ):
            raise ValueError("result_refs 必须与 n_pairs 一一对应且不得重复")
        for label, arm in (("parent", self.parent), ("candidate", self.candidate)):
            if arm.n_scored > self.n_pairs:
                raise ValueError(f"{label}.n_scored 超过 n_pairs")
            if arm.runtime_error_count > self.n_pairs:
                raise ValueError(f"{label}.runtime_error_count 超过 n_pairs")
            if arm.program_error_count > self.n_pairs:
                raise ValueError(f"{label}.program_error_count 超过 n_pairs")
            if arm.untrusted_geometry_use_count > self.n_pairs:
                raise ValueError(f"{label}.untrusted_geometry_use_count 超过 n_pairs")
        return self


class V11CampaignDecision(Spec):
    campaign_id: str
    question_type: str
    evidence_state: SkillEvidenceState
    stage: Literal["revision", "static_validation", "paired_evaluation"]
    outcome: Literal["promote", "reject", "no_change"]
    parent_snapshot_id: str
    parent_skill_version: str
    candidate_id: str = ""
    candidate_skill_version: str = ""
    conditions: dict[str, bool]
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _decision_consistent(self):
        if self.outcome == "no_change":
            if (
                self.stage != "revision"
                or self.candidate_id
                or self.candidate_skill_version
                or self.conditions != {"revision_no_change": True}
                or not self.reasons
            ):
                raise ValueError("no_change requires a revision decision without a candidate")
            return self
        if self.stage == "revision":
            raise ValueError("revision decisions must have outcome=no_change")
        promotes = bool(self.conditions) and all(self.conditions.values())
        expected = "promote" if promotes else "reject"
        if self.outcome != expected:
            raise ValueError("准入 outcome 与 conditions 不一致")
        if self.outcome == "reject" and not self.reasons:
            raise ValueError("拒绝决策必须记录原因")
        return self


class V11PublicationReceipt(Spec):
    campaign_id: str
    question_type: str
    evidence_state: SkillEvidenceState
    candidate_id: str
    skill_version: str
    parent_skill_version: str
    snapshot_before: str
    snapshot_after: str
    manifest_hash: str
    diff_sha256: str
    rollback_ref: str

    @field_validator("manifest_hash", "diff_sha256")
    @classmethod
    def _valid_hashes(cls, value: str, info) -> str:
        return _require_sha256(value, field_name=info.field_name)


class V11RollbackReceipt(Spec):
    campaign_id: str
    snapshot_from: str
    snapshot_to: str
    pointer_after: str
    reason: str
    verified: bool

    @model_validator(mode="after")
    def _rollback_consistent(self):
        if self.verified != (self.pointer_after == self.snapshot_to):
            raise ValueError("回滚 verified 与 pointer_after 不一致")
        return self


class V11PostPublishObservation(Spec):
    """Result returned by a new-learning probe after publication."""

    question_type: str
    evidence_state: SkillEvidenceState
    snapshot_id: str
    retrieved_skill_version: str
    delivered_skill_version: str
    delivered_skill_md: str
    learning_event_refs: list[str] = Field(default_factory=list)
    source_run_ref: str


class V11PostPublishReceipt(Spec):
    campaign_id: str
    snapshot_id: str
    expected_skill_version: str
    expected_content_sha256: str
    observation: V11PostPublishObservation
    observed_content_sha256: str
    checks: dict[str, bool]
    problems: list[str] = Field(default_factory=list)
    verified: bool

    @field_validator("expected_content_sha256", "observed_content_sha256")
    @classmethod
    def _valid_hashes(cls, value: str, info) -> str:
        return _require_sha256(value, field_name=info.field_name)

    @model_validator(mode="after")
    def _verification_consistent(self):
        expected = not self.problems and bool(self.checks) and all(self.checks.values())
        if self.verified != expected:
            raise ValueError("发布后 verified 与 checks/problems 不一致")
        return self


class V11ReceiptRef(Spec):
    path: str
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _valid_sha(cls, value: str) -> str:
        return _require_sha256(value, field_name="sha256")


V11CampaignStatus = Literal[
    "initialized",
    "collecting",
    "revising",
    "evaluating",
    "deciding",
    "publishing",
    "verifying",
    "promoted",
    "rejected",
    "no_change",
    "blocked",
]


class V11CampaignCheckpoint(Spec):
    schema_version: Literal["skill-evolution-campaign-v11/2.0"] = (
        V11_CAMPAIGN_SCHEMA_VERSION
    )
    campaign_id: str
    status: V11CampaignStatus
    question_type: str
    evidence_state: SkillEvidenceState
    library_root: str
    parent_snapshot_id: str
    parent_skill_version: str
    parent_content_sha256: str
    candidate_id: str = ""
    candidate_skill_version: str = ""
    candidate_content_sha256: str = ""
    revision_attempt: int = Field(default=0, ge=0)
    decision: str = ""
    published_snapshot_id: str = ""
    receipts: dict[str, V11ReceiptRef] = Field(default_factory=dict)
    blocked_reason: str = ""
    created_at: str
    updated_at: str

    @field_validator("parent_content_sha256")
    @classmethod
    def _valid_parent_sha(cls, value: str) -> str:
        return _require_sha256(value, field_name="parent_content_sha256")

    @field_validator("candidate_content_sha256")
    @classmethod
    def _valid_optional_candidate_sha(cls, value: str) -> str:
        return (
            _require_sha256(value, field_name="candidate_content_sha256")
            if value
            else ""
        )


__all__ = [
    "V11_CAMPAIGN_SCHEMA_VERSION",
    "V11_EVALUATION_SCHEMA_VERSION",
    "V11_EXPERIENCE_SCHEMA_VERSION",
    "V11CampaignCheckpoint",
    "V11CampaignDecision",
    "V11CampaignStatus",
    "V11EvaluationArm",
    "V11ExperienceBundle",
    "V11ExperienceCase",
    "V11PairedEvaluationReceipt",
    "V11PostPublishObservation",
    "V11PostPublishReceipt",
    "V11PublicationReceipt",
    "V11ReceiptRef",
    "V11RollbackReceipt",
    "V11RevisionAttemptReceipt",
    "V11RevisionProposal",
    "V11StaticValidationReceipt",
]
