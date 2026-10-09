"""CPU-only orchestration for resumable v11 complete-Skill evolution.

The expensive learning and evaluation implementations are injected callbacks.
This module owns identities, static validation, admission, publication, and
create-once receipts; it never calls a GPU service by itself.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel

from skill3d.memory.consolidation import leakage_scan_text
from skill3d.schemas import (
    SkillCandidateV11,
    SkillSpecV11,
    V11CampaignCheckpoint,
    V11CampaignDecision,
    V11ExperienceBundle,
    V11PairedEvaluationReceipt,
    V11PostPublishObservation,
    V11PostPublishReceipt,
    V11PublicationReceipt,
    V11ReceiptRef,
    V11RevisionAttemptReceipt,
    V11RevisionProposal,
    V11StaticValidationReceipt,
)
from skill3d.skills.delivery import (
    DEFAULT_METHOD_CONTEXT_MAX_CHARS,
    check_service_limit,
)
from skill3d.skills.promote_atomic import (
    publish_lock,
    read_active_snapshot,
    read_snapshot,
)
from skill3d.skills.registry import bump_semver, load_active_skills
from skill3d.skills.v11_library import (
    V11LibraryError,
    V11_SNAPSHOT_SCHEMA,
    publish_v11_candidate,
    resolve_source_like_ref,
    validate_v11_snapshot,
    write_v11_snapshot,
)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_CALL_RE = re.compile(r"(?<![.\w])([a-z][a-z0-9_]{2,})\s*\(")
_TOOLS_CALL_RE = re.compile(r"\btools\.([a-z][a-z0-9_]*)\s*\(")
_LOCAL_DEF_RE = re.compile(r"\bdef\s+([a-z][a-z0-9_]*)\s*\(")
_NON_TOOL_CALLS = frozenset({
    "abs", "all", "any", "array", "assert", "atan2", "bool", "ceil", "dict",
    "dot", "enumerate", "exp", "filter", "float", "floor", "format", "getattr",
    "int", "isinstance", "join", "len", "list", "log", "map", "max", "mean",
    "median", "min", "norm", "open", "pow", "print", "range", "round", "set",
    "sorted", "sqrt", "str", "sum", "tuple", "type", "zip",
})

ModelT = TypeVar("ModelT", bound=BaseModel)


class V11CampaignBlocked(RuntimeError):
    """The campaign cannot proceed without violating a v11 contract."""


@dataclass(frozen=True)
class V11CampaignConfig:
    """Stable orchestration inputs; model/evaluator details live in callbacks."""

    campaign_id: str = ""
    question_type: str = "object_counting"
    library_root: str = "data/evolution/library_v11"
    run_root: str = "data/evolution/runs_v11"
    max_revision_attempts: int = 3
    method_context_max_chars: int = DEFAULT_METHOD_CONTEXT_MAX_CHARS
    resume: bool = True

    def resolved_id(self) -> str:
        campaign_id = self.campaign_id or f"camp-v11-{uuid.uuid4().hex[:12]}"
        if not _SAFE_ID_RE.fullmatch(campaign_id):
            raise ValueError(f"campaign_id 不是安全标识符: {campaign_id!r}")
        return campaign_id


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: object) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_create_once(path: Path, data: bytes) -> None:
    """Atomically create an immutable receipt, accepting an identical retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise V11CampaignBlocked(f"不可变收据已存在且内容不同: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _coerce(model_type: type[ModelT], value: object) -> ModelT:
    if isinstance(value, model_type):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return model_type.model_validate(value)


def _candidate_id(skill_id: str, version: str, content_sha256: str) -> str:
    return f"cand-{skill_id.lower()}-{version}-{content_sha256[:16]}"


def _unknown_tool_calls(skill_md: str) -> list[str]:
    """Return call-like names that are neither registered Tools nor local helpers."""
    # Importing the package registers geometry and vision tools by side effect.
    from skill3d.tools import REGISTRY

    calls = set(_CALL_RE.findall(skill_md)) | set(_TOOLS_CALL_RE.findall(skill_md))
    local_helpers = set(_LOCAL_DEF_RE.findall(skill_md))
    known = set(REGISTRY.names())
    return sorted(calls - known - local_helpers - set(_NON_TOOL_CALLS))


def _normalized_text(value: str) -> str:
    return " ".join(str(value or "").split()).casefold()


def _sample_copy_leaks(
    skill_md: str,
    bundle: V11ExperienceBundle,
) -> list[str]:
    """Detect verbatim long sample content without rejecting generic short labels."""
    candidate = _normalized_text(skill_md)
    hits: list[str] = []
    for case in bundle.cases:
        question = _normalized_text(case.question_text)
        if len(question) >= 24 and question in candidate:
            hits.append(f"question_text:{case.case_id}")
        reference = _normalized_text(case.reference_answer)
        if len(reference) >= 16 and reference in candidate:
            hits.append(f"reference_answer:{case.case_id}")
        for index, option in enumerate(case.options):
            normalized = _normalized_text(option)
            if len(normalized) >= 24 and normalized in candidate:
                hits.append(f"option:{case.case_id}:{index}")
    return hits


class V11CampaignRunner:
    """Four-stage v11 campaign with injectable learning/revision/evaluation."""

    def __init__(
        self,
        cfg: V11CampaignConfig,
        *,
        collector: Callable[..., object],
        reviser: Callable[..., object],
        evaluator: Callable[..., object],
        post_publish_verifier: Callable[..., object],
    ) -> None:
        if int(cfg.max_revision_attempts) <= 0:
            raise ValueError("max_revision_attempts 必须为正整数")
        if int(cfg.method_context_max_chars) <= 0:
            raise ValueError("method_context_max_chars 必须为正整数")
        self.cfg = cfg
        self.campaign_id = cfg.resolved_id()
        self.library_root = Path(cfg.library_root).resolve()
        self.store_dir = self.library_root / "snapshots"
        self.root = Path(cfg.run_root).resolve() / self.campaign_id
        self.collector = collector
        self.reviser = reviser
        self.evaluator = evaluator
        self.post_publish_verifier = post_publish_verifier
        self.checkpoint: V11CampaignCheckpoint | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "campaign.json"

    def _receipt_path(self, key: str) -> Path:
        if not _SAFE_ID_RE.fullmatch(key):
            raise V11CampaignBlocked(f"非法收据键: {key!r}")
        return self.root / "receipts" / f"{key}.json"

    def _save_checkpoint(self) -> None:
        if self.checkpoint is None:
            raise RuntimeError("checkpoint 尚未初始化")
        self.checkpoint.updated_at = _now_iso()
        _write_atomic(self.checkpoint_path, _json_bytes(self.checkpoint))

    def _set_status(self, status: str) -> None:
        assert self.checkpoint is not None
        self.checkpoint.status = status  # type: ignore[assignment]
        self._save_checkpoint()

    def _block(self, reason: str) -> None:
        assert self.checkpoint is not None
        self.checkpoint.status = "blocked"
        self.checkpoint.blocked_reason = str(reason)
        self._save_checkpoint()

    def _record_receipt(self, key: str, receipt: BaseModel) -> None:
        assert self.checkpoint is not None
        path = self._receipt_path(key)
        data = _json_bytes(receipt)
        _write_create_once(path, data)
        relative = path.relative_to(self.root).as_posix()
        ref = V11ReceiptRef(path=relative, sha256=_sha256(data))
        existing = self.checkpoint.receipts.get(key)
        if existing is not None and existing != ref:
            raise V11CampaignBlocked(f"checkpoint 中的收据引用发生变化: {key}")
        self.checkpoint.receipts[key] = ref
        self._save_checkpoint()

    def _load_receipt(
        self,
        key: str,
        model_type: type[ModelT],
    ) -> ModelT | None:
        assert self.checkpoint is not None
        path = self._receipt_path(key)
        ref = self.checkpoint.receipts.get(key)
        if not path.exists():
            if ref is not None:
                raise V11CampaignBlocked(f"checkpoint 引用的收据不存在: {ref.path}")
            return None
        data = path.read_bytes()
        actual = V11ReceiptRef(
            path=path.relative_to(self.root).as_posix(),
            sha256=_sha256(data),
        )
        if ref is not None and ref != actual:
            raise V11CampaignBlocked(f"收据 hash/path 与 checkpoint 不一致: {key}")
        try:
            receipt = model_type.model_validate_json(data)
        except Exception as exc:
            raise V11CampaignBlocked(f"收据 {key} 损坏: {exc}") from exc
        if ref is None:
            self.checkpoint.receipts[key] = actual
            self._save_checkpoint()
        return receipt

    def _initialize(self) -> tuple[SkillSpecV11, dict]:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_path.exists():
            if not self.cfg.resume:
                raise V11CampaignBlocked(
                    f"campaign 已存在且 resume=False: {self.checkpoint_path}"
                )
            try:
                self.checkpoint = V11CampaignCheckpoint.model_validate_json(
                    self.checkpoint_path.read_bytes()
                )
            except Exception as exc:
                raise V11CampaignBlocked(f"campaign checkpoint 损坏: {exc}") from exc
            cp = self.checkpoint
            if cp.campaign_id != self.campaign_id:
                raise V11CampaignBlocked("checkpoint campaign_id 不一致")
            if cp.question_type != self.cfg.question_type:
                raise V11CampaignBlocked("checkpoint question_type 与配置不一致")
            if Path(cp.library_root).resolve() != self.library_root:
                raise V11CampaignBlocked("checkpoint library_root 与配置不一致")
            parent, snapshot = self._load_frozen_parent(cp.parent_snapshot_id)
            if (
                f"{parent.skill_id}@{parent.version}" != cp.parent_skill_version
                or parent.content_sha256 != cp.parent_content_sha256
            ):
                raise V11CampaignBlocked("checkpoint 父版本身份与不可变快照不一致")
            return parent, snapshot

        receipts = self.root / "receipts"
        if receipts.exists() and any(receipts.iterdir()):
            raise V11CampaignBlocked("存在收据但缺少 campaign checkpoint，拒绝猜测恢复")
        snapshot = read_active_snapshot(self.store_dir)
        if snapshot.get("schema_version") != V11_SNAPSHOT_SCHEMA:
            raise V11CampaignBlocked("v11 campaign 只接受 runtime-skill-snapshot/2.0")
        skills, warnings, snapshot_id = load_active_skills(self.store_dir)
        if warnings:
            raise V11CampaignBlocked(f"父快照加载警告: {warnings}")
        matching = [
            skill for skill in skills if skill.question_type == self.cfg.question_type
        ]
        if len(matching) != 1:
            raise V11CampaignBlocked(
                f"题型 {self.cfg.question_type!r} 必须恰有一个 active v11 Skill"
            )
        parent = matching[0]
        now = _now_iso()
        self.checkpoint = V11CampaignCheckpoint(
            campaign_id=self.campaign_id,
            status="initialized",
            question_type=self.cfg.question_type,
            library_root=str(self.library_root),
            parent_snapshot_id=snapshot_id,
            parent_skill_version=f"{parent.skill_id}@{parent.version}",
            parent_content_sha256=parent.content_sha256,
            created_at=now,
            updated_at=now,
        )
        self._save_checkpoint()
        return parent, snapshot

    def _load_frozen_parent(self, snapshot_id: str) -> tuple[SkillSpecV11, dict]:
        try:
            snapshot = read_snapshot(self.store_dir, snapshot_id)
            specs = validate_v11_snapshot(
                snapshot,
                library_root=self.library_root,
                method_context_max_chars=int(self.cfg.method_context_max_chars),
            )
        except Exception as exc:
            raise V11CampaignBlocked(
                f"无法重放 campaign 父快照 {snapshot_id!r}: {exc}"
            ) from exc
        matching = [
            spec for spec in specs if spec.question_type == self.cfg.question_type
        ]
        if len(matching) != 1:
            raise V11CampaignBlocked("父快照题型槽位不唯一")
        return matching[0], snapshot

    def _collect(self, parent: SkillSpecV11) -> V11ExperienceBundle:
        key = "experience_bundle"
        existing = self._load_receipt(key, V11ExperienceBundle)
        if existing is not None:
            self._validate_bundle(existing, parent)
            return existing
        self._set_status("collecting")
        value = self.collector(
            campaign_id=self.campaign_id,
            parent=parent.model_copy(deep=True),
            parent_snapshot_id=self.checkpoint.parent_snapshot_id,
            question_type=self.cfg.question_type,
            idempotency_key=f"{self.campaign_id}:collect",
        )
        bundle = _coerce(V11ExperienceBundle, value)
        self._validate_bundle(bundle, parent)
        self._record_receipt(key, bundle)
        return bundle

    def _validate_bundle(
        self,
        bundle: V11ExperienceBundle,
        parent: SkillSpecV11,
    ) -> None:
        assert self.checkpoint is not None
        expected = {
            "campaign_id": self.campaign_id,
            "parent_snapshot_id": self.checkpoint.parent_snapshot_id,
            "parent_skill_version": self.checkpoint.parent_skill_version,
            "parent_content_sha256": parent.content_sha256,
            "question_type": self.cfg.question_type,
        }
        actual = {
            "campaign_id": bundle.campaign_id,
            "parent_snapshot_id": bundle.parent_snapshot_id,
            "parent_skill_version": bundle.parent_skill_version,
            "parent_content_sha256": bundle.parent_content_sha256,
            "question_type": bundle.question_type,
        }
        if actual != expected:
            raise V11CampaignBlocked(
                f"经验包没有绑定 campaign 父版本: expected={expected}, actual={actual}"
            )
        if any(case.delivered_skill_md != parent.skill_md for case in bundle.cases):
            raise V11CampaignBlocked("经验包记录的实际交付正文不是父快照完整正文")

    def _proposal(
        self,
        parent: SkillSpecV11,
        bundle: V11ExperienceBundle,
        *,
        attempt: int,
        feedback: list[str],
    ) -> V11RevisionAttemptReceipt:
        key = f"revision_attempt_{attempt:02d}"
        experience_sha256 = _sha256(_json_bytes(bundle))
        existing = self._load_receipt(key, V11RevisionAttemptReceipt)
        if existing is not None:
            if (
                existing.campaign_id != self.campaign_id
                or existing.attempt != attempt
                or existing.parent_skill_version
                != self.checkpoint.parent_skill_version
                or existing.experience_sha256 != experience_sha256
            ):
                raise V11CampaignBlocked("修订收据与当前父版本/经验包不一致")
            return existing
        self._set_status("revising")
        value = self.reviser(
            campaign_id=self.campaign_id,
            parent=parent.model_copy(deep=True),
            experience=bundle.model_copy(deep=True),
            attempt=int(attempt),
            static_feedback=list(feedback),
            idempotency_key=f"{self.campaign_id}:revise:{attempt}",
        )
        proposal = _coerce(V11RevisionProposal, value)
        receipt = V11RevisionAttemptReceipt(
            campaign_id=self.campaign_id,
            attempt=int(attempt),
            parent_skill_version=self.checkpoint.parent_skill_version,
            experience_sha256=experience_sha256,
            proposal=proposal,
        )
        self._record_receipt(key, receipt)
        return receipt

    def _static_validate(
        self,
        parent: SkillSpecV11,
        bundle: V11ExperienceBundle,
        proposal_receipt: V11RevisionAttemptReceipt,
    ) -> V11StaticValidationReceipt:
        attempt = int(proposal_receipt.attempt)
        key = f"static_validation_{attempt:02d}"
        existing = self._load_receipt(key, V11StaticValidationReceipt)
        if existing is not None:
            return existing

        proposal = proposal_receipt.proposal
        raw = proposal.full_skill_md
        raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        version = bump_semver(parent.version, "MINOR")
        candidate_id = _candidate_id(parent.skill_id, version, raw_sha)
        source_diff = "".join(
            difflib.unified_diff(
                parent.skill_md.splitlines(keepends=True),
                raw.splitlines(keepends=True),
                fromfile=f"{parent.skill_id}@{parent.version}",
                tofile=f"{parent.skill_id}@{version}",
            )
        )
        diff_sha = hashlib.sha256(source_diff.encode("utf-8")).hexdigest()
        checks: dict[str, bool] = {}
        problems: list[str] = []

        checks["content_changed"] = raw != parent.skill_md
        if not checks["content_changed"]:
            problems.append("候选完整 SKILL.md 与父版本相同")

        candidate_spec: SkillSpecV11 | None = None
        candidate: SkillCandidateV11 | None = None
        try:
            candidate_spec = SkillSpecV11(
                skill_id=parent.skill_id,
                version=version,
                question_type=parent.question_type,
                skill_md=raw,
            )
            checks["complete_source_valid"] = True
        except Exception as exc:
            checks["complete_source_valid"] = False
            problems.append(f"完整 SKILL.md 非法: {type(exc).__name__}: {exc}")

        checks["same_lineage"] = candidate_spec is not None and (
            candidate_spec.skill_id == parent.skill_id
            and candidate_spec.question_type == parent.question_type
        )
        if not checks["same_lineage"]:
            problems.append("候选必须保持父 skill_id 和 question_type")
        checks["next_minor_version"] = candidate_spec is not None and (
            candidate_spec.version == version
        )
        if not checks["next_minor_version"]:
            problems.append("候选版本必须由框架生成下一 MINOR")

        service_problems = (
            check_service_limit(
                candidate_spec,
                max_chars=int(self.cfg.method_context_max_chars),
            )
            if candidate_spec is not None
            else ["候选无法构造，不能验证服务限制"]
        )
        checks["complete_body_deliverable"] = not service_problems
        problems.extend(service_problems)

        forbidden_ids = {
            token
            for case in bundle.cases
            for token in (case.case_id, case.qa_id, case.scene_id)
            if token
        }
        leaks = leakage_scan_text(raw, forbidden_sample_ids=forbidden_ids)
        sample_copy_leaks = _sample_copy_leaks(raw, bundle)
        leaks.extend(sample_copy_leaks)
        checks["no_leakage"] = not leaks
        if leaks:
            problems.append(f"候选命中泄漏模式: {sorted(set(leaks))}")

        unknown_tools = _unknown_tool_calls(raw)
        checks["tools_known"] = not unknown_tools
        if unknown_tools:
            problems.append(f"候选引用未知 Tool: {unknown_tools}")

        if candidate_spec is not None:
            try:
                candidate = SkillCandidateV11(
                    candidate_id=candidate_id,
                    parent_snapshot_id=self.checkpoint.parent_snapshot_id,
                    parent_skill_version=self.checkpoint.parent_skill_version,
                    full_skill_spec=candidate_spec,
                    modification_reason=proposal.modification_reason,
                    source_run_ref=proposal.source_run_ref,
                )
                checks["candidate_contract_valid"] = True
            except Exception as exc:
                checks["candidate_contract_valid"] = False
                problems.append(f"候选身份合同非法: {type(exc).__name__}: {exc}")
        else:
            checks["candidate_contract_valid"] = False
            problems.append("候选身份合同无法构造")

        receipt = V11StaticValidationReceipt(
            campaign_id=self.campaign_id,
            attempt=attempt,
            generated_version=version,
            candidate_id=candidate_id,
            raw_skill_md_sha256=raw_sha,
            content_sha256=(
                candidate_spec.content_sha256 if candidate_spec is not None else raw_sha
            ),
            source_diff=source_diff,
            diff_sha256=diff_sha,
            checks=checks,
            problems=problems,
            passed=not problems and all(checks.values()),
            candidate=candidate,
        )
        self._record_receipt(key, receipt)
        return receipt

    def _generate_candidate(
        self,
        parent: SkillSpecV11,
        bundle: V11ExperienceBundle,
    ) -> tuple[SkillCandidateV11 | None, V11StaticValidationReceipt]:
        feedback: list[str] = []
        last: V11StaticValidationReceipt | None = None
        for attempt in range(1, int(self.cfg.max_revision_attempts) + 1):
            proposal = self._proposal(
                parent,
                bundle,
                attempt=attempt,
                feedback=feedback,
            )
            last = self._static_validate(parent, bundle, proposal)
            self.checkpoint.revision_attempt = attempt
            self._save_checkpoint()
            if last.passed:
                assert last.candidate is not None
                candidate = last.candidate
                self.checkpoint.candidate_id = candidate.candidate_id
                self.checkpoint.candidate_skill_version = (
                    candidate.candidate_skill_version
                )
                self.checkpoint.candidate_content_sha256 = (
                    candidate.full_skill_spec.content_sha256
                )
                self._save_checkpoint()
                return candidate, last
            feedback = list(last.problems)
        assert last is not None
        return None, last

    def _evaluate(
        self,
        parent: SkillSpecV11,
        candidate: SkillCandidateV11,
        bundle: V11ExperienceBundle,
    ) -> V11PairedEvaluationReceipt:
        key = "paired_evaluation"
        existing = self._load_receipt(key, V11PairedEvaluationReceipt)
        if existing is not None:
            return existing
        self._set_status("evaluating")
        value = self.evaluator(
            campaign_id=self.campaign_id,
            parent=parent.model_copy(deep=True),
            candidate=candidate.model_copy(deep=True),
            experience=bundle.model_copy(deep=True),
            question_type=self.cfg.question_type,
            idempotency_key=f"{self.campaign_id}:evaluate",
        )
        receipt = _coerce(V11PairedEvaluationReceipt, value)
        self._record_receipt(key, receipt)
        return receipt

    def _static_rejection(
        self,
        last: V11StaticValidationReceipt,
    ) -> V11CampaignDecision:
        return V11CampaignDecision(
            campaign_id=self.campaign_id,
            stage="static_validation",
            outcome="reject",
            parent_snapshot_id=self.checkpoint.parent_snapshot_id,
            parent_skill_version=self.checkpoint.parent_skill_version,
            candidate_id=last.candidate_id,
            candidate_skill_version=(
                last.candidate.candidate_skill_version
                if last.candidate is not None
                else f"{self.checkpoint.parent_skill_version.partition('@')[0]}"
                f"@{last.generated_version}"
            ),
            conditions={"static_validation_passed": False},
            reasons=list(last.problems) or ["静态校验未通过"],
        )

    def _evaluation_decision(
        self,
        parent: SkillSpecV11,
        candidate: SkillCandidateV11,
        evaluation: V11PairedEvaluationReceipt,
    ) -> V11CampaignDecision:
        parent_key = f"{parent.skill_id}@{parent.version}"
        candidate_key = candidate.candidate_skill_version
        expected_parent_hash = parent.content_sha256
        expected_candidate_hash = candidate.full_skill_spec.content_sha256
        pa = evaluation.parent
        ca = evaluation.candidate
        finite_means = (
            pa.mean_score is not None
            and ca.mean_score is not None
            and math.isfinite(float(pa.mean_score))
            and math.isfinite(float(ca.mean_score))
        )
        conditions = {
            "campaign_matches": evaluation.campaign_id == self.campaign_id,
            "question_type_matches": (
                evaluation.question_type == self.cfg.question_type
            ),
            "evaluation_completed": evaluation.status == "completed",
            "formal_result_eligible": evaluation.formal_result_eligible,
            "frozen_panel": evaluation.frozen_panel,
            "independent_arms": evaluation.independent_arms,
            "nonempty_pairs": evaluation.n_pairs > 0,
            "complete_denominators": (
                evaluation.n_pairs > 0
                and pa.n_scored == evaluation.n_pairs
                and ca.n_scored == evaluation.n_pairs
            ),
            "parent_identity_matches": (
                pa.skill_version == parent_key
                and pa.content_sha256 == expected_parent_hash
                and pa.delivered_skill_version == parent_key
                and pa.delivered_content_sha256 == expected_parent_hash
                and pa.delivery_observed
            ),
            "candidate_identity_matches": (
                ca.skill_version == candidate_key
                and ca.content_sha256 == expected_candidate_hash
                and ca.delivered_skill_version == candidate_key
                and ca.delivered_content_sha256 == expected_candidate_hash
                and ca.delivery_observed
            ),
            "strict_score_improvement": bool(
                finite_means and float(ca.mean_score) > float(pa.mean_score)
            ),
            "runtime_errors_nonincreasing": (
                ca.runtime_error_count <= pa.runtime_error_count
            ),
            "legal_answer_rate_nondecreasing": (
                pa.legal_answer_rate is not None
                and ca.legal_answer_rate is not None
                and ca.legal_answer_rate >= pa.legal_answer_rate
            ),
        }
        reasons = [name for name, passed in conditions.items() if not passed]
        return V11CampaignDecision(
            campaign_id=self.campaign_id,
            stage="paired_evaluation",
            outcome="promote" if not reasons else "reject",
            parent_snapshot_id=self.checkpoint.parent_snapshot_id,
            parent_skill_version=parent_key,
            candidate_id=candidate.candidate_id,
            candidate_skill_version=candidate_key,
            conditions=conditions,
            reasons=reasons,
        )

    def _decision(
        self,
        *,
        parent: SkillSpecV11,
        candidate: SkillCandidateV11 | None,
        static_receipt: V11StaticValidationReceipt,
        evaluation: V11PairedEvaluationReceipt | None,
    ) -> V11CampaignDecision:
        key = "decision"
        existing = self._load_receipt(key, V11CampaignDecision)
        if existing is not None:
            return existing
        self._set_status("deciding")
        decision = (
            self._static_rejection(static_receipt)
            if candidate is None
            else self._evaluation_decision(parent, candidate, evaluation)
        )
        self._record_receipt(key, decision)
        self.checkpoint.decision = decision.outcome
        self._save_checkpoint()
        return decision

    def _library_promotion_receipt(self, candidate_id: str) -> dict | None:
        path = (
            self.library_root
            / "validation"
            / "promotion"
            / f"{candidate_id}.json"
        )
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise V11CampaignBlocked(f"库内 promotion receipt 损坏: {exc}") from exc
        if not isinstance(value, dict):
            raise V11CampaignBlocked("库内 promotion receipt 不是 mapping")
        return value

    def _finish_interrupted_pointer_switch(
        self,
        candidate: SkillCandidateV11,
        library_receipt: dict,
    ) -> None:
        """Finish only the pointer step of an already materialized publication."""
        before_id = str(library_receipt.get("snapshot_before") or "")
        after_id = str(library_receipt.get("snapshot_after") or "")
        with publish_lock(self.store_dir):
            active = read_active_snapshot(self.store_dir)
            active_id = str(active.get("snapshot_id") or "")
            if active_id == after_id:
                return
            if active_id != before_id:
                raise V11CampaignBlocked(
                    "promotion receipt 存在，但 active 指针已离开父/候选快照"
                )
            snapshot = read_snapshot(self.store_dir, after_id)
            entry = (snapshot.get("entries") or {}).get(
                candidate.candidate_skill_version
            )
            if (
                snapshot.get("parent_snapshot_id") != before_id
                or not isinstance(entry, dict)
                or entry.get("candidate_id") != candidate.candidate_id
                or entry.get("content_sha256")
                != candidate.full_skill_spec.content_sha256
                or (snapshot.get("active_by_question_type") or {}).get(
                    candidate.full_skill_spec.question_type
                )
                != candidate.candidate_skill_version
            ):
                raise V11CampaignBlocked("已物化候选快照与 campaign 候选身份不一致")
            manifest_path = resolve_source_like_ref(
                self.library_root,
                str(snapshot.get("manifest_ref") or ""),
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            write_v11_snapshot(
                self.library_root,
                snapshot,
                manifest,
                activate=True,
                method_context_max_chars=int(self.cfg.method_context_max_chars),
            )

    def _publication_from_library(
        self,
        candidate: SkillCandidateV11,
        library_receipt: dict,
    ) -> V11PublicationReceipt:
        expected = {
            "candidate_id": candidate.candidate_id,
            "skill_version": candidate.candidate_skill_version,
            "parent_skill_version": candidate.parent_skill_version,
            "snapshot_before": candidate.parent_snapshot_id,
        }
        actual = {key: str(library_receipt.get(key) or "") for key in expected}
        if actual != expected:
            raise V11CampaignBlocked(
                f"库内 promotion receipt 与候选不一致: {actual}"
            )
        return V11PublicationReceipt(
            campaign_id=self.campaign_id,
            candidate_id=candidate.candidate_id,
            skill_version=candidate.candidate_skill_version,
            parent_skill_version=candidate.parent_skill_version,
            snapshot_before=candidate.parent_snapshot_id,
            snapshot_after=str(library_receipt.get("snapshot_after") or ""),
            manifest_hash=str(library_receipt.get("manifest_hash") or ""),
            diff_sha256=str(library_receipt.get("diff_sha256") or ""),
            rollback_ref=str(library_receipt.get("rollback_ref") or ""),
        )

    def _verify_publication_active(
        self,
        candidate: SkillCandidateV11,
        receipt: V11PublicationReceipt,
    ) -> None:
        active = read_active_snapshot(self.store_dir)
        if str(active.get("snapshot_id") or "") != receipt.snapshot_after:
            raise V11CampaignBlocked("发布收据存在，但 active 指针不是候选快照")
        skills, warnings, snapshot_id = load_active_skills(self.store_dir)
        matching = [
            skill
            for skill in skills
            if skill.question_type == candidate.full_skill_spec.question_type
        ]
        if (
            warnings
            or snapshot_id != receipt.snapshot_after
            or len(matching) != 1
            or matching[0].version != candidate.full_skill_spec.version
            or matching[0].content_sha256
            != candidate.full_skill_spec.content_sha256
        ):
            raise V11CampaignBlocked("发布后的 active loader 未返回完整候选身份")

    def _publish(
        self,
        candidate: SkillCandidateV11,
    ) -> V11PublicationReceipt:
        key = "publication"
        existing = self._load_receipt(key, V11PublicationReceipt)
        if existing is not None:
            self._verify_publication_active(candidate, existing)
            return existing
        self._set_status("publishing")
        library_receipt = self._library_promotion_receipt(candidate.candidate_id)
        if library_receipt is None:
            _snapshot, library_receipt = publish_v11_candidate(
                self.library_root,
                candidate,
                expected_parent_snapshot_id=self.checkpoint.parent_snapshot_id,
                method_context_max_chars=int(self.cfg.method_context_max_chars),
            )
        else:
            self._finish_interrupted_pointer_switch(candidate, library_receipt)
        receipt = self._publication_from_library(candidate, library_receipt)
        self._verify_publication_active(candidate, receipt)
        self._record_receipt(key, receipt)
        self.checkpoint.published_snapshot_id = receipt.snapshot_after
        self._save_checkpoint()
        return receipt

    def _post_publish_receipt(
        self,
        candidate: SkillCandidateV11,
        publication: V11PublicationReceipt,
        observation: V11PostPublishObservation,
    ) -> V11PostPublishReceipt:
        observed_hash = hashlib.sha256(
            observation.delivered_skill_md.encode("utf-8")
        ).hexdigest()
        expected_version = candidate.candidate_skill_version
        expected_hash = candidate.full_skill_spec.content_sha256
        active = read_active_snapshot(self.store_dir)
        active_ok = str(active.get("snapshot_id") or "") == publication.snapshot_after
        try:
            skills, warnings, snapshot_id = load_active_skills(self.store_dir)
        except Exception:
            skills, warnings, snapshot_id = [], ["load failed"], ""
        loaded = [
            skill
            for skill in skills
            if skill.question_type == candidate.full_skill_spec.question_type
        ]
        loader_ok = (
            not warnings
            and snapshot_id == publication.snapshot_after
            and len(loaded) == 1
            and f"{loaded[0].skill_id}@{loaded[0].version}" == expected_version
            and loaded[0].content_sha256 == expected_hash
        )
        checks = {
            "active_snapshot_matches": active_ok,
            "active_loader_returns_candidate": loader_ok,
            "observation_snapshot_matches": (
                observation.snapshot_id == publication.snapshot_after
            ),
            "question_type_matches": (
                observation.question_type == candidate.full_skill_spec.question_type
            ),
            "candidate_retrieved": (
                observation.retrieved_skill_version == expected_version
            ),
            "candidate_delivered": (
                observation.delivered_skill_version == expected_version
            ),
            "delivered_body_exact": (
                observation.delivered_skill_md == candidate.full_skill_spec.skill_md
            ),
            "delivered_hash_matches": observed_hash == expected_hash,
            "new_learning_observed": bool(observation.learning_event_refs),
            "source_run_recorded": bool(observation.source_run_ref.strip()),
        }
        problems = [name for name, passed in checks.items() if not passed]
        return V11PostPublishReceipt(
            campaign_id=self.campaign_id,
            snapshot_id=publication.snapshot_after,
            expected_skill_version=expected_version,
            expected_content_sha256=expected_hash,
            observation=observation,
            observed_content_sha256=observed_hash,
            checks=checks,
            problems=problems,
            verified=not problems,
        )

    def _verify_post_publish(
        self,
        candidate: SkillCandidateV11,
        publication: V11PublicationReceipt,
    ) -> V11PostPublishReceipt:
        key = "post_publish"
        existing = self._load_receipt(key, V11PostPublishReceipt)
        if existing is not None:
            return existing
        self._set_status("verifying")
        value = self.post_publish_verifier(
            campaign_id=self.campaign_id,
            candidate=candidate.model_copy(deep=True),
            snapshot_id=publication.snapshot_after,
            question_type=self.cfg.question_type,
            idempotency_key=f"{self.campaign_id}:post-publish",
        )
        observation = _coerce(V11PostPublishObservation, value)
        receipt = self._post_publish_receipt(
            candidate,
            publication,
            observation,
        )
        self._record_receipt(key, receipt)
        return receipt

    def run(self) -> V11CampaignCheckpoint:
        """Run or resume one campaign through reject, promote, or blocked."""
        parent, _snapshot = self._initialize()
        assert self.checkpoint is not None
        if self.checkpoint.status in {"promoted", "rejected"}:
            return self.checkpoint
        if self.checkpoint.status == "blocked":
            raise V11CampaignBlocked(self.checkpoint.blocked_reason)

        bundle = self._collect(parent)
        candidate, static_receipt = self._generate_candidate(parent, bundle)
        evaluation = (
            self._evaluate(parent, candidate, bundle)
            if candidate is not None
            else None
        )
        decision = self._decision(
            parent=parent,
            candidate=candidate,
            static_receipt=static_receipt,
            evaluation=evaluation,
        )
        if decision.outcome == "reject":
            self.checkpoint.decision = "reject"
            self.checkpoint.status = "rejected"
            self._save_checkpoint()
            return self.checkpoint

        if candidate is None:
            self._block("promote 决策缺少完整候选")
            raise V11CampaignBlocked(self.checkpoint.blocked_reason)
        try:
            publication = self._publish(candidate)
        except (V11CampaignBlocked, V11LibraryError) as exc:
            reason = f"发布失败: {type(exc).__name__}: {exc}"
            self._block(reason)
            raise V11CampaignBlocked(reason) from exc
        except Exception:
            # The publishing state is already checkpointed. Transient I/O/lock
            # failures can therefore resume without repeating completed stages.
            self._save_checkpoint()
            raise
        post = self._verify_post_publish(candidate, publication)
        if not post.verified:
            reason = f"发布后新 learning 验证失败: {post.problems}"
            self._block(reason)
            raise V11CampaignBlocked(reason)
        self.checkpoint.decision = "promote"
        self.checkpoint.published_snapshot_id = publication.snapshot_after
        self.checkpoint.status = "promoted"
        self._save_checkpoint()
        return self.checkpoint


__all__ = [
    "V11CampaignBlocked",
    "V11CampaignConfig",
    "V11CampaignRunner",
]
