"""File-backed Skill library helpers for v9 candidate records.

Source compilation lives in :mod:`source_compiler`; the online process does not
import this module. Candidate records are immutable audit documents staged
outside the active snapshot until deterministic validation approves promotion.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from skill3d.schemas import CandidateRevision, SkillLibraryCandidate, SkillSpec


_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class SkillLibraryError(ValueError):
    """Raised for unsafe or conflicting Skill library writes."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_candidate_record(
    library_root: str | Path,
    candidate: SkillLibraryCandidate,
    *,
    future: bool = True,
) -> tuple[Path, str]:
    """Atomically write a create-once v9 candidate record.

    Future candidates are staged under ``candidates/future`` and cannot become
    online-visible merely by being present on disk. Rewriting an existing ID
    is rejected to preserve audit lineage.
    """
    if not _SAFE_ID.fullmatch(candidate.candidate_id):
        raise SkillLibraryError(f"不安全的 candidate_id: {candidate.candidate_id!r}")
    root = Path(library_root).resolve()
    subdir = "future" if future else "validated"
    target_dir = root / "candidates" / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{candidate.candidate_id}.json"
    payload = candidate.model_dump(mode="json")
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if target.exists():
        existing = target.read_bytes()
        if existing == data:
            return target, _digest(data)
        raise SkillLibraryError(f"candidate record 已存在且内容不同，禁止覆盖: {target}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{candidate.candidate_id}.", suffix=".tmp",
                                    dir=target_dir)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return target, _digest(data)


def load_candidate_record(path: str | Path) -> SkillLibraryCandidate:
    """Load and strictly validate a v9 candidate record."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return SkillLibraryCandidate.model_validate(raw)


def static_check_skill_spec(spec: SkillSpec, *, method_context_max_chars: int) -> list[str]:
    """§14.4 静态检查里与**服务限制**有关的那一项（§13.5 末句）。

    规范原文（§13.5）："超过服务限制的候选在静态检查中拒绝或修订。"

    这里只判"单条候选的完整方法正文是否超过方法上下文上限" —— 超过上限的候选
    **永远无法完整交付**（§13.5 禁止截断正文后仍称"完整 Skill 已交付"），因此不能
    让它进入 active。返回违规清单（空 = 通过）；调用方决定是 quarantine 还是拒绝。

    其余静态检查项（格式、来源权限、无样例泄漏、单题型、工具合同）已由
    `SkillSpec` 构造期校验与 `candidate_revision_from_record` 的题型一致性检查承担；
    方法正文长度此前没有任何检查点，是这里的补充。
    """
    from skill3d.skills.delivery import check_service_limit

    problems = list(check_service_limit(spec, max_chars=int(method_context_max_chars)))
    if not spec.applicable_question_types:
        problems.append("候选未声明任何适用题型（§13.5 硬隔离检索要求单题型分区）")
    if len(set(spec.applicable_question_types)) > 1:
        problems.append(
            "候选声明了多个题型（§14.2/§13.5：候选按规范题型单题型准入，"
            f"收到 {sorted(set(spec.applicable_question_types))}）")
    return problems


def candidate_revision_from_record(
    record: SkillLibraryCandidate,
    *,
    record_path: str | Path,
    spec_content: str,
    revision_id: str | None = None,
    status: str = "draft",
    created_by: str = "human",
) -> CandidateRevision:
    """Adapt an explanatory file record to the runtime revision contract.

    The record is governance metadata, not executable SkillSpec content. For a
    skill candidate the payload is parsed strictly here, while the returned
    revision keeps the record path and all content digests for lineage.
    """
    if record.operation.strip() == "" or record.canonical_question_type.strip() == "":
        raise SkillLibraryError("candidate record 缺少 operation 或 canonical_question_type")
    try:
        parsed = json.loads(spec_content)
    except json.JSONDecodeError as exc:
        raise SkillLibraryError(f"spec_content 不是 JSON SkillSpec: {exc}") from exc
    try:
        spec = SkillSpec.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001 - promotion must fail closed
        raise SkillLibraryError(f"spec_content 不是合法 SkillSpec: {exc}") from exc
    if record.canonical_question_type not in spec.applicable_question_types:
        raise SkillLibraryError(
            "candidate record 题型与 SkillSpec 不一致: "
            f"record={record.canonical_question_type} spec={spec.applicable_question_types}"
        )
    path = Path(record_path)
    return CandidateRevision(
        revision_id=revision_id or record.candidate_id,
        root_candidate_id=record.candidate_id,
        parent_version=(record.parent_skill_versions[-1]
                        if record.parent_skill_versions else None),
        candidate_type="skill",
        spec_content=spec.model_dump_json(),
        status=status,
        induction_trace_refs=list(record.source_trace_refs),
        evidence_lineage_ref=(record.inducer_receipt_ref or ""),
        created_by=created_by,  # type: ignore[arg-type]
        created_at=record.created_at,
        source_split=record.source_split,
        experience_relation=record.experience_relation,
        source_path=record.source_path,
        source_sha256=record.source_sha256,
        generated_spec_path=record.generated_spec_path,
        generated_sha256=record.generated_sha256,
        manifest_ref=record.manifest_ref,
        candidate_record_ref=str(path),
    )


def candidate_record_from_revision(
    revision: CandidateRevision,
    *,
    operation: str = "revise",
    canonical_question_type: str | None = None,
    hypothesis: str = "",
    patch: str = "",
    expected_scope: str = "",
    expected_effect: str = "",
    known_risks: str = "",
) -> SkillLibraryCandidate:
    """Project runtime provenance back into an immutable library record."""
    try:
        spec = SkillSpec.model_validate(json.loads(revision.spec_content))
    except Exception as exc:  # noqa: BLE001
        raise SkillLibraryError(f"revision.spec_content 不是合法 SkillSpec: {exc}") from exc
    qtype = canonical_question_type or (spec.applicable_question_types[0]
                                        if spec.applicable_question_types else "")
    return SkillLibraryCandidate(
        candidate_id=revision.root_candidate_id,
        operation=operation,
        canonical_question_type=qtype,
        parent_skill_versions=([revision.parent_version]
                               if revision.parent_version else []),
        source_trace_refs=list(revision.induction_trace_refs),
        source_split=revision.source_split or "unknown",
        experience_relation=revision.experience_relation or "unknown",
        hypothesis=hypothesis,
        patch=patch,
        expected_scope=expected_scope or qtype,
        expected_effect=expected_effect,
        known_risks=known_risks,
        source_path=revision.source_path,
        source_sha256=revision.source_sha256,
        generated_spec_path=revision.generated_spec_path,
        generated_sha256=revision.generated_sha256,
        manifest_ref=revision.manifest_ref,
        created_at=revision.created_at,
    )


__all__ = [
    "SkillLibraryError",
    "candidate_record_from_revision",
    "candidate_revision_from_record",
    "load_candidate_record",
    "write_candidate_record",
]
