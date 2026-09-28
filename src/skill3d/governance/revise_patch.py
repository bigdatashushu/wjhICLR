"""M16 基于反例的离线修订（v10 §5.4/§7.4）：**结构化变更 → 完整合法 SkillSpec**。

规范原文（§5.4）：

    「候选修订必须产生完整合法 `SkillSpec`，禁止把自由文本直接拼接到序列化 JSON 后。

    允许两种实现：

    * JSON Merge Patch / JSON Patch 作用于解析后的对象，再完整序列化；
    * 离线模型直接输出完整候选 `SkillSpec`，框架做父 / 子字段差分。

    首期推荐第二种：离线模型输出完整候选，框架负责验证和生成 diff。」

v9 的实现（`current.spec_content + "\\n\\n# PATCH\\n" + patch.patch_content`）正是 §5.4
禁止的那种：拼出来的东西不是合法 JSON，无法成为下一轮 `SkillSpec`（v10 §1.2 EV-04）。
本模块现在只提供**两条合法路径**：

1. `revise_with_full_spec()`：离线模型输出完整候选（首期推荐路径，与
   `governance/induce.py` 共用同一套校验）；
2. `apply_json_merge_patch()`：模型给结构化 patch（RFC 7386 语义的子集：对象成员
   递归合并、`null` 删除成员），作用在**解析后的对象**上，然后完整重新序列化。

两条路径的出口都是"通过 §5.4 静态检查的完整 SkillSpec"，不存在把自由文本追加到
JSON 之后的分支。

红线（保留）：

- **候选不可变（硬约束 11）**：绝不原地修改候选，修订必产新版本（父链）；
- **只见可见摘要（硬约束 19）**：离线模型只拿经验包 / 反例摘要（不含答案、不含 id）；
- **advisory（§3.3）**：修订产物是 draft 候选，**不**构成准入决定；
- patch 必须声明 `affected_task_types`（§8），否则静态检查失败。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from skill3d.evolution.firewall import scan_prompt_for_leakage
from skill3d.schemas import (
    CandidateRevision,
    CounterexampleBundle,
    GPT6Patch,
    SkillSpec,
)
from skill3d.schemas.evolution import StaticValidationReceipt
from skill3d.governance.induce import (
    CandidateStaticCheckError,
    parse_candidate_metadata,
    parse_full_spec_response,
    sha256_of_spec,
    structured_diff,
    validate_candidate_against_parent,
)

# prompt 模板版本（冻结配置项，进 RunManifest 的 prompt_version；§3.4）
REVISE_PROMPT_VERSION = "revise-v10.0"
# v5 名称别名（过渡期只读）
GPT6_REVISE_PROMPT_VERSION = REVISE_PROMPT_VERSION

# `CandidateRevision.created_by` 受控枚举定义在 `schemas/evolution.py`（本模块不拥有；
# 其字面量仍是 v5 取值）。集中在此，schema 改名后只需改这一行。
CREATED_BY_OFFLINE_REVISION = "gpt6_revision"


class PatchStaticCheckError(ValueError):
    """patch 静态检查失败（如未声明 affected_task_types / 目标版本不一致）。"""


def build_revision_prompt(current_spec_content: str,
                          gpt6_visible_summary: str,
                          rejected_similar: list[str] | None = None) -> str:
    """构造修订 prompt：只含当前 spec + 反例摘要（不含答案）+ 已拒相似候选。

    §5.4 第二条路径：要求模型输出**完整候选 SkillSpec**（不是文本补丁）。
    `gpt6_visible_summary` 形参名沿用 `CounterexampleBundle` 的字段名（该字段名定义在
    `schemas/evolution.py`，本模块不拥有）；内容是**可见摘要**，绝不含答案。
    """
    lines = [
        "你是 harness3D 的离线 Skill 修订器。",
        "基于下面的反例摘要，输出**完整的候选 SkillSpec JSON**（不是补丁文本），",
        "字段必须齐全；禁止出现任何具体题目答案、选项字母或 sample ID。",
        "你**不**执行实验、不评分、不决定 promote/reject —— 准入由确定性门决定。",
        "## 当前 SkillSpec",
        current_spec_content,
        "## 反例摘要（不含答案）",
        gpt6_visible_summary,
        "## 已被拒绝的相似候选（避免重复）",
        *[f"- {s}" for s in (rejected_similar or [])],
        "## 输出 JSON",
        '{"spec": {……完整 SkillSpec……}, "hypothesis": "…", "expected_effect": "…",'
        ' "known_risks": ["…"], "diff_summary": "…"}',
    ]
    prompt = "\n".join(lines)
    scan_prompt_for_leakage(prompt)  # 硬门：prompt 不得含答案/ID（硬约束 19）
    return prompt


def parse_patch(response_text: str, target_revision_id: str,
                prompt_version: str = REVISE_PROMPT_VERSION,
                model_id: str = "") -> GPT6Patch:
    """解析**结构化** patch，并做静态检查：`affected_task_types` 必填。

    v10：`patch_content` 必须是 JSON 文本（Merge Patch），不得是自由散文 —— 散文无法
    结构化作用到对象上，也无法复核 diff 一致性（§5.4）。

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
    try:
        parsed = json.loads(patch.patch_content)
    except json.JSONDecodeError as exc:
        raise PatchStaticCheckError(
            "patch_content 不是结构化 JSON（§5.4：禁止把自由文本拼接到 JSON 之后）："
            f"{exc}") from exc
    if not isinstance(parsed, dict):
        raise PatchStaticCheckError(
            f"patch_content 必须是 JSON 对象（Merge Patch），收到 {type(parsed).__name__}")
    return patch


def apply_json_merge_patch(spec_payload: dict, patch: dict) -> dict:
    """RFC 7386 语义的 JSON Merge Patch（作用在**解析后的对象**上）。

    - 对象成员递归合并；
    - `null` 值删除该成员（与 RFC 7386 一致）；
    - 其余类型整体替换。

    只处理这些语义，不做任何文本拼接 —— 输出的 dict 会重新完整序列化为 SkillSpec。
    """
    if not isinstance(patch, dict):
        raise PatchStaticCheckError("merge patch 必须是 JSON 对象")
    out = dict(spec_payload)
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = apply_json_merge_patch(out[key], value)
        else:
            out[key] = value
    return out


def new_revision_from_patch(current: CandidateRevision, patch: GPT6Patch) -> CandidateRevision:
    """由结构化 patch 产出新版本候选（绝不原地修改，硬约束 11；parent_version 链）。

    v10（§5.4）：对**解析后的对象**应用 Merge Patch，然后完整序列化 —— 新的
    `spec_content` 一定是可以再次 `model_validate` 的完整 SkillSpec。
    """
    if patch.target_revision_id != current.revision_id:
        raise PatchStaticCheckError(
            f"patch 目标 {patch.target_revision_id} 与当前版本 {current.revision_id} 不一致"
        )
    try:
        base_payload = json.loads(current.spec_content)
    except json.JSONDecodeError as exc:
        raise PatchStaticCheckError(
            f"当前候选 spec_content 不是 JSON，无法结构化修订: {exc}") from exc
    try:
        patch_payload = json.loads(patch.patch_content)
    except json.JSONDecodeError as exc:
        raise PatchStaticCheckError(
            f"patch_content 不是 JSON，禁止文本追加（§5.4）: {exc}") from exc
    merged = apply_json_merge_patch(base_payload, patch_payload)
    try:
        spec = SkillSpec.model_validate(merged)
    except Exception as exc:  # noqa: BLE001 - 修订产物必须是合法 SkillSpec
        raise PatchStaticCheckError(
            f"结构化修订结果不是合法 SkillSpec（§5.4）: {type(exc).__name__}: {exc}"
        ) from exc
    return _child_revision(current, spec.model_dump(mode="json"))


def _child_revision(current: CandidateRevision, spec_payload: dict) -> CandidateRevision:
    """把完整 SkillSpec 载荷装进子版本 `CandidateRevision`（父链 + 全部 provenance）。"""
    return CandidateRevision(
        revision_id=f"rev-{uuid.uuid4().hex[:12]}",
        root_candidate_id=current.root_candidate_id,
        parent_version=current.revision_id,  # 版本 DAG：父链
        candidate_type=current.candidate_type,
        spec_content=json.dumps(spec_payload, ensure_ascii=False, indent=2, sort_keys=True),
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
    """完整修订链（v9 形状的出口）：可见摘要 → prompt → 离线模型 → **完整 SkillSpec**。

    与 v9 的差别只有一处但很关键：模型输出被解析为**完整候选**并做 §5.4 静态检查，
    不再把文本追加到 `spec_content` 之后（EV-04）。异常按 §3.4 归族
    （`OfflineAuthError` / `OfflineServiceUnavailable` / ...）由调用方映射为
    "暂停/quarantine"，本函数不吞异常、不返回空串。
    """
    prompt = build_revision_prompt(
        current.spec_content, bundle.gpt6_visible_summary, rejected_similar)
    resp = offline_client.chat(prompt)
    spec, raw_text = parse_full_spec_response(resp)
    _static_gate(current, spec, raw_text)
    return _child_revision(current, spec.model_dump(mode="json"))


def revise_with_full_spec(parent_spec: SkillSpec,
                          visible_summary: str,
                          offline_client,
                          *,
                          canonical_question_type: str,
                          tool_names: Sequence[str],
                          forbidden_sample_ids: Optional[set[str]] = None,
                          method_context_max_chars: int = 8000,
                          rejected_similar: list[str] | None = None,
                          parent_content_sha256: str = "",
                          ) -> tuple[SkillSpec, StaticValidationReceipt, dict]:
    """§5.4 推荐路径：离线模型输出**完整候选 SkillSpec** → 框架校验 → `(spec, 收据, 元数据)`。

    `parent_spec` 是父版本完整 `SkillSpec`；`visible_summary` 是经验 / 反例摘要
    （不含答案）。校验与 `governance/induce.py` 共用同一个实现，因此"归纳"与"修订"
    两条路径不存在两套标准。
    """
    prompt = build_revision_prompt(
        json.dumps(parent_spec.model_dump(mode="json"), ensure_ascii=False,
                   indent=2, sort_keys=True),
        visible_summary, rejected_similar)
    response = offline_client.chat(prompt)
    spec, raw_text = parse_full_spec_response(response)
    parent_hash = parent_content_sha256 or sha256_of_spec(parent_spec)
    receipt = validate_candidate_against_parent(
        spec, parent_spec,
        parent_content_sha256=parent_hash,
        claimed_parent_content_sha256=parent_hash,
        canonical_question_type=canonical_question_type,
        tool_names=tool_names,
        forbidden_sample_ids=forbidden_sample_ids,
        method_context_max_chars=method_context_max_chars,
        raw_text=raw_text)
    return spec, receipt, parse_candidate_metadata(response)


def _static_gate(current: CandidateRevision, spec: SkillSpec, raw_text: str) -> None:
    """v9 形状入口的静态门：父/子必须同谱系、版本合法递增、完整可交付（§5.4）。"""
    try:
        parent_spec = SkillSpec.model_validate(json.loads(current.spec_content))
    except Exception as exc:  # noqa: BLE001
        raise PatchStaticCheckError(
            f"当前候选 spec_content 不是合法 SkillSpec，无法结构化修订: {exc}") from exc
    receipt = validate_candidate_against_parent(
        spec, parent_spec,
        parent_content_sha256=sha256_of_spec(parent_spec),
        claimed_parent_content_sha256=sha256_of_spec(parent_spec),
        canonical_question_type=str(spec.applicable_question_types[0]),
        tool_names=[],            # 未知工具清单时不做工具名校验（不虚构通过）
        raw_text=raw_text)
    # 工具清单未知 → `tools_known` 由调用方在 v10 路径复核；这里只拦硬错误。
    hard = [k for k, v in receipt.checks.items() if not v and k != "tools_known"]
    if hard:
        raise CandidateStaticCheckError(
            f"修订产物未通过静态检查（§5.4）: {sorted(hard)} {receipt.problems}",
            receipt)


def candidate_diff(parent_spec: SkillSpec, child_spec: SkillSpec) -> list[dict]:
    """父 → 子结构化 diff（供收据落盘；§14.1 candidate.json 的 `structured diff`）。"""
    return structured_diff(parent_spec, child_spec)


__all__ = [
    "CREATED_BY_OFFLINE_REVISION",
    "GPT6_REVISE_PROMPT_VERSION",
    "REVISE_PROMPT_VERSION",
    "PatchStaticCheckError",
    "apply_json_merge_patch",
    "build_revision_prompt",
    "candidate_diff",
    "new_revision_from_patch",
    "parse_patch",
    "revise_from_bundle",
    "revise_with_full_spec",
]
