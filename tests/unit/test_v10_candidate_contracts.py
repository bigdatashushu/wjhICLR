"""v10 §5.4/§7.2/§7.4 单元测试：候选必须是**完整合法 SkillSpec**（P0-1/P0-2 的回归锁）。

规范原文（§5.4）：

    「候选修订必须产生完整合法 `SkillSpec`，禁止把自由文本直接拼接到序列化 JSON 后。」

规范原文（§7.2）：候选要声明 `parent_snapshot_id` / `parent_skill_version` /
`candidate_skill_version` / `canonical_question_type` 与完整 `full_skill_spec`。

规范原文（§7.4）："输出不是合法 JSON：最多按同一操作规则重试；Schema 不合法：生成
结构化错误反馈，产生新 revision；超过最大 revision 数：reject；候选与父版本无行为
差异：reject，原因 `no_effective_diff`。"
"""

from __future__ import annotations

import json

import pytest

from skill3d.governance.induce import (
    CandidateStaticCheckError,
    parse_full_spec_response,
    structured_diff,
    validate_candidate_against_parent,
)
from skill3d.governance.revise_patch import (
    PatchStaticCheckError,
    apply_json_merge_patch,
    new_revision_from_patch,
    parse_patch,
)
from skill3d.schemas import CandidateRevision, GPT6Patch, SkillCandidate, SkillSpec
from skill3d.skills.delivery import skill_content_sha256

PARENT_PAYLOAD = {
    "skill_id": "S01", "version": "1.0.0",
    "applicable_question_types": ["object_counting"],
    "required_evidence_signature": {"object_detection": "available"},
    "skill_family": "counting", "source": "real",
    "description": "数出目标对象数量",
    "call_graph_template": "n = detect_objects(img, 'chair')\nReturnAnswer(str(len(n)))",
    "validation_assertions": ["n >= 0"],
}
TOOLS = ["detect_objects", "inspect_frames"]


def _parent() -> SkillSpec:
    return SkillSpec.model_validate(PARENT_PAYLOAD)


def _child(**updates) -> SkillSpec:
    payload = {**PARENT_PAYLOAD, "version": "1.1.0", **updates}
    return SkillSpec.model_validate(payload)


def _check(spec: SkillSpec, *, parent: SkillSpec | None = None, raw: str = "",
           tool_names=TOOLS, parent_hash_override: str | None = None):
    parent = parent or _parent()
    parent_hash = skill_content_sha256(parent)
    return validate_candidate_against_parent(
        spec, parent,
        parent_content_sha256=parent_hash_override if parent_hash_override is not None
        else parent_hash,
        claimed_parent_content_sha256=parent_hash,
        canonical_question_type="object_counting",
        tool_names=tool_names, raw_text=raw)


# ---------------------------------------------------------------- §5.4 完整候选 ----

def test_full_spec_round_trip_is_accepted():
    spec = _child(description="数出目标对象数量：先定类别，再跨帧覆盖")
    receipt = _check(spec, raw=json.dumps({"spec": spec.model_dump(mode="json")}))
    assert receipt.passed, receipt.problems
    assert all(receipt.checks.values())


def test_free_text_patch_shape_is_rejected():
    """§5.4：`# PATCH` 文本追加形态必须被拒（v9 的 EV-04 阻断点）。"""
    text = json.dumps({"spec": PARENT_PAYLOAD}) + "\n\n# PATCH\nassert n >= 0"
    with pytest.raises(ValueError):
        parse_full_spec_response(text)
    # 直接给 v9 形状的 patch 对象（没有 spec 字段）同样不是候选
    patch_like = json.dumps({"patch_type": "add_assertion", "patch_content": "assert n>=0",
                             "affected_task_types": ["object_counting"]})
    with pytest.raises(Exception):
        parse_full_spec_response(patch_like)


def test_parent_hash_mismatch_fails_static_check():
    """§5.4：父版本 hash 与候选记录不一致 → 静态检查失败。"""
    spec = _child(description="改了描述")
    receipt = _check(spec, raw=json.dumps({"spec": spec.model_dump(mode="json")}),
                     parent_hash_override="deadbeef")
    assert receipt.passed is False
    assert receipt.checks["parent_hash_matches"] is False


def test_precondition_change_requires_major_and_is_out_of_scope():
    """§5.2/§7.3：改检索前提（证据签名）必须 MAJOR → 首期超出范围，静态检查拒绝。"""
    spec = _child(required_evidence_signature={"object_detection": "degraded"})
    receipt = _check(spec, raw=json.dumps({"spec": spec.model_dump(mode="json")}))
    assert receipt.passed is False
    assert receipt.checks["version_bump_legal"] is False
    assert any("MAJOR" in p for p in receipt.problems)


def test_identical_content_with_bumped_version_is_no_effective_diff():
    """§7.4：候选与父版本**完全无差异**（只 bump 版本）→ reject，原因 no_effective_diff。

    注意与 §7.3 的区分：只改**描述**属于"修改描述 → MINOR"（合法候选）；
    "只修错字、不改变行为"这一类机制上无法与"改进描述"区分，因此由内容 diff 为空
    与下游配对效果评测把关，而不是靠文本差异猜测。
    """
    spec = _child()                                       # 除版本号外与父完全一致
    receipt = _check(spec, raw=json.dumps({"spec": spec.model_dump(mode="json")}))
    assert receipt.passed is False
    assert receipt.checks["diff_nonempty_and_consistent"] is False
    assert any("no_effective_diff" in p for p in receipt.problems)


def test_description_only_change_is_a_legal_minor_candidate():
    """§7.3："修改描述、步骤、检查、示例或失败教训：MINOR bump"。"""
    spec = _child(description="数出目标对象数量：先锁定目标类别，再跨帧去重计数")
    receipt = _check(spec, raw=json.dumps({"spec": spec.model_dump(mode="json")}))
    assert receipt.passed is True, receipt.problems


def test_unknown_tool_in_template_is_rejected():
    """§5.4：工具名必须来自当前 ToolSpec。"""
    spec = _child(call_graph_template="n = magic_tool(img)\nReturnAnswer(str(n))")
    receipt = _check(spec, raw=json.dumps({"spec": spec.model_dump(mode="json")}))
    assert receipt.checks["tools_known"] is False
    assert any("未知工具" in p for p in receipt.problems)


def test_forbidden_knob_and_leakage_are_rejected():
    """§12.3：检测阈值 / 检索 top-k 等首期禁止修改项不得出现在候选正文里。"""
    spec = _child(call_graph_template="if debug: box_threshold = 0.25\nReturnAnswer('1')")
    receipt = _check(spec, raw=json.dumps({"spec": spec.model_dump(mode="json")}))
    assert receipt.checks["no_leakage"] is False
    leaked = _child(description="本题标准答案是 3（ground truth）")
    receipt2 = _check(leaked, raw=json.dumps({"spec": leaked.model_dump(mode="json")}))
    assert receipt2.checks["no_leakage"] is False


def test_diff_must_be_nonempty_and_consistent():
    """§5.4：diff 非空且与候选完整内容一致（用**再应用**验证，不靠人读）。"""
    parent, child = _parent(), _child(description="新描述")
    diff = structured_diff(parent, child)
    assert diff
    from skill3d.governance.induce import apply_structured_diff

    rebuilt = apply_structured_diff(parent, diff)
    assert rebuilt.model_dump(mode="json") == child.model_dump(mode="json")


def test_candidate_schema_rejects_cross_lineage_and_empty_diff():
    spec = _child()
    base = dict(candidate_id="c1", campaign_id="C", generation=1, operation="revise",
                parent_snapshot_id="S0", parent_skill_version="S01@1.0.0",
                candidate_skill_version="S01@1.1.0",
                canonical_question_type="object_counting", full_skill_spec=spec,
                structured_diff=[{"field": "description"}])
    SkillCandidate(**base)                                     # 基线合法
    with pytest.raises(Exception):
        SkillCandidate(**{**base, "parent_skill_version": "S02@1.0.0"})   # 跨谱系
    with pytest.raises(Exception):
        SkillCandidate(**{**base, "structured_diff": []})                 # 空 diff
    with pytest.raises(Exception):
        SkillCandidate(**{**base, "canonical_question_type": "room_size_estimation"})


# ---------------------------------------------------------------- 结构化修订 ----

def test_json_merge_patch_semantics_and_full_spec_output():
    """§5.4 方式一：Merge Patch 作用在**解析后的对象**上，再完整序列化。"""
    merged = apply_json_merge_patch(
        {"a": 1, "b": {"c": 2, "d": 3}, "e": "x"}, {"b": {"c": 9}, "e": None, "f": True})
    assert merged == {"a": 1, "b": {"c": 9, "d": 3}, "f": True}

    current = CandidateRevision(
        revision_id="rev-1", root_candidate_id="cand-1", parent_version=None,
        candidate_type="skill", spec_content=json.dumps(PARENT_PAYLOAD), status="draft",
        induction_trace_refs=[], evidence_lineage_ref="", created_by="human",
        created_at="2026-01-01T00:00:00+00:00")
    patch = parse_patch(json.dumps({
        "patch_type": "add_assertion",
        "affected_task_types": ["object_counting"],
        "patch_content": json.dumps({"version": "1.1.0",
                                     "validation_assertions": ["n >= 0", "跨帧去重"]}),
    }), target_revision_id="rev-1")
    child = new_revision_from_patch(current, patch)
    parsed = json.loads(child.spec_content)              # 必须是**完整合法 JSON**
    assert parsed["version"] == "1.1.0"
    assert parsed["validation_assertions"] == ["n >= 0", "跨帧去重"]
    assert SkillSpec.model_validate(parsed).skill_id == "S01"
    assert child.parent_version == "rev-1"               # 父链保留（硬约束 11）
    assert "# PATCH" not in child.spec_content           # 不再有文本追加


def test_free_text_patch_content_is_rejected():
    """§5.4：`patch_content` 不是结构化 JSON → 静态检查失败（禁止文本路径）。"""
    with pytest.raises(PatchStaticCheckError):
        parse_patch(json.dumps({
            "patch_type": "add_assertion",
            "affected_task_types": ["object_counting"],
            "patch_content": "assert n >= 0  # 自由文本",
        }), target_revision_id="rev-1")


def test_invalid_merged_spec_is_rejected_not_silently_serialized():
    """结构化修订的结果必须能通过 SkillSpec 校验，否则拒绝（不产出坏候选）。"""
    current = CandidateRevision(
        revision_id="rev-1", root_candidate_id="cand-1", parent_version=None,
        candidate_type="skill", spec_content=json.dumps(PARENT_PAYLOAD), status="draft",
        induction_trace_refs=[], evidence_lineage_ref="", created_by="human",
        created_at="2026-01-01T00:00:00+00:00")
    patch = GPT6Patch(
        target_revision_id="rev-1", patch_type="modify_skill",
        affected_task_types=["object_counting"], affected_skills=["S01"],
        patch_content=json.dumps({"version": "1.1.0", "skill_family": "不存在的族"}),
        rationale="x", expected_improvement="y", risk_notes="z",
        prompt_version="test")
    with pytest.raises(PatchStaticCheckError):
        new_revision_from_patch(current, patch)


def test_inducer_static_error_carries_receipt():
    """§7.4：Schema 不合法要生成**结构化错误反馈**（收据里逐项检查结果）。"""
    spec = _child(call_graph_template="n = magic_tool(img)")
    receipt = _check(spec, raw=json.dumps({"spec": spec.model_dump(mode="json")}))
    assert receipt.passed is False and receipt.problems
    assert set(receipt.checks) == {
        "json_parseable", "schema_extra_forbid", "skill_id_matches_parent",
        "version_bump_legal", "single_canonical_question_type",
        "method_body_deliverable", "tools_known", "no_leakage",
        "parent_hash_matches", "diff_nonempty_and_consistent"}
    assert CandidateStaticCheckError("x", receipt).receipt is receipt
