"""P6 单测：§13.5"超过服务限制的候选在静态检查中拒绝"与 §14.4 quarantine。

规范原文（§13.5）：

    「超过服务限制的候选在静态检查中拒绝或修订。」

规范原文（§14.4）：

    「静态检查先验证格式、来源权限、无样例泄漏、单题型、工具合同、正文长度和
    可执行示例（若有）。不合格记 quarantine；结果不完整或不能配对时记
    validation_incomplete，不得晋升。」

覆盖的判定链：`manage_skill_library.py validate-candidate` 与
`promote_atomic.promote(strict_skill_specs=True)` —— 两条发布路径都必须在**改变
active 指针之前**拒绝超限候选。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill3d.schemas import SkillSpec
from skill3d.skills.delivery import (
    DEFAULT_METHOD_CONTEXT_MAX_CHARS,
    render_skill_entry,
)
from skill3d.skills.library import static_check_skill_spec
from skill3d.skills.promote_atomic import SnapshotValidationError, promote


def _spec(desc: str = "计数方法", template: str = "ReturnAnswer(str(count_objects('x')))",
          tasks=("object_counting",), family="counting") -> SkillSpec:
    return SkillSpec(skill_id="sk-1", version="1.0.0",
                     applicable_question_types=list(tasks),
                     skill_family=family, description=desc,
                     call_graph_template=template)


def test_static_check_accepts_a_normal_candidate():
    assert static_check_skill_spec(_spec(), method_context_max_chars=8000) == []


def test_static_check_rejects_body_over_the_service_limit():
    """单条正文超过上限 → 永远无法完整交付 → 静态检查必须拒绝。"""
    spec = _spec(desc="X" * 8100)
    assert len(render_skill_entry(spec)) > 8000
    problems = static_check_skill_spec(spec, method_context_max_chars=8000)
    assert problems and "服务限制" in problems[0]
    # 判据是**完整正文长度**：刚好放得下就不算超限
    assert static_check_skill_spec(
        spec, method_context_max_chars=len(render_skill_entry(spec))) == []


def test_static_check_rejects_multi_question_type_candidates():
    """§14.4"单题型"：候选只能属于一个规范题型分区（§13.5 硬隔离检索的前提）。"""
    # metric 族本身覆盖三个题型：声明两个同族题型能过 SkillSpec 构造期校验，
    # 但必须被 §14.4 的"单题型"静态检查拦下（每个候选只属于一个题型分区）。
    spec = _spec(tasks=("object_abs_distance", "room_size_estimation"), family="metric")
    problems = static_check_skill_spec(spec, method_context_max_chars=8000)
    assert any("单题型" in p or "多个题型" in p for p in problems)


def test_promote_fails_closed_on_oversized_candidate(tmp_path):
    """§13.5：发布路径必须在**切换 active 指针之前**拒绝超限候选。"""
    from skill3d.schemas import CandidateRevision

    store = tmp_path / "snapshots"
    store.mkdir(parents=True)
    big = _spec(desc="X" * 8100)
    rev = CandidateRevision(
        revision_id="rev-big", root_candidate_id="sk-1", candidate_type="skill",
        spec_content=big.model_dump_json(), status="promoted",
        created_by="human", created_at="2026-09-27T00:00:00Z",
        source_split="learning", experience_relation="revise",
        parent_version=None, induction_trace_refs=[], evidence_lineage_ref="")
    with pytest.raises(SnapshotValidationError) as exc:
        promote(store, rev, strict_skill_specs=True, method_context_max_chars=8000)
    assert "服务限制" in str(exc.value) or "静态检查" in str(exc.value)
    # 未切换 active：指针文件根本没被创建
    assert not (store / "active_snapshot.json").exists()


def test_promote_accepts_a_candidate_within_the_limit(tmp_path):
    from skill3d.schemas import CandidateRevision

    store = tmp_path / "snapshots"
    store.mkdir(parents=True)
    rev = CandidateRevision(
        revision_id="rev-ok", root_candidate_id="sk-1", candidate_type="skill",
        spec_content=_spec().model_dump_json(), status="promoted",
        created_by="human", created_at="2026-09-27T00:00:00Z",
        source_split="learning", experience_relation="revise",
        parent_version=None, induction_trace_refs=[], evidence_lineage_ref="")
    snap = promote(store, rev, strict_skill_specs=True,
                   method_context_max_chars=DEFAULT_METHOD_CONTEXT_MAX_CHARS)
    assert snap["snapshot_id"]
    assert (store / "active_snapshot.json").exists()


def test_validate_candidate_cli_reports_quarantine(tmp_path):
    """CLI 的 `validate-candidate` 不再只回显"validated"：不合格记 quarantine 并非零退出。"""
    import importlib.util

    spec_path = Path(__file__).resolve().parents[2] / "scripts" / "manage_skill_library.py"
    mod_spec = importlib.util.spec_from_file_location("manage_skill_library_cli", spec_path)
    cli = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(cli)

    big = _spec(desc="X" * 8100)
    record = tmp_path / "cand.json"
    record.write_text(json.dumps({
        "candidate_id": "cand-big", "operation": "revise",
        "canonical_question_type": "object_counting",
        "source_split": "learning", "experience_relation": "revise",
        "hypothesis": "h", "patch": "p", "expected_scope": "s",
        "expected_effect": "e", "known_risks": "r",
        "created_at": "2026-09-27T00:00:00Z"}), encoding="utf-8")
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(big.model_dump_json(), encoding="utf-8")

    with pytest.raises(SystemExit):
        cli.main(["validate-candidate", "--library-root", str(tmp_path / "lib"),
                  "--record", str(record), "--spec-file", str(spec_file)])

    ok_spec = tmp_path / "ok.json"
    ok_spec.write_text(_spec().model_dump_json(), encoding="utf-8")
    assert cli.main(["validate-candidate", "--library-root", str(tmp_path / "lib"),
                     "--record", str(record), "--spec-file", str(ok_spec)]) == 0
