"""M17 防火墙测试：split 物理隔离 + GPT-6 prompt 泄漏扫描（硬约束 9/19）。"""

import pytest

from skill3d.evolution.firewall import (
    PromptLeakageError,
    SplitContaminationError,
    assert_final_test_not_mounted,
    assert_split_isolation,
    scan_prompt_for_leakage,
)
from skill3d.governance.induce import build_induction_prompt
from skill3d.schemas import DataSplitConfig


def _split(induction, inner, outer, final) -> DataSplitConfig:
    return DataSplitConfig(
        induction_scene_ids=induction, inner_validation_scene_ids=inner,
        outer_holdout_scene_ids=outer, final_test_scene_ids=final,
        split_version="v1", contamination_check_log_ref="")


def test_disjoint_splits_pass():
    assert_split_isolation(_split(["s1", "s2"], ["s3"], ["s4"], ["s5"]))


def test_final_test_overlap_rejected():
    with pytest.raises(SplitContaminationError):
        assert_split_isolation(_split(["s1"], ["s2"], ["s3"], ["s1"]))
    with pytest.raises(SplitContaminationError):
        assert_split_isolation(_split(["s1"], ["s1"], ["s3"], ["s4"]))


def test_prompt_scan_blocks_ground_truth():
    with pytest.raises(PromptLeakageError):
        scan_prompt_for_leakage("该题 answer: B，请归纳模板")
    with pytest.raises(PromptLeakageError):
        scan_prompt_for_leakage("参考 ground_truth=3.5m")
    with pytest.raises(PromptLeakageError):
        scan_prompt_for_leakage("涉及 final_test 数据")
    with pytest.raises(PromptLeakageError):
        scan_prompt_for_leakage("样本 qa_ab12cd34ef 失败")


def test_induction_prompt_contains_no_ground_truth():
    """v10 §7.1 取代 v9 的 `build_induction_prompt(failure_summaries, input_features)`。

    规范原文（§7.1）——离线归纳器每次只收到："父 Skill 完整 `SkillSpec`；父版本的
    ExperienceBundle；允许使用的失败摘要和行为摘要；当前 ToolSpec 摘要；候选输出
    Schema；禁止泄漏和禁止修改项。"

    因此归纳 prompt 现在由**父 Skill + 经验包摘要 + 工具名清单**构成；本测试断言
    该 prompt 依然不含 ground truth / sample id / 答案（扫描不抛异常即通过）。
    """
    from skill3d.evolution.experience import (
        EpisodeEvidence, build_experience_bundle, build_experience_events,
    )
    from skill3d.schemas import SkillSpec

    parent = SkillSpec(skill_id="S01", version="1.0.0",
                       applicable_question_types=["object_counting"],
                       skill_family="counting", description="数对象",
                       call_graph_template="n = detect_objects(img)\nReturnAnswer(str(n))")
    record = {
        "retrieval_index": 1,
        "candidates": [{"skill_version": "S01@1.0.0", "hard_filter_passed": True,
                        "selected": True, "reason_code": "hit", "delivered": True,
                        "delivery_reason": "delivered"}],
        "delivered_content_sha256": {"S01@1.0.0": "sha"},
        "usage_clues": [{"skill_version": "S01@1.0.0", "declared_in_program": True}],
    }
    evidence = [
        EpisodeEvidence(episode_id=f"e{i}", scene_id=f"s{i}", split="learning",
                        snapshot_id="S0", trace={"retrieval_records": [record]},
                        answer_correct=(i % 2 == 0), failure_categories=("perception",),
                        outcome_ref=f"evaluation_result:e{i}")
        for i in range(4)
    ]
    events = build_experience_events(campaign_id="C", generation=1,
                                     parent_snapshot_id="S0",
                                     skills=["S01@1.0.0"], episodes=evidence)
    bundle = build_experience_bundle(
        campaign_id="C", generation=1, parent_snapshot_id="S0",
        parent_skill_key="S01@1.0.0", canonical_question_type="object_counting",
        events=events)
    prompt = build_induction_prompt(parent, bundle, ["detect_objects"])
    scan_prompt_for_leakage(prompt)          # 不抛异常即通过
    assert "e0" not in prompt and "s0" not in prompt     # sample / scene id 不进 prompt
    assert "S01" in prompt and '"failure_summary"' in prompt  # 父身份与失败摘要进了


def test_final_test_dir_not_mounted():
    assert_final_test_not_mounted(["/data/vsi", "/data/recon"], "/data/final_test")
    with pytest.raises(SplitContaminationError):
        assert_final_test_not_mounted(["/data/final_test"], "/data/final_test")
    with pytest.raises(SplitContaminationError):
        assert_final_test_not_mounted(["/data"], "/data/final_test")  # 父目录覆盖
