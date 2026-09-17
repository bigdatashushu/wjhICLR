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
    # 正常归纳 prompt（只含失败类型摘要与输入特征）应通过扫描
    prompt = build_induction_prompt(
        failure_summaries=["perception", "coordinate"],
        input_features=["abs_dist", "rel_direction"])
    scan_prompt_for_leakage(prompt)  # 不抛异常即通过


def test_final_test_dir_not_mounted():
    assert_final_test_not_mounted(["/data/vsi", "/data/recon"], "/data/final_test")
    with pytest.raises(SplitContaminationError):
        assert_final_test_not_mounted(["/data/final_test"], "/data/final_test")
    with pytest.raises(SplitContaminationError):
        assert_final_test_not_mounted(["/data"], "/data/final_test")  # 父目录覆盖
