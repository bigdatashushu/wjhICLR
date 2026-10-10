"""The online runtime exposes one fixed prompt and Tool-document contract."""

from __future__ import annotations

import inspect

import pytest

from skill3d.online.eval import build_parser, current_version_fields
from skill3d.online.runner import OnlineRunConfig
from skill3d.synthesis import prompt_builder
from skill3d.synthesis.prompt_builder import PROMPT_TEMPLATE_VERSION, PromptBuilder
from skill3d.tools.docs_v11 import TOOL_DOCS_VERSION


def test_current_protocol_identity_is_fixed_and_not_cli_selectable():
    parser = build_parser()
    assert "--prompt-template-version" not in parser.format_help()
    assert "prompt_template_version" not in {
        field.name for field in __import__("dataclasses").fields(OnlineRunConfig)
    }
    assert PromptBuilder().template_version == PROMPT_TEMPLATE_VERSION
    versions = current_version_fields()
    assert versions["template_version"] == "program_synth_v11_3"
    assert versions["execution_protocol_version"] == "solver-v11.4-eval-visual-fallback"
    assert versions["tool_docs_version"] == TOOL_DOCS_VERSION


def test_historical_prompt_selectors_are_absent():
    source = inspect.getsource(prompt_builder)
    for stale in (
        "program_synth_v8",
        "program_synth_v11_1",
        "V11_TEMPLATE_VERSIONS",
        "SUPPORTED_TEMPLATE_VERSIONS",
        "validate_template_version",
        "tool_docs_version_for",
    ):
        assert stale not in source


def test_online_config_rejects_legacy_skill_spec_before_execution():
    legacy = dict(
        skill_id="legacy",
        version="1.0.0",
        applicable_question_types=["object_counting"],
        skill_family="counting",
    )
    with pytest.raises(ValueError, match="只接受 SkillSpecV11"):
        OnlineRunConfig(skills=[legacy])


def test_online_config_rejects_retired_direct_answer_baseline():
    with pytest.raises(ValueError, match="只支持 C1_tools_program"):
        OnlineRunConfig(baseline="C0_direct_vlm")


@pytest.mark.parametrize("args", [
    ["--baseline", "C0_direct_vlm"],
    ["--skill-spec", "old.json"],
    ["--inject-wrong-skill"],
])
def test_cli_rejects_retired_solver_and_manual_injection(args):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(args)
    assert exc.value.code == 2


@pytest.mark.parametrize(("question_type", "meaning"), [
    ("object_counting", "非负整数"),
    ("object_abs_distance", "两个对象之间的最近距离"),
    ("object_rel_distance", "参照对象不是相机"),
    ("object_size_estimation", "最大物理维度"),
    ("room_size_estimation", "地面面积"),
    ("object_rel_direction", "面向指定对象"),
    ("obj_appearance_order", "首次出现顺序"),
    ("route_planning", "完整动作序列"),
])
def test_public_task_semantics_do_not_include_solving_recipes(question_type, meaning):
    text = PromptBuilder().render(
        question="题面要求",
        scene_summary="",
        scene_frame="world",
        tool_docs="",
        question_type=question_type,
    )
    assert meaning in text
    assert "ReturnAnswer" in text and "YieldObservations" in text
    assert "米转厘米乘以 100" in text
    assert "已失效结果不得继续支持答案" in text
    assert "## 参考方法" not in text
    assert "坐标系: world" not in text
    for stale in (
        "对象质心距离",
        "优先 `count_objects`",
        "relative_direction_of(",
        "先把所有候选都算出来",
        "## 题目口径",
    ):
        assert stale not in text


def test_return_answer_is_a_pending_submission():
    text = PromptBuilder().render(
        question="题面", scene_summary="", scene_frame="world", tool_docs="")
    assert "提交暂存答案" in text
    assert "验收失败时撤销提交" in text
