"""The online runtime exposes one fixed prompt and Tool-document contract."""

from __future__ import annotations

import inspect
import re

import pytest

from skill3d.online.eval import build_parser, current_version_fields
from skill3d.online.runner import OnlineRunConfig
from skill3d.synthesis import prompt_builder
from skill3d.synthesis.prompt_builder import PROMPT_TEMPLATE_VERSION, PromptBuilder
from skill3d.tools.docs_v11 import DESCRIPTIONS, TOOL_DOCS_VERSION


def test_current_protocol_identity_is_fixed_and_not_cli_selectable():
    parser = build_parser()
    assert "--prompt-template-version" not in parser.format_help()
    assert "prompt_template_version" not in {
        field.name for field in __import__("dataclasses").fields(OnlineRunConfig)
    }
    assert PromptBuilder().template_version == PROMPT_TEMPLATE_VERSION
    versions = current_version_fields()
    assert versions["template_version"] == "program_synth_v11_11_en"
    assert versions["execution_protocol_version"] == "solver-v11.10-response-envelope"
    assert versions["tool_docs_version"] == TOOL_DOCS_VERSION


def test_list_objects_documents_envelope_data_and_counting_contract():
    text = DESCRIPTIONS["list_objects"]
    assert "envelope's data is list[dict]" in text
    assert "result['data']" in text and "result['result_id']" in text
    assert "never obj_id values" in text
    assert "candidate inventory" in text
    assert "collection coverage preconditions" in text


def test_detect_objects_documents_no_repeat_transition():
    text = DESCRIPTIONS["detect_objects"]
    assert "do not repeat detect_objects with unchanged" in text
    assert "Analyze the registered candidates with list_objects" in text
    assert "selected views or crops with inspect_frames" in text


def test_historical_prompt_selectors_are_absent():
    source = inspect.getsource(prompt_builder)
    for stale in (
        "program_synth_v8",
        "program_synth_v11_1_en",
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
    ("object_counting", "non-negative integer"),
    ("object_abs_distance", "nearest distance between the two named objects"),
    ("object_rel_distance", "the reference is not the camera"),
    ("object_size_estimation", "largest physical dimension"),
    ("room_size_estimation", "floor area"),
    ("object_rel_direction", "facing the specified object"),
    ("obj_appearance_order", "first appear in the video"),
    ("route_planning", "complete action sequence"),
])
def test_public_task_semantics_do_not_include_solving_recipes(question_type, meaning):
    text = PromptBuilder().render(
        question="Question requirements",
        scene_summary="",
        scene_frame="world",
        tool_docs="",
        question_type=question_type,
    )
    assert meaning in text
    assert "ReturnAnswer" in text and "YieldObservations" in text
    assert "Multiply by 100 to convert meters to centimeters" in text
    assert "Invalidated results must not support the answer" in text
    assert "## Reference method" not in text
    assert "Coordinate frame: world" not in text
    assert not re.search(r"[\u4e00-\u9fff]", text)
    for stale in (
        "object centroid distance",
        "prefer `count_objects`",
        "relative_direction_of(",
        "first compute all candidates",
        "## Task-specific recipe",
    ):
        assert stale not in text


def test_return_answer_is_a_pending_submission():
    text = PromptBuilder().render(
        question="Question", scene_summary="", scene_frame="world", tool_docs="")
    assert "submits a provisional answer" in text
    assert "A rejected submission is revoked" in text


def test_question_and_option_data_remain_verbatim():
    question = "这把椅子距离桌子多远？"
    options = ["A. 左侧", "B. 右侧"]
    text = PromptBuilder().render(
        question=question, options=options, scene_summary="", scene_frame="world",
        tool_docs="", question_type="object_rel_direction")
    assert question in text
    assert all(option in text for option in options)
    instructions = text.replace(question, "")
    for option in options:
        instructions = instructions.replace(option, "")
    assert not re.search(r"[\u4e00-\u9fff]", instructions)
