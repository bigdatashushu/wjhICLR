"""v11 induction trace-to-experience conversion contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from skill3d.evolution.experience_v11 import (
    V11ExperienceBuildError,
    V11TraceCollector,
    build_v11_experience_bundle_from_trace_store,
)
from skill3d.online.submission import EXECUTION_PROTOCOL_VERSION
from skill3d.schemas import (
    EpisodeInputTrace,
    EpisodeTrace,
    EvaluationResultTrace,
    ProgramExecutionTrace,
    SkillSpecV11,
    ToolCall,
    ToolResult,
    V11ExperienceCase,
)
from skill3d.synthesis.prompt_builder import PROMPT_TEMPLATE_VERSION
from skill3d.tools.docs_v11 import TOOL_DOCS_VERSION
from skill3d.trace.store import TraceStore

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "skill_library"
    / "versions"
    / "S03"
    / "1.1.0"
    / "rank-object-distances"
    / "SKILL.md"
)


def _parent() -> SkillSpecV11:
    return SkillSpecV11(
        skill_id="S03",
        version="1.1.0",
        question_type="object_rel_distance",
        skill_md=SOURCE.read_text(encoding="utf-8"),
    )


def _write_run(store: TraceStore, *, split="induction", mode="real") -> None:
    store.append("online_run", {
        "run_id": "learning-run-1",
        "mode": mode,
        "split": split,
        "template_version": PROMPT_TEMPLATE_VERSION,
        "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
        "tool_docs_version": TOOL_DOCS_VERSION,
    })


def _write_case(
    store: TraceStore,
    parent: SkillSpecV11,
    *,
    qa_id: str,
    correct: bool,
    source_split: str = "induction",
    delivered_hash: str | None = None,
) -> None:
    key = f"{parent.skill_id}@{parent.version}"
    program_id = f"program-{qa_id}"
    model_answer = "B" if correct else "A"
    store.append("episode_input", EpisodeInputTrace(
        qa_id=qa_id,
        scene_id=f"scene-{qa_id}",
        dataset="vsibench",
        question_type=parent.question_type,
        question_text=f"Which object is closest to the chair in case {qa_id}?",
        options=["lamp", "table", "sofa", "door"],
        reference_answer="B",
        source_split=source_split,
        frame_set_hash=f"frames-{qa_id}",
    ))
    digest = delivered_hash or parent.content_sha256
    store.append("episode_trace", EpisodeTrace(
        episode_id=qa_id,
        qa_id=qa_id,
        final_state="answer",
        program_trace_ref=f"program_trace:{program_id}",
        geometry_check_ref="",
        evaluation_ref=f"evaluation_result:{qa_id}",
        failure=None,
        active_snapshot_ref="S0-v11-learning",
        frame_set_hash=f"frames-{qa_id}",
        episode_status="answered",
        retrieval_records=[{
            "candidates": [{
                "skill_version": key,
                "hard_filter_passed": True,
                "selected": True,
                "delivered": True,
            }],
            "delivered_skill_versions": [key],
            "delivered_content_sha256": {key: digest},
        }],
        retrieved_skill_versions=[key],
        delivered_skill_versions=[key],
    ))
    program_source = (
        "rank = relative_distance_rank('chair', ['lamp', 'table'])\n"
        f"return ReturnAnswer('{model_answer}')\n"
    )
    raw_response = f"```python\n{program_source}```\n"
    store.append("episode_program", {
        "qa_id": qa_id,
        "scene_name": f"scene-{qa_id}",
        "task": parent.question_type,
        "program_id": program_id,
        "program_source": program_source,
        "response_text": raw_response,
        "response_texts": [raw_response],
        "delivered_skill_versions": [key],
    })
    result = ToolResult(
        result_id=f"result-{qa_id}",
        source_tool="relative_distance_rank",
        tool="relative_distance_rank",
        args={"reference": "chair", "candidate_categories": ["lamp", "table"]},
        payload={"closest_category": "table"},
        value='{"closest_category":"table"}',
        source="real",
    )
    store.append("program_trace", ProgramExecutionTrace(
        program_id=program_id,
        calls=[ToolCall(
            tool="relative_distance_rank",
            args={"reference": "chair", "candidate_categories": ["lamp", "table"]},
            call_id=f"call-{qa_id}",
        )],
        results=[result],
        stdout_tail="",
        error_code=None,
        steps=2,
        wallclock_s=0.1,
    ))
    store.append("evaluation_result", EvaluationResultTrace(
        qa_id=qa_id,
        question_type=parent.question_type,
        task=parent.question_type,
        is_mca=True,
        predicted=model_answer,
        ground_truth="B",
        correct=correct,
        mra_value=None,
        answer_text=model_answer,
        answer_source="tool_program",
        episode_status="answered",
    ))


def _complete_trace(tmp_path: Path):
    parent = _parent()
    store = TraceStore(tmp_path)
    _write_run(store)
    _write_case(store, parent, qa_id="success", correct=True)
    _write_case(store, parent, qa_id="failure", correct=False)
    return parent


def test_builds_complete_learning_bundle_from_structured_trace(tmp_path):
    parent = _complete_trace(tmp_path)

    bundle = build_v11_experience_bundle_from_trace_store(
        tmp_path,
        campaign_id="campaign-trace",
        parent_snapshot_id="S0-v11-learning",
        parent=parent,
    )

    assert bundle.source_split == "induction"
    assert bundle.split == "learning"
    assert bundle.label_access is True
    assert bundle.source_run_ref == "online_run:learning-run-1"
    assert {case.outcome for case in bundle.cases} == {"success", "failure"}
    by_qa = {case.qa_id: case for case in bundle.cases}
    assert by_qa["success"].question_text.endswith("case success?")
    assert by_qa["success"].options == ["lamp", "table", "sofa", "door"]
    assert by_qa["success"].model_answer == "B"
    assert by_qa["success"].reference_answer == "B"
    assert by_qa["success"].score == 1.0
    assert by_qa["success"].tool_observations[0]["tool"] == (
        "relative_distance_rank"
    )
    assert by_qa["success"].response_text.startswith("```python")
    assert by_qa["success"].response_text != by_qa["success"].program_source
    assert by_qa["failure"].model_answer == "A"
    assert by_qa["failure"].score == 0.0
    assert by_qa["failure"].error_direction == "wrong_option"
    assert by_qa["failure"].failure_categories == ["answer_incorrect"]
    assert all(case.delivered_skill_md == parent.skill_md for case in bundle.cases)


def test_injectable_collector_passes_full_bundle(tmp_path):
    parent = _complete_trace(tmp_path)
    collector = V11TraceCollector(tmp_path)

    bundle = collector(
        campaign_id="campaign-trace",
        parent=parent,
        parent_snapshot_id="S0-v11-learning",
        question_type=parent.question_type,
    )

    assert bundle.cases[0].question_text
    assert bundle.cases[0].reference_answer == "B"


@pytest.mark.parametrize("split", [
    "inner_validation",
    "outer_holdout",
    "final_test",
])
def test_rejects_non_induction_run(tmp_path, split):
    parent = _parent()
    store = TraceStore(tmp_path)
    _write_run(store, split=split)
    _write_case(store, parent, qa_id="case", correct=False, source_split=split)

    with pytest.raises(V11ExperienceBuildError, match="split=induction"):
        build_v11_experience_bundle_from_trace_store(
            tmp_path,
            campaign_id="campaign-trace",
            parent_snapshot_id="S0-v11-learning",
            parent=parent,
        )


def test_rejects_case_whose_source_split_is_not_induction(tmp_path):
    parent = _parent()
    store = TraceStore(tmp_path)
    _write_run(store)
    _write_case(
        store,
        parent,
        qa_id="case",
        correct=False,
        source_split="inner_validation",
    )

    with pytest.raises(V11ExperienceBuildError, match="非 induction"):
        build_v11_experience_bundle_from_trace_store(
            tmp_path,
            campaign_id="campaign-trace",
            parent_snapshot_id="S0-v11-learning",
            parent=parent,
        )


def test_rejects_delivery_hash_that_is_not_parent_content(tmp_path):
    parent = _parent()
    store = TraceStore(tmp_path)
    _write_run(store)
    _write_case(
        store,
        parent,
        qa_id="case",
        correct=False,
        delivered_hash="0" * 64,
    )

    with pytest.raises(V11ExperienceBuildError, match="不是父 Skill hash"):
        build_v11_experience_bundle_from_trace_store(
            tmp_path,
            campaign_id="campaign-trace",
            parent_snapshot_id="S0-v11-learning",
            parent=parent,
        )


def test_rejects_mock_run_for_formal_revision(tmp_path):
    parent = _parent()
    store = TraceStore(tmp_path)
    _write_run(store, mode="mock_light")
    _write_case(store, parent, qa_id="case", correct=False)

    with pytest.raises(V11ExperienceBuildError, match="mode=real"):
        build_v11_experience_bundle_from_trace_store(
            tmp_path,
            campaign_id="campaign-trace",
            parent_snapshot_id="S0-v11-learning",
            parent=parent,
        )


@pytest.mark.parametrize("field", ["source_split", "split", "label_access"])
def test_case_schema_requires_explicit_learning_identity(tmp_path, field):
    parent = _parent()
    store = TraceStore(tmp_path)
    _write_run(store)
    _write_case(store, parent, qa_id="case", correct=False)
    bundle = build_v11_experience_bundle_from_trace_store(
        tmp_path,
        campaign_id="campaign-trace",
        parent_snapshot_id="S0-v11-learning",
        parent=parent,
    )
    payload = bundle.cases[0].model_dump(mode="json")
    payload.pop(field)

    with pytest.raises(ValidationError):
        V11ExperienceCase.model_validate(payload)
