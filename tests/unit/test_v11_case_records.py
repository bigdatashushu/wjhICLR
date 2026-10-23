"""Offline admission and honest all-round provenance contracts."""

from copy import deepcopy
from dataclasses import make_dataclass
import hashlib
from pathlib import Path

import pytest

from skill3d.evaluation.fitset_v11 import score_answer
from skill3d.evolution.case_records_v11 import (
    V11ExperienceBuildError,
    case_from_fitset_outcome,
    collect_case_tool_observations,
    extract_case_image_refs,
    extract_case_rounds,
)
from skill3d.schemas import DataAccessRecord, SkillSpecV11


ROOT = Path(__file__).resolve().parents[2]


def _inputs():
    parent = SkillSpecV11(
        skill_id="S03", version="1.1.0",
        question_type="object_rel_distance", evidence_state="g1m1",
        skill_md=(ROOT / "skill_library/versions/S03/1.1.0/rank-object-distances/SKILL.md").read_text(),
    )
    key = f"{parent.skill_id}@{parent.version}"
    initial = {"geometry_3d": "available", "metric_scale": "available"}
    final = {"geometry_3d": "available", "metric_scale": "unavailable"}
    digest = hashlib.sha256(b"frozen-test-frames").hexdigest()
    variant = {
        "variant_id": "qa-1@g1m1", "qa_id": "qa-1", "scene_id": "scene-1",
        "question_type": parent.question_type, "evidence_state": parent.evidence_state,
        "split": "induction",
    }
    outcome = {
        "qa_id": "qa-1", "scene_name": "scene-1", "question_type": parent.question_type,
        "task": parent.question_type, "final_state": "answer", "episode_status": "answered",
        "answer": "B", "label_access": False, "frame_set_hash": digest,
        "delivered_skill_versions": [key],
        "initial_skill_selection": {
            "question_type": parent.question_type, "evidence_state": parent.evidence_state,
            "skill_version": key, "content_sha256": parent.content_sha256,
        },
        "initial_evidence_states": initial, "evidence_states": final,
        "evidence_history": [
            {"trigger": "initial", "evidence_states": initial, "candidates": []},
            {"trigger": "final", "evidence_states": final,
             "candidates": [{"obj_id": "chair-1", "category_name": "chair"}]},
        ],
        "retrieval_records": [{
            "delivered_skill_versions": [key],
            "delivered_content_sha256": {key: parent.content_sha256},
            "candidates": [{"skill_version": key, "delivered": True}],
        }],
        "rounds": [
            {
                "index": 1, "trigger": "initial", "synthesis_source": "online_model_ok",
                "program_id": "p1", "program_source": "inspect_frames([0])",
                "response_text": "First original response\n```python\ninspect_frames([0])\n```",
                "executed": True, "observed_result_ids": ["r1"],
                "results": [{
                    "result_id": "r1", "tool": "inspect_frames", "source": "real",
                    "payload": {"images": ["crop-0"], "note": "original observation"},
                    "status": "ok", "resolved_object_ids": [],
                }],
                "observation_reads": [], "error_code": "",
                "custom_audit": {"keep": ["original"]},
            },
            {
                "index": 2, "trigger": "observation", "synthesis_source": "online_model_ok",
                "program_id": "p2", "program_source": "return ReturnAnswer('B')",
                "response_text": "Second original response\n```python\nreturn ReturnAnswer('B')\n```",
                "executed": True, "observed_result_ids": ["r2"],
                "used_result_ids": ["r1", "r2", "unrecorded-result"],
                "invalidated_result_ids": ["r1"], "error_code": "",
                "observation_reads": [{"result_id": "r1", "payload": {"actual": "read"}}],
                "results": [{
                    "result_id": "r2", "tool": "relative_distance_rank", "source": "real",
                    "payload": {"closest_category": "table"}, "status": "ok",
                    "resolved_object_ids": ["chair-1", "table-1"],
                    "result_dependencies": ["r1"],
                }],
            },
        ],
        "image_ledger": {
            "images": [{
                "image_id": "crop-0", "kind": "crop", "result_id": "r1",
                "source_frame_id": 0, "box_xyxy": [1, 2, 20, 30],
                "content_sha256": hashlib.sha256(b"recorded-image-identity").hexdigest(),
                "delivered_rounds": [2], "observed_rounds": [2],
            }],
            "rounds": [
                {"round_index": 1, "image_ids": [], "delivered": True, "observed": True},
                {"round_index": 2, "image_ids": ["crop-0"], "delivered": True, "observed": True},
            ],
        },
    }
    return {
        "variant": variant, "outcome": outcome, "parent": parent,
        "label": {"qa_id": "qa-1", "scene_id": "scene-1",
                  "question_type": parent.question_type,
                  "options": ["lamp", "table"], "reference_answer": "B"},
        "trace_ref": "outcome.json", "question_text": "Which object is closest?",
        "options": ["lamp", "table"],
    }


def test_all_rounds_preserve_originals_and_attribute_links():
    inputs = _inputs()
    outcome = inputs["outcome"]
    before = deepcopy(outcome)
    rounds = extract_case_rounds(outcome, trace_ref="outcome.json")

    assert len(rounds) == 2
    for row, original in zip(rounds, outcome["rounds"]):
        assert {key: row[key] for key in original} == original
    first, second = [row["experience_provenance"] for row in rounds]
    assert "used_result_ids" in first["missing_fields"]
    assert first["image_refs"] == []
    assert second["image_refs"][0]["image_id"] == "crop-0"
    links = {(link["relation"], link["result_id"]): link for link in second["result_links"]}
    assert links["used_result_ids", "r1"]["source_refs"] == ["outcome.json#/rounds/0/results/0"]
    assert links["used_result_ids", "unrecorded-result"]["status"] == "unresolved"
    assert links["invalidated_result_ids", "r1"]["status"] == "recorded"
    assert links["observation_reads", "r1"]["source_refs"] == ["outcome.json#/rounds/0/results/0"]
    assert links["result_dependencies", "r1"]["source_refs"] == ["outcome.json#/rounds/0/results/0"]
    assert rounds[1]["observation_reads"][0]["payload"] == {"actual": "read"}
    rounds[0]["results"][0]["payload"]["note"] = "changed copy"
    assert outcome == before


def test_missing_round_data_is_not_backfilled_from_final_evidence():
    rows = extract_case_rounds({
        "rounds": [{"response_text": "original partial row"}],
        "used_result_ids": ["episode-final"], "invalidated_result_ids": ["final-invalidation"],
        "program_source": "episode-final-source",
        "image_ledger": {"images": [{"image_id": "only-id"}],
                         "rounds": [{"round_index": 1, "image_ids": ["only-id"]}]},
    })
    assert "program_source" not in rows[0]
    provenance = rows[0]["experience_provenance"]
    assert provenance["result_links"] == []
    assert provenance["image_attribution"] == "unknown"
    assert {"index", "results", "program_source"} <= set(provenance["missing_fields"])
    assert extract_case_image_refs({"image_ledger": {"images": [{"image_id": "only-id"}]}})[0][
        "experience_provenance"]["observation_status"] == "unknown"


def test_flat_observations_keep_same_id_revisions_and_deduplicate_exact_copies():
    original = _inputs()["outcome"]["rounds"][0]["results"][0]
    invalidated = {**deepcopy(original), "invalidated_by": ["r2"]}
    observations = collect_case_tool_observations(
        {"rounds": [{"results": [original]}, {"results": [invalidated]}]},
        {"results": [invalidated]},
    )
    assert observations == [original, invalidated]
    assert "invalidated_by" not in observations[0]


def test_adapter_builds_production_case_and_records_offline_label_access():
    inputs = _inputs()
    before = deepcopy(inputs["outcome"])
    case = case_from_fitset_outcome(**inputs)

    assert case.qa_id == "qa-1"
    assert case.source_split == "induction" and case.split == "learning"
    assert case.label_access is True and case.score == 1 and case.outcome == "success"
    assert case.response_texts == [r["response_text"] for r in before["rounds"]]
    assert case.program_source == before["rounds"][-1]["program_source"]
    assert case.frame_set_hash == before["frame_set_hash"]
    assert case.image_refs[0]["source_frame_id"] == 0
    assert case.image_observations == before["image_ledger"]
    assert case.selected_object_ids == []
    assert case.tool_observations == [r["results"][0] for r in before["rounds"]]
    access = DataAccessRecord.model_validate(
        case.rounds[0]["experience_provenance"]["offline_label_access_record"])
    assert access.label_access is True and access.n_items == 1
    assert access.split == "induction" and access.component_role == "inducer"
    assert access.input_manifest_sha256 and access.at
    assert inputs["outcome"] == before
    assert inputs["outcome"]["label_access"] is False


@pytest.mark.parametrize("split", ["inner_validation", "outer_holdout", "final_test"])
def test_non_induction_rejected_before_any_label_read(split):
    class UnreadableLabel(dict):
        def get(self, *args):
            raise AssertionError("label must not be read")

    inputs = _inputs()
    inputs["variant"]["split"] = split
    inputs["label"] = UnreadableLabel()
    with pytest.raises(V11ExperienceBuildError, match="induction only"):
        case_from_fitset_outcome(**inputs)


@pytest.mark.parametrize("damage", [
    "qa", "scene", "task", "slot", "initial_evidence", "selection_hash",
    "delivery_hash", "delivery_missing", "candidate_not_delivered", "extra_parent",
    "program_hash", "online_label", "frame_hash", "label_scene", "label_options",
    "no_program", "missing_results", "mock_source", "mock_result",
])
def test_adapter_rejects_untrustworthy_identity_or_trace(damage):
    inputs = _inputs()
    outcome = inputs["outcome"]
    if damage == "qa":
        outcome["qa_id"] = "other"
    elif damage == "scene":
        outcome["scene_name"] = "other"
    elif damage == "task":
        outcome["task"] = "object_counting"
    elif damage == "slot":
        inputs["variant"]["evidence_state"] = "g0m0"
    elif damage == "initial_evidence":
        outcome["initial_evidence_states"] = {"geometry_3d": "unavailable", "metric_scale": "unavailable"}
    elif damage == "selection_hash":
        outcome["initial_skill_selection"]["content_sha256"] = "0" * 64
    elif damage == "delivery_hash":
        key = outcome["delivered_skill_versions"][0]
        outcome["retrieval_records"][0]["delivered_content_sha256"][key] = "0" * 64
    elif damage == "delivery_missing":
        outcome["retrieval_records"] = []
    elif damage == "candidate_not_delivered":
        outcome["retrieval_records"][0]["candidates"][0]["delivered"] = False
    elif damage == "extra_parent":
        outcome["delivered_skill_versions"].append("other@1.0.0")
    elif damage == "program_hash":
        outcome["rounds"][0]["program_sha256"] = "0" * 64
    elif damage == "online_label":
        outcome["label_access"] = True
    elif damage == "frame_hash":
        inputs["frame_set_hash"] = "0" * 64
    elif damage == "label_scene":
        inputs["label"]["scene_id"] = "other"
    elif damage == "label_options":
        inputs["label"]["options"] = ["different"]
    elif damage == "no_program":
        for row in outcome["rounds"]:
            row["executed"] = False
    elif damage == "missing_results":
        outcome["rounds"][0].pop("results")
    elif damage == "mock_source":
        outcome["rounds"][0]["synthesis_source"] = "mock_stub"
    else:
        outcome["rounds"][0]["results"][0]["source"] = "mock_light"
    with pytest.raises(V11ExperienceBuildError):
        case_from_fitset_outcome(**inputs)


@pytest.mark.parametrize("damage", ["unavailable", "input_error", "service",
                                    "tool_fault", "sigalrm", "oom"])
def test_infrastructure_is_rejected_even_with_a_correct_answer(damage):
    inputs = _inputs()
    outcome = inputs["outcome"]
    if damage in {"unavailable", "input_error"}:
        outcome["final_state"] = damage
    elif damage == "service":
        outcome["rounds"][0]["service_errors"] = [{"tool": "inspect_frames"}]
    elif damage == "tool_fault":
        outcome["rounds"][0]["results"][0]["payload"]["service_healthy"] = False
    elif damage == "sigalrm":
        outcome["rounds"][0]["execution_diagnostics"] = {
            "error_message": "cell timeout requires a main-thread SIGALRM runtime"}
    else:
        outcome["failure_code"] = "oom"
    with pytest.raises(V11ExperienceBuildError):
        case_from_fitset_outcome(**inputs)


def test_completed_program_failure_is_learning_not_an_infrastructure_failure():
    inputs = _inputs()
    inputs["outcome"].update(final_state="run_error", episode_status="run_error", answer=None)
    inputs["outcome"]["rounds"][-1]["error_code"] = "violation_runtime"
    case = case_from_fitset_outcome(**inputs)
    assert case.outcome == "failure" and case.score == 0
    assert case.failure_categories == ["violation_runtime"]
    assert case.error_direction == "run_error"
    assert case.rounds[0]["results"][0]["payload"]["note"] == "original observation"


def test_numeric_adapter_reuses_canonical_scoring():
    inputs = _inputs()
    parent = inputs["parent"].model_copy(update={"question_type": "object_size_estimation"})
    inputs["parent"] = parent
    for metadata in (inputs["variant"], inputs["outcome"], inputs["label"],
                     inputs["outcome"]["initial_skill_selection"]):
        metadata["question_type"] = parent.question_type
    inputs["outcome"]["task"] = parent.question_type
    inputs["options"] = inputs["label"]["options"] = []
    inputs["label"]["reference_answer"] = "100"
    inputs["outcome"]["answer"] = "80"
    case = case_from_fitset_outcome(**inputs)
    assert case.score == score_answer(answer="80", task=parent.question_type,
                                     options=[], reference="100", available=True)["score"]
    assert case.error_direction == "underestimate"


@pytest.mark.parametrize("ref", [
    {"image_id": "x", "bytes": b"pixels"},
    {"image_id": "x", "gt_3d": {"position": [0, 0, 0]}},
    {"image_id": "x", "source_frame_id": {"gt_3d": [0, 0, 0]}},
    {"image_id": "x", "url": "data:image/jpeg;base64,anything"},
])
def test_image_refs_never_embed_pixels_or_gt3d(ref):
    with pytest.raises(V11ExperienceBuildError):
        extract_case_image_refs({}, image_refs=[ref])


def test_adapter_accepts_recorded_dataclass_and_preserves_missing_frame_identity():
    inputs = _inputs()
    inputs["outcome"].pop("frame_set_hash")
    Outcome = make_dataclass("RecordedOutcome", [
        (name, object) for name in inputs["outcome"]])
    inputs["outcome"] = Outcome(**inputs["outcome"])
    case = case_from_fitset_outcome(**inputs)
    assert case.frame_set_hash == ""
    assert len(case.rounds) == 2


def test_summary_program_is_latest_executed_not_unexecuted_parse_failure():
    inputs = _inputs()
    inputs["outcome"]["rounds"].append({
        "index": 3, "executed": False, "synthesis_source": "online_model_parse_error",
        "program_source": "", "response_text": "original unparseable final reply",
        "error_code": "model_output_format_error",
    })
    inputs["outcome"].update(final_state="run_error", episode_status="run_error")
    case = case_from_fitset_outcome(**inputs)
    assert case.program_source == inputs["outcome"]["rounds"][1]["program_source"]
    assert case.rounds[-1]["program_source"] == ""
    assert case.response_text == "original unparseable final reply"
    assert case.rounds[-1]["experience_provenance"]["missing_fields"]


def test_explicit_selection_is_preserved_without_inventing_more_from_resolved_ids():
    inputs = _inputs()
    inputs["outcome"]["rounds"][1]["selected_object_ids"] = ["chair-1"]
    case = case_from_fitset_outcome(**inputs)
    assert case.selected_object_ids == ["chair-1"]
    assert case.tool_observations[-1]["resolved_object_ids"] == ["chair-1", "table-1"]


def test_image_round_attribution_uses_only_explicit_provenance():
    inputs = _inputs()
    inputs["outcome"]["image_ledger"].pop("rounds")
    rounds = extract_case_rounds(inputs["outcome"])
    assert rounds[0]["experience_provenance"]["image_attribution"] == "unknown"
    assert rounds[1]["experience_provenance"]["image_refs"][0]["observed_rounds"] == [2]
    assert rounds[1]["experience_provenance"]["image_rounds"] == []


@pytest.mark.parametrize("damage", ["fault", "sigalrm"])
def test_late_program_trace_cannot_poison_an_otherwise_valid_case(damage):
    inputs = _inputs()
    program_trace = {
        "program_id": "p2", "error_code": None,
        "results": deepcopy(inputs["outcome"]["rounds"][1]["results"]),
    }
    if damage == "fault":
        program_trace["results"][0]["payload"]["status"] = "fault"
    else:
        program_trace["error_message"] = "cell timeout requires a main-thread SIGALRM runtime"
    inputs["outcome"]["program_trace"] = program_trace
    with pytest.raises(V11ExperienceBuildError):
        case_from_fitset_outcome(**inputs)


def test_external_file_hash_provenance_does_not_claim_online_observation():
    inputs = _inputs()
    external = {"image_id": "original-0", "path": "frozen/frame-0.png",
                "file_sha256": "c" * 64,
                "frame_set_hash": inputs["outcome"]["frame_set_hash"]}
    inputs["image_refs"] = (external,)
    case = case_from_fitset_outcome(**inputs)
    ref = case.image_refs[-1]
    assert {key: ref[key] for key in external} == external
    assert ref["experience_provenance"]["content_identity"] == "recorded"
    assert ref["experience_provenance"]["observation_status"] == "unknown"
    assert "observed_rounds" not in ref


def test_parent_mismatch_is_rejected_before_any_label_read():
    class UnreadableLabel(dict):
        def get(self, *args):
            raise AssertionError("label must not be read")

    inputs = _inputs()
    inputs["label"] = UnreadableLabel()
    inputs["outcome"]["initial_skill_selection"]["content_sha256"] = "0" * 64
    with pytest.raises(V11ExperienceBuildError, match="parent slot/identity/hash mismatch"):
        case_from_fitset_outcome(**inputs)


def test_canonical_case_serializes_inside_the_separate_legacy_fitset_wrapper():
    from skill3d.evolution.fitset_stream_v11 import (
        FitSetExperienceCase, collect_fitset_case,
    )

    inputs = _inputs()
    legacy = collect_fitset_case(
        inputs["variant"], inputs["outcome"], inputs["parent"],
        inputs["label"], inputs["trace_ref"])
    assert legacy.full_case is None
    legacy.full_case = case_from_fitset_outcome(**inputs)
    restored = FitSetExperienceCase.model_validate_json(legacy.model_dump_json())
    assert restored.full_case.model_dump(mode="json") == legacy.full_case.model_dump(mode="json")
    assert restored.full_case.rounds[0]["results"][0]["payload"]["note"] == "original observation"
    assert restored.full_case.selected_object_ids == []
