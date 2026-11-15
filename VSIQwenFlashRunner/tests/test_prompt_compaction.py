"""Offline prompt/authorization checks for all eight tasks and three evidence states."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.evolution.fitset_revision_v11 import load_active_parent_slots
from skill3d.infra.vsi_runner_source import seed_library_root, source_report
from skill3d.online import runner
from skill3d.online.vsi_execution_contract import (
    MEASUREMENT_FIELDS, finalization_example, program_instructions, visual_example,
)
from skill3d.sandbox.ast_guard import ast_guard
from skill3d.schemas.answer import CANONICAL_UNIT_BY_QUESTION_TYPE, derivation_contract
from skill3d.schemas.evidence import EvidenceProfile
from skill3d.skills.delivery import plan_delivery, skill_content_sha256
from skill3d.synthesis import prompt_builder
from skill3d.synthesis.execution_contract import ExecutionContract
from skill3d.synthesis.prompt_compaction import (
    COMPACTION_VERSION, DESCRIPTIONS, compact_m5_coverage, compact_tool_docs,
    compact_rendered_tool_docs, task_derivation_contract,
)
from skill3d.tools import REGISTRY
from skill3d.tools.contract import KNOWN_ARTIFACTS
from skill3d.tools.task_policy import MODEL_TOOLS

ROOT = Path(__file__).resolve().parents[1]
TASKS = sorted(CANONICAL_UNIT_BY_QUESTION_TYPE)
STATES = ("g0m0", "g1m0", "g1m1")


def context(task, state):
    geometry = state != "g0m0"
    metric = state == "g1m1"
    scope = ("metric_enabled" if metric and task in {
        "object_abs_distance", "object_size_estimation", "room_size_estimation"}
        else "full_3d" if geometry else "fallback_2d_only")
    return dict(
        scope=scope, available=KNOWN_ARTIFACTS,
        evidence_profile=EvidenceProfile(
            geometry_3d="available" if geometry else "unavailable",
            metric_scale="available" if metric else "unavailable"),
        gate_passed=scope == "metric_enabled", question_type=task,
        local_facts={"world_frame": True})


def scene(task, state, *, coverage=None):
    ctx = context(task, state)
    return SimpleNamespace(
        question_type=task, question_tool_scope=ctx["scope"], scene_route=ctx["scope"],
        summary="Synthetic scene; no reference answer.", available_artifacts=ctx["available"],
        grounding_coverage=coverage or {}, local_facts=ctx["local_facts"],
        evidence_profile=ctx["evidence_profile"],
        metric_evidence_gate_result=SimpleNamespace(
            gate_passed=ctx["gate_passed"], missing_subconditions=[]))


def episode(task):
    options = (["A. left", "B. right"] if task == "object_rel_direction"
               else ["A. first", "B. second"] if CANONICAL_UNIT_BY_QUESTION_TYPE[task] == "option"
               else [])
    return SimpleNamespace(
        qa_id="synthetic", scene_name="synthetic", frame_set=None,
        question="Synthetic spatial question preserving the specified object roles and scope.",
        question_type=task, options=options)


@pytest.fixture(scope="module")
def parents():
    return load_active_parent_slots(seed_library_root(ROOT))[0]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("state", STATES)
def test_all_task_states_preserve_authorization_signatures_modes_and_complete_skill(task, state, parents):
    ctx = context(task, state)
    original = REGISTRY.docs(**ctx)
    compact = compact_tool_docs(REGISTRY, **ctx)

    def interfaces(text):
        result = {}
        for line in text.splitlines():
            if line.startswith("- "):
                prefix, data = line.split("): ", 1)
                name = prefix[2:].split("(", 1)[0]
                result[name] = (prefix, data[data.rfind(" [requires_artifacts: "):])
        return result

    assert interfaces(compact) == interfaces(original)
    assert set(interfaces(compact)) == set(REGISTRY.names_for_scope(**ctx))
    assert len(compact) < len(original)
    skill = parents[(task, state)]
    s = scene(task, state)
    p, delivery = runner._build_prompt_ex(episode(task), s, s, [skill], None)
    assert delivery.entries and delivery.entries[0].text == skill.skill_md
    assert delivery.entries[0].content_sha256 == skill_content_sha256(skill)
    assert skill.skill_md in p
    assert "Synthetic spatial question" in p
    assert f"Submission unit: {CANONICAL_UNIT_BY_QUESTION_TYPE[task]}" in p
    assert ExecutionContract(delivery.max_chars).sha256 in p
    assert COMPACTION_VERSION in p
    assert "raw_list_count_example" not in p
    # All example programs remain executable and legal in the actual Python phase.
    for code in re.findall(r"```python\n(.*?)```", program_instructions(
            task, authorized_tools=interfaces(compact)), re.S):
        compile(code, "<prompt-example>", "exec")
        assert ast_guard(code).ok
    if task in MEASUREMENT_FIELDS:
        assert ("MeasurementAnswer(result_or_id)" in p) == (
            MEASUREMENT_FIELDS[task][0] in interfaces(compact))


def test_task_parameter_help_is_subset_of_unchanged_shared_schema():
    full = derivation_contract()
    assert set(DESCRIPTIONS) == MODEL_TOOLS
    for task in TASKS:
        compact = task_derivation_contract(task)
        assert set(compact["registered_operations"]) == set(full["allowed_parameters"])
        for op, parameters in compact["task_parameter_help"].items():
            assert parameters == full["allowed_parameters"][op]


def test_unknown_interface_retains_full_description_and_requirements():
    line = ("- future_tool(value: 'float'): New mandatory semantic constraint. "
            "[requires_artifacts: frames; requires_evidence: none; local_facts: none]")
    assert line in compact_rendered_tool_docs("Original envelope header.\n" + line)


def test_dynamic_metric_mode_disappears_with_scale_or_question_authorization():
    ctx = context("object_abs_distance", "g1m1")
    assert "[allowed units: normalized, m;" in compact_tool_docs(REGISTRY, **ctx)
    ctx.update(evidence_profile=EvidenceProfile(geometry_3d="available", metric_scale="unavailable"),
               scope="full_3d", gate_passed=False)
    compact = compact_tool_docs(REGISTRY, **ctx)
    assert "[allowed units: normalized]" in compact
    assert "[allowed units: normalized, m;" not in compact
    for task in TASKS:
        if task != "object_abs_distance":
            assert "object_surface_distance(" not in compact_tool_docs(
                REGISTRY, **context(task, "g1m1"))


def test_failure_aggregation_preserves_all_distinct_reasons_coverage_and_source():
    errors = [{"candidate_id": f"{i:064x}", "blocking": True, "error": "uncertain"}
              for i in range(20)]
    errors += [{"candidate_id": "named-candidate", "blocking": False, "error": "diagnostic",
                "role": "target"}]
    coverage = {
        "candidate_scan_complete": False, "readable_slots": [0, 2, 31],
        "unresolved_names": ["chair"], "phase_receipts": [{"raw": "keep in source"}],
        "requirements": [{
            "name": "chair", "roles": [{"role": "target", "category": "chair"}],
            "matched_object_ids": ["obj_1"], "usable_object_ids": [],
            "scan_complete": True, "scanned_slots": [0, 2, 31],
            "scan_errors": ["frame 1 unreadable"], "binding_failures": errors,
            "blocking_failures": errors[:20], "detector_errors": ["private receipt"],
        }],
    }
    before = copy.deepcopy(coverage)
    compact = compact_m5_coverage(coverage)
    assert coverage == before
    row = compact["requirements"][0]
    assert row["blocking_failure_summary"] == [{"blocking": True, "error": "uncertain", "count": 20}]
    assert row["binding_failure_count"] == 21 and row["blocking_failure_count"] == 20
    assert row["scan_complete"] is True and compact["candidate_scan_complete"] is False
    for key in ("name", "roles", "matched_object_ids", "usable_object_ids", "scan_errors", "scanned_slots"):
        assert row[key] == before["requirements"][0][key]
    assert any(x.get("candidate_id") == "named-candidate" for x in row["binding_failure_summary"])
    coverage["requirements"][0]["blocking_failures"] = errors
    row = compact_m5_coverage(coverage)["requirements"][0]
    assert row["blocking_failures_same_as_binding"] is True
    assert "blocking_failure_summary" not in row


def test_direction_coverage_separates_incomplete_scan_from_live_usable_bindings():
    failure = {"candidate_id": "other-candidate", "blocking": True, "error": "uncertain"}
    coverage = {
        "status": "unavailable", "candidate_scan_complete": False,
        "unresolved_names": ["shelf"],
        "requirements": [{
            "name": "shelf", "status": "unavailable", "scan_complete": True,
            "usable_object_ids": ["shelf_a", "stale_shelf"],
            "binding_failures": [failure], "blocking_failures": [failure],
        }],
    }
    before = copy.deepcopy(coverage)
    public = compact_m5_coverage(
        coverage, question_type="object_rel_direction",
        binding_usable=lambda obj_id: obj_id == "shelf_a",
        authorized_tools=["list_objects", "relative_direction_of"])
    row = public["requirements"][0]
    assert coverage == before
    assert "status" not in public and "status" not in row
    assert public["candidate_coverage_complete"] is False
    assert row["candidate_coverage_complete"] is False
    assert row["usable_object_ids"] == ["shelf_a"]
    assert row["reported_usable_object_ids"] == ["shelf_a", "stale_shelf"]
    assert row["binding_failure_count"] == row["blocking_failure_count"] == 1
    assert public["relative_direction_of_authorized"] is True
    assert public["binding_check_source"] == "current_handle"
    # The same source still communicates an incomplete category to other tasks.
    unchanged = compact_m5_coverage(coverage)
    assert unchanged["status"] == unchanged["requirements"][0]["status"] == "unavailable"
    assert "direction_presentation_version" not in unchanged


@pytest.mark.parametrize("world_frame,geometry,scope_override,allowed", [
    (True, True, None, True),
    (False, True, None, False),
    (True, False, None, False),
    # A scope name alone does not revoke this objects-only tool in the registry;
    # actual recovery also withdraws geometry. Presentation must follow that gate.
    (True, True, "fallback_2d_only", True),
    (True, False, "fallback_2d_only", False),
])
def test_direction_prompt_uses_current_authorization_not_coverage_or_overview(
        world_frame, geometry, scope_override, allowed, parents):
    s = scene("object_rel_direction", "g1m0" if geometry else "g0m0")
    s.local_facts["world_frame"] = world_frame
    s.summary = "Synthetic scene world_frame_diagnostic=degraded."
    s.grounding_coverage = {
        "status": "unavailable", "candidate_scan_complete": False,
        "requirements": [{
            "name": "shelf", "status": "unavailable", "scan_complete": True,
            "usable_object_ids": ["shelf_a"],
            "binding_failures": [{"blocking": True, "error": "other candidate unresolved"}],
        }],
    }
    s.binding_usable = lambda obj_id: obj_id == "shelf_a"
    raw = copy.deepcopy(s.grounding_coverage)
    signature = s.evidence_profile.as_signature()
    prompt, _ = runner._build_prompt_ex(
        episode("object_rel_direction"), s, s,
        [parents[("object_rel_direction", "g1m0" if geometry else "g0m0")]], None,
        scope_override=scope_override)
    public = json.loads(prompt.split(
        "\nDirection candidate coverage and binding preconditions: ", 1)[1].split("\n", 1)[0])
    assert public["relative_direction_of_authorized"] is allowed
    assert public["requirements"][0]["usable_object_ids"] == ["shelf_a"]
    assert public["candidate_coverage_complete"] is False
    assert s.grounding_coverage == raw and s.evidence_profile.as_signature() == signature
    docs = prompt.split("## Authorized tools\n", 1)[1].split("\nSignatures, dependencies", 1)[0]
    assert ("- relative_direction_of(" in docs) is allowed
    assert "world_frame_diagnostic=degraded" in prompt
    assert "coverage gaps do not invalidate usable IDs" in prompt


def test_image_manifest_keeps_actual_noncontiguous_order_crops_and_omissions():
    crop = SimpleNamespace(
        kind="crop", image_id="crop-a", source_frame_id=7, box_xyxy=(1, 2, 10, 12),
        source_hw=(80, 100), sent_hw=(10, 9), scale=1.0)
    overview = SimpleNamespace(
        kind="scene_overview", image_id="overview-a", scene_id="synthetic", sent_hw=(64, 64),
        selected_azimuth_degrees=30, elevation_degrees=10, world_frame_status="unverified")
    records = {crop.image_id: crop, overview.image_id: overview}
    plan = SimpleNamespace(
        image_ids=["overview-a", "crop-a", "orig-0", "orig-7", "orig-31"],
        derived_image_ids=["overview-a", "crop-a"],
        original_image_ids=["orig-0", "orig-7", "orig-31"], omitted_originals=[1, 2])
    text = runner._image_manifest_text(SimpleNamespace(get=records.__getitem__), plan)
    assert text.index("overview-a") < text.index("crop-a") < text.index("[0,7,31]")
    assert "3..5. [Original frames]" in text
    assert "source_frame=7" in text and "crop_box(x0,y0,x1,y1)=(1, 2, 10, 12)" in text
    assert "[1, 2]" in text and "absent from this request" in text
    assert "no persistent visual memory" in prompt_builder.compact_execution_contract(8000)
    assert "Grants no metric" in text


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("finalizing", (False, True))
def test_actual_request_hashes_match_delivered_docs_and_complete_method(task, finalizing, parents):
    class OfflineClient:
        last_usage = {}
        last_generation = {}

        def set_request_context(self, value):
            self.context = value

        def chat(self, messages, **kwargs):
            self.messages = messages
            return (finalization_example(task) if finalizing else
                    "```python\n" + visual_example(task) + "```")

    s, ep = scene(task, "g1m1"), episode(task)
    client = OfflineClient()
    skill = parents[(task, "g1m1")]
    cfg = runner.OnlineRunConfig(mode="real", online_model_enabled=False)
    result = runner._synthesize(
        ep, s, s, [skill], cfg, client, pixels=[np.zeros((8, 8, 3), dtype=np.uint8)],
        trigger="finalize" if finalizing else "initial")
    assert result.program is not None, result.note
    if finalizing:
        docs = ""
        assert client.context["delivered_method_hashes"] == []
        assert skill.skill_md not in result.prompt
        assert "## Authorized tools" not in result.prompt
        assert "```python" not in result.prompt and "ReadNextObservation" not in result.prompt
    else:
        docs = result.prompt.split("## Authorized tools\n", 1)[1].split(
            "\nSignatures, dependencies", 1)[0]
        assert docs == compact_tool_docs(REGISTRY, **context(task, "g1m1"))
        assert client.context["delivered_method_hashes"] == [skill_content_sha256(skill)]
    assert client.context["tool_directory_sha256"] == hashlib.sha256(docs.encode()).hexdigest()
    assert client.context["tool_docs_delivered"] is (not finalizing)
    assert client.context["prompt_compaction_version"] == COMPACTION_VERSION


def test_local_source_loads_compact_builder_and_shared_enforcement():
    report = source_report()
    for name in ("skill3d.synthesis.prompt_builder", "skill3d.synthesis.prompt_compaction"):
        assert Path(report["local_module_paths"][name]).is_relative_to(ROOT / "src")
    spec = importlib.util.find_spec("skill3d.synthesis.execution_contract")
    assert Path(spec.origin).is_relative_to(ROOT.parent / "src")
    assert report["model_requests"] == 0
