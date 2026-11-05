"""Fresh-request repair continuity, with synthetic scenes and no transport."""
from copy import deepcopy
import json

import pytest

from Ominirunner import worker
from Ominirunner.audit import replay_contexts
from Ominirunner.common import Config, digest, read
from Ominirunner.evidence import EvidenceError, feedback, validate
from Ominirunner.jev import Client, Policy, host_review
from Ominirunner.native import ProgramFormatError, parse_program, pending_program, program_feedback
from Ominirunner.protocol import messages
from Ominirunner.review import apply_updates
from Ominirunner.submission import capture
from Ominirunner.test_contracts import OfflineClient, fenced, offline_only, scene_data
from Ominirunner.test_jev import Budget, Opener, SAMPLE, client, measurement, ratio


def text_of(context):
    return "\n".join(p["text"] for p in context[0]["content"] if p["type"] == "text")


def line_json(text, header):
    return json.loads(text.split("\n" + header, 1)[1].split("\n", 1)[0])


class SequenceOpener(Opener):
    def __init__(self, decisions):
        super().__init__()
        self.decisions = iter(decisions)

    def open(self, request, timeout):
        self.decision = next(self.decisions)
        return super().open(request, timeout)


def run_episode(tmp_path, monkeypatch, scene_data, outputs, *,
                question="What is the height ratio of the chair and desk?",
                decisions=("yes", "yes")):
    item = {**scene_data[0], "question": question, "task_type": "float"}
    rgb, frame, cache = scene_data[1:4]
    monkeypatch.setattr(worker, "load", lambda *a, **k: (rgb, frame, cache))
    budget, opener = Budget(), SequenceOpener(decisions)
    reviewer = Client(Policy(), budget, loader=lambda p: "private-test-key", opener=opener)
    job = {"item": item, "arm": "skill", "run_identity": "offline-host-context",
           "output_directory": str(tmp_path), "stop_file": str(tmp_path / "STOP"),
           "measurement_reviewer": reviewer.policy.public(),
           "initial_receipt": messages(item, rgb, frame, cache, arm="skill", config=Config(),
                                       measurement_reviewer=reviewer.policy.public())[1]}
    result = worker.solve(job, OfflineClient(outputs), budget, jev_client=reviewer)
    for index, payload in replay_contexts(item, result, rgb, frame, cache):
        assert payload == read(tmp_path / f"{index:02d}.messages.json")
    assert result == read(tmp_path / "result.json")
    assert not (tmp_path / "STOP").exists()
    return result, item, opener


def test_rejected_id_can_be_replaced_without_becoming_calculation_evidence(
        tmp_path, monkeypatch, scene_data):
    def replace(facts):
        assert len(facts) == 1 and facts[0]["target"] == "chair"
        text = text_of(read(tmp_path / "01.messages.json"))
        rejected = line_json(text, "RESELECT ")[0]
        assert rejected["target"] == "desk" and rejected["points"] == [[10, 10], [30, 20]]
        assert "value" not in rejected and "unit" not in rejected
        receipt = read(tmp_path / "01.intent.json")["request"]
        assert rejected["result_id"] in receipt["reselect_measurement_ids"]
        assert rejected["result_id"] not in receipt["visible_measurement_ids"]
        return fenced(
            'r=ctx.tools.pair_point_distance(view_index=0,a_xy=[11,10],b_xy=[30,20],'
            f'target="desk",quantity="height",replaces={rejected["result_id"]!r})\n'
            'YieldObservations([r["result_id"]])')

    result, item, opener = run_episode(
        tmp_path, monkeypatch, scene_data, [SAMPLE, replace, ratio], decisions=("yes", "no", "yes"))
    assert result["status"] == "answered" and result["request_count"] == 3
    assert opener.calls == 3 and result["answer_evidence"]["verified_arithmetic"]
    final_request = result["rounds"][2]["request"]
    old = result["rounds"][0]["new_measurement_ids"][1]
    new = result["rounds"][1]["new_measurement_ids"][0]
    assert final_request["superseded_measurement_ids"] == {old: new}
    assert old not in final_request["visible_measurement_ids"]
    assert new in final_request["visible_measurement_ids"]
    changed = deepcopy(result)
    changed["rounds"][1]["request"]["reselect_measurement_ids"] = []
    with pytest.raises(ValueError, match="reconstructed_public_context_changed"):
        list(replay_contexts(item, changed, *scene_data[1:4]))


@pytest.mark.parametrize("old", ["unknown", "not-delivered"])
def test_reselection_permission_does_not_authorize_unknown_or_hidden_ids(scene_data, old):
    from Ominirunner.data import problem
    from Ominirunner.native import NativeKernel
    from Ominirunner.tools import build_registry
    registry, scene, _ = build_registry(scene_data[3], problem(scene_data[0]), Config(), Policy().public())
    scene.delivered_reselect_ids = {"R"}
    kernel = NativeKernel(registry, scene, frames=[scene_data[2]])
    cell = kernel.run_cell(
        f'def solve(ctx):\n    ctx.tools.point_depth(view_index=0,x=10,y=10,'
        f'target="chair",quantity="camera_z",replaces={old!r})')
    assert cell.error_code == "tool_contract" and cell.answer_untrusted


def test_final_json_repairs_the_delivered_formula_without_copying_rejected_answer(
        tmp_path, monkeypatch, scene_data):
    def incorrect(facts):
        correct = ratio(facts)
        return correct.replace("ReturnAnswer(", "ReturnAnswer(1+")

    def final(facts):
        text = text_of(read(tmp_path / "05.messages.json"))
        state = line_json(text, "REPAIR EVIDENCE (unverified; recheck against task and approved facts) ")
        error = line_json(text, "ERRORS ")[0]
        assert error["code"] == "answer_calculation_mismatch" and "do not round" in error["repair"]
        assert state["status"] == "unverified_model_formula" and "answer" not in state
        assert "def solve(ctx)" not in text and "ReturnAnswer" not in text and "Tools closed" in text
        evidence = state["evidence"]
        values = {r["result_id"]: r["value"] for r in facts}
        answer = values[evidence["bindings"]["a"]["result_id"]] / values[evidence["bindings"]["b"]["result_id"]]
        return json.dumps({"answer": answer, "evidence": evidence})

    result, item, _ = run_episode(
        tmp_path, monkeypatch, scene_data,
        [SAMPLE, *[fenced("YieldObservations([])")] * 3, incorrect, final])
    assert result["rounds"][4]["status"] == "answer_evidence_required"
    assert result["status"] == "answered" and result["request_count"] == 6
    assert result["answer_evidence"]["verified_arithmetic"]
    changed = deepcopy(result)
    changed["rounds"][4]["last_submission_state"]["evidence"]["expression"] = "b/a"
    with pytest.raises(ValueError, match="round_last_submission_changed"):
        list(replay_contexts(item, changed, *scene_data[1:4]))


def test_physical_scale_survives_final_repair_and_keeps_public_conversion(
        tmp_path, monkeypatch, scene_data):
    sample = SAMPLE.replace('target="chair"', 'target="red chair"')

    def incorrect(facts):
        a, b = facts
        evidence = {"kind": "calculation", "expression": "s*b/a", "bindings": {
            "a": {"result_id": a["result_id"], "quantity": "height", "operand_index": 1},
            "b": {"result_id": b["result_id"], "quantity": "height", "operand_index": 0}},
            "scale": {"name": "s", "value": 5280.0, "unit": "ft", "anchor_binding": "a"}}
        return fenced(f"ReturnAnswer(1,evidence={evidence!r})")

    def final(facts):
        text = text_of(read(tmp_path / "05.messages.json"))
        state = line_json(text, "REPAIR EVIDENCE (unverified; recheck against task and approved facts) ")
        evidence = state["evidence"]
        assert evidence["scale"]["value"] == 5280.0 and evidence["scale"]["anchor_binding"] == "a"
        a, b = facts
        return json.dumps({"answer": 5280.0 * b["value"] / a["value"], "evidence": evidence})

    result, _, _ = run_episode(
        tmp_path, monkeypatch, scene_data,
        [sample, *[fenced("YieldObservations([])")] * 3, incorrect, final],
        question="If the red chair is 1 mile tall, what is the height of the desk in feet?")
    assert result["status"] == "answered" and result["answer_evidence"]["verified_arithmetic"]
    assert result["answer_evidence"]["scale"]["value"] == 5280.0


def test_repair_state_never_resurrects_blocked_numbers_or_host_geometry(scene_data):
    item, rgb, frame, cache, _, scene = scene_data
    row = measurement(scene)
    row["data"]["depth"] = 923.456789
    reviewer, _, _ = client("no")
    reviews = {"A": host_review(row, reviewer.review(row, rgb), rgb, reviewer.policy)}
    evidence = {"kind": "calculation", "expression": "a+923.456789", "bindings": {
        "a": {"result_id": "A", "quantity": "camera_z", "geometry": {"value": 923.456789}}}}
    snapshot = capture(1, 923.456789, evidence, [])
    context, receipt = messages(
        item, rgb, frame, cache, arm="skill", config=Config(), observations=[row], reviews=reviews,
        measurement_reviewer=reviewer.policy.public(), last_submission=snapshot,
        errors=[{"code": "measurement_surface_required", "actual": 923.456789,
                 "repair": "Put observed_surfaces inside measurement_reviews"}])
    text = text_of(context)
    assert "923.456789" not in text and "observed_surfaces" not in text
    assert "expression_omitted" in text and receipt["visible_measurement_ids"] == []
    with pytest.raises(EvidenceError, match="calculation_measurement_not_usable"):
        validate(923.456789, {"kind": "calculation", "expression": "a",
                              "bindings": {"a": {"result_id": "A", "quantity": "camera_z"}}},
                 [row], reviews, [], measurement_reviewer=reviewer.policy.public())


@pytest.mark.parametrize("unit,value", [("bad", 1), ("ft", 923.456789), ("ft", None),
                                        (["ft"], 5280), ("ft", 10**400)])
def test_invalid_scale_state_uses_public_task_instead_of_raising_or_copying_value(scene_data, unit, value):
    from Ominirunner.jev_protocol import calculation_state
    from Ominirunner.answer_contract import specification
    task = specification(scene_data[0] | {
        "question": "If the red chair is 1 mile tall, what is the height of the desk in feet?"})
    evidence = {"kind": "calculation", "expression": "s*a", "bindings": {},
                "scale": {"name": "s", "value": value, "unit": unit, "anchor_binding": "a"}}
    state = calculation_state(capture(1, 1, evidence, []), task, [], [])
    assert "value" not in state["evidence"]["scale"]
    assert state["scale_value_omitted"] == "use_public_task_span_in_answer_unit"


def test_unexecuted_plan_is_delivered_as_proposals_and_can_be_repaired_once(
        tmp_path, monkeypatch, scene_data):
    def repair(facts):
        assert facts == []
        text = text_of(read(tmp_path / "01.messages.json"))
        state = line_json(text, "UNEXECUTED TOOL PLAN (proposals only; no calls completed) ")
        assert state["execution_authorized"] is False and "code" not in state
        assert len(state["literal_tool_proposals"]) == 2
        assert "FIRST: measurement_reviews" not in text
        return SAMPLE
    result, _, opener = run_episode(
        tmp_path, monkeypatch, scene_data, ["Here is the plan:\n" + SAMPLE, repair, ratio])
    assert result["status"] == "answered" and opener.calls == 2
    assert result["rounds"][0]["executed"] is False
    assert result["rounds"][0]["tool_counts_before"]["calls"] == 0
    assert result["tool_counts"]["calls"] == 2


def test_final_unexecuted_state_does_not_request_program_output(scene_data):
    invalid = "Here is the plan:\n" + SAMPLE
    with pytest.raises(ProgramFormatError) as caught:
        parse_program(invalid)
    context, _ = messages(*scene_data[:4], arm="skill", config=Config(), round_index=5,
                          measurement_reviewer=Policy().public(),
                          pending_program=pending_program(invalid, caught.value),
                          errors=program_feedback(caught.value))
    text = text_of(context)
    assert "def solve(ctx)" not in text and "closed python fence" not in text
    assert '"proposals_omitted":"tools_closed"' in text
    assert '"executed":false' in text


def test_count_repair_retains_rgb_entities_without_promoting_them_to_tool_facts(
        tmp_path, monkeypatch, scene_data):
    scope = {"include": "visible chairs", "exclude": "", "entities": [
        {"entity_id": key, "category": "chair", "candidate_ids": [], "relationship": "rgb_only",
         "basis": "distinct chair visible in RGB", "included": True} for key in ("left", "right")]}
    evidence = {"kind": "visual_estimate", "basis": "count distinct RGB chairs", "object_scope": scope}

    def final(facts):
        assert facts == []
        text = text_of(read(tmp_path / "05.messages.json"))
        state = line_json(text, "REPAIR EVIDENCE (unverified; recheck against task and approved facts) ")
        assert state["status"] == "unverified_rgb_declaration"
        assert state["evidence"]["object_scope"] == scope
        return json.dumps({"answer": 2, "evidence": state["evidence"]})

    result, _, opener = run_episode(
        tmp_path, monkeypatch, scene_data,
        [*[fenced("YieldObservations([])")] * 4, fenced(f"ReturnAnswer(3,evidence={evidence!r})"), final],
        question="How many chairs are in the image?", decisions=())
    assert result["rounds"][4]["answer_feedback_errors"][0]["code"] == "object_scope_count_mismatch"
    assert result["status"] == "answered" and result["raw_answer"] == 2 and opener.calls == 0


def test_volume_repair_preserves_shape_and_scope_assumptions(
        tmp_path, monkeypatch, scene_data):
    assumptions = {"object_scope": "one chair", "volume_model": "bounding cuboid estimated from RGB"}
    evidence = {"kind": "visual_estimate", "basis": "x" * 201, "assumptions": assumptions}

    def final(facts):
        text = text_of(read(tmp_path / "05.messages.json"))
        state = line_json(text, "REPAIR EVIDENCE (unverified; recheck against task and approved facts) ")
        assert state["evidence"]["assumptions"] == assumptions
        state["evidence"]["basis"] = "RGB cuboid estimate"
        return json.dumps({"answer": .5, "evidence": state["evidence"]})

    result, _, opener = run_episode(
        tmp_path, monkeypatch, scene_data,
        [*[fenced("YieldObservations([])")] * 4, fenced(f"ReturnAnswer(.5,evidence={evidence!r})"), final],
        question="What is the volume of the chair in cubic meters?", decisions=())
    assert result["rounds"][4]["answer_feedback_errors"][0]["code"] == "visual_basis_required_max200"
    assert result["status"] == "answered" and opener.calls == 0
    assert result["answer_evidence"]["assumptions"] == assumptions


def test_candidate_context_matches_the_eight_ids_count_validation_can_use(scene_data):
    from Ominirunner.instances import ScopeError, validate_scope
    candidates = [{"candidate_id": f"C{n}", "category": "chair", "box_xyxy": [0, 0, 20, 20],
                   "view_index": 0} for n in range(10)]
    row = {"result_id": "candidates", "source_tool": "view_candidates", "status": "ok",
           "data": {"candidates": candidates, "count": 10}}
    context, receipt = messages(*scene_data[:4], arm="skill", config=Config(), observations=[row],
                               measurement_reviewer=Policy().public())
    facts = line_json(text_of(context), "TOOL FACTS ")
    assert [c["candidate_id"] for c in facts[0]["data"]["candidates"]] == [f"C{n}" for n in range(8)]
    assert facts[0]["data"]["omitted_candidates"] == 2 and facts[0]["data"]["exhaustive"] is False
    assert receipt["visible_result_ids"] == ["candidates"]
    scope = {"include": "chairs", "exclude": "", "entities": [
        {"entity_id": "one", "category": "chair", "candidate_ids": ["C9"],
         "relationship": "single", "basis": "RGB chair", "included": True}]}
    with pytest.raises(ScopeError, match="object_scope_identity_invalid"):
        validate_scope(scope, [row], ["candidates"], 1)
