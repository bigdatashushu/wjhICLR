"""Binary admission, compact context and worker replay without network or labels."""
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
import json
import urllib.error

import pytest
from PIL import Image

from Ominirunner import worker
from Ominirunner.audit import replay_contexts
from Ominirunner.common import Config, digest, read
from Ominirunner.evidence import EvidenceError, validate
from Ominirunner.jev import (Client, Policy, host_review, request_payload, seal)
from Ominirunner.protocol import messages
from Ominirunner.test_contracts import (OfflineClient, dimension, fenced, offline_only,
                                       point, scene_data, task)


class Budget:
    def __init__(self):
        self.categories = []
        self.charges = []

    @contextmanager
    def request(self, category="solver"):
        charge = {"usage": {}}
        self.categories.append(category)
        self.charges.append(charge)
        yield charge


class Opener:
    def __init__(self, decision="yes", model="jev-1.14.0", fail=False):
        self.calls, self.decision, self.model, self.fail = 0, decision, model, fail

    def open(self, request, timeout):
        self.calls += 1
        self.payload = json.loads(request.data)
        assert "assets" in self.payload and set(self.payload["questions"]) == {"usable"}
        assert request.headers["Typesafe-danger-zone-images"] == "true"
        if self.fail:
            raise urllib.error.HTTPError(request.full_url, 400, "SENSITIVE_BODY", {}, None)
        response = BytesIO(json.dumps({
            "model": self.model, "answers": {"usable": {"type": "choice", "choice": self.decision}},
            "usage": {"input_tokens": 12, "output_tokens": 1}}).encode())
        response.status = 200
        return response


def client(decision="yes", *, fail=False, model="jev-1.14.0", max_requests=8):
    budget = Budget()
    opener = Opener(decision, model, fail)
    reviewer = Client(Policy(max_requests=max_requests), budget, loader=lambda p: "private-test-key",
                      opener=opener)
    return reviewer, opener, budget


def measurement(scene, key="A", quantity="camera_z", target="chair"):
    row = point(scene, key, target) if quantity == "camera_z" else dimension(scene, key, target, quantity)[0]
    row["data"]["jev_quantity"] = quantity
    if quantity == "camera_z":
        row["data"]["intended_targets"][0]["surface"] = "visible_surface"
    return row


@pytest.mark.parametrize("decision,expected", [("yes", "usable"), ("no", "resample")])
def test_binary_gate_and_exact_image_identity(scene_data, decision, expected):
    row = measurement(scene_data[-1])
    reviewer, opener, budget = client(decision)
    certificate = reviewer.review(row, scene_data[1])
    update = host_review(row, certificate, scene_data[1], reviewer.policy)
    assert update["decision"] == expected and certificate["decision"] == decision
    assert budget.categories == ["gate"] and budget.charges[0]["usage"] == {
        "input_tokens": 12, "output_tokens": 1}
    state = opener.payload["state"]
    assert state["proposal"]["points"] == [[10, 10]]
    assert state["marked_points"] == [[10.0, 10.0]]
    import base64
    raw = base64.b64decode(next(iter(opener.payload["assets"].values()))["data_base64"])
    with Image.open(BytesIO(raw)) as image:
        assert image.size == (280, 280)
        assert image.getpixel((10, 10)) == (0, 0, 0)  # hollow marker
    changed = deepcopy(row)
    changed["data"]["pixel_xy"] = [11, 10]
    with pytest.raises(ValueError, match="jev_receipt_changed"):
        host_review(changed, certificate, scene_data[1], reviewer.policy)


@pytest.mark.parametrize("decision,model,fail", [
    ("yes", "jev-1.13.0", False), ("uncertain", "jev-1.14.0", False),
    ("yes", "jev-1.14.0", True)])
def test_service_deviation_never_approves_or_retries(scene_data, decision, model, fail):
    reviewer, opener, _ = client(decision, model=model, fail=fail)
    row = measurement(scene_data[-1])
    receipt = reviewer.review(row, scene_data[1])
    assert receipt["status"] == "service_error" and receipt["decision"] is None
    assert opener.calls == 1 and "SENSITIVE_BODY" not in json.dumps(receipt)
    assert "private-test-key" not in json.dumps(receipt)
    assert host_review(row, receipt, scene_data[1], reviewer.policy)["decision"] == "resample"


def test_cap_no_extra_request_and_receipt_tampering(scene_data):
    row = measurement(scene_data[-1])
    reviewer, opener, _ = client(max_requests=1)
    receipt = reviewer.review(row, scene_data[1])
    rejected = reviewer.review(row, scene_data[1])
    assert rejected["error_code"] == "jev_request_cap" and not rejected["request_attempted"]
    assert opener.calls == 1
    changed = {**receipt, "decision": "no"}
    with pytest.raises(ValueError, match="jev_receipt_changed"):
        host_review(row, changed, scene_data[1], reviewer.policy)


def test_binding_cannot_change_quantity_pixels_or_override_host_geometry(scene_data):
    row = measurement(scene_data[-1], quantity="height")
    reviewer, _, _ = client()
    receipt = reviewer.review(row, scene_data[1])
    reviews = {"A": host_review(row, receipt, scene_data[1], reviewer.policy)}
    evidence = {"kind": "calculation", "expression": "h", "bindings": {
        "h": {"result_id": "A", "quantity": "height"}}}
    checked = validate(row["data"]["distance"], evidence, [row], reviews, ["A"],
                       measurement_reviewer=reviewer.policy.public())
    assert checked["verified_arithmetic"] is True
    evidence["bindings"]["h"]["quantity"] = "width"
    with pytest.raises(EvidenceError, match="jev_review_quantity_mismatch"):
        validate(1, evidence, [row], reviews, ["A"], measurement_reviewer=reviewer.policy.public())
    evidence["bindings"]["h"].update(quantity="height", geometry={})
    with pytest.raises(EvidenceError, match="host_review_only"):
        validate(1, evidence, [row], reviews, ["A"], measurement_reviewer=reviewer.policy.public())
    row = measurement(scene_data[-1])
    reviews = {"A": host_review(row, reviewer.review(row, scene_data[1]), scene_data[1], reviewer.policy)}
    evidence["bindings"]["h"] = {"result_id": "A", "quantity": "camera_z", "sample_xy": [11, 10]}
    with pytest.raises(EvidenceError, match="measurement_surface_required"):
        validate(1, evidence, [row], reviews, ["A"], measurement_reviewer=reviewer.policy.public())


def test_compact_context_hides_rejected_values_and_review_images(scene_data):
    item, rgb, frame, cache, _, scene = scene_data
    row = measurement(scene)
    row["data"]["depth"] = 923.456789
    reviewer, _, _ = client("no")
    reviews = {"A": host_review(row, reviewer.review(row, rgb), rgb, reviewer.policy)}
    compact, receipt = messages(item, rgb, frame, cache, arm="skill", config=Config(),
                                measurement_reviewer=reviewer.policy.public(),
                                observations=[row], reviews=reviews, note="923.456789",
                                last_submission={"raw_answer": 923.456789})
    old = messages(item, rgb, frame, cache, arm="skill", config=Config(),
                   observations=[row], reviews=reviews)[1]
    text = "\n".join(p.get("text", "") for p in compact[0]["content"])
    assert "923.456789" not in text and "observed_surfaces" not in text
    assert "reselect points" in text and receipt["visible_measurement_ids"] == []
    assert receipt["image_count"] == 2 and receipt["text_chars"] < old["text_chars"] * .5


def episode(tmp_path, monkeypatch, scene_data, outputs, *, decision="yes", fail=False):
    item = {**scene_data[0], "question": "What is the height ratio of the chair and desk?",
            "task_type": "float"}
    rgb, frame, cache = scene_data[1:4]
    monkeypatch.setattr(worker, "load", lambda *a, **k: (rgb, frame, cache))
    reviewer, opener, budget = client(decision, fail=fail)
    solver = OfflineClient(outputs)
    job = {"item": item, "arm": "skill", "run_identity": "offline-jev",
           "output_directory": str(tmp_path), "stop_file": str(tmp_path / "STOP"),
           "measurement_reviewer": reviewer.policy.public(),
           "initial_receipt": messages(item, rgb, frame, cache, arm="skill", config=Config(),
                                       measurement_reviewer=reviewer.policy.public())[1]}
    result = worker.solve(job, solver, budget, jev_client=reviewer)
    stored = read(tmp_path / "result.json")
    assert stored == result
    assert digest({k: v for k, v in stored.items() if k != "receipt_sha256"}) == stored["receipt_sha256"]
    for index, payload in replay_contexts(item, stored, rgb, frame, cache):
        assert payload == read(tmp_path / f"{index:02d}.messages.json")
    return result, item, opener, budget


SAMPLE = fenced(
    'a=ctx.tools.pair_point_distance(view_index=0,a_xy=[10,10],b_xy=[20,20],'
    'target="chair",quantity="height")\n'
    'b=ctx.tools.pair_point_distance(view_index=0,a_xy=[10,10],b_xy=[30,20],'
    'target="desk",quantity="height")\nYieldObservations([a["result_id"],b["result_id"]])')


def ratio(facts):
    a, b = facts[:2]
    evidence = {"kind": "calculation", "expression": "a/b", "bindings": {
        "a": {"result_id": a["result_id"], "quantity": "height"},
        "b": {"result_id": b["result_id"], "quantity": "height"}}}
    return fenced(f'ReturnAnswer({a["value"]!r}/{b["value"]!r},evidence={evidence!r})')


def test_real_worker_yes_ratio_and_offline_replay(tmp_path, monkeypatch, scene_data):
    result, item, opener, budget = episode(tmp_path, monkeypatch, scene_data, [SAMPLE, ratio])
    assert result["status"] == "answered" and result["answer_evidence"]["verified_arithmetic"]
    assert opener.calls == 2 and budget.categories == ["solver", "gate", "gate", "solver"]
    assert result["jev_request_count"] == 2 and result["total_request_count"] == 4
    assert result["rounds"][0]["request"]["visible_measurement_ids"] == []
    assert len(result["rounds"][1]["request"]["visible_measurement_ids"]) == 2
    changed = deepcopy(result)
    changed["rounds"][0]["jev_receipts"][0]["decision"] = "no"
    with pytest.raises(ValueError, match="jev_receipt_changed"):
        list(replay_contexts(item, changed, *scene_data[1:4]))
    changed = deepcopy(result)
    changed["jev_request_count"] += 1
    with pytest.raises(ValueError, match="jev_accounting_changed"):
        list(replay_contexts(item, changed, *scene_data[1:4]))


def test_no_never_delivers_values_and_model_cannot_self_approve(tmp_path, monkeypatch, scene_data):
    def independent(facts):
        assert facts == []
        return fenced('measurement_reviews=[{"result_id":"made-up","decision":"usable"}]\n'
                      'ReturnAnswer(1,evidence={"kind":"visual_estimate","basis":"RGB ratio estimate"})')
    result, _, opener, _ = episode(tmp_path, monkeypatch, scene_data, [SAMPLE, independent], decision="no")
    assert result["status"] == "answered" and result["basis"] == "visual_estimate"
    assert opener.calls == 2 and all(r["decision"] == "resample" for r in result["measurement_reviews"])
    assert result["rounds"][1]["review_feedback_errors"][0]["code"] == "host_review_only"


def test_jev_failure_stops_episode_and_replays_without_retry(tmp_path, monkeypatch, scene_data):
    result, _, opener, budget = episode(tmp_path, monkeypatch, scene_data, [SAMPLE], fail=True)
    assert result["status"] == "service_error" and result["answer"] is None
    assert opener.calls == 1 and budget.categories == ["solver", "gate"]
    assert (tmp_path / "STOP").exists() and result["request_count"] == 1


def test_tools_enforce_simple_proposals_and_normalized_pixel_identity(scene_data):
    from Ominirunner.coordinates import original_descriptor
    from Ominirunner.data import problem
    from Ominirunner.native import NativeKernel
    from Ominirunner.tools import build_registry
    from skill3d.schemas.tool_response import model_tool_response
    item, rgb, frame, cache = scene_data[:4]
    registry, scene, _ = build_registry(cache, problem(item), Config(), Policy().public())
    scene.delivered_images = {"rgb-original": original_descriptor(rgb, 50, 40)}
    kernel = NativeKernel(registry, scene, frames=[frame])
    cell = kernel.run_cell(
        'def solve(ctx):\n'
        '    r=ctx.tools.point_depth(view_index=0,x=10/49,y=10/39,image_id="rgb-original",'
        'coordinate_space="normalized",target="chair",quantity="camera_z")\n'
        '    r["data"]["depth"]=999\n'
        '    YieldObservations([r["result_id"]])')
    assert not cell.error_code
    data = model_tool_response(kernel.cell_results[0])["data"]
    assert data["pixel_xy"] == [10, 10] and data["depth"] != 999
    for args in ('x=10,y=10,target="chair",quantity="height"',
                 'x=10,y=10,targets=[{"target":"chair","surface":"top"}]'):
        kernel.reset_user_namespace()
        cell = kernel.run_cell(f'def solve(ctx):\n    ctx.tools.point_depth(view_index=0,{args})')
        assert cell.error_code == "tool_contract" and cell.answer_untrusted


def test_same_cell_approval_still_requires_delivery_and_final_is_json(
        tmp_path, monkeypatch, scene_data):
    def premature(facts):
        assert facts == []
        return SAMPLE.replace('YieldObservations([a["result_id"],b["result_id"]])',
                              'ReturnAnswer(a["data"]["distance"]/b["data"]["distance"],'
                              'evidence={"kind":"calculation","expression":"a/b","bindings":{'
                              '"a":{"result_id":a["result_id"],"quantity":"height"},'
                              '"b":{"result_id":b["result_id"],"quantity":"height"}}})')
    result, _, _, _ = episode(tmp_path, monkeypatch, scene_data, [premature, ratio])
    assert result["rounds"][0]["status"] == "answer_evidence_required"
    assert result["rounds"][0]["answer_feedback_errors"][0]["code"] == "calculation_measurement_not_delivered"
    assert result["status"] == "answered"
    item, rgb, frame, cache = scene_data[:4]
    context, _ = messages(item, rgb, frame, cache, arm="skill", config=Config(),
                          measurement_reviewer=Policy().public(), round_index=5)
    text = "\n".join(p.get("text", "") for p in context[0]["content"])
    assert "Tools closed. Output only" in text and "one strict ```python" not in text


def test_replacement_and_renamed_id_need_their_own_certificate(scene_data):
    from Ominirunner.review import apply_updates
    old = measurement(scene_data[-1])
    reviewer, _, _ = client()
    certificate = reviewer.review(old, scene_data[1])
    reviews = {"A": host_review(old, certificate, scene_data[1], reviewer.policy)}
    new = deepcopy(old)
    new["result_id"] = "B"
    new["data"]["replaces_result_id"] = "A"
    with pytest.raises(ValueError, match="jev_receipt_changed"):
        host_review(new, certificate, scene_data[1], reviewer.policy)
    apply_updates(reviews, {}, [old, new])
    assert reviews["A"]["decision"] == "resample"
    evidence = {"kind": "calculation", "expression": "z", "bindings": {
        "z": {"result_id": "B", "quantity": "camera_z"}}}
    with pytest.raises(EvidenceError, match="calculation_measurement_not_usable"):
        validate(1, evidence, [old, new], reviews, ["B"], measurement_reviewer=Policy().public())


def test_sixth_json_keeps_host_reviews_and_exact_ratio(tmp_path, monkeypatch, scene_data):
    def final(facts):
        a, b = facts[:2]
        return json.dumps({"answer": a["value"]/b["value"], "evidence": {
            "kind": "calculation", "expression": "a/b", "bindings": {
                "a": {"result_id": a["result_id"], "quantity": "height"},
                "b": {"result_id": b["result_id"], "quantity": "height"}}}})
    result, _, opener, _ = episode(
        tmp_path, monkeypatch, scene_data,
        [SAMPLE, *[fenced("YieldObservations([])")]*4, final])
    assert result["status"] == "answered" and result["request_count"] == 6
    assert opener.calls == 2 and result["total_request_count"] == 8
    assert result["rounds"][-1]["phase"] == "final"
    assert result["answer_evidence"]["verified_arithmetic"]


def test_binary_yes_cannot_waive_task_operands_units_or_arithmetic(scene_data):
    row = measurement(scene_data[-1], quantity="height")
    reviewer, _, _ = client()
    reviews = {"A": host_review(row, reviewer.review(row, scene_data[1]), scene_data[1], reviewer.policy)}
    evidence = {"kind": "calculation", "expression": "h", "bindings": {
        "h": {"result_id": "A", "quantity": "height"}}}
    with pytest.raises(EvidenceError, match="task_operand_mismatch"):
        validate(row["data"]["distance"], evidence, [row], reviews, ["A"],
                 task("What is the height of the desk?", "float"),
                 measurement_reviewer=Policy().public())
    with pytest.raises(EvidenceError, match="task_unit_mismatch"):
        validate(row["data"]["distance"], evidence, [row], reviews, ["A"],
                 task("What is the height of the chair in meters?", "float"),
                 measurement_reviewer=Policy().public())
    with pytest.raises(EvidenceError, match="answer_calculation_mismatch"):
        validate(999, evidence, [row], reviews, ["A"], measurement_reviewer=Policy().public())


class InternOpener:
    def __init__(self, decision="yes", *, image_count=3, p_yes=None):
        from skill3d.infra.m5_identity import content_sha256
        self.decision, self.image_count, self.calls = decision, image_count, []
        self.p_yes = p_yes
        self.identity = {"model": "internlm/Intern-Decision-4B", "backend": "native-hf-model-package",
                         "dtype": "bfloat16", "model_revision": "frozen-test-revision",
                         "files": {"inference.py": "pinned"}, "implementation_sha256": "pinned",
                         "calibration": {"method": "temperature-scaling", "temperature": 1.99}}
        self.identity_sha = content_sha256(self.identity)

    def open(self, request, timeout):
        self.calls.append(request.full_url)
        if request.full_url.endswith("/health"):
            value = {"identity": self.identity, "identity_sha256": self.identity_sha}
        else:
            from skill3d.infra.m5_identity import content_sha256
            payload = json.loads(request.data)
            query = payload["request"]
            assert len(query["images"]) == 3 and set(query["questions"]) == {"usable"}
            assert query["questions"]["usable"]["type"] == "choice"
            assert content_sha256(query) == payload["request_sha256"]
            p = self.p_yes if self.p_yes is not None else (.9 if self.decision == "yes" else .1)
            answer = {"type": "choice", "choice": self.decision, "decision": self.decision,
                      "probabilities": {"yes": p, "no": 1-p}, "confidence": max(p, 1-p)}
            value = {"identity_sha256": self.identity_sha,
                     "request_sha256": payload["request_sha256"],
                     "result": {"model": "Intern-Decision-4B", "backend": "hf",
                                "calibration": self.identity["calibration"], "answers": {"usable": answer}},
                     "vision_evidence": {"image_count": self.image_count, "image_token_count": 96,
                                         "input_tokens": 150, "pixel_values_shape": [96, 10],
                                         "image_grid_thw": [[1, 4, 4]]*3,
                                         "decision_logit_positions": [149],
                                         "legal_symbol_token_ids": {"usable": [20, 21]}}}
        response = BytesIO(json.dumps(value).encode())
        response.status = 200
        return response


@pytest.mark.parametrize("decision", ["yes", "no"])
def test_intern_native_binary_images_and_receipt_replay(scene_data, decision):
    opener, budget = InternOpener(decision), Budget()
    policy = Policy(provider="intern", service_identity_sha256=opener.identity_sha)
    reviewer = Client(policy, budget, loader=lambda p: pytest.fail("Intern must not load an API key"),
                      opener=opener)
    row = measurement(scene_data[-1], quantity="height")
    receipt = reviewer.review(row, scene_data[1])
    assert receipt["status"] == "reviewed" and receipt["decision"] == decision
    assert receipt["actual_model"] == "Intern-Decision-4B"
    assert budget.categories == ["gate"] and len(opener.calls) == 2
    assert budget.charges[0]["usage"] == {"input_tokens": 150}
    host_review(row, receipt, scene_data[1], policy)
    forged = deepcopy(receipt)
    forged["vision_evidence"]["image_count"] = 1
    forged = seal({k: v for k, v in forged.items() if k != "receipt_sha256"})
    with pytest.raises(ValueError, match="intern_visual_evidence_invalid"):
        host_review(row, forged, scene_data[1], policy)


def test_intern_incomplete_image_encoding_and_changed_identity_fail_closed(scene_data):
    opener, budget = InternOpener(image_count=1), Budget()
    policy = Policy(provider="intern", service_identity_sha256=opener.identity_sha)
    row = measurement(scene_data[-1])
    receipt = Client(policy, budget, opener=opener).review(row, scene_data[1])
    assert receipt["status"] == "service_error" and receipt["decision"] is None
    assert receipt["error_code"] == "intern_visual_evidence_invalid"
    wrong = Policy(provider="intern", service_identity_sha256="0"*64)
    before = len(opener.calls)
    receipt = Client(wrong, Budget(), opener=opener).review(row, scene_data[1])
    assert not receipt["request_attempted"] and len(opener.calls) == before+1


@pytest.mark.parametrize("p_yes,threshold,decision", [
    (.5, .5, "no"), (.6, .5, "yes"), (.6, .7, "no"),
    (.7, .7, "no"), (.71, .7, "yes"), (.9, .9, "no"), (1., .9, "yes")])
def test_intern_calibrated_threshold_and_offline_replay(scene_data, p_yes, threshold, decision):
    opener = InternOpener(p_yes=p_yes)
    policy = Policy(provider="intern", service_identity_sha256=opener.identity_sha,
                    yes_threshold=threshold)
    assert Policy.parse(policy.public()) == policy
    row = measurement(scene_data[-1])
    receipt = Client(policy, Budget(), opener=opener).review(row, scene_data[1])
    assert receipt["status"] == "reviewed" and receipt["decision"] == decision
    assert receipt["binary_answer"]["choice"] == "yes"
    review = host_review(row, receipt, scene_data[1], policy)
    assert review["decision"] == ("usable" if decision == "yes" else "resample")
    forged = seal({**{k: v for k, v in receipt.items() if k != "receipt_sha256"},
                   "decision": "no" if decision == "yes" else "yes"})
    with pytest.raises(ValueError, match="intern_binary_answer_invalid"):
        host_review(row, forged, scene_data[1], policy)
    changed = Policy(provider="intern", service_identity_sha256=opener.identity_sha,
                     yes_threshold=.8 if threshold != .8 else .9)
    with pytest.raises(ValueError, match="jev_receipt_changed"):
        host_review(row, receipt, scene_data[1], changed)


@pytest.mark.parametrize("threshold", [True, ".7", .49, 1.01, float("nan"), float("inf")])
def test_intern_threshold_configuration_rejects_invalid_values(threshold):
    with pytest.raises(ValueError, match="intern_yes_threshold_invalid"):
        Policy(provider="intern", yes_threshold=threshold)


def test_existing_frozen_policies_keep_hashes_and_default_half():
    for provider in ("intern", "makers", "typesafe"):
        policy = Policy(provider=provider)
        assert "yes_threshold" not in policy.public()
        assert Policy.parse(policy.public()) == policy
        assert policy.effective_yes_threshold == .5
    with pytest.raises(ValueError, match="jev_policy_invalid"):
        Policy(provider="makers", yes_threshold=.7)


def test_full_worker_can_use_intern_without_changing_solver_context(tmp_path, monkeypatch, scene_data):
    opener = InternOpener()
    policy = Policy(provider="intern", service_identity_sha256=opener.identity_sha)
    budget = Budget()
    reviewer = Client(policy, budget, opener=opener)
    item = {**scene_data[0], "question": "What is the height ratio of the chair and desk?",
            "task_type": "float"}
    rgb, frame, cache = scene_data[1:4]
    monkeypatch.setattr(worker, "load", lambda *a, **k: (rgb, frame, cache))
    job = {"item": item, "arm": "skill", "run_identity": "offline-intern",
           "output_directory": str(tmp_path), "stop_file": str(tmp_path / "STOP"),
           "measurement_reviewer": policy.public(),
           "initial_receipt": messages(item, rgb, frame, cache, arm="skill", config=Config(),
                                       measurement_reviewer=policy.public())[1]}
    result = worker.solve(job, OfflineClient([SAMPLE, ratio]), budget, jev_client=reviewer)
    assert result["status"] == "answered" and result["jev_request_count"] == 2
    assert budget.categories == ["solver", "gate", "gate", "solver"]
    assert result["measurement_reviewer"]["provider"] == "intern"
    for index, payload in replay_contexts(item, result, rgb, frame, cache):
        assert payload == read(tmp_path / f"{index:02d}.messages.json")


def test_full_worker_threshold_rejection_hides_values_and_probabilities(
        tmp_path, monkeypatch, scene_data):
    opener, budget = InternOpener(p_yes=.62), Budget()
    policy = Policy(provider="intern", service_identity_sha256=opener.identity_sha,
                    yes_threshold=.7)
    item = {**scene_data[0], "question": "What is the height ratio of the chair and desk?",
            "task_type": "float"}
    rgb, frame, cache = scene_data[1:4]
    monkeypatch.setattr(worker, "load", lambda *a, **k: (rgb, frame, cache))

    def independent(facts):
        assert facts == []
        return fenced('ReturnAnswer(1,evidence={"kind":"visual_estimate","basis":"RGB ratio estimate"})')

    job = {"item": item, "arm": "skill", "run_identity": "offline-intern-threshold",
           "output_directory": str(tmp_path), "stop_file": str(tmp_path / "STOP"),
           "measurement_reviewer": policy.public(),
           "initial_receipt": messages(item, rgb, frame, cache, arm="skill", config=Config(),
                                       measurement_reviewer=policy.public())[1]}
    result = worker.solve(job, OfflineClient([SAMPLE, independent]), budget,
                          jev_client=Client(policy, budget, opener=opener))
    assert result["status"] == "answered" and result["basis"] == "visual_estimate"
    assert result["rounds"][1]["request"]["visible_measurement_ids"] == []
    receipts = result["rounds"][0]["jev_receipts"]
    assert len(receipts) == 2 and all(r["decision"] == "no" for r in receipts)
    assert all(r["binary_answer"]["choice"] == "yes" for r in receipts)
    for index, payload in replay_contexts(item, result, rgb, frame, cache):
        assert payload == read(tmp_path / f"{index:02d}.messages.json")
        text = "\n".join(p.get("text", "") for p in payload[0]["content"])
        assert "probabilities" not in text and "yes_threshold" not in text


@pytest.mark.parametrize("quantity", ["camera_z", "chord", "height", "width", "length", "diameter"])
def test_intern_question_matches_only_the_requested_measurement(scene_data, quantity):
    from Ominirunner.intern_review import VERSION as PROMPT_VERSION
    from Ominirunner.intern_threshold import row as pilot_row
    from Ominirunner.jev import ROLES
    count = 1 if quantity == "camera_z" else 2
    target = ["chair", "desk"] if quantity == "chord" else "chair"
    measurement = pilot_row({"case_id": "quantity-routing", "target": target,
                             "quantity": quantity, "points": [[10, 10], [20, 20]][:count]})
    request, identity = request_payload(measurement, scene_data[1], Policy(provider="intern"))
    question = request["questions"]["usable"]
    assert identity["binary_prompt_version"] == PROMPT_VERSION
    assert set(question["criteria"]) == {"yes", "no"} and len(request["images"]) == 3
    if quantity in ROLES:
        a, b = ROLES[quantity]
        assert f"A={a}, B={b}" in question["instructions"]
        assert f"COMPLETE visible {quantity}" in question["instructions"]
    else:
        assert "COMPLETE" not in question["instructions"]
        assert "extrema" not in question["criteria"]["yes"]
        if quantity == "chord":
            assert "respective" in question["criteria"]["yes"]
        else:
            assert "visible part" in question["instructions"]


def test_host_skill_is_loaded_and_bound_to_context_receipt(scene_data):
    from Ominirunner.common import skill_text
    from Ominirunner.jev_protocol import CONTEXT_VERSION
    import hashlib
    item, rgb, frame, cache = scene_data[:4]
    skill = skill_text(host_review=True)
    assert 0 < len(skill) <= 1800
    context, receipt = messages(item, rgb, frame, cache, arm="skill", config=Config(),
                                measurement_reviewer=Policy(provider="intern", yes_threshold=.5).public())
    text = "\n".join(p.get("text", "") for p in context[0]["content"])
    assert "MEASUREMENT SKILL\n" + skill in text
    assert "Geometry remains your visual declaration" not in text
    assert receipt["skill_sha256"] == hashlib.sha256(skill.encode()).hexdigest()
    assert receipt["context_version"] == CONTEXT_VERSION
    assert receipt["measurement_reviewer"]["yes_threshold"] == .5
    final, final_receipt = messages(item, rgb, frame, cache, arm="skill", config=Config(),
                                   measurement_reviewer=Policy(provider="intern", yes_threshold=.5).public(),
                                   round_index=5)
    final_text = "\n".join(p.get("text", "") for p in final[0]["content"])
    assert "MEASUREMENT SKILL" not in final_text
    assert final_receipt["skill_sha256"] == receipt["skill_sha256"]


def test_intern_overview_marks_original_centers_without_covering_surface(scene_data):
    import base64
    from Ominirunner.intern_review import RENDERING_VERSION
    buffer = BytesIO()
    Image.new("RGB", (900, 600), (0, 0, 0)).save(buffer, format="JPEG")
    row = measurement(scene_data[-1], quantity="height")
    request, identity = request_payload(row, buffer.getvalue(), Policy(provider="intern"))
    meta = json.loads(request["state"].split("\n", 1)[1])
    expected = [[(x+.5)*768/900-.5, (y+.5)*512/600-.5] for x, y in ((10, 10), (20, 20))]
    assert meta["overview_marked_points"] == expected
    assert identity["review_render_version"] == RENDERING_VERSION
    assert len(meta["marked_crop_points"]) == 2
    assert "scene overview with hollow A/B markers" in request["state"]
    crop = Image.open(BytesIO(base64.b64decode(request["images"][1].split(",", 1)[1])))
    assert crop.getpixel((10, 10)) == (0, 0, 0)
    overview = Image.open(BytesIO(base64.b64decode(request["images"][0].split(",", 1)[1])))
    assert overview.size == (768, 512) and overview.getbbox() is not None
