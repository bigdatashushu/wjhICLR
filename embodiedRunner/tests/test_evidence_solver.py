"""End-to-end offline current-RGB preparation, delivery and one-call solvers."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from action_contract import EpisodeBinding, plain, public_hash
from action_run import FakeClient, FakeDepth, fake_run
from action_solver import ActionSolver
from budget import RunBudget
from evidence import CombinedDINO, CurrentEvidencePreparer, PerceptionBackends, public_categories
from methods import load_methods
from protocol import ContractError, OperationalError, sha256
from test_action_contract import fixture


def evidence_setup(backends=None):
    _, snapshot, _, _ = fixture()
    binding = EpisodeBinding("run", "episode", "base:0", "eb_navigation", "base", sha256(b"input"))
    budget = RunBudget("run", ["eb_navigation"]).episode("episode", "eb_navigation")
    preparer = CurrentEvidencePreparer(backends or PerceptionBackends(depth_estimator=FakeDepth()))
    return snapshot, binding, budget, preparer


def finalized(snapshot, prepared):
    return replace(snapshot, evidence_revision=1, evidence_manifest_sha256=prepared.manifest_sha256,
                   capabilities=prepared.capabilities)


def test_public_categories_priority_and_no_method_dependency():
    assert public_categories("Put apple on coffee table.", ("FindRefrigerator",)) == (
        "apple", "coffee table", "refrigerator")
    assert public_categories("Find chairs", ()) == ("chair",)
    assert public_categories("foo unknown", ()) == ()


def test_prepare_uses_rgb_bytes_immutable_cache_and_scope():
    snapshot, binding, budget, preparer = evidence_setup()
    result = preparer(snapshot, binding, budget)
    current = finalized(snapshot, result)
    manifest = preparer.validate(current, result)
    assert manifest["reference_answer_visible"] is False
    assert manifest["scope"]["pixel_sha256"] == snapshot.observation.pixel_sha256
    assert result.prepared.problem["image_paths"] == ("current-rgb",) or \
        result.prepared.problem["image_paths"] == ["current-rgb"]
    assert result.prepared.frames[0].shape == (500, 500, 3)
    assert result.prepared.frames[0].tobytes() == b"\xff" * (500 * 500 * 3)
    with pytest.raises(ValueError):
        result.prepared.depths[0].setflags(write=True)
    assert preparer(snapshot, binding, budget) is result
    assert budget.counts["moge"] == 1
    with pytest.raises(ContractError):
        preparer.validate(replace(current, decision_id="stale"), result)


def test_same_rgb_next_observation_is_new_preparation_and_revokes_cache():
    snapshot, binding, budget, preparer = evidence_setup()
    first = preparer(snapshot, binding, budget)
    obs = replace(snapshot.observation, observation_revision=1, mutation_revision=2,
                  environment_step=1, view_id="episode:observation:1:view:0")
    second = preparer(replace(snapshot, observation=obs, decision_id="second"), binding, budget)
    assert first.manifest_sha256 != second.manifest_sha256 and budget.counts["moge"] == 2
    assert len(preparer.cache) == 1


def test_backend_failure_stops_without_later_calls_and_sanitizes_provider():
    class BrokenDepth(FakeDepth):
        def __call__(self, rgb):
            raise RuntimeError("private-token-url")
    class NeverDetector:
        identity = {"backend": "never"}
        def __call__(self, *args):
            pytest.fail("must stop at the failed service")
    snapshot, binding, budget, preparer = evidence_setup(
        PerceptionBackends(depth_estimator=BrokenDepth(), detector=NeverDetector()))
    with pytest.raises(OperationalError, match="perception_service_error") as error:
        preparer(snapshot, binding, budget)
    assert "private-token" not in repr(error)
    assert budget.counts["moge"] == 1 and not budget.counts["dino"]
    assert budget.receipts[0]["status"] == "error"
    assert not preparer.cache and "private-token" not in repr(budget.summary())


def test_combined_dino_one_actual_request_and_no_hidden_retries():
    calls = []
    def detect(rgb, prompt, **kwargs):
        calls.append((prompt, kwargs))
        return [SimpleNamespace(label="chair", bbox_xyxy=(1, 2, 40, 50), confidence=.9),
                SimpleNamespace(label="cup", bbox_xyxy=(2, 3, 40, 50), confidence=.8)]
    dino = CombinedDINO(identity={"checkpoint_sha256": sha256(b"fake")},
                        endpoint="private-endpoint", detect_fn=detect)
    result = dino(np.zeros((500, 500, 3), dtype=np.uint8), ("chair", "cup"))
    assert len(result) == 2 and len(calls) == 1
    assert calls[0][1]["max_attempts"] == 1 and calls[0][1]["total_timeout_s"] == 60
    assert "private-endpoint" not in repr(dino.identity)


def test_detection_sam_once_selects_one_candidate_without_surface_permission():
    calls = []
    class Detector:
        identity = {"backend": "fake-detector"}
        def __call__(self, rgb, categories):
            calls.append("dino")
            return [{"category": "chair", "box_xyxy": [1, 2, 40, 50], "confidence": .7},
                    {"category": "chair", "box_xyxy": [1, 2, 40, 50], "confidence": .9}]
    class SAM:
        identity = {"backend": "fake-sam"}
        def __call__(self, rgb, boxes):
            calls.append("sam")
            assert len(boxes) == 1
            return [np.ones((500, 500), dtype=bool)]
    snapshot, binding, budget, preparer = evidence_setup(
        PerceptionBackends(Detector(), SAM(), depth_estimator=FakeDepth()))
    result = preparer(snapshot, binding, budget)
    preparer.validate(finalized(snapshot, result), result)
    assert calls == ["dino", "sam"] and len(result.prepared.candidates) == 1
    assert result.prepared.candidates[0]["confidence"] == .9
    assert result.prepared.candidates[0]["geometry_usable"] is False
    assert result.prepared.candidates[0]["mask_quality_assessed"] is False
    assert budget.counts["dino"] == budget.counts["sam"] == 1


def test_native_review_requires_already_verified_health_without_calls():
    class Reviewer:
        identity = {"backend": "fake-native"}
        expected_identity_sha256 = sha256(b"pin")
        client = SimpleNamespace(_identity=None)
    with pytest.raises(ContractError, match="review_health_preflight_required"):
        CurrentEvidencePreparer(PerceptionBackends(reviewer=Reviewer()))


@pytest.mark.parametrize("environment", ("eb_navigation", "eb_alfred", "eb_habitat"))
def test_complete_method_frozen_on_normal_and_repair_requests(environment):
    methods = load_methods()
    method = methods[environment]
    _, snapshot, _, _ = fixture()
    snapshot = replace(snapshot, method_sha256=method.content_sha256,
                       observation=replace(snapshot.observation, environment=environment))
    client = FakeClient("B11")
    solver = ActionSolver(client, arm="B11", environment=environment, method=method)
    for feedback in ({}, {"code": "violation_syntax"}):
        messages, delivery, chars = solver.messages(snapshot, feedback)
        assert method.skill_md in messages[0]["content"]
        assert delivery.entries[0].text == method.skill_md and chars < 96000
        slot = {}
        proposal = solver.generate(snapshot, feedback=feedback, usage=slot)
        assert proposal.source and slot["usage"]["total_tokens"] == 16
        assert solver.receipts[-1]["delivery"]["delivered_skill_versions"] == [method.skill_id + "@1.0.0"]
        assert solver.receipts[-1]["delivery"]["entries"][0]["content_sha256"] == method.content_sha256
    assert client.calls == 2
    assert "program_source" not in repr(solver.receipts) and "data:image" not in repr(solver.receipts)


@pytest.mark.parametrize("text", (
    '{"program_source":"pass","extra":1}',
    '{"program_source":"pass","program_source":"pass"}',
    '```python\npass\n```', '{"program_source":null}',
    '{"program_source":"```python\\npass\\n```"}',
))
def test_strict_sole_program_source_no_extraction_fallback(text):
    _, snapshot, _, _ = fixture()
    class Client(FakeClient):
        def chat(self, *args, **kwargs):
            return text
    solver = ActionSolver(Client("B01"), arm="B01", environment="eb_navigation")
    with pytest.raises(ContractError, match="model_output_format_error"):
        solver.generate(snapshot, feedback={}, usage={})
    assert solver.receipts[0]["status"] == "format_error"


def test_b01_has_same_tool_contract_but_no_skill():
    _, snapshot, _, _ = fixture()
    solver = ActionSolver(FakeClient("B01"), arm="B01", environment="eb_navigation")
    messages, delivery, _ = solver.messages(snapshot, {})
    assert not delivery.entries and "Complete frozen environment Skill" not in messages[0]["content"]
    assert "point_depth" in messages[0]["content"] and "SubmitActions" in messages[0]["content"]


def test_method_body_tamper_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1] / "methods"
    for source in root.iterdir():
        (tmp_path / source.name).write_bytes(source.read_bytes())
    (tmp_path / "alfred.md").write_text("tampered\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_methods(tmp_path)


def test_all_four_environments_all_arms_use_real_prompts_and_sandbox():
    result = fake_run(run_id="p3-offline")
    assert result["status"] == "passed", [(r["arm"], r["environment"], r["terminal_reason"])
                                         for r in result["attempts"]]
    assert len(result["attempts"]) == 12 and result["live_requests"] == 0
    for report in result["attempts"]:
        assert report["environment_steps"] == 2
        assert len(report["model_requests"]) == 2
        assert all(row["reference_answer_visible"] is False for row in report["model_requests"])
        if report["arm"] == "Direct":
            assert not report["executions"]
            assert not report["budget"]["reserved"].get("moge")
            assert not report["decisions"][0]["evidence_manifest_sha256"]
        else:
            assert report["budget"]["reserved"]["moge"] == 2
            assert report["budget"]["reserved"]["tools"] == 2
            assert all(row["receipt"]["isolation_sealed"] for row in report["executions"])
        if report["arm"] == "B11":
            assert all(row["delivery"]["delivered_skill_versions"] for row in report["model_requests"])
