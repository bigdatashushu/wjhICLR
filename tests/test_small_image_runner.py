"""Fake-only solver/CLI coverage for actual small-image and sandbox contracts."""

from __future__ import annotations

import base64
import copy
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from skill3d.small_image.datasets import SmallImageProblem, freeze_manifest
from skill3d.small_image.precompute import SmallImagePrecomputer, prepared_content_sha256
from skill3d.small_image.runner import (
    SmallImageRunnerConfig, SmallImageSolver, build_category_messages,
    parse_options, parse_small_image_answer, preflight_small_image,
    score_small_image_answer, solve_small_image,
)
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig
from skill3d.tools.registry import REGISTRY


ANSWER = (
    "def solve(ctx):\n"
    "    ReturnAnswer(AnswerPayload(value='B', unit='option', basis='visual_estimate'))"
)
YIELD = (
    "def solve(ctx):\n"
    "    response = ctx.tools.view_depth_summary()\n"
    "    if response['status'] == 'ok':\n"
    "        YieldObservations([response['result_id']])"
)


class FakeClient:
    provider = "fake"
    model = "fake-small-image"
    last_usage = {}

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def audit_identity(self, *, seed=None):
        return {"provider": self.provider, "model": self.model,
                "model_revision": "fake-v1", "endpoint_sha256": "a" * 64,
                "credentials": "PRIVATE_IDENTITY_SENTINEL"}

    def chat(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        response = next(self.replies)
        if isinstance(response, Exception):
            raise response
        return response


def problem(tmp_path, benchmark="cv3d", n_images=None, *, options=None):
    n_images = n_images or (1 if benchmark == "cv3d" else 2)
    roots = {"blink": "BLINK_images", "cv3d": "CVBench/3D", "mmsi": "MMSI_images"}
    root = tmp_path / roots[benchmark]
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(n_images):
        path = root / f"image-{index}.png"
        Image.fromarray(np.full((8, 10, 3), 20 + index * 30, dtype=np.uint8)).save(path)
        paths.append(str(path))
    questions = {
        "blink": "These two video frames show a camera orbiting an object. Which way?",
        "cv3d": "Which object is closer, the red box or the blue box?",
        "mmsi": "From the last image observer facing the TV, where is the chair?",
    }
    choices = {
        "blink": ("clockwise (left)", "counter-clockwise (right)"),
        "cv3d": ("Red", "Blue"),
        "mmsi": ("left", "right"),
    }
    return SmallImageProblem(
        benchmark=benchmark, sample_id=f"{benchmark}-fixture", source_split="train",
        question=questions[benchmark], options=options or choices[benchmark],
        image_paths=tuple(paths), task_type="Depth" if benchmark == "cv3d" else "viewpoint")


def prepare(problem):
    return SmallImagePrecomputer().process(problem, categories=())


def fake_prepared(prepared, **changes):
    fake = SimpleNamespace(**{
        name: getattr(prepared, name) for name in prepared.__dataclass_fields__
        if name != "artifact_sha256"
    })
    fake.__dict__.update(changes)
    fake.artifact_sha256 = prepared_content_sha256(fake)
    return fake


@pytest.mark.parametrize("benchmark", ["blink", "cv3d", "mmsi"])
def test_three_benchmarks_use_callable_solver_original_images_and_explicit_option_payload(
        tmp_path, benchmark):
    p = problem(tmp_path, benchmark)
    client = FakeClient([ANSWER])
    before = REGISTRY.names()
    result = solve_small_image(p, client=client, prepared=prepare(p), categories=())
    assert result.status == "answered" and result.answer == "B"
    assert result.basis == "visual_estimate"
    assert result.benchmark == benchmark and result.source_split == "train"
    assert result.input_sha256 == p.input_sha256
    assert result.request_count == result.solve_requests == 1
    assert result.n_images == len(p.image_paths)
    assert result.model_identity["model_revision"] == "fake-v1"
    assert "credentials" not in result.model_identity
    assert len(result.receipt_sha256) == 64
    assert len(result.preparation_identity["cache_key"]) == 64
    assert result.preparation_identity["artifact_sha256"] == prepare(p).artifact_sha256
    assert REGISTRY.names() == before
    text = client.calls[0][0][0]["content"][0]["text"]
    assert "video when the question says so" in text
    assert "BOXES" in text and "POINT markers" in text
    assert "per_view_relative" in text and "world/camera transform" in text


@pytest.mark.parametrize("n_images", [1, 2, 3, 4, 5, 6])
def test_ordered_images_are_sent_in_every_solve_and_finalization_request(tmp_path, n_images):
    p = problem(tmp_path, "mmsi", n_images)
    client = FakeClient(['{"categories":[]}', YIELD, YIELD, '{"answer":"B"}'])
    result = SmallImageSolver(client)(p, prepared=prepare(p))
    assert result.status == "answered" and result.request_count == 4
    assert result.category_requests == 1 and result.solve_requests == 2
    assert result.finalization_requests == 1
    assert result.attempted_tool_calls == result.succeeded_tool_calls == 2
    assert result.authorization_count == 2 and result.denied_authorizations == 0
    assert result.basis == "mixed"
    assert [row["phase"] for row in result.rounds] == [
        "category", "solve", "solve", "finalization"]
    for messages, _ in client.calls[1:]:
        images = [row for row in messages[0]["content"] if row["type"] == "image_url"]
        assert len(images) == n_images
        for index, item in enumerate(images):
            raw = base64.b64decode(item["image_url"]["url"].split(",", 1)[1])
            with Image.open(io.BytesIO(raw)) as decoded:
                assert decoded.size == (10, 8)
                assert abs(float(np.array(decoded).mean()) - (20 + index * 30)) < 2
    feedback_prompt = client.calls[2][0][0]["content"][0]["text"]
    assert '"source_tool": "view_depth_summary"' in feedback_prompt
    assert '"status": "ok"' in feedback_prompt
    assert '"unit": "per_view_relative"' in feedback_prompt


def test_request_exhaustion_includes_category_and_denies_fourth_request(tmp_path):
    p = problem(tmp_path)
    client = FakeClient(['{"categories":[]}', YIELD, YIELD])
    result = SmallImageSolver(client, config=SmallImageRunnerConfig(max_requests=3))(
        p, prepared=prepare(p))
    assert result.status == "request_exhausted" and result.answer is None
    assert len(client.calls) == result.request_count == 3
    assert result.finalization_requests == 0
    assert result.rounds[-1]["status"] == "request_exhausted"


def test_global_governor_denial_is_not_a_model_error_or_request(tmp_path):
    p = problem(tmp_path)
    governor = RequestGovernor(tmp_path / "governor", RequestGovernorConfig(1, 1))
    with governor.request("solver", {}):
        pass
    client = FakeClient([ANSWER])
    result = SmallImageSolver(client, governor=governor)(p, prepared=prepare(p), categories=())
    assert result.status == "request_exhausted" and result.request_count == 0
    assert not client.calls and governor.summary()["reserved_requests"] == 1


def test_category_parser_receives_question_only_and_no_labels_or_answer_options(tmp_path):
    p = problem(tmp_path, options=("OPTION_ONLY_SENTINEL", "OTHER_OPTION_SENTINEL"))
    object.__setattr__(p, "ground_truth", "PRIVATE_GT_SENTINEL")
    object.__setattr__(p, "public_dict", lambda: {"answer": "PRIVATE_GT_SENTINEL"})
    client = FakeClient(['{"categories":["chair"]}', ANSWER])
    received = []

    def builder(problem, *, categories):
        received.append(categories)
        return SmallImagePrecomputer().process(problem, categories=categories)

    result = SmallImageSolver(client, prepare=builder)(p)
    assert result.status == "answered" and received == [("chair",)]
    category_text = json.dumps(client.calls[0][0])
    assert p.question in category_text
    assert "OPTION_ONLY_SENTINEL" not in category_text
    assert "PRIVATE_GT_SENTINEL" not in json.dumps(client.calls)
    receipt = json.dumps(result.to_dict())
    assert all(value not in receipt for value in (
        "PRIVATE_GT_SENTINEL", "OPTION_ONLY_SENTINEL", "PRIVATE_IDENTITY_SENTINEL",
        "ReturnAnswer", p.question))
    assert build_category_messages(p.question) == client.calls[0][0]


@pytest.mark.parametrize("categories", [(), None])
@pytest.mark.parametrize("key,value", [
    ("input_sha256", "b" * 64), ("source_split", "test"), ("benchmark", "mmsi"),
    ("sample_id", "other-question"), ("question", "other wording"),
    ("options", ["near", "far"]), ("task_type", "Distance"),
    ("image_paths", ["other.png"]),
])
def test_prepared_metadata_mismatch_is_rejected_before_registry_or_request(
        tmp_path, key, value, categories):
    p = problem(tmp_path)
    prepared = prepare(p)
    prepared = replace(prepared, problem={**dict(prepared.problem), key: value})
    client = FakeClient([ANSWER])
    built = []
    result = SmallImageSolver(
        client, registry_builder=lambda *args: built.append(args))(
            p, prepared=prepared, categories=categories)
    assert result.status == "prepared_identity_mismatch"
    assert result.request_count == 0 and not client.calls and not built


@pytest.mark.parametrize("categories", [(), None])
def test_reordered_and_resized_prepared_pixels_are_rejected(tmp_path, categories):
    p = problem(tmp_path, "blink")
    prepared = prepare(p)
    client = FakeClient([ANSWER])
    for changed in (prepared.frames[::-1], tuple(frame[:4] for frame in prepared.frames)):
        result = SmallImageSolver(client)(
            p, prepared=replace(prepared, frames=changed), categories=categories)
        assert result.status == "prepared_input_mismatch" and result.request_count == 0
    assert not client.calls


@pytest.mark.parametrize("categories", [(), None])
@pytest.mark.parametrize("mutation", [
    "candidate", "candidate_mask", "depth", "points", "intrinsics",
    "geometry_mask", "depth_map", "review", "provenance",
])
def test_actual_prepared_mutations_under_unchanged_cache_key_spend_no_requests(
        tmp_path, mutation, categories):
    p = problem(tmp_path)
    precomputer = SmallImagePrecomputer(
        detector=lambda rgb, categories: [
            {"category": "chair", "box_xyxy": [0, 0, 8, 8], "confidence": .8}],
        segmenter=lambda rgb, boxes: [np.ones(rgb.shape[:2], dtype=bool)],
        depth_estimator=lambda rgb: {
            "depth": np.ones(rgb.shape[:2]), "valid_mask": np.ones(rgb.shape[:2], dtype=bool),
            "intrinsics": np.eye(3)},
        model_identities={name: {"model": "fake-" + name, "revision": "one"}
                          for name in ("detector", "segmenter", "depth_estimator")})
    prepared = precomputer.process(p, categories=("chair",))
    assert not prepared.errors
    cache_key, digest = prepared.cache_key, prepared.artifact_sha256
    if mutation == "candidate":
        prepared.candidates[0]["confidence"] = .1
    elif mutation == "candidate_mask":
        mask = prepared.candidates[0]["mask"].copy()
        mask[0, 0] = False
        prepared.candidates[0]["mask"] = mask
    elif mutation in ("depth", "points", "intrinsics"):
        prepared.geometries[0][mutation] = prepared.geometries[0][mutation] + .1
    elif mutation == "geometry_mask":
        mask = prepared.geometries[0]["valid_mask"].copy()
        mask[0, 0] = False
        prepared.geometries[0]["valid_mask"] = mask
    elif mutation == "depth_map":
        prepared = replace(prepared, depths={0: prepared.depths[0] + .1})
    elif mutation == "review":
        prepared.candidates[0]["review"]["p_yes"] = .9
    else:
        prepared.provenance["config"]["models"]["detector"]["revision"] = "changed"
    assert prepared.cache_key == cache_key and prepared.artifact_sha256 == digest
    assert prepared_content_sha256(prepared) != digest
    client = FakeClient(['{"categories":[]}', ANSWER])
    governor = RequestGovernor(tmp_path / "governor", RequestGovernorConfig(4, 1))
    built = []
    result = SmallImageSolver(
        client, governor=governor, registry_builder=lambda *args: built.append(args))(
            p, prepared=prepared, categories=categories)
    assert result.status == "prepared_identity_mismatch"
    assert result.request_count == 0 and not client.calls and not built
    assert governor.summary()["reserved_requests"] == 0
    assert result.preparation_identity["cache_key"] == cache_key
    assert result.preparation_identity["artifact_sha256"] == digest


@pytest.mark.parametrize("digest", ["", "not-a-digest", "0" * 64, None])
def test_actual_prepared_missing_or_invalid_digest_is_not_silently_recomputed(tmp_path, digest):
    p = problem(tmp_path)
    prepared = replace(prepare(p), artifact_sha256=digest)
    client = FakeClient(['{"categories":[]}', ANSWER])
    result = SmallImageSolver(client)(p, prepared=prepared)
    assert result.status == "prepared_identity_mismatch"
    assert result.request_count == 0 and not client.calls


def test_injected_fake_requires_explicit_helper_digest(tmp_path):
    p = problem(tmp_path)
    fake = fake_prepared(prepare(p))
    del fake.artifact_sha256
    client = FakeClient([ANSWER])
    solver = SmallImageSolver(client)
    result = solver(p, prepared=fake, categories=())
    assert result.status == "prepared_identity_mismatch"
    assert result.request_count == 0 and not client.calls
    fake.artifact_sha256 = prepared_content_sha256(fake)
    result = solver(p, prepared=fake, categories=())
    assert result.status == "answered" and result.request_count == 1
    assert result.preparation_identity["artifact_sha256"] == fake.artifact_sha256


def test_generated_prepared_is_digest_validated_before_solve_request(tmp_path):
    p = problem(tmp_path)
    prepared = replace(prepare(p), provenance={"model": "changed"})
    client = FakeClient([ANSWER])
    result = SmallImageSolver(client, prepare=lambda *args, **kwargs: prepared)(
        p, categories=())
    assert result.status == "prepared_identity_mismatch"
    assert result.request_count == 0 and not client.calls


def test_input_image_changes_are_detected_before_any_request(tmp_path):
    p = problem(tmp_path)
    Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(p.image_paths[0])
    client = FakeClient([ANSWER])
    assert SmallImageSolver(client)(p, prepared=prepare(p)).status == "input_error"
    assert not client.calls


@pytest.mark.parametrize("source", [
    "import os\nReturnAnswer('B')",
    "import pathlib\nReturnAnswer('B')",
    "import numpy as np\nx = np.fromfile('PRIVATE_FILE')\nReturnAnswer('B')",
    "open('PRIVATE_FILE', 'w')\nReturnAnswer('B')",
    "ctx.tools.object_centroid('chair')\nReturnAnswer('B')",
    "ReturnAnswer(AnswerPayload(value='B', unit='option', basis='visual_estimate'))\n"
    "ReadObservation('forged')",
])
def test_unauthorized_tools_and_host_file_imports_fail_closed(tmp_path, source):
    p = problem(tmp_path)
    client = FakeClient([source])
    result = SmallImageSolver(client)(p, prepared=prepare(p), categories=())
    assert result.status == "program_rejected" and result.answer is None
    assert result.request_count == 1 and result.attempted_tool_calls == 0
    assert not (tmp_path / "PRIVATE_FILE").exists()
    assert source not in json.dumps(result.to_dict())


def test_successful_point_depth_uses_real_custom_envelope_and_authorization(tmp_path):
    p = problem(tmp_path)
    precomputer = SmallImagePrecomputer(
        depth_estimator=lambda rgb: {
            "depth": np.ones(rgb.shape[:2]), "valid_mask": np.ones(rgb.shape[:2], dtype=bool)},
        model_identities={"depth_estimator": {"model": "fake-depth-v1"}})
    prepared = precomputer.process(p, categories=())
    client = FakeClient([
        "def solve(ctx):\n"
        "    result = ctx.tools.point_depth(view_index=0, x=3, y=3)\n"
        "    if result['status'] == 'ok' and result['data']['depth'] == 1:\n"
        "        ReturnAnswer(AnswerPayload(value='B', unit='option', basis='mixed'))"
    ])
    result = SmallImageSolver(client)(p, prepared=prepared, categories=())
    assert result.status == "answered" and result.basis == "mixed"
    assert result.succeeded_tool_calls == result.authorization_count == 1
    assert result.geometry_views == (0,)


def test_category_confirmation_does_not_authorize_candidate_geometry(tmp_path):
    p = problem(tmp_path)
    precomputer = SmallImagePrecomputer(
        depth_estimator=lambda rgb: {
            "depth": np.ones(rgb.shape[:2]), "valid_mask": np.ones(rgb.shape[:2], dtype=bool)},
        model_identities={"depth_estimator": {"model": "fake-depth-v1"}})
    prepared = precomputer.process(p, categories=())
    prepared = fake_prepared(prepared, candidates=({
        "candidate_id": "chair-0", "view_index": 0, "category": "chair",
        "category_confirmed": True, "geometry_usable": False,
        "box_xyxy": [0, 0, 8, 8], "mask_quality_assessed": False,
    },))
    client = FakeClient([
        "def solve(ctx):\n"
        "    result = ctx.tools.candidate_depth_summary(candidate_id='chair-0')\n"
        "    ReturnAnswer(AnswerPayload(value='B', unit='option', basis='mixed'))"
    ])
    result = SmallImageSolver(client)(p, prepared=prepared, categories=())
    assert result.status == "program_rejected" and result.answer is None
    assert result.request_count == 1
    assert result.attempted_tool_calls == result.denied_authorizations == 0
    assert result.succeeded_tool_calls == 0


def test_service_failure_is_terminal_and_never_persists_exception_text(tmp_path):
    p = problem(tmp_path)
    client = FakeClient([RuntimeError("CREDENTIAL_AND_RAW_ERROR_SENTINEL"), ANSWER])
    result = SmallImageSolver(client)(p, prepared=prepare(p), categories=())
    assert result.status == "service_error" and result.request_count == 1
    assert len(client.calls) == 1 and result.answer is None
    assert "CREDENTIAL_AND_RAW_ERROR_SENTINEL" not in json.dumps(result.to_dict())


@pytest.mark.parametrize("categories", [(), None])
def test_preparation_failures_do_not_silently_become_visual_answers(tmp_path, categories):
    p = problem(tmp_path)
    prepared = replace(prepare(p), errors=({"stage": "detector", "error": "PRIVATE_ERROR"},))
    client = FakeClient([ANSWER])
    result = SmallImageSolver(client)(p, prepared=prepared, categories=categories)
    assert result.status == "precompute_error" and not client.calls
    assert result.request_count == 0
    assert "PRIVATE_ERROR" not in json.dumps(result.to_dict())


@pytest.mark.parametrize("text", [
    '{"answer":"B","tool":"point_depth"}',
    '{"answer":"B","answer":"A"}',
    '```json\n{"answer":"B"}\n```',
    'ReturnAnswer("B")',
    '{"answer":true}',
    '{"answer":"C"}',
])
def test_finalization_accepts_only_option_json_and_never_executes_tools(tmp_path, text):
    p = problem(tmp_path)
    client = FakeClient([YIELD, YIELD, text])
    result = SmallImageSolver(client)(p, prepared=prepare(p), categories=())
    assert result.status == "invalid_finalization" and result.answer is None
    assert result.request_count == 3 and result.finalization_requests == 1
    assert result.attempted_tool_calls == 2
    assert "No code, tool calls" in client.calls[-1][0][0]["content"][0]["text"]


@pytest.mark.parametrize("options,answer,wanted", [
    (("left", "right"), "b", "B"),
    (("left", "right"), "(B)", "B"),
    (("A. left", "B. right"), "right", "B"),
    (("(A) left", "(B) right"), "B. right", "B"),
    (("Red", "Blue"), "BLUE", "B"),
    (("left", "right"), "B)", "B"),
])
def test_options_and_host_only_scoring(options, answer, wanted):
    assert parse_small_image_answer(answer, options) == wanted
    assert score_small_image_answer(answer, wanted, options)
    assert not score_small_image_answer(None, wanted, options)


@pytest.mark.parametrize("options,answer", [
    (("left", "right"), "I think B is correct"),
    (("left", "right"), "C"),
    (("left", "right"), True),
    (("B", "right"), "B"),
    (("A. left", "right"), "B"),
    (("B. left", "A. right"), "B"),
    (("left", "LEFT"), "A"),
])
def test_ambiguous_or_invalid_options_are_never_guessed(options, answer):
    with pytest.raises(ValueError):
        parse_small_image_answer(answer, options)


def test_preflight_never_calls_client_and_unavailable_prepare_does_not_spend_category_request(tmp_path):
    p = problem(tmp_path)
    client = FakeClient([ANSWER])
    result = preflight_small_image(p, client=client)
    assert result.status == "preflight" and result.request_count == 0
    result = SmallImageSolver(client)(p)
    assert result.status == "precompute_unavailable" and result.request_count == 0
    assert not client.calls


def test_cli_default_is_no_request_no_backend_no_gt_and_verifies_public_manifest(tmp_path, monkeypatch, capsys):
    p = problem(tmp_path)
    manifest = tmp_path / "manifest.json"
    freeze_manifest([p], manifest)
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_small_image.py"
    spec = importlib.util.spec_from_file_location("run_small_image_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    def forbidden(*_args, **_kwargs):
        pytest.fail("preflight constructed GPU backends or issued a model request")

    monkeypatch.setattr(cli, "build_precomputer", forbidden)
    monkeypatch.setattr(cli.BailianClient, "chat", forbidden)
    monkeypatch.setattr(cli.BailianClient, "check_service", forbidden)
    status = cli.main(["--manifest", str(manifest), "--dataset-root", str(tmp_path)])
    assert status == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "preflight" and receipt["request_count"] == 0
    assert receipt["input_sha256"] == p.input_sha256
    assert "question" not in receipt and "image_paths" not in receipt


@pytest.mark.parametrize("kwargs", [{"max_requests": 5}, {"max_solve_rounds": 3}])
def test_runner_configuration_cannot_expand_request_or_round_caps(kwargs):
    with pytest.raises(ValueError):
        SmallImageRunnerConfig(**kwargs)
