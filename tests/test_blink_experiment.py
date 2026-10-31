"""No-network regression checks for the BLINK-only request and campaign changes."""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
from io import BytesIO
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
import pytest

from skill3d.evaluation import blink_experiment as experiment
from skill3d.small_image.blink_protocol import (
    DIRECT_SUFFIX, benchmark_prompt, explicit_answer, original_messages, sha_value,
)
from skill3d.small_image.datasets import SmallImageProblem, _sha256, freeze_manifest
from skill3d.small_image.precompute import SmallImagePrecomputer
from skill3d.small_image.runner import SmallImageRunnerConfig, SmallImageSolver
from skill3d.small_image.storage import save_prepared
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.client_errors import ServiceUnavailable
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"configs/experiments/blink.json"
YIELD = (
    "def solve(ctx):\n"
    "    r = ctx.tools.view_depth_summary()\n"
    "    YieldObservations([r['result_id']])"
)
ANSWER = "def solve(ctx):\n    ReturnAnswer(AnswerPayload(value='B', unit='option', basis='visual_estimate'))"
QUESTION = (
    "The images are frames from a video. The video is shooting a static scene. "
    "The camera is either moving clockwise (left) or counter-clockwise (right) around the object. "
    "The first image is from the beginning of the video and the second image is from the end. "
    "Is the camera moving left or right when shooting the video? Select from the following options."
)


@pytest.fixture(autouse=True)
def no_live_credentials(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


class Response(BytesIO):
    pass


def fake_client(texts, *, omit_model=False):
    texts = iter(texts)
    captured = []
    def open_response(request, timeout):
        payload = json.loads(request.data)
        captured.append(payload)
        text = next(texts)
        if isinstance(text, BaseException):
            raise text
        body = {
            "id": "test-response", "created": 1,
            "choices": [{"message": {"content": text,
                                     "reasoning_content": "provider-reasoning" if payload["enable_thinking"] else None},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12,
                      "completion_tokens_details": {"reasoning_tokens": 5}},
        }
        if not omit_model:
            body["model"] = "qwen3.8-flash"
        return Response(json.dumps(body).encode())
    client = BailianClient(max_output_tokens=8192, opener=open_response)
    client.captured = captured
    return client


def problem(tmp_path):
    paths = []
    image_root = tmp_path/"BLINK_images"
    image_root.mkdir(exist_ok=True)
    for i in range(2):
        p = image_root/f"{i}.jpg"
        pixels = np.random.default_rng(i).integers(0, 256, (8, 10, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(p, format="JPEG", quality=83)
        paths.append(str(p))
    return SmallImageProblem("blink", "blink-test", "train", QUESTION,
                             ("left", "right"), tuple(paths), "Multi-view_Reasoning")


def config():
    return SmallImageRunnerConfig(**experiment.profile_at(CONFIG)["solver"])


def test_thinking_original_bytes_category_and_finalization_share_protocol(tmp_path):
    p = problem(tmp_path)
    prepared = SmallImagePrecomputer().process(p, categories=())
    client = fake_client(['{"categories":[]}', YIELD, YIELD,
                          'The result follows from the views.\n```json\n{"reasoning":"brief","answer":"B"}\n```'])
    result = SmallImageSolver(client, config=config(), skill_text="unit skill")(
        p, prepared=prepared)
    assert result.status == "answered" and result.answer == "B"
    assert result.request_count == 4
    assert client.captured[0]["enable_thinking"] is False
    assert client.captured[0]["response_format"] == {"type": "json_object"}
    for payload, row in zip(client.captured[1:], result.rounds[1:]):
        assert payload["enable_thinking"] is True and payload["max_tokens"] == 8192
        assert "response_format" not in payload
        images = [c for c in payload["messages"][0]["content"] if c["type"] == "image_url"]
        assert [base64.b64decode(c["image_url"]["url"].split(",", 1)[1]) for c in images] == [
            Path(path).read_bytes() for path in p.image_paths]
        assert row["reasoning_content"] == "provider-reasoning"
        assert row["usage"]["completion_tokens_details"]["reasoning_tokens"] == 5
        assert row["delivered_skill_sha256"] == sha_value("unit skill")
        assert row["request_payload_sha256"] == sha_value(payload)
    assert "provider-reasoning" not in result.rounds[1]["program_source"]
    assert result.rounds[-1]["answer_parse_method"] == "final_fenced_json"


def test_client_clipping_and_missing_reported_model_stop_before_or_after_one_call(tmp_path):
    p = problem(tmp_path)
    prepared = SmallImagePrecomputer().process(p, categories=())
    clipped = fake_client([ANSWER])
    clipped.max_output_tokens = 2048
    result = SmallImageSolver(clipped, config=config())(p, prepared=prepared, categories=())
    assert result.status == "input_error" and not clipped.captured
    unreported = fake_client([ANSWER], omit_model=True)
    result = SmallImageSolver(unreported, config=config())(p, prepared=prepared, categories=())
    assert result.status == "service_error" and len(unreported.captured) == 1
    assert result.rounds[0]["model_response"] == ANSWER
    assert result.rounds[0]["response_identity"]["model_reported"] is False


@pytest.mark.parametrize("text", [
    '{"answer":"A","answer":"B"}', '{"answer":"B","tool":"point_depth"}',
    '```json\n{"answer":"B","program":"open(path)"}\n```',
    "I considered A and B; either may be plausible.", '{"answer":true}', '{"answer":"C"}',
])
def test_explicit_answer_rejects_ambiguous_or_executable_response(text):
    with pytest.raises((ValueError, TypeError)):
        explicit_answer(text, ("left", "right"))


def test_client_reasoning_is_reset_on_next_request_and_retained_when_content_missing():
    client = fake_client(["A", "B", None])
    client.chat([{"role": "user", "content": "test"}], enable_thinking=True)
    assert client.last_reasoning_content == "provider-reasoning"
    client.chat([{"role": "user", "content": "test"}], enable_thinking=False)
    assert client.last_reasoning_content is None
    with pytest.raises(RuntimeError, match="empty content"):
        client.chat([{"role": "user", "content": "test"}], enable_thinking=True)
    assert client.last_reasoning_content == "provider-reasoning"
    assert client.last_usage["total_tokens"] == 12


def test_request_intent_is_not_replayed_and_full_output_is_outside_budget(tmp_path):
    base = fake_client(["B"])
    governor = RequestGovernor(tmp_path/"budget", RequestGovernorConfig(4, 1))
    client = experiment.RecordedClient(base, governor, tmp_path, "operation", "solver")
    client.chat([{"role": "user", "content": "public question"}], enable_thinking=True)
    response = experiment.read(tmp_path/"requests/operation/001.response.json")
    assert response["reasoning_content"] == "provider-reasoning"
    assert "provider-reasoning" not in (tmp_path/"budget/state.json").read_text()
    replay = experiment.RecordedClient(base, governor, tmp_path, "operation", "solver")
    with pytest.raises(RuntimeError, match="request_replay_forbidden"):
        replay.chat([{"role": "user", "content": "test"}])
    assert len(base.captured) == 1


def test_failed_request_records_timing_and_safe_diagnostic_and_remains_counted(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append(timeout)
        raise TimeoutError("private prompt and unit-test-placeholder")

    base = BailianClient(opener=opener, timeout_s=600)
    governor = RequestGovernor(tmp_path/"budget", RequestGovernorConfig(4, 1))
    client = experiment.RecordedClient(base, governor, tmp_path, "operation", "solver")
    with pytest.raises(ServiceUnavailable):
        client.chat([{"role": "user", "content": "private question"}], enable_thinking=True)
    response_path = tmp_path/"requests/operation/001.response.json"
    before = response_path.read_bytes()
    response = experiment.read(response_path)
    intent = experiment.read(tmp_path/"requests/operation/001.intent.json")
    assert response["finished_at"] >= intent["started_at"]
    assert response["service_error"]["reason_code"] == "timeout"
    assert response["service_error"]["request_stage"] == "awaiting_headers"
    assert response["model_response"] is None and response["reasoning_content"] is None
    assert response["usage"] == {} and response["status"] == "error"
    assert "private" not in before.decode() and "unit-test-placeholder" not in before.decode()
    summary = governor.summary()
    assert summary["reserved_requests"] == summary["error_requests"] == 1
    assert summary["active_requests"] == summary["unknown_requests"] == 0
    replay = experiment.RecordedClient(base, governor, tmp_path, "operation", "solver")
    with pytest.raises(RuntimeError, match="request_replay_forbidden"):
        replay.chat([{"role": "user", "content": "private question"}])
    assert calls == [600] and response_path.read_bytes() == before
    assert governor.summary()["reserved_requests"] == 1


def make_gate_pairs():
    pairs, references = [], {}
    for seed in (137, 138):
        for i in range(8):
            references[str(i)] = "B"
            arms = []
            for name in ("parent", "candidate"):
                arms.append({
                    "status": "complete", "seed": seed, "input_sha256": f"input-{i}",
                    "artifact_sha256": f"cache-{i}", "solver": {"seed": seed},
                    "images": [f"image-{i}"], "model": {"model": "qwen3.8-flash"},
                    "skill_sha256": sha_value(name),
                    "result": {"sample_id": str(i), "source_split": "train",
                               "status": "answered", "answer": "B" if name == "candidate" else "A",
                               "denied_authorizations": 0,
                               "rounds": [{
                                   "phase": "solve", "status": "answered",
                                   "response_identity": {"model": "qwen3.8-flash",
                                                         "model_reported": True, "backend_fingerprint": None},
                                   "request_parameters": {"seed": seed},
                                   "delivered_skill_sha256": sha_value(name)}]},
                })
            pairs.append(tuple(arms))
    return pairs, references


def test_gate_checks_both_seeds_actual_delivery_and_backend_observations():
    pairs, refs = make_gate_pairs()
    decision = experiment.gate_decision(pairs, refs)
    assert decision["promoted"] and decision["backend_fingerprint_observed"] is False
    changed = deepcopy(pairs)
    changed[-1][1]["result"]["rounds"][0]["request_parameters"]["seed"] = 137
    with pytest.raises(ValueError, match="actual_gate_seed_mismatch"):
        experiment.gate_decision(changed, refs)
    changed = deepcopy(pairs)
    changed[-1][1]["result"]["rounds"][0]["delivered_skill_sha256"] = "other"
    with pytest.raises(ValueError, match="actual_gate_skill_mismatch"):
        experiment.gate_decision(changed, refs)
    changed = deepcopy(pairs)
    for a, b in changed[8:]:
        b["result"]["answer"] = a["result"]["answer"]
    assert not experiment.gate_decision(changed, refs)["promoted"]


def fixture_campaign(tmp_path, monkeypatch):
    template = problem(tmp_path)
    profile = experiment.profile_at(CONFIG, tmp_path)
    profile["dataset_root"] = str(tmp_path)
    profile["runtime"]["max_requests"] = 3000
    problems = [replace(template, sample_id=f"fixture-{split}-{i}", source_split=split)
                for split, n in (("train", 40), ("test", 93)) for i in range(n)]
    freeze_manifest(problems, tmp_path/"manifest.json")
    for p in problems:
        prepared = SmallImagePrecomputer().process(p, categories=())
        save_prepared(prepared, tmp_path/"prepared"/p.input_sha256)
    seed_body = {"benchmark": "blink", "task_type": "Multi-view_Reasoning",
                 "allowed_tools": experiment.TOOLS, "text": "x"*150}
    seed = {**seed_body, "content_sha256": _sha256(seed_body)}
    (tmp_path/"seed_skills.json").write_text(json.dumps({"skills": {"one": seed}}))
    (tmp_path/"host_evaluation").mkdir()
    labels = [{"benchmark": "blink", "sample_id": p.sample_id,
               "source_split": p.source_split, "reference_answer": "B"} for p in problems]
    (tmp_path/"host_evaluation/answers.json").write_text(json.dumps(labels))
    original_sha = experiment.file_sha
    def guard_sha(path):
        if str(path).endswith("host_evaluation/answers.json"):
            pytest.fail("preflight must not read ground-truth file contents")
        return original_sha(path)
    monkeypatch.setattr(experiment, "file_sha", guard_sha)
    inspection = experiment.preflight(profile)
    monkeypatch.setattr(experiment, "file_sha", original_sha)
    run_root = tmp_path/"run"
    for name in ("scripts", "configs/experiments", "src/skill3d"):
        (run_root/"code/source"/name).mkdir(parents=True, exist_ok=True)
    (run_root/"code/source/scripts/run_blink_experiment.py").write_text("# fixture\n")
    (run_root/"code/source/configs/experiments/blink.json").write_text("{}\n")
    manifest = experiment.write_once(run_root/"manifest.json", {
        "schema": experiment.SCHEMA, "profile": profile, "preflight": inspection,
        "source_files": experiment.source_files(run_root/"code/source"),
        "reference_file_sha256": original_sha(tmp_path/"host_evaluation/answers.json"),
    })
    return experiment.Campaign(run_root, manifest), template


def test_full_campaign_order_freezes_predictions_before_test_scoring_and_resume(tmp_path, monkeypatch):
    campaign, _ = fixture_campaign(tmp_path, monkeypatch)
    requests = fake_client(['{"no_change":true}']*3)
    monkeypatch.setattr(experiment, "BailianClient", lambda **kwargs: requests)
    events = []
    def operation(p, text, stage, seed=137, mode="system"):
        events.append(stage)
        cache = campaign.inputs["cache_identities"][p.sample_id]
        return experiment.write_once(campaign.root/"results"/stage/(p.sample_id+".json"), {
            "input_sha256": p.input_sha256, "images": cache["images"],
            "result": {"sample_id": p.sample_id, "status": "answered", "answer": "B",
                       "source_split": p.source_split, "rounds": [{"status": "answered"}],
                       "denied_authorizations": 0, "request_count": 1},
        })
    monkeypatch.setattr(campaign, "operation", operation)
    original_references = campaign.references
    def references(role):
        if role == "test":
            assert events.count("final") >= 93 and (campaign.root/"skills/frozen.json").exists()
        return original_references(role)
    monkeypatch.setattr(campaign, "references", references)
    result = campaign.run()
    assert events == ["direct"]*93+["no_skill"]*93+["initial"]*93+["training"]*40+["final"]*93
    assert all(score["questions"] == 93 for score in result["scores"].values())
    frozen = experiment.read(campaign.root/"skills/frozen.json")
    assert frozen["attempts"] == 3 and frozen["unrevised_tail_questions"] == 4
    campaign.run()  # reconstruct state from immutable training/revision records
    assert len(requests.captured) == 3


def test_unfinished_operation_stops_without_dispatching_worker(tmp_path, monkeypatch):
    campaign, _ = fixture_campaign(tmp_path, monkeypatch)
    class NoExecutor:
        def __init__(self, *args, **kwargs):
            pytest.fail("unfinished operation must not dispatch or repeat a model call")
    monkeypatch.setattr(experiment, "ProcessPoolExecutor", NoExecutor)
    p = campaign.tests[0]
    cache = campaign.inputs["cache_identities"][p.sample_id]
    identity = {"stage": "initial", "input_sha256": p.input_sha256,
                "skill_sha256": sha_value("skill"), "seed": 137,
                "solver": campaign.profile["solver"], "images": cache["images"],
                "artifact_sha256": cache["artifact_sha256"], "model": campaign.profile["model"],
                "campaign": campaign.manifest["content_sha256"],
                "mode": "system", "allowed_tools": experiment.TOOLS}
    experiment.write_once(campaign.root/"intents"/(sha_value(identity)+".json"), {"identity": identity})
    with pytest.raises(RuntimeError, match="unfinished_operation"):
        campaign.operation(p, "skill", "initial")


def test_direct_worker_payload_uses_same_official_prompt_without_reading_prepared(tmp_path, monkeypatch):
    p = problem(tmp_path)
    base = fake_client(['{"reasoning":"brief","answer":"B"}'])
    monkeypatch.setattr(experiment, "BailianClient", lambda **kwargs: base)
    monkeypatch.setattr(experiment, "load_prepared", lambda *args, **kwargs: pytest.fail("direct needs no geometry"))
    spec = {"problem": p.public_dict(), "model": {}, "run_root": str(tmp_path/"run"),
            "max_requests": 4, "request_concurrency": 1, "operation_id": "direct",
            "category": "solver", "solver": asdict(config()), "mode": "direct"}
    result = experiment.solve_worker(spec)
    assert result["result"]["answer"] == "B"
    assert base.captured[0]["messages"] == original_messages(p, benchmark_prompt(p)+DIRECT_SUFFIX)
    assert base.captured[0]["max_tokens"] == 8192 and "response_format" not in base.captured[0]


def test_formal_freeze_rejects_dirty_repo_and_copies_only_selected_source(tmp_path, monkeypatch):
    profile = experiment.profile_at(CONFIG)
    profile["runtime"]["max_requests"] = 10
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *args, **kwargs: " M file\n")
    with pytest.raises(ValueError, match="committed_and_clean"):
        experiment.create_run(ROOT, tmp_path/"new-run", profile, {})
    assert not (tmp_path/"new-run").exists()
    repo = tmp_path/"repo"
    for relative in ("src/skill3d/minimal.py", "scripts/run_blink_experiment.py",
                     "configs/experiments/blink.json", "credentials.txt"):
        target = repo/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# test fixture\n")
    answers = tmp_path/"references.json"
    answers.write_text("[]")
    profile["inputs"]["host_answers"] = str(answers)
    inspection = {"inputs": {"host_answers": experiment.file_identity(answers)}}
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda cmd, **kwargs: (
        "" if "status" in cmd else "a"*40+"\n"))
    manifest = experiment.create_run(repo, tmp_path/"new-run", profile, inspection)
    assert set(manifest["source_files"]) == {
        "src/skill3d/minimal.py", "scripts/run_blink_experiment.py", "configs/experiments/blink.json"}
    assert not (tmp_path/"new-run/code/source/credentials.txt").exists()
    assert experiment.source_files(tmp_path/"new-run/code/source") == manifest["source_files"]


def offline_spawn_worker(spec):
    """Mock only the provider inside a real spawned worker, retaining real solver/kernel."""
    original = experiment.BailianClient
    base = fake_client([ANSWER])
    experiment.BailianClient = lambda **kwargs: base
    try:
        result = experiment.solve_worker(spec)
        return {"result": result, "payloads": base.captured}
    finally:
        experiment.BailianClient = original


def test_spawn_worker_imports_and_executes_real_solver_in_process_main_thread(tmp_path):
    p = problem(tmp_path)
    prepared = SmallImagePrecomputer().process(p, categories=())
    save_prepared(prepared, tmp_path/"prepared"/p.input_sha256)
    spec = {"problem": p.public_dict(), "model": {}, "run_root": str(tmp_path/"run"),
            "max_requests": 4, "request_concurrency": 1, "operation_id": "spawned",
            "category": "solver", "solver": asdict(config()), "mode": "system",
            "prepared_root": str(tmp_path/"prepared"), "artifact_sha256": prepared.artifact_sha256,
            "skill_text": "test skill"}
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        result = pool.submit(offline_spawn_worker, spec).result(timeout=30)
    assert result["result"]["result"]["status"] == "answered"
    assert result["result"]["result"]["answer"] == "B"
    assert result["payloads"][0]["enable_thinking"] is True


def test_resume_uses_saved_budget_and_frozen_entry_without_new_development_preflight(tmp_path, monkeypatch):
    path = ROOT/"scripts/run_blink_experiment.py"
    spec = importlib.util.spec_from_file_location("blink_cli_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    run = tmp_path/"run"
    saved_profile = experiment.profile_at(CONFIG)
    saved_profile["runtime"]["max_requests"] = 1234
    experiment.write_once(run/"manifest.json", {"profile": saved_profile})
    called = []
    monkeypatch.setattr(cli.subprocess, "run", lambda argv, **kwargs: (
        called.append(argv) or SimpleNamespace(returncode=0)))
    monkeypatch.setattr(experiment, "preflight", lambda *args: pytest.fail("development preflight on resume"))
    assert cli.main(["--resume", "--check", "--output-dir", str(run)]) == 0
    assert called[0][2] == str(run/"code/source/scripts/run_blink_experiment.py")
    assert "--frozen-child" in called[0]
    with pytest.raises(ValueError, match="resume_request_budget_changed"):
        cli.main(["--resume", "--execute", "--max-requests", "999", "--output-dir", str(run)])
