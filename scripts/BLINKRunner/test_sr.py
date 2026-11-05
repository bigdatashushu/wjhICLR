"""Offline SR tests: public custody, real tool execution, original images, scoring."""
from __future__ import annotations

import base64
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr
import sr_data
from skill3d.small_image.blink_protocol import image_identities, sha_value
from sr_adapter import SRProblem as SmallImageProblem
from skill3d.small_image.precompute import SmallImagePrecomputer
from sr_adapter import SRSolver as SmallImageSolver
from sr_protocol import (
    VERSION, official_prompt, relation_for, validate_official_prompt,
)
from skill3d.small_image.storage import save_prepared


class FakeClient:
    model = "qwen3.8-flash"
    provider = "fake"
    temperature = 0
    max_output_tokens = 8192
    last_usage = {}
    last_response_identity = {"model": model, "model_reported": True}
    last_request_identity = {}
    last_generation = {}

    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []

    def build_payload(self, messages, **kwargs):
        return {"model": self.model, "messages": messages, **kwargs}

    def chat(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        response = next(self.replies)
        if isinstance(response, BaseException):
            raise response
        return response


def problem(tmp_path, question="Is the cat behind the car?", split="val"):
    image = tmp_path / "original.png"
    if not image.exists():
        Image.fromarray(np.full((12, 16, 3), 90, dtype=np.uint8)).save(image)
    return SmallImageProblem(
        benchmark="blink", sample_id=f"{split}:{split}_Spatial_Relation_1",
        source_split=split, task_type="Spatial_Relation", question=question,
        options=("yes", "no"), image_paths=(str(image),))


def prepared(problem):
    def depth(rgb):
        h, w = rgb.shape[:2]
        return {"depth": np.tile(np.arange(1, w + 1, dtype=np.float32), (h, 1)),
                "valid_mask": np.ones((h, w), dtype=bool), "relative_only": True}
    return SmallImagePrecomputer(
        depth_estimator=depth, model_identities={"depth_estimator": {"model": "fake-depth"}}
    ).process(problem, categories=("cat", "car"))


@pytest.mark.parametrize("question,expected", [
    ("Is the cat in front of the car?", ("cat", "in front of", "car", "depth")),
    ("Is the truck at the left side of the train?", ("truck", "at the left side of", "train", "horizontal")),
    ("Is the person facing away from the horse?", ("person", "facing away from", "horse", "orientation")),
    ("Does the suitcase contain the cat?", ("suitcase", "contain", "cat", "containment")),
    ("Does the cake have the cat as a part?", ("cake", "have as a part", "cat", "part_whole")),
    ("Is the laptop a part of the cake?", ("laptop", "a part of", "cake", "part_whole")),
    ("Is the potted plant at the edge of the dining table?", ("potted plant", "at the edge of", "dining table", "boundary")),
])
def test_relation_routes_evidence_and_preserves_subject_target(question, expected):
    value = relation_for(question)
    assert tuple(value[k] for k in ("subject", "relation", "target", "family")) == expected
    assert "answer" not in value


def test_direct_never_loads_geometry_and_preserves_original_bytes(tmp_path, monkeypatch):
    p = problem(tmp_path)
    profile, _, _ = sr.configuration()
    client = FakeClient(['{"reasoning":"visual fixture","answer":"A"}'])
    def forbidden(*_args, **_kwargs):
        raise AssertionError("direct loaded prepared evidence")
    monkeypatch.setattr(sr, "prepared_for", forbidden)
    spec = {"arm": "direct", "problem": p.public_dict(), "solver": profile["solver"],
            "official_prompt": official_prompt(p)}
    result = sr.solve_one(spec, client)
    assert result["answer"] == "A" and result["request_count"] == 1
    assert result["succeeded_tool_calls"] == 0 and result["reference_answer_visible"] is False
    content = client.calls[0][0][0]["content"]
    image = [r for r in content if r["type"] == "image_url"]
    assert len(image) == 1
    assert base64.b64decode(image[0]["image_url"]["url"].split(",", 1)[1]) == Path(p.image_paths[0]).read_bytes()
    assert content[0]["text"].startswith(official_prompt(p))
    assert "Task reasoning Skill:" not in content[0]["text"]
    assert "two supplied images" not in content[0]["text"]
    with pytest.raises(ValueError, match="extra_evidence"):
        sr.solve_one({**spec, "prepared_root": "/unused"}, client)


def test_skill_executes_relative_depth_tool_then_explicit_final_answer(tmp_path):
    p = problem(tmp_path)
    _, config, _ = sr.configuration()
    code = (
        "def solve(ctx):\n"
        "    r = ctx.tools.point_depth(view_index=0, x=7, y=6)\n"
        "    if r['status'] == 'ok':\n"
        "        YieldObservations([r['result_id']])"
    )
    client = FakeClient([code, code, '{"reasoning":"fake-only fixture","answer":"B"}'])
    result = SmallImageSolver(
        client, config=config, skill_text=sr.SKILL.read_text(),
        registry_builder=sr.scoped_registry, official_prompt=official_prompt(p)
    ).solve(p, prepared=prepared(p), categories=("cat", "car"))
    assert result.status == "answered" and result.answer == "B"
    assert result.prompt_version == VERSION
    assert result.request_count == 3 and result.category_requests == 0
    assert result.succeeded_tool_calls == 2 and result.denied_authorizations == 0
    for messages, kwargs in client.calls:
        text = messages[0]["content"][0]["text"]
        assert official_prompt(p) in text and "Task reasoning Skill:" in text
        assert "BOXES" not in text and "static scene" not in text
        assert kwargs["enable_thinking"] is True and kwargs["seed"] == 137
        raw = next(x for x in messages[0]["content"] if x["type"] == "image_url")
        assert base64.b64decode(raw["image_url"]["url"].split(",", 1)[1]) == Path(p.image_paths[0]).read_bytes()
    assert '"source_tool": "point_depth"' in client.calls[1][0][0]["content"][0]["text"]


def test_service_error_is_not_retried_and_wrong_task_has_zero_calls(tmp_path):
    p = problem(tmp_path)
    _, config, _ = sr.configuration()
    client = FakeClient([RuntimeError("FAKE_ONLY_ERROR"), '{"answer":"A"}'])
    result = SmallImageSolver(client, config=config, official_prompt=official_prompt(p)).solve(
        p, prepared=prepared(p), categories=())
    assert result.status == "service_error" and len(client.calls) == 1
    from skill3d.small_image.datasets import SmallImageProblem as BaseProblem
    other = BaseProblem(
        benchmark="blink", source_split="test", sample_id="wrong", task_type="Multi-view_Reasoning",
        question=p.question, options=p.options, image_paths=p.image_paths)
    client = FakeClient([])
    result = SmallImageSolver(client, config=config).solve(other, prepared=prepared(other), categories=())
    assert result.status == "input_error" and not client.calls
    with pytest.raises(ValueError):
        validate_official_prompt(p, official_prompt(p).replace("(A) yes", "(A) no"))


@pytest.fixture
def dataset(tmp_path):
    """Full denominator with tiny fake pixels; answer text is host-only."""
    root = tmp_path / "dataset"
    image_root = root / "data" / "images"
    image_root.mkdir(parents=True)
    image = image_root / "original.png"
    Image.fromarray(np.full((12, 16, 3), 90, dtype=np.uint8)).save(image)
    rows, references = [], []
    for split in ("val", "test"):
        for i in range(143):
            p = SmallImageProblem(
                benchmark="blink", sample_id=f"{split}:{split}_Spatial_Relation_{i + 1}",
                source_split=split, question="Is the cat behind the car?", options=("yes", "no"),
                image_paths=(str(image),), task_type="Spatial_Relation")
            identity = image_identities(p)[0]
            rows.append({
                "sample_id": p.sample_id, "idx": p.sample_id.split(":")[1], "split": split,
                "subset": "Spatial_Relation", "row_index": i, "sub_task": "Spatial Relation",
                "question": p.question, "choices": ["yes", "no"],
                "options": [{"label": "A", "text": "yes", "original": "yes"},
                            {"label": "B", "text": "no", "original": "no"}],
                "benchmark_prompt": official_prompt(p),
                "images": [{"field": "image_1", "path": str(image.relative_to(root)),
                            "size_bytes": image.stat().st_size,
                            **{k: v for k, v in identity.items() if k != "view_index"}}],
                "reference_answer_visible": False,
            })
            references.append({"sample_id": p.sample_id, "split": split,
                               "gt_available": split == "val",
                               "gt_answer": "A" if split == "val" else None,
                               "gt_explanation": "HOST_ONLY_SENTINEL"})
    for name, records in (("inputs", rows), ("references", references)):
        (root / "data" / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records))
    manifest = {
        "subset": "Spatial_Relation", "repository": "BLINK-Benchmark/BLINK",
        "resolved_revision": "fake-revision", "input_index_sha256": sr_data.file_sha(root / "data/inputs.jsonl"),
        "reference_index_sha256": sr_data.file_sha(root / "data/references.jsonl"),
        "splits": {"val": {"gt_available": 143, "gt_hidden": 0},
                   "test": {"gt_available": 0, "gt_hidden": 143}},
    }
    (root / "data" / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_full_offline_preflight_never_opens_references_or_calls_models(dataset, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("offline check accessed labels or model")
    monkeypatch.setattr(sr, "load_references", forbidden)
    monkeypatch.setattr(sr.BailianClient, "chat", forbidden)
    (dataset / "data/references.jsonl").unlink()
    result = sr.check(dataset, ("val", "test"))
    assert result["questions"] == 286 and result["model_requests"] == 0
    assert result["reference_answers_read"] is False and result["skill_tools_ready"] is False
    assert "HOST_ONLY_SENTINEL" not in json.dumps(result)
    assert {r["problem"]["source_split"] for r in result["inputs"]} == {"val", "test"}
    assert all(r["direct_payload_sha256"] != r["skill_initial_payload_sha256"] for r in result["inputs"])


def test_public_label_injection_and_original_image_changes_fail(dataset):
    path = dataset / "data/inputs.jsonl"
    rows = sr_data.read_rows(path)
    rows[0]["answer"] = "A"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest_path = dataset / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_index_sha256"] = sr_data.file_sha(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nonpublic"):
        sr_data.load_inputs(dataset)
    rows[0].pop("answer")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest["input_index_sha256"] = sr_data.file_sha(path)
    manifest_path.write_text(json.dumps(manifest))
    (dataset / "data/images/original.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image_changed"):
        sr_data.load_inputs(dataset)


def test_hidden_test_scoring_has_no_fake_accuracy_and_requires_completed_run(dataset, tmp_path):
    run_root = tmp_path / "run"
    inspection = sr.check(dataset, ("val", "test"))
    jobs = []
    for i, row in enumerate(inspection["inputs"]):
        spec = {"arm": "direct", "problem": row["problem"]}
        job = {"operation_id": f"sr-{i:04d}-direct", "spec": spec}
        jobs.append(job)
        result = {**row["problem"], "status": "answered", "answer": "A",
                  "reference_answer_visible": False}
        sr.write_once(run_root / "episodes" / (job["operation_id"] + ".json"),
                      {"job_sha256": sha_value(job), "result": result})
    manifest = sr.write_once(run_root / "manifest.json", {
        "schema": sr.SCHEMA, "data_root": str(dataset), "inspection": inspection,
        "arms": ["direct"], "jobs": jobs,
    })
    with pytest.raises(FileNotFoundError):
        sr.score(run_root)
    sr.write_once(run_root / "completed.json", {
        "manifest_sha256": manifest["content_sha256"], "episodes": len(jobs)})
    scores = sr.score(run_root)["groups"]
    assert scores["direct/val"]["accuracy"] == 1
    assert scores["direct/val"]["scored"] == 143
    assert scores["direct/test"]["accuracy"] is None
    assert scores["direct/test"]["correct"] is None
    assert scores["direct/test"]["scored"] == 0
    assert scores["direct/test"]["questions"] == scores["direct/test"]["legal_answers"] == 143


def test_prepared_cache_binds_to_sr_input_and_excludes_unrelated_metric_tools(tmp_path):
    p = problem(tmp_path)
    cache_root = tmp_path / "prepared"
    value = prepared(p)
    save_prepared(value, cache_root / p.input_sha256)
    replay = sr.prepared_for(p, cache_root)
    assert replay.artifact_sha256 == value.artifact_sha256
    registry, _ = sr.scoped_registry(replay, p)
    assert registry._model_tools == {"view_candidates", "view_depth_summary", "point_depth"}
    wrong = replace(p, question="Is the car behind the cat?")
    with pytest.raises((OSError, ValueError)):
        sr.prepared_for(wrong, cache_root)
