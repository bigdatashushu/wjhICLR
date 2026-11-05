from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import base64
import copy
import hashlib
import json
import io
import re
import subprocess
import sys
import threading

import numpy as np
from PIL import Image
import pytest

from skill3d.small_image.precompute import _prepared
from skill3d.small_image.storage import save_prepared
from skill3d.small_image.tools import build_registry as shared_registry
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig
from skill3d.synthesis.bailian_client import BailianClient

from MindcubeRunner.budget import LEDGERS, DualBudget, audit
from MindcubeRunner.common import Config, digest, read, sha, validate_preflight, write
from MindcubeRunner.context_stage import guidance
from MindcubeRunner.data import PUBLIC, digest_bytes, load, pair_identity, problem, select
from MindcubeRunner.depth_input import depth_preview
from MindcubeRunner.observation_context import build_observation_context, restore_observations
from MindcubeRunner.program import parse_json, parse_program
from MindcubeRunner.protocol import candidate_query_context, messages
from MindcubeRunner.run import preflight
from MindcubeRunner.score import load_answers, summarize
from MindcubeRunner.tools import ALLOWED, build_registry
from MindcubeRunner.transport import ProxyOpener, SCHEMA, validate_gateway
from MindcubeRunner.worker import solve


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    manifest, catalog, paths = [], [], {}
    for view in range(4):
        key = digest({"view": view})
        rgb = np.full((40, 48, 3), (view + 1) * 40, dtype=np.uint8)
        image_path = root / f"view{view}.png"
        Image.fromarray(rgb).save(image_path)
        public = {
            "benchmark": "MindCube", "source_split": "perview", "sample_id": key,
            "question": "POOLED OTHER QUESTION chair bed television",
            "image_paths": [str(image_path)], "task_type": "perview", "options": [], "input_sha256": key,
        }
        yy, xx = np.indices(rgb.shape[:2])
        # Smooth interiors for protocol tests; sharp transitions have their own fixture.
        depth = 10 + view * 10 + xx / 10 + yy / 20
        points = np.stack([xx / 10, yy / 10, depth], axis=-1)
        mask = np.ones(depth.shape, dtype=bool)
        candidates = tuple({
            "candidate_id": f"pooled-secret-{name}", "view_index": 0,
            "category": name, "box_xyxy": [4, 4, 30, 30],
            "category_confirmed": True, "review_status": "confirmed",
            "mask": mask, "raw_review": "SHOULD_NEVER_REACH_MODEL",
        } for name in ("chair", "bed", "television"))
        method = {"version": "offline-fixture/1", "model": "no-model"}
        prepared = _prepared(
            problem=public, frames=(rgb,), candidates=candidates,
            depths={0: depth}, points={0: points}, intrinsics={},
            geometry_masks={0: mask},
            geometries={0: {"relative_only": True}},
            categories=("chair", "bed", "television"),
            provenance={**({"method_identity": method} if view else {}),
                        "private": "UNION_QUESTION", "labels_used": False},
            capabilities={"relative_depth": True, "object_mask_geometry": False,
                          "metric_geometry": False, "world_geometry": False,
                          "cross_view_identity": False},
            cache_key=key, normalized_image_paths=(), errors=(),
        )
        directory = root / f"prepared{view}"
        save_prepared(prepared, directory)
        manifest.append({**public, "image_sha256": sha(image_path)})
        catalog.append({"input_sha256": key, "image_sha256": sha(image_path),
                        "prepared_directory": str(directory), "method_identity": method,
                        "artifact_sha256": prepared.artifact_sha256})
        paths[key] = str(image_path)
    keys = [row["input_sha256"] for row in manifest]
    questions = []
    for index, order in enumerate(([keys[2], keys[0]], [keys[3], keys[1], keys[2], keys[0]])):
        question = ("Where is the chair relative to the other chair? A. left B. right" if index == 0
                    else "Which chair is closest? A. front B. back C. bed D. above")
        questions.append({
            "benchmark": "MindCube", "source_split": "tinybench", "sample_id": f"q{index}",
            "question": question, "image_keys": list(order), "task_type": str(index),
            "options": [], "input_sha256": digest({"question": question, "images": order}),
        })
    # Another split must never be selected.
    other = {**questions[0], "source_split": "train", "sample_id": "train"}
    write(root / "manifest.json", {"images": manifest})
    write(root / "bindings.json", {"questions": [*questions, other]})
    write(root / "prepared_catalog.json", {"images": catalog, "question_bindings": [*questions, other]})
    write(root / "source_paths.json", paths)
    pins = {n: sha(root / n) for n in ("manifest.json", "bindings.json", "prepared_catalog.json")}
    return root, pins, select(root, pins=pins)


@pytest.fixture
def step_cache(snapshot):
    _, _, items = snapshot
    cache = load(items[0])
    for view, frame in enumerate(cache.frames):
        yy, xx = np.indices(frame.shape[:2])
        depth = np.where(xx < 16, 10.0, 22.0)
        cache.geometries[view] = {
            **cache.geometries[view], "depth": depth,
            "points": np.stack([xx / 10, yy / 10, depth], axis=-1),
            "valid_mask": np.ones(depth.shape, dtype=bool),
        }
    return items[0], cache


class FakeClient:
    def __init__(self, outputs):
        self.outputs, self.calls = iter(outputs), []
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        self.last_response_identity = {"model": "qwen3.8-flash", "model_reported": True}

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        value = next(self.outputs)
        if isinstance(value, BaseException):
            raise value
        return value


class FakeBudget:
    def __init__(self):
        self.receipts = []

    @contextmanager
    def request(self):
        receipt = {"usage": {}}
        self.receipts.append(receipt)
        yield receipt


def program(body):
    return "```python\ndef solve(ctx):\n" + "\n".join("    " + line for line in body.splitlines()) + "\n```"


def episode(tmp_path, item, outputs, *, config=None, arm="skill", cache=None):
    config = config or Config()
    cache = load(item, prepared=arm == "skill") if cache is None else cache
    directory = tmp_path / "episode"
    directory.mkdir()
    job = {"item": item, "arm": arm, "output_directory": str(directory),
           "stop_file": str(tmp_path / "STOP.json"), "config": config.public(), "run_identity": "test",
           "initial_receipt": messages(item, cache, config=config, arm=arm)[1]}
    client, budget = FakeClient(outputs), FakeBudget()
    result = solve(job, client, budget)
    return result, client, budget, directory


def content(client, index):
    return client.calls[index][0][0]["content"]


def text(client, index):
    return "\n".join(part["text"] for part in content(client, index) if part["type"] == "text")


def context_observations(client, index):
    line = next(line for line in text(client, index).splitlines()
                if line.startswith("TOOL RESULTS "))
    return restore_observations(json.loads(line[len("TOOL RESULTS "):]))


def context_feedback(client, index):
    line = next(line for line in text(client, index).splitlines()
                if line.startswith("RECENT FEEDBACK "))
    return json.loads(line[len("RECENT FEEDBACK "):])


def test_select_tiny_order_and_independent_geometry(snapshot):
    root, pins, items = snapshot
    assert [row["sample_id"] for row in items] == ["q0", "q1"]
    assert select(root, ["q1", "q0"], pins=pins)[0]["sample_id"] == "q1"
    with pytest.raises(ValueError):
        select(root, ["train"], pins=pins)
    cache = load(items[0])
    assert len(cache.frames) == 2
    assert cache.frames[0][0, 0, 0] == 120
    assert cache.frames[1][0, 0, 0] == 40
    assert cache.geometries[0]["depth"][10, 10] == 31.5
    assert cache.geometries[1]["depth"][10, 10] == 11.5
    with pytest.raises(ValueError):
        cache.frames[0].setflags(write=True)
    registry, scene, counters = build_registry(cache, problem(items[0]), Config())
    assert set(registry.model_names()) == ALLOWED
    result = registry.call_tool("point_depth", {"view_index": 0, "x": 10, "y": 10}, scene)
    assert result.payload["depth"] == 31.5
    result = registry.call_tool(
        "pair_point_distance", {"view_index": 1, "a_xy": [10, 10], "b_xy": [20, 10]}, scene)
    assert result.payload["distance"] == pytest.approx(np.sqrt(2))
    assert counters["measurements"] == 2


def test_original_bytes_and_question_only_in_context(snapshot):
    _, _, items = snapshot
    for item in items:
        cache = load(item)
        context, receipt = messages(item, cache, arm="skill", config=Config())
        parts = context[0]["content"]
        assert parts[0]["text"] == item["question"]
        count = len(item["views"])
        assert receipt["image_count"] == 2 * count
        assert receipt["rgb_image_count"] == receipt["depth_image_count"] == count
        urls = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
        raws = [base64.b64decode(u.split(",", 1)[1]) for u in urls]
        assert raws[::2] == list(cache.rgb)
        assert receipt["image_sha256"] == [digest_bytes(raw) for raw in raws]
        assert receipt["rgb_image_sha256"] == receipt["image_sha256"][::2]
        for index, frame in enumerate(cache.frames):
            h, w = frame.shape[:2]
            expected, caption, metadata = depth_preview(
                cache.geometries[index], view_index=index, width=w, height=h)
            assert raws[2 * index + 1] == expected
            assert receipt["depth_previews"][index] == metadata
            assert parts[2 + 4 * index]["text"].startswith(f"Original Image {index + 1}:")
            assert parts[4 + 4 * index]["text"] == caption
            assert metadata["view_index"] == index
            assert metadata["display_min"] > cache.geometries[index]["depth"].min()
            with Image.open(io.BytesIO(raws[2 * index + 1])) as depth_image:
                assert depth_image.format == "PNG" and depth_image.size == (w, h)
        assert messages(item, load(item), arm="skill", config=Config()) == (context, receipt)
        serialized = json.dumps(context)
        assert "POOLED OTHER QUESTION" not in serialized
        assert "SHOULD_NEVER_REACH_MODEL" not in serialized
        assert "UNION_QUESTION" not in serialized


def test_direct_context_retains_exact_original_rgb_only(snapshot):
    _, _, items = snapshot
    for item in items:
        cache = load(item, prepared=False)
        context, receipt = messages(item, cache, arm="direct", config=Config())
        expected = [
            {"type": "text", "text": item["question"]},
            {"type": "text", "text": f"Valid option letters: {', '.join(item['labels'])}."
             '\nTools are closed. Output only {"answer":"A"} with a valid current letter.'},
        ]
        for index, (raw, mime, frame) in enumerate(zip(cache.rgb, cache.mime, cache.frames)):
            h, w = frame.shape[:2]
            expected.extend([
                {"type": "text", "text": f"Original Image {index + 1}: view_index={index}, "
                 f"grid {w}x{h}, x right and y down."},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")}},
            ])
        assert context == [{"role": "user", "content": expected}]
        assert receipt["messages_sha256"] == digest(context)
        assert receipt["image_count"] == receipt["rgb_image_count"] == len(cache.rgb)
        assert receipt["depth_image_count"] == 0 and receipt["depth_previews"] == []
        assert receipt["image_sha256"] == receipt["rgb_image_sha256"]


def test_missing_or_misaligned_depth_is_rejected_before_request(snapshot):
    _, _, items = snapshot
    cache = load(items[0], prepared=False)
    with pytest.raises(ValueError, match="depth_preview_view_binding_changed"):
        messages(items[0], cache, arm="skill", config=Config())
    cache = load(items[0])
    cache.geometries[0]["depth"] = cache.geometries[0]["depth"][:-1]
    with pytest.raises(ValueError, match="depth_preview_grid_changed"):
        messages(items[0], cache, arm="skill", config=Config())


def test_candidate_filter_retains_stem_and_option_references(snapshot):
    _, _, items = snapshot
    cache = load(items[1])
    registry, scene, _ = build_registry(cache, problem(items[1]), Config())
    result = registry.call_tool("view_candidates", {}, scene)
    assert {c["category"] for c in result.payload["candidates"]} == {"chair", "bed"}
    assert "television" not in {c["category"] for c in scene.candidates}
    assert all("mask" not in c and "raw_review" not in c for c in scene.candidates)
    assert all(not c["geometry_usable"] for c in scene.candidates)
    # Pruning one instance does not alter shared defaults.
    other, _ = shared_registry(cache, problem(items[1]))
    assert "candidate_depth_summary" in other.names()
    assert "candidate_depth_summary" not in registry.names()


def test_candidate_filter_excludes_unconfirmed_and_missing_confirmation(snapshot):
    _, _, items = snapshot
    cache = load(items[1])
    for candidate in cache.candidates:
        if candidate["view_index"] == 0 and candidate["category"] == "chair":
            candidate.update(category_confirmed=False, review_status="not_established")
        elif candidate["view_index"] == 0 and candidate["category"] == "bed":
            candidate.pop("category_confirmed")
    registry, scene, _ = build_registry(cache, problem(items[1]), Config())
    result = registry.call_tool("view_candidates", {}, scene)
    assert result.payload["count"] == 6
    assert all(c["category_confirmed"] is True for c in result.payload["candidates"])
    assert {c["category"] for c in result.payload["candidates"]} == {"chair", "bed"}
    assert registry.call_tool("view_candidates", {"view_index": 0}, scene).payload["candidates"] == []


def test_edge_sampling_returns_reason_without_depth(step_cache):
    item, cache = step_cache
    registry, scene, counters = build_registry(cache, problem(item), Config())
    result = registry.call_tool("point_depth", {"view_index": 1, "x": 15, "y": 12}, scene)
    assert result.status == "ok"
    assert result.payload["status"] == "unreliable"
    assert result.payload["reason_code"] == "edge_sampling"
    assert "interior surface" in result.payload["reason"]
    assert "depth" not in result.payload
    assert result.payload["sample_stability"]["point"]["relative_depth_range"] > 0.10
    assert counters["calls"] == counters["measurements"] == 1
    # The shared registry keeps its original per-pixel measurement behaviour.
    other, other_scene = shared_registry(cache, problem(item))
    assert other.call_tool("point_depth", {"view_index": 1, "x": 15, "y": 12},
                           other_scene).payload["depth"] == 10


@pytest.mark.parametrize("endpoint", ["a", "b"])
def test_distance_checks_each_endpoint_and_omits_unreliable_value(step_cache, endpoint):
    item, cache = step_cache
    registry, scene, counters = build_registry(cache, problem(item), Config())
    args = {"view_index": 1, "a_xy": [8, 12], "b_xy": [30, 12]}
    args[endpoint + "_xy"] = [15, 12]
    result = registry.call_tool("pair_point_distance", args, scene)
    assert result.payload["status"] == "unreliable"
    assert result.payload["reason_code"] == "edge_sampling"
    assert result.payload["unreliable_endpoints"] == [endpoint]
    assert "distance" not in result.payload
    assert counters["calls"] == counters["measurements"] == 1


def test_missing_nearby_depth_returns_local_quality_notice(step_cache):
    item, cache = step_cache
    mask = np.zeros(cache.frames[1].shape[:2], dtype=bool)
    mask[7:10, 7:10] = True
    cache.geometries[1]["valid_mask"] = mask
    registry, scene, _ = build_registry(cache, problem(item), Config())
    result = registry.call_tool("point_depth", {"view_index": 1, "x": 8, "y": 8}, scene)
    assert result.payload["status"] == "unreliable"
    assert result.payload["reason_code"] == "insufficient_local_depth"
    assert "depth" not in result.payload


@pytest.mark.parametrize("resample", [False, True])
def test_unreliable_notice_reaches_next_request_and_can_be_resampled(
        step_cache, tmp_path, monkeypatch, resample):
    item, cache = step_cache
    monkeypatch.setattr("MindcubeRunner.worker.load", lambda *args, **kwargs: cache)
    outputs = [program('r = ctx.tools.point_depth(view_index=1, x=15, y=12)\nReturnAnswer("A")')]
    if resample:
        outputs.append(program('r = ctx.tools.point_depth(view_index=1, x=8, y=12)\nReturnAnswer("A")'))
    outputs.append(program('ReturnAnswer("B")'))
    result, client, _, _ = episode(tmp_path, item, outputs, cache=cache)
    first = result["rounds"][0]
    assert first["status"] == "measurement_review_required"
    assert first["unreliable_measurement_ids"] == first["new_measurement_ids"]
    assert first["tool_observations"][0]["data"]["reason_code"] == "edge_sampling"
    assert "depth" not in first["tool_observations"][0]["data"]
    assert '"status":"unreliable"' in text(client, 1)
    assert '"reason_code":"edge_sampling"' in text(client, 1)
    assert result["answer"] == "B"
    assert result["request_count"] == (3 if resample else 2)
    assert result["tool_counts"]["measurements"] == (2 if resample else 1)
    assert result["basis"] == ("mixed" if resample else "visual_estimate")
    if resample:
        assert result["rounds"][1]["tool_observations"][0]["data"]["depth"] == 10


@pytest.mark.parametrize("mutation", ["arrays", "rgb", "manifest", "method", "order"])
def test_input_tampering_fails_closed(snapshot, mutation):
    root, pins, items = snapshot
    item = copy.deepcopy(items[0])
    if mutation == "arrays":
        path = Path(item["views"][0]["prepared_directory"]) / "arrays.npz"
        path.write_bytes(path.read_bytes() + b"tampered")
    elif mutation == "rgb":
        Path(item["views"][0]["image_path"]).write_bytes(b"bad-image")
    elif mutation == "manifest":
        item["views"][0]["image_public"]["question"] = "changed"
    elif mutation == "method":
        item["views"][0]["method_identity"] = {"version": "changed"}
    else:
        item["views"].reverse()
    with pytest.raises(ValueError):
        load(item)


def test_measurement_answer_is_provisional_and_delivered_next_request(snapshot, tmp_path):
    _, _, items = snapshot
    result, client, budget, _ = episode(tmp_path, items[0], [
        program('r = ctx.tools.point_depth(view_index=1, x=10, y=10)\nReturnAnswer("A")'),
        program('ReturnAnswer("B")')])
    assert result["answer"] == "B" and result["request_count"] == 2
    row = result["rounds"][0]
    assert row["status"] == "measurement_review_required" and row["provisional_answer"] == "A"
    assert result["rounds"][1]["request"]["reviewed_measurement_ids"] == row["new_measurement_ids"]
    assert '"pixel_xy":[10,10]' in text(client, 1)
    assert '"view_index":1' in text(client, 1)
    assert len(budget.receipts) == 2 and budget.receipts[0]["usage"]["total_tokens"] == 12
    first_images = [p for p in content(client, 0) if p["type"] == "image_url"]
    assert len(first_images) == 2 * len(items[0]["views"])
    assert [p for p in content(client, 1) if p["type"] == "image_url"] == first_images
    assert result["rounds"][1]["request"]["depth_previews"] == row["request"]["depth_previews"]
    assert "PNG 48x40" in text(client, 0) and "x=0..47" in text(client, 1)


def test_fifth_measurement_review_in_sixth_final(snapshot, tmp_path):
    _, _, items = snapshot
    outputs = ["<xml>wrong format</xml>"] * 4 + [
        program('r = ctx.tools.point_depth(view_index=0, x=10, y=10)\nReturnAnswer("A")'),
        '{"answer":"B"}']
    result, client, _, _ = episode(tmp_path, items[0], outputs)
    assert result["request_count"] == 6 and result["answer"] == "B"
    assert result["rounds"][-1]["phase"] == "final"
    assert result["rounds"][-1]["request"]["reviewed_measurement_ids"]
    assert "Tools are closed" in text(client, 5)
    for row in result["rounds"]:
        assert row["request"]["depth_image_count"] == len(items[0]["views"])
        assert row["request"]["image_sha256"] == result["rounds"][0]["request"]["image_sha256"]


def test_new_candidates_are_inspected_before_next_measurement(snapshot, tmp_path):
    _, _, items = snapshot
    result, client, _, _ = episode(tmp_path, items[0], [
        program('r = ctx.tools.view_candidates(view_index=0)\n'
                'YieldObservations([r["result_id"]], reason="Inspect candidate locations")'),
        program('r = ctx.tools.point_depth(view_index=0, x=10, y=10)\n'
                'YieldObservations([r["result_id"]], reason="Review selected target pixel")'),
        program('ReturnAnswer("B")')])
    first, second, third = result["rounds"]
    assert second["request"]["context_actions"] == ["inspect_hints"]
    assert second["request"]["hint_result_ids"] == first["new_hint_ids"]
    assert third["request"]["context_actions"] == ["review_measurements"]
    assert third["request"]["hint_result_ids"] == []
    assert context_observations(client, 1) == first["tool_observations"]
    assert context_observations(client, 2) == [
        *first["tool_observations"], *second["tool_observations"]]


def test_prompt_tool_example_yields_its_actual_result_to_next_request(snapshot, tmp_path):
    examples = re.findall(r"```python\n([\s\S]*?)\n```",
                          guidance(["plan_or_measure"], final=False))
    source = next(example for example in examples if "ctx.tools.view_candidates(" in example)
    result, client, _, _ = episode(tmp_path, snapshot[2][0], [
        "```python\n" + source + "\n```", program('ReturnAnswer("B")')])
    first, second = result["rounds"]
    assert first["status"] == "yielded"
    assert result["status"] == "answered" and result["request_count"] == 2
    assert result["tool_counts"] == {"calls": 1, "measurements": 0}
    observation = first["tool_observations"][0]
    assert observation["source_tool"] == "view_candidates"
    assert observation["data"]["candidates"]
    assert first["cell"]["yielded_result_ids"] == [observation["result_id"]]
    assert second["request"]["hint_result_ids"] == [observation["result_id"]]
    assert second["request"]["context_actions"] == ["inspect_hints"]
    assert context_observations(client, 1) == [observation]


@pytest.mark.parametrize("arguments,view_index,category", [
    ("", -1, ""),
    ('view_index=1, category="unexposed-category"', 1, "unexposed-category"),
    ("view_index=0", 0, ""),
])
def test_empty_query_scope_survives_unexecuted_repairs_and_final(
        snapshot, tmp_path, monkeypatch, arguments, view_index, category):
    item = snapshot[2][0]
    cache = load(item)
    for candidate in cache.candidates:
        candidate["category_confirmed"] = False
    monkeypatch.setattr("MindcubeRunner.worker.load", lambda *args, **kwargs: cache)
    first = program(f"r = ctx.tools.view_candidates({arguments})\n"
                    'YieldObservations([r["result_id"]], reason="Inspect locations")')
    rejected = "Explanation outside the fence\n" + program(
        'r = ctx.tools.view_candidates(view_index=1, category="proposed-only")\n'
        'YieldObservations([r["result_id"]])')
    result, client, _, directory = episode(
        tmp_path, item, [first, *([rejected] * 4), '{"answer":"B"}'], cache=cache)
    assert result["status"] == "answered" and result["request_count"] == 6
    assert result["tool_counts"] == {"calls": 1, "measurements": 0}
    observation = result["rounds"][0]["tool_observations"][0]
    assert observation["data"]["candidates"] == [] and observation["data"]["count"] == 0
    expected = [{"result_id": observation["result_id"],
                 "view_index": view_index, "category": category}]
    first_trace = read(directory / "traces/00/trace.json")
    executed = next(e["data"]["results"][0] for e in first_trace["events"]
                    if e["kind"] == "tool_finished")
    assert executed["result_id"] == expected[0]["result_id"]
    assert executed["args"].get("view_index", -1) == view_index
    assert executed["args"].get("category", "") == category
    assert "EXECUTED CANDIDATE QUERIES [" not in text(client, 0)
    for index in range(1, 6):
        history = next(line.removeprefix("EXECUTED CANDIDATE QUERIES ")
                       for line in text(client, index).splitlines()
                       if line.startswith("EXECUTED CANDIDATE QUERIES "))
        assert json.loads(history) == expected
        assert context_observations(client, index) == [observation]
        assert result["rounds"][index]["request"]["candidate_query_history_sha256"] == digest(expected)
        assert result["rounds"][index]["request"]["candidate_query_count"] == 1
        trace = read(directory / "traces" / f"{index:02d}" / "trace.json")
        state = next(e["data"] for e in trace["events"] if e["kind"] == "state_before")
        assert state["candidate_queries"] == expected
        assert not any(e["kind"] == "tool_started" for e in trace["events"])
    assert all(row["status"] == "program_parse_error" for row in result["rounds"][1:5])
    assert result["rounds"][-1]["phase"] == "final"
    assert 'category="proposed-only"' not in text(client, 5)


@pytest.mark.parametrize("change", [
    {"result_id": "no-such-result"}, {"view_index": True}, {"view_index": 2},
    {"category": ["not-a-string"]}, {"extra": "not-public"},
])
def test_candidate_query_history_rejects_unbound_or_malformed_records(change):
    observations = [{"result_id": "query-1", "source_tool": "view_candidates",
                     "status": "ok", "data": {"candidates": [], "count": 0}}]
    query = {"result_id": "query-1", "view_index": -1, "category": "", **change}
    with pytest.raises(ValueError, match="invalid_candidate_query_record"):
        candidate_query_context(observations, [query], view_count=2)


@pytest.mark.parametrize("failure_after_measurement", [False, True])
def test_repair_keeps_measurement_review_and_exact_evidence(
        snapshot, tmp_path, failure_after_measurement):
    _, _, items = snapshot
    if failure_after_measurement:
        outputs = [
            program('r = ctx.tools.point_depth(view_index=0, x=10, y=10)\n'
                    'x = 1 / 0\nYieldObservations([r["result_id"]])'),
            program('ReturnAnswer("B")')]
        review_round = 1
    else:
        outputs = [
            program('r = ctx.tools.point_depth(view_index=0, x=10, y=10)\n'
                    'YieldObservations([r["result_id"]])'),
            "<xml>wrong format</xml>",
            program('ReturnAnswer("B")')]
        review_round = 2
    result, client, _, _ = episode(tmp_path, items[0], outputs)
    first = result["rounds"][0]
    reviewed = result["rounds"][review_round]["request"]
    assert first["new_measurement_ids"]
    assert reviewed["context_actions"] == ["repair", "review_measurements"]
    assert reviewed["reviewed_measurement_ids"] == first["new_measurement_ids"]
    assert context_observations(client, review_round) == first["tool_observations"]
    assert result["answer"] == "B"
    if failure_after_measurement:
        assert context_feedback(client, review_round)[0]["source"] == parse_program(outputs[0])
        assert result["tool_counts"]["measurements"] == 1
    else:
        assert context_feedback(client, review_round)[0]["previous_response"] == outputs[1]


def test_global_prefix_error_delivers_exact_program_and_syntax_location(snapshot, tmp_path):
    failed = program('GLOBAL ReturnAnswer("B")')
    source = failed.removeprefix("```python\n").removesuffix("\n```")
    result, client, _, directory = episode(tmp_path, snapshot[2][0], [
        failed, program('ReturnAnswer("B")')])
    error = result["rounds"][0]["feedback_errors"][0]
    assert error["code"] == "python_syntax_error" and error["type"] == "SyntaxError"
    assert error["line"] == 2 and error["column"] > 0
    assert error["syntax_message"] == "invalid syntax"
    assert error["source_line"] == source.splitlines()[1]
    assert error["source"] == source
    assert error["source_utf8_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert context_feedback(client, 1) == [error]
    assert "Do not put GLOBAL" in error["repair"]
    assert result["request_count"] == 2 and result["answer"] == "B"
    assert result["rounds"][1]["request"]["context_actions"] == ["repair"]
    assert len(client.calls[1][0]) == 1 and client.calls[1][0][0]["role"] == "user"
    assert [p for p in content(client, 1) if p["type"] == "image_url"] == [
        p for p in content(client, 0) if p["type"] == "image_url"]
    first = read(directory / "traces/00/trace.json")
    assert first["events"][-1]["data"]["state_after"]["errors"] == [error]
    assert not any(e["kind"] == "cell_completed" for e in first["events"])


def test_final_after_syntax_errors_excludes_previous_program_data(snapshot, tmp_path):
    failed = program('GLOBAL ReturnAnswer("B")')
    result, client, _, _ = episode(
        tmp_path, snapshot[2][0], [failed] * 5 + ['{"answer":"B"}'])
    assert result["request_count"] == 6 and result["answer"] == "B"
    final_feedback = context_feedback(client, 5)[0]
    assert final_feedback["code"] == "python_syntax_error" and final_feedback["line"] == 2
    assert set(final_feedback) == {"code", "type", "line", "column", "repair"}
    final_text = text(client, 5)
    for forbidden in ("GLOBAL ReturnAnswer", "def solve", "source_utf8_sha256",
                      "syntax_message", "```python"):
        assert forbidden not in final_text


def test_oversized_previous_program_stops_before_another_request(snapshot, tmp_path):
    failed = program("# " + "x" * 24000 + '\nGLOBAL ReturnAnswer("B")')
    result, client, _, directory = episode(tmp_path, snapshot[2][0], [failed])
    assert result["status"] == "context_limit"
    assert result["request_count"] == len(client.calls) == 1
    error = result["rounds"][0]["feedback_errors"][0]
    assert len(error["source"]) > 24000
    second = read(directory / "traces/01/trace.json")
    assert second["events"][-1]["data"]["state_after"]["errors"] == [error]
    assert not any(e["kind"] == "admitted" for e in second["events"])


def test_final_after_fifth_parse_error_has_only_current_json_format(snapshot, tmp_path):
    _, _, items = snapshot
    result, client, _, _ = episode(
        tmp_path, items[0], ["<xml>wrong format</xml>"] * 5 + ['{"answer":"B"}'])
    final = result["rounds"][-1]
    assert result["answer"] == "B" and result["request_count"] == 6
    assert final["request"]["context_actions"] == ["answer_json"]
    final_text = text(client, 5)
    assert "python_fence_required" in final_text
    assert "Tools are closed" in final_text
    assert "```python" not in final_text and "ctx.tools." not in final_text
    assert "Output one closed python fence" not in final_text
    assert "Use point_depth" not in final_text


def test_common_depth_legend_once_and_unavailable_captions_have_no_call_advice(snapshot):
    _, _, items = snapshot
    cache = load(items[1])
    for geometry in cache.geometries.values():
        geometry["valid_mask"] = np.zeros(geometry["depth"].shape, dtype=bool)
    for index in (0, 5):
        context, receipt = messages(items[1], cache, arm="skill", config=Config(), round_index=index)
        parts = context[0]["content"]
        assert receipt["depth_image_count"] == 4
        all_text = "\n".join(p["text"] for p in parts if p["type"] == "text")
        assert all_text.count("DEPTH LEGEND ") == 1
        for view in range(4):
            caption = parts[4 + 4 * view]["text"]
            assert "status=unavailable" in caption and "point_depth" not in caption
        if index == 5:
            assert receipt["context_actions"] == ["answer_json"]
            assert "ctx.tools." not in all_text


def test_pending_context_overflow_is_explicit_and_does_not_truncate(snapshot):
    _, _, items = snapshot
    cache = load(items[0])
    registry, scene, _ = build_registry(cache, problem(items[0]), Config())
    response = registry.call_tool("point_depth", {"view_index": 0, "x": 10, "y": 10}, scene)
    observations = [{
        "result_id": f"pending-{index:032d}", "source_tool": "point_depth",
        "status": "ok", "data": copy.deepcopy(response.payload),
    } for index in range(2000)]
    before = copy.deepcopy(observations)
    ids = [r["result_id"] for r in observations]
    packet, _ = build_observation_context(observations, review_ids=ids)
    assert restore_observations(packet) == observations
    with pytest.raises(ValueError, match="context_limit"):
        messages(items[0], cache, arm="skill", config=Config(max_context_chars=8000),
                 round_index=1, observations=observations, review_ids=ids)
    assert observations == before


def test_wrong_control_namespace_repairs_in_original_slots(snapshot, tmp_path):
    _, _, items = snapshot
    result, client, _, _ = episode(tmp_path, items[0], [
        program('ctx.tools.YieldObservations([])'),
        program('ReturnAnswer("A")')])
    assert result["answer"] == "A" and result["request_count"] == 2
    assert result["rounds"][0]["status"] == "repairable_program_error"
    assert result["rounds"][0]["executed"] is False
    assert context_feedback(client, 1)[0]["source"] == parse_program(
        program('ctx.tools.YieldObservations([])'))


def test_ordinary_runtime_repair_and_answer_slot_reset(snapshot, tmp_path):
    _, _, items = snapshot
    result, client, _, _ = episode(tmp_path, items[0], [
        program('ReturnAnswer("D")'), program('ReturnAnswer("B")')])
    assert result["answer"] == "B"
    assert result["rounds"][0]["cell"]["error_code"] == "violation_runtime"
    assert context_feedback(client, 1)[0]["source"] == parse_program(program('ReturnAnswer("D")'))


def test_unlimited_calls_execute_and_review_across_program_rounds(snapshot, tmp_path):
    _, _, items = snapshot
    config = Config(max_context_chars=100000, cell_timeout_s=5)
    result, client, _, _ = episode(tmp_path, items[0], [
        program('ids = []\n'
                'for i in range(12):\n'
                '    r = ctx.tools.view_depth_summary()\n'
                '    ids.append(r["result_id"])\n'
                'for i in range(12):\n'
                '    r = ctx.tools.point_depth(view_index=0, x=10, y=10)\n'
                '    ids.append(r["result_id"])\n'
                'for i in range(12):\n'
                '    r = ctx.tools.pair_point_distance('
                'view_index=0, a_xy=[10,10], b_xy=[20,10])\n'
                '    ids.append(r["result_id"])\n'
                'YieldObservations(ids, reason="Review all selected measurements")'),
        program('ids = []\n'
                'for i in range(8):\n'
                '    r = ctx.tools.point_depth(view_index=1, x=10, y=10)\n'
                '    ids.append(r["result_id"])\n'
                'YieldObservations(ids, reason="Review additional target samples")'),
        program('ReturnAnswer("B")')], config=config)
    assert result["answer"] == "B" and result["request_count"] == 3
    assert result["tool_counts"] == {"calls": 44, "measurements": 32}
    first, second, third = result["rounds"]
    assert first["status"] == second["status"] == "yielded"
    assert first["tool_counts_after"] == {"calls": 36, "measurements": 24}
    assert len(first["tool_observations"]) == 36
    assert len(first["new_measurement_ids"]) == 24
    assert len(second["new_measurement_ids"]) == 8
    assert second["request"]["reviewed_measurement_ids"] == first["new_measurement_ids"]
    assert third["request"]["reviewed_measurement_ids"] == second["new_measurement_ids"]
    assert all(row["phase"] == "program" for row in result["rounds"])
    assert all("quota_denials" not in row for row in result["rounds"])
    assert '"depth":31.5' in text(client, 1)


def test_call_counts_do_not_select_final_phase(snapshot):
    _, _, items = snapshot
    cache = load(items[0])
    config = Config()
    assert "max_tool_calls" not in config.public()
    assert "max_measurements" not in config.public()
    for index in range(config.request_limit("skill")):
        _, receipt = messages(
            items[0], cache, arm="skill", config=config, round_index=index,
            counters={"calls": 1000000, "measurements": 1000000})
        assert receipt["phase"] == ("final" if index == 5 else "program")
        assert receipt["tools_enabled"] == (index < 5)
        assert "remaining_quota" not in receipt


@pytest.mark.parametrize("body,expected", [
    ('import os\nReturnAnswer("A")', "program_rejected"),
    ('import numpy as np\nx = np.load("host-file")\nReturnAnswer("A")', "program_rejected"),
    ('x = ctx.scene.__dict__\nReturnAnswer("A")', "program_rejected"),
    ('r = ctx.tools.candidate_depth_summary(candidate_id="x")\nReturnAnswer("A")', "program_rejected"),
    ('r = ctx.tools.point_depth(view_index=9, x=10, y=10)\nReturnAnswer("A")', "run_error"),
    ('r = ctx.tools.pair_point_distance(view_index=0, a_xy=[-1,10], b_xy=[10,10])\n'
     'ReturnAnswer("A")', "run_error"),
    ('while True:\n    pass\nReturnAnswer("A")', "run_error"),
])
def test_security_tool_and_timeout_stop_episode(snapshot, tmp_path, body, expected):
    _, _, items = snapshot
    result, client, _, _ = episode(tmp_path, items[0], [program(body)],
                                  config=Config(cell_timeout_s=0.05))
    assert result["status"] == expected and result["answer"] is None
    assert len(client.calls) == 1


def test_service_failure_stops_without_retry_or_exception_text(snapshot, tmp_path):
    _, _, items = snapshot
    result, client, _, directory = episode(tmp_path, items[0], [RuntimeError("PRIVATE SECRET")])
    assert result["status"] == "service_error" and len(client.calls) == 1
    assert (tmp_path / "STOP.json").exists()
    assert "PRIVATE SECRET" not in (directory / "result.json").read_text()


def test_model_identity_deviation_stops(snapshot, tmp_path):
    _, _, items = snapshot
    directory = tmp_path / "episode"
    directory.mkdir()
    config = Config()
    job = {"item": items[0], "arm": "direct", "output_directory": str(directory),
           "stop_file": str(tmp_path / "STOP.json"), "config": config.public(), "run_identity": "test",
           "initial_receipt": messages(items[0], load(items[0]), arm="direct", config=config)[1]}
    client = FakeClient(['{"answer":"A"}'])
    client.last_response_identity = {"model": "wrong-model", "model_reported": True}
    assert solve(job, client, FakeBudget())["status"] == "service_error"
    assert len(client.calls) == 1


@pytest.mark.parametrize("text_value", [
    '{"answer":"C"}', '{"answer":0}', '{"answer":"A","answer":"B"}',
    '{"answer":"A","reason":"because"}', '{"answer":"a"}',
])
def test_strict_final_answer_contract(text_value):
    with pytest.raises((ValueError, TypeError)):
        parse_json(text_value, ["A", "B"])


def test_direct_and_preflight_freezing_zero_requests(snapshot, tmp_path):
    root, pins, items = snapshot
    output = tmp_path / "preflight"
    settings = {"cache_root": str(root), "output": str(output),
                "arms": ["skill", "direct"], "input_pins": pins}
    frozen, _ = preflight(settings)
    assert frozen["max_requests"] == 14
    assert frozen["model_requests"] == frozen["gpu_requests"] == 0
    first = (output / "preflight.json").read_bytes()
    preflight(settings)
    assert (output / "preflight.json").read_bytes() == first
    changed = {**settings, "selection": ["q0"]}
    with pytest.raises(ValueError, match="preflight_identity_changed"):
        preflight(changed)
    result, client, _, _ = episode(tmp_path, items[0], ['{"answer":"A"}'], arm="direct")
    assert result["answer"] == "A" and result["request_count"] == 1
    assert "STATIC SKILL" not in text(client, 0)


def test_cli_preflight_and_score_entry(snapshot, tmp_path):
    root, pins, _ = snapshot
    settings = {"cache_root": str(root), "output": str(tmp_path / "cli-output"), "input_pins": pins}
    path = tmp_path / "config.json"
    write(path, settings)
    script = Path(__file__).resolve().parents[1] / "run.py"
    process = subprocess.run([sys.executable, "-B", str(script), "--config", str(path)],
                             text=True, capture_output=True, check=True)
    report = json.loads(process.stdout)
    assert report["questions"] == 2 and report["model_requests"] == 0
    score = subprocess.run([sys.executable, "-B", str(script.parent / "score.py"), "--help"],
                           text=True, capture_output=True, check=True)
    assert "--answers-sha256" in score.stdout


def test_frozen_preflight_tampering_rejected(snapshot, tmp_path):
    root, pins, _ = snapshot
    frozen, _ = preflight({"cache_root": str(root), "output": str(tmp_path / "frozen"),
                          "input_pins": pins})
    validate_preflight(frozen)
    frozen["items"][0]["question"] = "tampered"
    with pytest.raises(ValueError, match="frozen_preflight_changed"):
        validate_preflight(frozen)


@pytest.mark.parametrize("arm", ["direct", "skill"])
def test_real_client_and_rpc_transport_without_network(snapshot, tmp_path, monkeypatch, arm):
    _, _, items = snapshot
    sent = []

    class OfflineHTTP:
        def open(self, request, timeout):
            payload = json.loads(request.data)
            sent.append(payload)
            assert payload["schema_version"] == SCHEMA
            assert payload["payload"]["model"] == "qwen3.8-flash"
            assert request.get_header("Authorization") == "Bearer dummy-offline-token"
            response = program('ReturnAnswer("B")') if arm == "skill" else '{"answer":"B"}'
            return io.BytesIO(json.dumps({
                "model": "qwen3.8-flash",
                "choices": [{"message": {"role": "assistant", "content": response},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }).encode())

    monkeypatch.setattr("MindcubeRunner.transport.urllib.request.build_opener",
                        lambda *args: OfflineHTTP())
    monkeypatch.setenv("MINDCUBE_OFFLINE_TEST_KEY", "dummy-local-placeholder")
    client = BailianClient(api_key_env="MINDCUBE_OFFLINE_TEST_KEY",
                           opener=ProxyOpener("http://127.0.0.1:1234",
                                              lambda: "dummy-offline-token", "operation", "test"))
    config = Config()
    directory = tmp_path / "rpc-episode"
    directory.mkdir()
    job = {"item": items[0], "arm": arm, "output_directory": str(directory),
           "stop_file": str(tmp_path / "STOP.json"), "config": config.public(), "run_identity": "test",
           "initial_receipt": messages(items[0], load(items[0]), arm=arm, config=config)[1]}
    result = solve(job, client, FakeBudget())
    assert result["status"] == "answered" and result["answer"] == "B"
    assert len(sent) == 1 and result["request_count"] == 1
    assert sent[0]["payload"]["enable_thinking"] is True
    assert sent[0]["payload"]["seed"] == 137
    sent_parts = sent[0]["payload"]["messages"][0]["content"]
    image_urls = [p["image_url"]["url"] for p in sent_parts if p["type"] == "image_url"]
    assert len(image_urls) == len(items[0]["views"]) * (2 if arm == "skill" else 1)
    assert [digest_bytes(base64.b64decode(url.split(",", 1)[1])) for url in image_urls] == (
        result["rounds"][0]["request"]["image_sha256"])
    assert "dummy-offline-token" not in (directory / "result.json").read_text()


def test_scoring_keeps_full_denominator_and_rejects_identity_changes(snapshot, tmp_path):
    _, _, items = snapshot
    result, _, _, _ = episode(tmp_path, items[0], ['{"answer":"B"}'], arm="direct")
    answers = {"q0": "B", "q1": "A"}
    summary = summarize(items, ["direct", "skill"], [result], run_identity="test", answers=answers)
    assert summary["arms"]["direct"]["accuracy"] == 0.5
    assert summary["arms"]["direct"]["statuses"] == {"answered": 1, "not_run": 1}
    assert summary["arms"]["skill"]["denominator"] == 2
    path = tmp_path / "host-only.json"
    write(path, answers)
    assert load_answers(path, items, expected_sha256=sha(path)) == answers
    changed = {**result, "pair_identity": "changed"}
    with pytest.raises(ValueError):
        summarize(items, ["direct"], [changed], run_identity="test", answers=answers)


def test_aggregate_budget_mirrors_and_started_rows(tmp_path):
    for name in LEDGERS:
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        write(directory / "state.json", {"policy": {"max_requests": 20000, "max_inflight": 1,
                                                  "requests_per_second": 0.0},
                                         "requests": []})
    campaign = "other-datasets-thinking-20261101"
    row = {"category": "solver", "context": {"campaign_id": campaign},
           "usage": {"total_tokens": 12}, "status": "completed"}
    primary = tmp_path / "smallimage/requests/state.json"
    state = read(primary)
    state["requests"] = [row, {"category": "solver", "context": {}, "status": "started"}]
    write(primary, state)
    mirror = tmp_path / "cv3d-mmsi-thinking-pilot-20261101/budget/state.json"
    state = read(mirror)
    state["requests"] = [row]
    write(mirror, state)
    accounting = audit(tmp_path)
    assert accounting["raw_count"] == 3 and accounting["excluded_mirror_count"] == 1
    assert accounting["used"] == 2


def test_dual_budget_records_only_sanitized_context(tmp_path):
    base = tmp_path / "base"
    for name in LEDGERS:
        directory = base / name
        directory.mkdir(parents=True, exist_ok=True)
        RequestGovernor(directory, RequestGovernorConfig(20000, 1))
    local = RequestGovernor(tmp_path / "local", RequestGovernorConfig(6, 1))
    shared = RequestGovernor(base / "smallimage/requests", RequestGovernorConfig(20000, 1))
    budget = DualBudget(local, shared, base, tmp_path / "STOP.json",
                        campaign="test", arm="skill", sample_id="q0", run_identity="identity")
    with budget.request() as receipt:
        receipt["usage"] = {"total_tokens": 12}
    assert audit(base)["used"] == 1
    state = read(shared.state_path)
    assert state["requests"][0]["usage"] == {"total_tokens": 12}
    assert set(state["requests"][0]["context"]) == {
        "campaign_id", "variant_id", "slot_key", "run_identity"}
    assert local.summary()["reserved_requests"] == 1


def test_two_budget_requests_can_overlap_after_atomic_admission(tmp_path):
    base = tmp_path / "base"
    for name in LEDGERS:
        directory = base / name
        directory.mkdir(parents=True, exist_ok=True)
        RequestGovernor(directory, RequestGovernorConfig(20000, 4))
    local = RequestGovernor(tmp_path / "local", RequestGovernorConfig(14, 2))
    shared = RequestGovernor(base / "smallimage/requests", RequestGovernorConfig(20000, 4))
    barrier = threading.Barrier(2)

    def request(arm):
        budget = DualBudget(local, shared, base, tmp_path / "STOP.json",
                            campaign="test", arm=arm, sample_id="q0", run_identity="identity")
        with budget.request() as receipt:
            barrier.wait(timeout=3)
            receipt["usage"] = {"total_tokens": 12}

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(request, arm) for arm in ("skill", "direct")]
        for future in futures:
            future.result()
    assert audit(base)["used"] == 2
    assert local.summary()["peak_inflight"] == 2


@pytest.mark.parametrize("gateway", [
    "http://remote.example", "https://user:password@example.org",
    "https://example.org?token=secret",
])
def test_gateway_rejects_unsafe_or_embedded_credentials(gateway):
    with pytest.raises(ValueError):
        validate_gateway(gateway)
