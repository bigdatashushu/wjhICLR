"""Offline evidence that both thinking arms honor their request and input contracts."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from skill3d.cluster import scene_compute
from skill3d.evaluation.vsi_rgb_inputs import FrozenRGBInputProvider
from skill3d.infra.experiment_stages_v11 import baseline32

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "tests/unit"))
from test_published_fitset_v11 import _published


def controller():
    spec = importlib.util.spec_from_file_location("_test_thinking40_controller",
                                                ROOT / "scripts/run_g1m1_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_system_forces_thinking_before_seeded_payload_audit(tmp_path):
    from skill3d.cluster.perception_client import RemoteBailianClient
    from skill3d.evolution.fitset_stream_v11 import SeededGateClient

    class Transport(RemoteBailianClient):
        def chat(self, *args, **kwargs):
            self.last_request_identity = {"parameters": kwargs}
            return "answer"

    delegate = Transport(endpoint="http://127.0.0.1:9999", transport_retries=0)
    seeded = SeededGateClient(delegate, 137, tmp_path / "seed.json")
    module = controller()
    client = module.FailClosedClient(seeded, enable_thinking=True)
    assert client.chat([{"role": "user", "content": "Synthetic"}],
                       enable_thinking=False) == "answer"
    assert seeded.requests[0]["request_identity"]["parameters"]["enable_thinking"] is True
    assert delegate.last_request_identity["parameters"]["seed"] == 137
    for _ in range(6):
        client.chat([])
    with pytest.raises(module.DiagnosticOperationalStop):
        client.chat([])
    assert len(seeded.requests) == 7


def test_rgb_provider_succeeds_without_any_geometry_files(tmp_path, monkeypatch):
    case = _published(tmp_path)
    provider = FrozenRGBInputProvider(case.manifest_path, max_scenes=2)
    frame_targets = {case.frozen.manifest["files"][f["source_path"]]
                     for f in case.frozen.manifest["frames"]}
    for target in case.frozen.manifest["files"].values():
        if target not in frame_targets:
            (case.frozen.root / target).unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError("geometry loader called by RGB arm")
    monkeypatch.setattr(scene_compute, "_load_scene", forbidden)
    opened = []
    original_read = Path.read_bytes
    def record(path):
        opened.append(path)
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", record)
    item = provider(case.variant)
    assert item.geometry is None and item.episode.ground_truth == ""
    assert len(item.pixels) == 32
    assert [int(p[0, 0, 0]) for p in item.pixels] == list(range(32))
    assert all(p.suffix not in {".npy", ".npz"} for p in opened)
    assert provider(case.variant).episode is not item.episode
    frame = case.frozen.root / next(iter(frame_targets))
    frame.write_bytes(b"changed")
    fresh = FrozenRGBInputProvider(case.manifest_path)
    with pytest.raises(ValueError, match="PNG changed"):
        fresh(case.variant)


def test_direct_actual_thinking_one_request_per_question(tmp_path, monkeypatch):
    import skill3d.infra.experiment_stages_v11 as stages
    from skill3d.online.config import load_config, online_model_from, resolve_runtime

    case = _published(tmp_path)
    qa = case.variant["qa_id"]
    item = case.frozen.items[0]
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(json.dumps({
        "qa_id": qa, "scene_name": item.episode.scene_name,
        "question_type": item.episode.question_type, "question": item.episode.question,
        "options": item.episode.options, "answer": "A"}) + "\n")
    config = tmp_path / "online.yaml"
    config.write_text("online_model:\n  base_url: https://offline.invalid/v1\n"
                      "  api_key_env: OFFLINE_TEST_API_KEY\n", encoding="utf-8")
    model = online_model_from(resolve_runtime(load_config(config), seed=137))
    calls = []
    class Client:
        configured = True
        transport_retries = 0
        last_usage = {"total_tokens": 3}
        last_generation = {}
        last_response_identity = {"model": model.model}
        def audit_identity(self):
            return {"model": model.model}
        def chat(self, messages, **kwargs):
            calls.append(kwargs)
            self.last_request_identity = {"parameters": kwargs}
            return "A"
    client = Client()
    monkeypatch.setattr(stages, "fitset_cli", lambda: SimpleNamespace(
        _RevisionClientFactory=lambda *args: lambda: client))
    manifest = {
        "run_root": str(tmp_path / "run"), "identity_sha256": "a" * 64,
        "profile": {
            "inputs": {"metadata": str(metadata), "published_inputs": str(case.manifest_path),
                       "online_config": str(config)},
            "runtime": {"seed": 137, "scene_cache_size": 2, "max_requests": 1,
                        "request_concurrency": 1, "requests_per_second": 0},
            "baseline": {"workers": 1},
        },
    }
    destination = tmp_path / "run/direct"
    baseline32(manifest, destination, variants=[case.variant], enable_thinking=True, rgb_only=True)
    response = json.loads(next((destination / "responses").glob("*.json")).read_text())
    assert len(calls) == 1 and calls[0]["enable_thinking"] is True
    assert response["actual_request_identity"]["parameters"]["enable_thinking"] is True
    assert response["tool_calls"] == 0 and response["reference_answer_visible"] is False
    baseline32(manifest, destination, variants=[case.variant], enable_thinking=True, rgb_only=True)
    assert len(calls) == 1


@pytest.mark.parametrize("subtype", ["object_rel_direction_hard", "object_rel_direction_medium"])
def test_rgb_provider_preserves_raw_direction_subtype(tmp_path, subtype):
    case = _published(tmp_path)
    path = case.frozen.root / "scene.json"
    scene = json.loads(path.read_text())
    for row in scene["questions"]:
        row["episode"]["question_type"] = subtype
    path.write_text(json.dumps(scene) + "\n")
    catalog = json.loads(case.manifest_path.read_text())
    for entry in catalog["entries"].values():
        entry["frozen_scene_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    case.manifest_path.write_text(json.dumps(catalog) + "\n")
    variant = {**case.variant, "question_type": "object_rel_direction"}
    item = FrozenRGBInputProvider(case.manifest_path)(variant)
    assert item.episode.question_type == subtype
    assert item.episode.ground_truth == "" and item.geometry is None
    assert len(item.pixels) == 32
