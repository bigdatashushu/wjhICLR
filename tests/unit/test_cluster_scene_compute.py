"""Frozen-scene contracts using real identities/stores, without heavy models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from skill3d.adapters.episode_source import EpisodeItem, _episode_from_row
from skill3d.adapters.frame_set import build_frame_set
from skill3d.cluster.scene_compute import (
    SceneComputeError, compute_scene, validate_published_scene,
    validate_scene_input, validate_scene_output,
)
from skill3d.evaluation.skill_ablation_v11 import input_identity
from skill3d.infra.m5_identity import content_sha256, file_identity
from skill3d.schemas import ObjectRecord, ReconstructionArtifact
from skill3d.storage.relocation import artifact_dependency_refs


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def _geometry(root, fs):
    root.mkdir(parents=True)
    arrays = {
        "depth_maps": np.ones((32, 3, 4), dtype=np.float32),
        "c2w_list": np.repeat(np.eye(4, dtype=np.float32)[None], 32, axis=0),
        "intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None], 32, axis=0),
        "point_map": np.ones((32, 3, 4, 3), dtype=np.float32),
        "point_conf": np.ones((32, 3, 4), dtype=np.float32),
        "depth_conf": np.ones((32, 3, 4), dtype=np.float32),
    }
    refs = {}
    for key, array in arrays.items():
        path = root / f"{key}.npy"
        np.save(path, array, allow_pickle=False)
        refs[key] = str(path)
    return ReconstructionArtifact(
        artifact_id="vggt-scene-one", artifact_version="frozen-v1",
        scene_name="scene-one", frame_ids=fs.frame_ids,
        frame_set_hash=fs.frame_set_hash, source_frame_indices=fs.source_frame_indices,
        timestamps=fs.timestamps, reconstruction_input_identity=fs.cache_identity(),
        confidence={"per_point_confidence": refs["point_conf"]},
        **refs,
    )


def _frozen(tmp_path, n_questions=2, existing=True):
    root = tmp_path / "input"
    root.mkdir()
    pixels = [np.full((3, 4, 3), slot, dtype=np.uint8) for slot in range(32)]
    manifest = {
        "schema": "harness3d-frozen-scene/1", "scene_id": "scene-one",
        "split": "inner_validation", "status": "frozen", "real_labels_available": False,
        "qa_ids": [f"qa-{i:02d}" for i in range(n_questions)],
        "questions": [], "frames": [], "files": {}, "artifact_source_path": None,
    }
    for slot, pixels_row in enumerate(pixels):
        original = f"/frozen/frames/{slot:04d}.png"
        ref = f"data/frames/{slot:04d}.png"
        path = root / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels_row).save(path)
        manifest["frames"].append({
            "source_path": original, "png_sha256": file_identity(path)["sha256"],
            "size": path.stat().st_size,
            "rgb_sha256": hashlib.sha256(pixels_row.tobytes()).hexdigest(),
            "shape": list(pixels_row.shape), "dtype": str(pixels_row.dtype),
        })
        manifest["files"][original] = ref
    items = []
    for qa_id in manifest["qa_ids"]:
        fs = build_frame_set(
            96, fps=24, episode_id=qa_id, dataset_id="scannet", video_id="scene-one",
            scene_name="scene-one",
        )
        row = {
            "qa_id": qa_id, "scene_name": "scene-one", "dataset": "scannet",
            "question_type": "object_rel_distance", "question": "Which chair is closer?",
            "options": ["A", "B"], "ground_truth": "",
        }
        ep = _episode_from_row(row, "inner_validation", pixels, 32, frame_set=fs)
        item = EpisodeItem(ep, pixels, source="vsi_bench",
                           video_path="/unavailable/raw/scene-one.mp4")
        items.append(item)
        public = ep.model_dump(mode="json")
        public.pop("ground_truth")
        manifest["questions"].append({
            "episode": public, "source": item.source, "video_path": item.video_path,
        })
    original_bytes = None
    if existing:
        original_root = tmp_path / "original-host"
        art = _geometry(original_root, items[0].episode.frame_set)
        artifact = original_root / "artifact.json"
        artifact.write_text(art.model_dump_json(), encoding="utf-8")
        original_bytes = artifact.read_bytes()
        manifest["artifact_source_path"] = str(artifact)
        for question, item in zip(manifest["questions"], items):
            question["logical_input_identity"] = input_identity(item, artifact)
        for source in [artifact, *map(Path, artifact_dependency_refs(art).values())]:
            ref = "data/geometry/" + source.name
            target = root / ref
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            manifest["files"][str(source)] = ref
        shutil.rmtree(original_root)
    _write(root / "scene.json", manifest)
    _write(root / "runtime.yaml", {})  # JSON is valid public YAML.
    return SimpleNamespace(root=root, manifest=manifest, items=items,
                           original_bytes=original_bytes)


def _never(*_args, **_kwargs):
    raise AssertionError("unexpected model/dataset/solver operation")


@pytest.fixture
def fake_perception(monkeypatch):
    from skill3d.online import runner
    from skill3d.reconstruction import vggt_runner

    monkeypatch.delenv("SKILL3D_ALLOW_HF_DOWNLOAD", raising=False)
    monkeypatch.setattr(vggt_runner, "run_vggt", _never)
    monkeypatch.setattr(runner, "run_episode", _never)
    monkeypatch.setattr(runner, "_prepare_m4_quality", lambda art, *_args, **_kwargs: art)
    calls = []

    def bind(art, pixels, _scene, ep, cfg, llm, *, m5_out_dir):
        assert ep.ground_truth == ""
        assert cfg.mode == "real" and len(pixels) == 32
        assert cfg.online_model == "qwen3.8-flash"
        assert (cfg.max_tokens, cfg.seed, cfg.temperature) == (8192, 0, 0)
        assert (cfg.semantic_min_pixels, cfg.semantic_max_pixels) == (65536, 1048576)
        assert llm is client
        calls.append((ep.qa_id, cfg, art))
        if ep.qa_id == "qa-00":
            return [], {"binding_error": "mask propagation unavailable"}, [], False
        path = Path(m5_out_dir)
        mask, points = path / "mask.npy", path / "points.npy"
        np.save(mask, np.ones((32, 3, 4), dtype=bool))
        np.save(points, np.ones((4, 3), dtype=np.float32))
        obj = ObjectRecord(
            obj_id=ep.qa_id + "-chair", category_name="chair",
            mask_per_frame=str(mask), pointcloud_world=str(points),
        )
        return [obj], {"status": "prepared"}, [], True

    client = SimpleNamespace(
        provider="bailian", model="qwen3.8-flash", model_revision="",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_s=180.0, temperature=0.0, max_output_tokens=8192,
        endpoint_hash=hashlib.sha256(
            b"https://dashscope.aliyuncs.com/compatible-mode/v1").hexdigest(),
    )
    monkeypatch.setattr(runner, "_bind_objects_best_effort", bind)
    constructor_calls = []

    def client_factory(**kwargs):
        constructor_calls.append(kwargs)
        return client

    return SimpleNamespace(calls=calls, client=client, client_factory=client_factory,
                           constructor_calls=constructor_calls)


def test_import_does_not_load_cpu_or_gpu_inference_dependencies():
    root = Path(__file__).resolve().parents[2] / "src"
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vggt', 'moge', 'sam2', 'numpy', 'PIL'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from skill3d.cluster.scene_compute import compute_scene, validate_scene_output
assert callable(compute_scene) and callable(validate_scene_output)
"""
    subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(root)!r})\n{code}"],
                   check=True, capture_output=True, text=True, timeout=15)


@pytest.mark.parametrize("existing", [False, True])
def test_valid_freeze_replays_without_original_host_or_video(tmp_path, existing):
    case = _frozen(tmp_path, existing=existing)
    summary = validate_scene_input(case.root)
    assert summary["qa_ids"] == case.manifest["qa_ids"]
    assert summary["n_frames"] == 32 and summary["status"] == "validated"
    assert summary["geometry_source"] == ("relocated" if existing else "reconstruct")
    assert summary["real_labels_available"] is False


@pytest.mark.parametrize("field", ["ground_truth", "answer", "reference_answer"])
@pytest.mark.parametrize("value", ["", "hidden-label"])
def test_public_label_fields_rejected_even_when_empty(tmp_path, field, value):
    case = _frozen(tmp_path)
    case.manifest["questions"][0]["episode"][field] = value
    _write(case.root / "scene.json", case.manifest)
    with pytest.raises(SceneComputeError, match="label field"):
        compute_scene(case.root, tmp_path / "output", {}, client_factory=_never)
    assert not (tmp_path / "output").exists()


def test_only_constant_empty_gt_allowed_in_saved_compatibility_identity(tmp_path):
    case = _frozen(tmp_path)
    logical = case.manifest["questions"][0]["logical_input_identity"]
    logical["episode"]["ground_truth"] = "hidden-label"
    logical["sha256"] = content_sha256({k: v for k, v in logical.items() if k != "sha256"})
    _write(case.root / "scene.json", case.manifest)
    with pytest.raises(SceneComputeError, match="label field"):
        validate_scene_input(case.root)


@pytest.mark.parametrize("tamper", [
    "png", "rgb_hash", "shape", "dtype", "frame_order", "source_indices",
    "timestamps", "video_path", "question_order", "geometry", "logical_episode",
    "missing_identity", "missing_dependency", "traversal", "absolute_target", "symlink",
])
def test_tampered_freeze_rejected_before_perception(tmp_path, tamper):
    case = _frozen(tmp_path)
    manifest = case.manifest
    fs = manifest["questions"][0]["episode"]["frame_set"]
    if tamper == "png":
        (case.root / manifest["files"][manifest["frames"][0]["source_path"]]).write_bytes(b"bad")
    elif tamper == "rgb_hash":
        manifest["frames"][0]["rgb_sha256"] = "0" * 64
    elif tamper in {"shape", "dtype"}:
        manifest["frames"][0][tamper] = [4, 3, 3] if tamper == "shape" else "float32"
    elif tamper == "frame_order":
        manifest["frames"][0], manifest["frames"][1] = manifest["frames"][1], manifest["frames"][0]
    elif tamper == "source_indices":
        fs["source_frame_indices"][0] = 1
    elif tamper == "timestamps":
        fs["timestamps"][0] = 0.5
    elif tamper == "video_path":
        manifest["questions"][0]["video_path"] = "/unavailable/raw/another.mp4"
    elif tamper == "question_order":
        manifest["questions"].reverse()
    elif tamper == "geometry":
        np.save(case.root / "data/geometry/depth_maps.npy", np.zeros((32, 3, 4)))
    elif tamper == "logical_episode":
        logical = manifest["questions"][0]["logical_input_identity"]
        logical["episode"]["question"] = "Another public question"
        logical["sha256"] = content_sha256({k: v for k, v in logical.items() if k != "sha256"})
    elif tamper == "missing_identity":
        manifest["questions"][1].pop("logical_input_identity")
    elif tamper == "missing_dependency":
        logical = manifest["questions"][0]["logical_input_identity"]
        manifest["files"].pop(logical["files"]["depth_maps"]["path"])
    elif tamper in {"traversal", "absolute_target"}:
        source = manifest["frames"][0]["source_path"]
        manifest["files"][source] = "../escape.png" if tamper == "traversal" else "/escape.png"
    elif tamper == "symlink":
        path = case.root / manifest["files"][manifest["frames"][0]["source_path"]]
        path.rename(path.with_suffix(".saved"))
        path.symlink_to(path.with_suffix(".saved"))
    _write(case.root / "scene.json", manifest)
    with pytest.raises(ValueError):
        compute_scene(case.root, tmp_path / "output", {}, client_factory=_never)
    assert not (tmp_path / "output").exists()


def test_full_scene_cap_cannot_be_bypassed_by_smaller_batches(tmp_path):
    case = _frozen(tmp_path, n_questions=9, existing=False)
    with pytest.raises(SceneComputeError, match="scene question cap"):
        compute_scene(case.root, tmp_path / "output", {}, client_factory=_never)
    with pytest.raises(SceneComputeError, match=r"\[1, 8\]"):
        compute_scene(case.root, tmp_path / "output",
                      {"max_questions_per_scene": 9}, client_factory=_never)


def test_output_local_bindings_survive_input_cleanup_and_recover_without_models(
        tmp_path, fake_perception, monkeypatch):
    from skill3d.evolution import prepared_bindings_v11 as pb
    from skill3d.segmentation.prepared_binding_store import load_prepared_binding_reference

    case = _frozen(tmp_path, n_questions=8)
    output = tmp_path / "output"
    method = {"runtime_config": "runtime.yaml", "min_free_gib": 0}
    before = {p.relative_to(case.root): p.read_bytes()
              for p in case.root.rglob("*") if p.is_file()}
    result = compute_scene(case.root, output, method,
                           client_factory=fake_perception.client_factory)
    assert len(fake_perception.calls) == 8
    assert result["scene_id"] == "scene-one" and result["qa_ids"] == case.manifest["qa_ids"]
    assert result["no_solver"] and result["no_labels"]
    base = result["variants"]["base"]
    assert base["artifact"] == "geometry/base/artifact.json"
    assert base["binding_manifest"] == "prepared/base/prepare_manifest.json"
    assert base["metric_status"] == "not_run"
    assert base["outcomes"]["qa-00"]["status"] == "binding_unavailable"
    assert base["outcomes"]["qa-00"]["materialized"] is False
    assert base["preparation"]["scene_groups"] == {"scene-one": result["qa_ids"]}
    assert all(r["request_identity"]["scene_audit_qa_ids"] == result["qa_ids"]
               for r in base["preparation"]["bindings"].values())
    assert before == {p.relative_to(case.root): p.read_bytes()
                      for p in case.root.rglob("*") if p.is_file()}
    artifact = ReconstructionArtifact.model_validate_json((output / base["artifact"]).read_bytes())
    assert all(Path(ref).is_relative_to(output) for ref in artifact_dependency_refs(artifact).values())
    identity = json.loads((output / base["input_identity_refs"]["qa-00"]).read_text())
    assert identity["artifact_path"] == str(output / base["artifact"])
    assert identity["sha256"] != case.manifest["questions"][0]["logical_input_identity"]["sha256"]
    monkeypatch.setattr(pb, "prepare_shared_bindings_v11", _never)
    assert compute_scene(case.root, output, method, client_factory=_never) == result
    shutil.rmtree(case.root)
    assert validate_scene_output(output) == result
    for qa_id, reference in base["preparation"]["bindings"].items():
        saved = json.loads((output / base["input_identity_refs"][qa_id]).read_text())
        binding = load_prepared_binding_reference(reference, scene_input_identity=saved)
        for obj in binding.objects:
            assert Path(obj.mask_per_frame).is_relative_to(output)
            assert Path(obj.pointcloud_world).is_relative_to(output)
            np.load(obj.mask_per_frame, allow_pickle=False)
            np.load(obj.pointcloud_world, allow_pickle=False)
    for reference, expected in result["files"].items():
        assert file_identity(output / reference) == expected
    assert json.loads((output / "source_provenance.json").read_text())["original_artifact"][
        "sha256"] == hashlib.sha256(case.original_bytes).hexdigest()
    published = tmp_path / "published"
    shutil.copytree(output, published)
    shutil.rmtree(output)
    checked = validate_published_scene(published, result["qa_ids"], "scene-one", method)
    assert checked["status"] == "validated"
    assert len(checked["variants"]["base"]["migration_receipts"]) == 8
    for qa_id, receipt in checked["variants"]["base"]["migration_receipts"].items():
        assert receipt["destination_root"] == str(published)
        assert receipt["logical_input_identity"]["episode"]["qa_id"] == qa_id
        assert receipt["logical_input_identity"]["episode"]["ground_truth"] == ""
    assert fake_perception.constructor_calls[0]["max_output_tokens"] == 8192
    assert fake_perception.constructor_calls[0]["timeout_s"] == 180


def test_missing_geometry_uses_only_frozen_frames_and_frameset(
        tmp_path, fake_perception, monkeypatch):
    from skill3d.reconstruction import vggt_runner

    case = _frozen(tmp_path, existing=False)
    checkpoint = tmp_path / "VGGT-1B"
    checkpoint.mkdir()
    calls = []

    def run(frames, scene_name, output_dir, *, checkpoint, frame_set, compute_quality):
        assert len(frames) == 32 and scene_name == "scene-one" and compute_quality
        assert all(np.array_equal(a, b) for a, b in zip(frames, case.items[0].pixels))
        assert frame_set == case.items[0].episode.frame_set
        calls.append(checkpoint)
        return _geometry(Path(output_dir), frame_set)

    monkeypatch.setattr(vggt_runner, "run_vggt", run)
    method = {"runtime_config": {}, "min_free_gib": 0, "vggt_checkpoint": str(checkpoint)}
    result = compute_scene(case.root, tmp_path / "output", method,
                           client_factory=fake_perception.client_factory)
    assert calls == [str(checkpoint)]
    assert result["validation"]["geometry_source"] == "reconstruct"


def test_metric_variant_prepares_all_qa_and_preserves_failed_evidence(
        tmp_path, fake_perception, monkeypatch):
    from skill3d.reconstruction import metric_evidence as me, metric_fusion as mf

    case = _frozen(tmp_path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"offline unit test model")
    output = tmp_path / "output"
    model = object()
    monkeypatch.setattr(mf, "make_moge2_model", lambda **_kwargs: model)

    def attach(art, frames, fs, supplied_model, out_dir):
        assert supplied_model is model and len(frames) == 32
        assert fs == case.items[0].episode.frame_set
        depth = np.load(art.depth_maps, allow_pickle=False)
        metric = np.full_like(depth, np.nan)
        valid = np.zeros_like(depth, dtype=bool)
        refs = {}
        for name, array in (("metric_depth_maps", metric), ("metric_valid_masks", valid)):
            path = Path(out_dir) / (name + ".npy")
            np.save(path, array, allow_pickle=False)
            refs[name] = str(path)
        fusion = mf.fuse_metric_scale(metric, depth, valid, model="moge2")
        request = {
            "input_identity": fs.cache_identity(),
            "sources": {name: file_identity(getattr(art, name))["sha256"]
                        for name in me.GEOMETRY_REFS if getattr(art, name)},
            "frame_ids": art.frame_ids, "frame_set_hash": art.frame_set_hash,
            "scene_name": art.scene_name, "source_frame_indices": art.source_frame_indices,
            "timestamps": art.timestamps,
        }
        receipt = mf.per_frame_receipt(fusion)
        receipt.update(request=request, request_sha256=me.digest(request), evidence_sha256={
            name: file_identity(ref)["sha256"] for name, ref in refs.items()
        })
        path = Path(out_dir) / "scale_receipt.json"
        _write(path, receipt)
        return art.model_copy(update={
            **refs, "artifact_version": "metric-v1", "per_frame_scale_ref": str(path),
            "metric_receipt_sha256": file_identity(path)["sha256"],
            "metric_input_identity": fs.cache_identity(), "metric_request_sha256": me.digest(request),
            "metric_model": fusion.model, "metric_fusion_version": fusion.version,
            "metric_scale": fusion.metric_scale, "scale_self_consistency": fusion.scale_self_consistency,
            "scale_fusion_status": fusion.status,
        })

    monkeypatch.setattr(me, "attach_metric_evidence", attach)
    method = {
        "runtime_config": {}, "metric_enabled": True, "min_free_gib": 0,
        "moge2_checkpoint": str(checkpoint),
    }
    result = compute_scene(case.root, output, method,
                           client_factory=fake_perception.client_factory)
    assert set(result["variants"]) == {"base", "metric"}
    assert len(fake_perception.calls) == 4
    metric = result["variants"]["metric"]
    assert metric["artifact"] == "geometry/metric/artifact.json"
    assert metric["binding_manifest"] == "prepared/metric/prepare_manifest.json"
    assert metric["metric_status"] != "success"
    assert set(metric["outcomes"]) == set(result["qa_ids"])
    art = ReconstructionArtifact.model_validate_json((output / metric["artifact"]).read_bytes())
    assert art.metric_scale is None
    me.validated_receipt(art)
    assert validate_scene_output(output) == result
    published = tmp_path / "published"
    shutil.copytree(output, published)
    shutil.rmtree(output)
    checked = validate_published_scene(published, result["qa_ids"], "scene-one", method)
    assert set(checked["variants"]) == {"base", "metric"}
    assert checked["variants"]["metric"]["metric_status"] == metric["metric_status"]


@pytest.mark.parametrize("change", ["file", "extra_file", "config"])
def test_completed_attempt_never_reruns_on_tamper_or_changed_config(
        tmp_path, fake_perception, monkeypatch, change):
    from skill3d.evolution import prepared_bindings_v11 as pb

    case = _frozen(tmp_path)
    output = tmp_path / "output"
    method = {"runtime_config": {}, "min_free_gib": 0}
    compute_scene(case.root, output, method, client_factory=fake_perception.client_factory)
    monkeypatch.setattr(pb, "prepare_shared_bindings_v11", _never)
    if change == "file":
        (output / "geometry/base/depth_maps.npy").write_bytes(b"tampered")
    elif change == "extra_file":
        (output / "unlisted.bin").write_bytes(b"unlisted")
    else:
        method["runtime_config"] = {"seed": 1}
    with pytest.raises(SceneComputeError):
        compute_scene(case.root, output, method, client_factory=_never)


def test_missing_or_incompatible_runtime_is_not_silently_defaulted(tmp_path):
    case = _frozen(tmp_path)
    with pytest.raises(SceneComputeError, match="reference"):
        compute_scene(case.root, tmp_path / "output", {}, client_factory=_never)
    with pytest.raises(ValueError):
        compute_scene(case.root, tmp_path / "output",
                      {"runtime_config": {"online_model": {"api_key": "forbidden"}}},
                      client_factory=_never)


@pytest.mark.parametrize("change", ["missing_question", "partial_group"])
def test_wrapper_rejects_partial_preparation_manifest(
        tmp_path, fake_perception, monkeypatch, change):
    from skill3d.evolution import prepared_bindings_v11 as pb

    case = _frozen(tmp_path)
    prepare = pb.prepare_shared_bindings_v11

    def incomplete(*args, **kwargs):
        result = prepare(*args, **kwargs)
        if change == "missing_question":
            result["bindings"].pop("qa-00")
        else:
            result["scene_groups"]["scene-one"] = ["qa-00"]
        return result

    monkeypatch.setattr(pb, "prepare_shared_bindings_v11", incomplete)
    with pytest.raises(SceneComputeError, match="full scene QA group"):
        compute_scene(case.root, tmp_path / "output",
                      {"runtime_config": {}, "min_free_gib": 0},
                      client_factory=fake_perception.client_factory)
    assert not (tmp_path / "output" / "inventory.json").exists()


def test_default_rpc_client_keeps_upstream_identity_without_worker_bailian_key(
        tmp_path, monkeypatch):
    from skill3d.cluster.scene_compute import _default_client_factory, _load_scene, _runtime_config
    from skill3d.cluster.perception_client import RemoteBailianClient

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("HARNESS3D_PERCEPTION_ENDPOINT", "http://127.0.0.1:27149")
    monkeypatch.setenv("HARNESS3D_PERCEPTION_TOKEN", "unit-rpc-token")
    case = _frozen(tmp_path)
    cfg, _ = _runtime_config(_load_scene(case.root, question_cap=8), tmp_path / "output", {
        "runtime_config": {"online_model": {"timeout_s": 180}},
    })
    client = _default_client_factory(cfg)("qa-00")
    assert isinstance(client, RemoteBailianClient) and client.configured
    assert client.endpoint_hash == hashlib.sha256(cfg.online_base_url.encode()).hexdigest()
    assert client.audit_identity(seed=0)["endpoint_sha256"] == client.endpoint_hash
    assert client.timeout_s == 180 and client.max_output_tokens == 8192


def test_duplicate_json_fields_and_top_level_labels_are_not_accepted(tmp_path):
    case = _frozen(tmp_path)
    raw = (case.root / "scene.json").read_text()
    (case.root / "scene.json").write_text('{"status":"frozen",' + raw[1:])
    with pytest.raises(SceneComputeError, match="duplicate"):
        validate_scene_input(case.root)
    case.manifest["labels"] = {}
    _write(case.root / "scene.json", case.manifest)
    with pytest.raises(SceneComputeError, match="label field"):
        validate_scene_input(case.root)


def test_injected_client_cannot_change_frozen_budget_or_upstream_identity(
        tmp_path, fake_perception):
    case = _frozen(tmp_path)
    fake_perception.client.max_output_tokens = 2048
    with pytest.raises(SceneComputeError, match="perception client"):
        compute_scene(case.root, tmp_path / "output",
                      {"runtime_config": {}, "min_free_gib": 0},
                      client_factory=fake_perception.client_factory)
    assert fake_perception.calls == []


@pytest.mark.parametrize("error", [
    "HTTPError: HTTP Error 429: Too Many Requests",
    "HTTPError: HTTP Error 503: Service Unavailable",
    "HTTPError: HTTP Error 504: Gateway Timeout",
    "ServiceUnavailable: perception transport unavailable",
])
def test_transport_outage_receipts_cannot_complete_a_scene(
        tmp_path, fake_perception, monkeypatch, error):
    from skill3d.online import runner

    case = _frozen(tmp_path)
    original = runner._bind_objects_best_effort

    def outage(*args, **kwargs):
        objects, stats, notes, materialized = original(*args, **kwargs)
        stats["candidates"] = [{
            "semantic_visual": {"status": "unavailable", "error": error},
        }]
        return objects, stats, notes, materialized

    monkeypatch.setattr(runner, "_bind_objects_best_effort", outage)
    with pytest.raises(SceneComputeError, match="infrastructure failure"):
        compute_scene(case.root, tmp_path / "output",
                      {"runtime_config": {}, "min_free_gib": 0},
                      client_factory=fake_perception.client_factory)
    assert not (tmp_path / "output" / "inventory.json").exists()


def test_worker_checkpoint_wrappers_validate_without_model_or_original_inputs(
        tmp_path, fake_perception):
    from skill3d.cluster.publication import ArtifactDigest, PublicationReceipt

    case = _frozen(tmp_path)
    output = tmp_path / "output"
    method = {"runtime_config": {}, "min_free_gib": 0}
    result = compute_scene(case.root, output, method,
                           client_factory=fake_perception.client_factory)
    _write(output / "scene_result.json", result)
    files = {**result["files"], "scene_result.json": file_identity(output / "scene_result.json")}
    receipt = PublicationReceipt(
        job_key="job-one", completed_key="completed-one", identity_hash="a" * 64,
        stage="gpu_aggregate", attempt=1, validator="harness3d-scene-qa-validator-v1",
        artifacts=tuple(ArtifactDigest(name, row["sha256"], row["size"])
                        for name, row in files.items()),
    )
    _write(output / "publication.json", receipt.to_dict())
    assert validate_scene_output(output) == result
    shutil.rmtree(case.root)
    assert validate_published_scene(
        output, case.manifest["qa_ids"], "scene-one", method)["status"] == "validated"
    _write(output / "scene_result.json", {**result, "no_solver": False})
    with pytest.raises(SceneComputeError, match="worker scene result"):
        validate_scene_output(output)


def test_independent_audit_replays_committed_output_and_detects_later_corruption(
        tmp_path, fake_perception, monkeypatch):
    import importlib
    from skill3d.cluster.identity import FrozenIdentity
    from skill3d.cluster.publication import ArtifactDigest, PublicationReceipt, hash_file
    from skill3d.cluster.queue import JobSpec, SQLiteJobQueue, WorkerIdentity

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    auditor = importlib.import_module("cluster_audit")
    case = _frozen(tmp_path)
    method = {"runtime_config": {}, "min_free_gib": 0}
    identity = FrozenIdentity.freeze(
        {"scenes": {"scene-one": {"qa_ids": case.manifest["qa_ids"],
                                   "split": "inner_validation"}}}, method)
    spec = JobSpec("run/scene-one/gpu_aggregate", "completed-one", identity, "gpu_aggregate")
    root = tmp_path / "coordinator"
    output = root / "results" / hashlib.sha256(spec.job_key.encode()).hexdigest() / "attempt-1"
    compute_scene(case.root, output, method, client_factory=fake_perception.client_factory)
    artifacts = tuple(
        ArtifactDigest(str(path.relative_to(output)), *hash_file(path))
        for path in sorted(output.rglob("*")) if path.is_file()
    )
    receipt = PublicationReceipt(
        spec.job_key, spec.completed_key, identity.identity_hash, spec.stage,
        1, "harness3d-scene-qa-validator-v1", artifacts)
    _write(output / "publication.json", receipt.to_dict())
    receipt_hash = hash_file(output / "publication.json")[0]
    queue = SQLiteJobQueue(root / "state/jobs.sqlite3")
    queue.enqueue_many([spec])
    lease = queue.claim(WorkerIdentity("audit-fixture", "local", "cpu"))
    lease = queue.start_attempt(lease)
    lease = queue.finish_attempt(lease, output_metadata={"receipt_sha256": receipt_hash})
    queue.complete(lease, receipt_path=output / "publication.json",
                   expected_receipt_hash=receipt_hash, artifact_root=output)
    validation = root / "receipts/scene_validation" / (
        hashlib.sha256(spec.job_key.encode()).hexdigest() + ".json")
    _write(validation, validate_published_scene(
        output, case.manifest["qa_ids"], "scene-one", method))
    _write(root / "coordinator_jobs.json",
           {"identity": identity.to_dict(), "jobs": [spec.to_dict()]})
    _write(root / "deliverables/coverage.json", {
        "identity_hash": identity.identity_hash, "updated_at": 1,
        "status": "completed", "integrity_errors": [],
        "queue": {"counts": {"complete": 1}},
        "scenes": {"scene-one": {
            "status": "complete", "model_attempts": 1, "receipt_sha256": receipt_hash,
            "output_root": str(output), "validation_receipt": str(validation),
            "qa_ids": case.manifest["qa_ids"], "split": "inner_validation",
        }},
    })
    before = queue.get_job(spec.job_key)
    verified = auditor.audit(root)
    assert verified["status"] == "verified" and verified["verified_questions"] == {"base": 2}
    assert verified["question_outcomes"]["base"]["binding_unavailable"] == 1
    assert queue.get_job(spec.job_key) == before
    (output / "geometry/base/depth_maps.npy").write_bytes(b"corrupted after publication")
    rejected = auditor.audit(root, recheck=True)
    assert rejected["status"] == "audit_errors" and rejected["verified_scenes"] == 0
    assert rejected["errors"][0]["error_type"] == "PublicationError"
    assert queue.get_job(spec.job_key) == before
