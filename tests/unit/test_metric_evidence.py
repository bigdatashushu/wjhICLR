"""Regression coverage for aligned real-model evidence and cache isolation."""

import json
from pathlib import Path

import numpy as np
import pytest

from skill3d.adapters.frame_set import build_frame_set, with_pixel_identity
from skill3d.reconstruction.image_grid import prepare_metric_frames, mask_on_vggt_grid
from skill3d.reconstruction.metric_evidence import (
    attach_metric_evidence, copy_geometry, fusion_contract, validated_receipt,
)
from skill3d.reconstruction.metric_fusion import MetricDepthOutput
from skill3d.reconstruction_gate.scene_state import _valid_frame_ratio
from skill3d.schemas.reconstruction import ReconstructionArtifact


class MetricModel:
    name = "moge2"
    identity = {"name": "unit-test-only", "checkpoint_sha256": "fixture"}

    def __init__(self, failures=(), wrong_grid=False):
        self.calls = 0
        self.failures = failures
        self.wrong_grid = wrong_grid

    def infer(self, rgb, intrinsics):
        idx = self.calls
        self.calls += 1
        if idx in self.failures:
            raise RuntimeError(f"frame {idx} failed")
        shape = (12, 12) if self.wrong_grid else rgb.shape[:2]
        return MetricDepthOutput(np.full(shape, 2.5), np.ones(shape, dtype=bool))


@pytest.fixture
def frozen(tmp_path):
    frames = [np.full((48, 64, 3), 100 + i, np.uint8) for i in range(4)]
    fs = with_pixel_identity(build_frame_set(4, n_frames=4, scene_name="fixture"), frames)
    rgb, _, _ = prepare_metric_frames(frames)
    h, w = rgb[0].shape[:2]
    refs = {}
    for key, value in {
        "depth_maps": np.ones((4, h, w), np.float32),
        "intrinsics": np.tile([[300, 0, w / 2], [0, 300, h / 2], [0, 0, 1]], (4, 1, 1)),
        "c2w_list": np.tile(np.eye(4), (4, 1, 1)),
        "point_map": np.zeros((4, h, w, 3), np.float32),
        "point_conf": np.ones((4, h, w), np.float32),
    }.items():
        path = tmp_path / f"{key}.npy"
        np.save(path, value)
        refs[key] = str(path)
    art = ReconstructionArtifact(
        artifact_id="unit", artifact_version="unit", scene_name="fixture",
        frame_ids=fs.readable_frame_ids, source_frame_indices=fs.source_frame_indices,
        timestamps=fs.timestamps, frame_set_hash=fs.frame_set_hash,
        reconstruction_input_identity=fs.cache_identity(), **refs,
        confidence={"per_point_confidence": refs["point_conf"],
                    "coverage_count_per_frame": ""},
    )
    return art, frames, fs


def test_portrait_crop_and_batch_padding_match_masks():
    portrait = np.zeros((100, 50, 3), np.uint8)
    portrait[:25] = 255
    landscape = np.full((50, 100, 3), 100, np.uint8)
    rgb, valid, rows = prepare_metric_frames([portrait, landscape])
    assert rows[0]["resized_hw"] == [1036, 518]
    assert rows[0]["crop_top"] == 259
    assert rows[1]["pad_top"] == 133  # round(18.5)=18, height=252
    assert not valid[1][:133].any()
    assert valid[1][133:385].all()
    original_mask = np.zeros((100, 50), dtype=bool)
    original_mask[:20] = True  # Outside the retained center crop.
    assert not mask_on_vggt_grid(original_mask, (518, 518)).any()
    assert rgb[0].shape == rgb[1].shape == (518, 518, 3)


def test_evidence_partial_failure_keeps_frame_and_binds_scalars(frozen, tmp_path):
    art, frames, fs = frozen
    result = attach_metric_evidence(art, frames, fs, MetricModel(failures={1}), tmp_path / "on")
    rec = validated_receipt(result)
    assert result.metric_scale == 2.5
    assert rec["n_frames_valid"] == 3 and _valid_frame_ratio(result) == 0.75
    assert rec["diagnostics"][1]["error"] == {"type": "RuntimeError", "message": "frame 1 failed"}
    assert np.isnan(np.load(result.metric_depth_maps)[1]).all()
    assert not np.load(result.metric_valid_masks)[1].any()
    assert rec["per_frame"][1]["s_k"] is None
    assert result.artifact_version != art.artifact_version
    assert _valid_frame_ratio(result.model_copy(update={"metric_scale": 25.0})) == 0
    assert _valid_frame_ratio(result.model_copy(update={"scene_name": "other"})) == 0
    assert _valid_frame_ratio(result.model_copy(update={"metric_input_identity": "other"})) == 0


@pytest.mark.parametrize("model", [MetricModel(failures={0, 1}), MetricModel(wrong_grid=True)])
def test_bad_depth_never_resized_or_silently_dropped(frozen, tmp_path, model):
    art, frames, fs = frozen
    result = attach_metric_evidence(art, frames, fs, model, tmp_path / "bad")
    rec = validated_receipt(result)
    assert result.scale_fusion_status == "failed" and result.metric_scale is None
    assert rec["n_frames_total"] == 4 and len(rec["diagnostics"]) == 4
    assert sum(row["error"] is not None for row in rec["diagnostics"]) >= 2


def test_mutated_depth_receipt_and_private_copy(frozen, tmp_path):
    from skill3d.evaluation.skill_ablation_v11 import _copy_base, _clone_tree

    art, frames, fs = frozen
    result = attach_metric_evidence(art, frames, fs, MetricModel(), tmp_path / "on")
    source = tmp_path / "source.json"
    source.write_text(result.model_dump_json())
    arm = _copy_base(source, tmp_path / "B11")
    assert validated_receipt(arm)["metric_scale"] == 2.5
    assert arm.metric_depth_maps != result.metric_depth_maps
    (tmp_path / "B11/artifact.json").write_text(arm.model_dump_json())
    _clone_tree(tmp_path / "B11", tmp_path / "B01")
    cloned = ReconstructionArtifact.model_validate_json(
        (tmp_path / "B01/artifact.json").read_bytes())
    assert _valid_frame_ratio(cloned) == 1
    assert Path(cloned.per_frame_scale_ref).read_bytes() == Path(arm.per_frame_scale_ref).read_bytes()
    depth = np.load(arm.metric_depth_maps)
    depth[0, 0, 0] = 99
    np.save(arm.metric_depth_maps, depth)
    assert _valid_frame_ratio(arm) == 0
    assert _valid_frame_ratio(result) == 1
    copied = copy_geometry(result, tmp_path / "off")
    assert copied.metric_scale is None and copied.per_frame_scale_ref is None
    assert copied.scale_fusion_status == "not_run"
    receipt = Path(result.per_frame_scale_ref)
    value = json.loads(receipt.read_text())
    value["valid_frame_ratio"] = 0.5
    receipt.write_text(json.dumps(value))
    assert _valid_frame_ratio(result) == 0


def test_cache_contract_changes_with_weights_and_pixels(frozen, tmp_path):
    from skill3d.adapters.episode_source import EpisodeItem
    from skill3d.online.synthetic import make_synthetic_episode
    from skill3d.reconstruction.run import artifact_path, plan_scene_jobs

    art, frames, fs = frozen
    model = MetricModel()
    result = attach_metric_evidence(art, frames, fs, model, tmp_path / "on")
    ep = make_synthetic_episode("room_size_estimation").episode.model_copy(update={
        "frame_set": fs, "scene_name": "fixture"})
    item = EpisodeItem(episode=ep, pixels=frames, source="synthetic")
    path = artifact_path(tmp_path, "fixture", "vggt", frame_set=fs)
    path.parent.mkdir()
    path.write_text(result.model_dump_json())
    assert plan_scene_jobs([item], tmp_path, "vggt", metric_depth_model=model)[0].status == "skipped"
    assert plan_scene_jobs([item], tmp_path, "vggt")[0].status == "failed"
    changed = MetricModel()
    changed.identity = {**model.identity, "checkpoint_sha256": "different"}
    assert fusion_contract(model) != fusion_contract(changed)
    assert plan_scene_jobs([item], tmp_path, "vggt", metric_depth_model=changed)[0].status == "failed"
    assert plan_scene_jobs([item], tmp_path, "vggt", force=True)[0].status == "pending"


def test_moge_identity_changes_with_attention_runtime(monkeypatch, tmp_path):
    from importlib import metadata
    from skill3d.reconstruction import metric_evidence

    package = tmp_path / "moge"
    model_source = package / "model/v2.py"
    model_source.parent.mkdir(parents=True)
    model_source.write_text("# frozen model source\n")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen weights")
    monkeypatch.setattr(metric_evidence.inspect, "getfile", lambda cls: str(model_source))
    versions = {"torch": "2.10.0", "utils3d": "1.0", "moge": "2.0",
                "xformers": "0.0.32.post1"}
    monkeypatch.setattr(metadata, "version", versions.__getitem__)
    before = metric_evidence.model_identity(MetricModel, checkpoint)
    versions["xformers"] = "0.0.34"
    after = metric_evidence.model_identity(MetricModel, checkpoint)
    assert before["source_sha256"] == after["source_sha256"]
    assert before["checkpoint_sha256"] == after["checkpoint_sha256"]
    assert before != after
    def without_xformers(name):
        if name == "xformers":
            raise metadata.PackageNotFoundError(name)
        return versions[name]
    monkeypatch.setattr(metadata, "version", without_xformers)
    assert metric_evidence.model_identity(
        MetricModel, checkpoint)["runtime_versions"]["xformers"] == "absent"


def test_pinned_moge_api_uses_meter_depth_without_extra_scale():
    torch = pytest.importorskip("torch")
    from skill3d.reconstruction.metric_fusion import _MoGe2MetricModel

    class Raw:
        def infer(self, image, **kwargs):
            assert kwargs["use_fp16"] is True
            assert kwargs["fov_x"] == pytest.approx(90)
            return self.output

    raw = Raw()
    raw.output = {"depth": torch.full((1, 8, 8), 2.0),
                  "mask": torch.ones((1, 8, 8), dtype=torch.bool)}
    model = _MoGe2MetricModel(raw, torch, "cpu", "unit-test-only")
    rgb, k = np.full((8, 8, 3), 100, np.uint8), np.array([[4, 0, 4], [0, 4, 4], [0, 0, 1]])
    assert np.all(model.infer(rgb, k).depth_metric == 2.0)
    raw.output["metric_scale"] = torch.tensor(float("nan"))
    with pytest.raises(RuntimeError, match="Unexpected metric_scale"):
        model.infer(rgb, k)


def test_invalid_mask_and_conf_sequence_fail_closed():
    from skill3d.reconstruction.metric_fusion import per_frame_scale, fuse_metric_scale

    depth = np.ones((40, 40))
    assert per_frame_scale(0, depth, depth, np.full(depth.shape, np.nan)).s_k is None
    result = fuse_metric_scale([depth] * 4, [depth] * 4, [depth > 0] * 4, [depth] * 2)
    assert result.status == "failed" and result.metric_scale is None
