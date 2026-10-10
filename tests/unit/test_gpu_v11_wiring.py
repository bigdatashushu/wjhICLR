"""GPU launcher contracts tested without weights, GPU, or network access."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "gpu_v11_launcher_test", ROOT / "scripts/run_gpu_experiment_v11.py")
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


def test_snapshot_hashing_keeps_files_under_user_cache(tmp_path):
    root = tmp_path / ".cache/huggingface/snapshot"
    root.mkdir(parents=True)
    (root / "model.bin").write_bytes(b"test-weights")
    metadata = root / ".cache/download"
    metadata.mkdir(parents=True)
    (metadata / "lock").write_text("metadata")
    assert [row["path"] for row in gpu._snapshot_files(root)] == ["model.bin"]


def test_existing_cache_uses_pinned_revision_not_main(tmp_path):
    wanted, main = "a" * 40, "b" * 40
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs/main").write_text(main)
    for revision in (wanted, main):
        (tmp_path / "snapshots" / revision).mkdir(parents=True)
    entry = {"existing_path": str(tmp_path), "revision": wanted}
    assert gpu._existing_weight_path(entry) == (tmp_path / "snapshots" / wanted, wanted)
    entry["existing_path"] = str(tmp_path / "snapshots" / main)
    assert gpu._existing_weight_path(entry) == (None, "")
    entry["existing_path"] = str(tmp_path / "mutable-local-weights")
    Path(entry["existing_path"]).mkdir()
    assert gpu._existing_weight_path(entry) == (None, "")


def test_runtime_plan_freezes_budget_seed_and_private_library(tmp_path):
    cfg = gpu._read_config(ROOT / "configs/gpu_experiment_v11.yaml")
    cfg["experiment"].update(video_root=str(tmp_path), run_evolution=True)
    cfg["runtime"].update(n_images=8, max_pixels=65536, max_model_len=16384)
    weights = {name: tmp_path / name for name in cfg["weights"]}
    commands = gpu._pipeline(ROOT, tmp_path, cfg, weights, dry_run=True)
    frozen = yaml.safe_load((tmp_path / "runtime_config.yaml").read_text())
    assert frozen["frame_sampling"]["n_frames"] == frozen["vllm"]["n_frames"] == 8
    assert frozen["vllm"]["max_pixels"] == 65536
    assert frozen["vllm"]["max_model_len"] == 16384
    assert frozen["paths"]["skill_library"] == str(tmp_path / "skill_library")
    for row in commands:
        command = row["command"]
        if row["stage"] != "serve_qwen":
            assert command[command.index("--seed") + 1] == "137"
            assert command[command.index("--config") + 1] == str(tmp_path / "runtime_config.yaml")
        if row["stage"] == "evolution":
            assert command[command.index("--library-root") + 1] == str(tmp_path / "skill_library")


def test_vggt_uses_launcher_checkpoint(monkeypatch, tmp_path):
    from skill3d.reconstruction.vggt_runner import run_vggt

    received = []

    def load(checkpoint):
        received.append(checkpoint)
        raise RuntimeError("stop before inference")

    monkeypatch.setenv("SKILL3D_VGGT_CHECKPOINT", str(tmp_path / "pinned-vggt"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True), bfloat16="bf16", float32="fp32"))
    monkeypatch.setitem(sys.modules, "vggt.models.vggt",
                        SimpleNamespace(VGGT=SimpleNamespace(from_pretrained=load)))
    monkeypatch.setitem(sys.modules, "vggt.utils.load_fn",
                        SimpleNamespace(load_and_preprocess_images=None))
    with pytest.raises(RuntimeError, match="stop before inference"):
        run_vggt([], "scene", tmp_path)
    assert received == [str(tmp_path / "pinned-vggt")]


def test_moge_uses_local_checkpoint_file(monkeypatch, tmp_path):
    from skill3d.reconstruction.metric_fusion import make_moge2_model

    received = []

    def load(checkpoint):
        received.append(checkpoint)
        raise RuntimeError("stop before inference")

    (tmp_path / "model.pt").write_bytes(b"test")
    monkeypatch.setenv("SKILL3D_MOGE2_CHECKPOINT", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, "moge.model.v2",
                        SimpleNamespace(MoGeModel=SimpleNamespace(from_pretrained=load)))
    with pytest.raises(RuntimeError, match="stop before inference"):
        make_moge2_model()
    assert received == [str(tmp_path / "model.pt")]
