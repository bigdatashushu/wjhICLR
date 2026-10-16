"""Configuration failures must precede decoding and inference (repair M1)."""

import json
from pathlib import Path

import pytest
import yaml

from skill3d.online.config import (
    config_receipt, load_config, load_yaml, online_model_from, resolve_runtime,
)
from skill3d.adapters.episode_source import (
    EpisodeSourceError, load_jsonl_items, load_vsi_bench_items,
)
from skill3d.adapters import vsibench_loader


def test_default_output_budget_is_8192(tmp_path):
    path = tmp_path / "runtime.yaml"
    path.write_text("{}\n")
    settings = online_model_from(load_config(path))
    assert settings.max_tokens == 8192


def test_repository_runtime_config_freezes_output_budget():
    path = Path(__file__).resolve().parents[2] / "configs/config.yaml"
    config = load_config(path)
    assert config["online_model"]["max_tokens"] == 8192
    assert config.sources["online_model.max_tokens"] == "yaml"
    model = online_model_from(config)
    assert model.model == "qwen3.8-flash"
    assert model.api_key_env == "DASHSCOPE_API_KEY"


def test_m5_config_stores_only_key_environment_name(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "must-not-appear")
    path = tmp_path / "runtime.yaml"
    path.write_text("online_model: {model: qwen3.8-flash}\n")
    receipt = config_receipt(load_config(path))
    text = json.dumps(receipt)
    assert "must-not-appear" not in text
    assert '"api_key_env": "<redacted>"' in text


def test_online_model_output_budget_is_effective(tmp_path):
    path = tmp_path / "runtime.yaml"
    path.write_text("seed: 137\nonline_model:\n  max_tokens: 2048\n")
    settings = online_model_from(load_config(path))
    assert settings.model == "qwen3.8-flash"
    assert settings.max_tokens == 2048


def test_yaml_expands_values_after_parsing(tmp_path, monkeypatch):
    monkeypatch.setenv("M1_PATH", 'path: "quoted"\nnext: value')
    monkeypatch.setenv("M1_ALIAS", "${M1_PATH}")
    path = tmp_path / "values.yaml"
    path.write_text("paths: ['${M1_ALIAS}', '${M1_ABSENT:-default}']\n")
    assert load_yaml(path) == {"paths": ['path: "quoted"\nnext: value', "default"]}


@pytest.mark.parametrize("text", ["${M1_MISSING}", "${M1_LOOP}", "${malformed"])
def test_missing_cyclic_and_unclosed_placeholders_fail(tmp_path, monkeypatch, text):
    monkeypatch.delenv("M1_MISSING", raising=False)
    monkeypatch.setenv("M1_LOOP", "${M1_LOOP}")
    path = tmp_path / "values.yaml"
    path.write_text(yaml.safe_dump({"value": text}))
    with pytest.raises(ValueError):
        load_yaml(path)


def test_missing_explicit_configs_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing-split.yaml")


@pytest.mark.parametrize("section", [
    {"online_model": {"unknown": 1}},
    {"online_model": {"max_tokens": 0}},
    {"online_model": {"max_tokens": 32768}},
    {"online_model": {"base_url": "http://insecure.example/v1"}},
    {"online_model": {"audit_min_pixels": 100, "audit_max_pixels": 10}},
    {"online_model": {"provider": "unknown"}},
    {"frame_sampling": {"n_frames": 33}},
    {"seed": True},
    {"paths": []},
])
def test_bad_runtime_settings_fail(tmp_path, section):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(section))
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("split,cfg", [
    ("induction", {}),
    ("unknown", {"induction_scene_ids": ["s"]}),
    ("induction", {"induction_scene_ids": []}),
    ("induction", {"induction_scene_ids": "s"}),
    ("induction", {"induction_scene_ids": ["s", "s"]}),
    ("induction", {"induction_scene_ids": ["s"], "inner_validation_scene_ids": ["s"]}),
])
def test_split_errors_precede_meta_and_decode(monkeypatch, split, cfg):
    def forbidden(**kwargs):
        pytest.fail("invalid split reached metadata loading")

    monkeypatch.setattr(vsibench_loader, "load_meta", forbidden)
    with pytest.raises(EpisodeSourceError):
        load_vsi_bench_items(split, cfg)


def test_jsonl_checks_entire_manifest_before_decode(tmp_path, monkeypatch):
    import cv2

    rows = [
        dict(qa_id="1", scene_name="s", question_type="object_counting",
             question="q", ground_truth="1", split="induction", frame_paths=["image"]),
        dict(qa_id="2", scene_name="other", question_type="object_counting",
             question="q", ground_truth="1", split="induction", frame_paths=["image"]),
    ]
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr(cv2, "imread", lambda *a: pytest.fail("invalid manifest decoded"))
    with pytest.raises(EpisodeSourceError):
        load_jsonl_items(path, "induction", limit=1,
                         split_cfg={"induction_scene_ids": ["s"]})


def test_explicit_missing_metadata_never_uses_hf(tmp_path):
    with pytest.raises(FileNotFoundError):
        vsibench_loader.load_meta(local_path=tmp_path / "missing.jsonl")


def test_cli_seed_zero_capacity_and_sources(tmp_path, monkeypatch):
    path = tmp_path / "runtime.yaml"
    path.write_text("seed: 137\nframe_sampling: {n_frames: 8}\nonline_model:\n"
                    "  max_tokens: 1024\n")
    monkeypatch.chdir(tmp_path)
    cfg = resolve_runtime(load_config(path), seed=0)
    assert cfg["seed"] == 0
    assert cfg["frame_sampling"]["n_frames"] == 8
    assert cfg.sources["seed"] == "cli"
    assert cfg.sources["online_model.max_tokens"] == "yaml"
    assert cfg.sources["online_model.max_model_len"] == "default"
    # The same source can execute from a checkout or a frozen development snapshot.
    expected_root = Path(__file__).resolve().parents[2]
    assert Path(cfg["split_config"]) == expected_root / "configs/vsi_bench_split.yaml"
    with pytest.raises(ValueError, match="32-frame"):
        resolve_runtime(cfg, formal=True)


def test_online_cli_propagates_config_before_solver(tmp_path, monkeypatch):
    from skill3d.online import eval as cli

    path = tmp_path / "runtime.yaml"
    path.write_text("seed: 137\nframe_sampling: {n_frames: 8}\nonline_model:\n"
                    "  max_tokens: 1024\n")
    captured = {}

    def load(split, **kwargs):
        captured["sampling"] = kwargs
        return []

    class Captured(Exception):
        pass

    def run(items, cfg):
        captured["solver"] = cfg
        raise Captured

    monkeypatch.setattr(cli, "load_synthetic_items", load)
    monkeypatch.setattr(cli, "run_split", run)
    with pytest.raises(Captured):
        cli.main(["--config", str(path), "--seed", "0", "--split", "induction"])
    assert captured["sampling"]["n_frames"] == 8
    assert captured["sampling"]["seed"] == captured["solver"].seed == 0
    assert captured["solver"].max_tokens == 1024
    assert captured["solver"].online_model == "qwen3.8-flash"


def test_reconstruction_cli_uses_yaml_frames(tmp_path, monkeypatch):
    from skill3d.reconstruction import run as cli

    path = tmp_path / "runtime.yaml"
    path.write_text("seed: 137\nframe_sampling: {n_frames: 8}\n")
    captured = []
    monkeypatch.setattr(cli, "load_vsi_bench_items",
                        lambda *a, **k: captured.append(k) or [])
    monkeypatch.setattr(cli, "plan_scene_jobs", lambda *a, **k: [])
    assert cli.main(["--config", str(path), "--split", "induction", "--plan-only"]) == 0
    assert captured[0]["n_frames"] == 8 and captured[0]["seed"] == 137
    assert captured[0]["local_meta_path"].endswith("/data/vsi_bench_meta")


@pytest.mark.parametrize("field,value", [
    ("source_frame_indices", []), ("source_frame_indices", "0"),
    ("source_frame_indices", [True]), ("timestamps", []),
    ("timestamps", [float("nan")]), ("frame_paths", [None]),
])
def test_jsonl_identity_arrays_are_checked_before_decode(tmp_path, monkeypatch, field, value):
    import cv2

    row = dict(qa_id="1", scene_name="s", question_type="object_counting",
               question="q", ground_truth="1", split="induction", frame_paths=["image"])
    row[field] = value
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(row))
    monkeypatch.setattr(cv2, "imread", lambda *a: pytest.fail("bad identity reached decoding"))
    with pytest.raises(EpisodeSourceError):
        load_jsonl_items(path, "induction")
