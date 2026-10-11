"""Regression checks for the legacy-cache offline detector startup failure."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skill3d.segmentation.groundingdino_startup import (
    BERT_REVISION, DetectorStartupError, bind_local_bert, load_local_bert,
    resolve_bert_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "detector_launcher_test", ROOT / "scripts/serve_groundingdino.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def _snapshot(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": "bert"}))
    (path / "vocab.txt").write_text("[PAD]\n[UNK]\n")
    (path / "model.safetensors").write_bytes(b"fixture")
    return path


def test_pinned_snapshot_in_legacy_transformers_cache_is_found(tmp_path):
    hf_home = tmp_path / "hf"
    expected = _snapshot(hf_home / "transformers/models--bert-base-uncased/snapshots" / BERT_REVISION)
    # A mutable main pointer or another revision must not change the chosen model.
    _snapshot(hf_home / "hub/models--bert-base-uncased/snapshots" / ("a" * 40))
    assert resolve_bert_snapshot(environ={"HF_HOME": str(hf_home)}) == expected


def test_invalid_explicit_path_does_not_silently_fall_back(tmp_path):
    _snapshot(tmp_path / "hf/transformers/models--bert-base-uncased/snapshots" / BERT_REVISION)
    with pytest.raises(DetectorStartupError, match="Cannot read local BERT config"):
        resolve_bert_snapshot(tmp_path / "missing", environ={"HF_HOME": str(tmp_path / "hf")})


def test_missing_weights_fail_before_model_loading(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path / "bert")
    (snapshot / "model.safetensors").unlink()
    monkeypatch.setattr(launcher, "load_local_bert",
                        lambda _: pytest.fail("Incomplete snapshot must not invoke a model loader"))
    log = tmp_path / "log"
    assert launcher.main(["--bert-path", str(snapshot), "--log-dir", str(log), "--check-only"]) == 1
    receipt = json.loads((log / "groundingdino_startup.json").read_text())
    assert receipt["status"] == "failed" and "weights are missing" in receipt["error"]
    assert "Traceback" in receipt["traceback"]


def test_loaders_receive_explicit_local_paths_and_disable_downloads(monkeypatch, tmp_path):
    import sys

    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return object()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load),
        BertModel=SimpleNamespace(from_pretrained=load)))
    tokenizer, encoder = load_local_bert(tmp_path)
    assert calls == [(str(tmp_path), {"local_files_only": True})] * 2
    loaders = SimpleNamespace()
    bind_local_bert(loaders, tmp_path, tokenizer, encoder)
    assert loaders.get_tokenlizer("bert-base-uncased") is tokenizer
    assert loaders.get_pretrained_language_model("bert-base-uncased") is encoder
    with pytest.raises(DetectorStartupError, match="unexpected text encoder"):
        loaders.get_pretrained_language_model("roberta-base")


def test_corrupt_weights_error_is_recorded_and_service_is_not_started(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path / "bert")
    def broken(_):
        raise ValueError("corrupt local safetensors")
    monkeypatch.setattr(launcher, "load_local_bert", broken)
    monkeypatch.setattr(launcher.runpy, "run_path",
                        lambda *_args, **_kwargs: pytest.fail("Do not start a service after a load failure"))
    assert launcher.main(["--bert-path", str(snapshot), "--log-dir", str(tmp_path / "log")]) == 1
    receipt = json.loads((tmp_path / "log/groundingdino_startup.json").read_text())
    assert receipt["status"] == "failed" and "corrupt local safetensors" in receipt["traceback"]
