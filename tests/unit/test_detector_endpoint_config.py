"""Shared launcher configuration precedence, including explicit disablement."""

import importlib.util
from pathlib import Path

import pytest

from skill3d.online.config import resolve_detector_endpoint


@pytest.mark.parametrize("experiment,env,endpoint,source", [
    ({"detector_endpoint": "http://127.0.0.1:20024"}, {},
     "http://127.0.0.1:20024", "yaml"),
    ({"detector_endpoint": "http://127.0.0.1:20024"},
     {"HARNESS3D_DETECTOR_ENDPOINT": "http://localhost:20026"},
     "http://localhost:20026", "env"),
    ({}, {}, "http://127.0.0.1:20022", "default"),
    ({"detector_endpoint": "http://127.0.0.1:20024"},
     {"HARNESS3D_DETECTOR_ENDPOINT": ""}, "", "env"),
    ({"detector_endpoint": ""}, {}, "", "yaml"),
    ({}, {"SKILL3D_DETECTOR_ENDPOINT": "http://old:20022"},
     "http://127.0.0.1:20022", "default"),
])
def test_endpoint_precedence(experiment, env, endpoint, source):
    assert resolve_detector_endpoint(experiment, environ=env) == {
        "endpoint": endpoint, "source": source, "enabled": bool(endpoint)}


@pytest.mark.parametrize("endpoint", [
    "ftp://host:20024", "http://:20024", "http://host:70000",
    "http://host", "http://user:password@host:20024", "http://host:20024/detect",
])
def test_bad_endpoint_fails_before_generating_inputs(endpoint):
    with pytest.raises(ValueError, match="Invalid detector endpoint"):
        resolve_detector_endpoint({"detector_endpoint": endpoint}, environ={})


def test_explicit_disabled_required_detector_is_error():
    with pytest.raises(ValueError, match="explicitly disabled"):
        resolve_detector_endpoint({"detector_endpoint": ""}, environ={},
                                  require_detector=True)


def test_child_environment_clears_inherited_old_endpoint(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "endpoint_gpu_launcher", root / "scripts/run_gpu_experiment_v11.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setenv("SKILL3D_DETECTOR_ENDPOINT", "http://old:20022")
    monkeypatch.setenv("SKILL3D_DETECTOR_RECEIPT", "/old/receipt")
    monkeypatch.delenv("HARNESS3D_DETECTOR_ENDPOINT", raising=False)
    cfg = {"experiment": {"detector_endpoint": ""},
           "weights": {"sam2": {"checkpoint_file": "sam2.pt", "config": "config.yaml"}}}
    env = launcher._command_env(root, cfg, {"vggt": tmp_path, "sam2": tmp_path})
    assert env["SKILL3D_DETECTOR_ENDPOINT"] == ""
    assert "SKILL3D_DETECTOR_RECEIPT" not in env
