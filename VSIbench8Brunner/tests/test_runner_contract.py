import copy
import importlib.util
import json
from pathlib import Path

import pytest

from skill3d.infra.experiment_startup import freeze_working_tree, verify_source_receipt, StartupError
from skill3d.infra.m5_identity import content_sha256
from skill3d.evaluation.qwen8b_published_fitset import read_manifest, SCHEMA

ROOT = Path(__file__).resolve().parents[1]


def test_standalone_source_snapshot_preserves_and_checks_every_code_file(tmp_path):
    source = tmp_path / "runner"
    source.mkdir()
    (source / "bundle_origin.json").write_text(json.dumps({
        "schema": "vsi8b-source-bundle/1", "upstream_commit": "a"*40}))
    (source / "run.py").write_text("x = 1\n")
    for name in ("S01", "S01-g0m0"):
        (source / "skills" / name).mkdir(parents=True)
        (source / "skills" / name / "skill.txt").write_text(name)
    (source / "prepared_cache").mkdir()
    (source / "prepared_cache" / "private.json").write_text("{}")
    receipt = freeze_working_tree(source, tmp_path / "snapshot")
    assert receipt["source_policy"] == "content-frozen-local-bundle"
    assert receipt["git_dirty"] is True
    frozen = tmp_path / "snapshot/source"
    assert not (frozen / "prepared_cache").exists()
    verify_source_receipt(tmp_path / "snapshot/code_identity.json", frozen)
    (frozen / "run.py").chmod(0o644)
    (frozen / "run.py").write_text("x = 2\n")
    with pytest.raises(StartupError):
        verify_source_receipt(tmp_path / "snapshot/code_identity.json", frozen)


def test_refreshed_binding_manifest_rejects_stale_or_incomplete_cache(tmp_path):
    path = tmp_path / "manifest.json"
    body = {"schema": SCHEMA, "status": "complete", "bindings": {"a": {}}}
    path.write_text(json.dumps({**body, "sha256": content_sha256(body)}))
    assert read_manifest(path)["status"] == "complete"
    body["status"] = "started"
    path.write_text(json.dumps({**body, "sha256": content_sha256(body)}))
    with pytest.raises(ValueError):
        read_manifest(path)
    body["status"] = "complete"
    path.write_text(json.dumps({**body, "sha256": "a"*64}))
    with pytest.raises(ValueError):
        read_manifest(path)


def test_profile_preserves_every_original_experiment_setting():
    old = json.loads((ROOT / "configs/experiments/vsi_m4_1000.json").read_text())
    new = json.loads((ROOT / "configs/experiments/vsi8b.json").read_text())
    new["inputs"]["online_config"] = old["inputs"]["online_config"]
    new["inputs"]["derived_binding_root"] = old["inputs"]["derived_binding_root"]
    new["runtime"]["binding_derivation_audit_mode"] = old["runtime"]["binding_derivation_audit_mode"]
    assert new == old


def test_system_stages_use_read_only_refreshed_cache_and_original_solver_arguments():
    from skill3d.infra.experiment_stages_v11 import system_argv
    new = json.loads((ROOT / "configs/experiments/vsi8b.json").read_text())
    manifest = {"run_root": "/tmp/vsi8b-test-run", "profile": new}
    argv = system_argv(manifest, "initial_validation", preflight=True)
    assert argv[argv.index("--binding-derivation-audit-mode") + 1] == "qwen8b_refresh"
    assert "--preflight-only" in argv
    assert "--scene-overview-manifest" in argv
    assert "--derived-binding-root" in argv
    assert argv[argv.index("--phase") + 1] == "final_system"
