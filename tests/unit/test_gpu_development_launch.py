"""Development launches must execute current source without installing or fetching."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from skill3d.infra.experiment_startup import (
    StartupError, configure_run, environment_contract,
    freeze_working_tree, verify_source_receipt,
)

ROOT = Path(__file__).resolve().parents[2]


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def working_repo(tmp_path):
    repo = tmp_path / "working source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "Launcher Tests")
    names = [
        "start_harness.sh", "start_harness3d.sh", "scripts/bootstrap_and_run_v11.sh",
        "scripts/prepare_gpu_run_v11.py", "scripts/run_gpu_experiment_v11.py",
        "src/skill3d/__init__.py", "src/skill3d/infra/__init__.py",
        "src/skill3d/infra/experiment_startup.py", "pyproject.toml",
        "src/skill3d/online/__init__.py", "src/skill3d/online/config.py",
        "configs/config.yaml", "configs/gpu_experiment_v11.yaml",
        "configs/v11/gpu_experiment_v11.local.yaml", "configs/vsi_bench_split.yaml",
    ]
    for name in names:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (repo / "skill_library").mkdir()
    (repo / "skill_library/SKILL.md").write_text("Original Skill")
    (repo / "src/keep.py").write_text("VALUE = 'before'\n")
    (repo / "src/deleted.py").write_text("VALUE = 'delete me'\n")
    (repo / ".gitignore").write_text(".venv/\ndata/\n*.log\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    (repo / "src/keep.py").write_text("VALUE = 'uncommitted'\n")
    (repo / "src/deleted.py").unlink()
    (repo / "src/new.py").write_text("VALUE = 'new code'\n")
    (repo / "skill_library/SKILL.md").write_text("Uncommitted Skill")
    (repo / ".venv").mkdir()
    (repo / ".venv/large.py").write_text("exclude")
    (repo / "data/vsi_bench_meta").mkdir(parents=True)
    (repo / "data/vsi_bench_meta/test.jsonl").write_text('{"id": "metadata"}\n')
    return repo


def test_snapshot_executes_uncommitted_add_edit_delete_and_is_independent(working_repo, tmp_path):
    receipt = freeze_working_tree(working_repo, tmp_path / "snapshot")
    source = Path(receipt["source_root"])
    assert not (source / "src/deleted.py").exists()
    assert (source / "src/new.py").read_text() == "VALUE = 'new code'\n"
    assert (source / "skill_library/SKILL.md").read_text() == "Uncommitted Skill"
    assert not (source / ".git").exists() and not (source / ".venv").exists()
    assert receipt["input_files"][0]["path"] == "data/vsi_bench_meta/test.jsonl"
    result = subprocess.check_output(
        [sys.executable, "-c", "from keep import VALUE; print(VALUE)"],
        env={**os.environ, "PYTHONPATH": str(source / "src"), "PYTHONDONTWRITEBYTECODE": "1"}, text=True)
    assert result.strip() == "uncommitted"
    (working_repo / "src/keep.py").write_text("VALUE = 'later edit'\n")
    assert "uncommitted" in (source / "src/keep.py").read_text()
    assert verify_source_receipt(source.parent / "code_identity.json", source) == receipt
    assert "src/new.py" in receipt["untracked_source_files"]
    assert "src/deleted.py" in receipt["deleted_source_files"]
    assert "-VALUE = 'before'" in (source.parent / "working_tree.diff").read_text()


def test_snapshot_verification_rejects_extra_or_mutated_code(working_repo, tmp_path):
    receipt = freeze_working_tree(working_repo, tmp_path / "snapshot")
    source = Path(receipt["source_root"])
    (source / "src/injected.py").write_text("bad")
    with pytest.raises(StartupError, match="additional code"):
        verify_source_receipt(source.parent / "code_identity.json", source)
    (source / "src/injected.py").unlink()
    (source / "src/keep.py").chmod(0o644)
    (source / "src/keep.py").write_text("tampered")
    with pytest.raises(StartupError, match="Frozen source changed"):
        verify_source_receipt(source.parent / "code_identity.json", source)


def test_staged_deletion_and_nested_vendor_data_source_are_preserved(working_repo, tmp_path):
    _git(working_repo, "rm", "--", "src/deleted.py")
    nested = working_repo / "third_party/model/training/data/loader.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("def load(): return 'source'\n")
    _git(working_repo, "add", "-f", str(nested.relative_to(working_repo)))
    receipt = freeze_working_tree(working_repo, tmp_path / "snapshot")
    source = Path(receipt["source_root"])
    assert "src/deleted.py" in receipt["deleted_source_files"]
    assert "delete me" in (source.parent / "working_tree.diff").read_text()
    assert (source / nested.relative_to(working_repo)).read_text() == nested.read_text()


def test_install_identity_ignores_source_commit_and_runtime_budget(working_repo):
    cfg = yaml.safe_load((ROOT / "configs/gpu_experiment_v11.yaml").read_text())
    original = environment_contract(working_repo, cfg)["sha256"]
    (working_repo / "src/keep.py").write_text("another program")
    cfg["experiment"].update(seed=138, inner_limit=64)
    cfg["runtime"].update(port=18111, max_model_len=16384)
    cfg["environment"]["min_free_disk_gib"] = 10
    assert environment_contract(working_repo, cfg)["sha256"] == original
    cfg["environment"]["vllm_version"] = "changed"
    assert environment_contract(working_repo, cfg)["sha256"] != original
    cfg["environment"]["vllm_version"] = "0.19.1"
    path = working_repo / "pyproject.toml"
    path.write_text(path.read_text().replace("pydantic>=2.13", "pydantic>=2.14"))
    assert environment_contract(working_repo, cfg)["sha256"] != original


@pytest.mark.parametrize("entry", ["start", "alias", "bootstrap", "python"])
def test_all_three_dry_run_entries_use_dirty_snapshot_without_environment(working_repo, tmp_path, entry):
    work = tmp_path / f"{entry} output"
    cfg = working_repo / "configs/gpu_experiment_v11.yaml"
    common = ["--config", str(cfg), "--work-dir", str(work), "--run-id", "dirty", "--dry-run"]
    if entry == "start":
        command = ["bash", str(working_repo / "start_harness3d.sh"), *common]
    elif entry == "alias":
        command = ["bash", str(working_repo / "start_harness.sh"), *common]
    elif entry == "bootstrap":
        command = ["bash", str(working_repo / "scripts/bootstrap_and_run_v11.sh"), *common]
    else:
        command = [sys.executable, str(working_repo / "scripts/run_gpu_experiment_v11.py"), *common]
    env = {**os.environ, "PYTHON_BIN": sys.executable, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("SKILL3D_CODE_RECEIPT", None)
    result = subprocess.run(command, env=env, cwd=working_repo,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout
    manifest = json.loads((work / "runs/dirty/pipeline_manifest.json").read_text())
    assert manifest["run_mode"] == "development" and manifest["preflight"]["git_dirty"]
    assert manifest["status"] == "dry_run" and manifest["formal_result_eligible"] is False
    assert not (work / "venv").exists() and not (work / "weights").exists()
    source = Path(manifest["code_identity"]["source_root"])
    assert not (source / "src/deleted.py").exists()
    assert (source / "src/keep.py").read_text() == "VALUE = 'uncommitted'\n"
    ablation = next(row["command"] for row in manifest["commands"] if row["stage"] == "b01_b11")
    assert str(source / "scripts/run_skill_ablation_v11.py") in ablation
    assert "--quality-confirmation" not in ablation
    assert manifest["commands"][0]["command"][-1] != "final_test"


def test_formal_entry_rejects_dirty_tree_before_fetching(working_repo, tmp_path):
    result = subprocess.run(
        ["bash", str(working_repo / "start_harness3d.sh"), "--work-dir", str(tmp_path / "formal")],
        env={**os.environ, "PYTHON_BIN": sys.executable}, capture_output=True, text=True)
    assert result.returncode == 2 and "working tree is dirty" in result.stderr
    assert not (tmp_path / "formal/source").exists()


@pytest.mark.parametrize("smoke", [False, True])
def test_metric_panel_route_never_runs_learning_or_evolution(working_repo, tmp_path, smoke):
    path = working_repo / "configs/gpu_experiment_v11.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["weights"]["moge2"]["enabled"] = True
    cfg["experiment"]["metric_validation"] = {
        "panel": "/external/panel.json", "archive_root": "/external/archive"}
    path.write_text(yaml.safe_dump(cfg))
    work = tmp_path / "panel-run"
    command = ["bash", str(working_repo / "start_harness.sh"), "--config", str(path),
               "--work-dir", str(work), "--run-id", "panel", "--dry-run"]
    if smoke:
        command.append("--smoke")
    result = subprocess.run(command, cwd=working_repo,
                            env={**os.environ, "PYTHON_BIN": sys.executable},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((work / "runs/panel/pipeline_manifest.json").read_text())
    stages = [row["stage"] for row in manifest["commands"]]
    assert stages == (["metric_prepare"] if smoke else
                      ["metric_prepare", "serve_qwen", "metric_b01_b11"])
    assert ("--first-scene-only" in manifest["commands"][0]["command"]) == smoke
    assert manifest["formal_result_eligible"] is False


def test_bootstrap_reuses_environment_for_code_edit_and_updates_for_dependency_edit(
    working_repo, tmp_path,
):
    """Exercise real shell control flow with local package/device seams."""
    work = tmp_path / "env-work"
    venv = work / "venv/bin"
    venv.mkdir(parents=True)
    shim = tmp_path / "shim.py"
    shim.write_text(
        "import sys, types\n"
        "sys.modules['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True), "
        "__version__='test', version=types.SimpleNamespace(cuda='test'))\n"
        "sys.modules['vllm'] = types.SimpleNamespace(__version__='test')\n"
        "import skill3d.env_preflight as preflight\n"
        "preflight.assert_runtime_dependencies=lambda **kwargs: {}\n"
        "exec(sys.stdin.read())\n")
    # The fixture's runtime is intentionally tiny. Verify the code the shell
    # chooses and keep model execution out of this environment reuse contract.
    runtime = working_repo / "scripts/run_gpu_experiment_v11.py"
    runtime.write_text(
        "import json, os, pathlib, sys\n"
        "root=pathlib.Path(__file__).resolve().parents[1]\n"
        "out=pathlib.Path(os.environ['TEST_RUN_RECORD'])\n"
        "out.write_text(json.dumps({'source':str(root),'value':(root/'src/keep.py').read_text()}))\n")
    preflight = working_repo / "src/skill3d/env_preflight.py"
    preflight.write_text("def assert_runtime_dependencies(**kwargs): return {}\n")
    python = venv / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ "$1" == "-" ]]; then shift; exec "{sys.executable}" "{shim}" "$@"; fi\n'
        f'exec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    pip = venv / "pip"
    pip.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"freeze\" ]]; then printf 'fixture==1\\n'; else printf '%s\\n' \"$*\" >> \"$TEST_PIP_LOG\"; fi\n")
    pip.chmod(0o755)
    cfg_path = working_repo / "configs/gpu_experiment_v11.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    marker = work / "environment/installed_contract_sha256.txt"
    marker.parent.mkdir()
    marker.write_text(environment_contract(working_repo, cfg)["sha256"] + "\n")
    pip_log, record = tmp_path / "pip.log", tmp_path / "execution.json"
    env = {**os.environ, "PYTHON_BIN": sys.executable, "TEST_PIP_LOG": str(pip_log),
           "TEST_RUN_RECORD": str(record)}
    def launch(name):
        result = subprocess.run(
            ["bash", str(working_repo / "scripts/bootstrap_and_run_v11.sh"),
             "--dev", "--config", str(cfg_path), "--work-dir", str(work), "--run-id", name],
            env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr + result.stdout
    launch("edit")
    assert not pip_log.exists()
    executed = json.loads(record.read_text())
    assert "source_snapshots/edit/source" in executed["source"]
    assert executed["value"] == "VALUE = 'uncommitted'\n"
    project = working_repo / "pyproject.toml"
    project.write_text(project.read_text().replace("pydantic>=2.13", "pydantic>=2.14"))
    launch("dependency")
    assert "pydantic>=2.14" in pip_log.read_text()
    assert marker.read_text().strip() == environment_contract(working_repo, cfg)["sha256"]


def test_mode_configuration_keeps_dev_budget_and_smoke_contract():
    cfg = yaml.safe_load((ROOT / "configs/gpu_experiment_v11.yaml").read_text())
    dev = configure_run(cfg, dev=True)
    assert dev["experiment"]["inner_limit"] == cfg["experiment"]["inner_limit"]
    assert dev["experiment"]["quality_confirmation"] == "" and not dev["experiment"]["formal"]
    smoke = configure_run(cfg, smoke=True)
    assert (smoke["experiment"]["induction_limit"], smoke["experiment"]["inner_limit"]) == (2, 2)
    assert smoke["experiment"]["run_evolution"] is False
    assert dev["runtime"] == smoke["runtime"] == cfg["runtime"]


def test_existing_environment_requires_explicit_dependency_exception(working_repo, monkeypatch):
    import importlib.metadata as metadata
    import types
    from skill3d.infra.experiment_startup import validate_existing_environment

    (working_repo / "pyproject.toml").write_text(
        '[build-system]\nrequires=["setuptools"]\n[project]\n'
        'dependencies=["pydantic>=2.13"]\nrequires-python=">=3.11"\n')
    cfg = yaml.safe_load((ROOT / "configs/gpu_experiment_v11.yaml").read_text())
    cfg["environment"]["python"] = f"{sys.version_info.major}.{sys.version_info.minor}"
    versions = {"pydantic": "2.12.5", "vllm": cfg["environment"]["vllm_version"]}
    monkeypatch.setattr(metadata, "version", versions.__getitem__)
    monkeypatch.setattr(metadata, "distributions", lambda: [])
    sam = types.ModuleType("sam2")
    sam.__file__ = __file__
    monkeypatch.setitem(sys.modules, "sam2", sam)
    with pytest.raises(StartupError, match="does not satisfy"):
        validate_existing_environment(working_repo, cfg)
    cfg["environment"]["existing_dependency_versions"] = {"pydantic": "2.12.5"}
    receipt = validate_existing_environment(working_repo, cfg)
    assert receipt["mode"] == "validated_existing"
    assert receipt["declared_dependency_exceptions"]["pydantic"]["installed"] == "2.12.5"
    versions["pydantic"] = "2.12.4"
    with pytest.raises(StartupError, match="explicit version pin"):
        validate_existing_environment(working_repo, cfg)
