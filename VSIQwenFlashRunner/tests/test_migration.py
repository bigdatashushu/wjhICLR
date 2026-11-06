"""Offline migration checks using synthetic inputs and real child processes."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from skill3d.infra import experiment_campaign as campaign
from skill3d.infra import experiment_stages_v11 as stages
from skill3d.infra import vsi_runner_source as source
from skill3d.infra.experiment_startup import verify_source_receipt

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT.parent


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def offline_env():
    env = {key: value for key, value in os.environ.items() if not any(
        word in key.upper() for word in (
            "KEY", "TOKEN", "SECRET", "PASSWORD", "ENDPOINT", "PROXY",
            "SKILL3D_CODE_RECEIPT", "SKILL3D_RUN_MODE"))}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    return env


def test_default_profile_and_local_stage_wiring():
    original = json.loads((SHARED / "configs/experiments/vsi_m4_1000.json").read_text())
    expected = json.loads(json.dumps(original))
    expected["runtime"]["expected_scenes"] = 160
    expected["inputs"]["seed_library"] = "{seed_root}"
    expected["inputs"].pop("derived_binding_root")
    expected["runtime"]["derived_binding_root"] = (
        "{run_root}/prepared_cache/derived_bindings_v4_metric")
    assert json.loads((ROOT / "configs/vsi_m4_1000.json").read_text()) == expected
    _, run_root, _, profile, _, profile_path, context = campaign.resolve_new(
        campaign.parser().parse_args(["--dry-run"]))
    assert profile_path == ROOT / "configs/vsi_m4_1000.json"
    assert run_root.is_relative_to(ROOT / "runs")
    assert Path(context["data_root"]) == (
        ROOT / "prepared_cache/vsibench-m4-success-g1m1-1000-seed137-metric-r02")
    assert Path(profile["inputs"]["seed_library"]) == source.seed_library_root(ROOT)
    assert Path(stages.fitset_cli().__file__) == ROOT / "scripts/run_fitset_v11.py"
    manifest = {"profile": profile, "run_root": str(run_root)}
    cli = stages.fitset_cli()
    learning = cli.build_parser().parse_args(stages.system_argv(manifest, "learning"))
    assert learning.gate_mode == "train_sample" and learning.phase == "accumulation"
    assert learning.seed == 137 and learning.gate_size == 8
    assert Path(learning.derived_binding_root).is_relative_to(run_root)
    assert "derived_binding_root" not in profile["inputs"]
    for stage in ("initial_validation", "final_validation"):
        args = cli.build_parser().parse_args(stages.system_argv(manifest, stage))
        assert args.execution_role == "validation" and args.phase == "final_system"
    assert profile["baseline"] == original["baseline"]


def test_loaded_core_is_local_and_utilities_are_shared():
    manifest = json.loads((ROOT / "migration.json").read_text())
    for row in manifest["files"]:
        if not row["path"].startswith("src/"):
            continue
        module = importlib.import_module(
            row["path"].removeprefix("src/").removesuffix(".py").replace("/", "."))
        assert Path(module.__file__) == ROOT / row["path"]
        assert hashlib.sha256((SHARED / row["path"]).read_bytes()).hexdigest() == row["original_sha256"]
    for name in ("skill3d.adapters.vsibench_loader", "skill3d.tools.category_match",
                 "skill3d.online.config", "skill3d.segmentation.question_binding"):
        assert Path(importlib.import_module(name).__file__).is_relative_to(SHARED / "src")


def test_default_preview_works_with_only_standard_library(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(ROOT / "run.py"), "--dry-run"],
        cwd=tmp_path, env=offline_env(), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    assert preview["model_requests"] == 0
    assert len(preview["stages"]) == 9
    assert not Path(preview["run_root"]).exists()
    assert all(Path(row["argv"][2]).is_relative_to(Path(preview["run_root"]) / "code/source")
               for row in preview["stages"])
    assert list(tmp_path.iterdir()) == []


def test_output_outside_runner_is_rejected(tmp_path):
    with pytest.raises(campaign.CampaignError, match="inside VSIQwenFlashRunner"):
        campaign.resolve_new(campaign.parser().parse_args(
            ["--dry-run", "--output-dir", str(tmp_path / "outside")]))


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    shared, local = tmp_path / "repo", tmp_path / "repo/VSIQwenFlashRunner"
    (shared / "src/skill3d/infra").mkdir(parents=True)
    shutil.copyfile(SHARED / "src/skill3d/infra/experiment_startup.py",
                    shared / "src/skill3d/infra/experiment_startup.py")
    (shared / "src/skill3d/shared_probe.py").write_text("VALUE = 'shared-original'\n")
    save(shared / "skill_library/snapshot.json", {"synthetic": True})
    (shared / "configs").mkdir()
    (shared / "configs/config.yaml").write_text("synthetic: true\n")
    (shared / "configs/vsi_bench_split.yaml").write_text("synthetic: true\n")
    (shared / "pyproject.toml").write_text("[project]\nname = 'synthetic'\n")
    for relative in ("src", "scripts", "configs"):
        (local / relative).mkdir(parents=True)
    for path in (ROOT / "src").rglob("*.py"):
        # The synthetic command backend needs only the source manager/controller.
        relative = path.relative_to(ROOT)
        if ("infra" in relative.parts and path.name in (
            "__init__.py", "vsi_runner_source.py", "experiment_campaign.py")
                or relative.as_posix() == "src/skill3d/__init__.py"):
            target = local / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    shutil.copyfile(ROOT / "run.py", local / "run.py")
    shutil.copyfile(ROOT / "scripts/run_experiment.py", local / "scripts/run_experiment.py")
    (local / "src/skill3d/shared_probe.py").write_text("VALUE = 'local-override'\n")
    (local / "scripts/fake_stage.py").write_text(
        "import argparse,json,pathlib,sys\n"
        "sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))\n"
        "from skill3d.shared_probe import VALUE\n"
        "p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--stage');"
        "p.add_argument('--mode',default='ok');p.add_argument('--resume',action='store_true');"
        "a=p.parse_args();assert VALUE=='local-override'\n"
        "r=pathlib.Path(a.root);d=r/'stages'/a.stage;d.mkdir(parents=True,exist_ok=True)\n"
        "with (r/'dispatches.txt').open('a') as f:f.write(a.stage+'\\n')\n"
        "if a.mode=='exit':sys.exit(7)\n"
        "if a.mode=='missing':sys.exit(0)\n"
        "(d/'receipt.json').write_text(json.dumps({'status':'complete','stage':a.stage}))\n"
        "if a.mode=='receipt_then_exit':sys.exit(7)\n")
    data = tmp_path / "inputs"
    data.mkdir()
    (data / "anchor.txt").write_text("immutable synthetic input")
    profile = {
        "schema": campaign.PROFILE_SCHEMA, "backend": "commands",
        "inputs": {"anchor": "{data_root}/anchor.txt"},
        "stages": [{
            "name": name,
            "argv": ["{python}", "-B", "{source_root}/scripts/fake_stage.py",
                     "--root", "{run_root}", "--stage", name],
            "resume": "receipt",
            "receipt": {"path": "{run_root}/stages/" + name + "/receipt.json",
                        "equals": {"status": "complete", "stage": name}},
            "artifacts": ["{run_root}/stages/" + name],
        } for name in campaign.STAGES],
    }
    config = local / "configs/profile.json"
    registry = local / "configs/datasets.json"
    save(config, profile)
    save(registry, {"schema": "harness3d-dataset-registry/1", "datasets": {
        "vsi-bench": {"aliases": ["vsi"], "profile": str(config),
                      "data_root": str(data), "description": "synthetic"}}})
    output = local / "runs/synthetic"
    monkeypatch.setattr(campaign, "ROOT", local)
    monkeypatch.setattr(source, "_git_identity", lambda _root: ("a" * 40, True))
    monkeypatch.setattr(campaign, "child_environment", lambda run: {
        **offline_env(), "PYTHONPATH": str(run / "code/source/src"),
        "SKILL3D_CODE_RECEIPT": str(run / "code/code_identity.json"), "SKILL3D_RUN_MODE": "formal"})
    return {"shared": shared, "local": local, "data": data, "profile": profile,
            "config": config, "output": output,
            "argv": ["--registry", str(registry), "--output-dir", str(output)]}


def test_nine_real_children_use_frozen_overrides_and_resume_without_dispatch(synthetic):
    case = synthetic
    assert campaign.main(case["argv"]) == 0
    assert (case["output"] / "dispatches.txt").read_text().splitlines() == list(campaign.STAGES)
    receipt = verify_source_receipt(case["output"] / "code/code_identity.json",
                                    case["output"] / "code/source")
    assert receipt["git_dirty"] is True
    assert receipt["file_origins"]["src/skill3d/shared_probe.py"] == "local"
    assert (case["shared"] / "src/skill3d/shared_probe.py").read_text() == "VALUE = 'shared-original'\n"
    # Existing runs survive later changes to mutable local/shared development code.
    (case["local"] / "src/skill3d/shared_probe.py").write_text("changed-local\n")
    (case["shared"] / "src/skill3d/shared_probe.py").write_text("changed-shared\n")
    assert campaign.main(["--output-dir", str(case["output"]), "--resume"]) == 0
    assert len((case["output"] / "dispatches.txt").read_text().splitlines()) == 9


@pytest.mark.parametrize("mode", ["missing", "exit"])
def test_interruption_does_not_automatically_repeat_requests(synthetic, mode):
    case = synthetic
    case["profile"]["stages"][1]["argv"] += ["--mode", mode]
    save(case["config"], case["profile"])
    assert campaign.main(case["argv"]) == 2
    assert campaign.main(["--output-dir", str(case["output"]), "--resume"]) == 2
    assert (case["output"] / "dispatches.txt").read_text().splitlines() == ["seed", "initial_validation"]


def test_complete_receipt_after_crash_is_adopted_without_repeat(synthetic):
    case = synthetic
    case["profile"]["stages"][1]["argv"] += ["--mode", "receipt_then_exit"]
    save(case["config"], case["profile"])
    assert campaign.main(case["argv"]) == 2
    assert campaign.main(["--output-dir", str(case["output"]), "--resume"]) == 0
    assert (case["output"] / "dispatches.txt").read_text().splitlines() == list(campaign.STAGES)


def test_unresolved_request_blocks_resume_before_dispatch(synthetic):
    case = synthetic
    case["profile"]["stages"][1]["argv"] += ["--mode", "receipt_then_exit"]
    save(case["config"], case["profile"])
    assert campaign.main(case["argv"]) == 2
    save(case["output"] / "requests/state.json", {"requests": [{"status": "started"}]})
    assert campaign.main(["--output-dir", str(case["output"]), "--resume"]) == 2
    assert (case["output"] / "dispatches.txt").read_text().splitlines() == ["seed", "initial_validation"]


def test_resume_refuses_other_runner_snapshot_policy(synthetic):
    case = synthetic
    assert campaign.main(case["argv"]) == 0
    path = case["output"] / "code/code_identity.json"
    value = json.loads(path.read_text())
    value.pop("snapshot_policy")
    save(path, value)
    assert campaign.main(["--output-dir", str(case["output"]), "--resume"]) == 2
    assert len((case["output"] / "dispatches.txt").read_text().splitlines()) == 9


@pytest.mark.parametrize("change", ["source", "input", "artifact", "extra_source", "manifest"])
def test_tampered_run_is_refused_before_child_dispatch(synthetic, change):
    case = synthetic
    assert campaign.main(case["argv"]) == 0
    output = case["output"]
    if change == "source":
        path = output / "code/source/scripts/fake_stage.py"
        path.chmod(0o644)
        path.write_text("changed")
    elif change == "extra_source":
        (output / "code/source/src/skill3d/extra.py").write_text("changed")
    elif change == "input":
        (case["data"] / "anchor.txt").write_text("changed")
    elif change == "artifact":
        (output / "stages/seed/receipt.json").write_text("{}")
    else:
        manifest = json.loads((output / "campaign_manifest.json").read_text())
        manifest["dataset"] = "changed"
        save(output / "campaign_manifest.json", manifest)
    assert campaign.main(["--output-dir", str(output), "--resume"]) == 2
    assert len((output / "dispatches.txt").read_text().splitlines()) == 9


def test_frozen_package_cannot_fall_back_to_mutable_shared_code(synthetic, tmp_path):
    case = synthetic
    receipt = source.freeze_source(case["local"], case["output"] / "code")
    snapshot = Path(receipt["source_root"])
    (case["shared"] / "src/skill3d/outside_probe.py").write_text("VALUE = 'outside'\n")
    env = offline_env()
    env["PYTHONPATH"] = str(snapshot / "src") + os.pathsep + str(case["shared"] / "src")
    result = subprocess.run(
        [sys.executable, "-B", "-c",
         "import skill3d,importlib.util; "
         "assert len(skill3d.__path__)==1; "
         "assert importlib.util.find_spec('skill3d.outside_probe') is None"],
        cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_source_freeze_excludes_credentials_other_runners_and_outputs(synthetic):
    case = synthetic
    (case["shared"] / "qwenkey.md").write_text("synthetic secret must stay outside")
    save(case["shared"] / "BLINK/config.json", {"synthetic": True})
    save(case["local"] / "runs/old/output.json", {"synthetic": True})
    (case["local"] / "configs/credentials.local.env").write_text("SYNTHETIC_SECRET=private")
    receipt = source.freeze_source(case["local"], case["output"] / "code")
    names = {row["path"] for row in receipt["files"]}
    assert not any("qwenkey" in name or "BLINK" in name or "runs/" in name or ".local.env" in name
                   for name in names)
    verify_source_receipt(case["output"] / "code/code_identity.json",
                          case["output"] / "code/source")


def test_local_seed_catalog_replaces_shared_seed_in_frozen_source(synthetic):
    case = synthetic
    local_seed = case["local"] / "skill_library"
    save(local_seed / "snapshots/active_snapshot.json", {"snapshot_id": "local-seed"})
    (local_seed / "SKILL.md").write_text("runner-owned method\n")
    assert source.seed_library_root(case["local"]) == local_seed
    receipt = source.freeze_source(case["local"], case["output"] / "code")
    snapshot = Path(receipt["source_root"])
    assert (snapshot / "skill_library/SKILL.md").read_text() == "runner-owned method\n"
    assert not (snapshot / "skill_library/snapshot.json").exists()
    assert receipt["file_origins"]["skill_library/SKILL.md"] == "local"
    assert source.seed_library_root(snapshot) == snapshot / "skill_library"
    verify_source_receipt(case["output"] / "code/code_identity.json", snapshot)


def test_normal_seed_stage_delivers_selected_catalog_without_revision(tmp_path):
    from skill3d.evaluation.fitset_v11 import load_fit_skill_bundle
    from skill3d.evolution.fitset_revision_v11 import load_active_parent_slots

    _, _, _, profile, _, _, _ = campaign.resolve_new(
        campaign.parser().parse_args(["--dry-run"]))
    parents, identity = load_active_parent_slots(profile["inputs"]["seed_library"])
    manifest = {"profile": profile, "identity_sha256": "1" * 64}
    receipt = stages.seed_bundle(manifest, tmp_path / "seed")
    bundle = load_fit_skill_bundle(tmp_path / "seed/bundle/manifest.json")
    assert receipt["physical_slots"] == len(parents) == 24
    assert receipt["bundle_sha256"] == bundle.manifest_sha256
    assert bundle.manifest["source_run_identity"]["seed_snapshot"] == identity
    for slot, parent in parents.items():
        delivered = bundle.slots[slot]
        assert (delivered.skill_id, delivered.version, delivered.content_sha256) == (
            parent.skill_id, parent.version, parent.content_sha256)
        assert delivered.skill_md == parent.skill_md
        assert delivered.role != "revised_candidate"


def test_real_snapshot_loads_pipeline_and_preserves_shared_m5_sources(tmp_path):
    receipt = source.freeze_source(ROOT, tmp_path / "code")
    snapshot = Path(receipt["source_root"])
    verify_source_receipt(tmp_path / "code/code_identity.json", snapshot)
    from skill3d.infra.m5_identity import source_identity
    original_m5 = source_identity(SHARED / "src/skill3d")
    # Preserve the original source identities, including retained absent refs.
    assert source_identity(snapshot / "src/skill3d") == original_m5
    for relative in ("src/skill3d/online/vsi_kernel.py",
                     "src/skill3d/online/vsi_execution_contract.py",
                     "src/skill3d/sandbox/ast_guard.py"):
        assert receipt["file_origins"][relative] == "local"
        assert source.sha256(snapshot / relative) == source.sha256(ROOT / relative)
    seed_root = source.seed_library_root(ROOT)
    original_seed = {path.relative_to(seed_root).as_posix(): source.sha256(path)
                     for path in seed_root.rglob("*") if path.is_file()}
    frozen_seed = {path.relative_to(snapshot / "skill_library").as_posix(): source.sha256(path)
                   for path in (snapshot / "skill_library").rglob("*") if path.is_file()}
    assert frozen_seed == original_seed
    code = (
        "import sys,pathlib;root=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(root/'src'));"
        "from skill3d.infra import experiment_stages_v11 as stages;"
        "from skill3d.online import runner,config,vsi_kernel,vsi_execution_contract;"
        "from skill3d.sandbox import ast_guard;"
        "from skill3d.evolution import fitset_stream_v11,fitset_revision_v11;"
        "from skill3d.evaluation import fitset_v11;"
        "assert all(pathlib.Path(m.__file__).is_relative_to(root) for m in "
        "(stages,runner,config,vsi_kernel,vsi_execution_contract,ast_guard,"
        "fitset_stream_v11,fitset_revision_v11,fitset_v11));"
        "assert runner.RestrictedNamespaceKernel is vsi_kernel.RestrictedNamespaceKernel;"
        "assert ast_guard.ast_guard('ReturnAnswer(MeasurementAnswer(\"actual-id\"))',set()).ok;"
        "assert pathlib.Path(stages.fitset_cli().__file__)==root/'scripts/run_fitset_v11.py';"
        "assert config.SOURCE_ROOT==root;"
        "assert fitset_v11._source_code_identity()['source_sha256']==sys.argv[2]"
    )
    env = offline_env()
    env.update(SKILL3D_CODE_RECEIPT=str(tmp_path / "code/code_identity.json"),
               SKILL3D_RUN_MODE="formal")
    result = subprocess.run([sys.executable, "-B", "-c", code, str(snapshot), receipt["source_sha256"]],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
