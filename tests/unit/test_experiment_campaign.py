"""Synthetic campaign tests: no models, real data, GPUs, or remote services."""

from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from skill3d.infra import experiment_campaign as campaign
from skill3d.infra import experiment_stages_v11 as stages
from test_qa_partition_v11 import qa_partition_case

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    # Enough of the real launcher/source freezer for real subprocess execution.
    for name in ("scripts/run_experiment.py", "src/skill3d/infra/experiment_campaign.py",
                 "src/skill3d/infra/experiment_startup.py"):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (source / "scripts/run_gpu_experiment_v11.py").write_text("# synthetic runner\n")
    (source / "scripts/fake_stage.py").write_text(
        "import argparse,json,pathlib,sys\n"
        "p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--stage');"
        "p.add_argument('--mode',default='ok');p.add_argument('--resume',action='store_true');"
        "a=p.parse_args()\n"
        "r=pathlib.Path(a.root);d=r/'stages'/a.stage;d.mkdir(parents=True,exist_ok=True)\n"
        "with (r/'dispatches.txt').open('a') as f:f.write(a.stage+'\\n')\n"
        "if a.mode=='exit':sys.exit(7)\n"
        "if a.mode=='missing':sys.exit(0)\n"
        "(d/'receipt.json').write_text(json.dumps({'status':'complete','stage':a.stage}))\n"
        "if a.mode=='receipt_then_exit':sys.exit(7)\n")
    (source / ".gitignore").write_text("/runs/\n__pycache__/\n")
    for argv in (["git", "init", "-q"], ["git", "add", "."],
                 ["git", "-c", "user.name=Synthetic", "-c", "user.email=synthetic@example.invalid",
                  "commit", "-qm", "synthetic fixture"]):
        subprocess.run(argv, cwd=source, check=True, capture_output=True)
    data = tmp_path / "inputs"
    data.mkdir()
    (data / "anchor.txt").write_text("synthetic immutable input")
    config = tmp_path / "profile.json"
    registry = tmp_path / "datasets.json"
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
    save(config, profile)
    save(registry, {"schema": "harness3d-dataset-registry/1", "datasets": {
        "vsi-bench": {"aliases": ["vsi"], "profile": str(config), "data_root": str(data),
                      "description": "synthetic"},
        "revsi": {"aliases": [], "profile": str(config), "data_root": str(data),
                  "description": "synthetic"},
    }})
    output = tmp_path / "new-campaign"
    monkeypatch.setattr(campaign, "ROOT", source)
    monkeypatch.setattr(socket, "create_connection",
                        lambda *_a, **_k: pytest.fail("Unexpected network access"))
    argv = ["--registry", str(registry), "--dataset", "vsi", "--output-dir", str(output)]
    return {"source": source, "data": data, "config": config, "registry": registry,
            "output": output, "profile": profile, "argv": argv}


def test_nine_real_subprocess_stages_and_completed_resume(isolated):
    case = isolated
    assert campaign.main(case["argv"]) == 0
    root = case["output"]
    assert (root / "dispatches.txt").read_text().splitlines() == list(campaign.STAGES)
    assert campaign.read_json(root / "campaign_state.json")["status"] == "complete"
    assert campaign.main(["--output-dir", str(root), "--resume"]) == 0
    assert len((root / "dispatches.txt").read_text().splitlines()) == 9
    assert not (case["source"] / "runs").exists()
    assert campaign.git(case["source"], "status", "--porcelain") == ""


def test_dry_run_is_dependency_free_and_makes_no_files(isolated, monkeypatch, capsys):
    def forbidden(*_a, **_k):
        pytest.fail("Dry run performed execution, writes, hashes, or imports")
    monkeypatch.setattr(campaign, "pins", forbidden)
    monkeypatch.setattr(campaign, "atomic", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert campaign.main([*isolated["argv"], "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["dataset"] == "vsi-bench"
    assert preview["model_requests"] == 0
    assert not isolated["output"].exists()


def test_local_check_is_readonly(isolated):
    assert campaign.main([*isolated["argv"], "--check"]) == 0
    assert not isolated["output"].exists()


@pytest.mark.parametrize("mode", ["missing", "exit"])
def test_no_receipt_or_nonzero_exit_stops_and_refuses_blind_resume(isolated, mode):
    profile = isolated["profile"]
    profile["stages"][1]["argv"] += ["--mode", mode]
    save(isolated["config"], profile)
    assert campaign.main(isolated["argv"]) == 2
    root = isolated["output"]
    assert (root / "dispatches.txt").read_text().splitlines() == ["seed", "initial_validation"]
    assert campaign.main(["--output-dir", str(root), "--resume"]) == 2
    assert len((root / "dispatches.txt").read_text().splitlines()) == 2


def test_valid_receipt_after_crash_is_adopted_without_dispatch(isolated):
    profile = isolated["profile"]
    profile["stages"][1]["argv"] += ["--mode", "receipt_then_exit"]
    save(isolated["config"], profile)
    assert campaign.main(isolated["argv"]) == 2
    assert campaign.main(["--output-dir", str(isolated["output"]), "--resume"]) == 0
    assert (isolated["output"] / "dispatches.txt").read_text().splitlines() == list(campaign.STAGES)


def test_unresolved_paid_request_prevents_continuation(isolated):
    profile = isolated["profile"]
    profile["stages"][1]["argv"] += ["--mode", "receipt_then_exit"]
    save(isolated["config"], profile)
    assert campaign.main(isolated["argv"]) == 2
    root = isolated["output"]
    save(root / "requests/state.json", {"requests": [{"status": "started", "request_id": 0}]})
    assert campaign.main(["--output-dir", str(root), "--resume"]) == 2
    assert len((root / "dispatches.txt").read_text().splitlines()) == 2


@pytest.mark.parametrize("change", ["input", "artifact", "extra_artifact", "source", "manifest"])
def test_changed_identities_fail_before_any_child_dispatch(isolated, change):
    assert campaign.main(isolated["argv"]) == 0
    root = isolated["output"]
    if change == "input":
        (isolated["data"] / "anchor.txt").write_text("changed")
    elif change == "artifact":
        save(root / "stages/seed/receipt.json", {"status": "complete", "stage": "seed", "changed": True})
    elif change == "extra_artifact":
        (root / "stages/seed/extra.txt").write_text("unexpected")
    elif change == "source":
        path = root / "code/source/scripts/fake_stage.py"
        path.chmod(0o644)
        path.write_text("changed")
    else:
        path = root / "campaign_manifest.json"
        value = campaign.read_json(path)
        value["dataset"] = "revsi"
        save(path, value)
    assert campaign.main(["--output-dir", str(root), "--resume"]) == 2
    assert len((root / "dispatches.txt").read_text().splitlines()) == 9


def test_existing_directory_and_dataset_collision_are_refused(isolated):
    assert campaign.main(isolated["argv"]) == 0
    assert campaign.main(isolated["argv"]) == 2
    assert campaign.main(["--output-dir", str(isolated["output"]), "--resume", "--dataset", "revsi"]) == 2


def test_run_lock_prevents_second_owner(isolated):
    assert campaign.main(isolated["argv"]) == 0
    root = isolated["output"]
    with (root / "campaign.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert campaign.main(["--output-dir", str(root), "--resume"]) == 2


def test_command_argv_has_no_shell_evaluation(isolated):
    payload = "$(touch should-never-exist) `touch also-never-exists`"
    profile = isolated["profile"]
    profile["stages"][0]["argv"] += ["--mode", payload]
    save(isolated["config"], profile)
    assert campaign.main(isolated["argv"]) == 0
    assert not (isolated["output"] / "code/source/should-never-exist").exists()
    assert not (isolated["output"] / "code/source/also-never-exists").exists()


def test_invalid_templates_and_inline_credentials_are_refused(isolated):
    profile = isolated["profile"]
    profile["stages"][0]["argv"].append("{unknown}")
    save(isolated["config"], profile)
    assert campaign.main([*isolated["argv"], "--dry-run"]) == 2
    profile["stages"][0]["argv"].pop()
    profile["api_key"] = "synthetic-secret"
    save(isolated["config"], profile)
    assert campaign.main([*isolated["argv"], "--dry-run"]) == 2


def native_manifest(tmp_path):
    profile = campaign.read_json(ROOT / "configs/experiments/prepared.json")
    context = {"data_root": str(tmp_path / "inputs"), "source_root": str(ROOT),
               "run_root": str(tmp_path / "run"), "python": sys.executable, "dataset": "vsi-bench"}
    return {"profile": campaign.expand(profile, context), "run_root": context["run_root"]}


@pytest.mark.parametrize("alias", [None, "vsi", "vsibench", "vsi-m4-1000"])
def test_default_vsi_profile_selects_1000_train_test_subset(alias, tmp_path):
    argv = ["--dry-run", "--output-dir", str(tmp_path / "new-run")]
    if alias is not None:
        argv.extend(["--dataset", alias])
    dataset, _, _, profile, _, profile_path, context = campaign.resolve_new(
        campaign.parser().parse_args(argv))
    assert dataset == "vsi-bench"
    assert profile_path.name == "vsi_m4_1000.json"
    assert Path(context["data_root"]) == ROOT.parent / "vsibench-m4-success-1000"
    runtime = profile["runtime"]
    assert runtime["gate_mode"] == "train_sample"
    assert (runtime["expected_train_questions"], runtime["expected_test_questions"],
            runtime["expected_scenes"]) == (300, 700, 164)
    assert runtime["learning_input_mode"] == "analysis_synthesis_v1"
    assert runtime["max_requests"] == 0


def test_new_vsi_stages_use_training_gate_and_test_only_validation(tmp_path):
    template = campaign.read_json(ROOT / "configs/experiments/vsi_m4_1000.json")
    context = {"data_root": str(tmp_path / "inputs"), "source_root": str(ROOT),
               "run_root": str(tmp_path / "run"), "python": sys.executable,
               "dataset": "vsi-bench"}
    manifest = {"profile": campaign.expand(template, context), "run_root": context["run_root"]}
    cli = stages.fitset_cli()
    learning = cli.build_parser().parse_args(stages.system_argv(manifest, "learning", preflight=True))
    assert learning.gate_mode == "train_sample" and learning.phase == "accumulation"
    assert learning.execution_role == ""
    for phase in ("initial_validation", "final_validation"):
        args = cli.build_parser().parse_args(stages.system_argv(manifest, phase, preflight=True))
        assert args.phase == "final_system" and args.execution_role == "validation"
        assert args.gate_mode == "scene_disjoint"
    assert manifest["profile"]["baseline"]["scope"] == "panel"


def test_optional_overview_reaches_system_stages_and_preserves_baseline(tmp_path):
    template = campaign.read_json(ROOT / "configs/experiments/vsi_m4_1000.json")
    template["inputs"]["scene_overviews"] = "{data_root}/scene_overviews"
    campaign.validate_profile(template)
    context = {"data_root": str(tmp_path / "inputs"), "source_root": str(ROOT),
               "run_root": str(tmp_path / "run"), "python": sys.executable,
               "dataset": "vsi-bench"}
    manifest = {"profile": campaign.expand(template, context), "run_root": context["run_root"]}
    for phase in ("learning", "initial_validation", "final_validation"):
        args = stages.fitset_cli().build_parser().parse_args(
            stages.system_argv(manifest, phase, preflight=True))
        assert args.scene_overview_manifest == str(
            tmp_path / "inputs/scene_overviews/manifest.json")
    assert manifest["profile"]["baseline"] == {
        "protocol": "prepared-32", "scope": "panel", "workers": 4}
    template["runtime"]["gate_mode"] = "qa_disjoint"
    with pytest.raises(campaign.CampaignError, match="Scene overviews require"):
        campaign.validate_profile(template)


def test_optional_derived_store_is_shared_by_train_and_test_stages(tmp_path):
    template = campaign.read_json(ROOT / "configs/experiments/vsi_m4_1000.json")
    campaign.validate_profile(template)
    context = {"data_root": str(tmp_path / "inputs"), "source_root": str(ROOT),
               "run_root": str(tmp_path / "run"), "python": sys.executable,
               "dataset": "vsi-bench"}
    manifest = {"profile": campaign.expand(template, context), "run_root": context["run_root"]}
    for phase in ("learning", "initial_validation", "final_validation"):
        args = stages.fitset_cli().build_parser().parse_args(
            stages.system_argv(manifest, phase, preflight=True))
        assert args.derived_binding_root == str(
            tmp_path / "inputs/derived_bindings_v4_vsi1000_reuse_audit_20261031")
        assert args.binding_derivation_audit_mode == "reuse_source_audit"
    legacy = native_manifest(tmp_path)
    assert not stages.fitset_cli().build_parser().parse_args(
        stages.system_argv(legacy, "learning")).derived_binding_root
    assert stages.fitset_cli().build_parser().parse_args(
        stages.system_argv(legacy, "learning")).binding_derivation_audit_mode == "reaudit"
    assert manifest["profile"]["baseline"]["protocol"] == "prepared-32"


def test_native_cli_wiring_uses_fixed_roles_shared_budget_and_four_revisers(tmp_path):
    manifest = native_manifest(tmp_path)
    manifest["profile"]["runtime"]["max_requests"] = 100
    cli = stages.fitset_cli()
    learning = cli.build_parser().parse_args(stages.system_argv(manifest, "learning", preflight=True))
    initial = cli.build_parser().parse_args(stages.system_argv(manifest, "initial_validation"))
    final = cli.build_parser().parse_args(stages.system_argv(manifest, "final_validation"))
    assert learning.reviser_concurrency == 4
    assert learning.execution_role == "" and learning.gate_mode == "qa_disjoint"
    assert learning.preflight_only and learning.phase == "accumulation"
    assert learning.library_root == str(ROOT / "skill_library")
    assert initial.phase == final.phase == "final_system"
    assert initial.execution_role == final.execution_role == "validation"
    assert initial.gate_mode == final.gate_mode == "scene_disjoint"
    assert initial.reviser_concurrency == final.reviser_concurrency == 1
    assert initial.request_governor_root == learning.request_governor_root == final.request_governor_root
    assert "seed/bundle" in initial.skill_bundle and "freeze/bundle" in final.skill_bundle


def test_native_seed_freezes_24_real_static_slots_without_model(tmp_path):
    manifest = native_manifest(tmp_path)
    manifest["identity_sha256"] = "b" * 64
    destination = tmp_path / "seed"
    details = stages.seed_bundle(manifest, destination)
    assert details["physical_slots"] == 24
    assert len(list((destination / "bundle").rglob("SKILL.md"))) == 24


def test_native_roles_are_accepted_by_original_selector(qa_partition_case, tmp_path, monkeypatch):
    case = qa_partition_case
    manifest = native_manifest(tmp_path)
    manifest["profile"]["inputs"].update(metadata=str(case.metadata), qa_split=str(case.path))
    cli = stages.fitset_cli()
    monkeypatch.setattr(cli, "select_panel_variants", lambda *_args: case.variants)
    selections = {}
    for name in ("initial_validation", "learning", "final_validation"):
        args = cli.build_parser().parse_args(stages.system_argv(manifest, name))
        selected, gate, overlay = cli._select_experiment_variants({}, args)
        selections[name] = {row["qa_id"] for row in selected}
        assert overlay["seed"] == 137
        assert bool(gate) == (name == "learning")
    assert selections["initial_validation"] == selections["final_validation"] == {"q1", "q3", "q5"}
    assert selections["learning"] == {"q0", "q2", "q4"}


def score_fixture(tmp_path, monkeypatch):
    from skill3d.evaluation import fitset_v11 as fit
    from skill3d.evaluation.direct_qwen_v11 import content_sha256
    manifest = native_manifest(tmp_path)
    manifest["identity_sha256"] = "b" * 64
    variant = {"variant_id": "synthetic-count@g0m0", "qa_id": "synthetic-count",
               "scene_id": "synthetic-scene", "split": "inner_validation",
               "question_type": "object_counting", "evidence_state": "g0m0"}
    monkeypatch.setattr(stages, "execution_variants", lambda *_a, **_k: [variant])
    label = {**variant, "options": [], "reference_answer": "2"}
    monkeypatch.setattr(fit, "load_label_metadata", lambda _path: {variant["qa_id"]: label})
    common = {
        "schema_version": fit.SYSTEM_RUN_SCHEMA, "variant_ids": [variant["variant_id"]],
        "variant_rows_sha256": content_sha256([
            {**variant, "slot_key": fit.fit_slot_key(variant["question_type"], variant["evidence_state"])}
        ]), "binding_refs_sha256": "c" * 64,
        "input_contract_identity": {"synthetic": "frozen"},
        "seed": 137, "online_model": "qwen3.8-flash", "online_model_revision": "synthetic",
        "prompt_version": "frozen", "tool_docs_version": "frozen", "code_identity": {},
        "solver_config_sha256": "d" * 64, "request_policy": {"max_requests": 100},
        "experiment_split_sha256": "e" * 64, "execution": {"variant_ids": [variant["variant_id"]]},
        "reference_answer_visible_to_online_model": False, "online_executor_input_is_label_free": True,
    }
    for name, answer, sha in (("initial_validation", "4", "a"), ("final_validation", "2", "f")):
        arm = Path(manifest["run_root"]) / "stages" / name / "system"
        value = {**common, "skill_identities_sha256": sha * 64}
        value["identity_sha256"] = content_sha256(value)
        save(arm / "manifest.json", value)
        save(arm / "summary.json", {
            "status": "complete", "manifest_identity_sha256": value["identity_sha256"],
            "selected_variants": 1, "completed_variants": 1, "pending_variants": 0,
        })
        skill = {"skill_version": "synthetic@1.0.0", "content_sha256": sha * 64}
        save(arm / "results" / (fit._variant_key(variant["variant_id"]) + ".json"), {
            **variant, "schema_version": fit.SYSTEM_RESULT_SCHEMA,
            "status": "completed", "final_state": "answer", "answer": answer,
            "skill_identity": skill, "service_errors": [],
        })
        save(arm / "preflight" / (fit._variant_key(variant["variant_id"]) + ".json"), {
            "model_requests": 0, "skill_identity": skill, "input_identity_sha256": "1" * 64,
            "binding_key": "frozen", "binding_manifest_sha256": "2" * 64,
            "m5_parameters_sha256": "3" * 64, "runtime_evidence_state": "g0m0",
        })
    return manifest, variant


def test_native_paired_scoring_preserves_true_values_and_full_denominator(tmp_path, monkeypatch):
    manifest, _ = score_fixture(tmp_path, monkeypatch)
    report = stages.paired_score(manifest, tmp_path / "scores")
    assert report["n_pairs"] == 1
    assert report["final_score"] == 1
    assert report["initial_score"] == 0
    assert report["mean_delta"] == 1
    assert report["paired_identity_verified"]


@pytest.mark.parametrize("mismatch", ["seed", "input", "missing_result"])
def test_native_score_refuses_changed_pair_identity(tmp_path, monkeypatch, mismatch):
    from skill3d.evaluation.direct_qwen_v11 import content_sha256
    manifest, _ = score_fixture(tmp_path, monkeypatch)
    arm = Path(manifest["run_root"]) / "stages/final_validation/system"
    if mismatch == "missing_result":
        for path in (arm / "results").glob("*.json"):
            path.unlink()
    else:
        path = arm / "manifest.json"
        value = campaign.read_json(path)
        value["seed" if mismatch == "seed" else "input_contract_identity"] = (
            138 if mismatch == "seed" else {"synthetic": "changed"})
        value.pop("identity_sha256")
        value["identity_sha256"] = content_sha256(value)
        save(path, value)
        summary = campaign.read_json(arm / "summary.json")
        summary["manifest_identity_sha256"] = value["identity_sha256"]
        save(arm / "summary.json", summary)
    with pytest.raises(campaign.CampaignError):
        stages.paired_score(manifest, tmp_path / "scores")


def test_selected_venv_interpreter_path_is_preserved(isolated, tmp_path):
    selected = tmp_path / "synthetic-venv/bin/python"
    selected.parent.mkdir(parents=True)
    selected.symlink_to(sys.executable)
    args = campaign.parser().parse_args([*isolated["argv"], "--python", str(selected), "--dry-run"])
    _, _, interpreter, _, _, _, _ = campaign.resolve_new(args)
    assert interpreter == str(selected)
    assert interpreter != str(selected.resolve())


def test_prepared_baseline_one_governed_call_and_no_repeat_after_resume(tmp_path, monkeypatch):
    import numpy as np
    from skill3d.online import config
    from skill3d.evaluation import published_fitset_v11 as published
    from skill3d.evaluation import direct_qwen_v11 as direct
    manifest = native_manifest(tmp_path)
    manifest["identity_sha256"] = "b" * 64
    manifest["profile"]["runtime"].update(max_requests=1, request_concurrency=1)
    manifest["profile"]["baseline"]["workers"] = 1
    variant = {"variant_id": "synthetic-count@g0m0", "qa_id": "synthetic-count",
               "scene_id": "synthetic-scene", "split": "inner_validation",
               "question_type": "object_counting", "evidence_state": "g0m0"}
    scene_root = tmp_path / "scene"
    scene_root.mkdir()
    (scene_root / "scene.json").write_text("{}")
    frame_set = SimpleNamespace(pixel_sha256=["f" * 64] * 32,
                                frame_set_hash="e" * 64, source_frame_indices=list(range(32)))
    item = SimpleNamespace(episode=SimpleNamespace(ground_truth="", frame_set=frame_set),
                           pixels=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(32)])

    class Provider:
        def __init__(self, *_args, **_kwargs):
            self.entries = {variant["qa_id"]: {"input_root": str(scene_root),
                                              "frozen_scene_sha256": campaign.file_hash(scene_root / "scene.json")}}
            self.input_contract_identity = {"source_catalog_sha256": "a" * 64}

        def __call__(self, _variant):
            return item

        def receipt(self, *_args):
            pytest.fail("Baseline must not materialize geometry bindings")

    calls = []

    class Client:
        configured = True
        last_response_identity = {"model": "qwen3.8-flash"}
        last_usage = {"prompt_tokens": 10, "completion_tokens": 1}

        def audit_identity(self):
            return {"model": "qwen3.8-flash", "provider": "bailian"}

        def chat(self, messages, **kwargs):
            assert len(messages[0]["content"]) == 33
            assert kwargs["temperature"] == 0 and kwargs["seed"] == 137
            assert kwargs["enable_thinking"] is False
            calls.append(1)
            return "2"

    raw = deepcopy(config._FALLBACK)
    raw["online_model"].update(enabled=True, provider="bailian", model="qwen3.8-flash",
                               n_frames=32, temperature=0)
    monkeypatch.setattr(config, "load_config", lambda _path: deepcopy(raw))
    monkeypatch.setattr(config, "resolve_runtime", lambda raw, *, seed: {**raw, "seed": seed})
    monkeypatch.setattr(published, "PublishedFitSetInputProvider", Provider)
    monkeypatch.setattr(stages, "baseline_variants", lambda _manifest: [variant])
    monkeypatch.setattr(direct, "load_safe_metadata", lambda _path: {variant["qa_id"]: {
        "scene_name": variant["scene_id"], "question_type": variant["question_type"],
        "question": "Synthetic count?", "options": []}})
    monkeypatch.setattr(stages, "fitset_cli", lambda: SimpleNamespace(
        _RevisionClientFactory=lambda *_args: lambda: Client()))
    destination = tmp_path / "baseline"
    assert stages.baseline32(manifest, destination)["n_qa"] == 1
    assert stages.baseline32(manifest, destination)["n_qa"] == 1
    assert len(calls) == 1
    state = campaign.read_json(Path(manifest["run_root"]) / "requests/state.json")
    assert len(state["requests"]) == 1 and state["requests"][0]["status"] == "completed"
    # Losing the response never converts the old intent into permission for a
    # second request, even when the request governor released its permit.
    for path in (destination / "responses").glob("*.json"):
        path.unlink()
    with pytest.raises(campaign.CampaignError, match="Unresolved baseline request"):
        stages.baseline32(manifest, destination)
    assert len(calls) == 1
