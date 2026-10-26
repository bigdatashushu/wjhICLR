"""QA-role CLI admission and runner wiring with all online entry points sealed."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import urllib.request

import pytest

from skill3d.evaluation.fitset_v11 import FitSetError, SystemVariantRunner
from skill3d.evolution.fitset_stream_v11 import FrozenSkillProvider
from skill3d.online.config import _FALLBACK
from skill3d.schemas.data_access import normalize_experiment_split
from skill3d.skills.registry import load_active_skills
from skill3d.synthesis.bailian_client import BailianClient

from test_qa_partition_v11 import qa_partition_case, _save


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_fitset_v11_qa_partition_tests", ROOT / "scripts/run_fitset_v11.py")
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
SPEC.loader.exec_module(cli)


def _forbidden(*_args, **_kwargs):
    pytest.fail("unexpected model, server, credential or production configuration access")


@pytest.fixture
def cli_case(qa_partition_case, monkeypatch):
    case = qa_partition_case
    panel = {
        "proposal_sha256": "b" * 64, "panel_file_sha256": "c" * 64,
        "system_variants": case.variants,
    }
    entries = {
        row["qa_id"]: {
            "scene_id": row["scene_id"], "split": row["split"],
            "reference": {"scene_artifact_ref": row["artifact_ref"]},
        } for row in case.variants
    }
    value = SimpleNamespace(case=case, panel=panel, runners=[], actions=[], config_reads=[],
                            bindings_union=[], entries=entries)
    raw = deepcopy(_FALLBACK)
    raw["online_model"]["api_key_env"] = "SYNTHETIC_KEY_MUST_NOT_BE_READ"
    raw["online_model"]["model_revision"] = "synthetic-offline-revision"
    skills, warnings, snapshot = load_active_skills(ROOT / "skill_library/snapshots")
    assert not warnings
    skill_provider = FrozenSkillProvider(skills, {
        "snapshot_ref": snapshot, "manifest_sha256": "d" * 64,
    })

    class PublishedProvider:
        def __init__(self, *_args, **_kwargs):
            self.entries = entries
            self.input_contract_identity = {"schema": "harness3d-published-fitset-inputs/2"}

        __call__ = _forbidden
        receipt = _forbidden

    class CaptureRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            value.runners.append(self)

        def preflight(self, **kwargs):
            value.actions.append(kwargs)
            return {"status": "captured-offline-preflight"}

        run = _forbidden

    def config(path):
        assert path == "synthetic-config-not-opened.yaml"
        value.config_reads.append(path)
        return deepcopy(raw)

    def bindings(_path, variants):
        value.bindings_union = deepcopy(variants)
        return {row["variant_id"]: entries[row["qa_id"]]["reference"] for row in variants}

    monkeypatch.setattr(socket, "create_connection", _forbidden)
    monkeypatch.setattr(urllib.request, "build_opener", _forbidden)
    monkeypatch.setattr(BailianClient, "chat", _forbidden)
    monkeypatch.setattr(BailianClient, "configured", property(_forbidden))
    monkeypatch.setattr(cli._RevisionClientFactory, "__call__", _forbidden)
    monkeypatch.setattr(cli, "load_config", config)
    monkeypatch.setattr(cli, "resolve_runtime", lambda raw, *, seed: {**raw, "seed": seed})
    monkeypatch.setattr(cli, "load_fitset_panel", lambda _path: panel)
    monkeypatch.setattr(cli, "select_panel_variants", lambda *_args: case.variants)
    monkeypatch.setattr(cli, "load_variant_bindings", bindings)
    monkeypatch.setattr(cli, "_FitSetItemProvider", _forbidden)
    monkeypatch.setattr(cli, "_active_skill_provider", lambda _path: skill_provider)
    monkeypatch.setattr(cli, "_fit_skill_provider", lambda _path: skill_provider)
    monkeypatch.setattr(cli, "load_label_metadata", lambda _path: {
        row["qa_id"]: {"scene_id": row["scene_id"], "question_type": row["question_type"],
                       "reference_answer": "synthetic-offline-answer"}
        for row in case.variants
    })
    monkeypatch.setattr(cli, "SystemVariantRunner", CaptureRunner)
    import skill3d.evaluation.published_fitset_v11 as published_module
    monkeypatch.setattr(published_module, "PublishedFitSetInputProvider", PublishedProvider)
    value.argv = [
        "run-system", "--panel", "synthetic-panel-not-opened.json",
        "--panel-name", "ready_579", "--metadata", str(case.metadata),
        "--published-inputs", "synthetic-published-v2-not-opened.json",
        "--config", "synthetic-config-not-opened.yaml", "--phase", "accumulation",
        "--output-dir", str(case.path.parent / "new-run"), "--preflight-only",
    ]
    value.qa_opts = ["--experiment-split-manifest", str(case.path), "--gate-mode", "qa_disjoint"]
    return value


def test_new_options_document_same_scene_question_disjoint_and_preserve_defaults(cli_case):
    args = cli.build_parser().parse_args(cli_case.argv)
    assert args.experiment_split_manifest == ""
    assert args.execution_role == ""
    assert args.gate_mode == "scene_disjoint"
    assert args.learning_input_mode == "full_cases_v1"
    assert args.gate_variant_ids == args.gate_panel_name == ""
    help_text = " ".join(
        cli.build_parser()._subparsers._group_actions[0].choices["run-system"].format_help().split())
    assert "same-scene question-disjoint" in help_text
    assert "source split stays unchanged" in help_text


def test_cli_partitions_before_runner_and_preserves_published_inner_learning(cli_case, capsys):
    value = cli_case
    before = deepcopy(value.case.variants)
    assert cli.main([*value.argv, *value.qa_opts]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "captured-offline-preflight"
    assert value.case.variants == before
    kwargs = value.runners[0].kwargs
    learning, gate = kwargs["variants"], kwargs["gate_variants"]
    assert {row["qa_id"] for row in learning} == {"q0", "q2", "q4"}
    assert {row["qa_id"] for row in gate} == {"q1", "q3", "q5"}
    assert not {row["qa_id"] for row in learning} & {row["qa_id"] for row in gate}
    assert {row["scene_id"] for row in learning} == {row["scene_id"] for row in gate}
    assert sum(row["split"] == "inner_validation" for row in learning) == 4
    assert all(row["source_split"] == value.entries[row["qa_id"]]["split"] for row in [*learning, *gate])
    assert all(row["experiment_role"] == "learning" for row in learning)
    assert all(row["experiment_role"] == "validation" for row in gate)
    assert kwargs["experiment_split"]["qa_roles"] == value.case.manifest["allocation"]["qa_assignment"]
    assert kwargs["evolution_config"].gate_mode == "qa_disjoint"
    assert kwargs["evolution_config"].gate_seeds == (137, 138)
    assert set(kwargs["label_metadata"]) == {f"q{i}" for i in range(6)}
    assert set(kwargs["binding_refs"]) == {row["variant_id"] for row in [*learning, *gate]}
    assert kwargs["item_provider"] is kwargs["published_inputs"]
    assert value.actions == [{"resume": False, "max_variants": 0, "variant_ids": []}]


def test_real_runner_freezes_overlay_in_new_run_identity_without_io(cli_case):
    value = cli_case
    assert cli.main([*value.argv, *value.qa_opts]) == 0
    kwargs = value.runners[0].kwargs
    runner = SystemVariantRunner(**kwargs)
    manifest = runner._manifest()
    assert manifest["experiment_split"] == kwargs["experiment_split"]
    assert manifest["variant_rows_sha256"] != manifest["gate_variant_rows_sha256"]
    assert set(manifest["gate_variant_ids"]) == {row["variant_id"] for row in kwargs["gate_variants"]}
    assert runner.learning_authorization(runner.variants[2])["experiment_role"] == "learning"
    changed = deepcopy(kwargs)
    changed["experiment_split"].pop("identity_sha256")
    changed["experiment_split"]["manifest_identity"]["sha256"] = "f" * 64
    other_manifest = SystemVariantRunner(**changed)._manifest()
    assert other_manifest["identity_sha256"] != manifest["identity_sha256"]
    assert not Path(kwargs["output_root"]).exists()


@pytest.mark.parametrize("opts", [
    ["--gate-mode", "qa_disjoint"],
    ["--gate-mode", "scene_disjoint", "--experiment-split-manifest", "use-case"],
    ["--gate-mode", "same_question", "--experiment-split-manifest", "use-case"],
])
def test_opt_in_flags_must_be_paired_before_config_or_runner(cli_case, opts, capsys):
    opts = [str(cli_case.case.path) if value == "use-case" else value for value in opts]
    assert cli.main([*cli_case.argv, *opts]) == 2
    assert "require each other" in capsys.readouterr().err
    assert not cli_case.runners and not cli_case.config_reads


@pytest.mark.parametrize("selector", [
    ["--gate-variant-ids", "q1@g0m0"], ["--gate-panel-name", "recommended_core_897"],
])
def test_legacy_gate_selectors_cannot_mix_with_manifest(cli_case, selector, capsys):
    assert cli.main([*cli_case.argv, *cli_case.qa_opts, *selector]) == 2
    assert "legacy gate selectors" in capsys.readouterr().err
    assert not cli_case.runners and not cli_case.config_reads


@pytest.mark.parametrize("ids", ["q1@g0m0", "unknown@g0m0", "q0@g0m0,q1@g1m0"])
def test_runtime_variant_selector_cannot_execute_validation_qa(cli_case, ids, capsys):
    assert cli.main([*cli_case.argv, *cli_case.qa_opts, "--variant-ids", ids]) == 2
    assert "manifest-authorized learning" in capsys.readouterr().err
    assert not cli_case.runners and not cli_case.config_reads


def test_runtime_learning_subset_keeps_full_frozen_qa_roles(cli_case):
    assert cli.main([*cli_case.argv, *cli_case.qa_opts, "--variant-ids", "q2@g0m0",
                     "--max-variants", "1", "--resume"]) == 0
    kwargs = cli_case.runners[0].kwargs
    assert len(kwargs["variants"]) == len(kwargs["gate_variants"]) == 6
    assert cli_case.actions == [{"resume": True, "max_variants": 1, "variant_ids": ["q2@g0m0"]}]


def test_manifest_gate_is_accumulation_only(cli_case, capsys):
    assert cli.main([*cli_case.argv, *cli_case.qa_opts, "--phase", "final_system"]) == 2
    assert "requires --execution-role validation" in capsys.readouterr().err
    assert not cli_case.runners and not cli_case.config_reads


def test_invalid_manifest_fails_before_config_or_runner(cli_case, capsys):
    case = cli_case.case
    case.manifest["allocation"]["qa_assignment"]["q2"] = "gate"
    _save(case)
    assert cli.main([*cli_case.argv, *cli_case.qa_opts]) == 2
    assert "unknown or invalid QA roles" in capsys.readouterr().err
    assert not cli_case.runners and not cli_case.config_reads


def test_manifest_metadata_hash_fails_before_config_or_runner(cli_case, capsys):
    cli_case.case.metadata.write_text("changed synthetic metadata")
    assert cli.main([*cli_case.argv, *cli_case.qa_opts]) == 2
    assert "metadata file hash changed" in capsys.readouterr().err
    assert not cli_case.runners and not cli_case.config_reads


def test_default_scene_gate_still_rejects_missing_selector(cli_case, capsys):
    assert cli.main(cli_case.argv) == 2
    assert "scene_disjoint requires unique explicit gate variant IDs" in capsys.readouterr().err
    assert not cli_case.runners and not cli_case.config_reads


def test_default_rich_learning_still_rejects_inner_learning(cli_case, capsys):
    assert cli.main([*cli_case.argv, "--gate-variant-ids", "q1@g0m0,q1@g1m0"]) == 2
    assert "induction-only" in capsys.readouterr().err
    assert not cli_case.runners and not cli_case.config_reads


def test_default_same_question_published_learning_still_rejects_inner(cli_case, capsys):
    assert cli.main([*cli_case.argv, "--gate-mode", "same_question",
                     "--learning-input-mode", "aggregate_v1"]) == 2
    assert "published live learning requires induction-only" in capsys.readouterr().err
    assert not cli_case.runners


def test_legacy_binding_path_gets_exact_partition_union(cli_case, monkeypatch):
    value = cli_case
    argv = value.argv.copy()
    index = argv.index("--published-inputs")
    del argv[index:index + 2]
    monkeypatch.setattr(cli, "_FitSetItemProvider", lambda **_kwargs: SimpleNamespace(
        receipt=_forbidden, input_contract_identity={"synthetic_legacy": True}))
    # The synthetic metadata is only hashed as the synthetic binding identity.
    assert cli.main([*argv, *value.qa_opts, "--bindings", str(value.case.metadata),
                     "--direct-run-root", "synthetic-direct-run-not-opened"]) == 0
    kwargs = value.runners[0].kwargs
    assert value.bindings_union == [*kwargs["variants"], *kwargs["gate_variants"]]
    assert kwargs["published_inputs"] is None
    assert kwargs["evolution_config"].gate_mode == "qa_disjoint"
    assert sum(row["split"] == "inner_validation" for row in kwargs["variants"]) == 4


def test_published_catalog_source_split_mismatch_rejected(cli_case, capsys):
    cli_case.entries["q2"]["split"] = "induction"
    assert cli.main([*cli_case.argv, *cli_case.qa_opts]) == 2
    assert "frozen union identity changed" in capsys.readouterr().err
    assert not cli_case.runners and not cli_case.config_reads


def test_accumulation_seed_must_match_frozen_manifest(cli_case, capsys):
    assert cli.main([*cli_case.argv, *cli_case.qa_opts, "--seed", "138"]) == 2
    assert "seed differs from frozen experiment split" in capsys.readouterr().err
    assert not cli_case.runners and not cli_case.config_reads


def test_final_system_keeps_original_published_split_without_manifest(cli_case):
    assert cli.main([*cli_case.argv, "--phase", "final_system", "--skill-bundle", "synthetic-bundle"]) == 0
    kwargs = cli_case.runners[0].kwargs
    assert sum(row["split"] == "inner_validation" for row in kwargs["variants"]) == 8
    assert kwargs["evolution_config"] is None and kwargs["gate_variants"] == []
    assert kwargs["label_metadata"] is None
    assert "experiment_split" not in kwargs
    assert kwargs["item_provider"] is kwargs["published_inputs"]


def test_final_system_validation_execution_has_hashed_overlay_without_evolution(cli_case):
    value = cli_case
    assert cli.main([*value.argv, "--phase", "final_system", "--skill-bundle", "synthetic-bundle",
                     "--experiment-split-manifest", str(value.case.path),
                     "--execution-role", "validation"]) == 0
    kwargs = value.runners[0].kwargs
    assert {row["qa_id"] for row in kwargs["variants"]} == {"q1", "q3", "q5"}
    assert all(row["experiment_role"] == "validation" for row in kwargs["variants"])
    assert sum(row["source_split"] == "inner_validation" for row in kwargs["variants"]) == 4
    assert kwargs["evolution_config"] is None and kwargs["gate_variants"] == []
    assert kwargs["label_metadata"] is None
    assert kwargs["experiment_split"]["execution_role"] == "validation"
    assert kwargs["experiment_split"] == normalize_experiment_split(kwargs["experiment_split"])
    assert set(kwargs["binding_refs"]) == {row["variant_id"] for row in kwargs["variants"]}


def test_real_final_runner_freezes_validation_overlay_without_learning(cli_case):
    value = cli_case
    assert cli.main([*value.argv, "--phase", "final_system", "--skill-bundle", "synthetic-bundle",
                     "--experiment-split-manifest", str(value.case.path),
                     "--execution-role", "validation"]) == 0
    kwargs = value.runners[0].kwargs
    runner = SystemVariantRunner(**kwargs)
    manifest = runner._manifest()
    assert manifest["phase"] == "final_system"
    assert manifest["experiment_split"]["execution_role"] == "validation"
    assert manifest["evolution"] is None
    assert manifest["gate_variant_ids"] == []
    assert runner.label_metadata == {}
    assert sum(row["split"] == "inner_validation" for row in runner.variants) == 4
    with pytest.raises(FitSetError, match="cannot enter learning"):
        runner.learning_authorization(runner.variants[0])
    assert not Path(kwargs["output_root"]).exists()


def test_final_validation_selector_cannot_select_learning_qa(cli_case, capsys):
    assert cli.main([*cli_case.argv, "--phase", "final_system", "--skill-bundle", "synthetic-bundle",
                     "--experiment-split-manifest", str(cli_case.case.path),
                     "--execution-role", "validation", "--variant-ids", "q2@g0m0"]) == 2
    assert "manifest-authorized validation" in capsys.readouterr().err
    assert not cli_case.runners and not cli_case.config_reads


@pytest.mark.parametrize("opts", [
    ["--execution-role", "validation"],
    ["--experiment-split-manifest", "use-case"],
    ["--experiment-split-manifest", "use-case", "--execution-role", "validation",
     "--gate-mode", "qa_disjoint"],
])
def test_final_validation_flags_must_be_explicit_and_have_no_gate(cli_case, opts, capsys):
    opts = [str(cli_case.case.path) if value == "use-case" else value for value in opts]
    assert cli.main([*cli_case.argv, "--phase", "final_system", *opts]) == 2
    error = capsys.readouterr().err
    assert "requires --execution-role validation" in error or "does not use qa_disjoint" in error
    assert not cli_case.runners and not cli_case.config_reads


def test_accumulation_cannot_accept_execution_role(cli_case, capsys):
    assert cli.main([*cli_case.argv, *cli_case.qa_opts, "--execution-role", "validation"]) == 2
    assert "only valid for final_system" in capsys.readouterr().err
    assert not cli_case.runners and not cli_case.config_reads
