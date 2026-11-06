"""Check real catalog selection, exact Skill slots, and fail-closed contracts."""

from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from skill3d.evaluation.fitset_v11 import (
    ALL_FIT_SLOTS, FitSetError, build_fit_skill_bundle, content_sha256,
    load_fit_skill_bundle, load_fitset_panel, select_panel_variants,
)
from skill3d.evaluation.published_fitset_v11 import (
    PublishedFitSetInputError, PublishedFitSetInputProvider,
)
from skill3d.evolution.fitset_revision_v11 import load_active_parent_slots, _tool_docs
from skill3d.infra import experiment_campaign as campaign
from skill3d.online.scene_overview import SceneOverviewStore
from skill3d.schemas.reconstruction import METRIC_TASK_TYPES

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "prepared_cache/vsibench-m4-success-g1m1-1000-seed137-r01"
NEW = ROOT / "prepared_cache/vsibench-m4-success-g1m1-1000-seed137-metric-r02"
catalog_required = pytest.mark.skipif(
    not all((path / "published_inputs.json").is_file() for path in (OLD, NEW)),
    reason="requires external frozen G1M1 prepared catalogs",
)


@catalog_required
def test_default_metric_catalog_and_all_qa_roles_validate():
    _, _, _, profile, _, _, _ = campaign.resolve_new(
        campaign.parser().parse_args(["--dry-run"]))
    campaign.check_train_test_inputs(profile)
    provider = PublishedFitSetInputProvider(NEW / "published_inputs.json", max_scenes=1)
    panel = load_fitset_panel(NEW / "panel.json")
    variants = select_panel_variants(panel, "ready_579")
    assert len(variants) == 1000
    assert provider.catalog_variant == "metric"
    assert {row["evidence_state"] for row in variants} == {"g1m1"}
    assert len({row["question_type"] for row in variants}) == 8
    sampling = json.loads((NEW / "sampling_manifest.json").read_bytes())
    from hashlib import sha256
    assert sampling["prepared_inputs"]["sha256"] == sha256(
        (NEW / "published_inputs.json").read_bytes()).hexdigest()
    assert sampling["prepared_inputs"]["package_root"] == str(NEW)
    assert all(entry["variant"] == "metric" for entry in provider._source_entries.values())
    assert provider.input_contract_identity["real_labels_available"] is False
    for row in variants:
        assert provider.public_entry(row)["reference"]["key"] == row["binding_key"]
    assert SceneOverviewStore(NEW / "scene_overviews/manifest.json").validate_entries(
        provider.entries)["n_scenes"] == 160


@catalog_required
def test_selection_frames_models_and_quotas_are_preserved():
    for filename in ("questions.jsonl", "train.jsonl", "test.jsonl", "split.json",
                     "online.yaml", "source_catalog.json"):
        assert (OLD / filename).read_bytes() == (NEW / filename).read_bytes()
    before = load_fitset_panel(OLD / "panel.json")
    after = load_fitset_panel(NEW / "panel.json")
    fields = ("qa_id", "scene_id", "question_type", "source_split", "split")
    assert [{k: row[k] for k in fields} for row in before["system_variants"]] == [
        {k: row[k] for k in fields} for row in after["system_variants"]]
    assert Counter(row["question_type"] for row in before["system_variants"]) == Counter(
        row["question_type"] for row in after["system_variants"])


@catalog_required
@pytest.mark.parametrize("mutation", ["base_entry", "base_policy", "unknown_policy"])
def test_metric_variant_cannot_be_silently_rebound(tmp_path, mutation):
    manifest = json.loads((NEW / "published_inputs.json").read_bytes())
    qa_id = next(iter(manifest["entries"]))
    if mutation == "base_entry":
        manifest["entries"][qa_id]["variant"] = "base"
    else:
        manifest["selection_policy"]["variant"] = (
            "base" if mutation == "base_policy" else "unknown")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(PublishedFitSetInputError):
        PublishedFitSetInputProvider(path)


@catalog_required
def test_metric_panel_rejects_original_base_artifact():
    provider = PublishedFitSetInputProvider(NEW / "published_inputs.json")
    row = copy.deepcopy(load_fitset_panel(NEW / "panel.json")["system_variants"][0])
    old = json.loads((OLD / "published_inputs.json").read_bytes())
    row["artifact_ref"] = old["entries"][row["qa_id"]]["reference"]["scene_artifact_ref"]
    with pytest.raises(PublishedFitSetInputError, match="panel/catalog identity changed"):
        provider.public_entry(row)


def test_new_seed_bundle_preserves_all_24_exact_skill_bodies(tmp_path):
    parents, identity = load_active_parent_slots(ROOT.parent / "skill_library")
    path = tmp_path / "bundle"
    build_fit_skill_bundle(output_dir=path, bundle_id="metric-seed", candidates={},
                           active_parents=parents, source_run_identity=identity)
    bundle = load_fit_skill_bundle(path / "manifest.json")
    assert bundle.manifest["reachable_slot_count"] == 24
    assert bundle.manifest["preserved_unreachable_slot_count"] == 0
    assert set(bundle.slots) == set(ALL_FIT_SLOTS)
    for slot, parent in parents.items():
        assert bundle.skill_for(*slot).skill_md == parent.skill_md
    # Existing sealed 19-slot bundles retain their original role interpretation.
    manifest = json.loads((path / "manifest.json").read_bytes())
    unreachable = [f"{task}/g1m1" for task, state in ALL_FIT_SLOTS
                   if state == "g1m1" and task not in METRIC_TASK_TYPES]
    manifest.update(reachable_slot_count=19, unreachable_slots=unreachable,
                    preserved_unreachable_slot_count=5,
                    preserved_reachable_pending_slot_count=19)
    for key in unreachable:
        manifest["slots"][key]["role"] = "active_parent_unreachable"
    manifest["manifest_sha256"] = content_sha256({
        key: value for key, value in manifest.items() if key != "manifest_sha256"})
    (path / "manifest.json").write_text(json.dumps(manifest))
    assert load_fit_skill_bundle(path / "manifest.json").manifest["reachable_slot_count"] == 19


def test_nonmetric_g1m1_does_not_gain_metric_tools():
    metric_doc = _tool_docs("object_abs_distance", "g1m1")
    nonmetric_doc = _tool_docs("object_counting", "g1m1")
    # These documented tool names are the actual capability surface.
    from skill3d.tools import REGISTRY
    from skill3d.schemas.evidence import EvidenceProfile
    context = dict(scope="metric_enabled", available=frozenset(),
                   evidence_profile=EvidenceProfile(
                       geometry_3d="available", metric_scale="available"),
                   gate_passed=True, question_type="object_abs_distance",
                   local_facts={"world_frame": True})
    # Full tool docs also expose interface signatures, making accidental task
    # authorization changes visible even when artifact preconditions differ.
    from skill3d.evolution.fitset_revision_v11 import KNOWN_ARTIFACTS
    context["available"] = KNOWN_ARTIFACTS
    metric = set(REGISTRY.names_for_scope(**context))
    context.update(scope="full_3d", gate_passed=False, question_type="object_counting")
    nonmetric = set(REGISTRY.names_for_scope(**context))
    assert metric - nonmetric
    assert all(f'"{name}"' in metric_doc and f'"{name}"' not in nonmetric_doc
               for name in metric - nonmetric)


def test_metric_package_builder_never_overwrites_a_sealed_package(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "configure_g1m1_dataset", ROOT / "scripts/configure_g1m1_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = tmp_path / "sealed"
    destination.mkdir()
    with pytest.raises(ValueError, match="separate, unused"):
        module.build(OLD, destination, Path("/unused"))
