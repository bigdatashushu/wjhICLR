"""Explicit train/test admissions retain original catalog and frozen QA anchors."""

from copy import deepcopy
from pathlib import Path
import pickle

import pytest

from skill3d.evaluation.published_fitset_v11 import (
    FULLQA_INPUT_SCHEMA,
    PUBLISHED_INPUT_SCHEMA,
    TRAIN_TEST_INPUT_SCHEMA,
    PublishedFitSetInputError,
    PublishedFitSetInputProvider,
)
from skill3d.infra.m5_identity import content_sha256, file_identity

from test_cluster_scene_compute import _write
from test_published_fitset_v11 import _assert_safe, _published, _rewrite_manifest
from test_published_physical_transport import _copy


SOURCE_SPLITS = ("induction", "inner_validation", "outer_holdout", "final_test")


def _train_test(case):
    source = case.manifest_path.parent / "train-test-source-catalog.json"
    _write(source, {
        "schema": "harness3d-published-binding-catalog/1",
        "variants": {"base": case.catalog},
    })
    case.manifest.update(
        schema=TRAIN_TEST_INPUT_SCHEMA,
        source_catalog_file=str(source),
        source_catalog_sha256=file_identity(source)["sha256"],
        selection_policy={
            "variant": "base",
            "train_test_manifest_sha256": "a" * 64,
            "original_source_split_preserved": True,
        },
        scene_scopes={"scene-one": {
            "qa_ids": list(case.frozen.manifest["qa_ids"]),
            "split": case.frozen.manifest["split"],
            "input_root": str(case.frozen.root),
            "frozen_scene_sha256": file_identity(case.frozen.root / "scene.json")["sha256"],
        }},
    )
    _rewrite_manifest(case)


@pytest.mark.parametrize("split", SOURCE_SPLITS)
def test_explicit_v3_replays_selected_qa_with_full_original_scope(tmp_path, split):
    case = _published(tmp_path, split=split, n_questions=9)
    _train_test(case)
    original_scene = (case.frozen.root / "scene.json").read_bytes()
    case.manifest["entries"] = {"qa-00": case.manifest["entries"]["qa-00"]}
    _rewrite_manifest(case)
    provider = PublishedFitSetInputProvider(case.manifest_path)
    variant = {**case.variant, "source_split": split, "role": "train_sample"}
    item = provider(variant)
    result = provider.prepare_published(variant, item)
    assert item.episode.split == split and item.episode.ground_truth == ""
    assert set(provider.entries) == {"qa-00"}
    assert provider.scene_scopes["scene-one"]["qa_ids"] == case.frozen.manifest["qa_ids"]
    assert len(next(iter(provider.cache.values())).items) == 9
    assert result["reference"] == case.catalog["qa-00"]["reference"]
    assert result["identity"] == case.frozen.manifest["questions"][0]["logical_input_identity"]
    assert result["transport_receipt"]["catalog_entry_sha256"] == content_sha256(
        case.catalog["qa-00"])
    assert provider.input_contract_identity["selection_policy"] == case.manifest["selection_policy"]
    assert (case.frozen.root / "scene.json").read_bytes() == original_scene
    spawned = pickle.loads(pickle.dumps(provider.for_spawned_episode(variant)))
    assert not spawned.cache
    assert spawned.scene_scopes == provider.scene_scopes
    assert spawned.input_contract_identity == provider.input_contract_identity
    assert spawned(variant).episode.split == split
    assert len(next(iter(spawned.cache.values())).items) == 9
    _assert_safe(provider.entries)
    _assert_safe(provider.input_contract_identity)
    _assert_safe(result["transport_receipt"])
    with pytest.raises(PublishedFitSetInputError, match="catalog entry is missing"):
        provider(case.variants["qa-01"])
    with pytest.raises(PublishedFitSetInputError, match="panel/catalog"):
        provider({**variant, "source_split": "train_sample"})
    with pytest.raises(PublishedFitSetInputError, match="panel/catalog"):
        provider({**variant, "split": "train_sample"})


@pytest.mark.parametrize("schema", [PUBLISHED_INPUT_SCHEMA, FULLQA_INPUT_SCHEMA])
@pytest.mark.parametrize("split", ["outer_holdout", "final_test"])
@pytest.mark.parametrize("admission", ["scope", "entry"])
def test_legacy_schemas_still_reject_original_outer_final_splits(
    tmp_path, schema, split, admission,
):
    case = _published(tmp_path, split=split)
    _train_test(case)
    case.manifest["schema"] = schema
    if admission == "entry":
        case.manifest["scene_scopes"]["scene-one"]["split"] = "induction"
    _rewrite_manifest(case)
    with pytest.raises(PublishedFitSetInputError):
        PublishedFitSetInputProvider(case.manifest_path)


@pytest.mark.parametrize("field", [
    "split", "scene_artifact", "publication_receipt_sha256", "reference",
])
def test_explicit_v3_requires_exact_original_catalog_entry(tmp_path, field):
    case = _published(tmp_path, split="final_test")
    _train_test(case)
    entry = case.manifest["entries"]["qa-00"]
    if field == "split":
        entry[field] = "induction"
    elif field == "reference":
        entry[field]["key"] = "b" * 64
    elif field == "scene_artifact":
        entry[field] = str(tmp_path / "rewritten-artifact.json")
    else:
        entry[field] = "b" * 64
    _rewrite_manifest(case)
    with pytest.raises(PublishedFitSetInputError, match="catalog entry changed"):
        PublishedFitSetInputProvider(case.manifest_path)


@pytest.mark.parametrize("mutation", [
    "missing_catalog_file", "missing_catalog_hash", "wrong_catalog_hash",
    "missing_train_test_hash", "wrong_train_test_hash", "missing_preservation",
    "false_preservation", "integer_preservation", "policy_not_object", "label",
])
def test_explicit_v3_requires_label_free_provenance_anchors(tmp_path, mutation):
    case = _published(tmp_path)
    _train_test(case)
    manifest, policy = case.manifest, case.manifest["selection_policy"]
    if mutation == "missing_catalog_file":
        del manifest["source_catalog_file"]
    elif mutation == "missing_catalog_hash":
        del manifest["source_catalog_sha256"]
    elif mutation == "wrong_catalog_hash":
        manifest["source_catalog_sha256"] = "b" * 64
    elif mutation == "missing_train_test_hash":
        del policy["train_test_manifest_sha256"]
    elif mutation == "wrong_train_test_hash":
        policy["train_test_manifest_sha256"] = "not-a-digest"
    elif mutation == "missing_preservation":
        del policy["original_source_split_preserved"]
    elif mutation == "false_preservation":
        policy["original_source_split_preserved"] = False
    elif mutation == "integer_preservation":
        policy["original_source_split_preserved"] = 1
    elif mutation == "policy_not_object":
        manifest["selection_policy"] = "train-test"
    else:
        policy["answer"] = "private"
    _rewrite_manifest(case)
    with pytest.raises(ValueError):
        PublishedFitSetInputProvider(case.manifest_path)


@pytest.mark.parametrize("mutation", ["subset_scope", "split", "root", "pin", "cap"])
def test_explicit_v3_cannot_replace_full_frozen_scene_scope(tmp_path, mutation):
    case = _published(tmp_path, split="outer_holdout", n_questions=9)
    _train_test(case)
    case.manifest["entries"] = {"qa-00": case.manifest["entries"]["qa-00"]}
    scope = case.manifest["scene_scopes"]["scene-one"]
    if mutation == "subset_scope":
        scope["qa_ids"] = ["qa-00"]
    elif mutation == "split":
        scope["split"] = "final_test"
    elif mutation == "root":
        scope["input_root"] = str(tmp_path / "other-root")
    elif mutation == "pin":
        scope["frozen_scene_sha256"] = "b" * 64
    else:
        scope["qa_ids"] = [f"qa-{i:03d}" for i in range(129)]
    _rewrite_manifest(case)
    with pytest.raises(ValueError):
        provider = PublishedFitSetInputProvider(case.manifest_path)
        provider(case.variant)


@pytest.mark.parametrize("schema", [TRAIN_TEST_INPUT_SCHEMA, FULLQA_INPUT_SCHEMA])
def test_verified_spawn_subset_keeps_complete_group_guard(tmp_path, schema):
    split = "final_test" if schema == TRAIN_TEST_INPUT_SCHEMA else "induction"
    case = _published(tmp_path, split=split, n_questions=9)
    _train_test(case)
    case.manifest["schema"] = schema
    case.manifest["entries"] = {"qa-00": case.manifest["entries"]["qa-00"]}
    _rewrite_manifest(case)
    copied = _copy(case)
    provider = PublishedFitSetInputProvider(
        copied.manifest_path, transport_locator=copied.descriptor)
    provider.reuse_verified_publications()
    identity = deepcopy(provider.input_contract_identity)
    with pytest.raises(PublishedFitSetInputError, match="complete original scene QA group"):
        provider.for_spawned_episode(case.variant)
    assert provider.input_contract_identity == identity


def test_selected_catalog_retains_private_verified_source_closure(tmp_path):
    case = _published(tmp_path, split="final_test", n_questions=9)
    _train_test(case)
    case.manifest["entries"] = {"qa-00": case.manifest["entries"]["qa-00"]}
    _rewrite_manifest(case)
    provider = PublishedFitSetInputProvider(case.manifest_path)
    assert set(provider.entries) == {"qa-00"}
    assert set(provider._source_entries) == set(case.catalog)
    omitted = "qa-08"
    original = provider._source_entries[omitted]
    item = next(item for item in provider._scene(original).items
                if item.episode.qa_id == omitted)
    transport = provider._transport_entry(case.variants[omitted], item, original)
    assert transport["identity"]["sha256"] == original["reference"]["scene_input_identity_sha256"]
    _assert_safe(transport["receipt"])
    with pytest.raises(PublishedFitSetInputError, match="catalog entry is missing"):
        provider(case.variants[omitted])
    with pytest.raises(PublishedFitSetInputError, match="catalog entry is missing"):
        provider.prepare_published(case.variants[omitted], item)


@pytest.mark.parametrize("mutation", ["missing", "scene", "split", "label"])
def test_omitted_source_catalog_entries_are_still_checked(tmp_path, mutation):
    case = _published(tmp_path, n_questions=9)
    _train_test(case)
    case.manifest["entries"] = {"qa-00": case.manifest["entries"]["qa-00"]}
    source = Path(case.manifest["source_catalog_file"])
    catalog = deepcopy(case.catalog)
    if mutation == "missing":
        del catalog["qa-08"]
    elif mutation == "scene":
        catalog["qa-08"]["scene_id"] = "other-scene"
    elif mutation == "split":
        catalog["qa-08"]["split"] = "final_test"
    else:
        catalog["qa-08"]["answer"] = "private"
    _write(source, {
        "schema": "harness3d-published-binding-catalog/1",
        "variants": {"base": catalog},
    })
    case.manifest["source_catalog_sha256"] = file_identity(source)["sha256"]
    _rewrite_manifest(case)
    with pytest.raises(PublishedFitSetInputError):
        PublishedFitSetInputProvider(case.manifest_path)
