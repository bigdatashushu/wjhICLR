import copy
import json
from pathlib import Path
import shutil

import pytest

from test_cache_relocation import _scene_copy, _receipt
from skill3d.evolution.prepared_bindings_v11 import prepare_shared_bindings_v11
from skill3d.online.runner import OnlineRunConfig
from skill3d.segmentation.prepared_binding_store import PreparedBindingStore


def fixture(tmp_path, monkeypatch):
    from skill3d.online import runner
    case = _scene_copy(tmp_path)
    migration = _receipt(case)
    resolver = case.store.validate_migration_receipt(case.key, migration)
    case.item.input_error = None
    case.item.episode.model_copy = lambda **_: copy.deepcopy(case.item.episode)
    shutil.rmtree(case.source)
    shutil.rmtree(case.old_store.root)
    monkeypatch.setattr(runner, "_prepare_m4_quality", lambda art, *_, **kwargs: art)
    def bind(*args, **kwargs):
        work = Path(kwargs["m5_out_dir"])
        (work / "raw_response.json").write_text(json.dumps({
            "description": "the lower half of a refrigerator",
            "reference_answer_visible": False}))
        return [], {"generation": {"sha256": "a"*64}}, [], False
    monkeypatch.setattr(runner, "_bind_objects_best_effort", bind)
    return case, migration, resolver


def test_new_preparation_preserves_original_logical_geometry_and_replays_without_the_source(
        tmp_path, monkeypatch):
    case, migration, resolver = fixture(tmp_path, monkeypatch)
    destination = tmp_path / "new-qwen8b-store"
    manifest = prepare_shared_bindings_v11(
        [case.item], artifact_paths={"qa-1": resolver.artifact_path},
        base_cfg=OnlineRunConfig(mode="real"), store_root=destination,
        identity_factory=lambda *_: case.logical,
        artifact_loader=lambda _: resolver.resolved_artifact(),
        scene_migration_factory=lambda _: migration,
        trace_root=tmp_path / "traces", min_free_gib=0)
    reference = manifest["bindings"]["qa-1"]
    assert reference["scene_artifact_ref"] == case.logical["artifact_path"]
    assert reference["scene_input_identity_sha256"] == case.logical["sha256"]
    store = PreparedBindingStore(destination)
    fresh = store.create_migration_receipt(
        reference["key"], scene_input_identity=case.logical,
        destination_root=case.destination, scene_artifact_ref="scene/artifact.json",
        dependency_refs=case.dependency_refs)
    prepared = store.load(
        reference["key"], expected_request_identity=reference["request_identity"],
        scene_input_identity=case.logical, migration_receipt=fresh,
        scene_artifact=resolver.artifact_path, migration_destination_root=case.destination)
    assert not prepared.objects
    assert len(list((tmp_path / "traces").rglob("raw_response.json"))) == 1
    # Repeat preparation checks the complete existing store and makes no new binding calls.
    from skill3d.online import runner
    monkeypatch.setattr(runner, "_bind_objects_best_effort",
                        lambda *_, **__: pytest.fail("sealed cache must not call a model"))
    replay = prepare_shared_bindings_v11(
        [case.item], artifact_paths={"qa-1": resolver.artifact_path},
        base_cfg=OnlineRunConfig(mode="real"), store_root=destination,
        identity_factory=lambda *_: case.logical,
        artifact_loader=lambda _: resolver.resolved_artifact(),
        scene_migration_factory=lambda _: migration, min_free_gib=0)
    assert replay["cache_hit_qa_ids"] == ["qa-1"]
    assert replay["bindings"] == manifest["bindings"]


def test_failed_preparation_retains_raw_json_before_work_cleanup(tmp_path, monkeypatch):
    case, migration, resolver = fixture(tmp_path, monkeypatch)
    def reject(_):
        raise RuntimeError("provider failure")
    with pytest.raises(RuntimeError):
        prepare_shared_bindings_v11(
            [case.item], artifact_paths={"qa-1": resolver.artifact_path},
            base_cfg=OnlineRunConfig(mode="real"), store_root=tmp_path / "failed-store",
            identity_factory=lambda *_: case.logical,
            artifact_loader=lambda _: resolver.resolved_artifact(),
            scene_migration_factory=lambda _: migration,
            outcome_validator=reject, trace_root=tmp_path / "traces", min_free_gib=0)
    traces = list((tmp_path / "traces").rglob("raw_response.json"))
    assert len(traces) == 1
    assert json.loads(traces[0].read_text())["reference_answer_visible"] is False
