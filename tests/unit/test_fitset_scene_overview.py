"""Optional overview identity and offline replay, without model/GPU calls."""

import copy
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from skill3d.evaluation.fitset_v11 import (
    FitSetError, SystemVariantRunner, _variant_key, content_sha256,
)
from skill3d.evolution.case_records_v11 import (
    V11ExperienceBuildError, case_from_fitset_outcome, extract_case_image_refs,
)
from skill3d.online.scene_overview import SceneOverviewStore
from skill3d.storage.offline_frames import (
    LEGACY_SCHEMA, OVERVIEW_SCHEMA, SHARED_SCHEMA, overview_ref,
)

from test_shared_offline_frames import capture, inputs
from test_v11_case_records import _inputs as case_inputs


def make_store(tmp_path, scene="scene-A"):
    root = tmp_path / "overview"
    root.mkdir()
    pixels = np.full((480, 640, 3), 67, dtype=np.uint8)
    image_path = root / f"{scene}.png"
    Image.fromarray(pixels).save(image_path)
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    entry = {
        "scene_id": scene, "scene_artifact": str(artifact),
        "frozen_scene_sha256": "f" * 64,
    }
    spec = {"resolution_wh": [640, 480], "test_only": True}
    manifest = {
        "schema": "harness3d-vsi1000-scene-overviews/1", "status": "complete",
        "render_spec": spec, "render_spec_sha256": content_sha256(spec),
        "n_scenes": 1,
        "images": [{
            **entry, "file": image_path.name, "n_frozen_frames": 32,
            "png_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "scene_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "world_frame_status": "available", "resolution_wh": [640, 480],
            "selected_azimuth_degrees": 45, "elevation_degrees": 30,
            "render_only_cutaway": True,
        }],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return SceneOverviewStore(path), entry


class PublishedStub:
    def __init__(self, entry, item=None, reference=None):
        self.entry, self.item, self.reference = entry, item, reference
        self.entries = {"qa": entry}

    def _entry(self, variant):
        assert variant["scene_id"] == self.entry["scene_id"]
        return self.entry

    def __call__(self, variant):
        self._entry(variant)
        return self.item

    def receipt(self, variant, item):
        return {"n_images": 32, "frame_set_hash": item.episode.frame_set.frame_set_hash}

    def prepare_published(self, variant, item):
        from skill3d.online.runner import PreparedObjectBinding
        return {
            "artifact_path": Path(self.entry["scene_artifact"]),
            "artifact": SimpleNamespace(
                depth_maps=None, c2w_list=None, intrinsics=None, depth_conf=None,
                point_map=None, reconstruction_input_identity="scene",
                frame_set_hash="", artifact_id="artifact"),
            "identity": {"sha256": "i" * 64},
            "reference": self.reference, "prepared": PreparedObjectBinding(),
            "artifact_relocation": None, "transport_receipt": None,
        }


def attach_store(runner, store, entry):
    runner.scene_overviews = store
    runner.published_inputs = PublishedStub(entry)
    runner.input_contract_identity = {"scene_overviews": store.identity}
    runner._scene_overview_descriptors = {
        entry["scene_id"]: store.descriptor(entry["scene_id"], entry=entry),
    }


@pytest.mark.parametrize("policy", ["per_variant_v1", "shared_scene_v2"])
def test_offline_32_originals_and_one_shared_typed_overview(tmp_path, policy):
    store, entry = make_store(tmp_path)
    runner = capture(tmp_path / "enabled", policy)
    legacy = capture(tmp_path / "disabled", policy)
    attach_store(runner, store, entry)
    variant, item = inputs()
    runner._freeze_offline_case_input(variant, item)
    legacy._freeze_offline_case_input(variant, item)
    second = {**variant, "variant_id": "qa-two", "qa_id": "two"}
    runner._freeze_offline_case_input(second, item)
    first, other, old = (run.offline_case_input(row) for run, row in (
        (runner, variant), (runner, second), (legacy, variant)))
    assert first["schema_version"] == OVERVIEW_SCHEMA
    assert first["frame_storage_schema"] == (
        SHARED_SCHEMA if policy == "shared_scene_v2" else LEGACY_SCHEMA)
    assert first["frame_set_hash"] == old["frame_set_hash"]
    assert len(first["image_refs"]) == len(old["image_refs"]) == 32
    for actual, original in zip(first["image_refs"], old["image_refs"]):
        assert {k: v for k, v in actual.items() if k != "path"} == {
            k: v for k, v in original.items() if k != "path"}
        assert Path(actual["path"]).read_bytes() == Path(original["path"]).read_bytes()
    ref = first["scene_overview_ref"]
    assert ref == other["scene_overview_ref"]
    assert ref["kind"] == "scene_overview" and "frame_set_hash" not in ref
    assert Path(ref["path"]).stat().st_nlink == 1
    assert len(list((runner.root / "variants" / "shared_frames").rglob(
        ref["png_sha256"] + ".png"))) == 1
    # Frozen replay has no dependency on the live renderer, provider, or cache.
    Path(ref["image_path"]).unlink()
    runner.published_inputs = None
    assert runner.offline_case_input(variant) == first


@pytest.mark.parametrize("field,value", [
    ("scene_id", "wrong-scene"), ("kind", "frame"), ("png_sha256", "0" * 64),
    ("rgb_sha256", "0" * 64), ("scene_artifact_sha256", "0" * 64),
    ("frozen_scene_sha256", "0" * 64), ("resolution_wh", [480, 640]),
    ("render_spec_sha256", "0" * 64), ("image_path", "/another/cache.png"),
    ("path", "/another/shared.png"), ("render_only_cutaway", False),
    ("ground_truth", "forbidden"),
])
def test_offline_overview_rejects_metadata_tamper(tmp_path, field, value):
    store, entry = make_store(tmp_path)
    runner = capture(tmp_path / "run", "shared_scene_v2")
    attach_store(runner, store, entry)
    variant, item = inputs()
    runner._freeze_offline_case_input(variant, item)
    path = runner.root / "variants" / _variant_key(variant["variant_id"]) / "offline_input.json"
    frozen = json.loads(path.read_bytes())
    frozen["scene_overview_ref"][field] = value
    path.write_text(json.dumps(frozen))
    with pytest.raises(FitSetError, match="overview provenance"):
        runner.offline_case_input(variant)


@pytest.mark.parametrize("mutation", ["bytes", "symlink", "hardlink", "missing", "schema"])
def test_offline_overview_fails_closed(tmp_path, mutation):
    store, entry = make_store(tmp_path)
    runner = capture(tmp_path / "run", "shared_scene_v2")
    attach_store(runner, store, entry)
    variant, item = inputs()
    runner._freeze_offline_case_input(variant, item)
    frozen = runner.offline_case_input(variant)
    path = Path(frozen["scene_overview_ref"]["path"])
    if mutation == "bytes":
        path.write_bytes(b"not the frozen PNG")
    elif mutation == "schema":
        receipt = runner.root / "variants" / _variant_key(variant["variant_id"]) / "offline_input.json"
        frozen["schema_version"] = SHARED_SCHEMA
        receipt.write_text(json.dumps(frozen))
    else:
        other = tmp_path / "other.png"
        other.write_bytes(path.read_bytes())
        if mutation == "hardlink":
            (tmp_path / "hardlink.png").hardlink_to(path)
        else:
            path.unlink()
            if mutation == "symlink":
                path.symlink_to(other)
    with pytest.raises(FitSetError):
        runner.offline_case_input(variant)


@pytest.mark.parametrize("policy", ["per_variant_v1", "shared_scene_v2"])
def test_old_schemas_reject_new_optional_fields(tmp_path, policy):
    runner = capture(tmp_path / "run", policy)
    variant, item = inputs()
    runner._freeze_offline_case_input(variant, item)
    path = runner.root / "variants" / _variant_key(variant["variant_id"]) / "offline_input.json"
    frozen = json.loads(path.read_bytes())
    frozen["scene_overview_ref"] = {}
    path.write_text(json.dumps(frozen))
    with pytest.raises(FitSetError, match="input identity"):
        runner.offline_case_input(variant)


class SkillsStub:
    def __init__(self, skill):
        self.skill = skill

    def __call__(self, variant):
        return [self.skill], {"snapshot_ref": "test", "manifest_sha256": "a" * 64}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from skill3d.adapters.episode_source import load_synthetic_items
    from skill3d.online import runner as online
    from skill3d.segmentation import question_binding
    from test_fitset_v11 import _method

    item = load_synthetic_items(
        "induction", question_types=["object_counting"], n_frames=32, frame_size=(8, 8))[0]
    item.episode.ground_truth = ""
    store, entry = make_store(tmp_path, item.episode.scene_name)
    variant = {
        "variant_id": "qa@g0m0", "qa_id": item.episode.qa_id,
        "scene_id": item.episode.scene_name, "question_type": "object_counting",
        "evidence_state": "g0m0", "split": "induction",
        "artifact_ref": entry["scene_artifact"],
    }
    reference = {"key": "b" * 64, "manifest_sha256": "c" * 64,
                 "request_identity": {"seed": 137, "parameters": {}}}
    published = PublishedStub(entry, item, reference)
    monkeypatch.setattr(online, "_prepare_m4_quality", lambda artifact, *a, **k: artifact)
    monkeypatch.setattr(online, "build_scene_state", lambda *a, **k: SimpleNamespace(
        evidence_profile={"geometry_3d": "unavailable", "metric_scale": "unavailable"}))
    monkeypatch.setattr(online, "_load_npy", lambda *a: None)
    monkeypatch.setattr(question_binding, "binding_parameters", lambda **k: {})
    monkeypatch.setattr(question_binding, "validate_prepared_binding", lambda *a, **k: None)
    monkeypatch.setattr("skill3d.evaluation.fitset_v11._source_code_identity", lambda: {"test": True})
    cfg = online.OnlineRunConfig(mode="real", semantic_review_config=None)
    kwargs = dict(
        output_root=tmp_path / "run",
        panel={"proposal_sha256": "p" * 64, "panel_file_sha256": "f" * 64},
        panel_name="ready_579", variants=[variant], item_provider=published,
        input_receipt_provider=published.receipt, input_contract_identity={"original": True},
        binding_refs={variant["variant_id"]: reference}, base_cfg=cfg,
        phase="accumulation", min_free_gib=0, published_inputs=published,
        skills_for_variant=SkillsStub(_method("object_counting", 1).model_copy(
            update={"evidence_state": "g0m0"})),
    )
    return SimpleNamespace(kwargs=kwargs, store=store, variant=variant, item=item, cfg=cfg)


def test_runtime_and_preflight_bind_overview_without_changing_input(runtime):
    run = SystemVariantRunner(**runtime.kwargs, scene_overviews=runtime.store)
    disabled = SystemVariantRunner(**runtime.kwargs)
    run.root.mkdir()
    expected = runtime.store.descriptor(runtime.variant["scene_id"])
    original = disabled._prepare_one(run.variants[0], preflight=True)
    prepared = run._prepare_one(run.variants[0])
    assert prepared["identity"] == original["identity"]
    assert prepared["plan"]["direct_input_receipt"] == original["plan"]["direct_input_receipt"]
    assert prepared["runtime_base_cfg"].max_images == 33
    assert prepared["runtime_base_cfg"].scene_overview == expected
    assert runtime.cfg.max_images == run.base_cfg.max_images == 32
    assert runtime.cfg.scene_overview is None
    assert prepared["plan"]["scene_overview_sha256"] == content_sha256(expected)
    assert len(prepared["item"].pixels) == 32
    assert prepared["item"].episode.frame_set == runtime.item.episode.frame_set
    receipt = run._preflight_one_leased(run.variants[0])
    assert receipt["model_requests"] == 0
    assert receipt["scene_overview"] == prepared["plan"]["scene_overview"]
    assert receipt["scene_overview_sha256"] == prepared["plan"]["scene_overview_sha256"]
    manifest = run._manifest()
    assert manifest["input_contract_identity"]["scene_overviews"] == runtime.store.identity
    assert manifest["request_input_config"]["offline_case_input_schema"] == OVERVIEW_SCHEMA
    assert manifest["shared_frame_consumers"]
    assert "scene_overviews" not in disabled.input_contract_identity


def test_gate_copy_and_spawn_keep_identical_descriptor_and_base_budget(runtime):
    run = SystemVariantRunner(**runtime.kwargs, scene_overviews=runtime.store)
    run.root.mkdir()
    prepared = run._prepare_one(run.variants[0], preflight=True)
    for clone in (copy.copy(run), pickle.loads(pickle.dumps(run))):
        clone.phase = "final_system"
        clone.base_cfg.seed += 1
        replay = clone._prepare_one(clone.variants[0], preflight=True)
        assert replay["plan"]["scene_overview_sha256"] == prepared["plan"]["scene_overview_sha256"]
        assert replay["runtime_base_cfg"].max_images == 33
        assert clone.base_cfg.max_images == 32
        assert clone.input_contract_identity == run.input_contract_identity
    old = object.__new__(SystemVariantRunner)
    old.__setstate__({"root": run.root})
    assert old.scene_overviews is None and old._scene_overview_descriptors == {}


def test_runtime_refuses_descriptor_or_published_scene_changes(runtime, monkeypatch):
    run = SystemVariantRunner(**runtime.kwargs, scene_overviews=runtime.store)
    run.root.mkdir()
    descriptor = runtime.store.descriptor(runtime.variant["scene_id"])
    monkeypatch.setattr(runtime.store, "descriptor", lambda *a, **k: {
        **descriptor, "selected_azimuth_degrees": 90})
    with pytest.raises(FitSetError, match="descriptor changed"):
        run._prepare_one(run.variants[0], preflight=True)


def test_constructor_refuses_unpublished_or_already_expanded_inputs(runtime):
    with pytest.raises(FitSetError, match="published inputs"):
        SystemVariantRunner(**{**runtime.kwargs, "published_inputs": None},
                            scene_overviews=runtime.store)
    runtime.cfg.max_images = 33
    with pytest.raises(FitSetError, match="32-image"):
        SystemVariantRunner(**runtime.kwargs, scene_overviews=runtime.store)


def test_actual_ledger_overview_survives_full_case_and_analysis(tmp_path):
    from skill3d.evolution.revision_view_v11 import decision_case_view
    from skill3d.tools.image_ledger import ImageLedger

    store, entry = make_store(tmp_path, "scene-1")
    descriptor = store.descriptor("scene-1", entry=entry)
    ledger = ImageLedger(max_images=33)
    record = ledger.register_scene_overview(descriptor, scene_id="scene-1")
    image = record.model_dump(mode="json")
    image.update(delivered_rounds=[1], observed_rounds=[],
                 sent_in_requests=["request-1"], response_completed_in_requests=[])
    arguments = case_inputs()
    arguments["outcome"]["image_ledger"]["images"].append(image)
    arguments["outcome"]["image_ledger"]["rounds"][0].update(
        image_ids=[record.image_id], scene_overview_image_ids=[record.image_id],
        observed=False)
    arguments["image_refs"] = [overview_ref(tmp_path, descriptor)]
    before = copy.deepcopy(arguments["outcome"])
    case = case_from_fitset_outcome(**arguments)
    refs = [ref for ref in case.image_refs if ref.get("kind") == "scene_overview"]
    assert len(refs) == 2
    assert refs[0]["delivered_rounds"] == [1] and refs[0]["observed_rounds"] == []
    assert refs[1]["experience_provenance"]["observation_status"] == "unknown"
    assert case.frame_set_hash == before["frame_set_hash"]
    assert case.image_observations == before["image_ledger"]
    assert any(ref["kind"] == "scene_overview"
               for ref in case.rounds[0]["experience_provenance"]["image_refs"])
    view = decision_case_view(case.model_dump(mode="json"))
    overview = next(row for row in view["initial_images"] if row["kind"] == "scene_overview")
    assert overview["images"][0]["image_id"] == record.image_id
    assert "frames" not in overview
    assert overview["render_only_cutaway"] is True
    assert overview["world_frame_status"] == descriptor["world_frame_status"]
    assert overview["observed_rounds"] == []
    assert arguments["outcome"] == before


@pytest.mark.parametrize("mutation", ["annotation", "frame", "wrong_kind"])
def test_overview_provenance_does_not_relax_other_refs(tmp_path, mutation):
    store, entry = make_store(tmp_path)
    ref = overview_ref(tmp_path, store.descriptor(entry["scene_id"]))
    if mutation == "annotation":
        ref["scene_artifact_sha256"] = {"ground_truth": "not allowed"}
    elif mutation == "frame":
        ref["source_frame_id"] = 32
    else:
        ref["kind"] = "frame"
    with pytest.raises(V11ExperienceBuildError):
        extract_case_image_refs({}, image_refs=[ref])


@pytest.mark.parametrize("mode", ["full_cases_v1", "analysis_synthesis_v1"])
def test_stream_forwards_separate_overview_to_canonical_cases(tmp_path, monkeypatch, mode):
    from skill3d.evolution.fitset_stream_v11 import FitSetWorkerEvolution
    from skill3d.tools.image_ledger import ImageLedger
    from test_fitset_stream_v11 import make_runner

    run = make_runner(tmp_path, n=8, learning_input_mode=mode)
    run.initialize(resume=False)
    worker = FitSetWorkerEvolution(run, 0, run.variants)
    run.skills_for_variant = worker.provider
    variant = run.variants[0]
    result = run._run_one(variant)
    source = run.offline_case_input(variant)
    store, entry = make_store(tmp_path, variant["scene_id"])
    descriptor = store.descriptor(variant["scene_id"], entry=entry)
    source["scene_overview_ref"] = overview_ref(run.root, descriptor)
    ledger = ImageLedger(max_images=33)
    image = ledger.register_scene_overview(descriptor).model_dump(mode="json")
    outcome_path = run.root / "variants" / _variant_key(variant["variant_id"]) / "outcome.json"
    outcome = json.loads(outcome_path.read_bytes())
    outcome["image_ledger"] = {"images": [image], "rounds": []}
    outcome_path.write_text(json.dumps(outcome))
    monkeypatch.setattr(run, "offline_case_input", lambda row: copy.deepcopy(source))
    worker.ingest_result(variant, result)
    canonical = json.loads(next((worker.root / "canonical_cases").glob("*.json")).read_bytes())
    overview_refs = [ref for ref in canonical["image_refs"] if ref.get("kind") == "scene_overview"]
    assert len(overview_refs) == 2
    assert len([ref for ref in canonical["image_refs"] if ref.get("kind") == "frame"]) == 32
    file_ref = next(ref for ref in overview_refs if "path" in ref)
    assert file_ref["path"] == source["scene_overview_ref"]["path"]
    assert file_ref["experience_provenance"]["observation_status"] == "unknown"
    assert len(source["image_refs"]) == 32


def test_case_rejects_overview_from_another_scene(tmp_path):
    store, entry = make_store(tmp_path, "wrong-scene")
    arguments = case_inputs()
    arguments["image_refs"] = [overview_ref(tmp_path, store.descriptor("wrong-scene"))]
    with pytest.raises(V11ExperienceBuildError, match="scene identity"):
        case_from_fitset_outcome(**arguments)


def test_case_ref_rejects_conflicting_recorded_render_identity(tmp_path):
    store, entry = make_store(tmp_path)
    descriptor = store.descriptor(entry["scene_id"])
    first = overview_ref(tmp_path, descriptor)
    second = {**first, "scene_artifact_sha256": "0" * 64}
    with pytest.raises(V11ExperienceBuildError, match="identities disagree"):
        extract_case_image_refs({}, image_refs=[first, second])
