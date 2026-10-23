"""CPU-only online reuse of byte-preserving, identity-checked scene migrations."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.evaluation.skill_ablation_v11 import input_identity
from skill3d.infra.m5_identity import content_sha256, file_identity, generation_identity
from skill3d.online import runner
from skill3d.schemas import InputFrame, VSIBenchEpisode
from skill3d.segmentation import question_binding as qb
from skill3d.segmentation.prepared_binding_store import PreparedBindingStore
from skill3d.storage.relocation import (
    RelocationError,
    ValidatedRelocation,
    create_relocation_receipt,
    validate_relocation_receipt,
)

from test_cache_relocation import _scene_copy


class RecordingClient:
    def __init__(self):
        self.calls = []
        self.last_usage = {}

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return (
            'ReturnAnswer(AnswerPayload(value="A", unit="option", '
            'basis="visual_estimate", used_result_ids=[]))\n'
        )


def _forbidden(*args, **kwargs):
    pytest.fail("unexpected live client, reconstruction, or M5 preparation")


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.setattr(runner, "_make_online_client", _forbidden)
    monkeypatch.setattr(runner, "reconstruct", _forbidden)
    monkeypatch.setattr(runner, "_bind_objects_best_effort", _forbidden)


@pytest.fixture(
    params=[(False, True), (True, False)], ids=["absolute-metric", "relative-refs"])
def relocated(tmp_path, request):
    relative, metric = request.param
    case = _scene_copy(tmp_path, relative=relative, metric=metric)
    episode = VSIBenchEpisode(
        **case.episode, ground_truth="A", split="inner_validation",
        frames=[
            InputFrame(
                frame_idx=fid, timestamp=timestamp, width=4, height=3,
                blur_var=1.0, overexposed_ratio=0.0, underexposed_ratio=0.0,
                quality_ok=True,
            )
            for fid, timestamp in zip(
                case.art.frame_ids, case.art.timestamps, strict=True)
        ],
    )
    item = EpisodeItem(
        episode, case.item.pixels, source=case.item.source,
        video_path=case.item.video_path,
    )
    cfg = runner.OnlineRunConfig(
        mode="real", seed=137, prepared_binding_seed=0,
        reuse_artifact=str(case.destination / "scene/artifact.json"),
        semantic_review_config=None, deterministic_replay=True,
        max_solver_rounds=2, finalization_rounds=0,
        trace_dir=str(tmp_path / "traces"),
    )
    binding = case.old_store.load(
        case.key, expected_request_identity=case.request,
        scene_artifact=case.scene, scene_input_identity=case.logical,
    )
    generation = generation_identity(
        scene_name=episode.scene_name, frame_set_hash=case.art.frame_set_hash,
        frames=item.pixels, depth_maps=np.load(case.source / "depth_maps.npy"),
        c2w_list=np.load(case.source / "c2w_list.npy"),
        intrinsics=np.load(case.source / "intrinsics.npy"),
        point_conf=np.load(case.source / "depth_conf.npy"),
        depth_conf=np.load(case.source / "depth_conf.npy"),
        point_map=np.load(case.source / "point_map.npy"),
        reconstruction_identity=(
            case.art.reconstruction_input_identity or case.art.frame_set_hash),
        question=episode.question, question_type=episode.question_type,
        options=episode.options, parameters=qb.binding_parameters(),
        model={"enabled": True, "model": "cpu-test-client", "seed": 0},
        detector={"enabled": False}, sam2={"enabled": False},
    )
    binding.stats.update(version=qb.M5_BINDING_VERSION, generation=generation)
    logical = input_identity(item, case.scene)
    request_body = {
        "schema_version": "online-relocation-test-v1",
        "input_sha256": logical["sha256"], "qa_id": episode.qa_id, "seed": 0,
    }
    sealed_request = {**request_body, "sha256": content_sha256(request_body)}
    source_store = PreparedBindingStore(tmp_path / "online-source-store")
    reference = source_store.publish(
        request_identity=sealed_request, scene_artifact=case.scene,
        scene_input_identity=logical, binding=binding,
    )
    shutil.copytree(source_store.root, case.destination / "online-store")
    store = PreparedBindingStore(case.destination / "online-store")
    receipt = store.create_migration_receipt(
        reference["key"], scene_input_identity=logical,
        destination_root=case.destination, scene_artifact_ref="scene/artifact.json",
        dependency_refs=case.dependency_refs,
    )
    resolver = store.validate_migration_receipt(reference["key"], receipt)
    actual = resolver.recompute_scene_input_identity(
        episode=episode.model_dump(mode="json"), source=item.source,
        video_path=item.video_path, pixels=item.pixels,
    )
    case.item, case.cfg, case.logical = item, cfg, logical
    case.receipt, case.resolver = receipt, resolver
    case.binding = store.load(
        reference["key"], expected_request_identity=sealed_request,
        scene_input_identity=actual, migration_receipt=receipt,
    )
    return case


def _run(case, client, **kwargs):
    arguments = {
        "prepared_binding": case.binding, "artifact_relocation": case.resolver,
        "original_item": case.item,
    }
    arguments.update(kwargs)
    return runner.run_episode(
        case.item.episode, case.item.pixels, case.cfg, llm=client, **arguments)


def _reject_before_m2(monkeypatch):
    monkeypatch.setattr(runner, "input_gate", _forbidden)


@pytest.mark.parametrize("entrypoint", ["episode", "split"])
def test_online_uses_local_refs_preserves_bytes_and_separates_seeds(
        relocated, monkeypatch, entrypoint):
    case = relocated
    before = {
        path: file_identity(path)
        for path in case.destination.rglob("*") if path.is_file()
    }
    original_episode = case.item.episode.model_dump(mode="json")
    original_binding = deepcopy(case.binding.stats)
    shutil.rmtree(case.source)
    monkeypatch.setattr("skill3d.legacy.readers.load_artifact_v5", _forbidden)
    seen = []
    build_scene = runner._scene_from_artifact
    validate_identity = runner._validate_artifact_identity
    identity_checks = []

    def check_identity(art, episode, pixels):
        identity_checks.append(art.artifact_id)
        return validate_identity(art, episode, pixels)

    def capture_scene(art, *args, **kwargs):
        for name, path in case.resolver.dependency_paths.items():
            if name.startswith("confidence."):
                ref = getattr(art.confidence, name.removeprefix("confidence."))
            else:
                ref = getattr(art, name)
            assert ref == str(path)
        assert kwargs["persist_quality"] is False
        result = build_scene(art, *args, **kwargs)
        assert np.array_equal(result[1].get_point_map(), np.load(art.point_map))
        seen.append(art)
        return result

    monkeypatch.setattr(runner, "_validate_artifact_identity", check_identity)
    monkeypatch.setattr(runner, "_scene_from_artifact", capture_scene)
    client = RecordingClient()
    if entrypoint == "episode":
        outcome = _run(case, client)
    else:
        outcomes, run = runner.run_split(
            [case.item], case.cfg, llm=client,
            prepared_bindings={case.item.episode.qa_id: case.binding},
            artifact_relocations={case.item.episode.qa_id: case.resolver},
        )
        assert run.n_episodes == 1
        outcome = outcomes[0]
    assert seen and identity_checks == [case.art.artifact_id]
    assert outcome.final_state == "answer"
    assert outcome.answer == "A" and outcome.receipts_ok
    assert outcome.artifact_ref == str(case.resolver.artifact_path)
    assert outcome.metric_scale == case.art.metric_scale
    assert outcome.per_frame_scale_ref == str(case.resolver.resolve(case.art.per_frame_scale_ref))
    assert client.calls and all(call[1]["seed"] == 137 for call in client.calls)
    assert case.cfg.seed == 137 and case.cfg.prepared_binding_seed == 0
    assert case.item.episode.model_dump(mode="json") == original_episode
    assert np.array_equal(case.binding.stats["array"], original_binding.pop("array"))
    assert {k: v for k, v in case.binding.stats.items() if k != "array"} == original_binding
    assert {path: file_identity(path) for path in before} == before
    assert {p for p in case.destination.rglob("*") if p.is_file()} == set(before)


@pytest.mark.parametrize("change", ["pixel", "pixel_order", "source", "video", "question"])
def test_changed_original_input_fails_before_m2_or_llm(relocated, monkeypatch, change):
    case = relocated
    if change == "pixel":
        case.item.pixels[0][0, 0, 0] ^= 1
    elif change == "pixel_order":
        case.item.pixels.reverse()
    elif change == "source":
        case.item.source = "other-source"
    elif change == "video":
        case.item.video_path = "/other/video.mp4"
    else:
        case.item.episode.question = "Another question?"
    _reject_before_m2(monkeypatch)
    client = RecordingClient()
    with pytest.raises(RelocationError, match="input"):
        _run(case, client)
    assert client.calls == []


def test_direct_episode_checks_supplied_pixels_not_only_original_item(
        relocated, monkeypatch):
    case = relocated
    pixels = [p.copy() for p in case.item.pixels]
    pixels[0][0, 0, 0] ^= 1
    _reject_before_m2(monkeypatch)
    client = RecordingClient()
    with pytest.raises(RelocationError, match="input"):
        runner.run_episode(
            case.item.episode, pixels, case.cfg, llm=client,
            artifact_relocation=case.resolver, original_item=case.item)
    assert client.calls == []


@pytest.mark.parametrize("change", ["missing", "different", "byte_copy"])
def test_reuse_ref_must_be_the_receipted_path(relocated, monkeypatch, change):
    case = relocated
    if change == "byte_copy":
        path = case.destination / "other-artifact.json"
        shutil.copyfile(case.resolver.artifact_path, path)
        case.cfg.reuse_artifact = str(path)
    elif change == "different":
        case.cfg.reuse_artifact = str(case.scene)
    else:
        case.cfg.reuse_artifact = None
    _reject_before_m2(monkeypatch)
    client = RecordingClient()
    with pytest.raises(ValueError, match="reuse_artifact"):
        _run(case, client, prepared_binding=None)
    assert client.calls == []


@pytest.mark.parametrize("override", ["artifact", "receipt", "object"])
def test_bare_artifact_or_receipt_is_not_a_validated_capability(
        relocated, monkeypatch, override):
    case = relocated
    invalid = {
        "artifact": case.resolver.resolved_artifact(),
        "receipt": case.receipt, "object": object(),
    }[override]
    _reject_before_m2(monkeypatch)
    client = RecordingClient()
    with pytest.raises(ValueError, match="ValidatedRelocation"):
        _run(case, client, artifact_relocation=invalid)
    assert client.calls == []


@pytest.mark.parametrize("original", ["missing", "untyped", "other_episode"])
def test_direct_episode_requires_original_typed_item(relocated, monkeypatch, original):
    case = relocated
    item = None if original == "missing" else case.item
    if original == "untyped":
        item = {"source": case.item.source, "video_path": case.item.video_path}
    elif original == "other_episode":
        episode = case.item.episode.model_copy(update={"qa_id": "qa-other"})
        item = replace(case.item, episode=episode)
    _reject_before_m2(monkeypatch)
    client = RecordingClient()
    with pytest.raises(ValueError, match="original"):
        _run(case, client, original_item=item)
    assert client.calls == []


@pytest.mark.parametrize("name", ["artifact", "depth_maps", "metric_depth_maps", "depth_conf"])
def test_changed_dependency_after_validation_fails_before_m2(
        relocated, monkeypatch, name):
    case = relocated
    path = (
        case.resolver.artifact_path if name == "artifact"
        else case.resolver.dependency_paths[name]
    )
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    _reject_before_m2(monkeypatch)
    client = RecordingClient()
    with pytest.raises(RelocationError, match="integrity changed"):
        _run(case, client)
    assert client.calls == []


@pytest.mark.parametrize("mapping", ["empty", "unknown", "extra", "bare", "none", "untyped"])
def test_split_requires_exact_qa_coverage_and_typed_resolvers(
        relocated, monkeypatch, mapping):
    case = relocated
    values = {
        "empty": {}, "unknown": {"other-qa": case.resolver},
        "extra": {"qa-1": case.resolver, "other-qa": case.resolver},
        "bare": {"qa-1": case.resolver.resolved_artifact()},
        "none": {"qa-1": None}, "untyped": [("qa-1", case.resolver)],
    }
    _reject_before_m2(monkeypatch)
    monkeypatch.setattr(runner, "run_episode", _forbidden)
    client = RecordingClient()
    with pytest.raises(ValueError, match="artifact_relocations|ValidatedRelocation"):
        runner.run_split(
            [case.item], case.cfg, llm=client,
            artifact_relocations=values[mapping])
    assert client.calls == []
    assert not Path(case.cfg.trace_dir).exists()


@pytest.mark.parametrize("change", ["second_pixel", "swapped_resolvers"])
def test_split_checks_all_inputs_before_starting_first_episode(
        relocated, monkeypatch, change):
    case = relocated
    episode = case.item.episode.model_copy(deep=True)
    episode.qa_id = "qa-2"
    episode.frame_set.episode_id = "qa-2"
    second = replace(case.item, episode=episode, pixels=[p.copy() for p in case.item.pixels])
    logical = input_identity(second, case.scene)
    receipt = create_relocation_receipt(
        logical_input_identity=logical, destination_root=case.destination,
        artifact_ref="scene/artifact.json", dependency_refs=case.dependency_refs,
    )
    resolver = validate_relocation_receipt(
        receipt, expected_input_sha256=logical["sha256"])
    assert isinstance(resolver, ValidatedRelocation)
    relocations = {"qa-1": case.resolver, "qa-2": resolver}
    if change == "second_pixel":
        second.pixels[0][0, 0, 0] ^= 1
    else:
        relocations = {"qa-1": resolver, "qa-2": case.resolver}
    _reject_before_m2(monkeypatch)
    monkeypatch.setattr(runner, "run_episode", _forbidden)
    client = RecordingClient()
    with pytest.raises(RelocationError, match="input"):
        runner.run_split(
            [case.item, second], case.cfg, llm=client,
            artifact_relocations=relocations)
    assert client.calls == []
    assert not Path(case.cfg.trace_dir).exists()


@pytest.mark.parametrize("seed", [137, 138])
def test_binding_seed_override_does_not_change_solver_seed(relocated, seed):
    case = relocated
    case.cfg.seed = seed
    client = RecordingClient()
    outcome = _run(case, client)
    assert outcome.final_state == "answer"
    assert client.calls and all(call[1]["seed"] == seed for call in client.calls)
    assert case.binding.stats["generation"]["identity"]["vlm"]["seed"] == 0


def test_default_binding_seed_remains_solver_seed(relocated):
    case = relocated
    case.cfg.prepared_binding_seed = None
    client = RecordingClient()
    with pytest.raises(ValueError, match="prepared M5 request seed changed"):
        _run(case, client)
    assert client.calls == []
    case.cfg.seed = 0
    assert _run(case, client).final_state == "answer"
    assert client.calls and all(call[1]["seed"] == 0 for call in client.calls)


@pytest.mark.parametrize("change", ["sources", "parameters", "geometry"])
def test_seed_override_does_not_bypass_binding_identity(relocated, change):
    case = relocated
    generation = case.binding.stats["generation"]
    identity = generation["identity"]
    if change == "sources":
        identity["sources"]["segmentation/question_binding.py"]["sha256"] = "f" * 64
    elif change == "parameters":
        identity["parameters"]["max_objects"] += 1
    else:
        identity["geometry"]["depth"]["sha256"] = "f" * 64
    generation["sha256"] = content_sha256(identity)
    client = RecordingClient()
    with pytest.raises(ValueError, match="prepared M5"):
        _run(case, client)
    assert client.calls == []


@pytest.mark.parametrize("value", [True, False, 0.0, "0", np.int64(0), object()])
def test_prepared_binding_seed_rejects_bool_and_non_int(value):
    with pytest.raises(ValueError, match="prepared_binding_seed"):
        runner.OnlineRunConfig(prepared_binding_seed=value)


def test_mutated_binding_seed_is_rejected_before_m2(relocated, monkeypatch):
    case = relocated
    case.cfg.prepared_binding_seed = False
    _reject_before_m2(monkeypatch)
    client = RecordingClient()
    with pytest.raises(ValueError, match="prepared_binding_seed"):
        _run(case, client)
    assert client.calls == []
