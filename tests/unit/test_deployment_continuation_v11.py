"""Source-only continuation checks using generated state, paths and faults."""

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.evaluation.fitset_v11 import (
    ALL_FIT_SLOTS, FitSetError, SystemVariantRunner, _variant_key,
    _write_create_once, content_sha256,
)
from skill3d.evolution.deployment_continuation_v11 import (
    revision_intents_used, write_deployment_continuation,
)
from skill3d.evolution.fitset_revision_v11 import (
    FIT_REVISION_PROMPT_VERSION, RevisionOperationalError,
)
from skill3d.evolution.fitset_stream_v11 import (
    FitSetEvolutionConfig, FitSetWorkerEvolution, _active, initialize_worker_library,
)
from skill3d.evolution.streaming_v11 import STREAM_SCHEMA_VERSION
from skill3d.schemas import SkillSpecV11


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=3) + "\n", encoding="utf-8")


def signed(value):
    return {**value, "identity_sha256": content_sha256(value)}


def runner_for(root, manifest):
    runner = object.__new__(SystemVariantRunner)
    runner.root = root
    runner._manifest = lambda: copy.deepcopy(manifest)
    runner.deployment_continuation = None
    runner.deployment_identity = None
    runner._continuation = None
    runner.published_inputs = None
    runner.run_identity = ""
    runner._episode_semaphore = None
    return runner


def stage(tmp_path):
    logical = tmp_path / "old" / "accumulation"
    physical = tmp_path / "working" / "accumulation"
    config = {"batch_size": 12, "gate_size": 8, "max_revision_requests_per_slot": 18}
    original = signed({
        "schema_version": "v11-fit-set-system-run/1.0", "phase": "accumulation",
        "seed": 137, "variant_rows_sha256": "f" * 64,
        "code_identity": {"source": "synthetic", "source_root": "/synthetic/old",
                          "source_sha256": "a" * 64},
        "input_contract_identity": {
            "manifest": {"path": str(logical / "public.json"), "sha256": "c" * 64}},
        "evolution": {"config": copy.deepcopy(config)},
        "request_input_config": {"evolution_config": copy.deepcopy(config)},
    })
    write_json(physical / "manifest.json", original)
    state = {
        "schema_version": STREAM_SCHEMA_VERSION,
        "stream_id": f"fit-worker-00-{original['identity_sha256'][:12]}",
        "config": {"batch_size": 12, "tail_for_unattempted_slots": False},
        "slots": {}, "attempts": [], "operational_failures": [], "seen_case_ids": [],
    }
    write_json(physical / "workers/00/buffer/stream.json", state)
    write_json(physical / "workers/00/worker_identity.json", {
        "run_identity": original["identity_sha256"], "owned_slots": ["object_counting/g0m0"]})
    ledger = {"policy": {"max_requests": 100}, "requests": [{
        "request_id": 0, "category": "reviser", "status": "error",
        "context": {"run_identity": original["identity_sha256"]}, "usage": {},
        "exception_type": "Synthetic502",
    }]}
    write_json(physical / "request_governor/state.json", ledger)
    current = copy.deepcopy(original)
    del current["identity_sha256"]
    current["code_identity"] = {"source": "synthetic", "source_root": "/synthetic/new",
                                "source_sha256": "b" * 64}
    current["input_contract_identity"]["manifest"]["path"] = str(physical / "public.json")
    for path in (current["evolution"]["config"], current["request_input_config"]["evolution_config"]):
        path.update(revision_input_max_bytes=128000, revision_input_max_chars=72000)
    current = signed(current)
    runner = runner_for(physical, current)
    return SimpleNamespace(
        logical=logical, root=physical, original=original, current=current, runner=runner,
        sidecar=physical / "deployments/new-code/continuation.json",
        ledgers=["request_governor/state.json"], path_map={})


def seal(staged):
    refs = [path.relative_to(staged.root).as_posix()
            for path in staged.root.rglob("*") if path.is_file()
            and "deployments" not in path.relative_to(staged.root).parts]
    value = write_deployment_continuation(
        staged.sidecar, system_root=staged.root, logical_root=staged.logical,
        deployment_id="new-code", current_manifest=staged.current,
        original_revision_protocol="fit-set-slot-revision-v5-synthetic",
        preserved_paths=refs, ledger_paths=staged.ledgers, path_map=staged.path_map)
    staged.runner.deployment_continuation = staged.sidecar
    return value


def test_explicit_sidecar_preserves_raw_state_and_original_lineage(tmp_path):
    staged = stage(tmp_path)
    before = {path: path.read_bytes() for path in staged.root.rglob("*") if path.is_file()}
    with pytest.raises(FitSetError, match="manifest changed"):
        staged.runner.initialize(resume=True)
    receipt = seal(staged)
    result = staged.runner.initialize(resume=True)
    assert result == staged.original
    assert staged.runner.run_identity == staged.original["identity_sha256"]
    assert staged.runner.deployment_identity == receipt["identity_sha256"]
    assert receipt["source_identity"] == staged.current["code_identity"]
    assert {change["kind"] for change in receipt["allowed_changes"]} == {
        "code", "location", "revision_protocol"}
    assert len([change for change in receipt["allowed_changes"]
                if change["kind"] == "revision_protocol"]) == 4
    assert all(path.read_bytes() == data for path, data in before.items())
    for ref in ("workers/00/buffer/stream.json", "request_governor/state.json"):
        assert (staged.sidecar.parent / "raw_state" / ref).read_bytes() == (
            staged.root / ref).read_bytes()
    assert staged.runner.initialize(resume=True) == staged.original
    with pytest.raises(FitSetError, match="use resume"):
        staged.runner.initialize(resume=False)


@pytest.mark.parametrize("field,value", [
    ("seed", 138), ("phase", "final_system"), ("variant_rows_sha256", "e" * 64),
    ("batch_size", 13), ("max_revision_requests_per_slot", 19),
    ("revision_input_max_bytes", 128001), ("revision_input_max_chars", 72001),
    ("revision_input_max_chars", True), ("unknown_revision_budget", 100),
])
def test_only_bounded_revision_additions_are_authorized(tmp_path, field, value):
    staged = stage(tmp_path)
    current = copy.deepcopy(staged.current)
    del current["identity_sha256"]
    if field in {"seed", "phase", "variant_rows_sha256"}:
        current[field] = value
    else:
        current["evolution"]["config"][field] = value
    staged.current = signed(current)
    with pytest.raises(FitSetError, match="unauthorized|unbounded"):
        seal(staged)
    assert not staged.sidecar.exists()


@pytest.mark.parametrize("target", ["manifest", "stream", "ledger", "source", "protocol", "sidecar"])
def test_changed_bound_inputs_are_rejected_before_activation(tmp_path, target):
    staged = stage(tmp_path)
    seal(staged)
    if target == "manifest":
        path = staged.root / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif target == "stream":
        path = staged.root / "workers/00/buffer/stream.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif target == "ledger":
        path = staged.root / "request_governor/state.json"
        write_json(path, {"policy": {"max_requests": 100}, "requests": []})
    elif target == "source":
        changed = copy.deepcopy(staged.current)
        del changed["identity_sha256"]
        changed["code_identity"]["source_sha256"] = "d" * 64
        staged.runner._manifest = lambda: signed(changed)
    else:
        changed = json.loads(staged.sidecar.read_bytes())
        if target == "protocol":
            changed["revision_protocol"]["current"] = "not-the-executing-protocol"
            del changed["identity_sha256"]
            changed = signed(changed)
        else:
            changed["run_identity"] = "0" * 64
        write_json(staged.sidecar, changed)
    with pytest.raises(FitSetError):
        staged.runner.initialize(resume=True)
    assert not (staged.sidecar.parent / "activated.json").exists()


def test_progress_keeps_raw_archives_and_cannot_reset_ledger(tmp_path):
    staged = stage(tmp_path)
    seal(staged)
    staged.runner.initialize(resume=True)
    ledger_path = staged.root / "request_governor/state.json"
    before = (staged.sidecar.parent / "raw_state/request_governor/state.json").read_bytes()
    ledger = json.loads(ledger_path.read_bytes())
    ledger["requests"].append({"request_id": 1, "status": "completed", "category": "reviser"})
    write_json(ledger_path, ledger)
    stream_path = staged.root / "workers/00/buffer/stream.json"
    state = json.loads(stream_path.read_bytes())
    state["seen_case_ids"].append("synthetic-new-case")
    write_json(stream_path, state)
    staged.runner.initialize(resume=True)
    assert (staged.sidecar.parent / "raw_state/request_governor/state.json").read_bytes() == before
    ledger["requests"] = []
    write_json(ledger_path, ledger)
    with pytest.raises(FitSetError, match="reservations were reset"):
        staged.runner.initialize(resume=True)


def test_external_sibling_ledger_and_nested_override_are_explicit(tmp_path):
    staged = stage(tmp_path)
    outer = tmp_path / "workspace"
    staged.path_map = {str(staged.logical.parent): str(outer)}
    staged.ledgers = ["../request_governor/state.json"]
    ledger_path = outer / "request_governor/state.json"
    write_json(ledger_path, json.loads((staged.root / "request_governor/state.json").read_bytes()))
    receipt = seal(staged)
    assert receipt["ledger_anchors"] == []
    logical_ledger = str(staged.logical.parent / "request_governor/state.json")
    anchor = receipt["external_ledger_anchors"][logical_ledger]
    assert set(anchor) == {"identity", "archive_ref"}
    staged.runner.initialize(resume=True)
    continuation = staged.runner._continuation
    assert staged.runner.request_governor_root == ledger_path.parent
    assert continuation.physical_path(staged.logical / "new/descendant.json") == (
        staged.root / "new/descendant.json")
    assert continuation.logical_path(staged.root / "new/descendant.json") == str(
        staged.logical / "new/descendant.json")
    with pytest.raises(FitSetError):
        continuation.logical_path(outer / "accumulation/shadow.json")
    write_json(ledger_path, {"policy": {"max_requests": 100}, "requests": []})
    with pytest.raises(FitSetError, match="reservations were reset"):
        staged.runner.initialize(resume=True)


def test_original_started_ledger_is_not_overwritten_by_settlement(tmp_path):
    staged = stage(tmp_path)
    ledger_path = staged.root / "request_governor/state.json"
    ledger = json.loads(ledger_path.read_bytes())
    ledger["requests"][0].update(status="started", completed_at=None, exception_type=None, usage={})
    write_json(ledger_path, ledger)
    seal(staged)
    staged.runner.initialize(resume=True)
    ledger["requests"][0].update(status="completed", completed_at=123, usage={"synthetic": 1})
    write_json(ledger_path, ledger)
    with pytest.raises(FitSetError, match="separate recovery receipt"):
        staged.runner.initialize(resume=True)


def test_supplied_external_anchor_allows_empty_internal_ledger_paths(tmp_path):
    staged = stage(tmp_path)
    outside = tmp_path / "sibling/request_governor/state.json"
    outside.parent.mkdir(parents=True)
    data = (staged.root / "request_governor/state.json").read_bytes()
    outside.write_bytes(data)
    logical = str(staged.logical.parent / "request_governor/state.json")
    identity = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    refs = ["manifest.json", "workers/00/worker_identity.json", "workers/00/buffer/stream.json"]
    value = write_deployment_continuation(
        staged.sidecar, system_root=staged.root, logical_root=staged.logical,
        deployment_id="new-code", current_manifest=staged.current,
        original_revision_protocol="fit-set-slot-revision-v5-synthetic",
        preserved_paths=refs, ledger_paths=[],
        path_map={logical: str(outside)},
        external_ledger_anchors={logical: {
            "identity": identity, "archive_ref": "raw_state/original_governor.json"}})
    staged.runner.deployment_continuation = staged.sidecar
    staged.runner.initialize(resume=True)
    assert staged.runner.request_governor_root == outside.parent
    assert value["ledger_anchors"] == []
    assert (staged.sidecar.parent / "raw_state/original_governor.json").read_bytes() == data


def test_runtime_cannot_use_a_new_unanchored_request_ledger(tmp_path):
    staged = stage(tmp_path)
    seal(staged)
    staged.runner._request_governor_root = tmp_path / "fresh-budget"
    with pytest.raises(FitSetError, match="runtime request ledger locator changed"):
        staged.runner.initialize(resume=True)
    assert not (staged.sidecar.parent / "activated.json").exists()


def test_raw_state_archive_cannot_write_through_a_directory_symlink(tmp_path):
    staged = stage(tmp_path)
    outside = tmp_path / "outside-archive"
    outside.mkdir()
    staged.sidecar.parent.mkdir(parents=True)
    (staged.sidecar.parent / "raw_state").symlink_to(outside, target_is_directory=True)
    with pytest.raises(FitSetError, match="archive escaped"):
        seal(staged)
    assert not list(outside.iterdir())
    assert not staged.sidecar.exists()


def offline_input(staged, variant, *, physical=False):
    variant_root = staged.root / "variants" / _variant_key(variant["variant_id"])
    refs = []
    for index in range(32):
        ref = f"variants/{_variant_key(variant['variant_id'])}/offline_images/{index:02d}.png"
        path = staged.root / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        data = f"synthetic image bytes {index}".encode()
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        refs.append({
            "path": str((staged.root if physical else staged.logical) / ref),
            "file_sha256": digest, "content_sha256": digest,
            "image_id": f"synthetic-frame-{index}", "kind": "frame", "frame_id": index,
            "source_frame_index": index, "source_hw": [4, 4], "frame_set_hash": "c" * 64,
        })
    value = {
        "schema_version": "v11-fit-set-offline-input/1.0",
        "variant_id": variant["variant_id"], "variant_sha256": content_sha256(variant),
        "qa_id": variant["qa_id"], "scene_id": variant["scene_id"], "split": variant["split"],
        "question_text": "Synthetic question record", "options": [],
        "frame_set_hash": "c" * 64, "image_refs": refs,
    }
    write_json(variant_root / "offline_input.json", value)
    return value


def synthetic_variant(index=0):
    return {"variant_id": f"synthetic-{index}@g0m0", "qa_id": f"synthetic-{index}",
            "scene_id": "synthetic-scene", "split": "induction",
            "question_type": "object_counting", "evidence_state": "g0m0"}


def test_old_image_refs_are_returned_unchanged_and_freeze_reuses_bytes(tmp_path):
    staged = stage(tmp_path)
    variant = synthetic_variant()
    old = offline_input(staged, variant)
    seal(staged)
    staged.runner.initialize(resume=True)
    input_path = staged.root / "variants" / _variant_key(variant["variant_id"]) / "offline_input.json"
    before = input_path.read_bytes()
    assert staged.runner.offline_case_input(variant) == old
    staged.runner._freeze_offline_case_input(variant, None)
    assert input_path.read_bytes() == before
    image = staged.runner.logical_file(old["image_refs"][0]["path"], old["image_refs"][0]["file_sha256"])
    image.write_bytes(b"synthetic corruption")
    with pytest.raises(FitSetError, match="content changed"):
        staged.runner.offline_case_input(variant)


@pytest.mark.parametrize("bad_ref", ["other-parent", "traversal", "physical-ref", "symlink"])
def test_offline_image_parent_and_locator_escapes_are_rejected(tmp_path, bad_ref):
    staged = stage(tmp_path)
    variant = synthetic_variant()
    old = offline_input(staged, variant)
    seal(staged)
    staged.runner.initialize(resume=True)
    ref = old["image_refs"][0]
    if bad_ref == "other-parent":
        ref["path"] = str(staged.logical / "another/offline_images/00.png")
    elif bad_ref == "traversal":
        ref["path"] = ref["path"].replace("/offline_images/", "/offline_images/../offline_images/")
    elif bad_ref == "physical-ref":
        ref["path"] = str(staged.runner._continuation.physical_path(ref["path"]))
    else:
        image = staged.runner._continuation.physical_path(ref["path"])
        external = tmp_path / "external.png"
        external.write_bytes(image.read_bytes())
        image.unlink()
        image.symlink_to(external)
    input_path = staged.root / "variants" / _variant_key(variant["variant_id"]) / "offline_input.json"
    write_json(input_path, old)
    with pytest.raises(FitSetError):
        staged.runner.offline_case_input(variant)


def test_new_frozen_images_use_old_logical_root_without_new_schema_fields(tmp_path):
    staged = stage(tmp_path)
    seal(staged)
    staged.runner.initialize(resume=True)
    variant = synthetic_variant(1)
    item = SimpleNamespace(
        episode=SimpleNamespace(frame_set=SimpleNamespace(
            frame_ids=list(range(32)), source_frame_indices=list(range(32)), frame_set_hash="c" * 64),
            question="Synthetic question", options=[]),
        pixels=[np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(32)])
    staged.runner._freeze_offline_case_input(variant, item)
    source = staged.runner.offline_case_input(variant)
    assert len(source["image_refs"]) == 32
    assert source["image_refs"][0]["path"].startswith(str(staged.logical / "variants"))
    assert "logical_path" not in source["image_refs"][0]


def methods():
    return [SkillSpecV11(
        skill_id=f"SYN-{index:02d}", version="1.0.0",
        question_type=task, evidence_state=state,
        skill_md=f"---\nname: synthetic-{index}\ndescription: Synthetic method.\n---\n\nInspect evidence.\n")
        for index, (task, state) in enumerate(ALL_FIT_SLOTS)]


def test_canonical_trace_identity_survives_physical_relocation(tmp_path):
    staged = stage(tmp_path)
    variant = synthetic_variant()
    offline_input(staged, variant)
    specs = methods()
    library = staged.root / "synthetic_library/library"
    initialize_worker_library(
        library, seed_skills=specs, owned_slots=["object_counting/g0m0"],
        run_identity=staged.original["identity_sha256"])
    parents, snapshot = _active(library)
    parent = parents[("object_counting", "g0m0")]
    key = f"{parent.skill_id}@{parent.version}"
    initial = {"geometry_3d": "unavailable", "metric_scale": "unavailable"}
    outcome = {
        "qa_id": variant["qa_id"], "scene_id": variant["scene_id"],
        "question_type": variant["question_type"], "answer": "2", "final_state": "answer",
        "episode_status": "answered",
        "delivered_skill_versions": [key],
        "initial_skill_selection": {
            "skill_version": key, "content_sha256": parent.content_sha256,
            "question_type": parent.question_type, "evidence_state": parent.evidence_state},
        "retrieval_records": [{
            "candidates": [{"skill_version": key, "delivered": True}],
            "delivered_skill_versions": [key], "delivered_content_sha256": {key: parent.content_sha256}}],
        "initial_evidence_states": initial, "evidence_history": [{"evidence_states": initial}],
        "rounds": [{"index": 0, "executed": True, "response_text": "synthetic",
                    "synthesis_source": "online_model_ok",
                    "program_source": "synthetic_record()", "results": []}],
    }
    outcome_path = staged.root / "variants" / _variant_key(variant["variant_id"]) / "outcome.json"
    write_json(outcome_path, outcome)
    runner = staged.runner
    runner._continuation = SimpleNamespace(
        logical_path=lambda path: str(staged.logical / Path(path).relative_to(staged.root)),
        physical_path=lambda path: staged.root / Path(path).relative_to(staged.logical))
    runner.experiment_split = None
    runner.learning_authorization = lambda _variant: {}
    runner.label_metadata = {variant["qa_id"]: {
        "scene_id": variant["scene_id"], "question_type": variant["question_type"],
        "reference_answer": "2", "options": []}}
    worker = object.__new__(FitSetWorkerEvolution)
    worker.runner, worker.library = runner, library
    worker.root = staged.root / "workers/00"
    worker.config = FitSetEvolutionConfig(learning_input_mode="full_cases_v1")
    collected = []
    worker.stream = SimpleNamespace(
        ingest=lambda case, **_kwargs: collected.append(case),
        attempt_ready=lambda *_args, **_kwargs: None)
    row = {"skill_identity": {"snapshot_ref": snapshot["snapshot_id"]}}
    worker.ingest_result(variant, row)
    canonical = next((worker.root / "canonical_cases").glob("*.json"))
    logical_trace = str(staged.logical / outcome_path.relative_to(staged.root))
    before = canonical.read_bytes()
    seal(staged)
    runner.initialize(resume=True)
    worker.ingest_result(variant, row)
    assert canonical.read_bytes() == before
    assert collected[-1].trace_ref == logical_trace
    assert collected[-1].full_case.trace_ref == logical_trace


def pending_campaign(staged, monkeypatch, *, intents=1):
    import skill3d.evolution.fitset_stream_v11 as stream_module

    config = FitSetEvolutionConfig(scheduling_mode="batch_v1")
    parent = next(skill for skill in methods() if skill.slot_key == "object_counting/g0m0")
    request = SimpleNamespace(
        slot_key=parent.slot_key, parent_skill_version=f"{parent.skill_id}@{parent.version}",
        parent_content_sha256=parent.content_sha256, attempt_index=1, new_case_count=12,
        question_type=parent.question_type, evidence_state=parent.evidence_state,
        cases=tuple(SimpleNamespace(case_id=f"synthetic-case-{index}") for index in range(12)))
    token = content_sha256({
        "slot": request.slot_key, "parent": request.parent_skill_version,
        "cases": [case.case_id for case in request.cases], "attempt": 1})[:20]
    root = staged.root / "workers/00/campaigns" / token
    _write_create_once(root / "request.json", {
        "slot": request.slot_key, "parent": parent.model_dump(mode="json"),
        "parent_content_sha256": parent.content_sha256,
        "case_ids": [case.case_id for case in request.cases], "new_case_count": 12,
        "run_identity": staged.original["identity_sha256"]})
    write_json(root / "experience_bundle.json", {"synthetic": "retained original bundle"})
    for index in range(intents):
        intent = {"request": {
            "prompt_version": "fit-set-slot-revision-v5-synthetic",
            "prompt_sha256": hashlib.sha256(f"synthetic-{index}".encode()).hexdigest()},
            "started_at": f"synthetic-start-{index}"}
        slot_root = root / "revision/slots/object_counting/g0m0"
        write_json(slot_root / f"request_intent_{index + 1:02d}.json", intent)
        write_json(slot_root / f"attempt_{index + 1:02d}.json", {
            "request": intent["request"], "service_error": True, "exception_type": "Synthetic502"})
    monkeypatch.setattr(stream_module, "sanitized_slot_feedback", lambda _cases: {"signatures": []})
    runner = staged.runner
    runner.evolution_config = config
    runner._require_storage_reserve = lambda _key: None
    runner.variants = [synthetic_variant()]
    runner.panel, runner.panel_name = {}, "synthetic"
    runner.experiment_split = None
    runner.base_cfg = SimpleNamespace(retrieval_policy=SimpleNamespace(method_context_max_chars=9000))
    runner.revision_client_factory = lambda: object()
    runner._govern_client = lambda client, *_args: client
    worker = object.__new__(FitSetWorkerEvolution)
    worker.runner, worker.config = runner, config
    worker.root, worker.library = staged.root / "workers/00", staged.root / "unused-library"
    worker.owned_slots = [request.slot_key]
    worker._parent_for = lambda _request: parent
    return worker, request, root


def test_old_502_counts_once_and_new_schedule_keeps_eighteen_slot_budget(tmp_path, monkeypatch):
    staged = stage(tmp_path)
    worker, request, root = pending_campaign(staged, monkeypatch)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    seal(staged)
    staged.runner.initialize(resume=True)
    seen = []

    class StopReviser:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def _revise_slot(self, **_kwargs):
            raise RevisionOperationalError("synthetic engineering stop")

    monkeypatch.setattr("skill3d.evolution.fitset_stream_v11.FitSetSlotReviser", StopReviser)
    checkpoint = (staged.root / "workers/00/buffer/stream.json").read_bytes()
    with pytest.raises(RevisionOperationalError, match="engineering stop"):
        worker.attempt(request)
    deployment = root / "deployments/new-code"
    schedule = json.loads((deployment / "schedule.json").read_bytes())
    assert schedule["revision_requests_used"] == 1
    assert schedule["remaining_revision_requests"] == 17
    assert schedule["max_revision_requests"] == 18
    assert seen[0]["output_root"] == deployment / "revision"
    assert not (root / "result.json").exists()
    assert (staged.root / "workers/00/buffer/stream.json").read_bytes() == checkpoint
    assert all(path.read_bytes() == data for path, data in before.items())
    original_intent = next(root.glob("revision/slots/*/*/request_intent_*.json"))
    copy_path = deployment / "revision/slots/object_counting/g0m0/request_intent_01.json"
    write_json(copy_path, json.loads(original_intent.read_bytes()))
    assert revision_intents_used(worker.root, request.slot_key) == 1
    intent = json.loads(original_intent.read_bytes())
    intent["request"]["prompt_version"] = FIT_REVISION_PROMPT_VERSION
    intent["started_at"] = "synthetic-new-start"
    write_json(copy_path, intent)
    assert revision_intents_used(worker.root, request.slot_key) == 2


def test_diagnosis_intents_are_counted_across_deployments_without_timestamps(tmp_path, monkeypatch):
    staged = stage(tmp_path)
    worker, request, root = pending_campaign(staged, monkeypatch)
    intent = {"input_sha256": "e" * 64, "request_identity": {"sha256": "f" * 64}}
    write_json(root / "diagnosis/synthetic/request_intent.json", intent)
    write_json(root / "deployments/new-code/diagnosis/synthetic/request_intent.json", intent)
    assert revision_intents_used(worker.root, request.slot_key) == 3


@pytest.mark.parametrize("mode", ["exhausted", "ambiguous", "same-protocol", "exchange-mismatch"])
def test_no_budget_or_unknown_old_request_cannot_advance_attempt(tmp_path, monkeypatch, mode):
    staged = stage(tmp_path)
    worker, request, root = pending_campaign(
        staged, monkeypatch, intents=18 if mode == "exhausted" else 1)
    if mode == "ambiguous":
        next(root.glob("revision/slots/*/*/attempt_*.json")).unlink()
    elif mode == "exchange-mismatch":
        exchange = next(root.glob("revision/slots/*/*/attempt_*.json"))
        recorded = json.loads(exchange.read_bytes())
        recorded["request"]["prompt_sha256"] = "0" * 64
        write_json(exchange, recorded)
    elif mode == "same-protocol":
        old = json.loads(next(root.glob("revision/slots/*/*/request_intent_*.json")).read_bytes())
        old["request"]["prompt_version"] = FIT_REVISION_PROMPT_VERSION
        prior = root / "deployments/prior-code/revision/slots/object_counting/g0m0"
        write_json(prior / "request_intent_01.json", old)
        write_json(prior / "attempt_01.json", {
            "request": old["request"], "service_error": True, "exception_type": "Synthetic502"})
    seal(staged)
    staged.runner.initialize(resume=True)
    before = (staged.root / "workers/00/buffer/stream.json").read_bytes()
    monkeypatch.setattr("skill3d.evolution.fitset_stream_v11.FitSetSlotReviser",
                        lambda **_kwargs: pytest.fail("no new request may be prepared"))
    with pytest.raises((RevisionOperationalError, FitSetError), match="unresolved|ambiguous|identity changed"):
        worker.attempt(request)
    assert not (root / "result.json").exists()
    assert not (root / "decision.json").exists()
    assert (staged.root / "workers/00/buffer/stream.json").read_bytes() == before


def test_later_campaign_exhaustion_is_not_an_unresolved_old_request(tmp_path, monkeypatch):
    staged = stage(tmp_path)
    worker, request, root = pending_campaign(staged, monkeypatch, intents=18)
    for path in root.glob("revision/slots/*/*/attempt_*.json"):
        exchange = json.loads(path.read_bytes())
        exchange.update(service_error=False, response_text='{"action":"no_change"}')
        write_json(path, exchange)
    _write_create_once(root / "result.json", {
        "outcome": "no_change", "campaign_id": root.name, "reason": "Synthetic completed method decision"})
    seal(staged)
    staged.runner.initialize(resume=True)
    request.attempt_index = 2
    monkeypatch.setattr("skill3d.evolution.fitset_stream_v11.FitSetSlotReviser",
                        lambda **_kwargs: pytest.fail("exhausted budget must not prepare a request"))
    result = worker.attempt(request)
    assert result.outcome == "no_change" and result.reason == "slot_revision_request_limit"
    assert result.campaign_id != root.name
    assert revision_intents_used(worker.root, request.slot_key) == 18


def test_optional_lease_covers_episode_and_preflight_even_on_failure(tmp_path):
    runner = runner_for(tmp_path, {})
    variant = synthetic_variant()
    events = []

    @contextmanager
    def lease(actual):
        assert actual == variant
        events.append("lease-enter")
        try:
            yield
        finally:
            events.append("lease-exit")

    def work(actual):
        assert actual == variant and events[-1] == "lease-enter"
        events.append("work")
        return {"synthetic": True}

    runner.published_inputs = SimpleNamespace(episode_lease=lease)
    runner._run_one_locked = work
    runner._preflight_one_leased = work
    assert runner._run_one(variant) == {"synthetic": True}
    assert runner._preflight_one(variant) == {"synthetic": True}
    assert events == ["lease-enter", "work", "lease-exit"] * 2
    runner._run_one_locked = lambda _variant: (_ for _ in ()).throw(FitSetError("synthetic stop"))
    with pytest.raises(FitSetError):
        runner._run_one(variant)
    assert events[-2:] == ["lease-enter", "lease-exit"]
    runner.published_inputs = None
    runner._run_one_locked = lambda _variant: {"default": True}
    assert runner._run_one(variant) == {"default": True}


def test_spawned_gate_provider_keeps_the_optional_lease(tmp_path):
    variant = synthetic_variant()
    runner = runner_for(tmp_path, {})
    calls = []

    @contextmanager
    def lease(actual):
        calls.append(actual["variant_id"])
        yield

    provider = SimpleNamespace(
        receipt=lambda *_args: {}, episode_lease=lease,
        for_spawned_episode=lambda actual: SimpleNamespace(
            receipt=lambda *_args: {}, episode_lease=lease))
    runner.item_provider = runner.published_inputs = provider
    clone = SystemVariantRunner._gate_spawn_runner(runner, variant)
    assert clone is not runner and clone.published_inputs is not provider
    assert clone.item_provider is clone.published_inputs
    with clone._input_episode_lease(variant):
        pass
    assert calls == [variant["variant_id"]]
