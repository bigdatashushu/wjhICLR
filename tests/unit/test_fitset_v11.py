from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import pytest

from skill3d.evaluation.direct_qwen_v11 import (
    DIRECT_RESPONSE_SCHEMA,
    DIRECT_RUN_SCHEMA,
)
from skill3d.evaluation.fitset_v11 import (
    ALL_FIT_SLOTS,
    FIT_SKILL_BUNDLE_SCHEMA,
    PANEL_SCHEMA,
    PROCESS_SCHEDULER,
    REACHABLE_FIT_SLOTS,
    SYSTEM_RESULT_SCHEMA,
    SYSTEM_RUN_SCHEMA,
    UNREACHABLE_FIT_SLOTS,
    SystemVariantRunner,
    _process_shard_worker,
    _stable_slot_shards,
    audit_fitset_pilot,
    build_fit_skill_bundle,
    build_fitset_comparison,
    direct_rows_for_variants,
    fit_slot_key,
    load_fit_skill_bundle,
    load_fitset_panel,
    load_variant_bindings,
    migrate_empty_m4_bindings,
    select_panel_variants,
)
from skill3d.evolution.fitset_revision_v11 import (
    FIT_FEEDBACK_SCHEMA,
    FitSetSlotReviser,
    _parse_response,
    build_method_feedback,
    load_active_parent_slots,
)
from skill3d.routing.task_classifier import TASK_TYPES
from skill3d.schemas import SkillSpecV11
from skill3d.schemas.answer import CANONICAL_UNIT_BY_QUESTION_TYPE


class _SpawnTimeoutProbe:
    def _run_one(self, variant):
        import threading

        from skill3d.sandbox.kernel import _Timeout

        with _Timeout(1):
            main_thread = threading.current_thread() is threading.main_thread()
        return {
            "variant_id": variant["variant_id"],
            "status": "completed",
            "main_thread": main_thread,
        }

    def _run_shard(self, index, owned, pending, stop_event, accept):
        for variant in pending:
            accept(self._run_one(variant))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _hash(value) -> str:
    from skill3d.evaluation.fitset_v11 import content_sha256

    return content_sha256(value)


def _variant(
    qa_id: str,
    state: str,
    *,
    status: str = "ready",
    task: str = "object_counting",
) -> dict:
    return {
        "variant_id": f"{qa_id}@{state}",
        "qa_id": qa_id,
        "scene_id": "scene-1",
        "question_type": task,
        "evidence_state": state,
        "split": "induction",
        "artifact_ref": f"/frozen/{state}/artifact.json",
        "binding_key": state[1] * 64,
        "binding_manifest_sha256": state[2] * 64,
        "system_input_status": status,
        "accumulation_role": True,
        "comparison_role": True,
    }


def _panel_rows() -> list[dict]:
    return [
        _variant("10", "g1m0", task="object_abs_distance"),
        _variant(
            "10",
            "g1m1",
            status="binding_preparation_required",
            task="object_abs_distance",
        ),
    ]


def _panel_file(tmp_path: Path) -> Path:
    path = tmp_path / "panel.json"
    rows = _panel_rows()
    body = {
        "schema_version": PANEL_SCHEMA,
        "panels": {
            "ready_579": {
                "system_input_variants": 1,
                "skill_slot_count": 1,
                "skill_slots": ["object_abs_distance/g1m0"],
            },
            "recommended_core_897": {
                "system_input_variants": 2,
                "skill_slot_count": 2,
                "skill_slots": [
                    "object_abs_distance/g1m0",
                    "object_abs_distance/g1m1",
                ],
            },
        },
        "skill_accumulation": {
            "grouping_key": ["question_type", "evidence_state"],
            "active_snapshot_physical_slot_count": 24,
            "reachable_slot_count": 19,
            "unreachable_slots": [
                fit_slot_key(*slot) for slot in UNREACHABLE_FIT_SLOTS
            ],
        },
        "system_variants": rows,
        "direct_qwen_rows": [
            {
                "qa_id": "10",
                "scene_id": "scene-1",
                "question_type": "object_abs_distance",
                "split": "induction",
                "system_variant_ids": ["10@g1m0", "10@g1m1"],
            },
        ],
    }
    _write(path, {**body, "proposal_sha256": _hash(body)})
    return path


def test_panel_selection_preserves_direct_reuse_mapping(tmp_path):
    panel = load_fitset_panel(_panel_file(tmp_path))
    ready = select_panel_variants(panel, "ready_579")
    core = select_panel_variants(panel, "recommended_core_897")
    assert [row["variant_id"] for row in ready] == ["10@g1m0"]
    assert [row["variant_id"] for row in core] == [
        "10@g1m0",
        "10@g1m1",
    ]
    direct = direct_rows_for_variants(panel, core)
    assert direct == [{
        "qa_id": "10",
        "scene_id": "scene-1",
        "question_type": "object_abs_distance",
        "split": "induction",
        "system_variant_ids": ["10@g1m0", "10@g1m1"],
    }]


def test_variant_binding_index_supports_qa_fallback_and_variant_override(
    tmp_path,
):
    variants = _panel_rows()[:2]
    path = tmp_path / "bindings.json"
    _write(path, {
        "bindings": {
            "10": {
                "key": "1" * 64,
                "manifest_sha256": "m" * 64,
                "scene_artifact_ref": "/frozen/g1m0/artifact.json",
            },
            "10@g1m1": {
                "key": "1" * 64,
                "manifest_sha256": "m" * 64,
                "scene_artifact_ref": "/frozen/g1m1/artifact.json",
            },
        },
    })
    for row in variants:
        row["binding_key"] = "1" * 64
        row["binding_manifest_sha256"] = "m" * 64
    loaded = load_variant_bindings(path, variants)
    assert loaded["10@g1m0"]["scene_artifact_ref"].endswith(
        "/g1m0/artifact.json")
    assert loaded["10@g1m1"]["scene_artifact_ref"].endswith(
        "/g1m1/artifact.json")


def test_empty_m4_binding_migration_is_explicit_and_replayable(tmp_path):
    from skill3d.infra.m5_identity import (
        content_sha256 as m5_content_sha256,
        file_identity as m5_file_identity,
        source_identity,
    )
    from skill3d.online.runner import PreparedObjectBinding
    from skill3d.segmentation import prepared_binding_store, question_binding
    from skill3d.segmentation.prepared_binding_store import PreparedBindingStore

    code_root = Path(question_binding.__file__).resolve().parents[1]
    current_sources = source_identity(code_root)
    parent_sources = copy.deepcopy(current_sources)
    parent_sources["segmentation/question_binding.py"] = {
        "sha256": (
            "945301bce9b0f5167bd5c8feb3878ec0873ac384288930bccb0e7b702a2cd72f"
        ),
        "size": 52507,
    }
    generation_identity = {"sources": parent_sources}
    generation = {
        "identity": generation_identity,
        "sha256": m5_content_sha256(generation_identity),
    }
    binding = PreparedObjectBinding(
        objects=[],
        stats={
            "status": "m4_unavailable",
            "generation": generation,
        },
        materialized=False,
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    input_sha = "i" * 64
    request = {
        "input_sha256": input_sha,
        "implementation_sources": parent_sources,
        "store_source": m5_file_identity(
            Path(prepared_binding_store.__file__)),
    }
    request["sha256"] = m5_content_sha256(request)
    source_store = PreparedBindingStore(tmp_path / "source-store")
    parent = source_store.publish(
        request_identity=request,
        scene_artifact=artifact,
        scene_input_identity={"sha256": input_sha},
        binding=binding,
    )
    source_manifest = tmp_path / "source-bindings.json"
    _write(source_manifest, {"bindings": {"qa-1": parent}})
    variant = {
        "variant_id": "qa-1@g0m0",
        "qa_id": "qa-1",
        "scene_id": "scene-1",
        "question_type": "object_counting",
        "evidence_state": "g0m0",
        "split": "induction",
        "artifact_ref": str(artifact.resolve()),
        "binding_key": parent["key"],
        "binding_manifest_sha256": parent["manifest_sha256"],
    }
    output = tmp_path / "migration"
    manifest = migrate_empty_m4_bindings(
        panel={
            "proposal_sha256": "p" * 64,
            "panel_file_sha256": "f" * 64,
        },
        variants=[variant],
        source_manifest_path=source_manifest,
        output_dir=output,
        min_free_gib=0,
    )
    assert manifest["n_migrated"] == 1
    resolved = load_variant_bindings(
        output / "prepared_bindings.json", [variant])
    migrated = resolved[variant["variant_id"]]
    assert migrated["key"] != parent["key"]
    loaded = PreparedBindingStore(migrated["store_root"]).load(
        migrated["key"],
        expected_request_identity=migrated["request_identity"],
        scene_artifact=artifact,
        scene_input_identity={"sha256": input_sha},
    )
    assert loaded.stats["generation"]["identity"]["sources"] == current_sources


def _method(
    task: str,
    index: int,
    *,
    state: str = "g0m0",
    skill_id: str | None = None,
    version: str = "2.0.0",
    name: str | None = None,
) -> SkillSpecV11:
    method_name = name or f"fit-method-{index}"
    text = (
        "---\n"
        f"name: {method_name}\n"
        f"description: Reusable method for exact slot {task}/{state}.\n"
        "---\n\n"
        "# Method\n\n"
        f"Inspect authorized evidence for {state}, use available tools, and "
        f"preserve exact-slot identity marker {index}.\n"
    )
    return SkillSpecV11(
        skill_id=skill_id or f"FIT-S{index:02d}",
        version=version,
        question_type=task,
        evidence_state=state,
        skill_md=text,
    )


def test_fit_bundle_freezes_exact_slots_and_preserves_other_parents(tmp_path):
    parents, _ = load_active_parent_slots(
        Path(__file__).resolve().parents[2] / "skill_library")
    selected = tuple(REACHABLE_FIT_SLOTS[:16])
    candidates = {}
    for index, slot in enumerate(selected, start=1):
        parent = parents[slot]
        candidates[slot] = SkillSpecV11(
            skill_id=parent.skill_id,
            version="1.1.0",
            question_type=parent.question_type,
            evidence_state=parent.evidence_state,
            skill_md=(
                parent.skill_md.rstrip("\n")
                + f"\n\nExact-slot fit marker {index}.\n"
            ),
        )
    root = tmp_path / "bundle"
    manifest = build_fit_skill_bundle(
        output_dir=root,
        bundle_id="fit-sixteen",
        candidates=candidates,
        active_parents=parents,
        source_run_identity={"sha256": "a" * 64},
    )
    assert manifest["schema_version"] == FIT_SKILL_BUNDLE_SCHEMA
    assert manifest["physical_slot_count"] == 24
    assert manifest["reachable_slot_count"] == 19
    assert manifest["revised_slot_count"] == 16
    assert manifest["preserved_parent_slot_count"] == 8
    assert len(list(root.glob("skills/*/*/SKILL.md"))) == 24
    loaded = load_fit_skill_bundle(root / "manifest.json")
    assert set(loaded.slots) == set(ALL_FIT_SLOTS)
    assert {
        skill.role for slot, skill in loaded.slots.items()
        if slot in selected
    } == {"revised_candidate"}
    assert {
        skill.role for slot, skill in loaded.slots.items()
        if slot in set(UNREACHABLE_FIT_SLOTS)
    } == {"active_parent_unreachable"}
    assert len({
        loaded.slots[slot].content_sha256 for slot in selected
    }) == 16


def test_comparison_reuses_full_direct_run_and_rescores_both_arms(tmp_path):
    panel_path = tmp_path / "panel.json"
    variant = _variant("10", "g1m0")
    body = {
        "schema_version": PANEL_SCHEMA,
        "panels": {
            name: {
                "system_input_variants": 1,
                "skill_slot_count": 1,
                "skill_slots": ["object_counting/g1m0"],
            }
            for name in ("ready_579", "recommended_core_897")
        },
        "skill_accumulation": {
            "grouping_key": ["question_type", "evidence_state"],
            "active_snapshot_physical_slot_count": 24,
            "reachable_slot_count": 19,
            "unreachable_slots": [
                fit_slot_key(*slot) for slot in UNREACHABLE_FIT_SLOTS
            ],
        },
        "system_variants": [variant],
        "direct_qwen_rows": [{
            "qa_id": "10",
            "scene_id": "scene-1",
            "question_type": "object_counting",
            "split": "induction",
            "system_variant_ids": ["10@g1m0"],
        }],
    }
    _write(panel_path, {**body, "proposal_sha256": _hash(body)})
    panel = load_fitset_panel(panel_path)

    system_root = tmp_path / "system"
    _write(system_root / "summary.json", {
        "schema_version": SYSTEM_RUN_SCHEMA,
        "status": "complete",
    })
    result_name = hashlib.sha256(
        b"10@g1m0").hexdigest()[:20] + ".json"
    _write(system_root / "results" / result_name, {
        "schema_version": SYSTEM_RESULT_SCHEMA,
        "variant_id": "10@g1m0",
        "question_type": "object_counting",
        "evidence_state": "g1m0",
        "status": "completed",
        "final_state": "answer",
        "answer": "1",
        "service_errors": [],
        "model_request_count": 2,
        "tool_call_count": 1,
        "elapsed_s": 3.0,
    })

    direct_root = tmp_path / "direct"
    _write(direct_root / "manifest.json", {
        "schema_version": DIRECT_RUN_SCHEMA,
        # A completed larger direct run can be reused for this subset.
        "selected_qa_ids": ["10", "outside-subset"],
    })
    _write(direct_root / "summary.json", {
        "schema_version": DIRECT_RUN_SCHEMA,
        "status": "complete",
    })
    _write(direct_root / "preflight" / "10.json", {
        "request_sha256": "r" * 64,
        "system_variant_ids": ["10@g1m0"],
    })
    _write(direct_root / "responses" / "10.json", {
        "schema_version": DIRECT_RESPONSE_SCHEMA,
        "qa_id": "10",
        "request_sha256": "r" * 64,
        "status": "accepted",
        "accepted": True,
        "answer": {"value": "2"},
        "elapsed_s": 1.0,
    })
    labels = {
        "10": {
            "qa_id": "10",
            "scene_id": "scene-1",
            "question_type": "object_counting",
            "options": [],
            "reference_answer": "2",
        },
    }
    report = build_fitset_comparison(
        panel=panel,
        panel_name="ready_579",
        system_root=system_root,
        direct_root=direct_root,
        label_metadata=labels,
        output_dir=tmp_path / "comparison",
    )
    all_rows = report["groups"]["all"]
    assert all_rows["system_score"] == 0.1
    assert all_rows["direct_score"] == 1.0
    assert all_rows["system_minus_direct"] == -0.9
    assert report["n_variants"] == report["n_unique_qa"] == 1


def _fake_accumulation(tmp_path: Path):
    root = tmp_path / "accumulation"
    variants = []
    labels = {}
    selected = [
        (task, state)
        for state in ("g0m0", "g1m0")
        for task in TASK_TYPES
    ]
    for index, (task, state) in enumerate(selected, start=1):
        qa_id = f"fitcase{index}"
        variant_id = f"{qa_id}@{state}"
        variants.append({
            "variant_id": variant_id,
            "qa_id": qa_id,
            "scene_id": f"fitscene{index}",
            "question_type": task,
            "evidence_state": state,
            "split": "induction",
        })
        is_option = CANONICAL_UNIT_BY_QUESTION_TYPE[task] == "option"
        answer = "A" if is_option else "2"
        labels[qa_id] = {
            "qa_id": qa_id,
            "scene_id": f"fitscene{index}",
            "question_type": task,
            "options": ["first", "second"] if is_option else [],
            "reference_answer": answer,
        }
        _write(root / "results" / f"{index:02d}.json", {
            "schema_version": SYSTEM_RESULT_SCHEMA,
            "variant_id": variant_id,
            "question_type": task,
            "evidence_state": state,
            "status": "completed",
            "final_state": "answer",
            "answer": answer,
            "service_errors": [],
            "legal_answer": True,
            "failure_code": "",
            "answer_basis": "visual_estimate",
            "answer_source": "tool_program",
            "main_gate_passed": False,
            "scene_route": "fallback_2d_only",
            "question_tool_scope": "visual_only",
            "model_request_count": 1,
            "tool_call_count": 0,
        })
    _write(root / "manifest.json", {
        "schema_version": SYSTEM_RUN_SCHEMA,
        "phase": "accumulation",
    })
    _write(root / "summary.json", {
        "schema_version": SYSTEM_RUN_SCHEMA,
        "status": "complete",
    })
    return root, variants, labels


class _RevisionClient:
    def __init__(self):
        self.calls = 0
        self.last_usage = {}
        self.last_generation = {}
        self.last_response_identity = {}

    def chat(self, messages, **kwargs):
        assert kwargs == {
            "max_tokens": 8192,
            "seed": 138,
            "temperature": 0.0,
            "enable_thinking": False,
        }
        prompt = messages[-1]["content"]
        prefix = "Parent complete SKILL.md:\n"
        suffix = "\nAuthorized Tool interfaces for this exact slot:\n"
        parent = prompt.split(prefix, 1)[1].split(suffix, 1)[0]
        candidate = (
            parent.rstrip("\n")
            + f"\n\nExact-slot revision marker {self.calls + 1}.\n"
        )
        self.calls += 1
        self.last_usage = {"prompt_tokens": 1, "completion_tokens": 1}
        self.last_generation = {"finish_reason": "stop"}
        self.last_response_identity = {"model": "qwen3.8-flash"}
        return json.dumps({
            "action": "revise",
            "full_skill_md": candidate,
            "modification_reason": "Add an exact-slot fail-closed check.",
            "change_hypothesis": "Add a consistency check.",
            "changed_sections": ["__body__"],
            "expected_process_effect": "More consistent decisions.",
            "evidence_signature_refs": [],
        })


@pytest.mark.parametrize(
    "skill_md",
    [
        "---\nname: count\ndescription: Count objects.\n---\n\nMethod",
        "---\r\nname: count\r\ndescription: Count objects.\r\n---\r\n\r\nMethod\r\n\r\n",
    ],
)
def test_fitset_revision_normalizes_canonical_skill_newlines(skill_md):
    value = _parse_response(json.dumps({
        "action": "revise",
        "full_skill_md": skill_md,
        "modification_reason": "Keep the complete method.",
        "change_hypothesis": "Keep complete source.",
        "changed_sections": ["__body__"],
        "expected_process_effect": "Preserve the method.",
        "evidence_signature_refs": [],
    }))
    assert "\r" not in value["full_skill_md"]
    assert value["full_skill_md"].endswith("\n")
    assert not value["full_skill_md"].endswith("\n\n")


def test_feedback_is_sanitized_and_ungated_freeze_is_refused(tmp_path):
    accumulation, variants, labels = _fake_accumulation(tmp_path)
    feedback = build_method_feedback(
        variants=variants,
        accumulation_root=accumulation,
        label_metadata=labels,
    )
    assert feedback["schema_version"] == FIT_FEEDBACK_SCHEMA
    encoded = json.dumps(feedback, ensure_ascii=False)
    assert all(variant["qa_id"] not in encoded for variant in variants)
    assert all(variant["scene_id"] not in encoded for variant in variants)
    assert all(
        row["reference_answer"] not in encoded
        for row in labels.values()
        if len(row["reference_answer"]) > 1
    )
    client = _RevisionClient()
    root = tmp_path / "revision"
    reviser = FitSetSlotReviser(
        output_root=root,
        panel={"proposal_sha256": "p" * 64},
        panel_name="ready_579",
        variants=variants,
        accumulation_root=accumulation,
        label_metadata=labels,
        library_root=Path(__file__).resolve().parents[2] / "skill_library",
        client=client,
    )
    with pytest.raises(Exception, match="一次性 reviser"):
        reviser.run(resume=False)
    assert client.calls == 0
    assert feedback["slot_count"] == 16


def test_system_runner_is_variant_keyed_and_resumes_without_reexecution(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    from skill3d.adapters.episode_source import load_synthetic_items
    from skill3d.evaluation import skill_ablation_v11 as ablation
    from skill3d.legacy import readers as legacy_readers
    from skill3d.online import runner as online_runner
    from skill3d.online.runner import (
        EpisodeOutcome,
        OnlineRunConfig,
        PreparedObjectBinding,
    )
    from skill3d.segmentation import prepared_binding_store
    from skill3d.segmentation import question_binding

    item = load_synthetic_items(
        "induction",
        question_types=["object_counting"],
        n_frames=32,
        frame_size=(8, 8),
    )[0]
    item.episode.ground_truth = ""
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    variant = {
        "variant_id": f"{item.episode.qa_id}@g0m0",
        "qa_id": item.episode.qa_id,
        "scene_id": item.episode.scene_name,
        "question_type": "object_counting",
        "evidence_state": "g0m0",
        "split": "induction",
        "artifact_ref": str(artifact),
        "binding_key": "b" * 64,
        "binding_manifest_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        ablation,
        "initial_state_for_item",
        lambda *_args, **_kwargs: "g0m0",
    )
    monkeypatch.setattr(
        ablation,
        "input_identity",
        lambda *_args, **_kwargs: {"sha256": "i" * 64},
    )
    monkeypatch.setattr(
        prepared_binding_store,
        "load_prepared_binding_reference",
        lambda *_args, **_kwargs: PreparedObjectBinding(),
    )
    monkeypatch.setattr(
        legacy_readers,
        "load_artifact_v5",
        lambda *_args, **_kwargs: SimpleNamespace(
            depth_maps="depth",
            c2w_list="c2w",
            intrinsics="intrinsics",
            depth_conf="confidence",
            point_map="points",
            reconstruction_input_identity="scene-input",
            frame_set_hash="",
            artifact_id="artifact",
        ),
    )
    monkeypatch.setattr(
        online_runner,
        "_load_npy",
        lambda _value: [],
    )
    monkeypatch.setattr(
        question_binding,
        "binding_parameters",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        question_binding,
        "validate_prepared_binding",
        lambda *_args, **_kwargs: None,
    )
    calls = []

    def fake_run_split(items, cfg, **kwargs):
        calls.append((items, cfg, kwargs))
        outcome = EpisodeOutcome(
            qa_id=item.episode.qa_id,
            scene_name=item.episode.scene_name,
            question_type="object_counting",
            final_state="answer",
            answer="2",
            predicted="2.0",
            is_mca=False,
            correct=None,
            mra_value=None,
            scoring_status="reference_unavailable",
            answer_parse_status="parsed",
            submission_status="accepted",
            legal_answer=True,
            task="object_counting",
            episode_status="answered",
                overall_quality=float("nan"),
            initial_evidence_states={
                "geometry_3d": "unavailable",
                "metric_scale": "unavailable",
            },
            initial_skill_selection={"evidence_state": "g0m0"},
        )
        return [outcome], SimpleNamespace(
            model_dump=lambda **_kwargs: {"status": "completed"})

    monkeypatch.setattr(online_runner, "run_split", fake_run_split)
    skill = _method("object_counting", 1).model_copy(update={
        "evidence_state": "g0m0",
    })
    panel = {
        "proposal_sha256": "p" * 64,
        "panel_file_sha256": "f" * 64,
    }

    def build_runner():
        return SystemVariantRunner(
            output_root=tmp_path / "system",
            panel=panel,
            panel_name="ready_579",
            variants=[variant],
            item_provider=lambda _variant: item,
            input_receipt_provider=lambda _variant, _item: {
                "request_sha256": "r" * 64,
            },
            input_contract_identity={
                "direct_manifest_sha256": "d" * 64,
            },
            binding_refs={
                variant["variant_id"]: {
                    "key": "b" * 64,
                    "manifest_sha256": "c" * 64,
                    "scene_input_identity_sha256": "i" * 64,
                },
            },
            base_cfg=OnlineRunConfig(
                mode="real",
                semantic_review_config=None,
            ),
            phase="accumulation",
            min_free_gib=0,
            skills_for_variant=lambda _variant: (
                [skill],
                {
                    "snapshot_ref": "test-snapshot",
                    "manifest_sha256": "a" * 64,
                },
            ),
        )

    preflight = build_runner().preflight(
        resume=False,
        variant_ids=[variant["variant_id"]],
    )
    assert preflight["model_requests"] == 0
    assert preflight["completed_preflights"] == 1
    assert calls == []
    manifest = json.loads(
        (tmp_path / "system" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["code_identity"]["base_commit"]
    assert manifest["code_identity"]["source"] in {
        "git_worktree", "startup_receipt"}
    first = build_runner().run(resume=True)
    second = build_runner().run(resume=True)
    assert first["status"] == second["status"] == "complete"
    assert len(calls) == 1
    result = json.loads(next(
        (tmp_path / "system" / "results").glob("*.json")
    ).read_text(encoding="utf-8"))
    assert result["variant_id"] == variant["variant_id"]
    assert result["scoring_status"] == "reference_unavailable"
    outcome = json.loads(next(
        (tmp_path / "system" / "variants").glob("*/outcome.json")
    ).read_text(encoding="utf-8"))
    assert outcome["overall_quality"] is None


def test_shared_reconstruction_file_hash_is_cached_by_stable_file_stamp(
    tmp_path,
    monkeypatch,
):
    from skill3d.evaluation import skill_ablation_v11 as ablation
    from skill3d.infra import m5_identity

    path = tmp_path / "shared.npy"
    with path.open("wb") as stream:
        stream.truncate(m5_identity.HASH_BLOCK_BYTES)
    stamp = m5_identity._file_stamp(path)
    monkeypatch.setattr(
        m5_identity.time,
        "time_ns",
        lambda: max(stamp[3:]) + m5_identity.FILE_HASH_SETTLE_NS + 1,
    )
    m5_identity._file_digest.cache_clear()
    first = ablation._file_hash(path)
    second = ablation._file_hash(path)
    info = m5_identity._file_digest.cache_info()
    assert first == second
    assert info.misses == 1
    assert info.hits == 1


def test_slot_shards_are_stable_and_never_split_a_slot():
    variants = [
        {
            "variant_id": f"qa-{index}@g0m0",
            "qa_id": f"qa-{index}",
            "scene_id": scene,
            "question_type": "object_counting",
            "evidence_state": "g0m0",
            "split": "induction",
        }
        for index, scene in enumerate(("scene-b", "scene-a", "scene-b", "scene-a"))
    ]
    shards = _stable_slot_shards(variants, 4)
    assert shards == _stable_slot_shards(variants, 4)
    slot_workers = {}
    for worker, shard in enumerate(shards):
        for row in shard:
            prior = slot_workers.setdefault(row["slot_key"], worker)
            assert prior == worker
    assert sorted(
        row["variant_id"] for shard in shards for row in shard
    ) == sorted(row["variant_id"] for row in variants)


def test_spawn_worker_runs_sigalrm_timeout_on_child_main_thread():
    context = multiprocessing.get_context("spawn")
    messages = context.Queue()
    stop_event = context.Event()
    process = context.Process(
        target=_process_shard_worker,
        args=(
            _SpawnTimeoutProbe(),
            0,
            [{"variant_id": "qa-1@g0m0", "scene_id": "scene-a"}],
            [{"variant_id": "qa-1@g0m0", "scene_id": "scene-a"}],
            stop_event,
            messages,
        ),
    )
    process.start()
    process.join(timeout=10)
    assert not process.is_alive()
    assert process.exitcode == 0
    first = messages.get(timeout=2)
    second = messages.get(timeout=2)
    assert first["kind"] == "result"
    assert first["result"]["main_thread"] is True
    assert second == {"kind": "done", "shard_index": 0}
    messages.close()
    messages.join_thread()
