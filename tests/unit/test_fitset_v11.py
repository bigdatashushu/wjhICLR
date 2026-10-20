from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time

from skill3d.evaluation.direct_qwen_v11 import (
    DIRECT_RESPONSE_SCHEMA,
    DIRECT_RUN_SCHEMA,
)
from skill3d.evaluation.fitset_v11 import (
    FIT_SKILL_BUNDLE_SCHEMA,
    SYSTEM_RESULT_SCHEMA,
    SYSTEM_RUN_SCHEMA,
    SystemVariantRunner,
    build_fit_skill_bundle,
    build_fitset_comparison,
    direct_rows_for_variants,
    load_fit_skill_bundle,
    load_fitset_panel,
    load_variant_bindings,
    select_panel_variants,
)
from skill3d.evolution.fitset_revision_v11 import (
    FIT_FEEDBACK_SCHEMA,
    FitSetEightMethodReviser,
    build_method_feedback,
)
from skill3d.routing.task_classifier import TASK_TYPES
from skill3d.schemas import SkillSpecV11
from skill3d.schemas.answer import CANONICAL_UNIT_BY_QUESTION_TYPE


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


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
        _variant(
            "11",
            "g1m1",
            status="binding_preparation_required",
            task="object_rel_distance",
        ),
    ]


def _panel_file(tmp_path: Path) -> Path:
    path = tmp_path / "panel.json"
    rows = _panel_rows()
    _write(path, {
        "schema_version": (
            "v11-same-question-system-vs-direct-qwen-proposal/1.0"),
        "proposal_sha256": "p" * 64,
        "panels": {
            "ready_579": {"system_input_variants": 1},
            "recommended_core_897": {"system_input_variants": 2},
            "maximum_1166": {"system_input_variants": 3},
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
            {
                "qa_id": "11",
                "scene_id": "scene-1",
                "question_type": "object_rel_distance",
                "split": "induction",
                "system_variant_ids": ["11@g1m1"],
            },
        ],
    })
    return path


def test_panel_selection_preserves_direct_reuse_mapping(tmp_path):
    panel = load_fitset_panel(_panel_file(tmp_path))
    ready = select_panel_variants(panel, "ready_579")
    core = select_panel_variants(panel, "recommended_core_897")
    maximum = select_panel_variants(panel, "maximum_1166")
    assert [row["variant_id"] for row in ready] == ["10@g1m0"]
    assert [row["variant_id"] for row in core] == [
        "10@g1m0",
        "10@g1m1",
    ]
    assert len(maximum) == 3
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


def _method(task: str, index: int) -> SkillSpecV11:
    text = (
        "---\n"
        f"name: fit-method-{index}\n"
        f"description: Reusable method for {task} across evidence states.\n"
        "---\n\n"
        "# Method\n\n"
        "Inspect authorized evidence, use available tools, and fall back to "
        "the original images when a capability is unavailable.\n"
    )
    return SkillSpecV11(
        skill_id=f"FIT-S{index:02d}",
        version="2.0.0",
        question_type=task,
        evidence_state="g0m0",
        skill_md=text,
    )


def test_fit_bundle_has_eight_bodies_and_read_only_state_adapters(tmp_path):
    methods = {
        task: _method(task, index)
        for index, task in enumerate(TASK_TYPES, start=1)
    }
    root = tmp_path / "bundle"
    manifest = build_fit_skill_bundle(
        output_dir=root,
        bundle_id="fit-eight",
        methods=methods,
        source_run_identity={"sha256": "a" * 64},
    )
    assert manifest["schema_version"] == FIT_SKILL_BUNDLE_SCHEMA
    assert len(list(root.glob("skills/*/SKILL.md"))) == 8
    loaded = load_fit_skill_bundle(root / "manifest.json")
    method = loaded.methods["object_counting"]
    adapters = [method.adapt(state) for state in ("g0m0", "g1m0", "g1m1")]
    assert len({adapter.content_sha256 for adapter in adapters}) == 1
    assert {adapter.evidence_state for adapter in adapters} == {
        "g0m0", "g1m0", "g1m1",
    }
    assert len({adapter.skill_id for adapter in adapters}) == 3


def test_comparison_reuses_full_direct_run_and_rescores_both_arms(tmp_path):
    panel_path = tmp_path / "panel.json"
    variant = _variant("10", "g1m0")
    _write(panel_path, {
        "schema_version": (
            "v11-same-question-system-vs-direct-qwen-proposal/1.0"),
        "proposal_sha256": "p" * 64,
        "panels": {
            "ready_579": {"system_input_variants": 1},
            "recommended_core_897": {"system_input_variants": 1},
            "maximum_1166": {"system_input_variants": 1},
        },
        "system_variants": [variant],
        "direct_qwen_rows": [{
            "qa_id": "10",
            "scene_id": "scene-1",
            "question_type": "object_counting",
            "split": "induction",
            "system_variant_ids": ["10@g1m0"],
        }],
    })
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
    for index, task in enumerate(TASK_TYPES, start=1):
        qa_id = f"fitcase{index}"
        variant_id = f"{qa_id}@g0m0"
        variants.append({
            "variant_id": variant_id,
            "qa_id": qa_id,
            "scene_id": f"fitscene{index}",
            "question_type": task,
            "evidence_state": "g0m0",
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
            "evidence_state": "g0m0",
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
        suffix = "\nAuthorized Tool interfaces by initial state:\n"
        parent = prompt.split(prefix, 1)[1].split(suffix, 1)[0]
        candidate = (
            parent.rstrip("\n")
            + "\n\nThe three initial evidence branches must preserve the "
            "same identity checks and fail closed on unavailable tools.\n"
        )
        self.calls += 1
        self.last_usage = {"prompt_tokens": 1, "completion_tokens": 1}
        self.last_generation = {"finish_reason": "stop"}
        self.last_response_identity = {"model": "qwen3.8-flash"}
        return json.dumps({
            "full_skill_md": candidate,
            "modification_reason": "Add a cross-state fail-closed check.",
        })


def test_feedback_is_sanitized_and_reviser_freezes_eight_methods(tmp_path):
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
    summary = FitSetEightMethodReviser(
        output_root=root,
        panel={"proposal_sha256": "p" * 64},
        panel_name="ready_579",
        variants=variants,
        accumulation_root=accumulation,
        label_metadata=labels,
        library_root=Path(__file__).resolve().parents[2] / "skill_library",
        client=client,
    ).run(resume=False)
    assert summary["status"] == "complete"
    assert summary["candidate_methods"] == client.calls == 8
    bundle = load_fit_skill_bundle(root / "frozen_bundle" / "manifest.json")
    assert set(bundle.methods) == set(TASK_TYPES)


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


def test_system_runner_uses_four_workers_and_stable_scene_schedule(
    tmp_path,
):
    from skill3d.online.runner import OnlineRunConfig

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
    skill = _method("object_counting", 1)
    runner = SystemVariantRunner(
        output_root=tmp_path / "parallel",
        panel={
            "proposal_sha256": "p" * 64,
            "panel_file_sha256": "f" * 64,
        },
        panel_name="ready_579",
        variants=variants,
        item_provider=lambda _variant: None,
        input_receipt_provider=lambda _variant, _item: {},
        input_contract_identity={"sha256": "d" * 64},
        binding_refs={},
        base_cfg=OnlineRunConfig(mode="real", semantic_review_config=None),
        phase="accumulation",
        min_free_gib=0,
        workers=4,
        skills_for_variant=lambda _variant: (
            [skill],
            {
                "snapshot_ref": "parallel-test",
                "manifest_sha256": "a" * 64,
            },
        ),
    )
    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def fake_run_one(variant):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            barrier.wait(timeout=2)
            time.sleep(0.02)
            return {
                "variant_id": variant["variant_id"],
                "status": "completed",
            }
        finally:
            with lock:
                state["active"] -= 1

    runner._run_one = fake_run_one
    summary = runner.run(resume=False)
    manifest = json.loads(
        (tmp_path / "parallel" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["workers"] == 4
    assert state["max_active"] == 4
    assert manifest["execution"] == {
        "workers": 4,
        "scheduler": "stable_scene_grouped_v1",
        "variant_ids": [
            "qa-1@g0m0",
            "qa-3@g0m0",
            "qa-0@g0m0",
            "qa-2@g0m0",
        ],
        "variant_ids_sha256": manifest["execution"]["variant_ids_sha256"],
    }
