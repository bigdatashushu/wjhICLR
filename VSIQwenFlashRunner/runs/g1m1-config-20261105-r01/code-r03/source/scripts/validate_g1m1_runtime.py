#!/usr/bin/env python3
"""Exercise real metric episode preparation for eight tasks without model calls."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in list(os.environ):
        if any(word in name.upper() for word in
               ("KEY", "TOKEN", "SECRET", "PASSWORD", "ENDPOINT", "PROXY")):
            os.environ.pop(name, None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source_root / "src"))
    from skill3d.evaluation.fitset_v11 import (
        SystemVariantRunner, build_fit_skill_bundle, load_fitset_panel, load_fit_skill_bundle,
    )
    from skill3d.evaluation.derived_published_fitset_v11 import DerivedPublishedFitSetInputProvider
    from skill3d.evolution.fitset_revision_v11 import load_active_parent_slots
    from skill3d.infra.experiment_stages_v11 import fitset_cli
    from skill3d.online.config import load_config
    from skill3d.online.scene_overview import SceneOverviewStore
    from skill3d.schemas.reconstruction import METRIC_TASK_TYPES
    from skill3d.tools import REGISTRY
    cli = fitset_cli()
    raw = load_config(args.package / "online.yaml")
    cfg = cli._base_config(raw)
    panel = load_fitset_panel(args.package / "panel.json")
    selected_ids = {"2976", "3252", "3586", "3772", "3879", "4405", "4661", "5121"}
    variants = [row for row in panel["system_variants"] if row["qa_id"] in selected_ids]
    assert len(variants) == 8 and len({row["question_type"] for row in variants}) == 8
    assert len({row["scene_id"] for row in variants}) == 2
    assert not args.output.exists(), "validation output already exists"
    args.output.mkdir(parents=True)
    parents, snapshot = load_active_parent_slots(args.source_root / "skill_library")
    bundle_root = args.output / "seed_bundle"
    build_fit_skill_bundle(output_dir=bundle_root, bundle_id="metric-preparation-validation",
                           candidates={}, active_parents=parents, source_run_identity=snapshot)
    bundle = load_fit_skill_bundle(bundle_root / "manifest.json")

    def skills_for(row):
        skill = bundle.skill_for(row["question_type"], row["evidence_state"])
        return [skill.as_spec()], {"content_sha256": skill.content_sha256}

    provider = DerivedPublishedFitSetInputProvider(
        args.package / "published_inputs.json",
        derivation_root=args.output / "derived_bindings_v4_metric", binding_cfg=cfg,
        max_scenes=2, audit_mode="reuse_source_audit")
    runner = SystemVariantRunner(
        output_root=args.output / "system", panel=panel, panel_name="ready_579",
        variants=variants, item_provider=provider, input_receipt_provider=provider.receipt,
        input_contract_identity=provider.input_contract_identity,
        binding_refs={row["variant_id"]: provider.public_entry(row)["reference"] for row in variants},
        base_cfg=cfg, phase="final_system", skills_for_variant=skills_for,
        min_free_gib=0, workers=1, published_inputs=provider,
        scene_overviews=SceneOverviewStore(args.package / "scene_overviews/manifest.json"),
    )
    runner.root.mkdir(parents=True)
    from skill3d.online import runner as online
    # The runner has no model client factories. Every verified path below is
    # the same _prepare_one used by formal preflight.
    checks = []
    for row in variants:
        print("preparing", row["qa_id"], row["question_type"], flush=True)
        prepared = runner._prepare_one(row, preflight=True)
        item = prepared["item"]
        assert prepared["state"] == "g1m1" and item.episode.ground_truth == {}
        assert len(item.pixels) == len(item.episode.frame_set.readable_frame_ids) == 32
        assert prepared["prepared"].stats["binding_derivation"]["sha256"]
        published = provider.prepare_published(row, item)
        artifact = online._prepare_m4_quality(published["artifact"], item.pixels, persist_quality=False)
        state = online.build_scene_state(artifact, question_type=row["question_type"])
        profile = state.evidence_profile
        metric_task = row["question_type"] in METRIC_TASK_TYPES
        gate_passed = state.metric_evidence_gate_result.gate_passed
        assert gate_passed is metric_task
        allowed = REGISTRY.names_for_scope(
            scope=state.question_tool_scope, available=state.available_artifacts,
            evidence_profile=profile, gate_passed=gate_passed,
            question_type=row["question_type"], local_facts={"world_frame": True})
        checks.append({
            "qa_id": row["qa_id"], "question_type": row["question_type"],
            "scene_id": row["scene_id"], "runtime_evidence_state": prepared["state"],
            "n_readable_frames": 32, "reference_answer_visible": False,
            "binding_derivation_validated": True, "exact_skill_state": "g1m1",
            "metric_task_authorized": gate_passed,
            "question_tool_scope": state.question_tool_scope, "tool_names": allowed,
        })
    result = {
        "status": "complete", "n_tasks": 8, "n_scenes": 2,
        "runtime_state_counts": dict(Counter(row["runtime_evidence_state"] for row in checks)),
        "model_requests": 0, "gpu_requests": 0, "checks": checks,
        "provider_module": sys.modules["skill3d.evaluation.published_fitset_v11"].__file__,
        "fitset_module": sys.modules["skill3d.evaluation.fitset_v11"].__file__,
        "scope": "eight-task representative preparation; not a full online experiment",
    }
    (args.output / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}), flush=True)


if __name__ == "__main__":
    main()
