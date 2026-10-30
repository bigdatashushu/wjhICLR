"""Reuse the existing v11 experiment engine without changing the solver.

Labels are opened only by offline scoring or the original learning framework.
The baseline uses the published chronological 32 RGB frames, with a single
governed request per QA. It is labelled prepared-32, not an official raw-video
sampler. Cluster-specific transport/audit can be supplied as a command profile.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import redirect_stdout
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import sys

from skill3d.infra.experiment_campaign import (
    CampaignError, atomic, digest, file_hash, read_json, require, unknown_requests,
    validate_run,
)

SOURCE = Path(__file__).resolve().parents[3]


def fitset_cli():
    name = "_unified_experiment_fitset_cli"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SOURCE / "scripts/run_fitset_v11.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def system_argv(manifest, stage, *, preflight=False):
    profile = manifest["profile"]
    inputs, runtime = profile["inputs"], profile["runtime"]
    root = Path(manifest["run_root"])
    learning = stage == "learning"
    argv = [
        "run-system", "--panel", inputs["panel"], "--metadata", inputs["metadata"],
        "--published-inputs", inputs["published_inputs"],
        "--experiment-split-manifest", inputs["qa_split"],
        "--config", inputs["online_config"],
        "--gate-mode", "qa_disjoint" if learning else "scene_disjoint",
        "--phase", "accumulation" if learning else "final_system",
        "--output-dir", str(root / "stages" / stage / "system"),
        "--library-root", inputs["seed_library"],
        "--scheduler", "episode_pool", "--request-governor-root", str(root / "requests"),
    ]
    options = {
        "seed", "panel_name", "workers", "skill_owners", "reviser_concurrency",
        "episode_concurrency", "request_concurrency", "requests_per_second", "max_requests",
        "min_free_gib", "scene_cache_size", "perception_endpoint", "perception_capacity",
        "learning_input_mode", "scheduling_mode", "max_revision_requests_per_slot",
        "max_gate_candidates_per_slot", "max_revision_attempts", "revision_max_tokens",
        "gate_size", "diagnosis_max_cases", "diagnosis_max_input_chars", "diagnosis_max_images",
        "diagnosis_max_output_tokens", "analysis_concurrency", "offline_frame_storage",
    }
    for name in sorted(options):
        if name in runtime and runtime[name] != "":
            value = 1 if name == "reviser_concurrency" and not learning else runtime[name]
            argv.extend(["--" + name.replace("_", "-"), str(value)])
    argv.append("--overlap-owner-slots" if runtime.get("overlap_owner_slots", False)
                else "--no-overlap-owner-slots")
    argv.append("--prefetch-scenes" if runtime.get("prefetch_scenes", False)
                else "--no-prefetch-scenes")
    if "offline_frame_storage" not in runtime:
        argv.extend(["--offline-frame-storage", "per_variant_v1"])
    if runtime.get("revision_request_caps"):
        argv.extend(["--revision-request-caps", json.dumps(
            runtime["revision_request_caps"], sort_keys=True)])
    if not learning:
        bundle_stage = "seed" if stage == "initial_validation" else "freeze"
        argv.extend(["--execution-role", "validation", "--skill-bundle",
                     str(root / "stages" / bundle_stage / "bundle/manifest.json")])
    output = root / "stages" / stage / "system"
    if (output / "manifest.json").exists():
        argv.append("--resume")
    if preflight:
        argv.append("--preflight-only")
    return argv


def run_system(manifest, stage):
    cli = fitset_cli()
    output = Path(manifest["run_root"]) / "stages" / stage / "system"
    preflight_path = output / "preflight_summary.json"
    if not preflight_path.is_file() or read_json(preflight_path).get("status") != "complete":
        args = cli.build_parser().parse_args(system_argv(manifest, stage, preflight=True))
        with redirect_stdout(io.StringIO()):
            require(cli._run_system(args) == 0, "System preflight failed")
        preflight = read_json(preflight_path)
        require(preflight["status"] == "complete" and preflight["model_requests"] == 0,
                "System preflight is incomplete")
    unknown_requests(Path(manifest["run_root"]))
    args = cli.build_parser().parse_args(system_argv(manifest, stage))
    with redirect_stdout(io.StringIO()):
        require(cli._run_system(args) == 0, "System execution failed")
    summary = read_json(output / "summary.json")
    require(summary.get("status") == "complete"
            and summary["completed_variants"] == summary["selected_variants"]
            and summary["pending_variants"] == 0, "System phase is incomplete")
    if stage == "learning":
        require(summary.get("evolution_finished") is True, "Skill evolution is unfinished")
    return {"selected_variants": summary["selected_variants"]}


def seed_bundle(manifest, destination):
    from skill3d.evaluation.fitset_v11 import build_fit_skill_bundle
    from skill3d.evolution.fitset_stream_v11 import _active
    parents, snapshot = _active(Path(manifest["profile"]["inputs"]["seed_library"]))
    bundle = build_fit_skill_bundle(
        output_dir=destination / "bundle", bundle_id="seed-" + manifest["identity_sha256"][:16],
        candidates={}, active_parents=parents,
        source_run_identity={"campaign_identity": manifest["identity_sha256"],
                             "seed_snapshot": snapshot},
    )
    return {"physical_slots": bundle["physical_slot_count"], "bundle_sha256": bundle["manifest_sha256"]}


def freeze_bundle(manifest, destination):
    from skill3d.evolution.fitset_stream_v11 import freeze_fitset_stream
    result = freeze_fitset_stream(
        Path(manifest["run_root"]) / "stages/learning/system", destination / "bundle")
    return {"physical_slots": result["physical_slot_count"],
            "revised_slots": result["revised_slot_count"], "bundle_sha256": result["manifest_sha256"]}


def validated_manifest(root):
    from skill3d.evaluation.direct_qwen_v11 import content_sha256
    value = read_json(root / "manifest.json")
    body = {key: child for key, child in value.items() if key != "identity_sha256"}
    require(content_sha256(body) == value.get("identity_sha256"), "System manifest hash changed")
    summary = read_json(root / "summary.json")
    require(summary.get("status") == "complete"
            and summary["manifest_identity_sha256"] == value["identity_sha256"]
            and summary["selected_variants"] == len(value["variant_ids"])
            and summary["completed_variants"] == len(value["variant_ids"])
            and summary["pending_variants"] == 0, "System result denominator changed")
    require(value.get("reference_answer_visible_to_online_model") is False
            and value.get("online_executor_input_is_label_free") is True,
            "System input label-access contract changed")
    return value


def results(root, expected):
    rows = {}
    for path in (root / "results").glob("*.json"):
        row = read_json(path)
        key = row.get("variant_id")
        require(key in expected and key not in rows, "Unexpected or duplicate system result")
        rows[key] = row
    require(set(rows) == set(expected), "System results do not cover the full denominator")
    return rows


def scored(row, label):
    from skill3d.evaluation.fitset_v11 import score_answer
    require(row["qa_id"] == label["qa_id"] and row["scene_id"] == label["scene_id"]
            and row["question_type"] == label["question_type"], "Scoring identity changed")
    return score_answer(
        answer=row.get("answer"), task=label["question_type"], options=label["options"],
        reference=label["reference_answer"],
        available=row.get("status") == "completed" and row.get("final_state") == "answer"
        and not row.get("service_errors"),
    )


def paired_score(manifest, destination):
    from skill3d.evaluation.fitset_v11 import load_label_metadata, _variant_key, fit_slot_key
    from skill3d.evaluation.direct_qwen_v11 import content_sha256
    root = Path(manifest["run_root"])
    initial_root, final_root = [root / "stages" / name / "system"
                                for name in ("initial_validation", "final_validation")]
    initial, final = validated_manifest(initial_root), validated_manifest(final_root)
    # Skills/bundle identity changes; every experimental input and request
    # contract below must stay equal.
    keys = (
        "variant_ids", "variant_rows_sha256", "binding_refs_sha256",
        "input_contract_identity", "seed", "online_model", "online_model_revision",
        "prompt_version", "tool_docs_version", "code_identity", "solver_config_sha256",
        "request_policy", "experiment_split_sha256", "execution",
    )
    require(all(key in initial and key in final and initial[key] == final[key] for key in keys),
            "Initial/final paired experiment identity differs")
    # The original runner adds slot_key before hashing its frozen rows.
    selected = [{**row, "slot_key": fit_slot_key(row["question_type"], row["evidence_state"])}
                for row in execution_variants(manifest, validation=True)]
    variants = {row["variant_id"]: row for row in selected}
    require(set(variants) == set(final["variant_ids"])
            and content_sha256(selected) == final["variant_rows_sha256"],
            "Frozen variant rows changed")
    left, right = results(initial_root, variants), results(final_root, variants)
    labels = load_label_metadata(manifest["profile"]["inputs"]["metadata"])
    pairs = []
    for key, variant in variants.items():
        for row in (left[key], right[key]):
            require(all(row.get(field) == variant[field] for field in (
                "variant_id", "qa_id", "scene_id", "split", "question_type", "evidence_state")),
                "System result identity differs from frozen variant")
            identity = row.get("skill_identity") or {}
            require(identity.get("content_sha256") and identity.get("skill_version"),
                    "System result lacks immutable Skill identity")
        receipts = []
        for arm_root, row in ((initial_root, left[key]), (final_root, right[key])):
            receipt = read_json(arm_root / "preflight" / (_variant_key(key) + ".json"))
            require(receipt.get("model_requests") == 0, "Preflight request count changed")
            require(receipt["skill_identity"]["content_sha256"]
                    == row["skill_identity"]["content_sha256"], "Executed Skill differs from preflight")
            receipts.append(receipt)
        require(all(receipts[0][field] == receipts[1][field] for field in (
            "input_identity_sha256", "binding_key", "binding_manifest_sha256",
            "m5_parameters_sha256", "runtime_evidence_state")),
            "Initial/final evidence identity changed")
        a, b = scored(left[key], labels[variant["qa_id"]]), scored(right[key], labels[variant["qa_id"]])
        pairs.append({"variant_id": key, "qa_id": variant["qa_id"], "metric": a["metric"],
                      "initial_score": a["score"], "final_score": b["score"],
                      "delta": b["score"] - a["score"],
                      "initial_legal": a["legal_answer"], "final_legal": b["legal_answer"]})
    atomic(destination / "pairs.json", pairs)
    return {
        "n_pairs": len(pairs),
        "initial_score": sum(row["initial_score"] for row in pairs) / len(pairs),
        "final_score": sum(row["final_score"] for row in pairs) / len(pairs),
        "mean_delta": sum(row["delta"] for row in pairs) / len(pairs),
        "paired_identity_verified": True,
        "interpretation": "Frozen initial/final validation; report observed delta, without a promotion claim",
    }


def execution_variants(manifest, *, validation):
    from skill3d.evaluation.fitset_v11 import load_fitset_panel, select_panel_variants
    from skill3d.evaluation.qa_partition_v11 import load_qa_partition_manifest, select_qa_execution_variants
    profile = manifest["profile"]
    panel = load_fitset_panel(profile["inputs"]["panel"])
    variants = select_panel_variants(panel, profile["runtime"]["panel_name"])
    if validation:
        partition = load_qa_partition_manifest(
            profile["inputs"]["qa_split"], metadata_path=profile["inputs"]["metadata"])
        variants = select_qa_execution_variants(variants, partition, role="validation")
    require(bool(variants), "Baseline scope is empty")
    return variants


def baseline_variants(manifest):
    return execution_variants(manifest, validation=manifest["profile"]["baseline"]["scope"] == "validation")


def baseline_input_receipt(provider, variant, item):
    fs = item.episode.frame_set
    require(fs is not None and len(fs.pixel_sha256) == 32, "Missing baseline pixel identity")
    entry = provider.entries[variant["qa_id"]]
    # No geometry replay or binding materialization is needed for direct VQA.
    frozen_scene = Path(entry["input_root"]) / "scene.json"
    require(file_hash(frozen_scene) == entry["frozen_scene_sha256"], "Frozen RGB input changed")
    return {
        "qa_id": variant["qa_id"], "n_images": 32, "frame_set_hash": fs.frame_set_hash,
        "source_frame_indices": list(fs.source_frame_indices), "pixel_sha256": list(fs.pixel_sha256),
        "frozen_scene_sha256": entry["frozen_scene_sha256"],
        "source_catalog_sha256": provider.input_contract_identity["source_catalog_sha256"],
        "reference_answer_visible": False,
    }


def baseline32(manifest, destination):
    from skill3d.evaluation.published_fitset_v11 import PublishedFitSetInputProvider
    from skill3d.evaluation.direct_qwen_v11 import load_safe_metadata
    from skill3d.online.config import load_config, online_model_from, resolve_runtime
    from skill3d.synthesis.eval_visual_fallback import build_visual_request, parse_visual_answer
    from skill3d.synthesis.request_context import request_phase
    from skill3d.synthesis.request_governor import GovernedClient, RequestGovernor, RequestGovernorConfig
    from skill3d.routing.task_classifier import canonical_task
    profile = manifest["profile"]
    runtime = profile["runtime"]
    variants = baseline_variants(manifest)
    by_qa = {}
    for variant in variants:
        by_qa.setdefault(variant["qa_id"], variant)
    metadata = load_safe_metadata(profile["inputs"]["metadata"])
    provider = PublishedFitSetInputProvider(
        profile["inputs"]["published_inputs"], max_scenes=runtime["scene_cache_size"])
    model = online_model_from(resolve_runtime(load_config(profile["inputs"]["online_config"]),
                                             seed=runtime["seed"]))
    require(model.enabled and model.provider == "bailian" and model.model == "qwen3.8-flash"
            and model.n_frames == 32 and model.temperature == 0,
            "Baseline requires the same enabled deterministic 32-frame Qwen model")
    client_config = {"online_model": asdict(model)}
    governor = RequestGovernor(Path(manifest["run_root"]) / "requests", RequestGovernorConfig(
        runtime["max_requests"], runtime["request_concurrency"], runtime["requests_per_second"]))
    baseline_identity = {
        "protocol": "prepared-32", "campaign_identity": manifest["identity_sha256"],
        "qa_ids": sorted(by_qa), "seed": runtime["seed"], "model": model.model,
        "model_revision": model.model_revision, "max_tokens": model.max_tokens,
        "temperature": 0.0, "n_frames": 32,
        "frame_encoding": "existing build_visual_request JPEG protocol",
        "system_max_pixels": model.max_pixels,
        "pixel_resize": "no additional baseline resize",
    }
    path = destination / "manifest.json"
    if path.exists():
        require(read_json(path) == baseline_identity, "Baseline run identity changed")
    else:
        atomic(path, baseline_identity)

    def one(qa_id):
        variant = by_qa[qa_id]
        key = digest(qa_id)[:24]
        output = destination / "responses" / f"{key}.json"
        intent_path = destination / "intents" / f"{key}.json"
        if output.is_file():
            row = read_json(output)
            require(row.get("qa_id") == qa_id and row.get("campaign_identity")
                    == manifest["identity_sha256"], "Baseline response identity changed")
            require(intent_path.is_file() and row["request_sha256"]
                    == read_json(intent_path)["request_sha256"], "Baseline request hash changed")
            return
        require(not intent_path.exists(),
                "Unresolved baseline request; a second model call is refused")
        item = provider(variant)
        require(not item.episode.ground_truth and len(item.pixels) == 32,
                "Baseline inputs must have 32 frames and no reference label")
        row = metadata[qa_id]
        require(row["scene_name"] == variant["scene_id"]
                and canonical_task(row["question_type"]) == variant["question_type"],
                "Baseline metadata changed")
        _, messages, messages_hash = build_visual_request(
            question=row["question"], options=row["options"], question_type=variant["question_type"],
            frames=item.pixels, max_images=32)
        client = fitset_cli()._RevisionClientFactory(
            client_config, model.max_tokens, runtime.get("perception_endpoint", ""))()
        if hasattr(client, "transport_retries"):
            client.transport_retries = 0
        require(client.configured, "Baseline model client is unconfigured")
        request_identity = {
            "messages_sha256": messages_hash, "model_identity": client.audit_identity(),
            "seed": runtime["seed"], "max_tokens": model.max_tokens,
            "temperature": 0.0, "enable_thinking": False,
        }
        preflight = {
            "qa_id": qa_id, "request_sha256": digest(request_identity),
            "request_identity": request_identity, "reference_answer_visible": False,
            "input_receipt": baseline_input_receipt(provider, variant, item),
        }
        atomic(destination / "preflight" / f"{key}.json", preflight)
        atomic(intent_path, {"qa_id": qa_id, "request_sha256": preflight["request_sha256"],
                             "status": "started"})
        governed = GovernedClient(client, governor, category="solver", context={
            "variant_id": "baseline:" + key, "campaign_id": manifest["identity_sha256"]})
        response = {
            "qa_id": qa_id, "scene_id": variant["scene_id"], "question_type": variant["question_type"],
            "campaign_identity": manifest["identity_sha256"],
            "request_sha256": preflight["request_sha256"], "answer": None,
            "status": "invalid_answer", "final_state": "unavailable",
        }
        try:
            with request_phase("direct_qwen_visual"):
                text = governed.chat(messages, max_tokens=model.max_tokens, seed=runtime["seed"],
                                     temperature=0.0, enable_thinking=False)
        except Exception as exc:
            response.update(status="service_error", exception_type=type(exc).__name__)
            atomic(output, response)
            raise CampaignError("Baseline service stopped; no automatic retry") from None
        actual_model = (getattr(client, "last_response_identity", {}) or {}).get("model")
        require(actual_model == model.model, "Baseline response model differs from frozen model")
        response["observed_model"] = actual_model
        try:
            answer, _ = parse_visual_answer(text, question_type=variant["question_type"],
                                           options=row["options"])
        except (ValueError, TypeError, AttributeError):
            pass
        else:
            response.update(status="completed", final_state="answer", answer=answer.value)
        atomic(output, response)

    # Bounded dispatch stops adding work after the first failure. Inflight calls
    # drain before the stage exits; no implicit retries or extra QA calls.
    pending_qa = iter(sorted(by_qa))
    workers = profile["baseline"]["workers"]
    failure = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running = set()
        for _ in range(workers):
            qa_id = next(pending_qa, None)
            if qa_id is not None:
                running.add(pool.submit(one, qa_id))
        while running:
            done, running = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    future.result()
                except BaseException as exc:
                    failure = failure or exc
            if failure is None:
                for _ in done:
                    qa_id = next(pending_qa, None)
                    if qa_id is not None:
                        running.add(pool.submit(one, qa_id))
    if failure is not None:
        raise failure
    require(len(list((destination / "responses").glob("*.json"))) == len(by_qa),
            "Baseline denominator is incomplete")
    return {"n_qa": len(by_qa), "protocol": "prepared-32", "no_automatic_request_retry": True}


def baseline_score(manifest, destination):
    from skill3d.evaluation.fitset_v11 import load_label_metadata, _variant_key
    root = Path(manifest["run_root"])
    baseline = root / "stages/baseline32"
    variants = baseline_variants(manifest)
    labels = load_label_metadata(manifest["profile"]["inputs"]["metadata"])
    direct = {}
    for path in (baseline / "responses").glob("*.json"):
        row = read_json(path)
        require(row["qa_id"] not in direct, "Duplicate baseline result")
        direct[row["qa_id"]] = row
    require(set(direct) == {row["qa_id"] for row in variants}, "Baseline denominator differs")
    rows = [{"qa_id": key, **scored(row, labels[key])} for key, row in sorted(direct.items())]
    atomic(destination / "scores.json", rows)
    final_root = root / "stages/final_validation/system"
    final_manifest = validated_manifest(final_root)
    final_rows = results(final_root, final_manifest["variant_ids"])
    comparisons = []
    for key, row in final_rows.items():
        system_input = read_json(final_root / "preflight" / (_variant_key(key) + ".json"))["published_transport"]
        baseline_input = read_json(baseline / "preflight" / (digest(row["qa_id"])[:24] + ".json"))["input_receipt"]
        require(all(system_input[field] == baseline_input[field] for field in (
            "qa_id", "n_images", "frame_set_hash", "source_frame_indices", "pixel_sha256")),
            "System/baseline RGB frame identity differs")
        score = scored(row, labels[row["qa_id"]])
        baseline_result = scored(direct[row["qa_id"]], labels[row["qa_id"]])
        comparisons.append({"variant_id": key, "qa_id": row["qa_id"],
                            "system_score": score["score"], "baseline_score": baseline_result["score"],
                            "delta": score["score"] - baseline_result["score"]})
    atomic(destination / "validation_pairs.json", comparisons)
    return {"n_qa": len(rows), "n_validation_pairs": len(comparisons),
            "baseline_score": sum(row["score"] for row in rows) / len(rows),
            "protocol": "prepared-32", "full_denominator_verified": True}


def seal(manifest, destination):
    from skill3d.infra.experiment_campaign import verify_files
    root = Path(manifest["run_root"])
    state = read_json(root / "campaign_state.json")
    stages = []
    for stage in manifest["stages"]:
        if stage["name"] == "seal":
            break
        saved = state["stages"][stage["name"]]
        require(saved["status"] == "complete", "Cannot seal unfinished stages")
        verify_files(saved["artifacts"])
        stages.append({"name": stage["name"], "artifact_sha256": digest(saved["artifacts"]),
                       "artifact_count": len(saved["artifacts"])})
    unknown_requests(root)
    atomic(destination / "inventory.json", stages)
    return {"completed_stages": len(stages), "source_sha256": manifest["source_sha256"],
            "integrity_verified": True,
            "audit_scope": "local source, input, stage receipt and artifact integrity"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = Path(args.run_root).resolve()
        manifest = read_json(root / "campaign_manifest.json")
        require(manifest["run_root"] == str(root), "Stage run root changed")
        validate_run(manifest)
        require(args.stage in {row["name"] for row in manifest["stages"]}, "Unknown campaign stage")
        destination = root / "stages" / args.stage
        if (destination / "receipt.json").exists():
            require(args.resume, "Stage already complete")
            return 0
        if args.stage in {"initial_validation", "learning", "final_validation"}:
            details = run_system(manifest, args.stage)
        else:
            handlers = {"seed": seed_bundle, "freeze": freeze_bundle, "paired_score": paired_score,
                        "baseline32": baseline32, "baseline_score": baseline_score, "seal": seal}
            require(args.stage in handlers, "Unknown native stage")
            details = handlers[args.stage](manifest, destination)
        unknown_requests(root)
        atomic(destination / "receipt.json", {
            "schema": "harness3d-experiment-stage/1", "status": "complete",
            "stage": args.stage, "campaign_identity": manifest["identity_sha256"], **details,
        })
        print(f"[{args.stage}] complete", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except BaseException as exc:
        print(f"Native stage stopped: {type(exc).__name__}; no automatic retry", file=sys.stderr)
        return 2
