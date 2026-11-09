#!/usr/bin/env python3
"""Operate a new panel with the exact measured solver source and policy."""
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "source_frozen/scripts"), str(ROOT / "source_frozen/src")]
import pilot40 as pilot

ORIGINAL_CONFIG = pilot.base_config
ORIGINAL_INITIALIZE = pilot.initialize


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_sources():
    spec = read(ROOT / "experiment_spec.json")
    assert sha(ROOT / "sampling_manifest.json") == spec["sampling_sha256"]
    assert sha(ROOT / "source_manifest.json") == spec["source_manifest_sha256"]
    assert sha(Path(__file__)) == spec["execution_adapter_sha256"]
    for row in read(ROOT / "source_manifest.json")["files"]:
        path = ROOT / "source_frozen" / row["file"]
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]


def base_config():
    cfg, snapshot = ORIGINAL_CONFIG()
    cfg.allow_final_test = True
    spec = read(ROOT / "experiment_spec.json")
    cfg.semantic_review_config = copy.deepcopy(cfg.semantic_review_config)
    cfg.semantic_review_config["target_description"] = copy.deepcopy(spec["description"])
    assert cfg.seed == 137 and cfg.temperature == 0 and cfg.max_tokens == 8192
    assert cfg.online_model == spec["model"] == "qwen3.8-flash"
    assert cfg.online_enable_thinking and cfg.online_timeout_s == 900
    assert cfg.max_solver_rounds == 6 and cfg.evaluation_execution_role == "validation"
    assert cfg.isolated_choice_visual_fallback and cfg.eval_visual_fallback
    assert snapshot == spec["skill_snapshot"] and len(cfg.skills) == 24
    return cfg, snapshot


def initialize(args):
    verify_sources()
    spec = read(ROOT / "experiment_spec.json")
    assert args.expected_questions == spec["n_questions"] == 80
    assert sha(args.sampling) == spec["sampling_sha256"]
    assert sha(args.published_inputs) == spec["published_inputs_sha256"]
    result = ORIGINAL_INITIALIZE(args)
    identity = read(result[0] / "runtime_identity.json")
    assert identity["sam2_identity_sha256"] == spec["sam2_identity_sha256"]
    assert identity["reviewer_identity_sha256"] == spec["reviewer_identity_sha256"]
    assert identity["description"] == spec["description"]
    return result


def preflight():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sampling", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--published-inputs", required=True, type=Path)
    p.add_argument("--expected-questions", required=True, type=int)
    args = p.parse_args()
    root, sampling, cfg, provider, overview, predictor = initialize(args)
    verified, scopes = [], {}
    for q in sorted(sampling["questions"], key=lambda q: (
            q["dataset"], q["scene_name"], int(q["qa_id"]))):
        qa = str(q["qa_id"])
        entry = provider.entries[qa]
        v = pilot.variant(q, entry)
        item = provider(v)
        ep = item.episode
        assert ep.qa_id == qa and ep.question == q["question"]
        assert (ep.options or []) == (q.get("options") or [])
        assert ep.dataset == q["dataset"] and ep.scene_name == q["scene_name"]
        assert ep.split == q["source_split"] and ep.ground_truth == ""
        assert pilot.canonical_task(ep.question_type) == v["question_type"]
        transport = provider.prepare_published(v, item)
        overview.descriptor(entry["scene_id"], entry=entry)
        prepared = transport["prepared"]
        stats = prepared.stats
        ids = (stats.get("review_selection") or {}).get("selected_candidate_ids", [])
        candidates = stats.get("candidate_receipts") or []
        assert len(ids) == len(set(ids))
        if (stats.get("object_parse") or {}).get("status") == "unavailable":
            assert pilot.reuse_unavailable_source_parser(prepared, entry["reference"])
        scopes[qa] = {
            "scene_id": entry["scene_id"], "review_candidate_ids": ids,
            "description_binary_request_cap": 2 * len(ids),
            "sam2_frame_batch_cap": len({c["frame_slot"] for c in candidates}),
            "source_parser_status": (stats.get("object_parse") or {}).get("status"),
            "source_generation_sha256": stats["generation"]["sha256"],
        }
        verified.append({
            "qa_id": qa, "scene_id": entry["scene_id"], "source_split": entry["split"],
            "frame_set_hash": ep.frame_set.frame_set_hash,
            "pixel_sha256": ep.frame_set.pixel_sha256,
            "source_reference_sha256": pilot.content_sha256(entry["reference"]),
            "binding_generation_sha256": stats["generation"]["sha256"],
            "transport_receipt_sha256": pilot.content_sha256(transport["transport_receipt"]),
            "reference_answer_visible": False,
        })
        provider.cache.clear()
        provider._publication_files.clear()
        provider._scene_checks.clear()
        provider._scene_transport_scopes.clear()
        if len(verified) % 10 == 0:
            print(json.dumps({"verified": len(verified), "total": 80,
                              "model_requests": 0, "gpu_requests": 0}), flush=True)
    pilot.write(root / "preflight_verified.json", {
        "schema": "vsi-published-panel-zero-request-preflight/1",
        "questions_verified": 80, "scenes_verified": len({x["scene_id"] for x in verified}),
        "sampling_sha256": sha(args.sampling), "model_requests": 0, "gpu_requests": 0,
        "reference_answers_read": False, "reference_answer_visible": False,
        "questions": verified,
    })
    pilot.write(ROOT / "preparation_scope.json", {
        "schema": "vsi-type80-frozen-preparation-scope/1", "questions": scopes,
        "preparation_request_cap": sum(x["description_binary_request_cap"] for x in scopes.values()),
        "sam2_frame_batch_cap": sum(x["sam2_frame_batch_cap"] for x in scopes.values()),
        "sampling_sha256": sha(args.sampling),
        "new_parser_requests": 0, "new_detector_requests": 0,
        "new_reconstruction_requests": 0, "automatic_retry": False,
        "reference_answers_read": False, "reference_answer_visible": False,
    })


def main():
    phase = sys.argv.pop(1)
    pilot.base_config = base_config
    pilot.initialize = initialize
    if phase == "preflight":
        preflight()
    elif phase == "execute":
        assert read(ROOT / "harness/preflight_verified.json")["questions_verified"] == 80
        import harness_runtime_identity_guard
        harness_runtime_identity_guard.install()
        import pilot40_parallel
        pilot40_parallel.main()
    else:
        raise ValueError("unknown phase")


if __name__ == "__main__":
    main()
