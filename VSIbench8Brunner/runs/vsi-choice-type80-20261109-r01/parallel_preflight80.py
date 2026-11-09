#!/usr/bin/env python3
"""Parallel label-free input checks, with persistent per-question receipts."""
import argparse
from collections import OrderedDict
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "source_frozen/scripts"), str(ROOT / "source_frozen/src")]
import type80_adapter as adapter
import pilot40 as pilot

PROVIDER_STATE = None
PROVIDER_CLASS = None
OVERVIEW = None
QUESTIONS = None
ARM = ROOT / "harness"


def worker(task):
    scene_id, ids = task
    provider = PROVIDER_CLASS.__new__(PROVIDER_CLASS)
    provider.__setstate__(copy.deepcopy(PROVIDER_STATE))
    results = []
    for qa in ids:
        q = QUESTIONS[qa]
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
        OVERVIEW.descriptor(entry["scene_id"], entry=entry)
        prepared = transport["prepared"]
        stats = prepared.stats
        selected = (stats.get("review_selection") or {}).get("selected_candidate_ids", [])
        candidates = stats.get("candidate_receipts") or []
        assert len(selected) == len(set(selected))
        if (stats.get("object_parse") or {}).get("status") == "unavailable":
            assert pilot.reuse_unavailable_source_parser(prepared, entry["reference"])
        receipt = {
            "qa_id": qa, "scene_id": entry["scene_id"], "source_split": entry["split"],
            "frame_set_hash": ep.frame_set.frame_set_hash,
            "pixel_sha256": ep.frame_set.pixel_sha256,
            "source_reference_sha256": pilot.content_sha256(entry["reference"]),
            "binding_generation_sha256": stats["generation"]["sha256"],
            "transport_receipt_sha256": pilot.content_sha256(transport["transport_receipt"]),
            "reference_answer_visible": False,
        }
        scope = {
            "scene_id": scene_id, "review_candidate_ids": selected,
            "description_binary_request_cap": 2 * len(selected),
            "sam2_frame_batch_cap": len({c["frame_slot"] for c in candidates}),
            "source_parser_status": (stats.get("object_parse") or {}).get("status"),
            "source_generation_sha256": stats["generation"]["sha256"],
        }
        result = {"receipt": receipt, "scope": scope, "model_requests": 0, "gpu_requests": 0}
        pilot.write(ARM / "parallel_preflight_questions" / (qa + ".json"), result)
        results.append(result)
    return results


def main():
    global PROVIDER_CLASS, PROVIDER_STATE, OVERVIEW, QUESTIONS
    p = argparse.ArgumentParser()
    p.add_argument("--sampling", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--published-inputs", required=True, type=Path)
    p.add_argument("--expected-questions", required=True, type=int)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    pilot.base_config = adapter.base_config
    pilot.initialize = adapter.initialize
    root, sampling, cfg, provider, overview, predictor = adapter.initialize(args)
    assert root == ARM
    assert not list((root / "solver_requests").glob("*/*.json"))
    assert not (root / "request_budget.json").exists()
    PROVIDER_CLASS = type(provider)
    PROVIDER_STATE = provider.__getstate__()
    OVERVIEW = overview
    QUESTIONS = {str(q["qa_id"]): q for q in sampling["questions"]}
    scenes = OrderedDict()
    for qa in QUESTIONS:
        scenes.setdefault(provider.entries[qa]["scene_id"], []).append(qa)
    started = time.time()
    results = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for group in pool.imap_unordered(worker, list(scenes.items()), chunksize=1):
            results.extend(group)
            progress = {
                "phase": "zero_request_parallel_preflight", "verified": len(results),
                "total": 80, "workers": args.workers, "elapsed_s": time.time() - started,
                "model_requests": 0, "gpu_requests": 0,
            }
            pilot.write(root / "parallel_preflight_progress.json", progress)
            print(json.dumps(progress), flush=True)
    by_qa = {x["receipt"]["qa_id"]: x for x in results}
    assert set(by_qa) == set(QUESTIONS) and len(results) == 80
    verified = [by_qa[qa]["receipt"] for qa in sorted(by_qa, key=int)]
    scopes = {qa: by_qa[qa]["scope"] for qa in sorted(by_qa, key=int)}
    pilot.write(root / "preflight_verified.json", {
        "schema": "vsi-published-panel-zero-request-preflight/1",
        "questions_verified": 80, "scenes_verified": len(scenes),
        "sampling_sha256": adapter.sha(args.sampling),
        "model_requests": 0, "gpu_requests": 0, "reference_answers_read": False,
        "reference_answer_visible": False, "questions": verified,
        "parallel_preflight_workers": args.workers,
        "parallel_preflight_source_sha256": adapter.sha(Path(__file__)),
    })
    pilot.write(ROOT / "preparation_scope.json", {
        "schema": "vsi-type80-frozen-preparation-scope/1", "questions": scopes,
        "preparation_request_cap": sum(x["description_binary_request_cap"] for x in scopes.values()),
        "sam2_frame_batch_cap": sum(x["sam2_frame_batch_cap"] for x in scopes.values()),
        "sampling_sha256": adapter.sha(args.sampling),
        "new_parser_requests": 0, "new_detector_requests": 0,
        "new_reconstruction_requests": 0, "automatic_retry": False,
        "reference_answers_read": False, "reference_answer_visible": False,
    })


if __name__ == "__main__":
    main()
