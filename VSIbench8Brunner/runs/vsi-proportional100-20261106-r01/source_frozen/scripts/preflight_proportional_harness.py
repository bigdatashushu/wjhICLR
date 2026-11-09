#!/usr/bin/env python3
"""Validate all selected published inputs without issuing inference requests."""
import argparse
import hashlib
import json
from pathlib import Path

import pilot40 as pilot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--published-inputs", required=True, type=Path)
    parser.add_argument("--expected-questions", required=True, type=int)
    args = parser.parse_args()
    root, sampling, cfg, provider, overview, predictor = pilot.initialize(args)
    verified = []
    rows = sorted(sampling["questions"], key=lambda row: (
        row["dataset"], row["scene_name"], int(row["qa_id"])))
    for row in rows:
        qa = str(row["qa_id"])
        entry = provider.entries[qa]
        variant = pilot.variant(row, entry)
        item = provider(variant)
        episode = item.episode
        if (episode.qa_id != qa or episode.question != row["question"]
                or (episode.options or []) != (row.get("options") or [])
                or episode.dataset != row["dataset"]
                or episode.scene_name != row["scene_name"]
                or episode.split != row["source_split"]
                or episode.ground_truth != ""
                or pilot.canonical_task(episode.question_type) != variant["question_type"]):
            raise ValueError("selected public question differs from published episode")
        transport = provider.prepare_published(variant, item)
        overview.descriptor(entry["scene_id"], entry=entry)
        prepared = transport["prepared"]
        verified.append({
            "qa_id": qa, "scene_id": entry["scene_id"], "source_split": entry["split"],
            "frame_set_hash": episode.frame_set.frame_set_hash,
            "pixel_sha256": episode.frame_set.pixel_sha256,
            "source_reference_sha256": pilot.content_sha256(entry["reference"]),
            "binding_generation_sha256": prepared.stats["generation"]["sha256"],
            "transport_receipt_sha256": pilot.content_sha256(transport["transport_receipt"]),
            "reference_answer_visible": False,
        })
        if len(verified) % 10 == 0:
            print(json.dumps({"verified": len(verified), "total": len(rows),
                              "model_requests": 0, "gpu_requests": 0}), flush=True)
    pilot.write(root / "preflight_verified.json", {
        "schema": "vsi-published-panel-zero-request-preflight/1",
        "questions_verified": len(verified),
        "scenes_verified": len({row["scene_id"] for row in verified}),
        "sampling_sha256": hashlib.sha256(args.sampling.read_bytes()).hexdigest(),
        "model_requests": 0, "gpu_requests": 0, "reference_answers_read": False,
        "reference_answer_visible": False, "questions": verified,
    })


if __name__ == "__main__":
    main()
