#!/usr/bin/env python3
"""Inspect a completed immutable input snapshot without any model request."""
import argparse
from collections import Counter
import json
from pathlib import Path

import fullqa_custody as c


def open_provider(snapshot):
    """Load the snapshot's pinned provider over the original frozen dependencies."""
    import sys
    snapshot = Path(snapshot)
    plan = c.read(snapshot / "extraction_plan.json")
    root = snapshot.parents[1] / "precompute-fullqa-20261024-r01"
    sys.path[:0] = [str(root / "source_frozen/src"), str(root / "source_frozen/scripts")]
    module = c.load_pinned("skill3d.evaluation.published_fitset_v11",
                          plan["provider_file"], plan["provider_sha256"])
    return module.PublishedFitSetInputProvider(snapshot / "published_inputs.json", max_scenes=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--qa-id")
    args = parser.parse_args()
    proof = c.read(args.snapshot / "preflight.json")
    path = args.snapshot / "published_inputs.json"
    c.require(proof["status"] == "verified"
              and proof["manifest_sha256"] == c.digest(path.read_bytes())
              and proof["model_requests"] == 0, "snapshot_not_preflight_verified")
    manifest = c.read(path)
    c.require(manifest["source_catalog_sha256"] ==
              c.digest(Path(manifest["source_catalog_file"]).read_bytes()), "source_catalog_changed")
    counts = Counter(entry["split"] for entry in manifest["entries"].values())
    c.require(set(counts) <= {"induction", "inner_validation"}, "reserved_split_present")
    result = {"status": "verified", "snapshot": str(args.snapshot),
                      "manifest_sha256": proof["manifest_sha256"],
                      "questions": dict(counts), "scenes": len(manifest["scene_scopes"]),
                      "normal_loader_checked_scenes": proof["normal_loader_checked_scenes"],
                      "model_requests": 0}
    if args.qa_id:
        provider = open_provider(args.snapshot)
        entry = provider.entries[args.qa_id]
        source = c.read(Path(entry["input_root"]) / "scene.json")
        question = next(row for row in source["questions"]
                        if row["episode"]["qa_id"] == args.qa_id)
        from skill3d.routing.task_classifier import canonical_task
        variant = {"variant_id": "snapshot/" + args.qa_id, "qa_id": args.qa_id,
                   "scene_id": entry["scene_id"], "split": entry["split"],
                   "source_split": entry["split"], "artifact_ref": entry["reference"]["scene_artifact_ref"],
                   "question_type": canonical_task(question["episode"]["question_type"])}
        item = provider(variant)
        transport = provider.prepare_published(variant, item)["transport_receipt"]
        result["replayed"] = {k: transport[k] for k in
            ("qa_id", "scene_id", "split", "n_images", "receipt_sha256", "reference_answer_visible")}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
