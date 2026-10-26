#!/usr/bin/env python3
"""Freeze audited catalog bytes and extract unchanged, label-free scene inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

import fullqa_custody as c

_PROVIDER_FILE = (Path(__file__).resolve().parents[1] / "src" / "skill3d" /
                  "evaluation" / "published_fitset_v11.py")


def freeze(root, output):
    c.require(not output.exists(), "snapshot_destination_exists")
    provider_sha256 = c.digest(_PROVIDER_FILE.read_bytes())
    raw = (root / "deliverables/prepared_catalog.json").read_bytes()
    catalog = __import__("json").loads(raw)
    audit = c.read(root / "audits/summary.json")
    c.require(catalog["identity_hash"] == audit["identity_hash"] == c.IDENTITY
              and catalog["updated_at"] == audit["updated_at"]
              and not audit["errors"], "snapshot_audit_not_current")
    selected = {q: e for q, e in catalog["variants"]["base"].items()
                if e["split"] in {"induction", "inner_validation"}}
    c.require(bool(selected), "no_usable_published_questions")
    catalog_path = output / "source_catalog.json"
    jobs = c.read(root / "coordinator_jobs.json")["jobs"]
    scopes = {s["job_key"].split("/")[-2]:
              s["identity"]["manifest"]["scenes"][s["job_key"].split("/")[-2]]
              for s in jobs}
    scene_info = {}
    for entry in selected.values():
        scene = entry["scene_id"]
        if scene in scene_info:
            continue
        key = f"precompute-fullqa-20261024-r01/{scene}/gpu_aggregate"
        hashed = c.digest(key.encode())
        proof = c.read(root / "audits/scenes" / (hashed + ".json"))
        c.require(proof["status"] == "verified"
                  and proof["publication_receipt_sha256"] == entry["publication_receipt_sha256"]
                  and proof["qa_ids"] == scopes[scene]["qa_ids"], "scene_audit_mismatch")
        archive = root / "inputs/jobs" / (hashed + ".tar")
        scene_info[scene] = {"archive": str(archive), "key": key, "scope": scopes[scene],
                            "output_root": str(output / "inputs" / hashed)}
    output.mkdir(mode=0o700, parents=True)
    with catalog_path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    catalog_path.chmod(0o400)
    c.durable(output / "source_audit.json", audit, immutable=True)
    c.durable(output / "extraction_plan.json",
        {"schema": "harness3d-fullqa-snapshot-extraction/1", "source_catalog_file": str(catalog_path),
         "source_catalog_sha256": c.digest(raw), "entries": selected, "scenes": scene_info,
         "provider_file": str(_PROVIDER_FILE),
         "provider_sha256": provider_sha256,
         "at": time.time(), "model_requests": 0, "label_access": False}, immutable=True)
    return {"status": "snapshot_frozen", "path": str(output),
            "questions": dict(Counter(e["split"] for e in selected.values())),
            "scenes": len(scene_info), "source_catalog_sha256": c.digest(raw)}


def extract(output):
    plan = c.read(output / "extraction_plan.json")
    c.require(c.digest(Path(plan["source_catalog_file"]).read_bytes()) ==
              plan["source_catalog_sha256"], "catalog_snapshot_changed")
    c.load_pinned("skill3d.evaluation.published_fitset_v11",
                  Path(plan["provider_file"]), plan["provider_sha256"])
    from cluster_coordinator import safe_extract
    from skill3d.cluster.scene_compute import validate_scene_input
    scopes, entries, extracted = {}, {}, {}
    for scene, info in plan["scenes"].items():
        root = Path(info["output_root"])
        receipt = output / "extracted" / (c.digest(info["key"].encode()) + ".json")
        archive = Path(info["archive"])
        if receipt.exists():
            proof = c.read(receipt)
            c.require(c.digest((root / "scene.json").read_bytes()) == proof["frozen_scene_sha256"],
                      "extracted_manifest_changed")
        else:
            c.require(not root.exists(), "unfinished_extraction_requires_review")
            temp = root.with_name(root.name + ".partial")
            c.require(not temp.exists(), "partial_extraction_retained")
            temp.mkdir(parents=True, mode=0o700)
            with archive.open("rb") as stream:
                sha = __import__("hashlib").sha256()
                while block := stream.read(4 * 1024 * 1024):
                    sha.update(block)
            archive_sha = sha.hexdigest()
            # Native extractor validates every member, including symlinks,
            # traversal, duplicates and total expanded bytes.
            safe_extract(archive, temp, 20 * c.GIB)
            manifest = c.read(temp / "scene.json")
            c.require(manifest["qa_ids"] == info["scope"]["qa_ids"]
                      and manifest["split"] == info["scope"]["split"], "input_scope_mismatch")
            validate_scene_input(temp, max_questions_per_scene=len(manifest["qa_ids"]))
            os.rename(temp, root)
            proof = {"schema": "harness3d-fullqa-extracted-input/1", "scene_id": scene,
                     "input_archive_sha256": archive_sha, "input_archive_bytes": archive.stat().st_size,
                     "frozen_scene_sha256": c.digest((root / "scene.json").read_bytes()),
                     "input_root": str(root), "model_requests": 0, "label_access": False}
            c.durable(receipt, proof, immutable=True)
        extracted[scene] = proof
        scopes[scene] = {"qa_ids": info["scope"]["qa_ids"], "split": info["scope"]["split"],
                        "frozen_scene_sha256": proof["frozen_scene_sha256"], "input_root": str(root)}
    for qa, entry in plan["entries"].items():
        proof = extracted[entry["scene_id"]]
        entries[qa] = {**entry, "input_root": proof["input_root"],
                      "frozen_scene_sha256": proof["frozen_scene_sha256"],
                      "input_archive_sha256": proof["input_archive_sha256"]}
    manifest = {"schema": "v11-published-fitset-inputs/2",
                "source_catalog_file": plan["source_catalog_file"],
                "source_catalog_sha256": plan["source_catalog_sha256"],
                "selection_policy": {"learning_source_split": "induction",
                    "validation_source_split": "inner_validation", "variant": "base",
                    "selection": "all audited questions at snapshot time",
                    "final_test_visible": False, "outer_holdout_visible": False},
                "scene_scopes": scopes, "entries": entries}
    path = output / "published_inputs.json"
    c.durable(path, manifest, immutable=True)
    path.chmod(0o400)
    from skill3d.evaluation.published_fitset_v11 import PublishedFitSetInputProvider
    provider = PublishedFitSetInputProvider(path, max_scenes=1)
    checked = 0
    for scene in scopes:
        qa, entry = next((q, e) for q, e in entries.items() if e["scene_id"] == scene)
        # Validate actual 32-frame inputs plus the original sealed binding.
        root = Path(entry["input_root"])
        source = c.read(root / "scene.json")
        question = next(v for v in source["questions"] if v["episode"]["qa_id"] == qa)
        from skill3d.routing.task_classifier import canonical_task
        variant = {"variant_id": f"snapshot/{qa}", "qa_id": qa, "scene_id": scene,
                   "split": entry["split"], "source_split": entry["split"],
                   "question_type": canonical_task(question["episode"]["question_type"]),
                   "artifact_ref": entry["reference"]["scene_artifact_ref"]}
        item = provider(variant)
        provider.prepare_published(variant, item)
        checked += 1
    result = {"schema": "harness3d-fullqa-snapshot-preflight/1", "status": "verified",
              "at": time.time(), "manifest": str(path), "manifest_sha256": c.digest(path.read_bytes()),
              "questions": dict(Counter(e["split"] for e in entries.values())),
              "scenes": len(scopes), "normal_loader_checked_scenes": checked,
              "model_requests": 0, "solver_ran": False, "scoring_ran": False,
              "label_access": False, "source_catalog_sha256": plan["source_catalog_sha256"]}
    preflight = output / "preflight.json"
    if preflight.exists():
        # Retain the first timestamp only after all live identity checks succeed.
        result["at"] = c.read(preflight)["at"]
    c.durable(preflight, result, immutable=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    result = extract(args.output) if args.extract else freeze(args.root, args.output)
    print(__import__("json").dumps(result), flush=True)


if __name__ == "__main__":
    main()
