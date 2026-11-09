#!/usr/bin/env python3
"""Sample public test questions and audit old candidate assets without requests."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

TASKS = (
    "object_counting", "object_abs_distance", "object_size_estimation",
    "room_size_estimation", "object_rel_distance", "object_rel_direction",
    "route_planning", "obj_appearance_order",
)
PRIVATE = {"answer", "answers", "ground_truth", "reference_answer",
           "correct_answer", "label", "labels", "score", "scores"}


def canonical(value):
    return "object_rel_direction" if value.startswith("object_rel_direction") else value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public(value):
    if isinstance(value, dict):
        if PRIVATE.intersection(value):
            raise ValueError("private reference fields in public question inputs")
        for child in value.values():
            public(child)
    elif isinstance(value, list):
        for child in value:
            public(child)


def write_once(path, value):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("existing pilot artifact identity changed")
        return
    path.write_text(text)


def sample(data_root, output_root, per_type=5, seed=137):
    data, output = Path(data_root), Path(output_root)
    questions = [json.loads(line) for line in (data / "test.jsonl").read_text().splitlines()]
    public(questions)
    if len(questions) != 700 or len({str(row["qa_id"]) for row in questions}) != 700:
        raise ValueError("original VSI1000 test split changed")
    rng = random.Random(seed)
    selected = []
    for task in TASKS:
        rows = sorted((row for row in questions if canonical(row["question_type"]) == task
                       and row["source_split"] in {"induction", "inner_validation"}),
                      key=lambda row: str(row["qa_id"]))
        selected.extend(rng.sample(rows, per_type))
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "vsi8b-pilot-sampling/1",
        "seed": seed, "questions_per_type": per_type, "n_questions": len(selected),
        "source_role": "test", "source_split_preserved": True,
        "source_split_filter": ["induction", "inner_validation"],
        "source_test_sha256": sha(data / "test.jsonl"),
        "local_published_inputs_sha256": sha(data / "published_inputs.json"),
        "sampling": "random.Random(seed); canonical TASKS order; sorted string qa_id",
        "type_counts": dict(Counter(canonical(row["question_type"]) for row in selected)),
        "questions": selected, "reference_answer_visible": False,
        "requested_execution": {
            "solver_model": "qwen3.8-flash", "enable_thinking": True,
            "description_model": "Qwen3-VL-8B-Instruct-FP8",
            "reviewer_model": "Intern-Decision-4B", "prompt_version": "P1",
            "yes_threshold": 0.77, "admission_policy": "category-only",
            "reuse_reconstruction": True, "reuse_detections": True,
            "reuse_original_candidate_masks_and_images": True,
            "retain_full_json_trajectories": True, "automatic_retry": False,
        },
    }
    write_once(output / "sampling_manifest.json", result)
    return result


def audit(published_inputs, sampling):
    manifest_path = Path(published_inputs)
    manifest = json.loads(manifest_path.read_text())
    selected = sampling if isinstance(sampling, dict) else json.loads(Path(sampling).read_text())
    sampling_hash = (hashlib.sha256((json.dumps(
        selected, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
        if isinstance(sampling, dict) else sha(sampling))
    public(selected["questions"])
    rows = []
    for question in selected["questions"]:
        qa_id = str(question["qa_id"])
        entry = manifest["entries"][qa_id]
        reference = entry["reference"]
        key = reference["key"]
        root = Path(reference["store_root"]) / "entries" / key[:2] / key
        binding_file = root / "payload/binding.json"
        store_manifest = json.loads((root / "manifest.json").read_text())
        manifest_body = {key: value for key, value in store_manifest.items()
                         if key != "manifest_sha256"}
        manifest_hash = hashlib.sha256(json.dumps(
            manifest_body, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if not manifest_hash == store_manifest["manifest_sha256"] == reference["manifest_sha256"]:
            raise ValueError("original binding manifest integrity changed")
        # Payload identities seal binding.json too, rather than trusting its name.
        payload_files = {path.relative_to(root).as_posix():
                         {"sha256": sha(path), "size": path.stat().st_size}
                         for path in (root / "payload").rglob("*") if path.is_file()}
        encoded = json.dumps(payload_files, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()
        observed_payload = {"files": payload_files,
                            "sha256": hashlib.sha256(encoded).hexdigest()}
        if observed_payload != store_manifest["payload"]:
            raise ValueError("original binding payload integrity changed")
        binding = json.loads(binding_file.read_text())
        candidates = []
        preserved = {str(cid) for obj in binding["objects"]
                     for cid in obj.get("candidate_ids", [])}
        for row in binding["stats"].get("candidate_receipts", []):
            visual = row.get("semantic_visual") or {}
            if not visual:
                continue
            request_path = Path(visual["request_file"]) if visual.get("request_file") else None
            present = bool(request_path and request_path.is_file())
            request_valid = bool(present and sha(request_path) ==
                                 visual.get("request_file_identity", {}).get("sha256"))
            image_valid = False
            if request_valid:
                import base64
                request = json.loads(request_path.read_text())
                urls = request.get("images", [])
                identities = visual["request_identity"]["images"]
                image_valid = len(urls) == len(identities) == 3 and all(
                    hashlib.sha256(base64.b64decode(url.split(",", 1)[1], validate=True)).hexdigest()
                    == identity["sha256"] for url, identity in zip(urls, identities))
            binary = visual.get("binary_review") or {}
            candidates.append({
                "candidate_id": row["candidate_id"], "category": row["category"],
                "frame_slot": row["frame_slot"], "bbox": row["bbox"],
                "old_p_yes": binary.get("p_yes"), "old_pass": binary.get("semantic_pass"),
                "request_file": str(request_path) if request_path else None,
                "request_file_identity": visual.get("request_file_identity"),
                "request_present_on_titan": present, "request_hash_valid": request_valid,
                "original_three_images_verified": image_valid,
                "original_image_identities": visual.get("request_identity", {}).get("images", []),
                "candidate_id_in_published_objects": row["candidate_id"] in preserved,
            })
        payload_paths = list((root / "payload").rglob("*"))
        private_paths = [str(path) for path in payload_paths
                         if path.is_file() and path.suffix in {".jpg", ".jpeg", ".png"}]
        rows.append({
            "qa_id": qa_id, "task": canonical(question["question_type"]),
            "scene_id": entry["scene_id"], "scene_artifact": entry["scene_artifact"],
            "scene_artifact_present": Path(entry["scene_artifact"]).is_file(),
            "frozen_input_present": Path(entry["input_root"]).is_dir(),
            "publication_root": entry["publication_root"],
            "binding_file": str(binding_file), "binding_file_sha256": sha(binding_file),
            "original_manifest_hash_verified": True,
            "original_payload_hash_verified": True,
            "payload_identity_observed": store_manifest.get("payload"),
            "payload_file_count_observed": len(payload_files),
            "payload_tree_sha256_observed": hashlib.sha256(encoded).hexdigest(),
            "published_object_count": len(binding["objects"]),
            "published_image_file_count": len(private_paths),
            "old_reviewed_candidates": len(candidates),
            "old_passed_candidates": sum(row["old_pass"] is True for row in candidates),
            "old_nonpassed_candidates": sum(row["old_pass"] is not True for row in candidates),
            "candidates_without_published_objects": sum(
                not row["candidate_id_in_published_objects"] for row in candidates),
            "verified_three_image_requests": sum(
                row["original_three_images_verified"] for row in candidates),
            "room_objects_not_required": canonical(question["question_type"]) == "room_size_estimation",
            "detection_receipts": len(binding["stats"].get("detection_receipts", [])),
            "detection_candidate_count": len(binding["stats"].get("candidate_receipts", [])),
            "candidates": candidates,
        })
    total = sum(row["old_reviewed_candidates"] for row in rows)
    reusable = sum(row["verified_three_image_requests"] for row in rows)
    return {
        "schema": "vsi8b-pilot-cache-audit/1",
        "queried_at": datetime.now(timezone.utc).isoformat(), "node": "titan",
        "published_inputs_sha256": sha(manifest_path), "sampling_sha256": sampling_hash,
        "status": "ready" if reusable == total else "blocked_missing_candidate_assets",
        "n_questions": len(rows), "n_scenes": len({row["scene_id"] for row in rows}),
        "scene_artifacts_present": sum(row["scene_artifact_present"] for row in rows),
        "frozen_inputs_present": sum(row["frozen_input_present"] for row in rows),
        "old_reviewed_candidates": total, "verified_three_image_requests": reusable,
        "missing_three_image_requests": total - reusable,
        "old_passed_candidates": sum(row["old_passed_candidates"] for row in rows),
        "old_nonpassed_candidates": sum(row["old_nonpassed_candidates"] for row in rows),
        "candidates_without_published_objects": sum(
            row["candidates_without_published_objects"] for row in rows),
        "model_requests": 0, "sam2_requests": 0, "detector_requests": 0,
        "reference_answer_visible": False, "questions": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("sample", "audit"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--published-inputs", type=Path)
    parser.add_argument("--sampling", type=Path)
    args = parser.parse_args()
    result = (sample(args.data_root, args.output_dir) if args.action == "sample"
              else audit(args.published_inputs, args.sampling))
    if args.action == "sample":
        result = {key: value for key, value in result.items() if key != "questions"}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
