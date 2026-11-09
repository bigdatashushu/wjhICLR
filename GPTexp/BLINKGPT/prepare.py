"""Build public-only jobs and copy independent method sources without changing originals."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

from .common import COUNTS, MODEL, PUBLIC_KEYS, ROOT, digest, file_digest, read, source_files, write


def copy_tree(source, target, suffixes=(".py", ".txt", ".json")):
    for p in sorted(source.rglob("*")):
        if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts:
            out = target / p.relative_to(source)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and file_digest(out) != file_digest(p):
                raise ValueError("vendor_source_already_exists_with_different_identity")
            if not out.exists():
                shutil.copyfile(p, out)


def build(repo, output, remote):
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError("use_existing_frozen_manifest_or_new_run_directory")
    output.mkdir(parents=True, exist_ok=True)
    mv_source = repo / "GPTexp/frozen/blink-multiview/code/source"
    sr_origins = [repo.parent / f"outputs/BLINK_SR_v2_{name}_20261105_r01/raw"
                  for name in ("retest20", "remaining80")]
    copy_tree(mv_source / "src", ROOT / "vendor/mv/src")
    copy_tree(sr_origins[1] / "source/src", ROOT / "vendor/sr/src")
    # Only SR integration code is needed; omit historical execution/deployment scripts.
    for p in (sr_origins[1] / "source/scripts/BLINKRunner").glob("sr*"):
        if p.suffix in (".py", ".json", ".txt") and p.name not in {"sr_retest.py"}:
            copy_tree_file(p, ROOT / "vendor/sr/scripts/BLINKRunner" / p.name)
    copy_tree(repo / "xcodeapi", ROOT / "vendor/xcodeapi", suffixes=(".py",))
    jobs, public_rows = [], {}
    remote_images = remote + "/public/images"
    for dataset, subroot in (("mv", repo / "BLINK"), ("sr", repo / "BLINK/spatial_relation")):
        metadata = read(subroot / "data/manifest.json")
        if metadata["resolved_revision"] != "a3666eb249237ba3d5eca8db21176cc47967e040":
            raise ValueError("official_dataset_revision_changed")
        if file_digest(subroot / "data/inputs.jsonl") != metadata["input_index_sha256"]:
            raise ValueError("official_public_index_changed")
        rows = [json.loads(line) for line in (subroot / "data/inputs.jsonl").read_text().splitlines()
                if line.strip()]
        rows = sorted((row for row in rows if row["split"] == "val"), key=lambda r: r["row_index"])
        if len(rows) != COUNTS[dataset + ":official"]:
            raise ValueError("official_validation_denominator_changed")
        for row in rows:
            if set(row) != PUBLIC_KEYS or row["reference_answer_visible"] is not False:
                raise ValueError("nonpublic_dataset_field")
            original = deepcopy(row)
            for image in row["images"]:
                path = subroot / image["path"]
                if file_digest(path) != image["sha256"]:
                    raise ValueError("original_image_changed")
                target = output / "public/images" / (image["sha256"] + path.suffix)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(path, target)
                image["path"] = remote_images + "/" + target.name
            public_rows[dataset, row["sample_id"]] = row
            jobs.append({"dataset": dataset, "arm": "official", "sample_id": row["sample_id"],
                         "official_id": row["sample_id"], "spec": {"row": row},
                         "images": row["images"], "logical_request_cap": 1})
        write(output / "public" / f"{dataset}-metadata.json", {
            "repository": metadata["repository"], "revision": metadata["resolved_revision"],
            "input_index_sha256": metadata["input_index_sha256"],
            "reference_index_sha256": metadata["reference_index_sha256"],
            "split": "val", "count": len(rows)}, once=True)
    old = read(repo / "GPTexp/frozen/blink-multiview/manifest.json")
    for job in old["jobs"]:
        worker = job["worker"]
        spec = {"problem": deepcopy(worker["problem"]),
                "solver": {**worker["solver"], "expected_model": ""},
                "official_prompt": worker["problem"]["question"] + "\n(A) left\n(B) right"}
        images = job["identity"]["images"]
        matches = [row for (d, _), row in public_rows.items() if d == "mv"
                   and [im["sha256"] for im in row["images"]] == [im["sha256"] for im in images]
                   and row["benchmark_prompt"] == spec["official_prompt"]]
        if len(matches) != 1:
            raise ValueError("legacy_mv_cannot_map_to_official_validation")
        arm = "direct" if worker["mode"] == "direct" else "harness"
        if arm == "harness":
            spec.update(prepared_root=worker["prepared_root"], artifact_sha256=worker["artifact_sha256"],
                        skill_text=worker["skill_text"])
        jobs.append({"dataset": "mv", "arm": arm, "sample_id": spec["problem"]["sample_id"],
                     "official_id": matches[0]["sample_id"], "spec": spec, "images": images,
                     "logical_request_cap": 1 if arm == "direct" else 4})
    cache_jobs = {}
    for origin in sr_origins:
        manifest = read(origin / "manifest.json")
        for job in manifest["jobs"]:
            if job["spec"]["arm"] == "skill_tools":
                cache_jobs[job["spec"]["problem"]["sample_id"]] = job["spec"]
    ids = read(repo / "GPTexp/frozen/blink-sr/split.json")["internal_test"]
    if set(ids) != set(cache_jobs) or len(ids) != 100:
        raise ValueError("sr_v2_cohort_changed")
    for sample in ids:
        old_spec = cache_jobs[sample]
        row = public_rows["sr", sample]
        if (old_spec["official_prompt"] != row["benchmark_prompt"]
                or old_spec["problem"]["question"] != row["question"]):
            raise ValueError("sr_official_public_binding_changed")
        images = [{**im, "view_index": i} for i, im in enumerate(row["images"])]
        images = [{k: im[k] for k in ("view_index", "sha256", "width", "height", "mime_type")}
                  for im in images]
        for arm in ("direct", "harness"):
            spec = {k: deepcopy(old_spec[k]) for k in ("problem", "solver", "official_prompt")}
            spec["solver"]["expected_model"] = ""
            if arm == "harness":
                spec.update({k: old_spec[k] for k in ("prepared_root", "artifact_sha256", "skill_text")})
            jobs.append({"dataset": "sr", "arm": arm, "sample_id": sample, "official_id": sample,
                         "spec": spec, "images": images,
                         "logical_request_cap": 1 if arm == "direct" else 3})
    # Interleave all six arms so a partial snapshot covers each comparison.
    groups = {key: [j for j in jobs if j["dataset"] + ":" + j["arm"] == key] for key in COUNTS}
    jobs = [groups[key][i] for i in range(max(COUNTS.values()))
            for key in COUNTS if i < len(groups[key])]
    for job in jobs:
        job["job_id"] = digest(job)[:24]
    if Counter(j["dataset"] + ":" + j["arm"] for j in jobs) != COUNTS:
        raise ValueError("job_denominator_changed")
    config = {"model": MODEL, "reasoning_effort": "high", "seed": 137, "temperature": 0,
              "max_output_tokens": 8192, "timeout_seconds": 36000, "attempts_per_request": 12,
              "backoff_min_seconds": 60, "backoff_max_seconds": 3600,
              "workers": 10, "automatic_retry": True,
              "response_metadata_policy": "request_context_with_reported_model_check"}
    body = {"schema": "blink-gpt/1", "counts": COUNTS, "jobs": jobs, "config": config,
            "logical_request_cap": sum(j["logical_request_cap"] for j in jobs),
            "source_files": source_files(), "reference_answers_visible": False,
            "official_prompt_suffix": "", "mv_legacy_split_crosswalk": "official_val_image_sha256",
            "sr_method_version": "small-image-blink-sr/2",
            "sr_prior_test_informed_prompt_development": True,
            "origin_manifest_sha256": {str(p.relative_to(repo.parent)): file_digest(p)
                for p in [repo / "GPTexp/frozen/blink-multiview/manifest.json",
                          *(o / "manifest.json" for o in sr_origins)]}}
    manifest = {**body, "run_identity": digest(body)}
    write(output / "manifest.json", manifest, once=True)
    return manifest


def copy_tree_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and file_digest(source) != file_digest(target):
        raise ValueError("vendor_file_identity_changed")
    if not target.exists():
        shutil.copyfile(source, target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--refresh-source-before-deploy", action="store_true")
    args = p.parse_args()
    if args.refresh_source_before_deploy:
        output = Path(args.output)
        if any((output / name).exists() for name in ("launch.json", "snapshot.json")):
            raise ValueError("cannot_refreeze_deployed_run")
        manifest = read(output / "manifest.json")
        manifest["source_files"] = source_files()
        manifest.pop("run_identity")
        manifest["run_identity"] = digest(manifest)
        write(output / "manifest.json", manifest)
    else:
        manifest = build(ROOT.parents[1], args.output, args.remote_root)
    print(json.dumps({"jobs": len(manifest["jobs"]), "counts": manifest["counts"],
                      "logical_request_cap": manifest["logical_request_cap"],
                      "run_identity": manifest["run_identity"], "model_requests": 0}))


if __name__ == "__main__":
    main()
