#!/usr/bin/env python3
"""Create a metric runtime package without changing the sealed VSI1000 selection."""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

SOURCE_NAME = "vsibench-m4-success-g1m1-1000-seed137-r01"
METRIC_NAME = "vsibench-m4-success-g1m1-1000-seed137-metric-r02"
RENDER_ARRAYS = (
    "point_map", "depth_maps", "c2w_list", "intrinsics", "point_conf", "depth_conf",
)
FRAME_FIELDS = (
    "frame_ids", "frame_set_hash", "source_frame_indices", "timestamps",
    "reconstruction_input_identity", "world_frame_status", "world_up", "handedness",
)
PRESERVED_FILES = (
    "questions.jsonl", "train.jsonl", "test.jsonl", "split.json", "online.yaml",
    "source_catalog.json", "source_m4_receipts.json",
    "host_evaluation/metadata.jsonl", "host_evaluation/labels.jsonl",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    Path(path).write_text(json.dumps(
        value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def content_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def probe(package):
    """Inspect existing Titan JSONs; no array decoding, model calls, or writes."""
    published = read(package / "published_inputs.json")
    catalog = read(package / "source_catalog.json")
    require(digest(package / "source_catalog.json") == published["source_catalog_sha256"],
            "source catalog anchor changed")
    first = {}
    for entry in published["entries"].values():
        first.setdefault(entry["scene_id"], entry["qa_id"])
    scenes = {}
    for scene_id, qa_id in sorted(first.items()):
        entries = {variant: catalog["variants"][variant][qa_id]
                   for variant in ("base", "metric")}
        artifacts = {variant: read(Path(entry["scene_artifact"]))
                     for variant, entry in entries.items()}
        scope = published["scene_scopes"][scene_id]
        frozen_path = Path(scope["input_root"]) / "scene.json"
        require(digest(frozen_path) == scope["frozen_scene_sha256"],
                "frozen scene anchor changed")
        logical = {}
        for variant, entry in entries.items():
            identity_path = Path(entry["scene_input_identity_file"])
            require(digest(identity_path) == entry["scene_input_identity_file_sha256"],
                    "logical scene input identity changed")
            logical[variant] = read(identity_path)
            require(logical[variant]["artifact_json_sha256"]
                    == digest(entry["scene_artifact"]),
                    "logical input differs from the published artifact bytes")
        identities = {}
        for name in RENDER_ARRAYS:
            for variant in ("base", "metric"):
                source = artifacts[variant][name]
                identity = logical[variant]["files"][name]
                require(identity["path"] == source, "render dependency locator changed")
                identities.setdefault(name, {})[variant] = identity["sha256"]
            require(identities[name]["base"] == identities[name]["metric"],
                    "base/metric rendering geometry differs")
        require(all(artifacts["base"][name] == artifacts["metric"][name]
                    for name in FRAME_FIELDS),
                "base/metric frame or orientation identity differs")
        require(artifacts["metric"]["scale_fusion_status"] == "success"
                and artifacts["metric"]["metric_scale"] > 0,
                "metric artifact has no successful scale fusion")
        scenes[scene_id] = {
            "base_artifact": entries["base"]["scene_artifact"],
            "base_artifact_sha256": digest(entries["base"]["scene_artifact"]),
            "metric_artifact": entries["metric"]["scene_artifact"],
            "metric_artifact_sha256": digest(entries["metric"]["scene_artifact"]),
            "frozen_scene_sha256": scope["frozen_scene_sha256"],
            "render_dependency_identities": identities,
            "frame_and_orientation_identity_preserved": True,
        }
    return {
        "schema": "vsi1000-metric-runtime-probe/1", "status": "complete",
        "source_catalog_sha256": published["source_catalog_sha256"],
        "n_scenes": len(scenes), "scenes": scenes,
        "model_requests": 0, "gpu_requests": 0, "array_readback_repeated": False,
    }


def build(source, destination, probe_path):
    require(destination.parent == source.parent and not destination.exists(),
            "new package must be a separate, unused runner cache")
    evidence = read(probe_path)
    published = read(source / "published_inputs.json")
    catalog = read(source / "source_catalog.json")
    require(evidence["status"] == "complete" and evidence["n_scenes"] == 160
            and evidence["source_catalog_sha256"] == digest(source / "source_catalog.json")
            == published["source_catalog_sha256"],
            "metric probe/catalog identity changed")
    require(set(evidence["scenes"]) == set(published["scene_scopes"]),
            "metric probe scene coverage changed")
    shutil.copytree(source, destination)
    for path in (destination, *destination.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    published["source_catalog_file"] = str(destination / "source_catalog.json")
    published["selection_policy"]["variant"] = "metric"
    for qa_id, old in list(published["entries"].items()):
        metric = copy.deepcopy(catalog["variants"]["metric"][qa_id])
        require(all(metric[name] == old[name] for name in ("qa_id", "scene_id", "split")),
                "metric catalog changed public QA identity")
        published["entries"][qa_id] = {
            **metric, **{name: old[name] for name in ("input_root", "frozen_scene_sha256")},
        }
    write(destination / "published_inputs.json", published)
    panel = read(source / "panel.json")
    for row in panel["system_variants"]:
        reference = published["entries"][row["qa_id"]]["reference"]
        row.update(evidence_state="g1m1", variant_id=f"{row['qa_id']}@g1m1",
                   artifact_ref=reference["scene_artifact_ref"],
                   binding_key=reference["key"],
                   binding_manifest_sha256=reference["manifest_sha256"])
    for row in panel["direct_qwen_rows"]:
        row["system_variant_ids"] = [f"{row['qa_id']}@g1m1"]
    slots = sorted({f"{row['question_type']}/g1m1" for row in panel["system_variants"]})
    for row in panel["panels"].values():
        row.update(skill_slots=slots, skill_slot_count=len(slots))
    panel["skill_accumulation"].update(reachable_slot_count=24, unreachable_slots=[])
    panel["experiment_id"] = METRIC_NAME
    panel["proposal_sha256"] = content_hash({
        name: value for name, value in panel.items() if name != "proposal_sha256"})
    write(destination / "panel.json", panel)
    references = read(source / "prepared_references.json")
    references["source_catalog_file"] = published["source_catalog_file"]
    write(destination / "prepared_references.json", references)
    overview = read(source / "scene_overviews/manifest.json")
    for row in overview["images"]:
        proof = evidence["scenes"][row["scene_id"]]
        require(row["scene_artifact"] == proof["base_artifact"]
                and row["scene_artifact_sha256"] == proof["base_artifact_sha256"]
                and row["frozen_scene_sha256"] == proof["frozen_scene_sha256"],
                "overview original rendering anchor differs from probe")
        row["scene_artifact"] = proof["metric_artifact"]
        row["scene_artifact_sha256"] = proof["metric_artifact_sha256"]
    overview["local_image_root"] = str(destination / "scene_overviews")
    overview["source_published_manifest_sha256"] = digest(destination / "published_inputs.json")
    overview["metric_runtime_rebinding"] = {
        "source_manifest_sha256": digest(source / "scene_overviews/manifest.json"),
        "probe_sha256": digest(probe_path), "render_dependency_bytes_preserved": True,
        "png_bytes_preserved": True, "reconstruction_rerun": False,
    }
    write(destination / "scene_overviews/manifest.json", overview)
    sampling = read(source / "sampling_manifest.json")
    sampling["prepared_inputs"].update(
        sha256=digest(destination / "published_inputs.json"), package_root=str(destination))
    sampling["runtime_variant"] = "metric"
    sampling["runtime_evidence_state"] = "g1m1"
    sampling["runtime_parent_package"] = str(source)
    write(destination / "sampling_manifest.json", sampling)
    shutil.copyfile(probe_path, destination / "eligibility/metric_runtime_probe.json")
    preserved = {name: digest(destination / name) for name in PRESERVED_FILES}
    require(all(value == digest(source / name) for name, value in preserved.items()),
            "sealed selection, labels, split, model config or catalog changed")
    require(all(digest(destination / "scene_overviews" / row["file"])
                == digest(source / "scene_overviews" / row["file"])
                == row["png_sha256"] for row in overview["images"]),
            "overview PNG bytes changed")
    result = {
        "schema": "vsi1000-metric-runtime-package/1", "status": "complete",
        "source_package": str(source), "metric_package": str(destination),
        "catalog_variant": "metric", "evidence_state": "g1m1",
        "n_questions": 1000, "n_scenes": 160, "n_skill_slots": len(slots),
        "task_counts": dict(Counter(row["question_type"] for row in panel["system_variants"])),
        "preserved_files": preserved, "png_images_preserved": 160,
        "published_inputs_sha256": digest(destination / "published_inputs.json"),
        "panel_sha256": digest(destination / "panel.json"),
        "model_requests": 0, "gpu_requests": 0,
    }
    write(destination / "metric_runtime.json", result)
    (destination / "README.md").write_text(
        "# VSI1000 metric/g1m1 runtime\n\n"
        "Uses the metric catalog and g1m1 Skill slots for all 1000 questions.\n"
        "Question selection, train300/test700 roles, 160 scenes, seed137, labels, "
        "32-frame inputs and source catalog remain unchanged from r01.\n"
        "Metric tools remain authorized only for absolute distance, object size "
        "and room size tasks. Original overview PNGs are reused after verifying "
        "the rendering dependency hashes.\n"
        "Current v4 bindings are derived without model requests into each run's "
        "own cache. See metric_runtime.json and eligibility/metric_runtime_probe.json.\n")
    for path in destination.rglob("*"):
        path.chmod(0o500 if path.is_dir() and path.name == "host_evaluation"
                   else 0o555 if path.is_dir()
                   else 0o400 if "host_evaluation" in path.parts else 0o444)
    destination.chmod(0o555)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, help="read-only source package on Titan")
    parser.add_argument("--build", type=Path, help="saved complete Titan probe JSON")
    args = parser.parse_args()
    if args.probe:
        result = probe(args.probe)
    else:
        require(args.build is not None, "--probe or --build is required")
        root = Path(__file__).resolve().parents[1]
        result = build(root / "prepared_cache" / SOURCE_NAME,
                       root / "prepared_cache" / METRIC_NAME, args.build)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
