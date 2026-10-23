#!/usr/bin/env python3
"""Attach immutable old binding entries to new scene packages by exact identity."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from skill3d.infra.m5_identity import file_identity
from cluster_coordinator import atomic_json


def verified_json(path, sha=None):
    if sha and file_identity(path)["sha256"] != sha:
        raise ValueError("source JSON identity changed")
    return json.loads(Path(path).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.root.resolve() == args.frozen_root.resolve():
        raise ValueError("reuse attachments require a new input root")
    catalog = verified_json(args.catalog)
    if catalog["status"] != "verified":
        raise ValueError("reuse source catalog is not verified")
    validation_cache = {}
    n_reused = 0
    for path in sorted((args.frozen_root / "scenes").glob("*.json")):
        scene = verified_json(path)
        package = {"schema_version": "v11-generation-reuse-package/1.0", "bindings": {}}
        for qa_id in scene["qa_ids"]:
            for name in ("base", "metric"):
                entry = catalog["variants"][name].get(qa_id)
                if entry is None:
                    continue
                if entry["scene_id"] != scene["scene_id"] or entry["qa_id"] != qa_id:
                    raise ValueError("catalog QA/scene changed")
                identity = verified_json(entry["scene_input_identity_file"],
                                         entry["scene_input_identity_file_sha256"])
                migration_file = entry["migration_receipt_file"]
                cache_key = (migration_file, entry["migration_receipt_file_sha256"])
                if cache_key not in validation_cache:
                    validation_cache[cache_key] = verified_json(migration_file, cache_key[1])
                migration = validation_cache[cache_key]["variants"][name]["migration_receipts"][qa_id]
                reference = copy.deepcopy(entry["reference"])
                key = reference["key"]
                entry_root = Path(reference["store_root"]) / "entries" / key[:2] / key
                prefix = f"data/generation_reuse/{name}"
                for source in sorted(entry_root.rglob("*")):
                    if source.is_symlink():
                        raise ValueError("source binding contains a symlink")
                    if not source.is_file():
                        continue
                    relative = prefix + "/entries/" + key[:2] + "/" + key + "/" + source.relative_to(entry_root).as_posix()
                    logical = str(source)
                    scene["files"][logical] = relative
                    scene["file_identities"][relative] = file_identity(source)
                    scene["local_sources"][relative] = logical
                artifact = scene["files"].get(identity["artifact_path"])
                if artifact is None:
                    raise ValueError("reuse source raw artifact is not in the frozen geometry closure")
                dependencies = {
                    field: scene["files"][record["path"]]
                    for field, record in identity["files"].items()
                }
                package["bindings"].setdefault(qa_id, {})[name] = {
                    "source_reference": reference, "source_input_identity": identity,
                    "source_migration_receipt": migration, "store_root_ref": prefix,
                    "scene_artifact_ref": artifact, "dependency_refs": dependencies,
                }
                n_reused += 1
        if package["bindings"]:
            package_file = args.root / "reuse_manifests" / (scene["scene_id"] + ".json")
            atomic_json(package_file, package)
            relative = "data/generation_reuse.json"
            scene["files"][str(package_file)] = relative
            scene["file_identities"][relative] = file_identity(package_file)
            scene["local_sources"][relative] = str(package_file)
            scene["generation_reuse_package_ref"] = relative
        target = args.root / "scenes" / path.name
        if target.exists():
            if verified_json(target) != scene:
                raise ValueError("existing attached scene has another identity")
        else:
            atomic_json(target, scene)
        print(json.dumps({"stage": "reuse_export", "scene_id": scene["scene_id"],
                          "source_bindings": sum(map(len, package["bindings"].values()))}), flush=True)
    atomic_json(args.root / "reuse_export_receipt.json", {
        "schema": "harness3d-generation-reuse-export/1",
        "source_catalog": str(args.catalog), "source_catalog_identity": file_identity(args.catalog),
        "exported_source_bindings": n_reused, "model_requests": 0, "label_access": False,
    })


if __name__ == "__main__":
    main()
