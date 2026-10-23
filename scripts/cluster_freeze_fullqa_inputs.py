#!/usr/bin/env python3
"""Freeze all public QA inputs, without labels, model requests, or solving.

Output: root/scenes/<scene>.json, root/new_frames/<new-scene>/<slot>.png,
root/freeze_spec.json, root/progress.json, and root/progress/<partition>.json.
Existing frames, artifacts, and dependencies are not copied or rewritten.
--new-frozen-root imports raw-only scene manifests and their saved PNGs without
opening any video. Missing import scenes fail explicitly, never fall back to
decoding; completed imports can be composed with the legacy scene partition.

Scenes use harness3d-frozen-scene/1. ``files`` maps logical absolute sources
to package-relative ``data/...`` targets; ``local_sources`` maps those targets
to verified absolute local files for the tar exporter. ``file_identities`` is
keyed by package-relative target. A manifest is not itself a materialized
scene input directory: an exporter must transfer the files and install it as
scene.json. Raw video paths remain SOURCE metadata, never transfer payloads.

Existing scenes have artifact_source_path / metric_artifact_source_path and
questions[].logical_input_identity / metric_logical_input_identity. Both are
full replayed logical identities, including the compatibility empty GT field.
Original saved artifact/dependency paths remain logical transfer keys; only
local_sources names current publication files.
Public episode objects never contain that field. New scenes have no geometry
or artifact identities until the separate scene-compute step.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image

from cluster_freeze_inputs import frozen_scene
from skill3d.adapters.episode_source import EpisodeItem, _episode_from_row
from skill3d.adapters.frame_set import (
    PREPROCESSING_VERSION, frame_set_hash, uniform_frame_ids, with_pixel_identity,
)
from skill3d.adapters.split_builder import SPLIT_KEYS, validate_split_mapping
from skill3d.cluster.scene_compute import _no_symlinks, _read_json, _reference
from skill3d.cluster.publication import inspect_receipt
from skill3d.infra.m5_identity import content_sha256, file_identity
from skill3d.online.config import load_yaml
from skill3d.routing.task_classifier import canonical_task
from skill3d.schemas import VSIBenchEpisode
from skill3d.segmentation.prepared_binding_store import (
    PreparedBindingStore, load_prepared_binding_reference,
)
from skill3d.storage.relocation import (
    create_relocation_receipt, validate_relocation_receipt,
)


SCHEMA = "harness3d-frozen-scene/1"
PROGRESS_SCHEMA = "harness3d-fullqa-input-freeze-progress/1"
LABEL_FIELDS = {
    "ground_truth", "answer", "reference_answer", "reference_answers",
    "correct_answer", "labels",
}


def reject_labels(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            location = (*path, key)
            compatibility = (
                len(location) >= 3
                and location[-3] in {
                    "logical_input_identity", "metric_logical_input_identity",
                }
                and location[-2:] == ("episode", "ground_truth") and child == ""
            )
            if key in LABEL_FIELDS and not compatibility:
                raise ValueError("forbidden label field at " + ".".join(map(str, location)))
            reject_labels(child, location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_labels(child, (*path, index))


def read(path):
    path = Path(path).absolute()
    _no_symlinks(path)
    return _read_json(path)


def local(path):
    path = Path(path).absolute()
    _no_symlinks(path)
    if not path.is_file():
        raise ValueError(f"missing local input file: {path}")
    return path.resolve()


def within(root, path):
    path = local(path)
    if not path.is_relative_to(root):
        raise ValueError(f"input escapes publication root: {path}")
    return path


def component(value):
    if not isinstance(value, str) or "/" in _reference(value):
        raise ValueError("scene ID must be one safe path component")
    return value


def write(path, value, *, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    _no_symlinks(path)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if read(path) != value:
                    raise ValueError(f"immutable freeze changed: {path}")
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    _no_symlinks(path)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def cached_metric_hashes():
    # Metric receipt validation uses this leaf repeatedly for shared arrays.
    # Use the same inode/size/mtime/ctime-checked cache as relocation identities.
    # This is process-local, restored on exit, and does not change source files.
    from skill3d.reconstruction import metric_evidence

    original = metric_evidence.file_sha256
    metric_evidence.file_sha256 = lambda path: file_identity(path)["sha256"]
    try:
        yield
    finally:
        metric_evidence.file_sha256 = original


def preflight(args):
    scope = read(args.scope)
    reject_labels(scope)
    if scope.get("schema") != "harness3d-full-question-precompute-scope/1":
        raise ValueError("expected the full public question scope schema")
    for flag in ("solver_enabled", "scoring_enabled", "reference_answers_in_scope"):
        if scope.get(flag, False) is not False:
            raise ValueError(f"public scope must disable {flag}")
    mapping = load_yaml(args.split_config)
    if not all(key in mapping for key in SPLIT_KEYS.values()):
        raise ValueError("historical split configuration requires all four scene lists")
    split_scenes = validate_split_mapping(mapping)
    splits = {scene: split for split, key in SPLIT_KEYS.items() for scene in mapping[key]}
    grouped, by_id = {}, {}
    rows = scope.get("questions")
    if not isinstance(rows, list):
        raise ValueError("scope questions must be a list")
    allowed = {"qa_id", "dataset", "scene_name", "question_type", "question",
               "options", "source_split"}
    for row in rows:
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError("scope contains non-public question fields")
        for key in ("qa_id", "dataset", "scene_name", "question_type", "question"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"public question requires nonempty {key}")
        scene = component(row["scene_name"])
        if row["qa_id"] in by_id:
            raise ValueError("duplicate public QA ID")
        if scene not in splits or row.get("source_split", splits[scene]) != splits[scene]:
            raise ValueError("public question differs from its historical scene split")
        canonical_task(row["question_type"])
        options = row.get("options")
        if options is not None and (
                not isinstance(options, list) or any(not isinstance(o, str) for o in options)):
            raise ValueError("public options must be a string list or null")
        by_id[row["qa_id"]] = row
        grouped.setdefault(scene, []).append({**row, "options": options})
    if (len(rows), len(grouped)) != (args.expected_questions, args.expected_scenes):
        raise ValueError("complete public scope question/scene counts changed")
    if split_scenes != set(grouped):
        raise ValueError("historical split lists do not cover the complete public scope")
    for key, actual in (("qa", len(rows)), ("scenes", len(grouped)),
                        ("max_qa_per_scene", max(map(len, grouped.values())))):
        if scope.get("counts", {}).get(key, actual) != actual:
            raise ValueError(f"public scope count differs: {key}")
    split_hash = file_identity(local(args.split_config))["sha256"]
    if scope.get("source_split_config_sha256", split_hash) != split_hash:
        raise ValueError("historical split config bytes changed")

    legacy = {}
    paths = sorted(args.legacy_frozen_root.glob("*.json")) if args.legacy_frozen_root else []
    for path in paths:
        value = read(path)
        reject_labels(value)
        scene = component(value.get("scene_id"))
        ids = value.get("qa_ids")
        if (value.get("schema") != SCHEMA or value.get("status") != "frozen"
                or value.get("real_labels_available") is not False
                or scene not in grouped or scene in legacy or value.get("split") != splits[scene]
                or not isinstance(ids, list) or not ids or any(not isinstance(q, str) for q in ids)
                or ids != sorted(set(ids)) or not set(ids) <= set(by_id)
                or any(by_id[q]["scene_name"] != scene for q in ids)
                or not isinstance(value.get("questions"), list)
                or len(value["questions"]) != len(ids)
                or len(value.get("frames", [])) != 32):
            raise ValueError(f"invalid legacy frozen scene: {path}")
        legacy[scene] = value
    if args.legacy_frozen_root and (len(legacy), sum(len(v["qa_ids"]) for v in legacy.values())) != (
            args.expected_legacy_scenes, args.expected_legacy_questions):
        raise ValueError("complete legacy frozen scene/question counts changed")
    catalog = read(args.geometry_catalog)
    reject_labels(catalog)
    if (catalog.get("schema") != "harness3d-published-binding-catalog/1"
            or catalog.get("status") != "verified"
            or catalog.get("no_labels") is not True or catalog.get("no_solver") is not True
            or set(catalog.get("variants", {})) != {"base", "metric"}
            or catalog.get("planned_questions") != args.expected_legacy_questions):
        raise ValueError("geometry catalog must be the fully verified base+metric catalog")
    old_ids = set(catalog["variants"]["base"])
    if len(old_ids) != args.expected_legacy_questions or not old_ids <= set(by_id):
        raise ValueError("catalog original question scope changed")
    old_scenes = {by_id[q]["scene_name"] for q in old_ids}
    if len(old_scenes) != args.expected_legacy_scenes:
        raise ValueError("catalog original scene scope changed")
    if args.legacy_frozen_root and (
            old_scenes != set(legacy)
            or any(set(value["qa_ids"]) != {q for q in old_ids if by_id[q]["scene_name"] == scene}
                   for scene, value in legacy.items())):
        raise ValueError("legacy manifests and catalog have different scene/QA coverage")
    if "previously_prepared_qa_ids" in scope:
        previous = scope["previously_prepared_qa_ids"]
        if (not isinstance(previous, list) or len(previous) != len(old_ids)
                or set(previous) != old_ids):
            raise ValueError("catalog QA coverage differs from public scope provenance")
    for name in ("base", "metric"):
        entries = catalog["variants"][name]
        if (not isinstance(entries, dict) or set(entries) != old_ids
                or catalog.get("verified_questions", {}).get(name) != len(old_ids)):
            raise ValueError("geometry catalog has incomplete original QA coverage")
        for qa_id, entry in entries.items():
            row = by_id[qa_id]
            if (entry.get("qa_id") != qa_id or entry.get("scene_id") != row["scene_name"]
                    or entry.get("split") != splits[row["scene_name"]]
                    or entry.get("variant") != name):
                raise ValueError("geometry catalog question/scene/split/variant changed")
    selected = sorted(set(args.scene or grouped))
    if set(selected) - set(grouped):
        raise ValueError("partition names scenes outside the complete public scope")
    if set(selected).intersection(old_scenes) - set(legacy):
        raise ValueError("catalog-covered scene partition requires --legacy-frozen-root")
    spec = {
        "schema": "harness3d-fullqa-input-freeze-spec/1",
        # Content-only pins let independent hosts freeze disjoint partitions.
        # Source video/FrameSet identities stay in the episodes, while current
        # transport locations are deliberately outside the logical freeze ID.
        "scope": file_identity(local(args.scope)),
        "split_config": {"sha256": split_hash},
        "geometry_catalog": file_identity(local(args.geometry_catalog)),
        "expected_questions": args.expected_questions, "expected_scenes": args.expected_scenes,
        "expected_legacy_questions": args.expected_legacy_questions,
        "expected_legacy_scenes": args.expected_legacy_scenes,
        "solver_ran": False, "scoring_ran": False, "model_requests": 0,
        "real_labels_available": False,
    }
    spec["sha256"] = content_sha256(spec)
    return grouped, splits, legacy, catalog, selected, spec


def frame_pixels(frames, sources):
    if not isinstance(frames, list) or len(frames) != 32:
        raise ValueError("scene requires exactly 32 saved frames")
    pixels, seen = [], set()
    for row in frames:
        if not isinstance(row, dict) or set(row) != {
                "source_path", "png_sha256", "size", "rgb_sha256", "shape", "dtype"}:
            raise ValueError("invalid frozen frame record")
        source = _reference(row["source_path"], absolute=True)
        if source in seen:
            raise ValueError("duplicate frozen frame source")
        seen.add(source)
        path = local(sources[source])
        if file_identity(path) != {"sha256": row["png_sha256"], "size": row["size"]}:
            raise ValueError("saved PNG hash/size changed")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["png_sha256"]:
            raise ValueError("PNG changed while reading")
        with Image.open(BytesIO(raw)) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError("saved frame is not an RGB PNG")
            array = np.array(image)
        if (array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3
                or list(array.shape) != row["shape"] or str(array.dtype) != row["dtype"]
                or hashlib.sha256(array.tobytes()).hexdigest() != row["rgb_sha256"]):
            raise ValueError("saved RGB identity changed")
        pixels.append(array)
    if len({str(local(sources[source])) for source in seen}) != 32:
        raise ValueError("frozen frames alias the same local file")
    return pixels


def episode_item(question, row, split, pixels, frame_sources):
    public = question.get("episode")
    if not isinstance(public, dict) or "ground_truth" in public:
        raise ValueError("public episode must not contain a label field")
    ep = VSIBenchEpisode.model_validate({**public, "ground_truth": ""})
    fs = ep.frame_set
    if (set(question) - {"episode", "source", "video_path", "logical_input_identity",
                        "metric_logical_input_identity"}
            or question.get("source") != "vsi_bench"
            or ep.qa_id != row["qa_id"] or ep.scene_name != row["scene_name"]
            or ep.split != split or fs is None or fs.n_frames != 32
            or fs.n_total_frames < 32 or fs.decode_status != "ok"
            or fs.readable_frame_ids != fs.frame_ids
            or fs.frame_ids != uniform_frame_ids(fs.n_total_frames, 32)
            or fs.source_frame_indices != fs.frame_ids
            or fs.frame_set_hash != frame_set_hash(fs.frame_ids)
            or not math.isfinite(fs.fps) or fs.fps < 0
            or fs.timestamps != [float(i) / fs.fps if fs.fps > 0 else float(i)
                                 for i in fs.frame_ids]
            or fs.episode_id != row["qa_id"] or fs.dataset_id != row["dataset"]
            or fs.scene_name != row["scene_name"] or not fs.video_id
            or fs.preprocessing_version != PREPROCESSING_VERSION
            or fs.frame_refs not in ([], frame_sources) or not fs.pixel_sha256):
        raise ValueError("frozen episode/source FrameSet changed")
    with_pixel_identity(fs, pixels)
    expected = _episode_from_row({**row, "ground_truth": ""}, split, pixels, 32, frame_set=fs)
    if ep.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise ValueError("frozen public question differs from scope or saved pixels")
    video = _reference(question.get("video_path"), absolute=True)
    return EpisodeItem(ep, pixels, source="vsi_bench", video_path=video)


def shared_source(items):
    first = items[0]
    expected = (first.video_path, first.episode.frame_set.model_dump(exclude={"episode_id"}))
    if any((item.video_path, item.episode.frame_set.model_dump(exclude={"episode_id"}))
           != expected for item in items):
        raise ValueError("one scene contains different source videos/FrameSets")


def replay(resolver, item):
    result = resolver.recompute_scene_input_identity(
        episode=item.episode.model_dump(mode="json"), source=item.source,
        video_path=item.video_path, pixels=item.pixels,
    )
    art = resolver.resolved_artifact()
    if art.reconstruction_input_identity != item.episode.frame_set.cache_identity():
        raise ValueError("artifact reconstruction identity differs from saved source pixels")
    return result


def verify_publication(entry, catalog_hash, scene):
    root = Path(entry["publication_root"]).absolute()
    _no_symlinks(root)
    root = root.resolve(strict=True)
    publication = inspect_receipt(
        within(root, root / "publication.json"),
        expected_receipt_hash=entry["publication_receipt_sha256"],
    )
    receipt = publication.receipt
    if (receipt.identity_hash != catalog_hash
            or receipt.validator != "harness3d-scene-qa-validator-v1"
            or receipt.stage != "gpu_aggregate" or receipt.job_key.split("/")[-2] != scene
            or receipt.completed_key != f"{catalog_hash}/{scene}/gpu_aggregate"):
        raise ValueError("catalog publication differs from the verified scene/run")
    inventory = {}
    for record in receipt.artifacts:
        path = within(root, root / record.name)
        if file_identity(path) != {"sha256": record.sha256, "size": record.size_bytes}:
            raise ValueError(f"current publication bytes changed: {record.name}")
        inventory[str(path)] = record.sha256
    return root, inventory


def verify_catalog_variant(name, old_items, catalog):
    scene = old_items[0].episode.scene_name
    first_entry = catalog["variants"][name][old_items[0].episode.qa_id]
    root, inventory = verify_publication(first_entry, catalog["identity_hash"], scene)
    seed, resolver, evidence = None, None, {}
    shared = None
    for item in old_items:
        qa_id = item.episode.qa_id
        entry = catalog["variants"][name][qa_id]
        if (entry["publication_root"] != first_entry["publication_root"]
                or entry["publication_receipt_sha256"] != first_entry["publication_receipt_sha256"]
                or entry["scene_artifact"] != first_entry["scene_artifact"]):
            raise ValueError("catalog scene group mixes different publications/artifacts")
        identity_path = within(root, entry["scene_input_identity_file"])
        artifact_path = within(root, entry["scene_artifact"])
        if str(artifact_path) not in inventory or inventory.get(str(identity_path)) != (
                entry["scene_input_identity_file_sha256"]):
            raise ValueError("catalog artifact/identity is not anchored in current publication")
        identity = read(identity_path)
        if file_identity(identity_path)["sha256"] != entry["scene_input_identity_file_sha256"]:
            raise ValueError("catalog input identity sidecar changed")
        reject_labels({"logical_input_identity": identity})
        validation_path = local(entry["migration_receipt_file"])
        if file_identity(validation_path)["sha256"] != entry["migration_receipt_file_sha256"]:
            raise ValueError("catalog migration receipt sidecar changed")
        validation = read(validation_path)
        if (validation.get("schema") != "harness3d-published-scene-validation/1"
                or validation.get("status") != "validated" or validation.get("scene_id") != scene
                or validation.get("qa_ids") != sorted(i.episode.qa_id for i in old_items)
                or validation.get("no_solver") is not True or validation.get("no_labels") is not True
                or set(validation.get("variants", {})) != {"base", "metric"}):
            raise ValueError("catalog migration validation differs from complete legacy scene")
        migration = validation["variants"][name]["migration_receipts"][qa_id]
        binding = load_prepared_binding_reference(
            entry["reference"], scene_input_identity=identity, migration_receipt=migration,
            scene_artifact=artifact_path, migration_destination_root=root,
        )
        outcome = {
            "status": binding.stats.get("status") or (
                "binding_unavailable" if not binding.materialized
                or binding.stats.get("binding_error") else "prepared"),
            "materialized": binding.materialized, "n_objects": len(binding.objects),
            "binding_error": binding.stats.get("binding_error"),
        }
        if outcome != entry["outcome"] or validation["variants"][name]["outcomes"][qa_id] != outcome:
            raise ValueError("current normal catalog loader outcome changed")
        resolver = PreparedBindingStore(entry["reference"]["store_root"]).validate_migration_receipt(
            entry["reference"]["key"], migration, scene_artifact=artifact_path,
            scene_input_identity=identity, destination_root=root,
        )
        actual = replay(resolver, item)
        geometry = {k: v for k, v in actual.items() if k not in {"episode", "sha256"}}
        if shared is not None and shared != geometry:
            raise ValueError("catalog legacy questions have different geometry/source identities")
        shared, seed = geometry, actual
        evidence[qa_id] = {
            "input_identity_sha256": actual["sha256"],
            "input_identity_file": str(identity_path),
            "input_identity_file_sha256": entry["scene_input_identity_file_sha256"],
            "migration_receipt_file": str(validation_path),
            "migration_receipt_file_sha256": entry["migration_receipt_file_sha256"],
            "normal_loader_outcome": outcome,
        }
    art = resolver.resolved_artifact()
    if name == "base" and art.metric_model not in (None, "none"):
        raise ValueError("catalog base artifact contains metric augmentation")
    if name == "metric" and art.metric_model in (None, "none"):
        raise ValueError("catalog metric artifact has no metric evidence")
    return seed, resolver, {
        "publication_root": str(root),
        "publication_receipt_sha256": first_entry["publication_receipt_sha256"],
        "artifact": {"path": str(resolver.artifact_path), **file_identity(resolver.artifact_path)},
        "verified_original_questions": evidence,
        "reconstruction_input_identity": art.reconstruction_input_identity,
        "metric_input_identity": art.metric_input_identity,
        "scale_fusion_status": art.scale_fusion_status,
    }


def expand_identity(seed, old_resolver, item):
    # Preserve every saved geometry locator while replaying each expanded
    # episode against the current, explicitly receipted publication bytes.
    body = copy.deepcopy(seed)
    body.pop("sha256")
    body.update(episode=item.episode.model_dump(mode="json"), source=item.source,
                video_path=item.video_path)
    candidate = {**body, "sha256": content_sha256(body)}
    root = old_resolver.artifact_path.parents[2]
    dependency_refs = {name: path.relative_to(root).as_posix()
                       for name, path in old_resolver.dependency_paths.items()}
    receipt = create_relocation_receipt(
        logical_input_identity=candidate, destination_root=root,
        artifact_ref=old_resolver.artifact_path.relative_to(root).as_posix(),
        dependency_refs=dependency_refs,
    )
    resolver = validate_relocation_receipt(
        receipt, expected_input_sha256=candidate["sha256"], destination_root=root,
    )
    return replay(resolver, item)


def transfer_mapping(value, sources):
    files = {row["source_path"]: f"data/frames/{slot:04d}.png"
             for slot, row in enumerate(value["frames"])}
    for name, field, identity_field in (
            ("base", "artifact_source_path", "logical_input_identity"),
            ("metric", "metric_artifact_source_path", "metric_logical_input_identity")):
        source = value.get(field)
        if source is None:
            continue
        files[source] = f"data/geometry/{name}/artifact.json"
        for key, record in sorted(value["questions"][0][identity_field]["files"].items()):
            source = record["path"]
            if source not in files:
                files[source] = f"data/geometry/{name}/{key}{Path(source).suffix}"
    if set(files) != set(sources) or len(set(files.values())) != len(files):
        raise ValueError("transfer mapping must cover exactly all scene input files")
    value["files"] = files
    value["local_sources"] = {target: str(local(sources[source])) for source, target in files.items()}
    value["file_identities"] = {target: file_identity(local(sources[source]))
                                for source, target in files.items()}
    for question in value["questions"]:
        for field, source_field in (("logical_input_identity", "artifact_source_path"),
                                    ("metric_logical_input_identity", "metric_artifact_source_path")):
            if field not in question:
                continue
            logical = question[field]
            if (logical["artifact_path"] != value[source_field]
                    or value["file_identities"][files[value[source_field]]]["sha256"]
                    != logical["artifact_json_sha256"]):
                raise ValueError("transfer artifact differs from replayed identity")
            for row in logical["files"].values():
                if value["file_identities"][files[row["path"]]]["sha256"] != row["sha256"]:
                    raise ValueError("transfer dependency differs from replayed identity")


def existing_scene(rows, scene, split, legacy, catalog, args, spec):
    original = legacy[scene]
    sources = {}
    for slot, frame in enumerate(original["frames"]):
        source = frame["source_path"]
        sibling = args.legacy_frozen_root.parent / "new_frames" / scene / f"{slot:04d}.png"
        mapped = original.get("local_sources", {}).get(original.get("files", {}).get(source))
        candidates = [sibling]
        if mapped:
            candidates.append(Path(mapped))
        candidates.append(Path(source))
        path = next((path for path in candidates if path.is_file()), None)
        if path is None:
            raise ValueError(f"missing saved legacy PNG: {scene}/{slot:04d}")
        sources[source] = str(local(path))
    pixels = frame_pixels(original["frames"], sources)
    by_id = {row["qa_id"]: row for row in rows}
    old_items = [episode_item(q, by_id[qa_id], split, pixels, list(sources))
                 for qa_id, q in zip(original["qa_ids"], original["questions"])]
    shared_source(old_items)
    first = old_items[0]
    questions, items = [], []
    for row in sorted(rows, key=lambda row: row["qa_id"]):
        fs = first.episode.frame_set.model_copy(update={"episode_id": row["qa_id"]})
        ep = _episode_from_row({**row, "ground_truth": ""}, split, pixels, 32, frame_set=fs)
        public = ep.model_dump(mode="json")
        public.pop("ground_truth")
        questions.append({"episode": public, "source": first.source, "video_path": first.video_path})
        items.append(EpisodeItem(ep, pixels, source=first.source, video_path=first.video_path))
    value = {
        "schema": SCHEMA, "scene_id": scene, "split": split,
        "qa_ids": sorted(by_id), "questions": questions,
        "frames": copy.deepcopy(original["frames"]), "status": "frozen",
        "real_labels_available": False, "reuse_evidence": {
            "schema": "harness3d-fullqa-geometry-reuse/1",
            "legacy_manifest_sha256": file_identity(
                local(args.legacy_frozen_root / f"{scene}.json"))["sha256"],
            "geometry_catalog_sha256": spec["geometry_catalog"]["sha256"],
            "catalog_identity_hash": catalog["identity_hash"],
            "raw_video_opened": False, "frozen_frames_recomputed": False, "variants": {},
        },
    }
    variants = {}
    for name, field, identity_field in (
            ("base", "artifact_source_path", "logical_input_identity"),
            ("metric", "metric_artifact_source_path", "metric_logical_input_identity")):
        seed, resolver, evidence = verify_catalog_variant(name, old_items, catalog)
        variants[name] = resolver.resolved_artifact()
        value[field] = seed["artifact_path"]
        sources[value[field]] = str(resolver.artifact_path)
        for key, path in resolver.dependency_paths.items():
            source = seed["files"][key]["path"]
            if source in sources and local(sources[source]) != path:
                raise ValueError("one logical source maps to different local files")
            sources[source] = str(path)
        for question, item in zip(questions, items):
            question[identity_field] = expand_identity(seed, resolver, item)
        value["reuse_evidence"]["variants"][name] = evidence
    from skill3d.reconstruction.metric_evidence import GEOMETRY_REFS

    for key in GEOMETRY_REFS:
        base, metric = getattr(variants["base"], key), getattr(variants["metric"], key)
        if bool(base) != bool(metric) or base and file_identity(base) != file_identity(metric):
            raise ValueError("base and metric variants do not share the same geometry bytes")
    transfer_mapping(value, sources)
    return value


def new_scene(rows, scene, split, args, target):
    if target.is_file():
        value = read(target)
        if value.get("artifact_source_path") is not None or value.get("metric_artifact_source_path") is not None:
            raise ValueError("new-scene resume unexpectedly contains geometry")
        sources = {source: value["local_sources"][relative]
                   for source, relative in value["files"].items()}
        pixels = frame_pixels(value["frames"], sources)
    elif args.new_frozen_root:
        source_manifest = local(args.new_frozen_root / "scenes" / f"{scene}.json")
        value = read(source_manifest)
        reject_labels(value)
        if (value.get("artifact_source_path") is not None
                or value.get("metric_artifact_source_path") is not None):
            raise ValueError("raw-only imported scene unexpectedly contains geometry")
        value["metric_artifact_source_path"] = None
        sources = {row["source_path"]: str(
            args.new_frozen_root / "new_frames" / scene / f"{slot:04d}.png")
                   for slot, row in enumerate(value["frames"])}
        pixels = frame_pixels(value["frames"], sources)
        transfer_mapping(value, sources)
        value["import_evidence"] = {
            "schema": "harness3d-fullqa-new-frame-import/1",
            "source_manifest_sha256": file_identity(source_manifest)["sha256"],
            "raw_video_opened": False, "frame_files_copied": False,
        }
    else:
        value = frozen_scene(rows, scene, split, None, args.root,
                             args.video_root, args.fallback_root)
        if value.get("status") != "frozen":
            raise ValueError("new-scene frame freeze failed: " + json.dumps(value, sort_keys=True))
        value["metric_artifact_source_path"] = None
        sources = {row["source_path"]: row["source_path"] for row in value["frames"]}
        pixels = frame_pixels(value["frames"], sources)
        transfer_mapping(value, sources)
    if (value.get("schema") != SCHEMA or value.get("status") != "frozen"
            or value.get("scene_id") != scene or value.get("split") != split
            or value.get("qa_ids") != sorted(row["qa_id"] for row in rows)
            or value.get("real_labels_available") is not False
            or len(value.get("questions", [])) != len(rows)):
        raise ValueError("new frozen scene differs from complete public QA group")
    frame_sources = [row["source_path"] for row in value["frames"]]
    items = [episode_item(q, row, split, pixels, frame_sources)
             for q, row in zip(value["questions"], sorted(rows, key=lambda row: row["qa_id"]))]
    if any("logical_input_identity" in q or "metric_logical_input_identity" in q
           for q in value["questions"]):
        raise ValueError("new frozen scene has premature artifact identities")
    shared_source(items)
    if target.is_file():
        expected = copy.deepcopy(value)
        transfer_mapping(expected, sources)
        if expected != value:
            raise ValueError("new-scene transfer mapping changed on resume")
    return value


def progress(root, grouped, selected, spec, failures, began):
    with lock(root / ".locks/progress.lock"):
        global_failures = {path.stem: read(path) for path in (root / "failures").glob("*.json")}
        scenes = {}
        for path in sorted((root / "scenes").glob("*.json")):
            value = read(path)
            scene = value.get("scene_id")
            if (scene not in grouped or path.name != f"{scene}.json"
                    or value.get("freeze_spec_sha256") != spec["sha256"]
                    or value.get("qa_ids") != sorted(row["qa_id"] for row in grouped[scene])
                    or value.get("status") != "frozen"):
                raise ValueError("existing output scene differs from complete freeze specification")
            if scene in global_failures:
                continue
            scenes[scene] = {"questions": len(value["qa_ids"]),
                             "manifest_sha256": file_identity(path)["sha256"]}
        pending = sorted(set(grouped) - set(scenes))
        selected_pending = sorted(set(selected) - set(scenes))
        common = {
            "schema": PROGRESS_SCHEMA, "freeze_spec_sha256": spec["sha256"],
            "updated_at": time.time(), "elapsed_s": time.monotonic() - began,
            "planned_scenes": len(grouped), "planned_questions": sum(map(len, grouped.values())),
            "frozen_scenes": len(scenes),
            "frozen_questions": sum(row["questions"] for row in scenes.values()),
            "pending_scene_ids": pending, "scenes": scenes,
            "solver_ran": False, "scoring_ran": False, "model_requests": 0,
            "real_labels_available": False,
        }
        write(root / "progress.json", {
            **common, "status": "frozen" if not pending else "incomplete" if global_failures else "partial",
            "failures": global_failures,
        })
        partition = content_sha256(selected)[:16]
        result = {
            **common, "partition_id": partition, "selected_scene_ids": selected,
            "selected_scenes": len(selected),
            "selected_questions": sum(len(grouped[scene]) for scene in selected),
            "selected_pending_scene_ids": selected_pending, "failures": failures,
            "status": "incomplete" if failures else "frozen" if not selected_pending else "running",
        }
        write(root / "progress" / f"{partition}.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--legacy-frozen-root", type=Path,
                        help="required for reused scenes, optional for new-scene-only partitions")
    parser.add_argument("--new-frozen-root", type=Path,
                        help="import raw-only scenes/ and new_frames/ without decoding videos")
    parser.add_argument("--geometry-catalog", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--fallback-root", action="append", default=[])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene", action="append", help="repeat for pilot/CPU partition; full scope is still validated")
    parser.add_argument("--expected-questions", type=int, default=5130)
    parser.add_argument("--expected-scenes", type=int, default=288)
    parser.add_argument("--expected-legacy-scenes", type=int, default=230)
    parser.add_argument("--expected-legacy-questions", type=int, default=1154)
    args = parser.parse_args()
    if (args.expected_questions < 1 or args.expected_scenes < 1
            or not 0 <= args.expected_legacy_scenes <= args.expected_scenes
            or not 0 <= args.expected_legacy_questions <= args.expected_questions):
        parser.error("invalid expected scope counts")
    args.root = args.root.absolute()
    _no_symlinks(args.root)
    if args.legacy_frozen_root:
        args.legacy_frozen_root = args.legacy_frozen_root.absolute()
        _no_symlinks(args.legacy_frozen_root)
    if args.new_frozen_root:
        args.new_frozen_root = args.new_frozen_root.absolute()
        _no_symlinks(args.new_frozen_root)
    if "source_frozen" in args.root.parts:
        parser.error("output must not be inside source_frozen")
    for source in (args.scope, args.geometry_catalog, args.split_config,
                   args.legacy_frozen_root, args.new_frozen_root):
        if source is None:
            continue
        source = source.absolute()
        if source.is_relative_to(args.root) or args.root.is_relative_to(source):
            parser.error("new task root must not overlap input manifests or legacy scene root")
    grouped, splits, legacy, catalog, selected, spec = preflight(args)
    write(args.root / "freeze_spec.json", spec, immutable=True)
    failures, began = {}, time.monotonic()
    progress(args.root, grouped, selected, spec, failures, began)
    with cached_metric_hashes():
        for scene in selected:
            target = args.root / "scenes" / f"{scene}.json"
            with lock(args.root / ".locks" / f"{scene}.lock"):
                try:
                    if target.is_file() and read(target).get("freeze_spec_sha256") != spec["sha256"]:
                        raise ValueError("resumed scene has a different freeze specification")
                    value = (existing_scene(grouped[scene], scene, splits[scene], legacy, catalog, args, spec)
                             if scene in legacy else new_scene(grouped[scene], scene, splits[scene], args, target))
                    value.update(freeze_spec_sha256=spec["sha256"],
                                 solver_ran=False, scoring_ran=False, model_requests=0)
                    reject_labels(value)
                    write(target, value, immutable=True)
                    (args.root / "failures" / f"{scene}.json").unlink(missing_ok=True)
                except Exception as exc:
                    failures[scene] = {
                        "scene_id": scene, "status": "input_error",
                        "error_type": type(exc).__name__, "detail": str(exc),
                    }
                    write(args.root / "failures" / f"{scene}.json", failures[scene])
            result = progress(args.root, grouped, selected, spec, failures, began)
            print(json.dumps({
                "scene_id": scene, "status": "input_error" if scene in failures else "frozen",
                "frozen_scenes": result["frozen_scenes"], "frozen_questions": result["frozen_questions"],
            }, sort_keys=True), flush=True)
    result = progress(args.root, grouped, selected, spec, failures, began)
    print(json.dumps({key: value for key, value in result.items() if key != "scenes"},
                     sort_keys=True, allow_nan=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
