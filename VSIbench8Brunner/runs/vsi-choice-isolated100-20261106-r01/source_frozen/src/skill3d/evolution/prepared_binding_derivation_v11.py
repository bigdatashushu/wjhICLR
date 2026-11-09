"""Explicit, model-free v3 category-only to v4 binding derivation.

The caller supplies trusted published source references, not a self-issued
source seal. Every derivation/replay loads those immutable entries again, checks
the original binary receipts and checks the complete destination scene group.
Explicit source-audit reuse preserves the old verdicts without repeating mask
conflict or numeric reviews. Historical generation identities are never resealed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.evaluation.skill_ablation_v11 import input_identity
from skill3d.infra.m5_identity import (
    M5_IDENTITY_VERSION, array_identity, content_sha256, file_identity, source_identity,
)
from skill3d.legacy.readers import load_artifact_v5
from skill3d.online import runner
from skill3d.online.runner import OnlineRunConfig, PreparedObjectBinding
from skill3d.segmentation.binding_validation import (
    BINDING_VALIDATION_VERSION, SEMANTIC_IDENTITY_KEY, _semantic_identity,
    _semantic_policy_passes, validate_binding,
)
from skill3d.segmentation.intern_binary_review import (
    admission_identity, validate_category_receipt,
)
from skill3d.segmentation.prepared_binding_store import (
    OBJECT_REFS, PreparedBindingStore, PreparedBindingStoreError,
    load_prepared_binding_reference, prepared_binding_request_identity,
)
from skill3d.segmentation.question_binding import (
    M5_BINDING_VERSION, question_input, refresh_coverage,
)
from skill3d.segmentation.sam2_tracker import audit_semantic_mask_conflicts
from skill3d.storage.relocation import (
    RelocationError, ValidatedRelocation, artifact_dependency_refs, read_relocation_receipt,
)

DERIVATION_SCHEMA_VERSION = "v11-prepared-binding-derivation/1.0"
LEGACY_BINDING_VERSION = "m5-question-binding-v3-grouped-review"
LEGACY_SOURCE_COMMIT = "3588bc8"
# Full SOURCE_REFS snapshot from the reviewed commit, including its null vllm ref.
LEGACY_SOURCES_SHA256 = "0fd603e51ad1eea00f786509c45b1008cfa5d3f9dbd3bd7a453accabe0dbbe3c"
LEGACY_STORE_SOURCE = {
    "size": 22304,
    "sha256": "ee39e7f64ce2414029859f5822760929a216ce01604b2669758415b9e49f151b",
}
AUDIT_MODES = ("reaudit", "reuse_source_audit")
LEGACY_VALIDATION_VERSION = "m5-binding-validation-v2-semantic-review"
_REUSED_AUDIT_SOURCES = {
    "segmentation/sam2_tracker.py": {
        "size": 40757,
        "sha256": "fcfbdc4f7fed3e64d7da6fa79313d90e5eebca8962fc0006a229ce1cb62ca3a3",
    },
    "tools/category_match.py": {
        "size": 10190,
        "sha256": "76032ba2a8975f9c6d2b209aedd2233528b665de4d5f1794630b15faf6112de7",
    },
}
_REQUEST_CHANGES = {
    "input_sha256", "scene_audit_qa_ids", "binding_version", "parameters",
    "implementation_sources", "store_source", "sha256",
}
_VISUAL_FIELDS = {
    "semantic_visual_request_sha256": "request_sha256",
    "semantic_request_identity": "request_identity",
    "semantic_model_identity_sha256": "model_identity_sha256",
    "semantic_admission": "admission",
    "semantic_binary_review": "binary_review",
    "semantic_binary_review_sha256": "binary_review_sha256",
    "semantic_binary_response": "raw_response",
    "semantic_observation": "observation",
    "semantic_observation_sha256": "observation_sha256",
    "semantic_frozen_quality": "frozen_quality",
}


@dataclass(frozen=True)
class PreparedBindingDerivationContext:
    """Worker-approved anchors, supplied out of band at validation time.

    ``sources[qa_id]`` accepts ``prepare_published()`` with its explicit original
    ``migration_receipt`` and ``migration_destination_root`` added by the caller.
    Non-provider callers supply ``reference``, ``identity``, ``artifact_path``
    and optionally ``migration_receipt``/``migration_destination_root``.
    ``prepared`` and ``artifact`` from the provider are checked, never trusted
    instead of reloading the store. Destination paths must identify real local
    artifacts; optional destination resolvers support unchanged logical inputs.
    Keep cfg.seed at the original M5 seed (0), independently of solver seeds.
    """

    sources: Mapping[str, Mapping]
    items: Sequence[EpisodeItem]
    target_inputs: Mapping[str, Mapping]
    target_requests: Mapping[str, Mapping]
    target_artifact_paths: Mapping[str, str | Path]
    cfg: OnlineRunConfig
    target_relocations: Mapping[str, ValidatedRelocation] | None = None
    target_migration_receipts: Mapping[str, Mapping | str | Path] | None = None
    audit_mode: str = "reaudit"
    _replay_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)


def _require(passed, message):
    if not passed:
        raise PreparedBindingStoreError(f"binding derivation: {message}")


def _digest(value, field, label):
    _require(isinstance(value, Mapping) and value.get(field) == content_sha256({
        key: child for key, child in value.items() if key != field
    }), f"{label} hash changed")


def _content(binding):
    """Path-independent output identity survives immutable store repacking."""
    def pack(value):
        if isinstance(value, np.ndarray):
            return {"array_identity": array_identity(value)}
        if isinstance(value, Mapping):
            return {key: pack(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(child) for child in value]
        return value

    objects = []
    for obj in binding.objects:
        row = obj.model_dump(mode="json")
        for field in OBJECT_REFS:
            if row.get(field):
                row[field] = file_identity(row[field])
        objects.append(row)
    return pack({
        "objects": objects,
        "stats": {key: value for key, value in binding.stats.items()
                  if key != "binding_derivation"},
        "notes": binding.notes, "materialized": binding.materialized,
        "prepared_measurements": binding.prepared_measurements,
    })


def _stamp(path):
    path = Path(path).absolute()
    _require(not any(parent.is_symlink() for parent in (path, *path.parents)),
             "symlink in derivation closure")
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _arrays(art, cache):
    values = {}
    for name, ref in (
        ("depth", "depth_maps"), ("c2w", "c2w_list"),
        ("intrinsics", "intrinsics"), ("point_conf", "depth_conf"),
        ("depth_conf", "depth_conf"), ("point_map", "point_map"),
    ):
        path = str(getattr(art, ref) or "")
        if not path:
            values[name] = None
            continue
        key = (path, _stamp(path))
        if key not in cache:
            cache[key] = np.load(path, allow_pickle=False, mmap_mode="r")
        values[name] = cache[key]
    return values


def _geometry(values, cache):
    result = {}
    for name, value in values.items():
        key = id(value)
        if key not in cache:
            cache[key] = array_identity(value)
        result[name] = cache[key]
    return result


def _source_receipt(source):
    return source.get("migration_receipt") or (
        source.get("transport_receipt") or {}).get("migration_receipt_file")


def _context_fingerprint(context):
    # Do not serialize cfg's runtime derivation-context back-reference.
    cfg_fields = (
        "mode", "seed", "online_model_provider", "online_model",
        "online_model_revision", "online_base_url", "model_weights_sha256",
        "semantic_min_pixels", "semantic_max_pixels", "semantic_relation_batch_size",
        "semantic_review_config", "temperature", "max_tokens",
    )
    return content_sha256({
        "sources": {
            qa: {
                "reference": source["reference"], "identity": source["identity"],
                "artifact_path": str(source["artifact_path"]),
                "migration_receipt": (
                    str(_source_receipt(source)) if isinstance(_source_receipt(source), Path)
                    else _source_receipt(source)),
                "migration_destination_root": str(source.get("migration_destination_root") or ""),
            }
            for qa, source in context.sources.items()
        },
        "items": [{
            "episode": item.episode.model_dump(mode="json"),
            "source": item.source, "video_path": item.video_path,
        } for item in context.items],
        "target_inputs": context.target_inputs, "target_requests": context.target_requests,
        "target_paths": {qa: str(path) for qa, path in context.target_artifact_paths.items()},
        "target_relocations": {
            qa: resolver.logical_input_identity
            for qa, resolver in (context.target_relocations or {}).items()},
        "target_migration_receipts": {
            qa: str(receipt) if isinstance(receipt, Path) else receipt
            for qa, receipt in (context.target_migration_receipts or {}).items()},
        "cfg": {name: getattr(context.cfg, name) for name in cfg_fields},
        "audit_mode": context.audit_mode,
        "implementation": _implementation(),
        "current_sources": source_identity(Path(__file__).resolve().parents[1]),
    })


def _closure_stamps(context):
    paths = set()
    artifacts = {}

    def add_artifact(path):
        path = Path(path).absolute()
        paths.add(path)
        if path not in artifacts:
            artifacts[path] = load_artifact_v5(path)
        paths.update(Path(ref).absolute() for ref in artifact_dependency_refs(artifacts[path]).values())

    for source in context.sources.values():
        store = PreparedBindingStore(source["reference"]["store_root"])
        key = source["reference"]["key"]
        manifest = store.manifest(key)
        entry = store.entry_path(key)
        paths.add(entry / "manifest.json")
        paths.add(entry / "payload")
        paths.update(path for path in (entry / "payload").rglob("*") if path.is_dir())
        paths.update(entry / ref for ref in manifest["payload"]["files"])
        receipt = _source_receipt(source)
        if receipt is not None:
            if isinstance(receipt, (str, Path)):
                paths.add(Path(receipt).absolute())
            data = read_relocation_receipt(receipt)
            root = Path(data["destination_root"])
            paths.update(root / row["target_ref"] for row in (
                data["artifact"], *data["dependencies"].values()))
        else:
            add_artifact(source["artifact_path"])
    for qa, path in context.target_artifact_paths.items():
        receipt = (context.target_migration_receipts or {}).get(qa)
        if isinstance(receipt, (str, Path)):
            paths.add(Path(receipt).absolute())
        resolver = (context.target_relocations or {}).get(qa)
        if resolver is not None:
            paths.add(resolver.artifact_path)
            paths.update(resolver.dependency_paths.values())
        else:
            add_artifact(path)
    return {str(path): _stamp(path) for path in paths}


def _check_cache(context):
    cache = context._replay_cache
    if not cache:
        return False
    _require(cache["fingerprint"] == _context_fingerprint(context),
             "cached derivation context or implementation changed")
    _require(all(_stamp(path) == stamp for path, stamp in cache["closure"].items()),
             "cached sealed source/destination closure changed")
    return True


def _scene(art):
    return str(art.reconstruction_input_identity or art.frame_set_hash or art.artifact_id)


def _prior_item(item, identity):
    old, new = identity["episode"], item.episode.model_dump(mode="json")
    _require(old.get("ground_truth") == new.get("ground_truth") == "",
             "only public questions may be derived")
    _require({k: v for k, v in old.items() if k not in {"split", "frame_set"}}
             == {k: v for k, v in new.items() if k not in {"split", "frame_set"}},
             "question or episode identity changed")
    old_fs, new_fs = old["frame_set"], new["frame_set"]
    locations = {"frame_refs", "pixel_sha256"}
    _require({k: v for k, v in old_fs.items() if k not in locations}
             == {k: v for k, v in new_fs.items() if k not in locations},
             "frame provenance changed")
    _require(item.source == identity["source"] and item.video_path == identity["video_path"],
             "episode source changed")
    prior = copy.copy(item)
    prior.episode = type(item.episode).model_validate(old)
    return prior


def _load_source(source, item):
    reference, identity = source["reference"], source["identity"]
    _digest(identity, "sha256", "source input")
    store = PreparedBindingStore(reference["store_root"])
    _require(store.reference(reference["key"]) == dict(reference),
             "trusted source reference changed")
    manifest = store.manifest(reference["key"])
    prior = _prior_item(item, identity)
    receipt = _source_receipt(source)
    root = source.get("migration_destination_root")
    path = Path(source["artifact_path"]).expanduser().resolve()
    if receipt is not None:
        resolver = store.validate_migration_receipt(
            reference["key"], receipt, scene_artifact=path,
            scene_input_identity=identity, destination_root=root)
        actual = resolver.recompute_scene_input_identity(
            episode=prior.episode.model_dump(mode="json"), source=prior.source,
            video_path=prior.video_path, pixels=prior.pixels)
        art = resolver.resolved_artifact()
        relocation_sha = file_identity(receipt)["sha256"] if isinstance(
            receipt, (str, Path)) else content_sha256(receipt)
    else:
        _require(source.get("artifact_relocation") is None,
                 "relocated source requires its original migration receipt")
        actual = input_identity(prior, path)
        art = load_artifact_v5(path)
        relocation_sha = ""
    _require(actual == dict(identity), "source input does not recompute")
    binding = load_prepared_binding_reference(
        reference, scene_input_identity=actual, scene_artifact=path,
        migration_receipt=receipt, migration_destination_root=root)
    if source.get("prepared") is not None:
        _require(_content(source["prepared"]) == _content(binding),
                 "provider binding differs from sealed source payload")
    if source.get("artifact") is not None:
        _require(source["artifact"].model_dump(mode="json") == art.model_dump(mode="json"),
                 "provider artifact differs from sealed source")
    generation = binding.stats.get("generation") or {}
    _require(bool(generation), "source has no sealed generation")
    _require(generation.get("sha256") == content_sha256(generation.get("identity"))
             == manifest["binding_generation_sha256"], "source generation changed")
    return binding, identity, manifest, art, {
        "reference": copy.deepcopy(dict(reference)),
        "input_sha256": identity["sha256"],
        "request_identity": copy.deepcopy(manifest["request_identity"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "payload_sha256": manifest["payload"]["sha256"],
        "generation_sha256": generation["sha256"],
        "generation_stats_sha256": content_sha256(generation),
        "binding_content_sha256": content_sha256(_content(binding)),
        "relocation_sha256": relocation_sha,
    }


def _legacy_contract(binding, old_request, target_request):
    generation = binding.stats["generation"]
    identity = generation["identity"]
    _require(binding.stats.get("version") == old_request.get("binding_version")
             == LEGACY_BINDING_VERSION, "source is not the reviewed v3 binding")
    _require(not binding.stats.get("binding_derivation"), "chained derivation is forbidden")
    _require(identity.get("version") == M5_IDENTITY_VERSION,
             "source generation identity version changed")
    _require(content_sha256(old_request.get("implementation_sources")) == LEGACY_SOURCES_SHA256
             and identity.get("sources") == old_request["implementation_sources"]
             and old_request.get("store_source") == LEGACY_STORE_SOURCE,
             "source implementation is not the exact reviewed 3588bc8 snapshot")
    _require(identity.get("parameters") == old_request.get("parameters"),
             "source generation/request parameters changed")
    old_params, new_params = old_request["parameters"], target_request["parameters"]
    normalized = copy.deepcopy(new_params)
    normalized.pop("binding_validation_version", None)
    normalized["semantic_visual_version"] = "m5-mask-observation-qwen38-v1"
    _require(old_params == normalized
             and "binding_validation_version" not in old_params
             and new_params.get("binding_validation_version") == BINDING_VALIDATION_VERSION,
             "parameters changed beyond the reviewed validation versions")
    expected_admission = admission_identity("category-only")
    review = old_params.get("experimental_semantic_review") or {}
    _require(old_params.get("binary_admission_identity") == expected_admission
             and review.get("mode") == "binary"
             and review.get("admission_policy") == "category-only",
             "source is not category-only")
    _require({k: v for k, v in old_request.items() if k not in _REQUEST_CHANGES}
             == {k: v for k, v in target_request.items() if k not in _REQUEST_CHANGES},
             "model, seed or request contract changed")
    model = identity.get("vlm") or {}
    pins = old_request["model"]
    _require(old_request.get("seed") == target_request.get("seed") == model.get("seed") == 0,
             "original M5 seed must remain 0")
    _require(model.get("enabled") is True
             and model.get("provider") == pins["provider"]
             and model.get("model") == pins["model_id"]
             and model.get("model_revision") == pins["model_revision"]
             and model.get("endpoint_sha256") == pins["endpoint_sha256"],
             "original generation model identity changed")
    _require(not binding.stats.get("binding_error"), "source is a binding error")
    parse = binding.stats.get("object_parse")
    if parse is not None and parse.get("status") == "unavailable":
        _require(binding.stats.get("status") == "parse_unavailable"
                 and not binding.objects and not parse.get("objects")
                 and not binding.stats.get("candidate_receipts")
                 and not binding.stats.get("detection_receipts"),
                 "unavailable source parse contains object evidence")
    else:
        _require(parse is None or parse.get("status") == "parsed",
                 "source question parse status is unsupported")


def _derive_policy(binding, parameters):
    receipts = {}
    for receipt in binding.stats.get("candidate_receipts") or []:
        cid = receipt["candidate_id"]
        _require(cid not in receipts, "duplicate original candidate receipt")
        receipts[cid] = receipt
    expected_admission = admission_identity("category-only")
    review = parameters["experimental_semantic_review"]
    pinned_models = {row["expected_identity_sha256"] for row in review["endpoints"]}
    for obj in binding.objects:
        _require(obj.grounding_status == "question_driven" and obj.binding_provenance,
                 "source object lacks question-driven provenance")
        _require([row.get("candidate_id") for row in obj.binding_provenance]
                 == obj.candidate_ids, "candidate provenance changed")
        for row in obj.binding_provenance:
            _require(row.get("category") == obj.category_name,
                     "object category differs from source candidate")
            receipt = receipts.get(row["candidate_id"]) or {}
            visual, relation = receipt.get("semantic_visual") or {}, receipt.get("semantic_relation") or {}
            _require(visual.get("status") == "observed"
                     and visual.get("provider") == row.get("semantic_provider")
                     == "intern-decision-binary-local"
                     and receipt.get("category") == obj.category_name
                     and receipt.get("frame_slot") == row.get("frame_slot")
                     and receipt.get("bbox") == row.get("bbox"),
                     "exact original semantic receipt is missing")
            for saved, original in _VISUAL_FIELDS.items():
                _require(saved in row and original in visual and row[saved] == visual[original],
                         f"original {saved} is missing or changed")
            _require(row.get("semantic_binary_response_sha256") == content_sha256(visual["raw_response"])
                     and relation.get("status") == "reviewed"
                     and relation.get("relation") == row.get("semantic_relation") == "match"
                     and relation.get("request_sha256") == row.get("semantic_relation_request_sha256")
                     == visual["request_sha256"], "original response/relation seal changed")
            _require(row["semantic_admission"] == expected_admission
                     == row["semantic_request_identity"].get("admission")
                     and row["semantic_model_identity_sha256"] in pinned_models
                     and row["semantic_binary_review"]["prompt_version"] == review["prompt_version"]
                     and row["semantic_binary_review"]["threshold"] == review["yes_threshold"],
                     "original admission/model/prompt/threshold changed")
            _require("semantic_policy" not in row or row["semantic_policy"] == "category-only",
                     "source semantic policy contradicts original admission")
            row["semantic_policy"] = "category-only"
            _require(validate_category_receipt(row, obj.category_name),
                     "original category-only binary evidence does not validate")


def _implementation():
    root = Path(__file__).resolve().parents[1]
    return {
        "migrator": file_identity(Path(__file__)),
        "validator": file_identity(root / "segmentation/binding_validation.py"),
        "prepared_validator": file_identity(root / "segmentation/question_binding.py"),
        "audit": file_identity(root / "segmentation/sam2_tracker.py"),
        "label_relation": file_identity(root / "tools/category_match.py"),
        "store": file_identity(root / "segmentation/prepared_binding_store.py"),
    }


def _reuse_validation(obj, generation_sha):
    """Upgrade only the semantic seal; retain every original numeric verdict."""
    receipt = obj.binding_validation
    _require(receipt is not None and receipt.version == LEGACY_VALIDATION_VERSION
             and receipt.generation_sha256 == generation_sha,
             f"original validation receipt missing or changed: {obj.obj_id}")
    required = {
        "nonempty_mask", "finite_points", "point_confidence", "no_binding_flags",
        "instance_not_suspect", "candidate_identity", "semantic_review",
        "intern_binary_integrity",
    }
    if receipt.checks.get("nonempty_mask"):
        required.update(("visible_frames", "mask_support"))
    if receipt.checks.get("finite_points"):
        required.update(("centroid", "bbox"))
    _require(required <= set(receipt.checks)
             and set(receipt.errors) == {k for k, passed in receipt.checks.items() if not passed}
             and receipt.status == ("unavailable" if receipt.errors else "validated"),
             f"original validation verdict is inconsistent: {obj.obj_id}")
    _require(receipt.checks["no_binding_flags"] == (not bool(obj.binding_flags))
             and receipt.checks["instance_not_suspect"] == (not obj.duplicate_suspect),
             f"original flags/verdict changed: {obj.obj_id}")
    files = {
        name: file_identity(getattr(obj, name))
        for name in ("mask_per_frame", "pointcloud_world", "pointconf_world")
        if getattr(obj, name, None)
    }
    _require(receipt.files == files
             and {"mask_per_frame", "pointcloud_world"} <= set(files),
             f"original validation file identities changed: {obj.obj_id}")
    checks = dict(receipt.checks)
    checks["semantic_policy"] = all(_semantic_policy_passes(row)
                                    for row in obj.binding_provenance)
    checks["candidate_provenance"] = (
        [row.get("candidate_id") for row in obj.binding_provenance] == obj.candidate_ids
        and all(row.get("category") == obj.category_name for row in obj.binding_provenance))
    _require(checks["semantic_policy"] and checks["candidate_provenance"],
             f"current semantic contract failed: {obj.obj_id}")
    return receipt.model_copy(update={
        "version": BINDING_VALIDATION_VERSION, "checks": checks,
        "files": {**files, SEMANTIC_IDENTITY_KEY: _semantic_identity(obj)},
    })


def _source_audit_contract(binding, request, qa_ids, scene_identity):
    _require(request.get("scene_audit_qa_ids") == qa_ids,
             "reuse requires the exact original full scene audit scope")
    audits = binding.stats.get("semantic_conflict_audits")
    _require(isinstance(audits, list), "original scene conflict audit receipts missing")
    for audit in audits:
        _digest(audit, "audit_sha256", "original scene conflict audit")
        _require(audit.get("scene_identity") == scene_identity
                 and bool(audit.get("qa_ids"))
                 and set(audit["qa_ids"]) <= set(qa_ids),
                 "original scene conflict audit scope changed")


def _derive(context):
    _require(isinstance(context, PreparedBindingDerivationContext),
             "explicit trusted derivation context is required")
    indexed = {str(item.episode.qa_id): item for item in context.items}
    qa_ids = sorted(indexed)
    _require(bool(indexed) and len(indexed) == len(context.items),
             "empty or duplicate destination questions")
    for mapping in (context.sources, context.target_inputs, context.target_requests,
                    context.target_artifact_paths):
        _require(set(mapping) == set(indexed), "destination scene group is incomplete")
    _require(context.cfg.mode == "real", "only real sealed inputs may be derived")
    _require(context.audit_mode in AUDIT_MODES, "unsupported source audit mode")
    reuse = context.audit_mode == "reuse_source_audit"
    if reuse:
        root = Path(__file__).resolve().parents[1]
        _require(all(file_identity(root / name) == identity
                     for name, identity in _REUSED_AUDIT_SOURCES.items()),
                 "current conflict/label rules differ from the reviewed source")
    scenes = {item.episode.scene_name for item in context.items}
    _require(len(scenes) == 1, "derive one complete destination scene group at a time")
    bindings, arrays, anchors, destinations, scene_ids = {}, {}, {}, {}, set()
    array_cache, geometry_cache = {}, {}
    for qa_id in qa_ids:
        item = indexed[qa_id]
        new_input, request = context.target_inputs[qa_id], context.target_requests[qa_id]
        _digest(new_input, "sha256", "destination input")
        _digest(request, "sha256", "destination request")
        _require(request == prepared_binding_request_identity(
            input_identity=new_input, cfg=context.cfg, scene_audit_qa_ids=qa_ids),
            "destination is not a normal current v4 prepared request")
        path = Path(context.target_artifact_paths[qa_id]).expanduser().resolve()
        relocation = (context.target_relocations or {}).get(qa_id)
        if relocation is not None:
            actual = relocation.recompute_scene_input_identity(
                episode=item.episode.model_dump(mode="json"), source=item.source,
                video_path=item.video_path, pixels=item.pixels)
            _require(path == relocation.artifact_path, "destination relocation path changed")
            art = relocation.resolved_artifact()
        else:
            actual = input_identity(item, path)
            art = load_artifact_v5(path)
        _require(actual == dict(new_input), "destination input does not recompute")
        old, old_input, manifest, old_art, anchor = _load_source(context.sources[qa_id], item)
        _legacy_contract(old, manifest["request_identity"], request)
        if reuse:
            _source_audit_contract(old, manifest["request_identity"], qa_ids, _scene(art))
        identity = old.stats["generation"]["identity"]
        values = _arrays(art, array_cache)
        geometry = _geometry(values, geometry_cache)
        _require(geometry == identity.get("geometry")
                 == _geometry(_arrays(old_art, array_cache), geometry_cache),
                 "destination geometry differs from original generation")
        _require(identity["vlm"].get("temperature") == context.cfg.temperature
                 and identity["vlm"].get("max_tokens") == context.cfg.max_tokens,
                 "original M5 temperature/token budget changed")
        _require(identity.get("frames") == [array_identity(frame) for frame in item.pixels]
                 and old_input["pixels"] == new_input["pixels"]
                 and identity.get("question") == question_input(
                     item.episode.question, runner.canonical_task(item.episode.question_type),
                     item.episode.options)
                 and identity.get("scene") == art.scene_name
                 and identity.get("frame_set_hash") == art.frame_set_hash
                 and identity.get("reconstruction_identity") == _scene(art)
                 and identity.get("frame_provenance") == runner._episode_frame_provenance(
                     item.episode, item.pixels), "destination frames/question/scene changed")
        _require(bool(old_art.quality and old_art.quality.main_gate_passed)
                 == bool(art.quality and art.quality.main_gate_passed),
                 "destination reconstruction admission changed")
        derived = copy.deepcopy(old)
        derived.prepared_measurements = {}
        _derive_policy(derived, request["parameters"])
        if reuse:
            for obj in derived.objects:
                obj.binding_validation = _reuse_validation(
                    obj, derived.stats["generation"]["sha256"])
        derived.stats["version"] = M5_BINDING_VERSION
        bindings[qa_id], arrays[qa_id], anchors[qa_id] = derived, values, anchor
        scene_ids.add(_scene(art))
        destinations[qa_id] = {
            "input_sha256": new_input["sha256"], "request_identity": copy.deepcopy(dict(request)),
        }
    _require(len(scene_ids) == 1, "destination group reconstruction identities differ")
    audited, audits = ({}, []) if reuse else audit_semantic_mask_conflicts(
        [(qa_id, bindings[qa_id].objects) for qa_id in qa_ids],
        scene_identity=next(iter(scene_ids)))
    for qa_id in qa_ids:
        binding, values = bindings[qa_id], arrays[qa_id]
        generation_sha = binding.stats["generation"]["sha256"]
        if not reuse:
            binding.objects = []
            for obj in audited[qa_id]:
                verdict = validate_binding(
                    obj, depth_maps=values["depth"], n_frames=len(indexed[qa_id].pixels),
                    generation_sha256=generation_sha)
                _require(set(verdict.errors) <= {"no_binding_flags", "instance_not_suspect"},
                         f"current array/semantic validation failed for {qa_id}/{obj.obj_id}: {verdict.errors}")
                binding.objects.append(obj.model_copy(update={"binding_validation": verdict}))
            binding.stats["semantic_conflict_audits"] = [
                audit for audit in audits if qa_id in audit["qa_ids"]]
            if binding.stats.get("grounding"):
                binding.stats["grounding"] = refresh_coverage(binding.stats["grounding"], binding.objects)
        body = {
            "schema_version": DERIVATION_SCHEMA_VERSION,
            "source_commit": LEGACY_SOURCE_COMMIT, "semantic_policy": "category-only",
            "source": anchors[qa_id], "scene_sources": anchors,
            "destination": destinations[qa_id], "scene_destinations": destinations,
            "implementation": _implementation(),
            "scene_audit_qa_ids": qa_ids,
            "output_binding_sha256": content_sha256(_content(binding)),
            "new_model_requests": 0, "new_detector_requests": 0, "new_sam2_tracks": 0,
            "generation_resealed": False, "quality_assessed": False,
            "audit_mode": context.audit_mode,
            "source_audit_reused": reuse,
            "new_mask_conflict_audits": 0 if reuse else 1,
            "new_numeric_validations": 0 if reuse else len(binding.objects),
        }
        binding.stats["binding_derivation"] = {**body, "sha256": content_sha256(body)}
    return bindings


def derive_prepared_binding_scene_v11(context: PreparedBindingDerivationContext):
    """Return new bindings, or raise; this API never requests perception fallback."""
    try:
        _require(isinstance(context, PreparedBindingDerivationContext),
                 "explicit trusted derivation context is required")
        if context._replay_cache:
            _check_cache(context)
        fingerprint, closure = _context_fingerprint(context), _closure_stamps(context)
        bindings = _derive(context)
        _require(fingerprint == _context_fingerprint(context)
                 and all(_stamp(path) == stamp for path, stamp in closure.items()),
                 "derivation closure/context changed during validation")
        context._replay_cache.update(
            fingerprint=fingerprint, closure=closure,
            expected={qa: copy.deepcopy(binding.stats["binding_derivation"])
                      for qa, binding in bindings.items()})
        return bindings
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RelocationError) as exc:
        raise PreparedBindingStoreError(f"binding derivation rejected: {exc}") from exc


def publish_derived_binding_scene_v11(
    context: PreparedBindingDerivationContext, *, store_root, publish_qa_ids=None,
):
    """Publish into a new store and return the normal current store references."""
    store = PreparedBindingStore(store_root)
    _require(all(store.root != PreparedBindingStore(source["reference"]["store_root"]).root
                 for source in context.sources.values()), "destination store must be new")
    ids = list(context.sources) if publish_qa_ids is None else list(publish_qa_ids)
    _require(bool(ids) and len(set(ids)) == len(ids) and set(ids) <= set(context.sources),
             "publication QA selection must be a nonempty subset of the full scene")
    publish_ids = set(ids)
    # Derive and audit the full source scene even when only selected experiment
    # outputs are persisted. Every persisted receipt retains the full closure.
    bindings = derive_prepared_binding_scene_v11(context)
    receipts = {
        qa: (context.target_migration_receipts or {}).get(qa) or _source_receipt(context.sources[qa])
        for qa in bindings if qa in (context.target_relocations or {})
    }
    _require(all(receipts.values()), "relocated publication requires explicit scene receipts")

    def publish(qa_id, binding):
        request = context.target_requests[qa_id]
        key = store.key_for(request)
        if (store.entry_path(key) / "manifest.json").is_file():
            reference = store.reference(key)
            _require(reference["request_identity"] == request,
                     "existing derived request changed")
            migration = None
            destination = None
            if qa_id in receipts:
                original = read_relocation_receipt(receipts[qa_id])
                destination = original["destination_root"]
                migration = store.create_migration_receipt(
                    key, scene_input_identity=context.target_inputs[qa_id],
                    destination_root=destination,
                    scene_artifact_ref=original["artifact"]["target_ref"],
                    dependency_refs={
                        name: row["target_ref"]
                        for name, row in original["dependencies"].items()})
            existing = load_prepared_binding_reference(
                reference, scene_input_identity=context.target_inputs[qa_id],
                scene_artifact=context.target_artifact_paths[qa_id],
                migration_receipt=migration, migration_destination_root=destination)
            _require(existing.stats.get("binding_derivation")
                     == binding.stats.get("binding_derivation")
                     and content_sha256(_content(existing)) == content_sha256(_content(binding)),
                     "existing derived output differs from the full scene audit")
            return reference
        return store.publish(
            request_identity=context.target_requests[qa_id],
            scene_artifact=context.target_artifact_paths[qa_id],
            scene_input_identity=context.target_inputs[qa_id], binding=binding,
            scene_migration_receipt=receipts.get(qa_id))

    return {qa_id: publish(qa_id, binding)
            for qa_id, binding in bindings.items() if qa_id in publish_ids}


def validate_derived_prepared_binding(binding, *, context, frames, depth, c2w, intrinsics,
                                     point_conf, depth_conf, point_map, inputs,
                                     scene_identity, seed=None, model_weights_sha256="",
                                     parameters=None):
    """Replay against caller-pinned sources and compare every output field/file."""
    _require(isinstance(context, PreparedBindingDerivationContext),
             "explicit trusted derivation context is required")
    receipt = binding.stats.get("binding_derivation")
    _digest(receipt, "sha256", "derivation receipt")
    _require(receipt.get("schema_version") == DERIVATION_SCHEMA_VERSION,
             "unsupported derivation schema")
    qa_id = receipt["destination"]["request_identity"]["qa_id"]
    _require(qa_id in context.target_requests, "destination question is not trusted")
    request = context.target_requests[qa_id]
    _require(receipt["destination"]["request_identity"] == request,
             "destination request differs from trusted context")
    identity = binding.stats["generation"]["identity"]
    geometry = {name: array_identity(value) for name, value in (
        ("depth", depth), ("c2w", c2w), ("intrinsics", intrinsics),
        ("point_conf", point_conf), ("depth_conf", depth_conf), ("point_map", point_map))}
    _require(identity["geometry"] == geometry
             and identity["frames"] == [array_identity(frame) for frame in frames]
             and identity["question"] == inputs
             and identity["reconstruction_identity"] == scene_identity,
             "consumer arrays/question/scene changed")
    _require(parameters is not None and parameters == request["parameters"],
             "consumer must supply exact current parameters")
    _require(seed is None or seed == request["seed"], "consumer M5 seed changed")
    _require(not model_weights_sha256
             or model_weights_sha256 == request["model"]["model_weights_sha256"],
             "consumer model weights changed")
    if not _check_cache(context):
        derive_prepared_binding_scene_v11(context)
    expected_receipt = context._replay_cache["expected"][qa_id]
    _require(receipt == expected_receipt,
             "source, policy, implementation, audit or output seal changed")
    if context.audit_mode != "reuse_source_audit":
        for obj in binding.objects:
            verdict = validate_binding(
                obj, depth_maps=depth, n_frames=len(frames),
                generation_sha256=binding.stats["generation"]["sha256"])
            _require(verdict == obj.binding_validation,
                     f"consumer current validation receipt changed: {obj.obj_id}")
    _require(content_sha256(_content(binding)) == expected_receipt["output_binding_sha256"],
             "derived binding output/files changed")


__all__ = [
    "DERIVATION_SCHEMA_VERSION", "PreparedBindingDerivationContext",
    "derive_prepared_binding_scene_v11", "publish_derived_binding_scene_v11",
    "validate_derived_prepared_binding",
]
