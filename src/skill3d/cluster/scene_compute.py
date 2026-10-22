"""Frozen scene preparation, without dataset access, solving, or scoring.

``input_root/scene.json`` uses ``harness3d-frozen-scene/1`` and a ``files``
mapping from original absolute filenames to input-root-relative transferred
files. Original video filenames are metadata only; they are never opened.

``method_config`` accepts ``runtime_config`` (a public runtime dict or an
input-relative public YAML file),
``metric_enabled`` (default False), ``vggt_checkpoint``, ``moge2_checkpoint``,
``device`` (default ``cuda:0``), ``max_questions_per_scene`` (at most eight),
and ``min_free_gib`` (default 80). Checkpoints must already be local; their
source/model freeze is the caller's responsibility. The default perception
client uses main's authenticated RPC, not a worker-side Bailian credential.
An optional keyword-only ``client_factory(**bailian_fields)`` supplies that
same interface and must honor the supplied frozen operating point.

All inference imports are inside functions. The returned validation summary
and output inventory are JSON values, not model instances. The caller owns
job durability, input authentication, publication, and subsequent relocation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable

FROZEN_SCENE_SCHEMA = "harness3d-frozen-scene/1"
SCENE_PREPARATION_SCHEMA = "harness3d-scene-preparation/1"
MAX_QUESTIONS_PER_SCENE = 8
_LABEL_FIELDS = {
    "ground_truth", "answer", "reference_answer", "reference_answers",
    "correct_answer", "labels",
}


class SceneComputeError(ValueError):
    """Frozen inputs, configuration, or published outputs fail validation."""


def _unique_pairs(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise SceneComputeError(f"duplicate JSON field: {key}")
        value[key] = child
    return value


def _read_json(path: Path) -> dict:
    def invalid_constant(value):
        raise SceneComputeError(f"non-finite frozen JSON value: {value}")

    value = json.loads(path.read_bytes(), object_pairs_hook=_unique_pairs,
                       parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise SceneComputeError(f"expected a JSON object: {path.name}")
    return value


def _no_symlinks(path: Path) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise SceneComputeError(f"symlink is forbidden: {path}")


def _root(path: Path) -> Path:
    path = Path(path).absolute()
    _no_symlinks(path)
    if not path.is_dir():
        raise SceneComputeError(f"input root is not a directory: {path}")
    return path.resolve()


def _reference(value: object, *, absolute: bool = False) -> str:
    if (not isinstance(value, str) or not value or "\\" in value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise SceneComputeError("invalid transfer reference")
    path = PurePosixPath(value)
    if (str(path) != value or path.is_absolute() != absolute or value == "/"
            or value.startswith("//")
            or any(p in {".", ".."} or ":" in p or p.startswith("~")
                   for p in value.split("/") if p)):
        raise SceneComputeError(f"unsafe transfer reference: {value}")
    return value


def _local_file(root: Path, reference: object) -> Path:
    path = root / _reference(reference)
    _no_symlinks(path)
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise SceneComputeError(f"missing or escaping transferred file: {reference}")
    return path.resolve()


def _digest(value: object) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise SceneComputeError("expected a lowercase SHA256")
    return value


def _file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "size": size}


def _reject_labels(value, path=()) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            location = (*path, key)
            if key in _LABEL_FIELDS:
                compatibility_field = (
                    len(location) == 5 and location[0] == "questions"
                    and location[2:] == (
                        "logical_input_identity", "episode", "ground_truth",
                    )
                )
                # The only label-shaped input allowed is the saved empty GT.
                if not compatibility_field or child != "":
                    raise SceneComputeError("frozen scene contains a forbidden label field")
            _reject_labels(child, location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_labels(child, (*path, index))


def _question_cap(method_config: dict) -> int:
    value = method_config.get("max_questions_per_scene", MAX_QUESTIONS_PER_SCENE)
    if type(value) is not int or not 1 <= value <= MAX_QUESTIONS_PER_SCENE:
        raise SceneComputeError("max_questions_per_scene must be an integer in [1, 8]")
    return value


@dataclass
class _FrozenScene:
    root: Path
    manifest: dict
    items: list[Any]
    receipts: dict[str, dict]
    artifact: Any
    summary: dict


def _load_scene(input_root: Path, *, question_cap: int) -> _FrozenScene:
    from io import BytesIO

    import numpy as np
    from PIL import Image

    from skill3d.adapters.episode_source import EpisodeItem
    from skill3d.adapters.frame_set import (
        PREPROCESSING_VERSION, frame_set_hash, uniform_frame_ids, with_pixel_identity,
    )
    from skill3d.schemas import VSIBenchEpisode
    from skill3d.storage.relocation import (
        create_relocation_receipt, validate_relocation_receipt,
    )

    root = _root(input_root)
    scene_path = _local_file(root, "scene.json")
    manifest = _read_json(scene_path)
    _reject_labels(manifest)
    if (manifest.get("schema") != FROZEN_SCENE_SCHEMA
            or manifest.get("status") != "frozen"
            or manifest.get("real_labels_available") is not False):
        raise SceneComputeError("scene must be frozen with real_labels_available=false")
    scene_id, split = manifest.get("scene_id"), manifest.get("split")
    if (not isinstance(scene_id, str) or not scene_id
            or split not in {"induction", "inner_validation", "outer_holdout"}):
        raise SceneComputeError("invalid frozen scene/split")
    qa_ids, questions = manifest.get("qa_ids"), manifest.get("questions")
    if (not isinstance(qa_ids, list) or not qa_ids
            or any(not isinstance(q, str) or not q for q in qa_ids)
            or qa_ids != sorted(set(qa_ids))
            or not isinstance(questions, list) or len(questions) != len(qa_ids)):
        raise SceneComputeError("invalid complete scene QA group")
    if len(qa_ids) > question_cap:
        raise SceneComputeError(f"scene question cap {question_cap} exceeded")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SceneComputeError("scene requires a files transfer mapping")
    for source, target in files.items():
        _reference(source, absolute=True)
        _local_file(root, target)
    transferred = manifest.get("file_identities")
    if transferred is not None:
        if not isinstance(transferred, dict) or set(transferred) != set(files.values()):
            raise SceneComputeError("transferred file identities do not cover the files mapping")
        for reference, expected in transferred.items():
            if _file_identity(_local_file(root, reference)) != expected:
                raise SceneComputeError("transferred file integrity changed")
    frame_rows = manifest.get("frames")
    if not isinstance(frame_rows, list) or len(frame_rows) != 32:
        raise SceneComputeError("frozen scene requires exactly 32 frames")
    pixels, frame_sources = [], []
    for row in frame_rows:
        if (not isinstance(row, dict) or set(row) != {
                "source_path", "png_sha256", "size", "rgb_sha256", "shape", "dtype"}):
            raise SceneComputeError("invalid frozen frame record")
        source = _reference(row["source_path"], absolute=True)
        if source not in files:
            raise SceneComputeError("missing frozen frame transfer mapping")
        path = _local_file(root, files[source])
        raw = path.read_bytes()
        if (type(row["size"]) is not int or row["size"] != len(raw)
                or hashlib.sha256(raw).hexdigest() != _digest(row["png_sha256"])):
            raise SceneComputeError("frozen PNG integrity changed")
        with Image.open(BytesIO(raw)) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise SceneComputeError("frozen image must be an RGB PNG")
            array = np.array(image)
        if (list(array.shape) != row["shape"] or str(array.dtype) != row["dtype"]
                or array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3
                or hashlib.sha256(array.tobytes()).hexdigest() != _digest(row["rgb_sha256"])):
            raise SceneComputeError("frozen RGB integrity changed")
        pixels.append(array)
        frame_sources.append(source)
    if (len(set(frame_sources)) != 32
            or len({files[source] for source in frame_sources}) != 32):
        raise SceneComputeError("duplicate frozen frame source")

    items = []
    shared_metadata = None
    for qa_id, question in zip(qa_ids, questions):
        if (not isinstance(question, dict) or set(question) - {
                "episode", "source", "video_path", "logical_input_identity"}):
            raise SceneComputeError("invalid public frozen question")
        public = question.get("episode")
        if not isinstance(public, dict) or "ground_truth" in public:
            raise SceneComputeError("public frozen episode must not contain ground_truth")
        ep = VSIBenchEpisode.model_validate({**public, "ground_truth": ""})
        if (ep.qa_id != qa_id or ep.scene_name != scene_id or ep.split != split
                or not ep.dataset or not ep.question):
            raise SceneComputeError("frozen question order/scene metadata changed")
        if question.get("source") != "vsi_bench":
            raise SceneComputeError("unsupported frozen source")
        video_path = _reference(question.get("video_path"), absolute=True)
        fs = ep.frame_set
        if (fs is None or fs.n_frames != 32 or fs.n_total_frames < 32
                or fs.decode_status != "ok" or fs.readable_frame_ids != fs.frame_ids
                or len(fs.frame_ids) != 32
                or fs.frame_set_hash != frame_set_hash(fs.frame_ids)
                or not math.isfinite(fs.fps) or fs.fps < 0
                or fs.frame_ids != uniform_frame_ids(fs.n_total_frames, 32)
                or fs.source_frame_indices != fs.frame_ids
                or fs.timestamps != [float(i) / fs.fps if fs.fps > 0 else float(i)
                                     for i in fs.frame_ids]
                or fs.episode_id != qa_id or fs.dataset_id != ep.dataset
                or fs.scene_name != scene_id or not fs.video_id
                or fs.preprocessing_version != PREPROCESSING_VERSION
                or fs.frame_refs and fs.frame_refs != frame_sources):
            raise SceneComputeError("frozen frame source metadata changed")
        actual_fs = with_pixel_identity(fs, pixels)
        if not fs.pixel_sha256 or actual_fs.pixel_sha256 != fs.pixel_sha256:
            raise SceneComputeError("frozen FrameSet pixel identity missing or changed")
        if len(ep.frames) != 32 or any(
                frame.frame_idx != slot or frame.timestamp != float(slot)
                or (frame.height, frame.width) != image.shape[:2]
                for slot, (frame, image) in enumerate(zip(ep.frames, pixels))):
            raise SceneComputeError("frozen episode frame order/shape changed")
        metadata = (ep.dataset, video_path, actual_fs.cache_identity(),
                    fs.model_dump(mode="json", exclude={"episode_id"}))
        if shared_metadata is not None and metadata != shared_metadata:
            raise SceneComputeError("one scene group contains different frozen sources")
        shared_metadata = metadata
        items.append(EpisodeItem(ep, pixels, source=question["source"], video_path=video_path))

    receipts, artifact = {}, None
    source_artifact = manifest.get("artifact_source_path")
    if source_artifact is not None:
        _reference(source_artifact, absolute=True)
        if source_artifact not in files:
            raise SceneComputeError("missing artifact transfer mapping")
        for item, question in zip(items, questions):
            logical = question.get("logical_input_identity")
            if not isinstance(logical, dict) or logical.get("artifact_path") != source_artifact:
                raise SceneComputeError("existing geometry requires every question's logical identity")
            dependencies = {}
            for name, record in logical.get("files", {}).items():
                if record["path"] not in files:
                    raise SceneComputeError(f"missing geometry transfer mapping: {name}")
                dependencies[name] = files[record["path"]]
            receipt = create_relocation_receipt(
                logical_input_identity=logical, destination_root=root,
                artifact_ref=files[source_artifact], dependency_refs=dependencies,
            )
            resolver = validate_relocation_receipt(
                receipt, expected_input_sha256=logical["sha256"], destination_root=root,
                expected_source_artifact=source_artifact,
            )
            resolver.recompute_scene_input_identity(
                episode=item.episode.model_dump(mode="json"), source=item.source,
                video_path=item.video_path, pixels=item.pixels,
            )
            resolved = resolver.resolved_artifact()
            if resolved.reconstruction_input_identity != item.episode.frame_set.cache_identity():
                raise SceneComputeError("frozen reconstruction source identity changed")
            artifact = resolved
            receipts[item.episode.qa_id] = receipt
    elif any("logical_input_identity" in question for question in questions):
        raise SceneComputeError("logical identity without existing geometry")
    summary = {
        "schema": FROZEN_SCENE_SCHEMA, "status": "validated",
        "scene_id": scene_id, "split": split, "qa_ids": qa_ids,
        "n_questions": len(items), "n_frames": 32,
        "scene_json": _file_identity(scene_path),
        "reconstruction_input_identity": items[0].episode.frame_set.cache_identity(),
        "geometry_source": "relocated" if artifact is not None else "reconstruct",
        "logical_input_sha256": {q: r["logical_input_identity"]["sha256"]
                                 for q, r in receipts.items()},
        "real_labels_available": False,
    }
    return _FrozenScene(root, manifest, items, receipts, artifact, summary)


def validate_scene_input(input_root: Path, *, max_questions_per_scene: int = 8) -> dict:
    """CPU-only replay of frozen pixels, metadata and transferred geometry."""
    cap = _question_cap({"max_questions_per_scene": max_questions_per_scene})
    return _load_scene(input_root, question_cap=cap).summary


def _runtime_config(scene: _FrozenScene, output_root: Path, method_config: dict):
    from skill3d.online.config import (
        active_vision_from, load_config, online_model_from, resolve_runtime,
        retrieval_policy_from, sandbox_from,
    )
    from skill3d.online.runner import OnlineRunConfig

    reference = method_config.get("runtime_config", scene.manifest.get("runtime_config"))
    if isinstance(reference, dict):
        import tempfile
        import yaml

        # Use the existing strict loader, without changing the frozen input tree.
        with tempfile.TemporaryDirectory(prefix="skill3d-public-runtime-") as directory:
            path = Path(directory) / "runtime.yaml"
            path.write_text(yaml.safe_dump(reference), encoding="utf-8")
            raw = resolve_runtime(load_config(path), formal=True)
        runtime_identity = {
            "kind": "public_dict", "sha256": hashlib.sha256(json.dumps(
                reference, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode()).hexdigest(),
        }
    else:
        path = _local_file(scene.root, reference)
        raw = resolve_runtime(load_config(path), formal=True)
        runtime_identity = {"reference": reference, **_file_identity(path)}
    model, sandbox, vision = online_model_from(raw), sandbox_from(raw), active_vision_from(raw)
    if not model.enabled:
        raise SceneComputeError("M5 preparation requires the public online model to be enabled")
    cfg = OnlineRunConfig(
        mode="real", seed=raw["seed"], online_model_enabled=model.enabled,
        online_model_provider=model.provider, online_model=model.model,
        online_model_revision=model.model_revision, online_base_url=model.base_url,
        online_api_key_env=model.api_key_env, online_timeout_s=model.timeout_s,
        semantic_min_pixels=model.audit_min_pixels, semantic_max_pixels=model.audit_max_pixels,
        semantic_relation_batch_size=model.relation_batch_size,
        semantic_review_config=model.mask_review, max_images=model.n_frames,
        max_pixels=model.max_pixels, max_model_len=model.max_model_len,
        max_tokens=model.max_tokens, temperature=model.temperature,
        cell_timeout_s=sandbox.cell_timeout_s, max_regen=sandbox.max_regenerate,
        max_solver_rounds=raw["max_solver_rounds"],
        max_retries_per_operation=raw["max_retries_per_operation"],
        finalization_rounds=raw["finalization_rounds"],
        eval_visual_fallback=raw["eval_visual_fallback"],
        input_diagnostics=raw.get("input_diagnostics", False),
        image_layout=vision.layout, max_derived_images=vision.max_derived_images,
        retrieval_policy=retrieval_policy_from(raw),
        trace_dir=str(output_root / "unused_traces"),
        recon_dir=str(output_root / "geometry"), work_dir=str(output_root / "work"),
    )
    return cfg, runtime_identity


def _checkpoint(method_config: dict, key: str, env: str, default=None) -> str:
    value = method_config.get(key) or os.environ.get(env) or default
    if not isinstance(value, (str, Path)) or not value:
        raise SceneComputeError(f"{key} must name an existing offline checkpoint")
    path = Path(value).expanduser()
    if key == "moge2_checkpoint" and path.is_dir():
        path = path / "model.pt"
    if not path.exists() or not (path.is_file() or path.is_dir()):
        raise SceneComputeError(f"{key} must name an existing offline checkpoint")
    return str(path.resolve())


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _write_artifact(path: Path, artifact) -> None:
    # Schema-authorized NaN quality observations must not become invented zeros.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _reject_infrastructure_failures(stats) -> None:
    """Transport outages are retryable task failures, not perception evidence."""
    outage = re.compile(
        r"HTTP(?:Error)?(?:\s+Error)?[\s:]*(?:401|403|408|429|5\d\d)\b"
        r"|ServiceUnavailable|Connection(?:Refused)?Error|URLError"
        r"|TimeoutError|ReadTimeout|ConnectTimeout"
        r"|perception transport unavailable|invalid perception response evidence"
        r"|frozen text model or request seed unavailable",
        re.IGNORECASE,
    )

    def visit(value, *, error_context=False):
        if isinstance(value, dict):
            for key, child in value.items():
                field = str(key).lower()
                is_error = error_context or "error" in field or field in {
                    "exception", "failure_reason", "transport_failure",
                }
                if (field in {"http_status", "status_code", "status"} and type(child) is int
                        and (child in {401, 403, 408, 429} or 500 <= child <= 599)):
                    raise SceneComputeError("perception infrastructure failure in prepared binding")
                visit(child, error_context=is_error)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, error_context=error_context)
        elif error_context and isinstance(value, str) and outage.search(value):
            raise SceneComputeError("perception infrastructure failure in prepared binding")

    visit(stats)


def _prepare_variant(scene: _FrozenScene, cfg, output: Path, name: str, artifact,
                     *, cap: int, min_free_gib: float, client_factory: Callable) -> dict:
    from skill3d.evaluation.skill_ablation_v11 import input_identity
    from skill3d.evolution.prepared_bindings_v11 import prepare_shared_bindings_v11
    from skill3d.schemas import ReconstructionArtifact
    from skill3d.segmentation.prepared_binding_store import (
        OBJECT_REFS, PreparedBindingStore, load_prepared_binding_reference,
    )
    from skill3d.storage.relocation import artifact_dependency_refs
    for reference in artifact_dependency_refs(artifact).values():
        path = Path(reference)
        _no_symlinks(path)
        if not path.is_absolute() or not path.is_file() or not path.is_relative_to(output):
            raise SceneComputeError("new artifact dependency is not output-local")
    path = output / "geometry" / name / "artifact.json"
    _write_artifact(path, artifact)
    paths = {item.episode.qa_id: str(path) for item in scene.items}

    def recompute(item, artifact_path):
        if Path(artifact_path) != path:
            raise SceneComputeError("preparation changed the scene artifact path")
        return input_identity(item, artifact_path)

    identities = {item.episode.qa_id: recompute(item, path) for item in scene.items}

    def identify(item, artifact_path):
        actual = recompute(item, artifact_path)
        if actual != identities[item.episode.qa_id]:
            raise SceneComputeError("new scene input identity changed during preparation")
        return actual

    def load_artifact(artifact_path):
        return ReconstructionArtifact.model_validate_json(Path(artifact_path).read_bytes())

    store_root = output / "prepared" / name
    prepared = prepare_shared_bindings_v11(
        scene.items, artifact_paths=paths, base_cfg=cfg, store_root=store_root,
        client_factory=client_factory, max_questions_per_scene=cap,
        min_free_gib=min_free_gib, identity_factory=identify, artifact_loader=load_artifact,
    )
    qa_ids, scene_id = scene.summary["qa_ids"], scene.summary["scene_id"]
    if (prepared.get("n_questions") != len(qa_ids) or prepared.get("n_scenes") != 1
            or prepared.get("scene_groups") != {scene_id: qa_ids}
            or set(prepared.get("bindings", {})) != set(qa_ids)
            or prepared.get("store_root") != str(store_root)):
        raise SceneComputeError("prepared manifest omitted or changed the full scene QA group")
    outcomes = {}
    for qa_id, reference in prepared["bindings"].items():
        if (reference.get("store_root") != str(store_root)
                or reference.get("scene_artifact_ref") != str(path)
                or reference.get("request_identity", {}).get("scene_audit_qa_ids") != qa_ids):
            raise SceneComputeError("prepared binding changed the full scene audit group")
        identity = identify(next(i for i in scene.items if i.episode.qa_id == qa_id), path)
        binding = load_prepared_binding_reference(reference, scene_input_identity=identity)
        _reject_infrastructure_failures(binding.stats)
        for obj in binding.objects:
            for field in OBJECT_REFS:
                ref = getattr(obj, field)
                if ref and not Path(ref).is_relative_to(output):
                    raise SceneComputeError("prepared object dependency is not output-local")
        stats = binding.stats
        outcomes[qa_id] = {
            "status": stats.get("status") or (
                "binding_unavailable" if not binding.materialized or stats.get("binding_error")
                else "prepared"),
            "materialized": binding.materialized, "n_objects": len(binding.objects),
            "binding_error": stats.get("binding_error"),
        }
        store = PreparedBindingStore(store_root)
        _write_json(output / "input_identities" / name / f"{reference['key']}.json", identity)
        if store.manifest(reference["key"])["scene_input_identity_sha256"] != identity["sha256"]:
            raise SceneComputeError("published input identity changed")
    _write_json(store_root / "prepare_manifest.json", prepared)
    return {
        "artifact": path.relative_to(output).as_posix(),
        "binding_manifest": (store_root / "prepare_manifest.json").relative_to(output).as_posix(),
        "artifact_identity": _file_identity(path),
        "artifact_dependencies": {
            key: {"path": ref, **_file_identity(Path(ref))}
            for key, ref in artifact_dependency_refs(artifact).items()
        },
        "quality_status": artifact.quality_status,
        "scale_fusion_status": artifact.scale_fusion_status,
        "metric_status": artifact.scale_fusion_status,
        "metric_model": artifact.metric_model,
        "preparation": prepared, "outcomes": outcomes,
        "input_identities": identities,
        "input_identity_refs": {
            qa_id: f"input_identities/{name}/{reference['key']}.json"
            for qa_id, reference in prepared["bindings"].items()
        },
        "source_provenance": "source_provenance.json",
    }


def _default_client_factory(cfg, factory: Callable | None = None) -> Callable:
    if factory is None:
        from skill3d.cluster.perception_client import RemoteBailianClient

        factory = RemoteBailianClient

    def create(_qa_id):
        return factory(
            model=cfg.online_model, model_revision=cfg.online_model_revision,
            base_url=cfg.online_base_url, api_key_env=cfg.online_api_key_env,
            timeout_s=cfg.online_timeout_s,
            temperature=cfg.temperature, max_output_tokens=cfg.max_tokens,
        )

    return create


def _checked_client_factory(factory: Callable, cfg) -> Callable:
    expected = {
        "provider": cfg.online_model_provider, "model": cfg.online_model,
        "model_revision": cfg.online_model_revision,
        "base_url": cfg.online_base_url.rstrip("/"), "timeout_s": cfg.online_timeout_s,
        "temperature": cfg.temperature, "max_output_tokens": cfg.max_tokens,
        "endpoint_hash": hashlib.sha256(cfg.online_base_url.rstrip("/").encode()).hexdigest(),
    }

    def create(qa_id):
        client = factory(qa_id)
        if any(getattr(client, key, None) != value for key, value in expected.items()):
            raise SceneComputeError("perception client differs from the frozen runtime contract")
        return client

    return create


def _output_summary(output: Path, manifest: dict, inventory: dict) -> dict:
    return {
        "schema": SCENE_PREPARATION_SCHEMA, "status": "prepared",
        "scene_id": manifest["scene_id"], "qa_ids": manifest["qa_ids"],
        "validation": manifest["validation"], "scene_groups": manifest["scene_groups"],
        "variants": {
            name: {key: value for key, value in result.items()
                   if key not in {"input_identities", "artifact_dependencies"}}
            for name, result in manifest["variants"].items()
        },
        "prepare_manifest": {"path": "prepare_manifest.json",
                             **inventory["prepare_manifest.json"]},
        "inventory": {"path": "inventory.json", **_file_identity(output / "inventory.json")},
        "files": {**inventory, "inventory.json": _file_identity(output / "inventory.json")},
        "solver_ran": False, "scoring_ran": False, "real_labels_available": False,
        "no_solver": True, "no_labels": True,
    }


def validate_scene_output(output_root: Path) -> dict:
    """Recheck a completed local attempt for transport recovery, without models."""
    output = _root(output_root)
    inventory = _read_json(_local_file(output, "inventory.json"))
    if (inventory.get("schema") != "harness3d-scene-output-inventory/1"
            or not isinstance(inventory.get("files"), dict)
            or "prepare_manifest.json" not in inventory["files"]
            or "inventory.json" in inventory["files"]):
        raise SceneComputeError("invalid scene output inventory")
    files = inventory["files"]
    for reference, expected in files.items():
        if _file_identity(_local_file(output, reference)) != expected:
            raise SceneComputeError(f"scene output integrity changed: {reference}")
    actual_files = set()
    for path in output.rglob("*"):
        _no_symlinks(path)
        if path.is_file():
            actual_files.add(path.relative_to(output).as_posix())
    wrappers = actual_files & {"scene_result.json", "publication.json"}
    if actual_files != set(files) | {"inventory.json"} | wrappers:
        raise SceneComputeError("scene output inventory is incomplete")
    manifest = _read_json(output / "prepare_manifest.json")
    if (manifest.get("schema") != SCENE_PREPARATION_SCHEMA
            or manifest.get("status") != "prepared"
            or manifest.get("solver_ran") is not False
            or manifest.get("scoring_ran") is not False
            or manifest.get("real_labels_available") is not False):
        raise SceneComputeError("invalid completed scene preparation")
    summary = _output_summary(output, manifest, files)
    if "scene_result.json" in wrappers and _read_json(output / "scene_result.json") != summary:
        raise SceneComputeError("worker scene result differs from validated outputs")
    if "publication.json" in wrappers:
        from skill3d.cluster.publication import PublicationReceipt

        receipt = PublicationReceipt.from_dict(_read_json(output / "publication.json"))
        published = {row.name: {"sha256": row.sha256, "size": row.size_bytes}
                     for row in receipt.artifacts}
        expected = {ref: _file_identity(_local_file(output, ref))
                    for ref in actual_files - {"publication.json"}}
        if published != expected:
            raise SceneComputeError("worker publication does not cover the complete scene output")
    return summary


def compute_scene(input_root: Path, output_root: Path, method_config: dict, *,
                  client_factory: Callable[..., object] | None = None) -> dict:
    """Prepare exactly one complete frozen scene and inventory its local outputs.

    Invalid freezes/configuration raise before inference. Model and integrity
    failures propagate to the job owner; valid failed metric receipts and
    binding-unavailable payloads retain their original statuses. Use a fresh
    output directory per attempt; no originals or sealed entries are rewritten.
    An already completed, matching attempt is validated for transport recovery
    without invoking models. Partial attempts are never silently overwritten.
    """
    if not isinstance(method_config, dict):
        raise SceneComputeError("method_config must be a mapping")
    cap = _question_cap(method_config)
    enabled = method_config.get("metric_enabled", False)
    reserve = method_config.get("min_free_gib", 80.0)
    if type(enabled) is not bool:
        raise SceneComputeError("metric_enabled must be boolean")
    if (type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve < 0):
        raise SceneComputeError("min_free_gib must be finite and non-negative")
    scene = _load_scene(input_root, question_cap=cap)
    output = Path(output_root).absolute()
    _no_symlinks(output)
    output = output.resolve()
    if output.is_relative_to(scene.root) or scene.root.is_relative_to(output):
        raise SceneComputeError("input and output roots must not overlap")
    cfg, runtime_identity = _runtime_config(scene, output, method_config)
    method_sha256 = hashlib.sha256(json.dumps(
        method_config, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        recovered = validate_scene_output(output)
        previous = _read_json(output / "prepare_manifest.json")
        if (previous["validation"] != scene.summary
                or previous.get("runtime_config") != runtime_identity
                or previous.get("method_config_sha256") != method_sha256):
            raise SceneComputeError("completed attempt has different frozen inputs/configuration")
        return recovered
    if os.environ.get("SKILL3D_ALLOW_HF_DOWNLOAD"):
        raise SceneComputeError("automatic HF downloads must be disabled for scene preparation")

    from skill3d.reconstruction import metric_evidence as me
    from skill3d.reconstruction.vggt_runner import VGGT_CHECKPOINT, run_vggt

    vggt_checkpoint = None if scene.artifact is not None else _checkpoint(
        method_config, "vggt_checkpoint", "SKILL3D_VGGT_CHECKPOINT", VGGT_CHECKPOINT)
    moge_checkpoint = _checkpoint(
        method_config, "moge2_checkpoint", "SKILL3D_MOGE2_CHECKPOINT") if enabled else None
    make_client = _checked_client_factory(_default_client_factory(cfg, client_factory), cfg)
    output.mkdir(parents=True, exist_ok=True)
    import shutil

    if shutil.disk_usage(output).free / 1024 ** 3 < reserve:
        raise SceneComputeError("free disk is below the runtime reserve")
    base_dir = output / "geometry" / "base"
    first = scene.items[0]
    provenance = {
        "schema": "harness3d-scene-source-provenance/1",
        "geometry_source": scene.summary["geometry_source"],
        "reconstruction_input_identity": scene.summary["reconstruction_input_identity"],
        "logical_input_sha256": scene.summary["logical_input_sha256"],
        "original_artifact": None, "original_dependencies": {},
    }
    if scene.artifact is not None:
        if scene.artifact.metric_model not in (None, "none"):
            raise SceneComputeError("existing base geometry must not contain metric augmentation")
        receipt = next(iter(scene.receipts.values()))
        provenance["original_artifact"] = {
            "source_path": receipt["artifact"]["source_path"], **receipt["artifact"]["identity"],
        }
        provenance["original_dependencies"] = {
            name: {"source_path": row["source_path"], **row["identity"]}
            for name, row in receipt["dependencies"].items()
        }
        base = me.copy_geometry(scene.artifact, base_dir)
        for name in me.GEOMETRY_REFS:
            if name in receipt["dependencies"] and _file_identity(
                    Path(getattr(base, name))) != receipt["dependencies"][name]["identity"]:
                raise SceneComputeError("base copy differs from verified source geometry")
    else:
        base = run_vggt(
            first.pixels, first.episode.scene_name, base_dir, checkpoint=vggt_checkpoint,
            frame_set=first.episode.frame_set, compute_quality=True,
        )
    _write_json(output / "source_provenance.json", provenance)
    if base.reconstruction_input_identity != scene.summary["reconstruction_input_identity"]:
        raise SceneComputeError("new geometry changed the frozen reconstruction identity")
    variants = {
        "base": _prepare_variant(scene, cfg, output, "base", base,
                                 cap=cap, min_free_gib=float(reserve),
                                 client_factory=make_client),
    }
    if enabled:
        from skill3d.reconstruction.metric_fusion import make_moge2_model

        metric_dir = output / "geometry" / "metric"
        metric = me.copy_geometry(base, metric_dir)
        model = make_moge2_model(device=method_config.get("device", "cuda:0"),
                                 checkpoint=moge_checkpoint)
        metric = me.attach_metric_evidence(
            metric, first.pixels, first.episode.frame_set, model, metric_dir)
        del model
        me.validated_receipt(metric)
        variants["metric"] = _prepare_variant(
            scene, cfg, output, "metric", metric, cap=cap, min_free_gib=float(reserve),
            client_factory=make_client)
    manifest = {
        "schema": SCENE_PREPARATION_SCHEMA, "status": "prepared",
        "validation": scene.summary, "runtime_config": runtime_identity,
        "method_config_sha256": method_sha256,
        "scene_id": scene.summary["scene_id"], "qa_ids": scene.summary["qa_ids"],
        "scene_groups": {scene.summary["scene_id"]: scene.summary["qa_ids"]},
        "metric_enabled": enabled, "variants": variants,
        "solver_ran": False, "scoring_ran": False, "real_labels_available": False,
        "no_solver": True, "no_labels": True,
    }
    _write_json(output / "prepare_manifest.json", manifest)
    inventory = {}
    for path in sorted(output.rglob("*")):
        _no_symlinks(path)
        if path.is_file():
            inventory[path.relative_to(output).as_posix()] = _file_identity(path)
    _write_json(output / "inventory.json", {
        "schema": "harness3d-scene-output-inventory/1", "files": inventory,
    })
    return validate_scene_output(output)


def validate_published_scene(
    root: Path,
    expected_qa_ids: list[str],
    expected_scene_id: str,
    expected_method_config: dict,
) -> dict:
    """CPU replay after byte-preserving transfer to a new publication root.

    Returns per-QA migration receipts for the caller to persist as separate
    sidecars, outside the immutable output tree. Original JSON, store keys and
    binding manifests are never rewritten. Publication authentication and
    model/source freeze checks belong to the coordinator.
    """
    from skill3d.segmentation.prepared_binding_store import OBJECT_REFS, PreparedBindingStore

    output = _root(root)
    summary = validate_scene_output(output)
    manifest = _read_json(output / "prepare_manifest.json")
    qa_ids = list(expected_qa_ids)
    if (not qa_ids or qa_ids != sorted(set(qa_ids))
            or len(qa_ids) > _question_cap(expected_method_config)
            or summary["qa_ids"] != qa_ids or summary["scene_id"] != expected_scene_id
            or summary["scene_groups"] != {expected_scene_id: qa_ids}
            or summary["validation"]["qa_ids"] != qa_ids
            or summary["validation"]["scene_id"] != expected_scene_id):
        raise SceneComputeError("published scene does not match the authorized complete QA group")
    method_hash = hashlib.sha256(json.dumps(
        expected_method_config, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    enabled = expected_method_config.get("metric_enabled", False)
    expected_variants = {"base", "metric"} if enabled else {"base"}
    if (type(enabled) is not bool or manifest.get("method_config_sha256") != method_hash
            or manifest.get("metric_enabled") is not enabled
            or set(manifest["variants"]) != expected_variants):
        raise SceneComputeError("published scene method/variant configuration changed")

    cfg = None
    if isinstance(expected_method_config.get("runtime_config"), dict):
        cfg, runtime_identity = _runtime_config(
            _FrozenScene(output, {}, [], {}, None, {}), output, expected_method_config)
        if manifest["runtime_config"] != runtime_identity:
            raise SceneComputeError("published public runtime configuration changed")
    old_root = None
    validated = {}
    for name, variant in manifest["variants"].items():
        artifact_ref = _reference(variant["artifact"])
        if artifact_ref != f"geometry/{name}/artifact.json":
            raise SceneComputeError("published artifact path changed")
        _local_file(output, artifact_ref)
        binding_ref = f"prepared/{name}/prepare_manifest.json"
        if variant["binding_manifest"] != binding_ref:
            raise SceneComputeError("published binding manifest path changed")
        prepared = _read_json(_local_file(output, binding_ref))
        if (prepared != variant["preparation"] or prepared["n_questions"] != len(qa_ids)
                or prepared["n_scenes"] != 1
                or prepared["scene_groups"] != {expected_scene_id: qa_ids}
                or set(prepared["bindings"]) != set(qa_ids)
                or set(variant["input_identity_refs"]) != set(qa_ids)
                or set(variant["input_identities"]) != set(qa_ids)
                or set(variant["outcomes"]) != set(qa_ids)):
            raise SceneComputeError("published preparation omitted part of the scene QA group")
        store = PreparedBindingStore(output / "prepared" / name)
        receipts, outcomes = {}, {}
        resolved_artifact = None
        for qa_id, reference in prepared["bindings"].items():
            identity_ref = _reference(variant["input_identity_refs"][qa_id])
            if identity_ref != f"input_identities/{name}/{reference['key']}.json":
                raise SceneComputeError("published input identity path changed")
            identity = _read_json(_local_file(output, identity_ref))
            ep = identity["episode"]
            if (identity != variant["input_identities"][qa_id] or ep["qa_id"] != qa_id
                    or ep["scene_name"] != expected_scene_id or ep.get("ground_truth") != ""):
                raise SceneComputeError("published question identity or constant empty GT changed")
            _reject_labels({**identity, "episode": {
                key: value for key, value in ep.items() if key != "ground_truth"
            }})
            original_artifact = PurePosixPath(_reference(identity["artifact_path"], absolute=True))
            relative_artifact = PurePosixPath(artifact_ref)
            try:
                candidate_root = original_artifact.parents[len(relative_artifact.parts) - 1]
                if original_artifact.relative_to(candidate_root) != relative_artifact:
                    raise ValueError
            except (IndexError, ValueError) as exc:
                raise SceneComputeError("published artifact has no unambiguous worker root") from exc
            if old_root is not None and candidate_root != old_root:
                raise SceneComputeError("published identities name different worker roots")
            old_root = candidate_root
            request = reference["request_identity"]
            sealed = store.manifest(reference["key"])
            if (reference["store_root"] != str(old_root / "prepared" / name)
                    or prepared["store_root"] != reference["store_root"]
                    or reference["scene_artifact_ref"] != identity["artifact_path"]
                    or reference["scene_input_identity_sha256"] != identity["sha256"]
                    or reference["manifest_sha256"] != sealed["manifest_sha256"]
                    or request["qa_id"] != qa_id or request["scene_name"] != expected_scene_id
                    or request["scene_audit_qa_ids"] != qa_ids):
                raise SceneComputeError("published binding request/scene audit identity changed")
            if cfg is not None:
                from skill3d.segmentation.question_binding import binding_parameters

                model = request["model"]
                if (request["seed"] != cfg.seed
                        or model["provider"] != cfg.online_model_provider
                        or model["model_id"] != cfg.online_model
                        or model["model_revision"] != cfg.online_model_revision
                        or model["endpoint_sha256"] != hashlib.sha256(
                            cfg.online_base_url.encode()).hexdigest()
                        or request["parameters"] != binding_parameters(
                            semantic_min_pixels=cfg.semantic_min_pixels,
                            semantic_max_pixels=cfg.semantic_max_pixels,
                            semantic_relation_batch_size=cfg.semantic_relation_batch_size,
                            semantic_review_config=cfg.semantic_review_config)):
                    raise SceneComputeError("published M5 operating point changed")
            dependencies = {}
            for field, record in identity["files"].items():
                source = PurePosixPath(_reference(record["path"], absolute=True))
                try:
                    relative = source.relative_to(old_root).as_posix()
                except ValueError as exc:
                    raise SceneComputeError("published geometry dependency is not output-local") from exc
                _local_file(output, relative)
                dependencies[field] = relative
            receipt = store.create_migration_receipt(
                reference["key"], scene_input_identity=identity, destination_root=output,
                scene_artifact_ref=artifact_ref, dependency_refs=dependencies,
            )
            resolver = store.validate_migration_receipt(
                reference["key"], receipt, scene_artifact=output / artifact_ref,
                scene_input_identity=identity, destination_root=output,
            )
            binding = store.load(
                reference["key"], expected_request_identity=request,
                scene_input_identity=identity, migration_receipt=receipt,
                migration_destination_root=output,
            )
            _reject_infrastructure_failures(binding.stats)
            for obj in binding.objects:
                for field in OBJECT_REFS:
                    ref = getattr(obj, field)
                    if ref:
                        _local_file(output, Path(ref).relative_to(output).as_posix())
            outcomes[qa_id] = {
                "status": binding.stats.get("status") or (
                    "binding_unavailable" if not binding.materialized
                    or binding.stats.get("binding_error") else "prepared"),
                "materialized": binding.materialized, "n_objects": len(binding.objects),
                "binding_error": binding.stats.get("binding_error"),
            }
            if outcomes[qa_id] != variant["outcomes"][qa_id]:
                raise SceneComputeError("published binding outcome changed")
            receipts[qa_id] = receipt
            resolved_artifact = resolver.resolved_artifact()
        if (resolved_artifact.scene_name != expected_scene_id
                or resolved_artifact.reconstruction_input_identity
                != summary["validation"]["reconstruction_input_identity"]
                or resolved_artifact.scale_fusion_status != variant["metric_status"]):
            raise SceneComputeError("published reconstruction/metric status changed")
        if name == "metric":
            from skill3d.reconstruction.metric_evidence import validated_receipt

            validated_receipt(resolved_artifact)
        elif resolved_artifact.metric_model not in (None, "none"):
            raise SceneComputeError("published base unexpectedly contains metric evidence")
        validated[name] = {
            "artifact": artifact_ref, "binding_manifest": binding_ref,
            "n_questions": len(qa_ids), "outcomes": outcomes,
            "migration_receipts": receipts, "metric_status": variant["metric_status"],
        }
    return {
        "schema": "harness3d-published-scene-validation/1", "status": "validated",
        "scene_id": expected_scene_id, "qa_ids": qa_ids, "variants": validated,
        "no_solver": True, "no_labels": True,
    }


__all__ = [
    "SceneComputeError", "compute_scene", "validate_scene_input",
    "validate_scene_output", "validate_published_scene",
]
