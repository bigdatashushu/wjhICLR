"""Frozen B01/B11 evaluation. No evolution, active-pointer lookup or mock fallback.

M5 is prepared once per question, before either solver sees a Skill. Each solver
gets a physical copy of that tree and fresh runtime state. JSON receipts contain
content identities, not just filenames; results are joined by qa_id.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import time
import uuid
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.adapters.frame_set import frame_set_hash
from skill3d.online import runner
from skill3d.online.runner import OnlineRunConfig, PreparedObjectBinding
from skill3d.online.submission import EXECUTION_PROTOCOL_VERSION
from skill3d.routing.task_classifier import canonical_task, classify
from skill3d.schemas import (
    ObjectRecord,
    ReconstructionArtifact,
    SkillEvaluationBinding,
    SkillSpecV11,
)
from skill3d.schemas.trace import EVALUATION_CONTRACT_VERSION
from skill3d.schemas.evidence import skill_evidence_state
from skill3d.segmentation.open_vocab_detector import capture_detector_failures
from skill3d.segmentation.prepared_binding_store import (
    PreparedBindingStore,
    prepared_binding_request_identity,
)
from skill3d.skills.v11_library import (
    _manifest_for,
    canonical_json_sha256,
    resolve_source_like_ref,
    validate_v11_snapshot,
)
from skill3d.synthesis.prompt_builder import PROMPT_TEMPLATE_VERSION
from skill3d.synthesis.eval_visual_fallback import EVAL_VISUAL_PROMPT_VERSION
from skill3d.synthesis.request_context import current_request_phase
from skill3d.tools.docs_v11 import TOOL_DOCS_VERSION
from skill3d.tools.registry import TOOL_FACE_VERSION
from skill3d.tools.scene_handle import (
    canonical_payload_sha256,
    file_sha256,
    frozen_reconstruction_identity,
    prepared_room_request,
)
from skill3d.trace.store import TraceStore
from skill3d.infra.experiment_startup import MODE_ENV, current_code_identity

SNAPSHOT_ID = "S0-v11-task-tools"
DEFAULT_LIBRARY = Path(__file__).resolve().parents[3] / "skill_library"
ARMS = ("B01", "B11")
ARTIFACT_REFS = (
    "c2w_list", "intrinsics", "depth_maps", "point_map", "point_conf",
    "depth_conf", "track_list", "per_frame_scale_ref",
    "metric_depth_maps", "metric_valid_masks",
)
OBJECT_REFS = ("mask_per_frame", "pointcloud_world", "pointconf_world", "centroid_ref")


class PairingError(ValueError):
    """Missing, duplicate or inconsistent experiment identity."""


def _prepared_room_worker(connection, point_map_path: str, metric_scale: float,
                          world_up: list[float] | None) -> None:
    try:
        from skill3d.tools.distance_primitives import room_size_from_planes

        point_map = np.load(point_map_path, allow_pickle=False)
        payload = room_size_from_planes(
            np.asarray(point_map, dtype=np.float64).reshape(-1, 3),
            metric_scale=float(metric_scale), up_hint=world_up)
        connection.send({"status": "success", "payload": _clean(payload)})
    except BaseException as exc:  # noqa: BLE001 - failure is frozen into a shared receipt
        connection.send({
            "status": "failure",
            "error": f"{type(exc).__name__}: {exc}",
        })
    finally:
        connection.close()


def _prepare_room_measurement(
    artifact: ReconstructionArtifact,
    *,
    timeout_s: float = 300.0,
) -> dict:
    """Compute one deterministic room payload in a child process and bind its identity."""
    point_map_path = Path(artifact.point_map)
    scale = getattr(artifact, "metric_scale", None)
    if not point_map_path.is_file() or scale is None:
        raise PairingError("authorized room preparation requires point_map and metric_scale")
    request = prepared_room_request(
        point_map_sha256=file_sha256(point_map_path),
        metric_scale=float(scale),
        frozen_input_identity=frozen_reconstruction_identity(artifact),
        world_up=artifact.world_up)
    timeout = max(0.0, float(timeout_s))
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_prepared_room_worker,
        args=(child, str(point_map_path), float(scale), request["world_up"]),
        daemon=True)
    started = time.perf_counter()
    try:
        process.start()
    except BaseException as exc:  # noqa: BLE001 - startup failure is a shared fact
        child.close()
        parent.close()
        outcome = {
            "status": "failure",
            "error": f"{type(exc).__name__}: failed to start prepared worker: {exc}",
        }
    else:
        child.close()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            outcome = {
                "status": "timeout",
                "error": f"prepared room measurement exceeded {timeout:.1f}s",
            }
        elif parent.poll():
            outcome = parent.recv()
        else:
            outcome = {
                "status": "failure",
                "error": f"worker exited with code {process.exitcode} without a payload",
            }
        parent.close()
    runtime_s = time.perf_counter() - started
    receipt = {
        "request": request,
        "status": str(outcome["status"]),
        "runtime_s": float(runtime_s),
        "timeout_s": timeout,
        "payload": outcome.get("payload"),
        "payload_sha256": (
            canonical_payload_sha256(outcome["payload"])
            if outcome.get("status") == "success" else None),
        "error": outcome.get("error"),
    }
    return {**receipt, "receipt_sha256": canonical_payload_sha256(receipt)}


def _clean(value):
    if hasattr(value, "model_dump"):
        return _clean(value.model_dump(mode="json"))
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in (sorted(value) if isinstance(value, set) else value)]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _encoded(value) -> bytes:
    return json.dumps(_clean(value), ensure_ascii=False, sort_keys=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _hash(value) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _file_hash(path: Path) -> str:
    # Reconstruction arrays are shared by every question in one scene. Reuse a
    # digest only while inode, size, mtime and ctime are unchanged; this keeps
    # the exact SHA256 identity while avoiding full array rereads per question.
    from skill3d.infra.m5_identity import file_identity

    return str(file_identity(path)["sha256"])


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(_encoded(value) + b"\n")
    temp.replace(path)


def _index(rows, *, key="qa_id") -> dict:
    out = {}
    for row in rows:
        value = str(row[key])
        if not value or value in out:
            raise PairingError(f"empty or duplicate {key}: {value!r}")
        out[value] = row
    return out


def _same_ids(expected, actual, label: str) -> None:
    if set(expected) != set(actual):
        raise PairingError(
            f"{label}: missing={sorted(set(expected) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(expected))}")


def _ref_path(ref: str, artifact_path: Path) -> Path:
    # Existing reconstruction writers use cwd-relative paths; portable manifests
    # can use artifact-relative paths. Reject ambiguity instead of guessing.
    p = Path(ref)
    choices = {p.resolve()} if p.is_absolute() else {
        p.resolve(), (artifact_path.parent / p).resolve()}
    found = [p for p in choices if p.is_file()]
    if len(found) != 1:
        raise PairingError(f"missing or ambiguous artifact reference: {ref!r}")
    return found[0]


def _artifact_files(art: ReconstructionArtifact, path: Path) -> dict[str, Path]:
    refs = {key: getattr(art, key) for key in ARTIFACT_REFS}
    refs.update({f"confidence.{key}": value
                 for key, value in art.confidence.model_dump().items()})
    return {key: _ref_path(value, path) for key, value in refs.items() if value}


def _pixels_identity(pixels) -> list[dict]:
    return [{"shape": list(p.shape), "dtype": str(p.dtype),
             "sha256": hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()}
            for p in pixels]


def input_identity(item: EpisodeItem, artifact_path: str | Path) -> dict:
    """Validate actual frame order, source identity and every declared artifact ref."""
    ep = item.episode
    fs = ep.frame_set
    path = Path(artifact_path).resolve()
    art = ReconstructionArtifact.model_validate_json(path.read_bytes())
    if not ep.qa_id or not ep.dataset or not ep.scene_name or fs is None:
        raise PairingError(f"{ep.qa_id}: missing question/source/frame identity")
    if not art.artifact_id or not art.artifact_version:
        raise PairingError(f"{ep.qa_id}: missing artifact identity")
    if fs.frame_set_hash != frame_set_hash(fs.frame_ids):
        raise PairingError(f"{ep.qa_id}: invalid frame_set_hash")
    if art.scene_name != ep.scene_name or art.frame_set_hash != fs.frame_set_hash:
        raise PairingError(f"{ep.qa_id}: artifact scene/frame_set_hash mismatch")
    if len(set(fs.readable_frame_ids)) != len(fs.readable_frame_ids):
        raise PairingError(f"{ep.qa_id}: duplicate readable frame IDs")
    slots = [fs.frame_ids.index(fid) for fid in fs.readable_frame_ids]
    if slots != sorted(slots):
        raise PairingError(f"{ep.qa_id}: readable frame order mismatch")
    for key, expected in (
        ("frame_ids", fs.readable_frame_ids),
        ("source_frame_indices", [fs.source_frame_indices[i] for i in slots]),
        ("timestamps", [fs.timestamps[i] for i in slots]),
    ):
        if list(getattr(art, key)) != list(expected):
            raise PairingError(f"{ep.qa_id}: artifact {key} mismatch")
    if len(item.pixels) != len(slots):
        raise PairingError(f"{ep.qa_id}: actual pixel count mismatch")
    if any(p.ndim != 3 or p.shape[2] != 3 or p.size == 0 for p in item.pixels):
        raise PairingError(f"{ep.qa_id}: invalid RGB pixels")
    from skill3d.adapters.frame_set import with_pixel_identity

    actual_frames = with_pixel_identity(fs, item.pixels)
    if art.metric_model not in (None, "none"):
        from skill3d.reconstruction.metric_evidence import validated_receipt

        if actual_frames.cache_identity() != art.metric_input_identity:
            raise PairingError(f"{ep.qa_id}: metric input pixels mismatch")
        try:
            validated_receipt(art)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise PairingError(f"{ep.qa_id}: invalid metric evidence: {exc}") from exc
    for key, actual, expected in (
        ("dataset", fs.dataset_id, ep.dataset),
        ("scene", fs.scene_name, ep.scene_name),
        ("episode", fs.episode_id, ep.qa_id),
    ):
        if actual and actual != expected:
            raise PairingError(f"{ep.qa_id}: FrameSet {key} mismatch")
    files = _artifact_files(art, path)
    for key in ("depth_maps", "c2w_list", "point_map", "depth_conf"):
        if key in files:
            arr = np.load(files[key], mmap_mode="r", allow_pickle=False)
            if arr.shape[0] != len(slots):
                raise PairingError(f"{ep.qa_id}: {key} frame dimension mismatch")
    value = {
        "episode": ep.model_dump(mode="json"), "source": item.source,
        "video_path": item.video_path, "pixels": _pixels_identity(item.pixels),
        "artifact_path": str(path), "artifact_json_sha256": _file_hash(path),
        "artifact_id": art.artifact_id, "artifact_version": art.artifact_version,
        "files": {key: {"path": str(p), "sha256": _file_hash(p)}
                  for key, p in sorted(files.items())},
    }
    return {**value, "sha256": _hash(value)}


def initial_state_for_item(item: EpisodeItem, artifact_path: str | Path) -> str:
    """Determine the slot from shared geometry before any model request or score."""
    path = Path(artifact_path).resolve()
    art = ReconstructionArtifact.model_validate_json(path.read_bytes())
    art = art.model_copy(update={
        key: str(value) for key, value in _artifact_files(art, path).items()
        if key in ARTIFACT_REFS})
    art = runner._prepare_m4_quality(art, item.pixels, persist_quality=False)
    scene = runner.build_scene_state(
        art, question_type=canonical_task(item.episode.question_type))
    return skill_evidence_state(scene.evidence_profile)


def input_error_identity(item: EpisodeItem) -> dict:
    """Content identity for a preregistered question with no readable image input."""
    if item.input_error is None:
        raise PairingError(f"{item.episode.qa_id}: missing input_error record")
    value = {
        "episode": item.episode.model_dump(mode="json"),
        "source": item.source,
        "input_error": item.input_error.model_dump(mode="json"),
    }
    return {**value, "sha256": _hash(value)}


def _copy_base(
    source: Path,
    dest: Path,
    *,
    link_files: bool = False,
) -> ReconstructionArtifact:
    """Copy declared arrays only: old M5 caches must not enter the experiment."""
    art = ReconstructionArtifact.model_validate_json(source.read_bytes())
    data = art.model_dump()
    dest.mkdir(parents=True)
    for key, src in _artifact_files(art, source).items():
        target = dest / (key.replace(".", "_") + src.suffix)
        if link_files:
            try:
                os.link(src, target)
            except OSError as exc:
                raise PairingError(
                    f"immutable scene payload cannot be hard-linked: {src}"
                ) from exc
        else:
            shutil.copyfile(src, target)
        if key.startswith("confidence."):
            data["confidence"][key.split(".")[1]] = str(target)
        else:
            data[key] = str(target)
    return ReconstructionArtifact.model_validate(data)


def _rebase(value, source: Path, dest: Path):
    if isinstance(value, str):
        return value.replace(str(source) + "/", str(dest) + "/")
    if isinstance(value, dict):
        return {_rebase(k, source, dest): _rebase(v, source, dest) for k, v in value.items()}
    if isinstance(value, list):
        return [_rebase(v, source, dest) for v in value]
    return value


def _save_binding(binding: PreparedObjectBinding, root: Path) -> None:
    def pack(value, key="stats"):
        if isinstance(value, np.ndarray):
            path = root / f"binding_{key}.npy"
            np.save(path, value)
            return {"__array_ref__": str(path)}
        if isinstance(value, dict):
            return {k: pack(v, f"{key}_{k}") for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [pack(v, f"{key}_{i}") for i, v in enumerate(value)]
        return value

    for obj in binding.objects:
        for name in OBJECT_REFS:
            ref = getattr(obj, name)
            if ref:
                p = Path(ref).resolve()
                if not p.is_relative_to(root) or not p.is_file():
                    raise PairingError(f"M5 object reference escapes private tree: {ref}")
    _write(root / "binding.json", {
        "objects": binding.objects, "stats": pack(binding.stats),
        "notes": binding.notes, "materialized": binding.materialized,
        "prepared_measurements": pack(binding.prepared_measurements,
                                      "prepared_measurements"),
    })


def _load_binding(root: Path) -> PreparedObjectBinding:
    store_reference = root / "binding_store_ref.json"
    if store_reference.is_file():
        reference = json.loads(store_reference.read_text(encoding="utf-8"))
        scene_input_identity = reference.pop("scene_input_identity")
        store = PreparedBindingStore(reference["store_root"])
        binding = store.load(
            reference["key"],
            expected_request_identity=reference["request_identity"],
            scene_artifact=reference["scene_artifact_ref"],
            scene_input_identity=scene_input_identity,
        )
        overlay_path = root / "binding_overlay.json"
        if overlay_path.is_file():
            overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
            binding.prepared_measurements = overlay.get(
                "prepared_measurements") or {}
            binding.notes = list(overlay.get("notes") or binding.notes)
            binding.stats.update(overlay.get("stats_updates") or {})
        return binding

    def unpack(value):
        if isinstance(value, dict):
            if set(value) == {"__array_ref__"}:
                return np.load(value["__array_ref__"], allow_pickle=False)
            return {k: unpack(v) for k, v in value.items()}
        if isinstance(value, list):
            return [unpack(v) for v in value]
        return value

    data = unpack(json.loads((root / "binding.json").read_text()))
    data["objects"] = [ObjectRecord.model_validate(o) for o in data["objects"]]
    return PreparedObjectBinding(**data)


def tree_identity(root: Path) -> dict:
    """Path-independent identity, including all M5 caches and object arrays."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PairingError(f"symlink in frozen artifact tree: {path}")
        if not path.is_file():
            continue
        digest = (_hash(_rebase(json.loads(path.read_text()), root, Path("$TREE")))
                  if path.suffix == ".json" else _file_hash(path))
        result[path.relative_to(root).as_posix()] = digest
    return {"files": result, "sha256": _hash(result)}


def _clone_tree(
    source: Path,
    dest: Path,
    *,
    link_binary: bool = False,
) -> None:
    tree_identity(source)  # reject links before copying

    def clone_file(src, dst):
        if link_binary and Path(src).suffix != ".json":
            try:
                os.link(src, dst)
            except OSError as exc:
                raise PairingError(
                    f"immutable prepared payload cannot be hard-linked: {src}"
                ) from exc
            return dst
        return shutil.copyfile(src, dst)

    shutil.copytree(source, dest, copy_function=clone_file)
    for path in dest.rglob("*.json"):
        # Runtime geometry schema uses NaN for unavailable diagnostic values.
        # Keep that existing serialization contract; report JSON uses null.
        value = json.loads(path.read_text())
        # Metric receipts contain content hashes, not mutable file references.
        # Keep their exact bytes: the artifact binds the receipt's SHA256.
        if isinstance(value, dict) and value.get("request_sha256") and value.get("metric_fusion_version"):
            continue
        path.write_text(json.dumps(_rebase(value, source, dest),
                                   ensure_ascii=False, sort_keys=True), encoding="utf-8")


def quality_contract() -> dict:
    """Record current code/thresholds; never claim TODO defaults are calibrated."""
    from skill3d.reconstruction_gate import m4_main_gate, scene_state

    root = Path(m4_main_gate.__file__).parent
    evidence_sources = (
        "tools/category_match.py", "tools/scene_handle.py", "tools/geometry_tools.py",
        "tools/distance_primitives.py", "tools/ranking_contract.py",
        "segmentation/sam2_tracker.py", "segmentation/open_vocab_detector.py",
        "segmentation/question_binding.py", "segmentation/binding_validation.py",
        "verifier/geometry_oracle.py", "verifier/derivation.py",
        "online/runner.py", "online/submission.py",
        "synthesis/eval_visual_fallback.py", "synthesis/request_context.py",
        "synthesis/bailian_client.py", "synthesis/client_errors.py", "fsm/online_fsm.py",
        "online/config.py", "online/recovery.py",
        "synthesis/program_assembler.py", "synthesis/prompt_builder.py", "tools/docs_v11.py",
        "adapters/frame_set.py", "adapters/episode_source.py",
        "reconstruction/vggt_runner.py", "reconstruction/metric_fusion.py",
        "reconstruction/metric_evidence.py", "reconstruction/image_grid.py",
        "sandbox/kernel.py", "sandbox/receipt.py",
        "infra/m5_identity.py", "infra/version_lock.py", "coords.py",
        "segmentation/groundingdino_startup.py", "schemas/reconstruction.py",
        "schemas/tool.py", "schemas/trace.py", "schemas/authorization.py",
        "tools/task_policy.py", "tools/registry.py", "tools/contract.py",
    )
    from skill3d.reconstruction.metric_fusion import threshold_snapshot, METRIC_FUSION_VERSION

    value = {
        "enabled": True, "metric_version": m4_main_gate.M4_GATE_VERSION,
        "metric_fusion_version": METRIC_FUSION_VERSION,
        "metric_fusion_thresholds": threshold_snapshot(),
        "default_thresholds": m4_main_gate.default_thresholds(),
        "overall_quality_threshold": scene_state.TH_OVERALL_QUALITY,
        "source_sha256": {p.name: _file_hash(p) for p in sorted(root.glob("*.py"))},
        "evidence_source_sha256": {
            path: _file_hash(root.parent / path) for path in evidence_sources},
        "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
        "tool_docs_version": TOOL_DOCS_VERSION,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "eval_visual_prompt_version": EVAL_VISUAL_PROMPT_VERSION,
    }
    return {**value, "sha256": _hash(value)}


def _common_config(cfg: OnlineRunConfig) -> dict:
    private = {"skills", "active_snapshot_ref", "active_snapshot_manifest_sha256",
               "trace_dir", "work_dir", "recon_dir", "reuse_artifact",
               "evaluation_binding"}
    value = {f.name: getattr(cfg, f.name) for f in fields(cfg)
             if f.name not in private and f.name != "retrieval_policy"}
    value["retrieval_policy"] = cfg.retrieval_policy.to_dict()
    value["temperature"] = cfg.temperature
    value["prompt_template_version"] = PROMPT_TEMPLATE_VERSION
    value["execution_protocol_version"] = EXECUTION_PROTOCOL_VERSION
    value["tool_face_version"] = TOOL_FACE_VERSION
    value["tool_docs_version"] = TOOL_DOCS_VERSION
    from skill3d.tools.task_policy import TOOL_TASK_POLICY_VERSION
    from skill3d.schemas.authorization import DECISION_VERSION
    value["tool_task_policy_version"] = TOOL_TASK_POLICY_VERSION
    value["tool_authorization_version"] = DECISION_VERSION
    value["eval_visual_prompt_version"] = EVAL_VISUAL_PROMPT_VERSION
    return _clean(value)


class _BoundTrace(TraceStore):
    def __init__(self, root, identity):
        super().__init__(root)
        self.identity = identity

    def append(self, topic, record):
        data = record.model_dump(mode="json") if hasattr(record, "model_dump") else dict(record)
        super().append(topic, {**data, **self.identity})


class _RecordingClient:
    def __init__(self, client, *, seed, specs, store):
        self.client, self.seed, self.specs, self.store = client, seed, specs, store
        self.requests: list[dict] = []
        self.failures: list[str] = []
        self.contract_errors: list[str] = []

    @property
    def last_usage(self):
        return getattr(self.client, "last_usage", {})

    @property
    def last_generation(self):
        return getattr(self.client, "last_generation", {})

    @property
    def last_response_identity(self):
        return getattr(self.client, "last_response_identity", {})

    @property
    def model(self):
        return str(getattr(self.client, "model", ""))

    def audit_identity(self, *, seed=None):
        method = getattr(self.client, "audit_identity", None)
        return (method(seed=seed) if callable(method)
                else {"model": self.model, "seed": seed, "cache_eligible": False})

    def chat(self, messages, max_tokens=8192, **kwargs):
        plain = copy.deepcopy(messages)
        delivered = {}
        phase = current_request_phase()
        for message in plain:
            content = message.get("content")
            parts = ([{"text": content}] if isinstance(content, str) else content or [])
            for part in parts:
                text = part.get("text", "")
                # A candidate may extend the complete parent text verbatim.
                # Match the longest full body first so the parent prefix cannot
                # be counted as delivery or left behind in the common prompt.
                for spec in sorted(self.specs, key=lambda s: len(s.skill_md), reverse=True):
                    block = "\n## Reference method\n" + spec.skill_md
                    if block in text:
                        key = f"{spec.skill_id}@{spec.version}"
                        delivered[key] = spec.content_sha256
                        text = text.replace(block, "")
                if "text" in part:
                    part["text"] = text
            if isinstance(content, str):
                message["content"] = parts[0]["text"]
        row = {"request_index": len(self.requests), "solver": phase == "program",
               "phase": phase,
               "seed": kwargs.get("seed"), "max_tokens": max_tokens,
               "request_sha256": _hash(messages), "common_request_sha256": _hash(plain),
               "skill_content_sha256": delivered, "success": False}
        self.requests.append(row)
        try:
            if kwargs.get("seed") != self.seed:
                raise PairingError(f"actual model seed {kwargs.get('seed')} != {self.seed}")
            if self.client is None:
                raise RuntimeError("model endpoint not configured")
            answer = self.client.chat(messages, max_tokens=max_tokens, **kwargs)
            row["success"] = True
            row["response_text"] = answer
            return answer
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            self.failures.append(row["error"])
            if isinstance(exc, PairingError):
                self.contract_errors.append(row["error"])
            raise
        finally:
            row["usage"] = dict(self.last_usage or {})
            row["usage_status"] = "measured" if row["usage"] else "unknown"
            row["generation"] = dict(self.last_generation or {})
            self.store.append("model_requests", row)


_PROGRAM_ERROR_CODES = frozenset({
    "ast_violation",
    "no_answer",
    "violation_runtime",
    "violation_syntax",
    "online_model_parse_error",
})


def _arm_result(
    out,
    client,
    identity,
    input_hash,
    config_hash,
    expected,
    *,
    options,
    reconstruction_cost_s=None,
) -> dict:
    delivered = {}
    for record in out.retrieval_records:
        data = record.model_dump(mode="json") if hasattr(record, "model_dump") else record
        delivered.update(data.get("delivered_content_sha256") or {})
    program_requests = [r for r in client.requests if r["phase"] == "program"]
    visual_requests = [r for r in client.requests if r["phase"] == "eval_visual_fallback"]
    requests = program_requests + visual_requests
    successful = [r for r in program_requests if r["success"]]
    service_errors = list(client.failures) + [
        e for r in out.rounds for e in r.get("service_errors", [])]
    incomplete = bool(service_errors) or out.final_state == "unavailable"
    errors = [r for r in out.rounds if r.get("error_code")]
    error_codes = {str(row.get("error_code") or "") for row in errors}
    delivery_ok = (all(r["skill_content_sha256"] == expected for r in program_requests)
                   and all(not r["skill_content_sha256"] for r in visual_requests)
                   and len(visual_requests) <= 1
                   and (not successful or delivered == expected)
                   and set(out.delivered_skill_versions) == set(delivered))
    scoring_status = "incomplete" if incomplete else out.scoring_status
    score = out.score if scoring_status == "scored" else None
    # Parsing is the same official parsing used by M12; it does not depend on GT.
    legal = out.answer is not None and out.predicted is not None
    if out.is_mca:
        legal = legal and out.predicted in list("ABCD"[:len(options or [])])
    return {
        **identity, "input_sha256": input_hash, "config_sha256": config_hash,
        "status": "incomplete" if incomplete else "completed",
        "final_state": out.final_state, "answer": out.answer, "predicted": out.predicted,
        "task": out.task or canonical_task(out.question_type),
        "metric": "Accuracy" if out.is_mca else "MRA", "score": score,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "answer_parse_status": out.answer_parse_status,
        "submission_status": out.submission_status,
        "scoring_status": scoring_status,
        "correct": out.correct if scoring_status == "scored" else None,
        "mra_value": out.mra_value if scoring_status == "scored" else None,
        "legal_answer": bool(legal),
        "program_error": bool(
            error_codes & _PROGRAM_ERROR_CODES
            or out.static_check_errors
            or out.synthesis_source == "online_model_parse_error"),
        "runtime_error": out.final_state == "run_error",
        "tool_contract_error": "tool_contract" in error_codes,
        "geometry_rejection": "geometry_rejected" in error_codes,
        "untrusted_geometry_used": bool(
            out.answer
            and (
                out.answer_untrusted
                or set(out.used_result_ids) & set(out.invalidated_result_ids)
            )
        ),
        "agent_rounds": int(out.agent_rounds),
        "yield_count": int(out.yield_count),
        "model_request_count": len(requests),
        "program_request_count": len(program_requests),
        "eval_visual_request_count": len(visual_requests),
        "eval_visual_accepted": bool(out.eval_visual_fallback.get("accepted")),
        "eval_visual_request_sha256": (
            visual_requests[0]["request_sha256"] if visual_requests else None),
        "eval_visual_fallback": dict(out.eval_visual_fallback),
        "original_score": out.eval_visual_fallback.get("original_result", {}).get("score", score),
        "original_final_state": out.eval_visual_fallback.get("original_result", {}).get(
            "final_state", out.final_state),
        "tool_call_count": sum(
            len(row.get("results") or []) for row in out.rounds
        ),
        "reconstruction_cost_s": (
            None if reconstruction_cost_s is None else float(reconstruction_cost_s)
        ),
        "errors": errors, "static_check_errors": out.static_check_errors,
        "service_errors": service_errors,
        "failure_code": out.failure_code, "notes": list(out.notes),
        "terminal_failure": dict(out.terminal_failure),
        "detector_attempts": list(out.detector_attempts),
        "answer_basis": out.answer_basis,
        "used_result_ids": list(out.used_result_ids),
        "used_evidence_flags": list(out.used_evidence_flags),
        "geometry_verify": out.verify.model_dump(mode="json") if out.verify is not None else None,
        "delivered_skill_versions": out.delivered_skill_versions,
        "delivered_content_sha256": delivered, "delivery_ok": delivery_ok,
        "initial_evidence_states": dict(out.initial_evidence_states),
        "initial_skill_selection": dict(out.initial_skill_selection),
        "requests": client.requests,
        "first_common_request_sha256": (
            program_requests[0]["common_request_sha256"] if program_requests else None),
        "quality_status": out.quality_status, "main_gate_passed": out.main_gate_passed,
        "scene_route": out.scene_route, "question_tool_scope": out.question_tool_scope,
        "answer_source": out.answer_source, "image_ledger": out.image_ledger,
        "episode_status": out.episode_status,
        "input_error": out.episode_status == "input_error",
        "input_error_reason": out.input_error_reason,
    }


def _incomplete_row(identity, *, task, is_mca, input_hash, config_hash, errors):
    return {
        **identity, "input_sha256": input_hash, "config_sha256": config_hash,
        "status": "incomplete", "final_state": "unavailable", "task": task,
        "metric": "Accuracy" if is_mca else "MRA",
        "answer": None, "score": None, "legal_answer": False,
        "predicted": None, "correct": None, "mra_value": None,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "answer_parse_status": "no_answer", "submission_status": "not_submitted",
        "scoring_status": "incomplete",
        "program_error": False, "runtime_error": False, "input_error": False,
        "tool_contract_error": False, "geometry_rejection": False,
        "untrusted_geometry_used": False, "agent_rounds": 0, "yield_count": 0,
        "model_request_count": 0, "tool_call_count": 0, "reconstruction_cost_s": None,
        "program_request_count": 0, "eval_visual_request_count": 0,
        "eval_visual_accepted": False, "eval_visual_request_sha256": None,
        "eval_visual_fallback": {},
        "used_result_ids": [], "used_evidence_flags": [], "geometry_verify": None,
        "original_score": None, "original_final_state": "unavailable",
        "input_error_reason": "", "episode_status": "run_error", "delivery_ok": True,
        "delivered_skill_versions": [], "delivered_content_sha256": {},
        "service_errors": errors, "requests": [],
    }


def pair_results(b01: Sequence[dict], b11: Sequence[dict], expected_qa_ids) -> list[dict]:
    """Strict qa_id join, independent of result-list ordering."""
    expected = _index([{"qa_id": q} for q in expected_qa_ids])
    left, right = _index(b01), _index(b11)
    _same_ids(expected, left, "B01 results")
    _same_ids(expected, right, "B11 results")
    pairs = []
    for qa_id in expected:
        a, b = (_result_states(left[qa_id]), _result_states(right[qa_id]))
        for key in ("input_sha256", "config_sha256", "task", "metric"):
            if not a.get(key) or a[key] != b.get(key):
                raise PairingError(f"{qa_id}: paired {key} missing/mismatch")
        prompts = [a.get("first_common_request_sha256"), b.get("first_common_request_sha256")]
        if all(prompts) and prompts[0] != prompts[1]:
            raise PairingError(f"{qa_id}: actual common first requests differ")
        visual_prompts = [
            a.get("eval_visual_request_sha256"), b.get("eval_visual_request_sha256")]
        if all(visual_prompts) and visual_prompts[0] != visual_prompts[1]:
            raise PairingError(f"{qa_id}: actual evaluation visual requests differ")
        if not a["delivery_ok"] or not b["delivery_ok"]:
            raise PairingError(f"{qa_id}: actual Skill delivery mismatch")
        completed = a["status"] == b["status"] == "completed"
        scored = completed and a["scoring_status"] == b["scoring_status"] == "scored"
        input_error_pair = bool(a.get("input_error") and b.get("input_error"))
        pairs.append({
            "qa_id": qa_id, "task": a["task"], "metric": a["metric"],
            "input_sha256": a["input_sha256"], "config_sha256": a["config_sha256"],
            "status": "completed" if completed else "incomplete",
            "comparable": completed and (all(prompts) or input_error_pair),
            "eval_visual_common_request_sha256": (
                visual_prompts[0] if all(visual_prompts) else None),
            "arms": {"B01": a, "B11": b},
            "scoring_status": ("scored" if scored else "incomplete" if not completed
                               else "invalid_reference" if "invalid_reference" in
                               (a["scoring_status"], b["scoring_status"])
                               else "reference_unavailable"),
            "delta": b["score"] - a["score"] if scored else None,
        })
    return pairs


def _result_states(record):
    """只为无新合同标记的历史/替身记录显式推导，禁止补写原记录。"""
    if record.get("evaluation_contract_version") == EVALUATION_CONTRACT_VERSION:
        if not all(key in record for key in
                   ("answer_parse_status", "submission_status", "scoring_status")):
            raise PairingError("current result lacks explicit evaluation states")
        if (record["scoring_status"] == "scored") != (record.get("score") is not None):
            raise PairingError("score conflicts with explicit scoring state")
        return record
    return {**record, "state_provenance": "legacy_score_status_adapter",
            "scoring_status": record.get("scoring_status", (
                "incomplete" if record.get("status") != "completed" else
                "scored" if record.get("score") is not None else "reference_unavailable"))}


def summarize_pairs(pairs: Sequence[dict]) -> dict:
    """All aggregate values can be recomputed from paired_results.jsonl."""
    if not pairs:
        raise PairingError("cannot summarize an empty result panel")
    _index(pairs)
    groups = {"all": list(pairs)}
    for p in pairs:
        groups.setdefault(p["task"], []).append(p)
    report = {}
    rate_fields = {
        "legal_answer_rate": "legal_answer",
        "program_error_rate": "program_error",
        "runtime_error_rate": "runtime_error",
        "input_error_rate": "input_error",
        "tool_contract_error_rate": "tool_contract_error",
        "geometry_rejection_rate": "geometry_rejection",
        "untrusted_geometry_use_rate": "untrusted_geometry_used",
    }
    mean_fields = {
        "mean_agent_rounds": "agent_rounds",
        "mean_yield_count": "yield_count",
        "mean_model_requests": "model_request_count",
        "mean_tool_calls": "tool_call_count",
        "mean_reconstruction_cost_s": "reconstruction_cost_s",
    }
    for name, rows in groups.items():
        arms = {}
        for arm in ARMS:
            records = [_result_states(p["arms"][arm]) for p in rows]
            completed = [r for r in records if r["status"] == "completed"]
            # Both arm means use exactly the same paired scoring subset.
            scored = [_result_states(p["arms"][arm]) for p in rows if all(
                _result_states(p["arms"][a])["scoring_status"] == "scored" for a in ARMS)
                and p["status"] == "completed"]
            def mean(key, selected):
                return sum(float(r[key]) for r in selected) / len(selected) if selected else None
            arms[arm] = {
                "n_planned": len(records), "n_scored": len(scored),
                "n_completed": len(completed),
                "n_unscored": sum(r["status"] == "completed"
                                  and r["scoring_status"] != "scored" for r in records),
                "n_incomplete": len(records) - len(completed),
                "scoring_status_counts": {
                    state: sum(r["scoring_status"] == state for r in records)
                    for state in ("scored", "reference_unavailable", "invalid_reference", "incomplete")
                },
                "Accuracy": mean("score", [r for r in scored if r["metric"] == "Accuracy"]),
                "MRA": mean("score", [r for r in scored if r["metric"] == "MRA"]),
                "original_Accuracy": mean(
                    "original_score", [r for r in records if r["metric"] == "Accuracy"
                                       and r.get("original_score") is not None]),
                "original_MRA": mean(
                    "original_score", [r for r in records if r["metric"] == "MRA"
                                       and r.get("original_score") is not None]),
                "eval_visual_requests": sum(r.get("eval_visual_request_count", 0)
                                            for r in records),
                "eval_visual_accepted": sum(bool(r.get("eval_visual_accepted"))
                                            for r in records),
                "eval_visual_acceptance_rate": mean(
                    "eval_visual_accepted",
                    [r for r in records if r.get("eval_visual_request_count")]),
                "eval_visual_Accuracy": mean(
                    "score", [r for r in scored if r.get("eval_visual_request_count")
                              and r["metric"] == "Accuracy"]),
                "eval_visual_MRA": mean(
                    "score", [r for r in scored if r.get("eval_visual_request_count")
                              and r["metric"] == "MRA"]),
                "answer_basis_counts": {
                    basis: sum(r.get("answer_basis", "") == basis for r in records)
                    for basis in ("tool_derived", "visual_estimate", "mixed", "")
                },
                **{
                    output: mean(field, completed)
                    for output, field in rate_fields.items()
                },
                **{
                    output: mean(
                        field,
                        [record for record in completed if record.get(field) is not None],
                    )
                    for output, field in mean_fields.items()
                },
            }
        # Never compare two averages based on different available subsets.
        complete = [p for p in rows if p["status"] == "completed"]
        delta = {}
        delta_fields = {
            "Accuracy": ("score", "Accuracy"),
            "MRA": ("score", "MRA"),
            **{output: (field, None) for output, field in rate_fields.items()},
            **{output: (field, None) for output, field in mean_fields.items()},
        }
        for key, (field, metric) in delta_fields.items():
            selected = [
                pair for pair in complete
                if (metric is None or pair["metric"] == metric)
                and pair["arms"]["B01"].get(field) is not None
                and pair["arms"]["B11"].get(field) is not None
            ]
            delta[key] = (sum(float(p["arms"]["B11"][field]) -
                              float(p["arms"]["B01"][field]) for p in selected) /
                          len(selected) if selected else None)
        report[name] = {"arms": arms, "B11_minus_B01": delta,
                        "n_complete_pairs": len(complete),
                        "n_scored_pairs": sum(p.get("delta") is not None for p in complete)}
    return {
        "status": "completed" if all(p["status"] == "completed" for p in pairs) else "incomplete",
        "n_pairs": len(pairs),
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "formal_scoring_eligible": all(
            _result_states(p["arms"][arm])["scoring_status"] == "scored"
            for p in pairs for arm in ARMS),
        "n_input_error_pairs": sum(
            1 for pair in pairs
            if all(pair["arms"][arm].get("input_error") for arm in ARMS)),
        "groups": report,
        "denominator_contract": (
            "Input/runtime/program/format failures score zero and remain in denominators. "
            "Unavailable/invalid references have null scores; completed engineering pairs remain complete. "
            "Service failures have null scores and make the experiment incomplete. "
            "Scores use the same scored pair subset; process rates use completed records."),
    }


def run_skill_ablation_v11(
    items: Sequence[EpisodeItem], *, artifact_paths: Mapping[str, str | Path],
    output_dir: str | Path, base_cfg: OnlineRunConfig, seed: int,
    question_types: Sequence[str] = ("object_rel_distance",),
    expected_qa_ids: Sequence[str] | None = None,
    library_root: str | Path = DEFAULT_LIBRARY,
    client_factory: Callable[[str, str], object] | None = None,
    quality_confirmation: dict | None = None,
    quality_confirmations: Mapping[str, dict] | None = None,
    reconstruction_costs: Mapping[str, float] | None = None,
    prepare_only: bool = False,
    prepared_manifest: str | Path | None = None,
    prepared_binding_store: str | Path | None = None,
    prepared_binding_refs: Mapping[str, Mapping] | None = None,
    prepared_room_cache: dict[str, dict] | None = None,
    _fixed_skill_arms: Mapping[str, SkillSpecV11] | None = None,
    _fixed_arm_snapshot_refs: Mapping[str, str] | None = None,
    _fixed_arm_manifest_hashes: Mapping[str, str] | None = None,
) -> dict:
    """Run frozen S0 Skill on/off with a real runner and optional test clients.

    client_factory(phase, qa_id) creates fresh clients for prepare/B01/B11.
    It is the sole fake-model seam; the program runner is never substituted.
    Output directories are create-only, so retries cannot mix experiments.
    """
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    experiment_id = f"v11-{uuid.uuid4().hex}"
    manifest = {"schema_version": "skill-ablation-v11/1.1",
                "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
                "experiment_id": experiment_id, "status": "preparing", "seed": int(seed),
                "run_mode": os.environ.get(MODE_ENV, "formal"),
                "code_identity": current_code_identity(),
                "requested_qa_ids": list(expected_qa_ids) if expected_qa_ids is not None
                else [it.episode.qa_id for it in items]}
    _write(root / "manifest.json", manifest)
    try:
        snapshot_id = SNAPSHOT_ID
        if base_cfg.mode != "real" or base_cfg.baseline != "C1_tools_program":
            raise PairingError("v11 requires real C1_tools_program (fake clients may be injected)")
        if base_cfg.evaluation_binding or base_cfg.metric_depth_model:
            raise PairingError("v11 requires normal retrieval, program solving and frozen geometry")
        binding_store_enabled = prepared_binding_store is not None
        if binding_store_enabled != (prepared_binding_refs is not None):
            raise PairingError(
                "prepared binding store and references must be supplied together"
            )
        if binding_store_enabled and prepared_manifest is not None:
            raise PairingError(
                "prepared binding store cannot be combined with a legacy prepared manifest"
            )
        wanted = {canonical_task(t) for t in question_types}
        if not wanted:
            raise PairingError("question_types must not be empty")
        fixed_pair = _fixed_skill_arms is not None
        fixed_arms = dict(_fixed_skill_arms or {})
        if fixed_pair:
            if set(fixed_arms) != set(ARMS):
                raise PairingError(
                    f"fixed Skill arms must be exactly {list(ARMS)}")
            if len(wanted) != 1:
                raise PairingError("fixed Skill evaluation requires one question type")
            for arm, spec in fixed_arms.items():
                if not isinstance(spec, SkillSpecV11):
                    raise PairingError(f"{arm}: fixed arm is not SkillSpecV11")
                if spec.question_type not in wanted:
                    raise PairingError(
                        f"{arm}: Skill question_type {spec.question_type!r} "
                        f"does not match panel {sorted(wanted)}")
            snapshot_refs = dict(_fixed_arm_snapshot_refs or {})
            manifest_hashes = dict(_fixed_arm_manifest_hashes or {})
            if set(snapshot_refs) != set(ARMS) or set(manifest_hashes) != set(ARMS):
                raise PairingError(
                    "fixed Skill evaluation requires snapshot ref/hash for both arms")
            if any(not value for value in snapshot_refs.values()):
                raise PairingError("fixed arm snapshot refs must not be empty")
            if any(
                len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                for value in manifest_hashes.values()
            ):
                raise PairingError(
                    "fixed arm manifest hashes must be lowercase sha256 values")
        else:
            snapshot_refs = {}
            manifest_hashes = {}
        all_items = _index([{"qa_id": it.episode.qa_id, "item": it} for it in items])
        chosen = {q: r["item"] for q, r in all_items.items()
                  if canonical_task(r["item"].episode.question_type) in wanted}
        expected = list(expected_qa_ids) if expected_qa_ids is not None else list(chosen)
        _index([{"qa_id": q} for q in expected])
        _same_ids(expected, chosen, "question panel")
        if not chosen:
            raise PairingError("empty question panel")
        initial_states = {
            q: initial_state_for_item(item, artifact_paths[q])
            for q, item in chosen.items() if item.input_error is None
            and q in artifact_paths}
        excluded = []
        if fixed_pair:
            target_states = {s.evidence_state for s in fixed_arms.values()}
            if len(target_states) != 1:
                raise PairingError("parent/candidate initial evidence slots must match")
            target_state = next(iter(target_states))
            excluded = [q for q in expected if initial_states.get(q) != target_state]
            expected = [q for q in expected if q not in excluded]
            chosen = {q: chosen[q] for q in expected}
            manifest.update(target_evidence_state=target_state,
                            excluded_qa_ids=excluded, initial_evidence_states=initial_states)
            if not expected:
                manifest.update(status="incomplete", qa_ids=[], reason="no_target_slot_samples")
                _write(root / "manifest.json", manifest)
                summary = {"status": "incomplete", "reason": "no_target_slot_samples",
                           "n_pairs": 0, "n_solver_runs": 0,
                           "formal_result_eligible": False, "excluded_qa_ids": excluded}
                _write(root / "summary.json", summary)
                (root / "paired_results.jsonl").write_text("")
                return summary
        costs = {
            str(key): float(value)
            for key, value in (reconstruction_costs or {}).items() if key not in excluded
        }
        if set(costs) - set(expected):
            raise PairingError(
                f"reconstruction costs contain unexpected qa_ids: "
                f"{sorted(set(costs) - set(expected))}")
        if any(not math.isfinite(value) or value < 0.0 for value in costs.values()):
            raise PairingError("reconstruction costs must be finite non-negative seconds")
        runnable = {q for q, item in chosen.items() if item.input_error is None}
        if runnable - set(artifact_paths):
            raise PairingError(
                f"missing artifact mappings: {sorted(runnable - set(artifact_paths))}")
        if any(it.episode.split == "final_test" for it in chosen.values()) and not base_cfg.allow_final_test:
            raise PairingError("final_test requires allow_final_test")
        cfg = replace(copy.deepcopy(base_cfg), seed=int(seed), skills=[],
                      active_snapshot_ref=snapshot_id,
                      active_snapshot_manifest_sha256="")
        common = _common_config(cfg)
        config_hash = _hash(common)
        library = Path(library_root).resolve()
        if fixed_pair:
            specs = list(fixed_arms.values())
            identities_by_arm = {
                arm: {
                    "skill_version": f"{spec.skill_id}@{spec.version}",
                    "content_sha256": spec.content_sha256,
                    "snapshot_ref": snapshot_refs[arm],
                    "manifest_sha256": manifest_hashes[arm],
                }
                for arm, spec in fixed_arms.items()
            }
            fixed_identity_sha = _hash(identities_by_arm)
            snapshot = {
                "schema_version": "fixed-skill-pair-v11/1.1",
                "snapshot_id": f"E11-{fixed_identity_sha[:20]}",
                "manifest_hash": fixed_identity_sha,
                "arms": identities_by_arm,
            }
            skill_manifest = {
                "schema_version": "fixed-skill-pair-manifest-v11/1.1",
                "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
                "manifest_sha256": fixed_identity_sha,
                "arms": identities_by_arm,
            }
            by_slot = {}
        else:
            snapshot = json.loads(
                (library / f"snapshots/snapshot_{snapshot_id}.json").read_text())
            if snapshot.get("snapshot_id") != snapshot_id:
                raise PairingError("wrong frozen S0 snapshot")
            specs = validate_v11_snapshot(
                snapshot, library_root=library,
                method_context_max_chars=cfg.retrieval_policy.method_context_max_chars)
            skill_manifest = json.loads(
                resolve_source_like_ref(library, snapshot["manifest_ref"]).read_text())
            payload = {k: v for k, v in skill_manifest.items()
                       if k != "manifest_sha256"}
            if payload != _manifest_for(snapshot) or canonical_json_sha256(payload) != \
                    snapshot.get("manifest_hash") or skill_manifest.get(
                        "manifest_sha256") != snapshot.get("manifest_hash"):
                raise PairingError("frozen Skill manifest mismatch")
            by_slot = {(s.question_type, s.evidence_state): s for s in specs}
            if any((task, state) not in by_slot for task in wanted
                   for state in ("g0m0", "g1m0", "g1m1")):
                raise PairingError(
                    "missing frozen task/evidence Skills")
        quality = quality_contract()
        if quality_confirmations is not None:
            unknown = set(quality_confirmations) - set(expected)
            if unknown:
                raise PairingError(
                    f"quality confirmations contain unexpected qa_ids: {sorted(unknown)}")
        confirmation_by_qa = {
            q: (quality_confirmations.get(q) if quality_confirmations is not None
                else quality_confirmation)
            for q in expected
        }
        formal_requested = (
            client_factory is None and os.environ.get(MODE_ENV) != "development"
            and any(c is not None and c.get("confirmed") is True
                    for c in confirmation_by_qa.values()))
        if formal_requested and (
                cfg.max_images < 32
                or any(it.episode.frame_set is None
                       or (len(it.episode.frame_set.frame_ids) != 32
                           and it.episode.frame_set.n_total_frames >= 32)
                       for it in chosen.values() if it.input_error is None)):
            raise PairingError("Formal experiments require the frozen 32-frame input contract")
        for q, confirmation in confirmation_by_qa.items():
            if confirmation is None:
                continue
            if not (
                confirmation.get("confirmed") in (True, False)
                and confirmation.get("sha256", confirmation.get("quality_contract_sha256"))
                == quality["sha256"]
                and confirmation.get("evidence_ref")
            ):
                raise PairingError(
                    f"{q}: quality confirmation must match current contract and include evidence_ref")
        identities = {
            q: (input_error_identity(it) if it.input_error is not None
                else input_identity(it, artifact_paths[q]))
            for q, it in chosen.items()
        }
        # M2: confirm the already validated nested identity before M5/client creation.
        # Contract confirmation is independent of the per-scene geometry decision.
        for q, confirmation in confirmation_by_qa.items():
            if confirmation is None or chosen[q].input_error is not None:
                continue
            identity = identities[q]
            episode = identity["episode"]
            if (confirmation.get("scene_name") is not None
                    and confirmation["scene_name"] != episode["scene_name"]):
                raise PairingError(f"{q}: scene differs from quality confirmation")
            expected_artifact = confirmation.get("artifact", {})
            if not isinstance(expected_artifact, dict):
                raise PairingError(f"{q}: quality confirmation artifact must be a mapping")
            for key, actual in (
                ("sha256", identity["artifact_json_sha256"]),
                ("frame_set_hash", episode["frame_set"]["frame_set_hash"]),
                ("artifact_id", identity["artifact_id"]),
                ("artifact_version", identity["artifact_version"]),
            ):
                if key in expected_artifact and expected_artifact[key] != actual:
                    raise PairingError(f"{q}: {key} differs from quality confirmation")
            original = ReconstructionArtifact.model_validate_json(
                Path(artifact_paths[q]).read_bytes())
            if confirmation.get("confirmed") is True and (
                    original.quality is None
                    or original.quality.gate_thresholds != quality["default_thresholds"]):
                raise PairingError(f"{q}: artifact quality thresholds differ from confirmed contract")
        # One alleged artifact identity cannot refer to different data across questions.
        seen = {}
        for q, identity in identities.items():
            if chosen[q].input_error is not None:
                continue
            key = (chosen[q].episode.dataset, identity["artifact_id"], identity["artifact_version"])
            value = {k: identity[k] for k in ("artifact_json_sha256", "pixels")}
            if key in seen and seen[key] != value:
                raise PairingError(f"{q}: reused artifact identity has different content/frames")
            seen[key] = value
        binding_store = (
            PreparedBindingStore(prepared_binding_store)
            if binding_store_enabled
            else None
        )
        binding_references = {
            str(qa_id): copy.deepcopy(dict(reference))
            for qa_id, reference in (prepared_binding_refs or {}).items()
        }
        runnable_ids = {
            q for q in expected if chosen[q].input_error is None
        }
        if binding_store_enabled and set(binding_references) != runnable_ids:
            raise PairingError(
                "prepared binding references must exactly cover runnable panel: "
                f"expected={sorted(runnable_ids)}, "
                f"actual={sorted(binding_references)}"
            )
        scene_audit_groups: dict[str, list[str]] = {}
        for q in sorted(runnable_ids):
            scene_audit_groups.setdefault(
                chosen[q].episode.scene_name,
                [],
            ).append(q)
        code_root = Path(runner.__file__).parents[1]
        manifest.update({
            "qa_ids": expected, "question_types": sorted(wanted),
            "evaluation_mode": (
                "fixed_parent_candidate" if fixed_pair else "skill_ablation"),
            "common_config": common, "config_sha256": config_hash,
            "inputs": identities, "snapshot": snapshot, "skill_manifest": skill_manifest,
            "quality_contract": quality,
            "quality_confirmation": quality_confirmation,
            "quality_confirmations": confirmation_by_qa if quality_confirmations is not None else None,
            "quality_confirmation_status": (
                "confirmed" if all(
                    c is not None and c.get("confirmed") is True
                    for c in confirmation_by_qa.values()
                ) else "unconfirmed"),
            "reconstruction_costs_s": dict(sorted(costs.items())),
            "client_mode": "injected" if client_factory else "bailian",
            "prepared_binding_store": (
                {
                    "root": str(binding_store.root),
                    "keys": {
                        q: binding_references[q]["key"]
                        for q in sorted(binding_references)
                    },
                    "scene_audit_groups": scene_audit_groups,
                }
                if binding_store is not None
                else None
            ),
            "run_mode": os.environ.get(MODE_ENV, "formal"),
            "code_identity": current_code_identity(),
            "source_sha256": {str(p.relative_to(code_root)): _file_hash(p)
                              for p in sorted(code_root.rglob("*.py"))},
        })
        _write(root / "manifest.json", manifest)
        _write(root / "frozen_skill_snapshot.json", snapshot)
        for spec in specs:
            path = root / "skills" / spec.skill_id / spec.version / "SKILL.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(spec.skill_md, encoding="utf-8")

        prior = None
        if prepared_manifest is not None:
            prior = json.loads(Path(prepared_manifest).read_bytes())
            if prior.get("status") != "prepared":
                raise PairingError("supplied preparation is not sealed")
            for name, actual in (
                ("seed", int(seed)), ("config_sha256", config_hash),
                ("snapshot", snapshot), ("quality_contract", quality),
                ("source_sha256", manifest["source_sha256"]),
            ):
                if prior.get(name) != actual:
                    raise PairingError(f"supplied preparation {name} changed")
            if not set(expected).issubset(prior.get("qa_ids", [])):
                raise PairingError("supplied preparation does not cover selected questions")
            if any(prior["inputs"].get(q) != identities[q] for q in expected):
                raise PairingError("supplied preparation input identity changed")

        def client(phase, qa_id, store):
            inner = (client_factory(phase, qa_id) if client_factory
                     else runner._make_online_client(cfg))
            return _RecordingClient(inner, seed=cfg.seed, specs=specs, store=store)

        prepared, prepare_errors, preparation_detector_attempts = {}, {}, {}
        preparation_work = {}
        # Freeze *all* common inputs before starting either arm.
        for q in expected:
            item = chosen[q]
            if item.input_error is not None:
                prepared[q] = {
                    "input_error": item.input_error.model_dump(mode="json"),
                    "sha256": identities[q]["sha256"],
                }
                continue
            key = hashlib.sha256(q.encode()).hexdigest()
            frozen = root / "frozen" / key
            if prior is not None:
                original = prior["prepared"][q]
                source = Path(original["directory"])
                if tree_identity(source)["sha256"] != original["sha256"]:
                    raise PairingError(f"{q}: supplied preparation tree changed")
                _clone_tree(source, frozen)
                prepared[q] = {**original, "directory": str(frozen)}
                if tree_identity(frozen)["sha256"] != original["sha256"]:
                    raise PairingError(f"{q}: copied preparation tree changed")
                if q in prior.get("preparation_errors", {}):
                    prepare_errors[q] = prior["preparation_errors"][q]
                preparation_detector_attempts[q] = prior.get(
                    "preparation_detector_attempts", {}).get(q, [])
                continue
            art = _copy_base(
                Path(artifact_paths[q]).resolve(),
                frozen,
                link_files=binding_store is not None,
            )
            art = runner._prepare_m4_quality(art, item.pixels, persist_quality=False)
            _write(frozen / "episode.json", item.episode)
            for i, pixels in enumerate(item.pixels):
                np.save(frozen / f"frame_{i:04d}.npy", pixels)
            local_pixels = [p.copy() for p in item.pixels]
            if binding_store is not None:
                reference = binding_references[q]
                audit_qa_ids = list(
                    (
                        reference.get("request_identity") or {}
                    ).get("scene_audit_qa_ids") or []
                )
                if (
                    q not in audit_qa_ids
                    or len(audit_qa_ids) > 8
                    or len(set(audit_qa_ids)) != len(audit_qa_ids)
                ):
                    raise PairingError(
                        f"{q}: invalid sealed scene audit group"
                    )
                expected_request = prepared_binding_request_identity(
                    input_identity=identities[q],
                    cfg=cfg,
                    scene_audit_qa_ids=audit_qa_ids,
                )
                expected_reference = binding_store.reference(
                    binding_store.key_for(expected_request)
                )
                for name in (
                    "store_root",
                    "key",
                    "request_identity",
                    "scene_artifact_ref",
                    "scene_input_identity_sha256",
                    "manifest_sha256",
                ):
                    if reference.get(name) != expected_reference.get(name):
                        raise PairingError(
                            f"{q}: prepared binding reference {name} changed"
                        )
                binding = binding_store.load(
                    reference["key"],
                    expected_request_identity=expected_request,
                    scene_artifact=artifact_paths[q],
                    scene_input_identity=identities[q],
                )
                preparation_detector_attempts[q] = list(
                    binding.stats.get("detector_attempts") or []
                )
            else:
                trace = _BoundTrace(root / "preparation" / key, {
                    "experiment_id": experiment_id,
                    "arm": "prepare",
                    "run_id": f"{experiment_id}:prepare:{q}",
                    "qa_id": q,
                })
                llm = client("prepare", q, trace)
                with capture_detector_failures() as detection_errors:
                    binding = PreparedObjectBinding(
                        *runner._bind_objects_best_effort(
                            art,
                            local_pixels,
                            None,
                            item.episode.model_copy(deep=True),
                            cfg,
                            llm,
                        )
                    )
                preparation_detector_attempts[q] = list(
                    binding.stats.get("detector_attempts") or []
                )
                if llm.contract_errors:
                    raise PairingError("; ".join(llm.contract_errors))
                llm.failures.extend(detection_errors)
                if binding.stats.get("binding_error"):
                    llm.failures.append(binding.stats["binding_error"])
                if llm.failures:
                    prepare_errors[q] = list(llm.failures)
            _, _, art = runner._scene_from_artifact(
                art, local_pixels, binding.stats, binding.objects, item.episode,
                objects_materialized=binding.materialized, persist_quality=False)
            if _pixels_identity(local_pixels) != identities[q]["pixels"]:
                raise PairingError(f"{q}: preparation modified frozen pixels")
            preparation_work[q] = {
                "artifact": art, "binding": binding, "frozen": frozen,
                "binding_reference": (
                    copy.deepcopy(binding_references[q])
                    if binding_store is not None
                    else None
                ),
            }

        # Cross-question M5 integrity audit runs before any frozen tree identity is computed.
        from skill3d.segmentation.sam2_tracker import (
            audit_semantic_mask_conflicts, refresh_grounding_coverage,
        )

        scene_groups: dict[str, list[str]] = {}
        for q, work in preparation_work.items():
            art = work["artifact"]
            scene_identity = str(
                art.reconstruction_input_identity
                or _hash({
                    "artifact_id": art.artifact_id,
                    "artifact_version": art.artifact_version,
                    "frame_set_hash": art.frame_set_hash,
                }))
            work["scene_identity"] = scene_identity
            if work["binding_reference"] is None:
                scene_groups.setdefault(scene_identity, []).append(q)
        for scene_identity, qa_ids in sorted(scene_groups.items()):
            audited, audits = audit_semantic_mask_conflicts(
                [(q, preparation_work[q]["binding"].objects) for q in sorted(qa_ids)],
                scene_identity=scene_identity)
            for q in qa_ids:
                binding = preparation_work[q]["binding"]
                binding.objects = audited[q]
                from skill3d.segmentation.binding_validation import validate_binding
                binding.objects = [
                    obj.model_copy(update={"binding_validation": validate_binding(
                        obj, generation_sha256=(binding.stats.get("generation") or {}).get(
                            "sha256", ""))})
                    if obj.grounding_status == "question_driven" else obj
                    for obj in binding.objects]
                binding.stats["semantic_conflict_audits"] = [
                    audit for audit in audits if q in audit["qa_ids"]]
                if binding.stats.get("grounding"):
                    binding.stats["grounding"] = refresh_grounding_coverage(
                        binding.stats["grounding"], binding.objects)
                # M4 must consume the audited coverage, before the tree is frozen.
                _, _, updated_art = runner._scene_from_artifact(
                    preparation_work[q]["artifact"], chosen[q].pixels, binding.stats,
                    binding.objects, chosen[q].episode,
                    objects_materialized=binding.materialized, persist_quality=False)
                preparation_work[q]["artifact"] = updated_art

        for q in expected:
            if chosen[q].input_error is not None:
                continue
            art = (preparation_work[q]["artifact"] if q in preparation_work else
                   ReconstructionArtifact.model_validate_json(
                       (Path(prepared[q]["directory"]) / "artifact.json").read_bytes()))
            state = skill_evidence_state(runner.build_scene_state(
                art, question_type=canonical_task(chosen[q].episode.question_type)).evidence_profile)
            if state != initial_states[q]:
                raise PairingError(f"{q}: shared preparation changed the initial evidence slot")
        manifest["initial_skill_states"] = dict(initial_states)

        # Room measurements are scene-level deterministic infrastructure. Compute once per
        # request identity before either arm and freeze success/failure identically.
        room_receipts: dict[str, dict] = (
            prepared_room_cache if prepared_room_cache is not None else {})
        new_room_receipts = []
        prepared_measurement_by_qa: dict[str, dict] = {}
        for q, work in sorted(preparation_work.items()):
            task = canonical_task(chosen[q].episode.question_type)
            if task != "room_size_estimation":
                continue
            art = work["artifact"]
            scoped = runner.build_scene_state(art, question_type=task)
            if not scoped.metric_task_authorized(task):
                continue
            request = prepared_room_request(
                point_map_sha256=file_sha256(art.point_map),
                metric_scale=float(art.metric_scale),
                frozen_input_identity=frozen_reconstruction_identity(art),
                world_up=art.world_up)
            request_sha = request["request_sha256"]
            if request_sha not in room_receipts:
                room_receipts[request_sha] = _prepare_room_measurement(
                    art, timeout_s=300.0)
                new_room_receipts.append(room_receipts[request_sha])
            receipt = copy.deepcopy(room_receipts[request_sha])
            work["binding"].prepared_measurements["plane_fit_room_size"] = receipt
            work["binding"].stats["prepared_measurement_receipts"] = {
                "plane_fit_room_size": receipt["receipt_sha256"]}
            work["binding"].notes.append(
                "M6 room measurement prepared before paired solving: "
                f"status={receipt['status']} receipt={receipt['receipt_sha256']}")
            prepared_measurement_by_qa[q] = {
                "tool": "plane_fit_room_size",
                "status": receipt["status"],
                "request_sha256": request_sha,
                "receipt_sha256": receipt["receipt_sha256"],
                "runtime_s": receipt["runtime_s"],
            }

        # Only audited bindings enter the immutable trees shared by B01 and B11.
        for q in expected:
            if q not in preparation_work:
                continue
            work = preparation_work[q]
            art, binding, frozen = (
                work["artifact"], work["binding"], work["frozen"])
            if work["binding_reference"] is None:
                _save_binding(binding, frozen)
            else:
                _write(
                    frozen / "binding_store_ref.json",
                    {
                        **work["binding_reference"],
                        "scene_input_identity": identities[q],
                    },
                )
                if binding.prepared_measurements:
                    _write(
                        frozen / "binding_overlay.json",
                        {
                            "prepared_measurements": (
                                binding.prepared_measurements
                            ),
                            "notes": binding.notes,
                            "stats_updates": {
                                "prepared_measurement_receipts": (
                                    binding.stats.get(
                                        "prepared_measurement_receipts"
                                    )
                                    or {}
                                ),
                            },
                        },
                    )
            (frozen / "artifact.json").write_text(art.model_dump_json(), encoding="utf-8")
            prepared[q] = {"directory": str(frozen), **tree_identity(frozen),
                           "gate_thresholds": art.quality.gate_thresholds if art.quality else None}
            if q in prepared_measurement_by_qa:
                prepared[q]["prepared_measurement"] = prepared_measurement_by_qa[q]
            confirmation = confirmation_by_qa[q]
            if confirmation and confirmation.get("confirmed") is True and (
                prepared[q]["gate_thresholds"] != quality["default_thresholds"]
            ):
                raise PairingError(f"{q}: artifact quality thresholds differ from confirmed contract")
        manifest.update({"prepared": prepared, "preparation_errors": prepare_errors,
                         "preparation_detector_attempts": preparation_detector_attempts,
                         "prepared_measurement_receipts": prepared_measurement_by_qa,
                         "prepared_measurement_cost_s": sum(
                             float(receipt["runtime_s"])
                             for receipt in new_room_receipts),
                         "status": "running"})
        _write(root / "manifest.json", manifest)
        if prepare_only:
            manifest["status"] = "prepared"
            _write(root / "manifest.json", manifest)
            summary = {
                "experiment_id": experiment_id, "status": "prepared",
                "n_planned": len(expected), "n_solver_runs": 0,
                "label_access": False,
            }
            _write(root / "summary.json", summary)
            return summary
        results = {arm: [] for arm in ARMS}
        pairs = []
        for q in expected:
            item = chosen[q]
            task = canonical_task(item.episode.question_type)
            if item.input_error is not None:
                input_hash = identities[q]["sha256"]
                for arm in ARMS:
                    spec = fixed_arms[arm] if fixed_pair else (
                        None)
                    binding = (
                        SkillEvaluationBinding(
                            mode="fixed_skill_evaluation",
                            arm="parent" if arm == "B01" else "candidate",
                            skill_id=spec.skill_id,
                            skill_version=f"{spec.skill_id}@{spec.version}",
                            content_sha256=spec.content_sha256,
                            bypassed_component="retrieval_selection",
                        )
                        if fixed_pair and spec is not None else None
                    )
                    arm_root = root / "arms" / arm / hashlib.sha256(q.encode()).hexdigest()
                    identity = {
                        "experiment_id": experiment_id,
                        "arm": arm,
                        "qa_id": q,
                        "run_id": f"{experiment_id}:{arm}:{q}",
                    }
                    trace = _BoundTrace(arm_root / "trace", identity)
                    arm_cfg = replace(
                        copy.deepcopy(cfg),
                        skills=[copy.deepcopy(spec)] if spec is not None else [],
                        reuse_artifact=None,
                        active_snapshot_ref=(
                            snapshot_refs[arm] if fixed_pair
                            else snapshot_id if arm == "B11" else "none"),
                        active_snapshot_manifest_sha256=(
                            manifest_hashes[arm] if fixed_pair
                            else snapshot["manifest_hash"] if arm == "B11" else ""),
                        evaluation_binding=binding,
                        trace_dir=str(trace.root),
                        recon_dir=str(arm_root / "artifacts"),
                        work_dir=str(arm_root / "work"),
                    )
                    llm = client(arm, q, trace)
                    out = runner.run_episode(
                        item.episode.model_copy(deep=True),
                        [],
                        arm_cfg,
                        trace_store=trace,
                        llm=llm,
                        input_error=item.input_error,
                    )
                    _write(arm_root / "outcome.json", asdict(out))
                    row = _arm_result(
                        out,
                        llm,
                        identity,
                        input_hash,
                        config_hash,
                        ({
                            f"{spec.skill_id}@{spec.version}": spec.content_sha256
                        } if spec is not None else {}),
                        options=item.episode.options,
                        reconstruction_cost_s=costs.get(q),
                    )
                    row["initial_tree_sha256"] = None
                    _write(arm_root / "result.json", row)
                    results[arm].append(row)
                pairs = pair_results(
                    results["B01"], results["B11"], expected[:len(results["B01"])])
                (root / "paired_results.jsonl").write_bytes(
                    b"".join(_encoded(pair) + b"\n" for pair in pairs))
                progress = summarize_pairs(pairs)
                progress.update({"experiment_id": experiment_id, "n_planned": len(expected)})
                if len(pairs) != len(expected):
                    progress["status"] = "running"
                _write(root / "summary.json", progress)
                continue
            frozen = Path(prepared[q]["directory"])
            input_hash = _hash({"original": identities[q]["sha256"],
                                "prepared": prepared[q]["sha256"]})
            for arm in ARMS:
                spec = fixed_arms[arm] if fixed_pair else (
                    by_slot[(task, initial_states[q])] if arm == "B11" else None)
                binding = (
                    SkillEvaluationBinding(
                        mode="fixed_skill_evaluation",
                        arm="parent" if arm == "B01" else "candidate",
                        skill_id=spec.skill_id,
                        skill_version=f"{spec.skill_id}@{spec.version}",
                        content_sha256=spec.content_sha256,
                        bypassed_component="retrieval_selection",
                    )
                    if fixed_pair and spec is not None else None
                )
                arm_root = root / "arms" / arm / frozen.name
                local = arm_root / "artifacts"
                _clone_tree(
                    frozen,
                    local,
                    link_binary=binding_store is not None,
                )
                actual = tree_identity(local)["sha256"]
                if actual != prepared[q]["sha256"]:
                    raise PairingError(f"{q}/{arm}: copied initial inputs differ")
                identity = {"experiment_id": experiment_id, "arm": arm, "qa_id": q,
                            "run_id": f"{experiment_id}:{arm}:{q}"}
                trace = _BoundTrace(arm_root / "trace", identity)
                arm_cfg = replace(copy.deepcopy(cfg),
                                  skills=[copy.deepcopy(spec)] if spec is not None else [],
                                  reuse_artifact=str(local / "artifact.json"),
                                  active_snapshot_ref=(
                                      snapshot_refs[arm] if fixed_pair
                                      else snapshot_id if arm == "B11" else "none"),
                                  active_snapshot_manifest_sha256=(
                                      manifest_hashes[arm] if fixed_pair
                                      else snapshot["manifest_hash"]
                                      if arm == "B11" else ""),
                                  evaluation_binding=binding,
                                  trace_dir=str(trace.root), recon_dir=str(local),
                                  work_dir=str(arm_root / "work"))
                if _hash(_common_config(arm_cfg)) != config_hash:
                    raise PairingError(f"{q}/{arm}: common config changed")
                llm = client(arm, q, trace)
                if q in prepare_errors:
                    row = _incomplete_row(
                        identity, task=task, is_mca=classify(item.episode).is_mca,
                        input_hash=input_hash, config_hash=config_hash, errors=prepare_errors[q])
                    row["reconstruction_cost_s"] = costs.get(q)
                else:
                    try:
                        with capture_detector_failures() as detection_errors:
                            out = runner.run_episode(
                                item.episode.model_copy(deep=True),
                                [np.load(p, allow_pickle=False) for p in sorted(local.glob("frame_*.npy"))],
                                arm_cfg, trace_store=trace, llm=llm, prepared_binding=_load_binding(local))
                        llm.failures.extend(detection_errors)
                        _write(arm_root / "outcome.json", asdict(out))
                        row = _arm_result(
                            out, llm, identity, input_hash, config_hash,
                            ({
                                f"{spec.skill_id}@{spec.version}": spec.content_sha256
                            } if spec is not None else {}),
                            options=item.episode.options,
                            reconstruction_cost_s=costs.get(q))
                        if out.initial_skill_selection.get("evidence_state") != initial_states[q]:
                            raise PairingError(f"{q}/{arm}: runtime initial slot differs from shared preparation")
                    except Exception as exc:
                        row = _incomplete_row(
                            identity, task=task, is_mca=classify(item.episode).is_mca,
                            input_hash=input_hash, config_hash=config_hash,
                            errors=[f"runner exception: {type(exc).__name__}: {exc}"])
                        row["requests"] = llm.requests
                if llm.contract_errors:
                    raise PairingError("; ".join(llm.contract_errors))
                row["initial_tree_sha256"] = actual
                _write(arm_root / "result.json", row)
                results[arm].append(row)
                if tree_identity(frozen)["sha256"] != prepared[q]["sha256"]:
                    raise PairingError(f"{q}/{arm}: frozen prepared source mutated")
                if input_identity(item, artifact_paths[q]) != identities[q]:
                    raise PairingError(f"{q}/{arm}: original input mutated")
            pairs = pair_results(results["B01"], results["B11"], expected[:len(results["B01"])])
            (root / "paired_results.jsonl").write_bytes(
                b"".join(_encoded(p) + b"\n" for p in pairs))
            progress = summarize_pairs(pairs)
            progress.update({"experiment_id": experiment_id, "n_planned": len(expected)})
            if len(pairs) != len(expected):
                progress["status"] = "running"
            _write(root / "summary.json", progress)
        for q in expected:
            if chosen[q].input_error is not None:
                continue
            if tree_identity(Path(prepared[q]["directory"]))["sha256"] != prepared[q]["sha256"]:
                raise PairingError(f"{q}: frozen source changed later in the experiment")
            if input_identity(chosen[q], artifact_paths[q]) != identities[q]:
                raise PairingError(f"{q}: original input changed later in the experiment")
        summary = summarize_pairs(pairs)
        summary.update({"experiment_id": experiment_id, "n_planned": len(expected),
                        "formal_result_eligible": (
                        all(c is not None and c.get("confirmed") is True
                            for c in confirmation_by_qa.values())
                        and client_factory is None
                        and os.environ.get(MODE_ENV) != "development"
                        and bool(cfg.model_weights_sha256) and bool(cfg.environment_sha256))
                        and summary["status"] == "completed"
                        and summary["formal_scoring_eligible"]
                        and summary["n_input_error_pairs"] < summary["n_pairs"]
                        and all(p["comparable"] for p in pairs)})
        summary["run_mode"] = manifest["run_mode"]
        manifest["status"] = summary["status"]
        _write(root / "manifest.json", manifest)
        _write(root / "summary.json", summary)
        return summary
    except Exception as exc:
        manifest.update({"status": "invalid" if isinstance(exc, PairingError) else "incomplete",
                         "error": f"{type(exc).__name__}: {exc}"})
        _write(root / "manifest.json", manifest)
        _write(root / "failure.json", {"experiment_id": experiment_id,
                                      "status": manifest["status"], "error": manifest["error"]})
        summary_path = root / "summary.json"
        failed_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        failed_summary.update({"experiment_id": experiment_id, "status": manifest["status"],
                               "error": manifest["error"], "formal_result_eligible": False})
        _write(summary_path, failed_summary)
        raise


def run_skill_pair_evaluation_v11(
    items: Sequence[EpisodeItem],
    *,
    artifact_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    base_cfg: OnlineRunConfig,
    seed: int,
    parent: SkillSpecV11,
    candidate: SkillSpecV11,
    parent_snapshot_ref: str,
    parent_manifest_sha256: str,
    candidate_snapshot_ref: str,
    candidate_manifest_sha256: str,
    expected_qa_ids: Sequence[str] | None = None,
    client_factory: Callable[[str, str], object] | None = None,
    quality_confirmation: dict | None = None,
    reconstruction_costs: Mapping[str, float] | None = None,
    prepared_binding_store: str | Path | None = None,
    prepared_binding_refs: Mapping[str, Mapping] | None = None,
) -> dict:
    """Run E11 with explicit parent/candidate fixed injection on frozen inputs."""
    if parent.slot_key != candidate.slot_key:
        raise PairingError("parent and candidate task/evidence slot must match")
    return run_skill_ablation_v11(
        items,
        artifact_paths=artifact_paths,
        output_dir=output_dir,
        base_cfg=base_cfg,
        seed=seed,
        question_types=[parent.question_type],
        expected_qa_ids=expected_qa_ids,
        client_factory=client_factory,
        quality_confirmation=quality_confirmation,
        reconstruction_costs=reconstruction_costs,
        prepared_binding_store=prepared_binding_store,
        prepared_binding_refs=prepared_binding_refs,
        _fixed_skill_arms={"B01": parent, "B11": candidate},
        _fixed_arm_snapshot_refs={
            "B01": parent_snapshot_ref,
            "B11": candidate_snapshot_ref,
        },
        _fixed_arm_manifest_hashes={
            "B01": parent_manifest_sha256,
            "B11": candidate_manifest_sha256,
        },
    )
