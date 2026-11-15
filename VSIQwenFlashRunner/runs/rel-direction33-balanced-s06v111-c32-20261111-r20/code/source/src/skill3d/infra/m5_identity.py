"""Content identities for M5 generation; incomplete identities disable disk reuse."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from importlib import metadata, util
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

M5_IDENTITY_VERSION = "m5-generation-identity-v2-question-driven"
HASH_BLOCK_BYTES = 8 * 1024 * 1024
FILE_HASH_SETTLE_NS = 2_000_000_000
SOURCE_REFS = (
    "infra/m5_identity.py", "infra/version_lock.py",
    "segmentation/sam2_tracker.py", "segmentation/open_vocab_detector.py",
    "segmentation/question_binding.py", "segmentation/binding_validation.py",
    "segmentation/mask_semantic_review.py",
    "segmentation/intern_decision_client.py", "segmentation/intern_decision_protocol.py",
    "segmentation/intern_binary_review.py",
    "segmentation/intern_review_pool.py",
    "segmentation/groundingdino_startup.py", "coords.py",
    "reconstruction/image_grid.py", "tools/category_match.py",
    "reconstruction_gate/confidence_map.py",
    "schemas/reconstruction.py", "synthesis/vllm_client.py",
    "synthesis/bailian_client.py",
)


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def content_sha256(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _stat_stamp(stat) -> tuple:
    return (stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def _file_stamp(path: Path) -> tuple:
    return _stat_stamp(path.stat())


@lru_cache(maxsize=8192)
def _file_digest(path: str, stamp: tuple) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        if _stat_stamp(os.fstat(stream.fileno())) != stamp:
            raise ValueError(f"identity file changed during hashing: {path}")
        for block in iter(lambda: stream.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
        if _stat_stamp(os.fstat(stream.fileno())) != stamp:
            raise ValueError(f"identity file changed during hashing: {path}")
    if _file_stamp(Path(path)) != stamp:
        raise ValueError(f"identity file changed during hashing: {path}")
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict:
    """Reuse hashes in this process only while inode, size and timestamps match."""
    # Preserve the OS meaning of symlink/.. while avoiding component-by-component
    # resolution on every lookup. The opened descriptor must match the path stamp.
    path = Path(path).absolute()
    stamp = _file_stamp(path)
    settled = time.time_ns() - max(stamp[3:]) > FILE_HASH_SETTLE_NS
    digest = (_file_digest(str(path), stamp)
              if settled
              else _file_digest.__wrapped__(str(path), stamp))
    result = {"size": stamp[2], "sha256": digest}
    if _file_stamp(path) != stamp:
        raise ValueError(f"identity file changed during lookup: {path}")
    return result


def array_identity(value, *, block_bytes: int = HASH_BLOCK_BYTES):
    """Hash logical C-order bytes in bounded chunks, including noncontiguous arrays."""
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("M5 identity cannot hash object arrays")
    if block_bytes <= 0:
        raise ValueError("hash block size must be positive")
    digest = hashlib.sha256()
    if array.flags.c_contiguous:
        view = memoryview(array.reshape(-1)).cast("B")
        for start in range(0, len(view), block_bytes):
            digest.update(view[start:start + block_bytes])
    else:
        with np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                       op_flags=["readonly"], order="C",
                       buffersize=max(1, block_bytes // array.dtype.itemsize)) as it:
            for block in it:
                digest.update(block.tobytes(order="C"))
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "sha256": digest.hexdigest()}


def source_identity(root: Path, refs=SOURCE_REFS) -> dict:
    result = {}
    for name in refs:
        path = root / name
        if path.is_dir():
            for child in sorted(path.rglob("*.py")):
                result[child.relative_to(root).as_posix()] = file_identity(child)
        elif path.is_file():
            result[name] = file_identity(path)
        else:
            result[name] = None
    return result


def package_identity(name: str, *, suffixes=(".py", ".yaml", ".so")):
    """Hash the actual installed implementation, without importing CUDA extensions."""
    try:
        spec = util.find_spec(name)
        if spec is None or not spec.origin:
            return None
        root = Path(spec.origin).parent
        files = {p.relative_to(root).as_posix(): file_identity(p)
                 for p in sorted(root.rglob("*"))
                 if p.is_file() and p.suffix in suffixes}
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = "unregistered"
        return {"version": version, "files": files} if files else None
    except (OSError, ImportError, ValueError):
        return None


def sam2_identity(checkpoint, config, predictor=None,
                  inference_mode: str = "video_bidirectional") -> dict | None:
    if predictor is not None:
        # A custom predictor is never identified by its class name or shape.
        return getattr(predictor, "_skill3d_m5_identity", None)
    from skill3d.infra.version_lock import sam2_runtime_identity
    try:
        package = package_identity("sam2")
        if not package or not checkpoint or not Path(checkpoint).is_file():
            return None
        from hydra import compose
        from omegaconf import OmegaConf
        import sam2  # noqa: F401 - installs the package's Hydra search path

        # These are build_sam2_video_predictor's default inference overrides.
        overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=8",
        ]
        resolved = OmegaConf.to_container(compose(config_name=config, overrides=overrides),
                                         resolve=True)
        settings = {
            "mode": "eval", "apply_postprocessing": True,
            "vos_optimized": False, "autocast": "cuda:bfloat16",
            "inference_api": str(inference_mode),
        }
        if inference_mode == "video_bidirectional":
            settings.update(
                offload_video_to_cpu=True, offload_state_to_cpu=True,
                propagation="forward_then_reverse", jpeg_quality=92)
        return {
            "checkpoint": file_identity(checkpoint), "config": resolved,
            "package": package, "runtime": sam2_runtime_identity(),
            "settings": settings,
        }
    except Exception:  # noqa: BLE001 - usable inference need not imply cache eligibility
        return None


def process_identity(pid: int) -> dict:
    """Bind receipts to an actual process lifetime, protecting against PID reuse."""
    base = Path("/proc") / str(int(pid))
    stat = (base / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": int(pid), "start_ticks": stat[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def _listens_on(pid: int, port: int) -> bool:
    inodes = set()
    for path in (Path("/proc") / str(pid) / "fd").iterdir():
        try:
            target = os.readlink(path)
            if target.startswith("socket:["):
                inodes.add(target[8:-1])
        except OSError:
            continue
    for name in ("tcp", "tcp6"):
        for line in (Path("/proc/net") / name).read_text().splitlines()[1:]:
            row = line.split()
            if (int(row[1].rsplit(":", 1)[1], 16) == port and row[3] == "0A"
                    and row[9] in inodes):
                return True
    return False


def detector_identity(endpoint: str, receipt_path: str | Path | None) -> dict | None:
    if not endpoint:
        return {"enabled": False}
    if not receipt_path:
        return None
    try:
        parsed = urlparse(endpoint)
        if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            return None
        receipt = json.loads(Path(receipt_path).read_text())
        pid = int(receipt["process"]["pid"])
        if (receipt["process"] != process_identity(pid)
                or receipt["port"] != parsed.port
                or receipt["status"] != "starting_service"
                or not _listens_on(pid, parsed.port)):
            return None
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        if not any(b"serve_groundingdino.py" in word for word in command):
            return None
        model = receipt["generation_identity"]
        # Startup hashes must still describe the files used by the live service.
        for row in receipt["identity_files"]:
            if file_identity(row["path"]) != row["identity"]:
                return None
        service_root = Path(receipt["service_root"]).resolve()
        if (not service_root.is_dir()
                or Path(os.readlink(f"/proc/{pid}/cwd")).resolve() != service_root):
            return None
        return {"enabled": True, "model": model}
    except (KeyError, OSError, TypeError, ValueError):
        return None


def detector_startup_identity(*, service_root: Path, checkpoint: Path,
                              model_config: Path, bert_snapshot: Path,
                              groundingdino_root: Path, transformers_root: Path,
                              launcher: Path) -> tuple[dict, list[dict]]:
    """Capture actual local detector assets before the first model load."""
    roots = {
        "groundingdino": groundingdino_root, "transformers": transformers_root,
        "service": service_root / "skill3d/external_experts/GroundingDINO",
        "bert": bert_snapshot,
    }
    files, provenance = {}, []
    selected = {"checkpoint": checkpoint, "model_config": model_config, "launcher": launcher}
    for prefix, root in roots.items():
        for path in sorted(root.rglob("*")):
            if (path.is_file() and ".cache" not in path.relative_to(root).parts
                    and (prefix == "bert" or path.suffix in (".py", ".so", ".yaml"))):
                selected[f"{prefix}/{path.relative_to(root).as_posix()}"] = path
    for name, path in sorted(selected.items()):
        identity = file_identity(path)
        files[name] = identity
        provenance.append({"path": str(path.resolve()), "identity": identity})
    return {"model": "GroundingDINO_SwinB", "files": files,
            "bert_model": "bert-base-uncased", "mode": "eval",
            "annotate_fallback": os.environ.get("SKILL3D_GDINO_ANNOTATE_FALLBACK", "0")}, provenance


def generation_identity(*, scene_name, frame_set_hash, frames, depth_maps,
                        c2w_list=None, intrinsics=None, point_conf=None, depth_conf=None,
                        grid_transform=None, frame_provenance=None, model=None,
                        detector=None, sam2=None, parameters=None, sources=None,
                        question="", question_type="", options=None,
                        reconstruction_identity="", point_map=None) -> dict:
    identity = {
        "version": M5_IDENTITY_VERSION, "scene": str(scene_name),
        "frame_set_hash": str(frame_set_hash or ""),
        "question": {"question": str(question), "question_type": str(question_type),
                     "options": options or []},
        "reconstruction_identity": str(reconstruction_identity),
        "frames": [array_identity(frame) for frame in frames],
        "frame_provenance": [
            {k: v for k, v in row.items() if k != "qa_id"}
            for row in (frame_provenance or [])],
        "geometry": {name: array_identity(value) for name, value in (
            ("depth", depth_maps), ("c2w", c2w_list), ("intrinsics", intrinsics),
            ("point_conf", point_conf), ("depth_conf", depth_conf),
            ("point_map", point_map))},
        "grid_contract": "rgb-mask-vggt-crop-or-nearest-v1",
        "grid_transform": grid_transform,
        "sam2": sam2, "vlm": model, "detector": detector,
        "parameters": parameters or {},
        "environment": {
            name: _package_version(name) for name in (
                "numpy", "scipy", "opencv-python", "Pillow", "torch", "transformers")},
        "sources": source_identity(Path(__file__).resolve().parents[1])
                   if sources is None else sources,
    }
    missing = [name for name, value in (("sam2", sam2), ("vlm", model),
                                        ("detector", detector)) if value is None]
    if isinstance(model, dict) and not model.get("cache_eligible", True):
        missing.append("mutable_vlm_model")
    if any(value is None for value in identity["sources"].values()):
        missing.append("source_files")
    if depth_maps is None or c2w_list is None:
        missing.append("geometry")
    identity = json.loads(canonical_bytes(identity))
    return {"identity": identity, "sha256": content_sha256(identity),
            "cache_eligible": not missing, "missing": missing}


def _package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "absent"


def inventory_generation_identity(*, scene_name, frame_set_hash, frames, depth_maps,
                                  c2w_list, intrinsics, point_conf, depth_conf,
                                  grid_transform, frame_provenance, predictor,
                                  vlm_client, seed, parameters, model_weights_sha256="",
                                  question="", question_type="", options=None,
                                  reconstruction_identity="", point_map=None):
    from skill3d.segmentation import sam2_tracker as st, open_vocab_detector as ovd

    checkpoint, config = st.resolve_sam2_config()
    checkpoint = st.ensure_checkpoint(checkpoint)
    model = {"enabled": False} if vlm_client is None else None
    if vlm_client is not None:
        actual_client = getattr(vlm_client, "client", vlm_client)
        weights = (model_weights_sha256
                   or getattr(actual_client, "model_weights_sha256", "")
                   or os.environ.get("SKILL3D_MODEL_WEIGHTS_SHA256", ""))
        audit_identity = getattr(actual_client, "audit_identity", None)
        if callable(audit_identity):
            model = audit_identity(seed=seed)
        if model is None and len(weights) == 64 and all(
                c in "0123456789abcdef" for c in weights):
            model = {"enabled": True, "model": str(getattr(actual_client, "model", "")),
                     "weights_sha256": weights, "seed": seed,
                     "temperature": 0.0,
                     "max_tokens": min(2048, int(getattr(
                         actual_client, "max_output_tokens", 2048))),
                     "cache_eligible": True}
            if not model["model"]:
                model = None
    return generation_identity(
        scene_name=scene_name, frame_set_hash=frame_set_hash, frames=frames,
        depth_maps=depth_maps, c2w_list=c2w_list, intrinsics=intrinsics,
        point_conf=point_conf, depth_conf=depth_conf, grid_transform=grid_transform,
        frame_provenance=frame_provenance, model=model,
        detector=detector_identity(ovd.detector_endpoint(),
                                   os.environ.get("SKILL3D_DETECTOR_RECEIPT")),
        sam2=sam2_identity(
            checkpoint, config, predictor,
            inference_mode=str((parameters or {}).get(
                "segmentation_mode", "video_bidirectional"))),
        parameters=parameters, question=question, question_type=question_type,
        options=options, reconstruction_identity=reconstruction_identity,
        point_map=point_map)
