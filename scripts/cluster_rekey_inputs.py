#!/usr/bin/env python3
"""Prepare an isolated run from frozen input tars; never reuse results or enqueue.

The public freeze manifest is unchanged. Only cluster_identity_hash changes in
each scene.json; payloads and saved logical input identities remain byte-frozen.
This is an archive/metadata integrity check, not image decoding or model replay.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tarfile
import tempfile

from skill3d.cluster import FrozenIdentity, JobSpec, hash_file
from skill3d.cluster.identity import canonical_json, check_digest, safe_text
from skill3d.cluster.scene_compute import _no_symlinks, _reference, _reject_labels


PROGRESS_SCHEMA = "harness3d-cluster-package-progress/1"
BLOCK_BYTES = 4 * 1024 * 1024
MAX_SCENE_BYTES = 32 * 1024 * 1024


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _object(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _read(path: Path) -> dict:
    _no_symlinks(path)
    return _object(path.read_bytes())


def _file_record(value: dict) -> dict:
    if (not isinstance(value, dict) or set(value) != {"sha256", "size"}
            or type(value["size"]) is not int or value["size"] < 0):
        raise ValueError("file identity requires SHA256 and nonnegative size")
    check_digest(value["sha256"], "file sha256")
    return value


def _archive_record(row: dict) -> tuple[str, int]:
    value = _file_record({"sha256": row["archive_sha256"], "size": row["archive_bytes"]})
    return value["sha256"], value["size"]


def _segment(value: str) -> str:
    safe_text(value, field="run/scene identifier", limit=256)
    if _reference(value) != value or "/" in value:
        raise ValueError("run/scene identifier must be one safe path component")
    return value


def _archive_path(root: Path, job_key: str) -> Path:
    return root / "inputs/jobs" / (hashlib.sha256(job_key.encode("utf-8")).hexdigest() + ".tar")


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _install(temporary: Path, target: Path, *, replace: bool = False) -> None:
    _no_symlinks(target)
    if replace:
        os.replace(temporary, target)
    else:
        # A hard-link install is atomic and, unlike rename, cannot clobber.
        os.link(temporary, target)
        temporary.unlink()
    _fsync_directory(target.parent)


def _write_json(path: Path, value: dict, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _no_symlinks(path)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        _install(temporary, path, replace=replace)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_scene(scene: dict, scene_id: str, scope: dict,
                    identity_hash: str, geometry: str) -> None:
    _reject_labels(scene)
    questions = scene.get("questions")
    if (not isinstance(questions, list) or len(questions) != len(scope["qa_ids"])
            or any(not isinstance(q, dict) for q in questions)):
        raise ValueError("source scene lacks the full question group")
    public = copy.deepcopy(scene)
    for question in public["questions"]:
        logical = question.get("logical_input_identity", {})
        if isinstance(logical, dict) and isinstance(logical.get("episode"), dict):
            logical["episode"].pop("ground_truth", None)
    canonical_json(public)
    if (scene.get("schema") != "harness3d-frozen-scene/1"
            or scene.get("status") != "frozen"
            or scene.get("real_labels_available") is not False
            or scene.get("cluster_identity_hash") != identity_hash
            or scene.get("scene_id") != scene_id or scene.get("split") != scope["split"]
            or scene.get("qa_ids") != scope["qa_ids"]):
        raise ValueError("source scene identity/split/full QA group differs from public freeze")
    files, identities = scene.get("files"), scene.get("file_identities")
    if (not isinstance(files, dict) or not isinstance(identities, dict)
            or len(set(files.values())) != len(files)
            or set(identities) != set(files.values())):
        raise ValueError("scene file identities must cover an unambiguous files mapping")
    for original, relative in files.items():
        _reference(original, absolute=True)
        _reference(relative)
        if not relative.startswith("data/"):
            raise ValueError("payloads must be under data/")
        _file_record(identities[relative])
    frames = scene.get("frames")
    if not isinstance(frames, list) or len(frames) != 32:
        raise ValueError("scene requires exactly 32 frozen frames")
    frame_sources = []
    for frame in frames:
        original = frame["source_path"]
        if (original not in files or identities[files[original]] !=
                _file_record({"sha256": frame["png_sha256"], "size": frame["size"]})):
            raise ValueError("frozen frame hash/size differs from payload identity")
        check_digest(frame["rgb_sha256"], "frozen RGB sha256")
        if (frame.get("dtype") != "uint8" or not isinstance(frame.get("shape"), list)
                or len(frame["shape"]) != 3 or frame["shape"][2] != 3
                or any(type(n) is not int or n <= 0 for n in frame["shape"])):
            raise ValueError("invalid frozen RGB metadata")
        frame_sources.append(original)
    if len(set(frame_sources)) != 32:
        raise ValueError("duplicate frozen frame source")
    required_sources = set(frame_sources)
    artifact = scene.get("artifact_source_path")
    if (geometry != ("verified_reuse" if artifact else "fresh_required")
            or bool(artifact) != (scope["legacy_artifact_sha256"] is not None)):
        raise ValueError("geometry/package/public freeze agreement changed")
    if artifact:
        if artifact not in files or identities[files[artifact]]["sha256"] != scope["legacy_artifact_sha256"]:
            raise ValueError("legacy artifact differs from public freeze")
        required_sources.add(artifact)
    shared_source, shared_geometry = None, None
    for qa_id, question in zip(scope["qa_ids"], questions):
        episode = question.get("episode")
        if (not isinstance(episode, dict) or episode.get("qa_id") != qa_id
                or episode.get("scene_name") != scene_id or episode.get("split") != scope["split"]
                or not episode.get("dataset") or not episode.get("question")
                or question.get("source") != "vsi_bench"):
            raise ValueError("frozen question order/scene/split differs from full QA group")
        _reference(question.get("video_path"), absolute=True)
        if not isinstance(episode.get("frames"), list) or len(episode["frames"]) != 32:
            raise ValueError("question requires exactly 32 frozen frames")
        for slot, (record, frozen) in enumerate(zip(episode["frames"], frames)):
            if (not isinstance(record, dict) or record.get("frame_idx") != slot
                    or record.get("timestamp") != float(slot)
                    or (record.get("height"), record.get("width")) != tuple(frozen["shape"][:2])):
                raise ValueError("question frame order/shape differs from frozen frames")
        frame_set = episode.get("frame_set")
        if (not isinstance(frame_set, dict) or frame_set.get("n_frames") != 32
                or frame_set.get("episode_id") != qa_id
                or frame_set.get("scene_name") != scene_id
                or frame_set.get("dataset_id") != episode["dataset"]
                or not isinstance(frame_set.get("frame_ids"), list)
                or len(frame_set["frame_ids"]) != 32
                or any(type(i) is not int or i < 0 for i in frame_set["frame_ids"])
                or len(set(frame_set["frame_ids"])) != 32
                or frame_set["frame_ids"] != sorted(frame_set["frame_ids"])
                or frame_set.get("readable_frame_ids") != frame_set["frame_ids"]
                or frame_set.get("decode_status") != "ok"):
            raise ValueError("question frozen frame metadata changed")
        frame_ids, fps = frame_set["frame_ids"], frame_set.get("fps")
        if (type(frame_set.get("n_total_frames")) is not int or frame_set["n_total_frames"] < 32
                or frame_ids[-1] >= frame_set["n_total_frames"]
                or type(fps) not in (int, float) or not math.isfinite(fps) or fps < 0
                or frame_set.get("source_frame_indices") != frame_ids
                or frame_set.get("timestamps") != [float(i) / fps if fps > 0 else float(i) for i in frame_ids]
                or frame_set.get("frame_set_hash") != hashlib.sha256(
                    json.dumps(frame_ids, sort_keys=True).encode("utf-8")).hexdigest()
                or not frame_set.get("video_id") or frame_set.get("preprocessing_version") != "preprocess-v1"
                or frame_set.get("frame_refs") not in ([], frame_sources)
                or not isinstance(frame_set.get("pixel_sha256"), list)
                or len(frame_set["pixel_sha256"]) != 32):
            raise ValueError("question frozen frame source identity changed")
        for digest in frame_set["pixel_sha256"]:
            check_digest(digest, "frozen pixel sha256")
        metadata = (episode["dataset"], question["video_path"],
                    {k: v for k, v in frame_set.items() if k != "episode_id"})
        if shared_source is not None and metadata != shared_source:
            raise ValueError("question group mixes frozen frame sources")
        shared_source = metadata
        logical = question.get("logical_input_identity")
        if not artifact:
            if "logical_input_identity" in question:
                raise ValueError("logical input identity without frozen geometry")
            continue
        if (not isinstance(logical, dict) or set(logical) != {
                "episode", "source", "video_path", "pixels", "artifact_path", "artifact_json_sha256",
                "artifact_id", "artifact_version", "files", "sha256"}
                or logical.get("episode") != {**episode, "ground_truth": ""}
                or logical.get("source") != question["source"]
                or logical.get("video_path") != question["video_path"]
                or logical.get("artifact_path") != artifact
                or logical.get("artifact_json_sha256") != scope["legacy_artifact_sha256"]
                or not isinstance(logical.get("artifact_id"), str) or not logical["artifact_id"]
                or not isinstance(logical.get("artifact_version"), str) or not logical["artifact_version"]
                or logical.get("pixels") != [
                    {"shape": f["shape"], "dtype": f["dtype"], "sha256": f["rgb_sha256"]} for f in frames]):
            raise ValueError("frozen logical input identity differs from question/geometry")
        check_digest(logical.get("sha256"), "logical input sha256")
        body = {k: v for k, v in logical.items() if k != "sha256"}
        if hashlib.sha256(_json_bytes(body)).hexdigest() != logical["sha256"]:
            raise ValueError("frozen logical input identity hash mismatch")
        dependencies = logical.get("files")
        if not isinstance(dependencies, dict):
            raise ValueError("missing logical geometry dependencies")
        for record in dependencies.values():
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise ValueError("invalid logical geometry dependency identity")
            original = record["path"]
            if original not in files or identities[files[original]]["sha256"] != record["sha256"]:
                raise ValueError("logical geometry dependency differs from payload identity")
            required_sources.add(original)
        geometry_identity = {k: v for k, v in logical.items() if k not in {"episode", "sha256"}}
        if shared_geometry is not None and geometry_identity != shared_geometry:
            raise ValueError("question group mixes frozen geometry identities")
        shared_geometry = geometry_identity
    if set(files) != required_sources:
        raise ValueError("files mapping contains non-input payloads or missing geometry")


def _inspect_archive(path: Path, *, expected_archive: tuple[str, int] | None = None,
                     expected_scene: dict | None = None) -> tuple[dict, str, tuple[str, int]]:
    _no_symlinks(path)
    archive_identity = hash_file(path)
    if expected_archive is not None and archive_identity != expected_archive:
        raise ValueError("archive SHA256/size mismatch")
    with tarfile.open(path, "r:") as stream:
        members = {}
        for member in stream:
            _reference(member.name)
            if (member.name in members or not member.isfile() or member.issparse()
                    or member.linkname):
                raise ValueError("duplicate or unsafe tar member")
            members[member.name] = member
        for name in members:
            if any(str(parent) in members for parent in Path(name).parents if str(parent) != "."):
                raise ValueError("tar payload conflicts with a parent file")
        if archive_identity[1] % 512 or archive_identity[1] - stream.offset < 1024:
            raise ValueError("incomplete tar end marker")
        stream.fileobj.seek(stream.offset)
        for block in iter(lambda: stream.fileobj.read(BLOCK_BYTES), b""):
            if any(block):
                raise ValueError("nonzero data after tar end marker")
        if "scene.json" not in members or not 0 < members["scene.json"].size <= MAX_SCENE_BYTES:
            raise ValueError("missing or oversized scene.json")
        with stream.extractfile(members["scene.json"]) as source:
            raw = source.read(MAX_SCENE_BYTES + 1)
        scene = _object(raw)
        if expected_scene is not None and _json_bytes(scene) != _json_bytes(expected_scene):
            raise ValueError("existing archive scene differs from expected rekey")
        identities = scene.get("file_identities")
        if not isinstance(identities, dict) or set(members) != {"scene.json", *identities}:
            raise ValueError("archive members do not exactly cover frozen payloads")
        for name, expected in identities.items():
            _file_record(expected)
            if members[name].size != expected["size"]:
                raise ValueError("tar payload size mismatch")
            digest = hashlib.sha256()
            size = 0
            with stream.extractfile(members[name]) as source:
                for block in iter(lambda: source.read(BLOCK_BYTES), b""):
                    digest.update(block)
                    size += len(block)
            if (digest.hexdigest(), size) != (expected["sha256"], expected["size"]):
                raise ValueError("tar payload SHA256/size mismatch")
    if hash_file(path) != archive_identity:
        raise ValueError("archive changed while validating")
    return scene, hashlib.sha256(raw).hexdigest(), archive_identity


def _write_archive(source: Path, target: Path, scene: dict,
                   source_identity: tuple[str, int]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _no_symlinks(target)
    fd, name = tempfile.mkstemp(prefix="." + target.name + ".", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            with tarfile.open(source, "r:") as original, tarfile.open(fileobj=output, mode="w:") as repack:
                for member in original:
                    header = copy.copy(member)
                    if member.name == "scene.json":
                        raw = _json_bytes(scene) + b"\n"
                        header.size = len(raw)
                        header.pax_headers = {k: v for k, v in member.pax_headers.items() if k != "size"}
                        repack.addfile(header, io.BytesIO(raw))
                    else:
                        with original.extractfile(member) as payload:
                            repack.addfile(header, payload)
            output.flush()
            os.fsync(output.fileno())
        if hash_file(source) != source_identity:
            raise ValueError("source archive changed while repackaging")
        _inspect_archive(temporary, expected_scene=scene)
        _install(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _source_plan(source: Path, method: dict, run_id: str) -> tuple[dict, dict, dict, dict]:
    jobs_path, receipt_path = source / "coordinator_jobs.json", source / "receipts/package_progress.json"
    jobs_digest, receipt_digest = hash_file(jobs_path), hash_file(receipt_path)
    envelope, progress = _read(jobs_path), _read(receipt_path)
    identity = FrozenIdentity.from_dict(envelope["identity"])
    manifest = json.loads(identity.manifest_json)
    if manifest.get("schema") not in {"harness3d-cluster-scope/1", "cluster-scope/1"}:
        raise ValueError("source identity is not a frozen public scene scope")
    scopes = manifest.get("scenes")
    if not isinstance(scopes, dict) or not scopes:
        raise ValueError("source public scene scope is empty")
    all_qa_ids = []
    for scene_id, scope in scopes.items():
        _segment(scene_id)
        qa_ids = scope["qa_ids"]
        if (not isinstance(qa_ids, list) or not qa_ids
                or any(not isinstance(q, str) or not q for q in qa_ids)
                or qa_ids != sorted(set(qa_ids))
                or scope["split"] not in {"induction", "inner_validation", "outer_holdout"}):
            raise ValueError("invalid full public QA group/split")
        check_digest(scope["frozen_scene_sha256"], "public frozen scene sha256")
        if scope["legacy_artifact_sha256"] is not None:
            check_digest(scope["legacy_artifact_sha256"], "public legacy artifact sha256")
        all_qa_ids.extend(qa_ids)
    if len(set(all_qa_ids)) != len(all_qa_ids):
        raise ValueError("public QA groups overlap")
    if (progress.get("schema") != PROGRESS_SCHEMA or progress.get("status") != "packaged"
            or progress.get("identity_hash") != identity.identity_hash
            or progress.get("labels_accessed") is not False
            or set(progress.get("scenes", {})) != set(scopes)
            or progress.get("total_scenes") != len(scopes)
            or progress.get("total_questions") != len(all_qa_ids)):
        raise ValueError("source jobs/package progress agreement changed")
    new_identity = FrozenIdentity.freeze(manifest, method, expected_manifest_hash=identity.manifest_hash)
    if new_identity.identity_hash == identity.identity_hash:
        raise ValueError("new method identity must differ from source identity")
    specs, source_run = {}, None
    for row in envelope["jobs"]:
        spec = JobSpec.from_dict(row)
        parts = spec.job_key.split("/")
        if (len(parts) != 3 or parts[1] not in scopes or parts[2] != "gpu_aggregate"
                or spec.stage != "gpu_aggregate" or spec.dependencies
                or spec.identity != identity
                or spec.completed_key != f"{identity.identity_hash}/{parts[1]}/gpu_aggregate"):
            raise ValueError("source jobs must be root gpu_aggregate jobs with one frozen identity")
        _segment(parts[0])
        if source_run is not None and source_run != parts[0]:
            raise ValueError("source jobs mix run identifiers")
        source_run = parts[0]
        scene_id = parts[1]
        receipt = progress["scenes"][scene_id]
        _archive_record(receipt)
        if (scene_id in specs or receipt["job_key"] != spec.job_key
                or type(receipt["questions"]) is not int
                or receipt["questions"] != len(scopes[scene_id]["qa_ids"])
                or receipt["geometry"] not in {"verified_reuse", "fresh_required"}):
            raise ValueError("source jobs/package scene agreement changed")
        specs[scene_id] = spec
    if set(specs) != set(scopes) or source_run == run_id:
        raise ValueError("source job scope incomplete or new run identifier equals source")
    new_specs = {
        scene_id: JobSpec(f"{run_id}/{scene_id}/gpu_aggregate",
                          f"{new_identity.identity_hash}/{scene_id}/gpu_aggregate",
                          new_identity, "gpu_aggregate", dependencies=spec.dependencies,
                          max_attempts=spec.max_attempts, priority=spec.priority,
                          dependency_policy=spec.dependency_policy)
        for scene_id, spec in sorted(specs.items())
    }
    output = {"identity": new_identity.to_dict(), "jobs": [s.to_dict() for s in new_specs.values()]}
    base = {
        "schema": PROGRESS_SCHEMA, "preparation": "frozen-input-rekey/1",
        "identity_hash": new_identity.identity_hash, "run_id": run_id,
        "total_scenes": len(scopes), "total_questions": len(all_qa_ids), "labels_accessed": False,
        "source": {
            "root": str(source), "identity_hash": identity.identity_hash,
            "coordinator_jobs_sha256": jobs_digest[0], "package_progress_sha256": receipt_digest[0],
            "archives": {s: {"sha256": r["archive_sha256"], "size": r["archive_bytes"]}
                         for s, r in sorted(progress["scenes"].items())},
        },
    }
    if hash_file(jobs_path) != jobs_digest or hash_file(receipt_path) != receipt_digest:
        raise ValueError("source metadata changed while planning")
    return output, base, specs, progress


def _destination(root: Path, output: dict, base: dict) -> dict:
    archive_names = {
        str(_archive_path(root, row["job_key"]).relative_to(root)) for row in output["jobs"]
    }
    files = archive_names | {"jobs.json", "coordinator_jobs.json", "receipts/package_progress.json"}
    directories = {"inputs", "inputs/jobs", "receipts"}
    if root.exists():
        if not root.is_dir():
            raise ValueError("destination is not a directory")
        for path in root.rglob("*"):
            _no_symlinks(path)
            relative = path.relative_to(root).as_posix()
            if not (path.is_file() and relative in files or path.is_dir() and relative in directories):
                raise ValueError(f"destination contains unrelated state/results: {relative}")
    path = root / "receipts/package_progress.json"
    if not path.exists():
        if root.exists() and any(root.iterdir()):
            raise ValueError("nonempty destination has no rekey provenance")
        return {**base, "status": "running", "scenes": {}}
    progress = _read(path)
    if (_json_bytes({k: v for k, v in progress.items() if k not in {"status", "scenes"}}) != _json_bytes(base)
            or progress.get("status") not in {"running", "packaged"}
            or not isinstance(progress.get("scenes"), dict)
            or not set(progress["scenes"]) <= set(base["source"]["archives"])):
        raise ValueError("destination rekey identity/source provenance conflict")
    for name in ("jobs.json", "coordinator_jobs.json"):
        target = root / name
        if target.exists() and _json_bytes(_read(target)) != _json_bytes(output):
            raise ValueError("destination job manifest identity conflict")
        if target.exists() and len(progress["scenes"]) != base["total_scenes"]:
            raise ValueError("destination exposes job manifest for partial preparation")
        if progress["status"] == "packaged" and not target.exists():
            raise ValueError("packaged destination is missing a job manifest")
    if progress["status"] == "packaged" and len(progress["scenes"]) != base["total_scenes"]:
        raise ValueError("packaged destination is incomplete")
    return progress


def prepare(source_root: Path, root: Path, method_config: Path, run_id: str) -> dict:
    """Prepare or strictly verify/resume a run. This function never opens a queue."""
    source, root = Path(source_root).absolute(), Path(root).absolute()
    _no_symlinks(source)
    _no_symlinks(root)
    source, root = source.resolve(), root.resolve()
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("source and destination roots must be distinct and non-overlapping")
    _segment(run_id)
    output, base, specs, source_progress = _source_plan(source, _read(Path(method_config)), run_id)
    progress = _destination(root, output, base)
    progress_path = root / "receipts/package_progress.json"
    if not progress_path.exists():
        _write_json(progress_path, progress)
    new_specs = {s["job_key"].split("/")[1]: JobSpec.from_dict(s) for s in output["jobs"]}
    scopes = output["identity"]["manifest"]["scenes"]
    for scene_id, spec in sorted(specs.items()):
        source_archive = _archive_path(source, spec.job_key)
        source_receipt = source_progress["scenes"][scene_id]
        scene, source_scene_sha, source_archive_identity = _inspect_archive(
            source_archive, expected_archive=_archive_record(source_receipt))
        _validate_scene(scene, scene_id, scopes[scene_id], spec.identity.identity_hash,
                        source_receipt["geometry"])
        expected = {**scene, "cluster_identity_hash": base["identity_hash"]}
        target = _archive_path(root, new_specs[scene_id].job_key)
        if not target.exists():
            if scene_id in progress["scenes"] or progress["status"] == "packaged":
                raise ValueError("recorded output archive is missing")
            _write_archive(source_archive, target, expected, source_archive_identity)
        _, output_scene_sha, output_archive_identity = _inspect_archive(target, expected_scene=expected)
        receipt = {
            "job_key": new_specs[scene_id].job_key,
            "archive_sha256": output_archive_identity[0], "archive_bytes": output_archive_identity[1],
            "questions": source_receipt["questions"], "geometry": source_receipt["geometry"],
            "source_job_key": spec.job_key, "source_archive_sha256": source_archive_identity[0],
            "source_archive_bytes": source_archive_identity[1],
            "source_scene_sha256": source_scene_sha, "output_scene_sha256": output_scene_sha,
            "frozen_scene_sha256": scopes[scene_id]["frozen_scene_sha256"],
            "preserved_file_identities": scene["file_identities"],
        }
        if scene_id in progress["scenes"]:
            if _json_bytes(progress["scenes"][scene_id]) != _json_bytes(receipt):
                raise ValueError("destination archive/provenance receipt conflict")
        else:
            progress["scenes"][scene_id] = receipt
            _write_json(progress_path, progress, replace=True)
    for name, digest_field in (("coordinator_jobs.json", "coordinator_jobs_sha256"),
                               ("receipts/package_progress.json", "package_progress_sha256")):
        if hash_file(source / name)[0] != base["source"][digest_field]:
            raise ValueError("source metadata changed during preparation")
    for name in ("coordinator_jobs.json", "jobs.json"):
        path = root / name
        if path.exists():
            if _json_bytes(_read(path)) != _json_bytes(output):
                raise ValueError("destination job manifest identity conflict")
        else:
            _write_json(path, output)
    if progress["status"] != "packaged":
        progress["status"] = "packaged"
        _write_json(progress_path, progress, replace=True)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    progress = prepare(args.source_root, args.root, args.method_config, args.run_id)
    print(json.dumps({k: v for k, v in progress.items() if k not in {"source", "scenes"}},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
