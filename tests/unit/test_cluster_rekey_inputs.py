"""Isolated, byte-preserving preparation from trusted frozen input packages."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zlib

import pytest

from skill3d.cluster import FrozenIdentity, JobSpec, hash_file


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/cluster_rekey_inputs.py"
SPEC = importlib.util.spec_from_file_location("cluster_rekey_inputs_test", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

# Frozen public questions from r02 scene 0d2ee665be (QA 1584 and 2700).
MULTILINE_QUESTIONS = (
    "If I am standing by the door and facing the cutting board, is the computer mouse "
    "to my left, right, or back?\n"
    "An object is to my back if I would have to turn at least 135 degrees in order to face it.",
    "What is the size of this room (in square meters)? \n"
    "If multiple rooms are shown, estimate the size of the combined space.",
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(value) + b"\n")


def _read(path):
    return json.loads(path.read_bytes())


def _snapshot(root):
    return {p.relative_to(root).as_posix(): hash_file(p) for p in root.rglob("*") if p.is_file()}


def _png(slot):
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))

    pixels = bytes([slot, 37, 91]) * 4
    rows = b"\0" + pixels[:6] + b"\0" + pixels[6:]
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")), pixels


def _tar(path, scene, payloads, extra=(), omit=(), scene_raw=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:") as stream:
        records = [("scene.json", _bytes(scene) + b"\n" if scene_raw is None else scene_raw)]
        records.extend(payloads.items())
        for name, raw in records:
            if name in omit:
                continue
            header = tarfile.TarInfo(name)
            header.size, header.mtime, header.mode = len(raw), 123456789, 0o640
            stream.addfile(header, io.BytesIO(raw))
        for header, raw in extra:
            stream.addfile(header, io.BytesIO(raw) if raw is not None else None)


def _source(tmp_path, counts=(2, 9), geometry_bytes=b"frozen geometry dependency",
            question_texts=None):
    root = tmp_path / "r02"
    scenes, payloads, scopes = {}, {}, {}
    for index, count in enumerate(counts):
        scene_id = f"scene-{index}"
        split = ("induction", "outer_holdout")[index % 2]
        qa_ids = [f"{scene_id}-qa-{q:03d}" for q in range(count)]
        files, identities, frames, pixels = {}, {}, [], []
        data = {}
        for slot in range(32):
            raw, rgb = _png(slot)
            original, relative = f"/frozen/{scene_id}/{slot:04d}.png", f"data/frames/{slot:04d}.png"
            files[original], data[relative] = relative, raw
            identities[relative] = {"sha256": _sha(raw), "size": len(raw)}
            frames.append({"source_path": original, "png_sha256": _sha(raw), "size": len(raw),
                           "rgb_sha256": _sha(rgb), "shape": [2, 2, 3], "dtype": "uint8"})
            pixels.append({"shape": [2, 2, 3], "dtype": "uint8", "sha256": _sha(rgb)})
        artifact = f"/legacy/{scene_id}/artifact.json" if index == 0 else None
        artifact_sha = None
        dependency = f"/legacy/{scene_id}/points.bin"
        if artifact:
            artifact_raw = _bytes({"scene_name": scene_id, "artifact_id": "frozen-geometry",
                                   "artifact_version": "v1", "point_map": "points.bin"})
            artifact_sha = _sha(artifact_raw)
            for original, relative, raw in (
                (artifact, "data/geometry/artifact.json", artifact_raw),
                (dependency, "data/geometry/points.bin", geometry_bytes),
            ):
                files[original], data[relative] = relative, raw
                identities[relative] = {"sha256": _sha(raw), "size": len(raw)}
        questions = []
        for offset, qa_id in enumerate(qa_ids):
            frame_ids = list(range(32))
            frame_set = {
                "frame_ids": frame_ids, "source_frame_indices": frame_ids,
                "timestamps": [float(i) for i in frame_ids], "n_frames": 32,
                "n_total_frames": 32, "fps": 1.0,
                "frame_set_hash": _sha(json.dumps(frame_ids, sort_keys=True).encode()),
                "episode_id": qa_id, "dataset_id": "scannet", "scene_name": scene_id,
                "video_id": scene_id, "readable_frame_ids": frame_ids, "decode_status": "ok",
                "preprocessing_version": "preprocess-v1", "frame_refs": [],
                "pixel_sha256": [_sha(str(((2, 2, 3), "|u1")).encode() + _png(i)[1]) for i in frame_ids],
            }
            episode = {
                "qa_id": qa_id, "scene_name": scene_id, "split": split, "dataset": "scannet",
                "question": (question_texts[offset] if question_texts is not None
                             else f"Public question {qa_id}?"),
                "question_type": "object_counting",
                "options": None,
                "frames": [{"frame_idx": i, "timestamp": float(i), "height": 2, "width": 2}
                           for i in range(32)],
                "frame_set": frame_set,
            }
            question = {"episode": episode, "source": "vsi_bench", "video_path": f"/video/{scene_id}.mp4"}
            if artifact:
                logical = {
                    "episode": {**episode, "ground_truth": ""}, "source": question["source"],
                    "video_path": question["video_path"], "pixels": pixels, "artifact_path": artifact,
                    "artifact_json_sha256": artifact_sha, "artifact_id": "frozen-geometry",
                    "artifact_version": "v1",
                    "files": {"point_map": {"path": dependency, "sha256": _sha(geometry_bytes)}},
                }
                logical["sha256"] = _sha(_bytes(logical))
                question["logical_input_identity"] = logical
            questions.append(question)
        scene = {
            "schema": "harness3d-frozen-scene/1", "status": "frozen", "real_labels_available": False,
            "scene_id": scene_id, "split": split, "qa_ids": qa_ids, "questions": questions,
            "frames": frames, "artifact_source_path": artifact,
        }
        scopes[scene_id] = {"qa_ids": qa_ids, "split": split, "frozen_scene_sha256": _sha(_bytes(scene)),
                            "legacy_artifact_sha256": artifact_sha}
        scene["files"], scene["file_identities"] = files, identities
        scenes[scene_id], payloads[scene_id] = scene, data
    identity = FrozenIdentity.freeze({"schema": "harness3d-cluster-scope/1", "scenes": scopes},
                                     {"detector": "r02", "threshold": 0.1})
    jobs, receipts = [], {}
    for index, (scene_id, scene) in enumerate(scenes.items()):
        scene["cluster_identity_hash"] = identity.identity_hash
        spec = JobSpec(f"old-r02/{scene_id}/gpu_aggregate",
                       f"{identity.identity_hash}/{scene_id}/gpu_aggregate", identity, "gpu_aggregate",
                       max_attempts=2 + index, priority=100 - index,
                       dependency_policy="complete_or_skipped")
        jobs.append(spec.to_dict())
        archive = module._archive_path(root, spec.job_key)
        _tar(archive, scene, payloads[scene_id])
        digest, size = hash_file(archive)
        receipts[scene_id] = {
            "job_key": spec.job_key, "archive_sha256": digest, "archive_bytes": size,
            "questions": len(scene["qa_ids"]),
            "geometry": "verified_reuse" if scene["artifact_source_path"] else "fresh_required",
        }
    _write(root / "coordinator_jobs.json", {"identity": identity.to_dict(), "jobs": jobs})
    _write(root / "receipts/package_progress.json", {
        "schema": module.PROGRESS_SCHEMA, "status": "packaged", "identity_hash": identity.identity_hash,
        "scenes": receipts, "total_scenes": len(counts), "total_questions": sum(counts),
        "labels_accessed": False, "elapsed_s": 3.14,
    })
    for relative in ("results/old.bin", "bindings/old.json", "state/jobs.sqlite3"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old live r02 state must never be copied")
    method = tmp_path / "method.json"
    _write(method, {"detector": "r03", "threshold": 0.25})
    return SimpleNamespace(root=root, destination=tmp_path / "r03", method=method,
                           scenes=scenes, payloads=payloads, identity=identity, jobs=jobs)


def _prepare(case, **kwargs):
    return module.prepare(case.root, kwargs.get("root", case.destination),
                          kwargs.get("method", case.method), kwargs.get("run_id", "new-r03"))


def _replace_source_tar(case, scene_id="scene-0", **kwargs):
    receipt_path = case.root / "receipts/package_progress.json"
    progress = _read(receipt_path)
    archive = module._archive_path(case.root, progress["scenes"][scene_id]["job_key"])
    _tar(archive, case.scenes[scene_id], case.payloads[scene_id], **kwargs)
    digest, size = hash_file(archive)
    progress["scenes"][scene_id].update(archive_sha256=digest, archive_bytes=size)
    _write(receipt_path, progress)
    return archive


def test_rekeys_exact_public_manifest_and_only_scene_identity(tmp_path):
    case = _source(tmp_path)
    before = _snapshot(case.root)
    progress = _prepare(case)
    assert _snapshot(case.root) == before
    output = _read(case.destination / "coordinator_jobs.json")
    assert output == _read(case.destination / "jobs.json")
    new_identity = FrozenIdentity.from_dict(output["identity"])
    assert new_identity.manifest_json == case.identity.manifest_json
    assert new_identity.manifest_hash == case.identity.manifest_hash
    assert new_identity.method_config_hash != case.identity.method_config_hash
    assert new_identity.identity_hash != case.identity.identity_hash
    assert progress["status"] == "packaged"
    assert (progress["total_scenes"], progress["total_questions"]) == (2, 11)
    assert not any((case.destination / p).exists() for p in ("results", "bindings", "state"))
    for old, new in zip(case.jobs, output["jobs"]):
        scene_id = new["job_key"].split("/")[1]
        assert new["job_key"] == f"new-r03/{scene_id}/gpu_aggregate"
        assert new["completed_key"] == f"{new_identity.identity_hash}/{scene_id}/gpu_aggregate"
        assert new["identity"] == output["identity"]
        for field in ("priority", "max_attempts", "dependencies", "stage", "dependency_policy"):
            assert new[field] == old[field]
        archive = module._archive_path(case.destination, new["job_key"])
        with tarfile.open(archive, "r:") as stream:
            scene = _read_tar_scene(stream)
            expected = {**case.scenes[scene_id], "cluster_identity_hash": new_identity.identity_hash}
            assert scene == expected
            for name, raw in case.payloads[scene_id].items():
                assert stream.extractfile(name).read() == raw
        receipt = progress["scenes"][scene_id]
        assert (receipt["archive_sha256"], receipt["archive_bytes"]) == hash_file(archive)
        assert (receipt["source_archive_sha256"], receipt["source_archive_bytes"]) == hash_file(
            module._archive_path(case.root, old["job_key"]))
        assert receipt["source_scene_sha256"] != receipt["frozen_scene_sha256"]
        assert receipt["source_scene_sha256"] != receipt["output_scene_sha256"]
        assert receipt["preserved_file_identities"] == scene["file_identities"]
        for question in scene["questions"]:
            assert "ground_truth" not in question["episode"]
            if "logical_input_identity" in question:
                assert question["logical_input_identity"]["episode"]["ground_truth"] == ""
    destination_before = _snapshot(case.destination)
    assert _prepare(case) == progress
    assert _snapshot(case.destination) == destination_before
    assert _snapshot(case.root) == before


def _read_tar_scene(stream):
    return json.load(stream.extractfile("scene.json"))


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_rekeys_real_multiline_questions_without_normalizing_frozen_bytes(tmp_path, line_ending):
    texts = tuple(text.replace("\n", line_ending) for text in MULTILINE_QUESTIONS)
    case = _source(tmp_path, counts=(2, 2), question_texts=texts)
    source_before, scenes_before = _snapshot(case.root), copy.deepcopy(case.scenes)
    progress = _prepare(case)
    output = _read(case.destination / "jobs.json")
    new_identity = FrozenIdentity.from_dict(output["identity"])
    assert new_identity.manifest_json == case.identity.manifest_json
    assert new_identity.manifest_hash == case.identity.manifest_hash
    assert new_identity.identity_hash != case.identity.identity_hash
    assert {row["geometry"] for row in progress["scenes"].values()} == {
        "verified_reuse", "fresh_required",
    }
    for row in output["jobs"]:
        scene_id = row["job_key"].split("/")[1]
        archive = module._archive_path(case.destination, row["job_key"])
        with tarfile.open(archive, "r:") as stream:
            scene = _read_tar_scene(stream)
            expected = {**scenes_before[scene_id], "cluster_identity_hash": new_identity.identity_hash}
            assert _bytes(scene) == _bytes(expected)
            for name, raw in case.payloads[scene_id].items():
                assert stream.extractfile(name).read() == raw
        for question, text in zip(scene["questions"], texts):
            assert question["episode"]["question"].encode("utf-8") == text.encode("utf-8")
            if "logical_input_identity" in question:
                logical = question["logical_input_identity"]
                assert logical["episode"]["question"] == text
                assert logical["sha256"] == _sha(_bytes({
                    k: v for k, v in logical.items() if k != "sha256"
                }))
    destination_before = _snapshot(case.destination)
    assert _prepare(case) == progress
    assert _snapshot(case.destination) == destination_before
    assert _snapshot(case.root) == source_before
    assert case.scenes == scenes_before


@pytest.mark.parametrize("snapshot", ["manifest", "method"])
def test_canonical_identity_contract_still_rejects_multiline_text(snapshot):
    with pytest.raises(ValueError, match="control characters"):
        module.canonical_json({"question": MULTILINE_QUESTIONS[0]})
    manifest, method = {"schema": "cluster-scope/1"}, {"detector": "r03"}
    (manifest if snapshot == "manifest" else method)["question"] = MULTILINE_QUESTIONS[0]
    with pytest.raises(ValueError, match="control characters"):
        FrozenIdentity.freeze(manifest, method)


@pytest.mark.parametrize("location", ["public", "logical"])
@pytest.mark.parametrize("control", ["\0", "\t", "\v", "\f", "\x1f"])
def test_question_line_break_exception_does_not_allow_other_controls(tmp_path, location, control):
    case = _source(tmp_path, counts=(1,), question_texts=MULTILINE_QUESTIONS)
    question = case.scenes["scene-0"]["questions"][0]
    logical = question["logical_input_identity"]
    episode = question["episode"] if location == "public" else logical["episode"]
    episode["question"] += control
    logical["sha256"] = _sha(_bytes({k: v for k, v in logical.items() if k != "sha256"}))
    _replace_source_tar(case)
    before = _snapshot(case.root)
    with pytest.raises(ValueError, match="control characters"):
        _prepare(case)
    assert _snapshot(case.root) == before
    assert not (case.destination / "jobs.json").exists()
    assert not list((case.destination / "inputs/jobs").glob("*.tar"))


@pytest.mark.parametrize("path", [
    ("scene_id",),
    ("questions", 0, "source"),
    ("questions", 0, "video_path"),
    ("questions", 0, "episode", "qa_id"),
    ("questions", 0, "episode", "dataset"),
    ("questions", 0, "episode", "question_type"),
    ("questions", 0, "episode", "frame_set", "video_id"),
    ("questions", 0, "logical_input_identity", "artifact_id"),
    ("questions", 0, "logical_input_identity", "episode", "dataset"),
    ("questions", 0, "logical_input_identity", "files", "point_map", "path"),
])
def test_multiline_questions_do_not_relax_identifier_or_metadata_validation(tmp_path, path):
    case = _source(tmp_path, counts=(1,), question_texts=MULTILINE_QUESTIONS)
    owner = case.scenes["scene-0"]
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] += "\n"
    _replace_source_tar(case)
    before = _snapshot(case.root)
    with pytest.raises(ValueError, match="control characters"):
        _prepare(case)
    assert _snapshot(case.root) == before
    assert not list((case.destination / "inputs/jobs").glob("*.tar"))


@pytest.mark.parametrize("field", ["nested_question", "options", "key", "credential", "label"])
def test_multiline_question_exception_keeps_recursive_safety_checks(tmp_path, field):
    case = _source(tmp_path, counts=(1,), question_texts=MULTILINE_QUESTIONS)
    scene = case.scenes["scene-0"]
    episode = scene["questions"][0]["episode"]
    if field == "nested_question":
        scene["metadata"] = {"question": MULTILINE_QUESTIONS[0]}
    elif field == "options":
        episode["options"] = ["A\nB"]
    elif field == "key":
        episode["extra\nkey"] = "value"
    elif field == "credential":
        episode["question"] += "\napi_key=forbidden"
    else:
        episode["reference_answer"] = "forbidden"
    _replace_source_tar(case)
    before = _snapshot(case.root)
    with pytest.raises(ValueError, match="control characters|credential|forbidden"):
        _prepare(case)
    assert _snapshot(case.root) == before
    assert not list((case.destination / "inputs/jobs").glob("*.tar"))


@pytest.mark.parametrize("location", ["public", "logical"])
@pytest.mark.parametrize("value", [None, 1, True, [], {}, "", " \r\n "])
def test_frozen_question_requires_nonempty_text(tmp_path, location, value):
    case = _source(tmp_path, counts=(1,))
    question = case.scenes["scene-0"]["questions"][0]
    logical = question["logical_input_identity"]
    episode = question["episode"] if location == "public" else logical["episode"]
    episode["question"] = value
    logical["sha256"] = _sha(_bytes({k: v for k, v in logical.items() if k != "sha256"}))
    _replace_source_tar(case)
    with pytest.raises(ValueError):
        _prepare(case)
    assert not list((case.destination / "inputs/jobs").glob("*.tar"))


@pytest.mark.parametrize("invalid", [
    "duplicate_question", "escaped_duplicate_question", "literal_newline",
    "NaN", "Infinity", "-Infinity", "1e999",
])
def test_multiline_questions_keep_json_duplicate_and_nonfinite_rejection(tmp_path, invalid):
    case = _source(tmp_path, counts=(1,), question_texts=MULTILINE_QUESTIONS)
    raw = _bytes(case.scenes["scene-0"])
    if invalid == "duplicate_question":
        raw = raw.replace(b'"question":', b'"question":"duplicate","question":', 1)
    elif invalid == "escaped_duplicate_question":
        raw = raw.replace(b'"question":', b'"\\u0071uestion":"duplicate","question":', 1)
    elif invalid == "literal_newline":
        raw = raw.replace(b"\\n", b"\n", 1)
    else:
        raw = b'{"extra_number":' + invalid.encode("ascii") + b"," + raw[1:]
    _replace_source_tar(case, scene_raw=raw)
    before = _snapshot(case.root)
    with pytest.raises(ValueError, match="duplicate JSON|Invalid control|nonfinite"):
        _prepare(case)
    assert _snapshot(case.root) == before
    assert not list((case.destination / "inputs/jobs").glob("*.tar"))


@pytest.mark.parametrize("change", ["logical_question", "logical_hash"])
def test_question_validation_projection_is_not_used_for_logical_equality_or_hash(tmp_path, change):
    case = _source(tmp_path, counts=(1,), question_texts=MULTILINE_QUESTIONS)
    logical = case.scenes["scene-0"]["questions"][0]["logical_input_identity"]
    body = copy.deepcopy({k: v for k, v in logical.items() if k != "sha256"})
    body["episode"]["question"] = body["episode"]["question"].replace("\n", " ")
    if change == "logical_question":
        logical["episode"]["question"] = body["episode"]["question"]
    logical["sha256"] = _sha(_bytes(body))
    _replace_source_tar(case)
    before = _snapshot(case.root)
    with pytest.raises(ValueError, match="logical input identity"):
        _prepare(case)
    assert _snapshot(case.root) == before
    assert not list((case.destination / "inputs/jobs").glob("*.tar"))


@pytest.mark.parametrize("replacement", [" ", "\r\n"])
def test_resume_rejects_normalized_question_bytes_without_overwriting(tmp_path, replacement):
    case = _source(tmp_path, counts=(1,), question_texts=MULTILINE_QUESTIONS)
    progress = _prepare(case)
    progress.update(status="running", scenes={})
    _write(case.destination / "receipts/package_progress.json", progress)
    for name in ("jobs.json", "coordinator_jobs.json"):
        (case.destination / name).unlink()
    archive = next((case.destination / "inputs/jobs").glob("*.tar"))
    with tarfile.open(archive, "r:") as stream:
        scene = _read_tar_scene(stream)
    question = scene["questions"][0]
    for episode in (question["episode"], question["logical_input_identity"]["episode"]):
        episode["question"] = episode["question"].replace("\n", replacement)
    _tar(archive, scene, case.payloads["scene-0"])
    before = _snapshot(case.destination)
    with pytest.raises(ValueError, match="expected rekey"):
        _prepare(case)
    assert _snapshot(case.destination) == before


def test_cli_runs_with_stdlib_only_and_all_arguments_required(tmp_path):
    case = _source(tmp_path, counts=(1,))
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, "-S", str(SCRIPT)]
    flags = ["--source-root", str(case.root), "--root", str(case.destination),
             "--method-config", str(case.method), "--run-id", "new-r03"]
    result = subprocess.run(command + flags, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "packaged"
    for offset in range(0, len(flags), 2):
        result = subprocess.run(command + flags[:offset] + flags[offset + 2:],
                                env=env, capture_output=True, text=True)
        assert result.returncode == 2
        assert "required" in result.stderr


def test_streams_large_payload_without_extraction(tmp_path, monkeypatch):
    case = _source(tmp_path, counts=(1,), geometry_bytes=b"\xa5" * (2 * module.BLOCK_BYTES + 111))
    read = tarfile.ExFileObject.read
    reads = []

    def bounded(stream, size=-1):
        assert 0 <= size <= max(module.BLOCK_BYTES, module.MAX_SCENE_BYTES + 1)
        reads.append(size)
        return read(stream, size)

    def forbidden(*args, **kwargs):
        raise AssertionError("archive extraction is forbidden")

    monkeypatch.setattr(tarfile.ExFileObject, "read", bounded)
    monkeypatch.setattr(tarfile.TarFile, "extract", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extractall", forbidden)
    _prepare(case)
    assert module.BLOCK_BYTES in reads
    assert not (case.destination / "data").exists()


def test_interrupted_preparation_stays_running_without_job_manifests_and_resumes(tmp_path, monkeypatch):
    case = _source(tmp_path)
    before = _snapshot(case.root)
    writer = module._write_archive
    writes = []

    def fail_second(source, target, scene, source_identity):
        writes.append(scene["scene_id"])
        if len(writes) == 2:
            raise OSError("simulated interruption")
        return writer(source, target, scene, source_identity)

    monkeypatch.setattr(module, "_write_archive", fail_second)
    with pytest.raises(OSError, match="interruption"):
        _prepare(case)
    progress = _read(case.destination / "receipts/package_progress.json")
    assert progress["status"] == "running" and set(progress["scenes"]) == {"scene-0"}
    assert not any((case.destination / p).exists() for p in ("jobs.json", "coordinator_jobs.json", "state"))
    first = next((case.destination / "inputs/jobs").glob("*.tar"))
    first_identity = hash_file(first)
    monkeypatch.setattr(module, "_write_archive", writer)
    assert _prepare(case)["status"] == "packaged"
    assert hash_file(first) == first_identity
    assert _snapshot(case.root) == before


def test_resume_validates_archive_installed_before_scene_receipt(tmp_path, monkeypatch):
    case = _source(tmp_path, counts=(1,))
    writer = module._write_json

    def interrupted(path, value, **kwargs):
        if kwargs.get("replace"):
            raise OSError("receipt interruption")
        return writer(path, value, **kwargs)

    monkeypatch.setattr(module, "_write_json", interrupted)
    with pytest.raises(OSError, match="receipt interruption"):
        _prepare(case)
    archive = next((case.destination / "inputs/jobs").glob("*.tar"))
    identity = hash_file(archive)
    assert _read(case.destination / "receipts/package_progress.json")["scenes"] == {}
    monkeypatch.setattr(module, "_write_json", writer)
    assert _prepare(case)["status"] == "packaged"
    assert hash_file(archive) == identity


@pytest.mark.parametrize("where", ["source", "destination"])
def test_resume_rejects_later_archive_corruption_without_overwriting(tmp_path, where):
    case = _source(tmp_path)
    _prepare(case)
    root = case.root if where == "source" else case.destination
    archive = next((root / "inputs/jobs").glob("*.tar"))
    raw = bytearray(archive.read_bytes())
    raw[-1] ^= 1
    archive.write_bytes(raw)
    before = _snapshot(case.destination)
    with pytest.raises(ValueError, match="mismatch|conflict|end marker"):
        _prepare(case)
    assert _snapshot(case.destination) == before


def test_corrupt_second_source_leaves_partial_running_and_no_queue(tmp_path):
    case = _source(tmp_path)
    archive = module._archive_path(case.root, case.jobs[1]["job_key"])
    original = archive.read_bytes()
    archive.write_bytes(original[:-1024])
    with pytest.raises(ValueError, match="archive SHA256/size"):
        _prepare(case)
    progress = _read(case.destination / "receipts/package_progress.json")
    assert progress["status"] == "running" and set(progress["scenes"]) == {"scene-0"}
    assert not (case.destination / "coordinator_jobs.json").exists()
    assert not (case.destination / "jobs.json").exists()
    archive.write_bytes(original)
    assert _prepare(case)["status"] == "packaged"


@pytest.mark.parametrize("change", ["method", "run", "source_receipt", "jobs_manifest", "results", "queue"])
def test_resume_rejects_mixed_identity_or_foreign_state(tmp_path, change):
    case = _source(tmp_path, counts=(1,))
    _prepare(case)
    kwargs = {}
    if change == "method":
        _write(case.method, {"detector": "another-method"})
    elif change == "run":
        kwargs["run_id"] = "other-run"
    elif change == "source_receipt":
        path = case.root / "receipts/package_progress.json"
        progress = _read(path)
        progress["elapsed_s"] += 1
        _write(path, progress)
    elif change == "jobs_manifest":
        path = case.destination / "jobs.json"
        jobs = _read(path)
        jobs["jobs"][0]["priority"] += 1
        _write(path, jobs)
    else:
        path = case.destination / ("results/old.json" if change == "results" else "state/jobs.sqlite3")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"foreign run state")
    before = _snapshot(case.destination)
    with pytest.raises(ValueError, match="conflict|unrelated"):
        _prepare(case, **kwargs)
    assert _snapshot(case.destination) == before


@pytest.mark.parametrize("relation", ["same", "nested", "ancestor", "symlink"])
def test_rejects_overlapping_or_symlinked_roots_without_source_mutation(tmp_path, relation):
    case = _source(tmp_path, counts=(1,))
    before = _snapshot(case.root)
    root = {"same": case.root, "nested": case.root / "new", "ancestor": tmp_path,
            "symlink": tmp_path / "alias"}[relation]
    if relation == "symlink":
        root.symlink_to(case.root, target_is_directory=True)
    with pytest.raises(ValueError, match="distinct|symlink"):
        _prepare(case, root=root)
    assert _snapshot(case.root) == before


@pytest.mark.parametrize("change", ["qa", "question", "split", "identity", "frames", "label",
                                    "logical_label", "logical_hash", "frame_hash", "dependency_hash",
                                    "frame_order", "frame_set_hash", "source_indices", "logical_pixels",
                                    "non_input_payload"])
def test_rejects_source_scene_drift_even_with_updated_archive_receipt(tmp_path, change):
    case = _source(tmp_path, counts=(2,))
    scene = case.scenes["scene-0"]
    if change == "qa":
        scene["qa_ids"].pop()
    elif change == "question":
        scene["questions"][0]["episode"]["qa_id"] = "different-qa"
    elif change == "split":
        scene["split"] = "outer_holdout"
    elif change == "identity":
        scene["cluster_identity_hash"] = "a" * 64
    elif change == "frames":
        scene["frames"].pop()
    elif change == "label":
        scene["questions"][0]["episode"]["ground_truth"] = ""
    elif change == "logical_label":
        scene["questions"][0]["logical_input_identity"]["episode"]["ground_truth"] = "real label"
    elif change == "logical_hash":
        scene["questions"][0]["logical_input_identity"]["sha256"] = "b" * 64
    elif change == "frame_hash":
        scene["frames"][0]["png_sha256"] = "c" * 64
    elif change == "dependency_hash":
        logical = scene["questions"][0]["logical_input_identity"]
        logical["files"]["point_map"]["sha256"] = "d" * 64
        logical["sha256"] = _sha(_bytes({k: v for k, v in logical.items() if k != "sha256"}))
    elif change == "frame_order":
        scene["questions"][0]["episode"]["frames"][0]["frame_idx"] = 1
    elif change == "frame_set_hash":
        scene["questions"][0]["episode"]["frame_set"]["frame_set_hash"] = "e" * 64
    elif change == "source_indices":
        scene["questions"][0]["episode"]["frame_set"]["source_frame_indices"] = list(range(1, 33))
    elif change == "logical_pixels":
        logical = scene["questions"][0]["logical_input_identity"]
        logical["pixels"] = logical["pixels"][:-1]
        logical["sha256"] = _sha(_bytes({k: v for k, v in logical.items() if k != "sha256"}))
    else:
        scene["files"]["/old/bindings.json"] = "data/old-bindings.json"
        scene["file_identities"]["data/old-bindings.json"] = {"sha256": _sha(b"old"), "size": 3}
        case.payloads["scene-0"]["data/old-bindings.json"] = b"old"
    _replace_source_tar(case)
    before = _snapshot(case.root)
    with pytest.raises(ValueError):
        _prepare(case)
    assert not (case.destination / "jobs.json").exists()
    assert not list((case.destination / "inputs/jobs").glob("*.tar"))
    assert _snapshot(case.root) == before


@pytest.mark.parametrize("change", ["status", "job_key", "questions", "geometry", "count",
                                    "duplicate_job", "mixed_identity", "stage", "dependencies"])
def test_source_jobs_and_package_progress_must_agree(tmp_path, change):
    case = _source(tmp_path, counts=(1,))
    jobs = _read(case.root / "coordinator_jobs.json")
    progress = _read(case.root / "receipts/package_progress.json")
    row = progress["scenes"]["scene-0"]
    if change == "status":
        progress["status"] = "running"
    elif change == "job_key":
        row["job_key"] = "another/scene-0/gpu_aggregate"
    elif change == "questions":
        row["questions"] += 1
    elif change == "geometry":
        row["geometry"] = "fresh_required"
    elif change == "count":
        progress["total_scenes"] += 1
    elif change == "duplicate_job":
        jobs["jobs"].append(copy.deepcopy(jobs["jobs"][0]))
    elif change == "mixed_identity":
        other = FrozenIdentity.freeze(json.loads(case.identity.manifest_json), {"method": "other"})
        jobs["jobs"][0]["identity"] = other.to_dict()
    elif change == "stage":
        jobs["jobs"][0]["stage"] = "reconstruction"
    else:
        jobs["jobs"][0]["dependencies"] = ["some-old-run/upstream"]
    _write(case.root / "coordinator_jobs.json", jobs)
    _write(case.root / "receipts/package_progress.json", progress)
    with pytest.raises(ValueError):
        _prepare(case)
    assert not (case.destination / "jobs.json").exists()


@pytest.mark.parametrize("member", ["duplicate", "traversal", "absolute", "symlink", "hardlink",
                                    "unexpected", "missing", "payload_corrupt", "duplicate_json"])
def test_rejects_unsafe_or_incomplete_tars_and_original_payload_corruption(tmp_path, member):
    case = _source(tmp_path, counts=(1,))
    kwargs = {}
    if member == "missing":
        kwargs["omit"] = ["data/frames/0000.png"]
    elif member == "payload_corrupt":
        raw = case.payloads["scene-0"]["data/frames/0000.png"]
        case.payloads["scene-0"]["data/frames/0000.png"] = bytes([raw[0] ^ 1]) + raw[1:]
    elif member == "duplicate_json":
        raw = _bytes(case.scenes["scene-0"])
        kwargs["scene_raw"] = b'{"scene_id":"duplicate",' + raw[1:]
    else:
        names = {"duplicate": "scene.json", "traversal": "data/../escape", "absolute": "/escape",
                 "symlink": "data/link", "hardlink": "data/link", "unexpected": "data/extra"}
        header = tarfile.TarInfo(names[member])
        raw = b"unexpected"
        if member in {"symlink", "hardlink"}:
            header.type = tarfile.SYMTYPE if member == "symlink" else tarfile.LNKTYPE
            header.linkname, raw = "../../escape", None
        else:
            header.size = len(raw)
        kwargs["extra"] = [(header, raw)]
    _replace_source_tar(case, **kwargs)
    before = _snapshot(case.root)
    with pytest.raises(ValueError):
        _prepare(case)
    assert not (case.destination / "jobs.json").exists()
    assert _snapshot(case.root) == before


def test_refuses_same_method_or_run_and_never_overwrites_conflicting_archive(tmp_path):
    case = _source(tmp_path, counts=(1,))
    same_method = tmp_path / "same.json"
    _write(same_method, json.loads(case.identity.method_config_json))
    with pytest.raises(ValueError, match="method identity"):
        _prepare(case, method=same_method)
    with pytest.raises(ValueError, match="run identifier"):
        _prepare(case, run_id="old-r02")
    _prepare(case)
    archive = next((case.destination / "inputs/jobs").glob("*.tar"))
    wrong = {**case.scenes["scene-0"], "cluster_identity_hash": case.identity.identity_hash}
    _tar(archive, wrong, case.payloads["scene-0"])
    before = _snapshot(case.destination)
    with pytest.raises(ValueError, match="expected rekey"):
        _prepare(case)
    assert _snapshot(case.destination) == before


def test_atomic_archive_install_cannot_clobber_an_existing_file(tmp_path):
    temporary, target = tmp_path / "temp", tmp_path / "final"
    temporary.write_bytes(b"new package")
    target.write_bytes(b"existing conflict")
    with pytest.raises(FileExistsError):
        module._install(temporary, target)
    assert target.read_bytes() == b"existing conflict"


@pytest.mark.parametrize("tail", ["missing_end", "hidden_archive"])
def test_rejects_incomplete_or_hidden_tar_tail_with_updated_receipt(tmp_path, tail):
    case = _source(tmp_path, counts=(1,))
    archive = module._archive_path(case.root, case.jobs[0]["job_key"])
    raw = archive.read_bytes()
    with tarfile.open(archive, "r:") as stream:
        list(stream)
        offset = stream.offset
    archive.write_bytes(raw[:offset] if tail == "missing_end" else raw + raw)
    path = case.root / "receipts/package_progress.json"
    progress = _read(path)
    digest, size = hash_file(archive)
    progress["scenes"]["scene-0"].update(archive_sha256=digest, archive_bytes=size)
    _write(path, progress)
    with pytest.raises(ValueError, match="end marker"):
        _prepare(case)
    assert not (case.destination / "jobs.json").exists()


def test_resume_requires_exact_json_types_and_original_payload_bytes(tmp_path):
    case = _source(tmp_path, counts=(1,))
    progress = _prepare(case)
    receipt_path = case.destination / "receipts/package_progress.json"
    progress.update(status="running", scenes={})
    _write(receipt_path, progress)
    for name in ("jobs.json", "coordinator_jobs.json"):
        (case.destination / name).unlink()
    archive = next((case.destination / "inputs/jobs").glob("*.tar"))
    with tarfile.open(archive, "r:") as stream:
        expected = _read_tar_scene(stream)
    changed = copy.deepcopy(expected)
    changed["questions"][0]["episode"]["frames"][0]["timestamp"] = 0
    _tar(archive, changed, case.payloads["scene-0"])
    before = _snapshot(case.destination)
    with pytest.raises(ValueError, match="expected rekey"):
        _prepare(case)
    assert _snapshot(case.destination) == before
    data = dict(case.payloads["scene-0"])
    data["data/geometry/points.bin"] = b"X" * len(data["data/geometry/points.bin"])
    _tar(archive, expected, data)
    before = _snapshot(case.destination)
    with pytest.raises(ValueError, match="payload SHA256"):
        _prepare(case)
    assert _snapshot(case.destination) == before
