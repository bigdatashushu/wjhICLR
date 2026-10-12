#!/usr/bin/env python3
"""Preregister, seal and execute the expanded, label-isolated S0 B01/B11 panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import copy
from dataclasses import asdict, replace
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[1] / "third_party/vggt")]

TASKS = (
    "obj_appearance_order", "object_abs_distance", "object_counting",
    "object_rel_direction", "object_rel_distance", "object_size_estimation",
    "room_size_estimation", "route_planning",
)
SCHEMA = "full-experiment-v11/1"
RESERVE = 12 * 1024 ** 3


class FullExperimentError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise FullExperimentError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    from skill3d.evaluation.skill_ablation_v11 import _encoded
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        stream.write(_encoded(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def safe_row(row):
    from skill3d.routing.task_classifier import canonical_task
    # A whitelist makes the sampling algorithm independent of labels/quality/files.
    keys = ("qa_id", "id", "dataset", "scene_name", "question_type", "question",
            "options", "video_path", "video", "video_id", "language")
    out = {k: copy.deepcopy(row[k]) for k in keys if k in row}
    out["qa_id"] = str(row.get("qa_id") or row.get("id"))
    out["task"] = canonical_task(out["question_type"])
    out["split"] = "inner_validation"
    return out


def scene_cycle(rows, seed, task, covered):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene_name"]].append(row)
    rng = random.Random(f"{seed}:{task}")
    for unseen in (True, False):
        scenes = sorted(s for s in groups if (s not in covered) == unseen)
        rng.shuffle(scenes)
        queues = {s: deque(sorted(groups[s], key=lambda r: r["qa_id"])) for s in scenes}
        while any(queues.values()):
            for scene in scenes:
                if queues[scene]:
                    yield queues[scene].popleft()


def select_panel(metadata, original, allowed, per_task=24, sampling_seed=137):
    import re
    candidates = {}
    for raw in metadata:
        row = safe_row(raw)
        if (row["scene_name"] not in allowed or row["dataset"] not in {"scannet", "scannetpp"}
                or row["task"] not in TASKS):
            continue
        require(not re.search(r"[\u4e00-\u9fff]", json.dumps(
            [row.get("question", ""), row.get("options", [])], ensure_ascii=False)),
            f"non-English question {row['qa_id']}")
        require(row["qa_id"] not in candidates, "duplicate metadata ID")
        candidates[row["qa_id"]] = row
    anchors = {}
    for panel in original["panels"].values():
        for row in panel["questions"]:
            q = str(row["qa_id"])
            require(q in candidates, f"anchor missing from permitted metadata: {q}")
            require(all(candidates[q][k] == row[k] for k in
                        ("dataset", "scene_name", "question_type", "task")),
                    f"anchor metadata changed: {q}")
            anchors[q] = candidates[q]
    selected, receipts = [], {}
    for task in TASKS:
        chosen = sorted((r for r in anchors.values() if r["task"] == task),
                        key=lambda r: r["qa_id"])
        require(len(chosen) <= per_task, f"anchors exceed quota for {task}")
        covered = {r["scene_name"] for r in chosen}
        remaining = [r for q, r in candidates.items() if r["task"] == task and q not in anchors]
        counts = Counter(r["question_type"].rsplit("_", 1)[-1] for r in chosen)
        if task == "object_rel_direction":
            levels = ("easy", "medium", "hard")
            queues = {level: deque(scene_cycle(
                [r for r in remaining if r["question_type"].endswith("_" + level)],
                sampling_seed, task, covered)) for level in levels}
            while len(chosen) < per_task and any(queues.values()):
                level = min((s for s in levels if queues[s]),
                            key=lambda s: (counts[s], levels.index(s)))
                row = queues[level].popleft()
                chosen.append(row)
                counts[level] += 1
        else:
            for row in scene_cycle(remaining, sampling_seed, task, covered):
                if len(chosen) == per_task:
                    break
                chosen.append(row)
        require(len(chosen) == per_task, f"insufficient candidates for {task}")
        for index, row in enumerate(chosen):
            receipts[row["qa_id"]] = {
                "task": task, "selection_index": index, "anchor": row["qa_id"] in anchors,
                "scene_new_to_task_anchors": row["scene_name"] not in covered,
                "sampling_seed": sampling_seed, "safe_metadata_sha256": digest(row),
                "ground_truth_used": False, "availability_used": False,
            }
        selected.extend(chosen)
    require(len({r["qa_id"] for r in selected}) == len(TASKS) * per_task, "panel duplicate")
    return selected, receipts


def make_manifest(rows, original, run_id, seeds=(137, 138), **identity):
    old_runs = {r["run_id"]: r for r in original["runs"]}
    first = []
    for run in original["first_batch"]["run_ids"]:
        r = old_runs[run]
        key = (int(r["seed"]), str(r["qa_id"]))
        if key not in first:
            first.append(key)
    order = first + [(s, r["qa_id"]) for s in seeds for r in rows
                     if (s, r["qa_id"]) not in first]
    valid = {(s, r["qa_id"]) for s in seeds for r in rows}
    require(len(order) == len(valid) and set(order) == valid, "first batch/panel mismatch")
    by_id = {r["qa_id"]: r for r in rows}
    runs = [{**by_id[q], "seed": s, "arm": arm,
             "run_id": f"{run_id}-s{s}-{q}-{arm}", "max_tokens": 8192}
            for s, q in order for arm in ("B01", "B11")]
    value = {
        "schema": SCHEMA, "run_id": run_id, "status": "registered",
        "sampling_seed": identity.pop("sampling_seed", 137), "solver_seeds": list(seeds),
        "questions": rows, "runs": runs, "pair_order": [list(k) for k in order],
        "first_batch_pairs": [list(k) for k in first],
        "planned_solver_count": len(runs), "unique_question_count": len(rows),
        "frozen_skill": "S0-v11-english", "language": "en",
        "allow_evolution": False, "allow_final_test": False,
        "budgets": {"n_frames": 32, "max_pixels": 131072, "max_model_len": 32768,
                    "max_tokens": 8192, "temperature": 0, "max_solver_rounds": 6,
                    "max_retries_per_operation": 3, "finalization_rounds": 1},
        **identity,
    }
    return {**value, "sha256": digest(value)}


def gpu_snapshot():
    import psutil
    def query(args):
        return subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                                       text=True)
    gpus = []
    for line in query(["--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu"]).splitlines():
        idx, uuid, total, used, utilization = [s.strip() for s in line.split(",")]
        gpus.append(dict(index=int(idx), uuid=uuid, total_mib=int(total),
                         used_mib=int(used), utilization=int(utilization)))
    processes = []
    for line in query(["--query-compute-apps=gpu_uuid,pid,used_gpu_memory"]).splitlines():
        gpu, pid, memory = [s.strip() for s in line.split(",")]
        row = {"gpu_uuid": gpu, "pid": int(pid), "memory_mib": memory}
        try:
            p = psutil.Process(int(pid))
            row.update(owner=p.username(), command=p.cmdline(), create_time=p.create_time())
        except psutil.Error as exc:
            row["identity_error"] = str(exc)
        processes.append(row)
    return {"time": time.time(), "gpus": gpus, "processes": processes}


def select_gpus(before, after, preferred=(2, 3, 4)):
    def free(snapshot):
        busy = {p["gpu_uuid"] for p in snapshot["processes"]}
        return {g["index"]: g for g in snapshot["gpus"]
                if g["uuid"] not in busy and g["used_mib"] < 100 and g["total_mib"] >= 20480}
    a, b = free(before), free(after)
    available = {i: b[i] for i in set(a) & set(b) if a[i]["uuid"] == b[i]["uuid"]}
    roles, used = {}, set()
    for role, candidate in zip(("qwen", "geometry", "detector"), preferred):
        options = ([candidate] if candidate in available else []) + sorted(available)
        idx = next((i for i in options if i not in used), None)
        require(idx is not None, "waiting for three unused GPUs with at least 20 GiB each")
        roles[role] = available[idx]
        used.add(idx)
    return roles


def _blob(store, sha):
    require(len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "invalid blob SHA")
    return store / sha[:2] / (sha + ".gz")


def verify_blob(path, sha, size):
    h, count = hashlib.sha256(), 0
    with gzip.open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
            count += len(block)
    require(h.hexdigest() == sha and count == size, f"archive corruption: {path}")


def seal_tree(source, store, receipt_path, *, binary_only=False):
    """Content-addressed gzip retains exact bytes; publish all refs before unlink."""
    from skill3d.evaluation import skill_ablation_v11 as ab
    source, store = Path(source).resolve(), Path(store).resolve()
    require(source != store and not store.is_relative_to(source), "overlapping archive roots")
    identity = ab.tree_identity(source)
    files = {}
    for path in sorted(source.rglob("*")):
        require(not path.is_symlink(), "archive symlink")
        if not path.is_file() or (binary_only and path.suffix not in
                                 {".npy", ".npz", ".png", ".jpg", ".jpeg"}):
            continue
        sha, size = file_hash(path), path.stat().st_size
        blob = _blob(store, sha)
        if not blob.exists():
            blob.parent.mkdir(parents=True, exist_ok=True)
            tmp = blob.with_suffix(".tmp")
            with path.open("rb") as src, tmp.open("wb") as dst:
                with gzip.GzipFile(fileobj=dst, mode="wb", compresslevel=1, mtime=0) as zipped:
                    shutil.copyfileobj(src, zipped, 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            verify_blob(tmp, sha, size)
            os.replace(tmp, blob)
        verify_blob(blob, sha, size)
        files[path.relative_to(source).as_posix()] = {"sha256": sha, "size": size}
    require(ab.tree_identity(source) == identity, "tree changed while sealing")
    receipt = {"schema": "sealed-tree-gzip-cas/1", "directory": str(source),
               "tree": identity, "files": files, "binary_only": binary_only,
               "uncompressed_bytes": sum(r["size"] for r in files.values())}
    write(receipt_path, receipt)
    for name, row in files.items():
        path = source / name
        require(file_hash(path) == row["sha256"], "source changed before archive removal")
        path.unlink()
    return receipt


def restore_tree(receipt, store):
    from skill3d.evaluation import skill_ablation_v11 as ab
    root = Path(receipt["directory"])
    for name, row in receipt["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe archive path")
        path = root / name
        if path.exists():
            require(file_hash(path) == row["sha256"], "changed restored file")
            continue
        blob = _blob(Path(store), row["sha256"])
        verify_blob(blob, row["sha256"], row["size"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".restore")
        with gzip.open(blob, "rb") as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        require(file_hash(tmp) == row["sha256"], "restore identity changed")
        os.replace(tmp, path)
    require(ab.tree_identity(root) == receipt["tree"], "restored tree identity changed")
    return root


def archive_path(path, archive):
    """Map only the documented old campaign prefix; never guess by basename."""
    p = Path(path)
    if p.is_file():
        return p
    prefixes = ("/tmp/harness3d-english-formal-20261012",
                "/home/cvailab/experiments/harness3d-v11/campaigns/english-formal-20261012")
    choices = [archive / p.relative_to(prefix) for prefix in prefixes
               if p.is_relative_to(prefix)]
    found = [p for p in choices if p.is_file()]
    require(len(found) == 1, f"unmapped historical file: {path}")
    return found[0]


def load_frozen(path):
    import numpy as np
    from skill3d.adapters.episode_source import EpisodeItem
    from skill3d.schemas import VSIBenchEpisode, InputErrorRecord
    from skill3d.evaluation import skill_ablation_v11 as ab
    value = read(path)
    require(value["sha256"] == ab._hash({k: v for k, v in value.items() if k != "sha256"}),
            "frozen input payload changed")
    episode = VSIBenchEpisode.model_validate(value["episode"])
    require(episode.ground_truth == "" and episode.split == "inner_validation", "unsafe episode")
    pixels = []
    for ref in value["frame_files"]:
        require(file_hash(ref) == value["frame_file_hashes"][Path(ref).name], "frame changed")
        pixels.append(np.load(ref, allow_pickle=False))
    if pixels:
        require(ab._pixels_identity(pixels) == value["pixels"], "pixels changed")
    error = InputErrorRecord.model_validate(value["input_error"]) if value["input_error"] else None
    return EpisodeItem(episode=episode, pixels=pixels, source=value["source"],
                       video_path=value["video_path"], input_error=error), value


def freeze_inputs(root, rows, archive, raw_cfg):
    import numpy as np
    from skill3d.adapters import episode_source as es, vsibench_loader as vl
    from skill3d.adapters.frame_set import with_pixel_identity
    from skill3d.evaluation import skill_ablation_v11 as ab
    old = read(archive / "input_manifest.json")
    index = read(root / "input_manifest.json") if (root / "input_manifest.json").exists() else {}
    sources = {}
    for row in rows:
        q, path = row["qa_id"], root / "inputs" / (row["qa_id"] + ".json")
        if q in index:
            require(file_hash(path) == index[q]["sha256"], "input resume identity changed")
            load_frozen(path)
            continue
        if q in old:
            original = archive_path(old[q]["path"], archive)
            require(file_hash(original) == old[q]["sha256"], "historical input changed")
            value = read(original)
            require(value["sha256"] == ab._hash({k: v for k, v in value.items() if k != "sha256"}),
                    "historical payload changed")
            value["frame_files"] = [str(archive_path(p, archive)) for p in value["frame_files"]]
        else:
            safe = {**row, "ground_truth": ""}
            key = (row["dataset"], row["scene_name"])
            if key not in sources:
                sources[key] = es._load_from_sources(
                    safe, raw_cfg["paths"]["raw_videos"],
                    raw_cfg["paths"].get("raw_video_fallbacks"), 32)
            loaded, attempts = sources[key]
            if loaded is None:
                err = es._input_error_record(safe, "inner_validation",
                                            es._source_failure_reason(attempts), attempts)
                value = {"qa_id": q, "episode": err.as_episode(), "input_error": err,
                         "frame_files": [], "source": "vsi_bench", "video_path": ""}
            else:
                pixels, missing = loaded["pixels"], set(loaded["missing"])
                fs = vl.build_frame_set(
                    loaded["total"], n_frames=32, fps=loaded["fps"], episode_id=q,
                    dataset_id=row["dataset"], video_id=vl.video_id_for(row, loaded["source"]),
                    scene_name=row["scene_name"],
                    readable_frame_ids=[i for i in loaded["planned"] if i not in missing])
                fs = with_pixel_identity(fs, pixels)
                directory = root / "frames" / fs.cache_identity()
                directory.mkdir(parents=True, exist_ok=True)
                refs = []
                for i, image in enumerate(pixels):
                    ref = directory / f"frame_{i:04d}.npy"
                    if not ref.exists():
                        np.save(ref, image, allow_pickle=False)
                    refs.append(str(ref))
                value = {
                    "qa_id": q, "episode": es._episode_from_row(
                        safe, "inner_validation", pixels, 32, frame_set=fs), "input_error": None,
                    "frame_files": refs, "source": "vsi_bench",
                    "video_path": str(loaded["source"].path), "source_attempts": attempts,
                    "cache_identity": fs.cache_identity(), "pixels": ab._pixels_identity(pixels),
                    "frame_file_hashes": {Path(p).name: file_hash(p) for p in refs},
                }
        value = ab._clean(value)
        value["sha256"] = ab._hash({k: v for k, v in value.items() if k != "sha256"})
        write(path, value)
        item, value = load_frozen(path)
        require(all(getattr(item.episode, k) == row[k] for k in
                    ("qa_id", "scene_name", "dataset", "question_type")), "panel/input mismatch")
        index[q] = {"path": str(path), "sha256": file_hash(path),
                    "cache_identity": value.get("cache_identity"), "input_error": bool(item.input_error)}
        write(root / "input_manifest.json", index)
        print(f"frozen {q}", flush=True)
    return index


def runtime_cfg(path, seed):
    from skill3d.online.config import (load_config, resolve_runtime, sandbox_from,
                                      active_vision_from, retrieval_policy_from)
    from skill3d.online.runner import OnlineRunConfig
    import yaml
    envelope = yaml.safe_load(Path(path).read_text())
    extension = envelope.pop("experiment", {})
    online_path = Path(path).with_name(Path(path).stem + ".online.yaml")
    if online_path.exists():
        require(yaml.safe_load(online_path.read_text()) == envelope, "derived runtime config changed")
    else:
        online_path.write_text(yaml.safe_dump(envelope, sort_keys=False))
    raw = resolve_runtime(load_config(online_path), seed=seed, formal=False)
    vision, sandbox, model = active_vision_from(raw), sandbox_from(raw), raw["vllm"]
    cfg = OnlineRunConfig(
        mode="real", seed=seed, vllm_endpoints=model["endpoints"], vllm_model=model["model"],
        max_images=model["n_frames"], max_pixels=model["max_pixels"],
        max_model_len=model["max_model_len"], max_tokens=model["max_tokens"],
        temperature=model["temperature"], cell_timeout_s=sandbox.cell_timeout_s,
        max_regen=sandbox.max_regenerate, max_solver_rounds=raw["max_solver_rounds"],
        max_retries_per_operation=raw["max_retries_per_operation"],
        finalization_rounds=raw["finalization_rounds"], eval_visual_fallback=raw["eval_visual_fallback"],
        input_diagnostics=raw.get("input_diagnostics", False), image_layout=vision.layout,
        max_derived_images=vision.max_derived_images, retrieval_policy=retrieval_policy_from(raw),
        model_weights_sha256=os.environ.get("SKILL3D_MODEL_WEIGHTS_SHA256", ""),
        environment_sha256=os.environ.get("SKILL3D_ENVIRONMENT_SHA256", ""))
    require((cfg.max_images, cfg.max_pixels, cfg.max_model_len, cfg.max_tokens,
             cfg.temperature, cfg.max_solver_rounds, cfg.max_retries_per_operation,
             cfg.finalization_rounds) == (32, 131072, 32768, 8192, 0, 6, 3, 1),
            "full experiment budget changed")
    require(len(cfg.vllm_endpoints) == 1 and cfg.eval_visual_fallback
            and not cfg.allow_final_test, "full experiment endpoint/fallback/split changed")
    raw["experiment"] = extension
    return cfg, raw


def register(args):
    import yaml
    from skill3d.adapters.split_builder import validate_split_mapping
    original = read(args.manifest)
    require(file_hash(args.metadata) == original["metadata_sha256"], "metadata SHA changed")
    split_path = Path(__file__).resolve().parents[1] / "configs/vsi_bench_split.yaml"
    require(file_hash(split_path) == original["split_config_sha256"], "split SHA changed")
    allowed = validate_split_mapping(yaml.safe_load(split_path.read_text()), "inner_validation")
    source = [json.loads(line) for line in Path(args.metadata).read_text().splitlines() if line]
    rows, receipts = select_panel(source, original, allowed, args.per_task, args.sampling_seed)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    labels = {str(r.get("qa_id") or r.get("id")): r.get("ground_truth") for r in source}
    write(root / "scoring/reference_answers.json", {r["qa_id"]: labels[r["qa_id"]] for r in rows})
    manifest = make_manifest(rows, original, root.name, tuple(args.seeds),
                             code_ref=args.code_ref, metadata_sha256=file_hash(args.metadata),
                             original_manifest_sha256=file_hash(args.manifest),
                             split_config_sha256=file_hash(split_path),
                             sampling_seed=args.sampling_seed, config_sha256=file_hash(args.config))
    write(root / "expanded_panel_192.json", {
        "questions": rows, "sampling_receipts": receipts, "per_task": dict(Counter(
            r["task"] for r in rows)), "scenes": dict(Counter(r["scene_name"] for r in rows)),
        "anchor_mapping": {q: [r["run_id"] for r in original["runs"] if r["qa_id"] == q]
                           for q, v in receipts.items() if v["anchor"]}})
    write(root / "execution_manifest_768.json", manifest)
    print(json.dumps({"status": "registered", "questions": len(rows),
                      "solver_runs": len(manifest["runs"])}), flush=True)
    return manifest


def verify_run(args):
    root = Path(args.output).resolve()
    manifest = read(root / "execution_manifest_768.json")
    require(manifest["sha256"] == digest({k: v for k, v in manifest.items() if k != "sha256"}),
            "execution manifest changed")
    require(manifest["config_sha256"] == file_hash(args.config)
            and manifest["code_ref"] == args.code_ref, "run config/code differs")
    actual = subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]),
                                      "rev-parse", "HEAD"], text=True).strip()
    require(actual == args.code_ref, "executing commit differs from frozen code")
    status_text = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "status", "--porcelain"],
        text=True).strip()
    require(not status_text, "full experiment requires a clean immutable checkout")
    return root, manifest


def freeze_runtime_identity(args):
    """Rehash existing weights and the executing environment; never install/download."""
    import importlib.metadata
    import yaml
    from skill3d.infra.experiment_startup import _source_paths, encoded
    from skill3d.evaluation import skill_ablation_v11 as ab
    root, manifest = verify_run(args)
    code = Path(__file__).resolve().parents[1]
    settings = yaml.safe_load(Path(args.config).read_text())["experiment"]["full_services"]
    weights = read(settings["weights_manifest"])
    verified = []
    for name, model in weights["models"].items():
        for record in model["files"]:
            path = Path(model["local_path"]) / record["path"]
            require(path.is_file() and path.stat().st_size == record["size"]
                    and file_hash(path) == record["sha256"], f"weight identity changed: {path}")
            verified.append({"model": name, "path": str(path), **record})
    detector = weights["detector"]
    require(file_hash(detector["path"]) == detector["sha256"], "detector weight changed")
    qwen = weights["models"]["qwen"]
    model_id = {k: qwen[k] for k in ("repo_id", "revision")}
    model_id["files"] = [{k: r[k] for k in ("path", "size", "sha256")} for r in qwen["files"]]
    model_sha = digest(model_id)
    environment = {
        "python": sys.executable, "version": sys.version,
        "packages": sorted((d.metadata["Name"], d.version)
                           for d in importlib.metadata.distributions() if d.metadata["Name"]),
        "weights": verified, "detector": detector,
    }
    env_sha = digest(environment)
    names, untracked = _source_paths(code)
    require(not untracked, "untracked runtime source")
    files = [{"path": n, "sha256": file_hash(code / n), "size": (code / n).stat().st_size,
              "executable": bool((code / n).stat().st_mode & 0o111)}
             for n in names if (code / n).is_file()]
    receipt = {
        "schema_version": "experiment-source/1.0", "run_mode": "formal",
        "base_commit": args.code_ref, "git_dirty": False, "source_root": str(code),
        "files": files, "source_sha256": hashlib.sha256(encoded(files)).hexdigest(),
        "input_files": [],
    }
    identity = {"manifest_sha256": manifest["sha256"], "code": receipt,
                "environment": environment, "environment_sha256": env_sha,
                "model_weights_sha256": model_sha, "quality_contract": ab.quality_contract()}
    path = root / "runtime_identity.json"
    if path.exists():
        require(read(path) == ab._clean(identity), "runtime identity differs from registered environment")
    else:
        write(path, identity)
        write(root / "code_identity.json", receipt)
    os.environ.update(
        SKILL3D_CODE_RECEIPT=str(root / "code_identity.json"), SKILL3D_RUN_MODE="formal",
        SKILL3D_MODEL_WEIGHTS_SHA256=model_sha, SKILL3D_ENVIRONMENT_SHA256=env_sha,
        SKILL3D_VGGT_CHECKPOINT=weights["models"]["vggt"]["local_path"],
        SKILL3D_SAM2_CHECKPOINT=str(
            Path(weights["models"]["sam2"]["local_path"]) / "sam2.1_hiera_large.pt"),
        SKILL3D_MOGE2_CHECKPOINT=settings["moge2_checkpoint"],
        SKILL3D_DETECTOR_RECEIPT=str(root / "services/detector/groundingdino_startup.json"),
    )
    return identity


def verified_historical_geometry(entry, item, value, archive):
    """Migrate a legacy missing input ID only from its content-bound evidence."""
    from skill3d.evaluation import skill_ablation_v11 as ab
    from skill3d.schemas import ReconstructionArtifact

    source = archive_path(entry["path"], archive)
    require(file_hash(source) == entry["sha256"], "historical geometry changed")
    art = ReconstructionArtifact.model_validate_json(source.read_bytes())
    data = art.model_dump()
    for name in ab.ARTIFACT_REFS:
        if data.get(name):
            data[name] = str(archive_path(data[name], archive))
    for name, ref in data["confidence"].items():
        if ref:
            data["confidence"][name] = str(archive_path(ref, archive))
    art = ReconstructionArtifact.model_validate(data)
    fs = item.episode.frame_set
    slots = [fs.frame_ids.index(fid) for fid in fs.readable_frame_ids]
    key = value["cache_identity"]
    require(art.scene_name == item.episode.scene_name
            and art.frame_set_hash == fs.frame_set_hash
            and art.frame_ids == fs.readable_frame_ids
            and art.source_frame_indices == [fs.source_frame_indices[i] for i in slots]
            and art.timestamps == [fs.timestamps[i] for i in slots],
            "historical geometry frame identity mismatch")
    require(not art.reconstruction_input_identity or art.reconstruction_input_identity == key,
            "historical geometry input mismatch")
    require(bool(entry.get("quality_confirmation")), "historical geometry evidence missing")
    confirmation_path = archive_path(entry["quality_confirmation"], archive)
    confirmation = read(confirmation_path)
    evidence_path = archive_path(confirmation["evidence_ref"], archive)
    require(file_hash(evidence_path) == confirmation["evidence_sha256"],
            "historical geometry evidence changed")
    evidence = read(evidence_path)
    actual = {name: file_hash(path) for name, path in ab._artifact_files(art, source).items()}
    require(evidence["artifact"]["sha256"] == entry["sha256"]
            and {name: row["sha256"] for name, row in evidence["array_hashes"].items()} == actual
            and evidence["cache_identity"] == key
            and evidence["input_pixels"] == value["pixels"] == ab._pixels_identity(item.pixels)
            and evidence["source_frame_indices"] == art.source_frame_indices,
            "historical geometry content/input evidence mismatch")
    return art.model_copy(update={"reconstruction_input_identity": key}), {
        "source": str(source), "sha256": entry["sha256"],
        "evidence_ref": str(evidence_path), "evidence_sha256": file_hash(evidence_path),
        "input_identity": key, "legacy_input_identity_migrated": not bool(
            data.get("reconstruction_input_identity")),
    }


def historical_geometry_index(archive):
    """Cache lookup follows frozen RGB identity across questions."""
    from skill3d.evaluation import skill_ablation_v11 as ab

    artifacts, inputs = read(archive / "artifact_manifest.json"), read(archive / "input_manifest.json")
    by_input = {}
    for q, entry in sorted(artifacts.items()):
        require(q in inputs, "historical geometry has no frozen input")
        path = archive_path(inputs[q]["path"], archive)
        require(file_hash(path) == inputs[q]["sha256"], "historical cached input changed")
        value = read(path)
        require(value["sha256"] == ab._hash({k: v for k, v in value.items() if k != "sha256"}),
                "historical cached input payload changed")
        key = value.get("cache_identity")
        if not key or value.get("input_error"):
            continue
        if key in by_input:
            require(by_input[key]["sha256"] == entry["sha256"],
                    "historical geometry identity has multiple contents")
        else:
            by_input[key] = {**entry, "historical_qa_id": q}
    return by_input


def _prepare_geometry(root, inputs, archive):
    from skill3d.evaluation import skill_ablation_v11 as ab
    from skill3d.reconstruction import metric_evidence as me
    from skill3d.reconstruction.metric_fusion import make_moge2_model
    from skill3d.reconstruction.vggt_runner import reconstruct, ReconstructionFailed
    from skill3d.reconstruction_gate.quality_metrics import compute_and_store_quality
    from skill3d.schemas import ReconstructionArtifact
    import numpy as np
    old = historical_geometry_index(archive)
    geometry = read(root / "geometry_manifest.json") if (root / "geometry_manifest.json").exists() else {}
    model = make_moge2_model(device="cuda:0")
    contract = me.fusion_contract(model)
    for q, record in inputs.items():
        if record["input_error"]:
            continue
        item, value = load_frozen(record["path"])
        key = value["cache_identity"]
        if key in geometry:
            prior = geometry[key]
            require(file_hash(prior["artifact"]) == prior["sha256"], "geometry resume changed")
            if prior.get("metric_contract_sha256"):
                require(prior["metric_contract_sha256"] == digest(contract), "metric contract changed")
            continue
        dest = root / "geometry" / key
        provenance = {}
        started = time.monotonic()
        try:
            if key in old:
                old_art, provenance = verified_historical_geometry(old[key], item, value, archive)
                provenance["historical_qa_id"] = old[key]["historical_qa_id"]
                art = me.copy_geometry(old_art, dest)
            else:
                art = reconstruct(item.pixels, item.episode.scene_name, dest,
                                  frame_set=item.episode.frame_set, compute_quality=False)
            art = me.attach_metric_evidence(art, item.pixels, item.episode.frame_set, model, dest)
            art = compute_and_store_quality(
                art, frames=item.pixels, depth_maps=np.load(art.depth_maps),
                c2w_list=np.load(art.c2w_list), intrinsics=np.load(art.intrinsics),
                point_map=np.load(art.point_map),
                depth_conf=np.load(art.depth_conf) if art.depth_conf else None)
            status, error = "completed", None
        except ReconstructionFailed as exc:
            # Runner's existing 2D route consumes an explicit unavailable artifact.
            from skill3d.schemas import ConfidenceMap
            dest.mkdir(parents=True, exist_ok=True)
            fs = item.episode.frame_set
            art = ReconstructionArtifact(
                artifact_id=f"failed-{item.episode.scene_name}", artifact_version=key[:16],
                scene_name=item.episode.scene_name, recon_method="vggt",
                frame_ids=fs.readable_frame_ids, source_frame_indices=[
                    fs.source_frame_indices[fs.frame_ids.index(f)] for f in fs.readable_frame_ids],
                timestamps=[fs.timestamps[fs.frame_ids.index(f)] for f in fs.readable_frame_ids],
                frame_set_hash=fs.frame_set_hash, reconstruction_input_identity=key,
                c2w_list="", intrinsics="", depth_maps="",
                point_map="", point_conf="", quality_status="not_computed",
                confidence=ConfidenceMap(per_point_confidence=""))
            status, error = "reconstruction_failed", str(exc)
        path = dest / "artifact.json"
        path.write_text(art.model_dump_json())
        geometry[key] = {"artifact": str(path), "sha256": file_hash(path), "status": status,
                         "error": error, "provenance": provenance,
                         "runtime_s": time.monotonic() - started,
                         "metric_contract_sha256": digest(contract) if status == "completed" else None}
        evidence = {
            "artifact": geometry[key], "input": ab.input_identity(item, path),
            "quality_contract": ab.quality_contract(),
            "metric_receipt": me.validated_receipt(art) if art.metric_model == "moge2" else None,
            "reference_access": False,
        }
        evidence_path = dest / "current_quality_evidence.json"
        write(evidence_path, evidence)
        geometry[key]["evidence_ref"] = str(evidence_path)
        geometry[key]["evidence_sha256"] = file_hash(evidence_path)
        write(root / "geometry_manifest.json", geometry)
        print(f"geometry {item.episode.scene_name}: {status}", flush=True)
    del model
    import torch
    torch.cuda.empty_cache()
    return {q: geometry[r["cache_identity"]]["artifact"] for q, r in inputs.items()
            if not r["input_error"]}


def prepare(args):
    from skill3d.evaluation import skill_ablation_v11 as ab
    from scripts.run_metric_validation_v11 import raw_requests
    root, manifest = verify_run(args)
    freeze_runtime_identity(args)
    cfg, raw = runtime_cfg(args.config, args.seeds[0])
    confirmation_ref = raw.get("experiment", {}).get("full_services", {}).get("quality_confirmation")
    confirmation = read(confirmation_ref) if confirmation_ref else None
    if confirmation:
        require(confirmation.get("confirmed") is True
                and confirmation.get("sha256") == ab.quality_contract()["sha256"]
                and confirmation.get("evidence_ref"), "six-case quality confirmation changed")
    require(shutil.disk_usage(root).free > RESERVE, "insufficient free disk reserve")
    inputs = freeze_inputs(root, manifest["questions"], Path(args.archive_root), raw)
    artifacts = _prepare_geometry(root, inputs, Path(args.archive_root))
    groups = defaultdict(list)
    for row in manifest["questions"]:
        groups[inputs[row["qa_id"]].get("cache_identity") or ("error-" + row["qa_id"])].append(row)
    prepared = read(root / "preparation_manifest.json") if (root / "preparation_manifest.json").exists() else {
        "manifest_sha256": manifest["sha256"], "status": "preparing", "pairs": {}}
    require(prepared["manifest_sha256"] == manifest["sha256"], "preparation manifest changed")
    for scene, rows in sorted(groups.items()):
        qids = [r["qa_id"] for r in rows]
        done = [f"{s}:{q}" in prepared["pairs"] for s in args.seeds for q in qids]
        if all(done):
            continue
        require(not any(done), "partially sealed scene requires retained-scene recovery")
        items = [load_frozen(inputs[q]["path"])[0] for q in qids]
        work = []
        room_cache = {}
        for seed in args.seeds:
            output = root / "preparation" / f"s{seed}" / scene
            if output.exists():
                prior = read(output / "manifest.json")
                require(prior["status"] == "prepared", "unfinished preparation; inspect in-flight evidence")
            else:
                with raw_requests(output):
                    ab.run_skill_ablation_v11(
                        items, artifact_paths={q: artifacts[q] for q in qids if q in artifacts},
                        output_dir=output, base_cfg=replace(cfg, seed=seed), seed=seed,
                        question_types=TASKS, expected_qa_ids=qids, library_root=args.library_root,
                        prepare_only=True, quality_confirmation=confirmation,
                        prepared_room_cache=room_cache)
            work.append((seed, output, read(output / "manifest.json")))
        # Audit both seeds' candidates together before publishing the final trees.
        from skill3d.segmentation.sam2_tracker import audit_semantic_mask_conflicts, refresh_grounding_coverage
        from skill3d.online import runner
        bindings = {}
        for seed, output, prior in work:
            for q in qids:
                if not inputs[q]["input_error"]:
                    tree = Path(prior["prepared"][q]["directory"])
                    bindings[f"{seed}:{q}"] = (tree, ab._load_binding(tree), q)
        audited, audits = audit_semantic_mask_conflicts(
            [(key, value[1].objects) for key, value in sorted(bindings.items())],
            scene_identity=scene)
        for key, (tree, binding, q) in bindings.items():
            binding.objects = audited[key]
            binding.stats["semantic_conflict_audits"] = [a for a in audits if key in a["qa_ids"]]
            if binding.stats.get("grounding"):
                binding.stats["grounding"] = refresh_grounding_coverage(binding.stats["grounding"], binding.objects)
            from skill3d.schemas import ReconstructionArtifact
            art = ReconstructionArtifact.model_validate_json((tree / "artifact.json").read_bytes())
            item = next(i for i in items if i.episode.qa_id == q)
            _, _, art = runner._scene_from_artifact(
                art, item.pixels, binding.stats, binding.objects, item.episode,
                objects_materialized=binding.materialized, persist_quality=False)
            ab._save_binding(binding, tree)
            (tree / "artifact.json").write_text(art.model_dump_json())
        for seed, output, prior in work:
            for q in qids:
                entry = prior["prepared"][q]
                if not inputs[q]["input_error"]:
                    entry.update(ab.tree_identity(Path(entry["directory"])))
            write(output / "manifest.json", prior)
            for q in qids:
                key = f"{seed}:{q}"
                if inputs[q]["input_error"]:
                    prepared["pairs"][key] = {"preparation_manifest": str(output / "manifest.json"),
                                              "input_error": True}
                else:
                    tree = Path(prior["prepared"][q]["directory"])
                    receipt_path = root / "sealed" / f"s{seed}-{q}.json"
                    receipt = seal_tree(tree, root / "blobs", receipt_path)
                    prepared["pairs"][key] = {
                        "preparation_manifest": str(output / "manifest.json"),
                        "preparation_manifest_sha256": file_hash(output / "manifest.json"),
                        "receipt": str(receipt_path), "receipt_sha256": file_hash(receipt_path),
                        "tree_sha256": receipt["tree"]["sha256"]}
        write(root / "preparation_manifest.json", prepared)
        require(shutil.disk_usage(root).free > RESERVE, "disk reserve exhausted; retained prepared evidence")
        print(f"sealed scene {scene}: {len(rows) * len(args.seeds)} pairs", flush=True)
    require(len(prepared["pairs"]) == len(manifest["pair_order"]), "preparation coverage incomplete")
    prepared["status"] = "prepared"
    write(root / "preparation_manifest.json", prepared)


def check_completed(root, key, manifest):
    path = root / "pairs" / key / "completion.json"
    if not path.exists():
        return None
    value = read(path)
    require(value["manifest_sha256"] == manifest["sha256"], "completed pair belongs to another run")
    for name, sha in value["refs"].items():
        require(file_hash(root / name) == sha, f"completed evidence changed: {name}")
    if value.get("binary_archive"):
        archive = read(root / value["binary_archive"])
        for name, row in archive["files"].items():
            blob = _blob(root / "blobs", row["sha256"])
            require(blob.is_file(), f"completed archive blob missing: {name}")
    for name, row in value.get("surviving_binary_files", {}).items():
        require(file_hash(root / name) == row["sha256"], "completed binary changed")
    for packet in (root / "pairs" / key / "full_requests").glob("*.json"):
        require(read(packet)["status"] != "in_flight", "unresolved completed request")
    return value


def run_schedule(root, manifest, callback, first_check):
    """Never resubmit an unresolved pair; the first-batch gate automatically continues."""
    first_size = len(manifest["first_batch_pairs"])
    for index, (seed, q) in enumerate(manifest["pair_order"]):
        key = f"s{seed}-{q}"
        if check_completed(root, key, manifest) is None:
            require(not (root / "pairs" / key).exists(), f"unfinished pair requires resolution: {key}")
            callback(seed, q, key)
        if index + 1 == first_size:
            gate = root / "first_batch_acceptance.json"
            if not gate.exists():
                first_check(manifest["first_batch_pairs"])
                write(gate, {"manifest_sha256": manifest["sha256"], "status": "passed",
                             "pairs": first_size, "labels_read": False, "auto_continue": True})
    return status(root, manifest)


def solve(args):
    from skill3d.evaluation import skill_ablation_v11 as ab
    from scripts.run_metric_validation_v11 import raw_requests
    root, manifest = verify_run(args)
    freeze_runtime_identity(args)
    prep = read(root / "preparation_manifest.json")
    require(prep["status"] == "prepared" and prep["manifest_sha256"] == manifest["sha256"],
            "all public inputs must be sealed before solving")
    inputs, geometry = read(root / "input_manifest.json"), read(root / "geometry_manifest.json")
    require(set(prep["pairs"]) == {f"{s}:{q}" for s, q in manifest["pair_order"]}, "prepared pair coverage")

    def one(seed, q, key):
        cfg, _ = runtime_cfg(args.config, seed)
        p = prep["pairs"][f"{seed}:{q}"]
        if p.get("receipt"):
            require(file_hash(p["receipt"]) == p["receipt_sha256"], "sealed preparation changed")
            require(file_hash(p["preparation_manifest"]) == p["preparation_manifest_sha256"],
                    "preparation manifest changed")
            receipt = read(p["receipt"])
            need = receipt["uncompressed_bytes"] * 5
            require(shutil.disk_usage(root).free > RESERVE + need, "insufficient space for private arm copies")
            restored = restore_tree(receipt, root / "blobs")
        else:
            restored = None
        item, _ = load_frozen(inputs[q]["path"])
        artifact = None if item.input_error else geometry[inputs[q]["cache_identity"]]["artifact"]
        output = root / "pairs" / key
        with raw_requests(output):
            prior = read(p["preparation_manifest"])
            ab.run_skill_ablation_v11(
                [item], artifact_paths={q: artifact} if artifact else {}, output_dir=output,
                base_cfg=cfg, seed=seed, question_types=TASKS, expected_qa_ids=[q],
                library_root=args.library_root, prepared_manifest=p["preparation_manifest"],
                quality_confirmation=prior.get("quality_confirmation"),
                reconstruction_costs=({q: geometry[inputs[q]["cache_identity"]].get("runtime_s", 0)}
                                      if artifact else {}))
        pair = json.loads((output / "paired_results.jsonl").read_text())
        require(pair["status"] == "completed" and pair["comparable"], "incomplete/noncomparable pair retained")
        for row in pair["arms"].values():
            require(row["scoring_status"] == "reference_unavailable" and row["score"] is None,
                    "online solver read labels")
        refs = {str(path.relative_to(root)): file_hash(path)
                for path in output.rglob("*") if path.is_file() and path.suffix in {".json", ".jsonl"}}
        # Only this now-quiescent completed pair loses duplicate binary files.
        archive_path = root / "completed_archives" / (key + ".json")
        seal_tree(output, root / "blobs", archive_path, binary_only=True)
        refs[str(archive_path.relative_to(root))] = file_hash(archive_path)
        write(output / "completion.json", {
            "manifest_sha256": manifest["sha256"], "seed": seed, "qa_id": q, "refs": refs,
            "binary_archive": str(archive_path.relative_to(root))})
        if restored:
            require(ab.tree_identity(restored) == receipt["tree"], "solver mutated sealed source")
            for name in receipt["files"]:
                (restored / name).unlink()
        write(root / "progress.json", status(root, manifest))
        print(f"completed {key}", flush=True)

    def first_check(pairs):
        for seed, q in pairs:
            output = root / "pairs" / f"s{seed}-{q}"
            pair = json.loads((output / "paired_results.jsonl").read_text())
            require(pair["comparable"], "first batch pair identity failed")
            for arm, row in pair["arms"].items():
                require(row["delivery_ok"] and row["initial_tree_sha256"] ==
                        prep["pairs"][f"{seed}:{q}"].get("tree_sha256"),
                        "first batch Skill/tree mismatch")
                require(row["eval_visual_request_count"] <= 1, "visual request budget changed")
                for request in row["requests"]:
                    require(request["seed"] == seed, "solver seed changed")
            require(pair["arms"]["B01"]["first_common_request_sha256"] ==
                    pair["arms"]["B11"]["first_common_request_sha256"], "common first prompt changed")
    result = run_schedule(root, manifest, one, first_check)
    if result["completed_solver_count"] == manifest["planned_solver_count"]:
        refs = {str(p.relative_to(root)): file_hash(p) for p in (root / "pairs").rglob("*")
                if p.is_file() and p.name in {"paired_results.jsonl", "completion.json"}}
        write(root / "answers_frozen.json", {"manifest_sha256": manifest["sha256"], "refs": refs,
                                            "solver_count": result["completed_solver_count"]})
    return result


def status(root, manifest):
    completed, failed = 0, []
    for seed, q in manifest["pair_order"]:
        key = f"s{seed}-{q}"
        if check_completed(root, key, manifest):
            completed += 2
        elif (root / "pairs" / key).exists():
            failed.append(key)
    return {"manifest_sha256": manifest["sha256"], "planned_solver_count": manifest["planned_solver_count"],
            "completed_solver_count": completed, "incomplete_pairs": failed,
            "status": "completed" if completed == manifest["planned_solver_count"] else "incomplete"}


def score(args):
    from skill3d.online.runner import _evaluation_values
    from skill3d.routing.task_classifier import classify
    from skill3d.schemas import VSIBenchEpisode
    root, manifest = verify_run(args)
    frozen = read(root / "answers_frozen.json")
    require(frozen["solver_count"] == manifest["planned_solver_count"]
            and frozen["manifest_sha256"] == manifest["sha256"], "answers not all frozen")
    for name, sha in frozen["refs"].items():
        require(file_hash(root / name) == sha, "frozen answer changed")
    labels = read(root / "scoring/reference_answers.json")
    by_q = {r["qa_id"]: r for r in manifest["questions"]}
    results = []
    for seed, q in manifest["pair_order"]:
        pair = json.loads((root / "pairs" / f"s{seed}-{q}" / "paired_results.jsonl").read_text())
        for arm, row in pair["arms"].items():
            episode = load_frozen(read(root / "input_manifest.json")[q]["path"])[0].episode
            episode.ground_truth = str(labels[q] or "")
            evaluated = _evaluation_values(
                episode, row["answer"], classify(episode),
                force_wrong=row["final_state"] in {"unanswerable", "abstain", "run_error", "input_error"})
            results.append({"seed": seed, "qa_id": q, "arm": arm, "task": row["task"],
                            "frozen_answer": row["answer"], **evaluated})
    groups = defaultdict(list)
    for row in results:
        for key in (f"s{row['seed']}/{row['task']}/{row['arm']}",
                    f"s{row['seed']}/all/{row['arm']}"):
            groups[key].append(row)
    summaries = {key: {"n": len(values), "n_scored": sum(v["scoring_status"] == "scored" for v in values),
                      "mean_score": (sum(v["score"] for v in values if v["score"] is not None)
                                     / sum(v["score"] is not None for v in values)
                                     if any(v["score"] is not None for v in values) else None)}
                 for key, values in groups.items()}
    write(root / "scoring/results.json", {"answers_sha256": file_hash(root / "answers_frozen.json"),
                                          "reference_sha256": file_hash(root / "scoring/reference_answers.json"),
                                          "rows": results, "groups": summaries})


def port_available(port):
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def monitor(root, stop):
    while not stop.is_set():
        sample = {"time": time.time(), "free_bytes": shutil.disk_usage(root).free}
        try:
            sample.update(gpu_snapshot())
        except Exception as exc:
            sample["sampling_error"] = f"{type(exc).__name__}: {exc}"
        with (root / "resources.jsonl").open("a") as stream:
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
        stop.wait(5)


def start_services(args):
    """Commands/weights are explicit configuration; never attach to an unrelated service."""
    import yaml
    import psutil
    root, manifest = verify_run(args)
    raw = yaml.safe_load(Path(args.config).read_text())
    settings = raw["experiment"]["full_services"]
    require(raw["vllm"]["endpoints"] == [f"http://127.0.0.1:{args.qwen_port}"],
            "config endpoint differs from requested Qwen port")
    while True:
        before = gpu_snapshot()
        time.sleep(5)
        after = gpu_snapshot()
        write(root / "gpu_checks" / (str(time.time_ns()) + ".json"), {"before": before, "after": after})
        try:
            roles = select_gpus(before, after, (args.qwen_gpu, args.geometry_gpu, args.detector_gpu))
            break
        except FullExperimentError as exc:
            write(root / "waiting.json", {"status": "waiting_for_gpus", "reason": str(exc)})
            print(str(exc), flush=True)
            time.sleep(55)
    require(port_available(args.qwen_port) and port_available(args.detector_port), "requested service port busy")
    require(select_gpus(after, gpu_snapshot(), tuple(r["index"] for r in roles.values())) == roles,
            "GPU allocation changed before startup")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    qcommand = list(settings["qwen_command"])
    for flag, value in (("--port", args.qwen_port), ("--max-model-len", 32768),
                        ("--tensor-parallel-size", 1)):
        require(flag in qcommand, f"Qwen command lacks {flag}")
        qcommand[qcommand.index(flag) + 1] = str(value)
    dcommand = [settings["detector_python"], str(Path(__file__).with_name("serve_groundingdino.py")),
                "--gpu", str(roles["detector"]["index"]), "--port", str(args.detector_port),
                "--checkpoint-path", settings["detector_checkpoint"],
                "--log-dir", str(root / "services/detector")]
    processes = {}
    for role, command in (("qwen", qcommand), ("detector", dcommand)):
        log = root / "services" / (role + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as stream:
            p = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                 env=dict(env, CUDA_VISIBLE_DEVICES=str(roles[role]["index"])),
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        processes[role] = {"pid": p.pid, "create_time": psutil.Process(p.pid).create_time(),
                           "command": command, "gpu": roles[role], "log": str(log)}
        write(root / "services.json", {"roles": roles, "processes": processes})
    os.environ["CUDA_VISIBLE_DEVICES"] = str(roles["geometry"]["index"])
    os.environ["SKILL3D_DETECTOR_ENDPOINT"] = f"http://127.0.0.1:{args.detector_port}"
    os.environ["SKILL3D_DETECTOR_RECEIPT"] = str(
        root / "services/detector/groundingdino_startup.json")
    deadline = time.monotonic() + 900
    urls = [f"http://127.0.0.1:{args.qwen_port}/health",
            f"http://127.0.0.1:{args.detector_port}/health"]
    while time.monotonic() < deadline:
        require(all(psutil.Process(p["pid"]).is_running() for p in processes.values()),
                "new service exited; inspect retained log")
        try:
            if all(urllib.request.urlopen(url, timeout=2).status == 200 for url in urls):
                write(root / "services_ready.json", {"status": "ready", "processes": processes})
                return
        except Exception:
            pass
        time.sleep(2)
    raise FullExperimentError("new services did not become ready")


def stop_services(root):
    """Signal only processes whose full launch identity still matches this run."""
    import psutil
    import signal
    path = root / "services.json"
    if not path.exists():
        return
    outcomes = {}
    for name, row in read(path)["processes"].items():
        try:
            p = psutil.Process(row["pid"])
            require(p.create_time() == row["create_time"] and p.cmdline() == row["command"],
                    f"owned service identity differs: {name}")
            os.killpg(p.pid, signal.SIGTERM)
            outcomes[name] = {"status": "termination_requested", "pid": p.pid}
        except psutil.NoSuchProcess:
            outcomes[name] = {"status": "already_exited", "pid": row["pid"]}
    write(root / "service_cleanup.json", outcomes)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("run", "register", "prepare", "solve", "score", "status"))
    for name in ("manifest", "archive-root", "config", "library-root", "output", "code-ref", "metadata"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--sampling-seed", type=int, default=137)
    p.add_argument("--per-task", type=int, default=24)
    p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[137, 138])
    p.add_argument("--qwen-gpu", type=int, default=2)
    p.add_argument("--geometry-gpu", type=int, default=3)
    p.add_argument("--detector-gpu", type=int, default=4)
    p.add_argument("--qwen-port", type=int, default=18110)
    p.add_argument("--detector-port", type=int, default=20024)
    return p


def main(argv=None):
    arguments = list(argv) if argv is not None else sys.argv[1:]
    args = parser().parse_args(arguments)
    root = Path(args.output).resolve()
    if args.stage == "register":
        register(args)
        return 0
    if args.stage == "run" and not root.exists():
        register(args)
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "score":
        score(args)
        return 0
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "status":
            print(json.dumps(status(root, verify_run(args)[1]), sort_keys=True))
            return 0
        stop = threading.Event()
        worker = threading.Thread(target=monitor, args=(root, stop), daemon=True)
        worker.start()
        try:
            if args.stage == "run":
                freeze_runtime_identity(args)
                start_services(args)
            if args.stage in {"run", "prepare"}:
                prepare(args)
            if args.stage in {"run", "solve"}:
                result = solve(args)
                require(result["status"] == "completed", "full campaign incomplete")
            if args.stage == "run":
                # Labels are joined in a separate process after the answer seal.
                command = [sys.executable, str(Path(__file__).resolve()), "score", *arguments[1:]]
                subprocess.run(command, check=True)
        finally:
            stop.set()
            worker.join(timeout=10)
            if args.stage == "run":
                stop_services(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
