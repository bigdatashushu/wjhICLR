"""Synthetic-only, network-disabled check of 30-pair isolation and score gating."""
from __future__ import annotations

# Clears inherited service variables and disables every network connection.
from . import vggt_task_selfcheck as fixtures

from collections import Counter
import contextlib
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile
import threading

import numpy as np
from PIL import Image

from ..traces import credential_free, digest, file_digest, read, write
from . import vggt_pose30_trial as trial, vggt_pose30_host as host
from .depth_comparison import runner as shared
from .prototype.client import ServiceError
from .prototype.inputs import load_images, original_input_sha256
from .vggt_pose import relative_cameras


class FakeClient:
    calls = Counter()
    guard = threading.Lock()

    def __init__(self, api, key):
        self.api, self.key = api, key

    def assert_public(self, value):
        credential_free(value, self.key)

    def call(self, body):
        with self.guard:
            self.calls[self.api] += 1
        return {
            "text": "`A`", "tool_call_count": 0, "reasoning_summary": None,
            "reported": {"model": "gpt-5.6-terra", "reasoning_effort": "high",
                         "temperature": 1, "seed": None, "max_output_tokens": 8192},
            "usage": {"input_tokens": 10, "output_tokens": 1}, "elapsed_seconds": 0.01,
        }


class FailedClient(FakeClient):
    def call(self, body):
        raise ServiceError("synthetic_transport_failure")


def inputs(directory):
    directory.mkdir()
    originals = directory / "originals"
    originals.mkdir()
    paths = []
    # One JPEG has a misleading PNG extension, matching the observed source.
    for index, fmt in enumerate(("PNG", "JPEG")):
        path = originals / f"original-{index}.png"
        Image.new("RGB", (16, 12), (20 + index * 40, 50, 80)).save(path, format=fmt)
        paths.append(str(path))
    hashes = [file_digest(path) for path in paths]
    items = []
    for index in range(30):
        rank, task = divmod(index, 6)
        problem = {
            "benchmark": "mmsi", "source_split": "test", "sample_id": f"synthetic_{index:03d}",
            "task_type": list(fixtures.TASKS)[task], "question": fixtures.QUESTIONS[task],
            "options": ["left", "right"], "image_paths": paths,
        }
        problem["input_sha256"] = original_input_sha256(problem, hashes)
        items.append({
            "index": index, "problem": problem, "images": [r for r, _ in load_images(problem)],
            "api_id": f"pool-{(rank + task) % 6 + 1:02d}",
            "arm_order": ["direct", "pose"] if (rank + task) % 2 == 0 else ["pose", "direct"],
        })
    selected = {
        "schema": "mmsi-vggt30-public-selection/1", "questions": 30, "per_type": 5,
        "seed": 137, "max_requests": 60, "api_ids": list(fixtures.base.APIS), "items": items,
        "reference_answer_visible": False, "selection_reads_labels": False,
        "source_manifest_sha256": trial.SOURCE_SHA, "excluded_sample_ids": [],
        "sample_role": "synthetic_only",
    }
    selected["selection_identity"] = digest(selected)
    write(directory / "selection.json", selected, exclusive=True)
    gpu = directory / "gpu"
    for name in ("code", "poses", "images"):
        (gpu / name).mkdir(parents=True)
    for name in ("vggt_pose.py", "vggt_pose_worker.py", "vggt_pose30_worker.py"):
        shutil.copyfile(trial.base.PACKAGE / name, gpu / "code" / name)
    evidence = relative_cameras(np.stack([
        np.c_[np.eye(3), np.zeros(3)], np.c_[np.eye(3), [-2, 0, 0]]]))
    config_items, files = [], {}
    environment = {"gpu": "synthetic_no_gpu"}
    for item in items:
        sample = item["problem"]["sample_id"]
        receipts = []
        for index, (receipt, raw) in enumerate(load_images(item["problem"], item["images"])):
            suffix = ".png" if receipt["mime_type"] == "image/png" else ".jpg"
            path = f"images/{sample}-v{index+1}{suffix}"
            (gpu / path).write_bytes(raw)
            receipts.append({**receipt, "path": path})
        config_items.append({"sample_id": sample,
                             "input_sha256": item["problem"]["input_sha256"], "images": receipts})
        pose = {
            "schema": "mmsi-joint-vggt-camera/1", "sample_id": sample,
            "input_sha256": item["problem"]["input_sha256"], "original_image_sha256": hashes,
            "weight_sha256": trial.base.WEIGHT_SHA, "environment": environment,
            "camera_evidence": evidence, "joint_forward": True, "model_requests": 0,
            "reference_answer_visible": False, "depth_head_executed": False,
            "point_head_executed": False, "track_head_executed": False,
        }
        path = "poses/" + sample + ".json"
        write(gpu / path, pose, exclusive=True)
        files[path] = file_digest(gpu / path)
    config = {
        "schema": "mmsi-pose30-gpu-input/1", "selection_identity": selected["selection_identity"],
        "weight_sha256": trial.base.WEIGHT_SHA, "vggt_source_identity": {"synthetic": "no_gpu"},
        "worker_identity": {p.name: file_digest(p) for p in (gpu / "code").iterdir()},
        "environment": environment, "items": config_items, "model_requests": 0,
        "reference_answer_visible": False, "depth_enabled": False,
        "point_enabled": False, "track_enabled": False,
    }
    write(gpu / "gpu-input.json", config, exclusive=True)
    write(gpu / "gpu-complete.json", {
        "status": "complete", "joint_forwards": 30, "files": files, "model_requests": 0,
        "config_sha256": file_digest(gpu / "gpu-input.json"),
        "reference_answer_visible": False, "depth_enabled": False,
    }, exclusive=True)
    for name in ("gpu-started.json", "launch.json"):
        write(gpu / name, {"synthetic": True}, exclusive=True)


def run():
    original_globals = (shared.validate, shared.request, shared.execute.__globals__["validate"])
    source_binding = trial.SOURCE, trial.SOURCE_SHA
    keys = {api: f"synthetic-auth-{index}" for index, api in enumerate(fixtures.base.APIS)}
    try:
        with tempfile.TemporaryDirectory(prefix="mmsi-pose30-synthetic-") as temporary:
            root = Path(temporary)
            trial.SOURCE = root / "synthetic-source.json"
            write(trial.SOURCE, {"reference_answer_visible": False, "synthetic": True})
            trial.SOURCE_SHA = file_digest(trial.SOURCE)
            directory = root / "complete"
            inputs(directory)
            manifest = trial.prepare(directory)
            assert manifest["max_requests"] == 60 and manifest["max_inflight"] == 6
            for item in manifest["items"]:
                direct, pose = [trial.request(directory, item, arm) for arm in ("direct", "pose")]
                d, p = direct["input"][0]["content"], pose["input"][0]["content"]
                assert len(d) == 3 and len(p) == 4 and d[:2] == p[:2] and d[-1] == p[-1]
                minimal = {k: item[k] for k in ("problem", "images", "api_id")}
                assert trial.request(directory, minimal, "direct") == direct
            fixtures.rejects(lambda: host.score_internal(directory, root / "never-read", "absent"))
            item = manifest["items"][0]
            path = directory / item["pose_path"]
            saved = path.read_bytes()
            value = read(path)
            value["camera_evidence"]["relative_cameras"][0]["camera_center_xyz"][0] += 1
            write(path, value)
            fixtures.rejects(lambda: trial.validate(directory, verify_requests=True))
            path.write_bytes(saved)
            with contextlib.redirect_stdout(StringIO()):
                result = trial.execute(directory, keys, client_factory=FakeClient)
                frozen = trial.freeze(directory)
            assert result["status"] == "complete" and result["dispatched"] == 60
            assert result["peak_inflight"] <= 6 and result["inflight"] == 0
            assert len(frozen["predictions"]) == 60
            assert FakeClient.calls == {api: 10 for api in keys}
            assert all(p["status"] == "answered" for p in frozen["predictions"])
            assert all("depth" not in name for name in frozen["files"])
            fixtures.rejects(lambda: trial.execute(directory, keys, client_factory=FakeClient))
            refs = root / "synthetic-references.json"
            write(refs, {"references": [{
                "benchmark": "mmsi", "sample_id": item["problem"]["sample_id"],
                "reference_answer": "A" if item["index"] % 2 else "B",
            } for item in manifest["items"]]})
            scoring = host.score_internal(directory, refs, file_digest(refs))
            assert scoring["overall"]["direct_correct"] == 15
            assert scoring["overall"]["pose_correct"] == 15
            assert all(v["questions"] == 5 for v in scoring["by_task"].values())
            with contextlib.redirect_stdout(StringIO()):
                host.report(directory)
            assert (directory / "实验结果.md").exists()
            failed = root / "failed"
            inputs(failed)
            trial.prepare(failed)
            with contextlib.redirect_stdout(StringIO()):
                stopped = trial.execute(failed, keys, client_factory=FailedClient)
                trial.freeze(failed)
            assert stopped["status"] == "stopped_service_error" and stopped["dispatched"] < 60
            fixtures.rejects(lambda: host.score_internal(failed, root / "never-read", "absent"))
    finally:
        trial.SOURCE, trial.SOURCE_SHA = source_binding
    assert original_globals == (
        shared.validate, shared.request, shared.execute.__globals__["validate"])
    return {
        "status": "passed", "synthetic_only": True, "network_disabled": True,
        "real_model_requests": 0, "real_gpu_requests": 0, "real_references_read": False,
        "paired_60_budget_verified": True, "service_failure_stops_and_blocks_score": True,
        "shared_runner_globals_unchanged": True,
    }


if __name__ == "__main__":
    print(json.dumps(run()), flush=True)
