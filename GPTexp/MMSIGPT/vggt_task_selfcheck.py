"""Synthetic-only, network-disabled checks of task-bound camera evidence."""
from __future__ import annotations

import os
import socket

# Run in a dedicated process. Real auth and endpoints are never inherited by tests.
_allowed = {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "PYTHONPATH"}
_environment = {k: v for k, v in os.environ.items() if k in _allowed}
os.environ.clear()
os.environ.update(_environment, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def _no_network(*args, **kwargs):
    raise AssertionError("network_disabled_in_synthetic_selfcheck")


socket.socket.connect = _no_network
socket.socket.connect_ex = _no_network
socket.create_connection = _no_network

import argparse
import contextlib
from io import StringIO
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from ..traces import digest, file_digest, read, write
from . import vggt_pose_trial as base, vggt_task_trial as trial
from .prototype.client import ServiceError
from .prototype.inputs import load_images, original_input_sha256
from .vggt_pose import pose_note as old_note, relative_cameras
from .vggt_task_prompt import TASKS, pose_note, task_binding

QUESTIONS = (
    "Suppose I am taking the first picture. From which direction relative to me "
    "is the camera positioned when taking the second picture?",
    "The images are continuous. At the moment of the last image, in which "
    "direction is the blue stool relative to you?",
    "From the position in the first image, I want to reach the reading area "
    "shown in the second image. What should I do?",
    "When you sit on the yellow chair facing the direction of the upper left "
    "area in Figure 2, where is the round clock located relative to you?",
    "The bookcase is in front of the lounge. In which direction is the "
    "lamp relative to the lounge?",
    "The piano is on the east side of the hall. In which direction from "
    "the entrance is the sofa located?",
)


class FakeClient:
    calls = 0

    def __init__(self, api, key):
        pass

    def call(self, body):
        type(self).calls += 1
        return {"text": "`A`", "tool_call_count": 0}


class FailedClient(FakeClient):
    def call(self, body):
        raise ServiceError("synthetic_transport_failure")


def rejects(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("invalid_input_accepted")


def synthetic_parent(directory):
    directory.mkdir()
    paths = []
    for index, color in enumerate(((20, 50, 80), (90, 130, 170))):
        path = directory / f"original-{index}.png"
        Image.new("RGB", (16, 12), color).save(path)
        paths.append(str(path))
    import hashlib
    hashes = [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths]
    evidence = relative_cameras(np.stack([
        np.c_[np.eye(3), np.zeros(3)], np.c_[np.eye(3), [-2, 0, 0]]]))
    items, selected, files = [], [], {}
    for index, (task, question, sample, api) in enumerate(zip(
            TASKS, QUESTIONS, base.SAMPLES, base.APIS)):
        problem = {
            "benchmark": "mmsi", "source_split": "test", "sample_id": sample,
            "task_type": task, "question": question, "options": ["left", "right"],
            "image_paths": paths,
        }
        problem["input_sha256"] = original_input_sha256(problem, hashes)
        item = {
            "index": index, "source_index": index, "problem": problem,
            "images": [r for r, _ in load_images(problem)], "api_id": api,
            "arm_order": ["direct", "pose"] if index % 2 == 0 else ["pose", "direct"],
        }
        selected.append(dict(item))
        item["pose_path"] = "gpu/poses/" + sample + ".json"
        pose = {
            "sample_id": sample, "input_sha256": problem["input_sha256"],
            "original_image_sha256": hashes, "weight_sha256": base.WEIGHT_SHA,
            "joint_forward": True, "reference_answer_visible": False,
            "depth_head_executed": False, "point_head_executed": False,
            "track_head_executed": False, "camera_evidence": evidence,
        }
        write(directory / item["pose_path"], pose, exclusive=True)
        item["pose_sha256"] = file_digest(directory / item["pose_path"])
        files[item["pose_path"][4:]] = item["pose_sha256"]
        item["pose_text"] = old_note(evidence)
        item["payload_sha256"] = {arm: digest(base.request(item, arm)) for arm in ("direct", "pose")}
        items.append(item)
    config = {"weight_sha256": base.WEIGHT_SHA, "vggt_source_identity": {"synthetic": "no_gpu_run"}}
    write(directory / "gpu/gpu-input.json", config, exclusive=True)
    write(directory / "gpu/gpu-complete.json", {
        "status": "complete", "joint_forwards": 6,
        "config_sha256": file_digest(directory / "gpu/gpu-input.json"), "files": files,
        "reference_answer_visible": False, "depth_enabled": False,
    }, exclusive=True)
    write(directory / "selection.json", {"items": selected}, exclusive=True)
    manifest = {
        "schema": "mmsi-joint-camera-pair/1", "items": items,
        "code_identity": base.code_identity(), "environment": base.environment(),
        "api_ids": list(base.APIS), "questions": 6, "max_requests": 12, "max_inflight": 6,
        "inference_by_api": {api: base.settings(api) for api in base.APIS},
        "reference_answer_visible": False, "automatic_retry": False, "tools_enabled": False,
        "depth_images_sent": False, "weight_sha256": base.WEIGHT_SHA,
        "prompt_version": base.PROMPT_VERSION, "parser_version": base.PARSER_VERSION,
        "selection_sha256": file_digest(directory / "selection.json"),
        "gpu_input_sha256": file_digest(directory / "gpu/gpu-input.json"),
        "gpu_complete_sha256": file_digest(directory / "gpu/gpu-complete.json"),
        "vggt_source_identity": config["vggt_source_identity"], "sample_role": "synthetic_only",
    }
    manifest["run_identity"] = digest(manifest)
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "preflight.json", {
        "run_identity": manifest["run_identity"], "model_requests": 0, "reference_answers_read": False,
    }, exclusive=True)
    return manifest


def run():
    globals_before = (base.validate, base.request, base.execute.__globals__["validate"])
    keys = {api: f"synthetic-auth-{index}" for index, api in enumerate(base.APIS)}
    keys.update({f"unused-{index}": f"synthetic-auth-{index}" for index in range(6, 11)})
    with tempfile.TemporaryDirectory(prefix="mmsi-task-pose-") as temporary:
        root = Path(temporary)
        parent = root / "parent"
        original = synthetic_parent(parent)
        with contextlib.redirect_stdout(StringIO()):
            base.execute(parent, keys, client_factory=FakeClient)
            base.freeze(parent)
        base.validate_frozen(parent, original)
        directory = root / "adapted"
        manifest = trial.prepare(directory, parent)
        binding = [item["task_binding"] for item in manifest["items"]]
        assert binding[0]["reference_image"] == 1 and binding[0]["target_camera_image"] == 2
        assert binding[1]["reference_image"] == 2 and binding[1]["target_entity"] == "the blue stool"
        assert binding[2]["target_entity"] == "the reading area"
        assert binding[3]["reference_origin"] == "Person seated on the yellow chair"
        assert binding[3]["target_entity"] == "the round clock"
        assert "upper left" in binding[3]["reference_forward"]
        assert binding[4]["reference_origin"] == "the lounge"
        assert "bookcase is in front" in binding[4]["reference_forward"]
        assert binding[5]["target_entity"] == "the sofa"
        assert "piano is on the east" in binding[5]["reference_forward"]
        for source, item in zip(original["items"], manifest["items"]):
            direct, pose = [trial.request(item, arm)["input"][0]["content"] for arm in ("direct", "pose")]
            assert direct[:2] == pose[:2] and direct[-1] == pose[-1]
            assert len(direct) == 3 and len(pose) == 4
            assert digest(trial.request(item, "direct")) == source["payload_sha256"]["direct"]
            direct_only = {k: item[k] for k in ("problem", "images", "api_id")}
            assert trial.request(direct_only, "direct") == trial.request(item, "direct")
            assert "photographer position" in item["pose_text"]
            assert "viewing direction" in item["pose_text"]
            changed_id = {**item["problem"], "sample_id": "unseen_sample", "options": ["up", "down"]}
            assert pose_note(changed_id, read(directory / item["pose_path"])["camera_evidence"]) == item["pose_text"]
        assert manifest["items"][1]["pose_text"].index("camera1_center_in_camera2") < (
            manifest["items"][1]["pose_text"].index("camera2_center_in_camera1"))
        probe = manifest["items"][1]["problem"]
        assert task_binding({**probe, "question": QUESTIONS[1].replace("last", "first")}, 2)["reference_image"] == 1
        assert task_binding({**probe, "question": QUESTIONS[1].replace("last", "Last")}, 4)["reference_image"] == 4
        assert task_binding({**probe, "question": "Where is the stool relative to me?"}, 2)["reference_image"] is None
        rejects(lambda: task_binding({**probe, "task_type": "unknown"}, 2))
        rejects(lambda: trial.request({**manifest["items"][0], "problem": {**probe, "answer": "A"}}, "direct"))
        path = directory / manifest["items"][0]["pose_path"]
        saved = path.read_bytes()
        pose_value = read(path)
        pose_value["camera_evidence"]["relative_cameras"][0]["camera_center_xyz"][0] += 1
        write(path, pose_value)
        rejects(lambda: trial.validate(directory, requests=True))
        path.write_bytes(saved)
        trial.validate(directory, requests=True)
        with contextlib.redirect_stdout(StringIO()):
            execution = trial.execute(directory, keys, client_factory=FakeClient)
        frozen = trial.freeze(directory)
        trial.validate_frozen(directory, manifest)
        assert execution["status"] == "complete" and execution["dispatched"] == 12
        assert execution["inflight"] == 0 and execution["peak_inflight"] <= 6
        assert len(frozen["predictions"]) == 12
        rejects(lambda: trial.execute(directory, keys, client_factory=FakeClient))
        result_path = base.paths(directory, manifest["items"][0], "direct")[1]
        result = read(result_path)
        result["answer"] = "B"
        write(result_path, result)
        rejects(lambda: trial.validate_frozen(directory, manifest))
        failed = root / "failed"
        trial.prepare(failed, parent)
        with contextlib.redirect_stdout(StringIO()):
            execution = trial.execute(failed, keys, client_factory=FailedClient)
        frozen = trial.freeze(failed)
        assert execution["status"] == "stopped_service_error"
        assert execution["dispatched"] <= 6 and execution["inflight"] == 0
        assert all(p["status"] in ("not_dispatched", "service_error") for p in frozen["predictions"])
        # Scoring must reject incomplete predictions before it can read any reference file.
        rejects(lambda: trial.score(failed))
    assert globals_before == (base.validate, base.request, base.execute.__globals__["validate"])
    return {
        "status": "passed", "actual_model_requests": 0, "reference_answers_read": False,
        "credentials_loaded": False, "real_dataset_loaded": False, "network_disabled": True,
        "six_task_bindings_verified": True, "first_last_and_unknown_reference_verified": True,
        "routing_independent_of_ids_and_options": True, "direct_payload_equal_to_parent": True,
        "twelve_request_fake_client_and_freeze_verified": True,
        "pose_and_prediction_tampering_rejected": True, "no_restart_or_automatic_retry": True,
        "service_failure_stops_followup": True, "incomplete_score_blocks_reference_access": True,
        "parent_module_globals_unchanged": True, "selfcheck_sha256": file_digest(__file__),
    }


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run()
    if args.output:
        write(args.output, result, exclusive=True)
    print(json.dumps(result))
