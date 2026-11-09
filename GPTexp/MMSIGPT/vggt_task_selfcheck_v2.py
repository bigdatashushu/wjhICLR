"""Synthetic-only regression of compact context and isolated v2 dispatch."""
from __future__ import annotations

# This fixture module clears inherited auth and disables socket connections.
from . import vggt_task_selfcheck as fixtures

import argparse
import contextlib
from io import StringIO
import json
from pathlib import Path
import tempfile

from ..traces import digest, file_digest, read, write
from . import vggt_pose_trial as base, vggt_task_trial as previous, vggt_task_trial_v2 as trial
from .vggt_task_prompt_v2 import PROMPT_VERSION, pose_note, relevant_rows, task_binding


def run():
    originals = (base.validate, previous.validate, previous.execute.__globals__["validate"])
    keys = {api: f"synthetic-auth-{index}" for index, api in enumerate(base.APIS)}
    keys.update({f"unused-{index}": f"synthetic-auth-{index}" for index in range(6, 11)})
    with tempfile.TemporaryDirectory(prefix="mmsi-task-pose-v2-") as temporary:
        root = Path(temporary)
        parent = root / "parent"
        original = fixtures.synthetic_parent(parent)
        with contextlib.redirect_stdout(StringIO()):
            base.execute(parent, keys, client_factory=fixtures.FakeClient)
            base.freeze(parent)
        directory = root / "adapted"
        manifest = trial.prepare(directory, parent)
        assert manifest["pose_prompt_version"] == PROMPT_VERSION
        assert "GPTexp/MMSIGPT/vggt_task_prompt_v2.py" in manifest["code_identity"]
        bindings = [item["task_binding"] for item in manifest["items"]]
        assert bindings[0]["reference_image"] == 1 and bindings[0]["target_camera_image"] == 2
        assert bindings[1]["reference_image"] == 2 and bindings[1]["target_entity"] == "the blue stool"
        assert bindings[2]["target_entity"] == "the reading area"
        assert bindings[3]["reference_origin"] == "Person seated on the yellow chair"
        assert bindings[3]["target_entity"] == "the round clock"
        assert bindings[4]["reference_origin"] == "the lounge"
        assert "bookcase is in front" in bindings[4]["reference_forward"]
        assert "piano is on the east" in bindings[5]["reference_forward"]
        assert [b["navigation_requested"] for b in bindings] == [False, False, True, False, False, False]
        for source, item in zip(original["items"], manifest["items"]):
            evidence = read(directory / item["pose_path"])["camera_evidence"]
            assert len(relevant_rows(evidence, item["task_binding"])) == 1
            assert len(item["pose_text"].split()) < 150
            assert item["pose_text"].count("_center_in_camera") == 1
            direct, pose = [trial.request(item, arm)["input"][0]["content"] for arm in ("direct", "pose")]
            assert len(direct) == 3 and len(pose) == 4
            assert direct[:2] == pose[:2] and direct[-1] == pose[-1]
            assert trial.request(item, "direct") == base.request(source, "direct")
            minimal = {k: item[k] for k in ("problem", "images", "api_id")}
            assert trial.request(minimal, "direct") == trial.request(item, "direct")
            assert digest(trial.request(item, "pose")) != source["payload_sha256"]["pose"]
            other_id = {**item["problem"], "sample_id": "unseen_sample", "options": ["up", "down"]}
            assert pose_note(other_id, evidence) == item["pose_text"]
        assert "_rotation_rows" not in manifest["items"][0]["pose_text"]
        assert "_forward_in_camera" not in manifest["items"][0]["pose_text"]
        assert "camera1_center_in_camera2" in manifest["items"][1]["pose_text"]
        assert "camera2_center_in_camera1" not in manifest["items"][1]["pose_text"]
        assert "forward = viewing direction" in manifest["items"][2]["pose_text"]
        assert "do not automatically define body or room axes" in manifest["items"][3]["pose_text"]

        item = manifest["items"][2]
        evidence = read(directory / item["pose_path"])["camera_evidence"]
        positional = {
            **item["problem"],
            "question": "From the position in the first image, in which direction "
                        "is the reading area relative to you?",
        }
        binding = task_binding(positional, 2)
        note = pose_note(positional, evidence)
        assert binding["target_entity"] == "the reading area" and not binding["navigation_requested"]
        assert "requested action" not in note and "_forward_in_camera" not in note
        probe = manifest["items"][1]["problem"]
        assert task_binding({**probe, "question": fixtures.QUESTIONS[1].replace("last", "first")}, 2)["reference_image"] == 1
        assert task_binding({**probe, "question": "Where is the stool relative to me?"}, 2)["reference_image"] is None
        fixtures.rejects(lambda: trial.request({
            **manifest["items"][0], "problem": {**manifest["items"][0]["problem"], "answer": "A"},
        }, "direct"))

        path = directory / manifest["items"][0]["pose_path"]
        saved = path.read_bytes()
        pose_value = read(path)
        pose_value["camera_evidence"]["relative_cameras"][0]["camera_center_xyz"][0] += 1
        write(path, pose_value)
        fixtures.rejects(lambda: trial.validate(directory, requests=True))
        path.write_bytes(saved)
        with contextlib.redirect_stdout(StringIO()):
            execution = trial.execute(directory, keys, client_factory=fixtures.FakeClient)
        frozen = trial.freeze(directory)
        trial.validate_frozen(directory, manifest)
        assert execution["status"] == "complete" and execution["dispatched"] == 12
        assert execution["inflight"] == 0 and execution["peak_inflight"] <= 6
        assert len(frozen["predictions"]) == 12
        fixtures.rejects(lambda: trial.execute(directory, keys, client_factory=fixtures.FakeClient))
        result_path = base.paths(directory, manifest["items"][0], "direct")[1]
        result = read(result_path)
        result["answer"] = "B"
        write(result_path, result)
        fixtures.rejects(lambda: trial.validate_frozen(directory, manifest))

        failed = root / "failed"
        trial.prepare(failed, parent)
        with contextlib.redirect_stdout(StringIO()):
            execution = trial.execute(failed, keys, client_factory=fixtures.FailedClient)
        frozen = trial.freeze(failed)
        assert execution["status"] == "stopped_service_error" and execution["dispatched"] <= 6
        assert execution["inflight"] == 0
        assert all(p["status"] in ("not_dispatched", "service_error") for p in frozen["predictions"])
        fixtures.rejects(lambda: trial.score(failed))
    assert originals == (base.validate, previous.validate, previous.execute.__globals__["validate"])
    return {
        "status": "passed", "pose_prompt_version": PROMPT_VERSION,
        "actual_model_requests": 0, "reference_answers_read": False, "credentials_loaded": False,
        "real_dataset_loaded": False, "network_disabled": True,
        "six_task_bindings_verified": True, "navigation_conditional_on_question": True,
        "one_relevant_transform_and_camera_center_only_verified": True,
        "compact_prompt_and_body_frame_wording_verified": True,
        "direct_payload_equal_to_parent": True, "routing_independent_of_ids_and_options": True,
        "fake_execute_freeze_and_failure_stop_verified": True,
        "pose_and_prediction_tampering_rejected": True,
        "previous_entry_globals_unchanged": True, "selfcheck_sha256": file_digest(__file__),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run()
    if args.output:
        write(args.output, result, exclusive=True)
    print(json.dumps(result))
