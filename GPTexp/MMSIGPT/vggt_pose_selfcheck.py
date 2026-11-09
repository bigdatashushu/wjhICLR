"""Uncharged mock checks for dispatch budget, failure stop and immutable collection."""
from __future__ import annotations

import contextlib
from io import StringIO
import tempfile
from pathlib import Path

import numpy as np

from ..traces import digest, file_digest, read, write
from . import vggt_pose_trial as trial
from .prototype.client import ServiceError
from .vggt_pose import pose_note, relative_cameras


def run():
    _, originals = trial.selection()
    extrinsics = np.stack([np.c_[np.eye(3), np.zeros(3)],
                           np.c_[np.eye(3), [-2, 0, 0]]])
    note = pose_note(relative_cameras(extrinsics))
    items = []
    for item in originals:
        item = {**item, "pose_text": note}
        item["payload_sha256"] = {arm: digest(trial.request(item, arm)) for arm in ("direct", "pose")}
        direct, pose = [trial.request(item, arm)["input"][0]["content"] for arm in ("direct", "pose")]
        assert direct[:2] == pose[:2] and direct[-1] == pose[-1]
        assert len(direct) == 3 and len(pose) == 4
        items.append(item)

    class MockClient:
        def __init__(self, api, key):
            pass

        def call(self, body):
            return {"text": "`A`", "tool_call_count": 0}

    class FailedClient(MockClient):
        def call(self, body):
            raise ServiceError("synthetic_transport_failure")

    original_validate = trial.validate
    keys = {f"pool-{i:02d}": f"synthetic-slot-{i}" for i in range(1, 11)}
    keys["extra-11"] = "synthetic-slot-11"
    try:
        with tempfile.TemporaryDirectory(prefix="mmsi-pose-mock-") as temporary:
            root = Path(temporary)
            manifest = {"items": items, "run_identity": "synthetic_not_real_run"}
            trial.validate = lambda directory, **kwargs: manifest
            for case, factory in (("complete", MockClient), ("failure", FailedClient)):
                directory = root / case
                directory.mkdir()
                for name in ("manifest.json", "preflight.json", "selection.json",
                             "gpu/gpu-input.json", "gpu/gpu-complete.json"):
                    write(directory / name, {"synthetic": True}, exclusive=True)
                for item in items:
                    item["pose_path"] = "gpu/poses/" + item["problem"]["sample_id"] + ".json"
                    write(directory / item["pose_path"], {"synthetic": True}, exclusive=True)
                with contextlib.redirect_stdout(StringIO()):
                    execution = trial.execute(directory, keys, client_factory=factory)
                frozen = trial.freeze(directory)
                trial.validate_frozen(directory, manifest)
                assert execution["inflight"] == 0 and execution["peak_inflight"] <= 6
                assert execution["dispatched"] <= 12
                if case == "complete":
                    assert execution["dispatched"] == 12 and execution["status"] == "complete"
                    assert len(frozen["predictions"]) == 12
                    path = trial.paths(directory, items[0], "direct")[1]
                    result = read(path)
                    result["answer"] = "B"
                    write(path, result)
                    try:
                        trial.validate_frozen(directory, manifest)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("tampered_prediction_accepted")
                else:
                    assert execution["status"] == "stopped_service_error"
                    assert execution["dispatched"] <= 6
                    assert all(p["status"] in ("service_error", "not_dispatched")
                               for p in frozen["predictions"])
    finally:
        trial.validate = original_validate
    return {
        "status": "passed", "actual_model_requests": 0, "reference_answers_read": False,
        "credentials_loaded": False, "original_rgb_and_question_equal_across_arms": True,
        "twelve_request_budget_and_six_inflight_verified": True,
        "service_failure_stops_followup_without_retry": True,
        "prediction_tampering_rejected": True, "source_sha256": file_digest(__file__),
    }


if __name__ == "__main__":
    import json
    result = run()
    write(trial.DESTINATION / "selfcheck.json", result, exclusive=True)
    print(json.dumps(result))
