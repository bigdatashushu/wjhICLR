"""Network-disabled synthetic continuation, no repeated dispatched sample/arm."""
from __future__ import annotations

from . import vggt_pose30_selfcheck as fixtures

from collections import Counter
import contextlib
import copy
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile

from ..traces import digest, file_digest, read, write
from . import vggt_pose30_trial as parent, vggt_pose30_recovery as recovery
from .prototype.inputs import original_input_sha256


def failed_parent(directory):
    fixtures.inputs(directory)
    manifest = parent.prepare(directory)
    write(directory / "execution.started.json", {
        "run_identity": manifest["run_identity"], "max_requests": 60, "max_inflight": 6,
        "reference_answer_visible": False, "automatic_retry": False,
    }, exclusive=True)
    for item in manifest["items"][:6]:
        arm = item["arm_order"][0]
        dispatch_path, result_path = parent.paths(directory, item, arm)
        dispatch = {
            "index": item["index"], "sample_id": item["problem"]["sample_id"],
            "input_sha256": item["problem"]["input_sha256"], "arm": arm, "api_id": item["api_id"],
            "run_identity": manifest["run_identity"], "payload_sha256": item["payload_sha256"][arm],
            "requested": parent.base.settings(item["api_id"]),
            "reference_answer_visible": False, "automatic_retry": False,
        }
        write(dispatch_path, dispatch, exclusive=True)
        result = {k: dispatch[k] for k in (
            "index", "sample_id", "input_sha256", "arm", "api_id", "run_identity",
            "reference_answer_visible", "automatic_retry")}
        result["dispatch_sha256"] = file_digest(dispatch_path)
        if item["index"] == 3:
            result.update(status="service_error", answer=None, parse_status="not_parsed",
                          error_code="http_error", http_status=503)
        else:
            result.update(status="answered", answer="A", parse_status="legal", response={
                "text": "`A`", "tool_call_count": 0, "reasoning_summary": None,
                "reported": {"model": "gpt-5.6-terra", "reasoning_effort": "high",
                             "temperature": 1, "seed": None, "max_output_tokens": None},
                "usage": {"input_tokens": 10, "output_tokens": 1}, "elapsed_seconds": 0.01,
            })
        write(result_path, result, exclusive=True)
    write(directory / "execution.json", {
        "run_identity": manifest["run_identity"], "dispatched": 6, "inflight": 0,
        "peak_inflight": 6, "status": "stopped_service_error",
        "reference_answer_visible": False, "automatic_retry": False,
    }, exclusive=True)
    parent.freeze(directory)
    return manifest


def replacement(directory, manifest):
    directory.mkdir()
    original = manifest["items"][3]
    row = {k: copy.deepcopy(original[k]) for k in ("problem", "images")}
    row["problem"]["sample_id"] = "synthetic_replacement"
    hashes = [r["sha256"] for r in row["images"]]
    row["problem"]["input_sha256"] = original_input_sha256(row["problem"], hashes)
    value = {
        "failed_sample_id": original["problem"]["sample_id"], "failed_api_id": original["api_id"],
        "replacement": row, "replacement_reads_labels": False, "reference_answer_visible": False,
        "parent_run_identity": manifest["run_identity"],
    }
    value["replacement_identity"] = digest(value)
    write(directory / "replacement.json", value, exclusive=True)
    gpu = directory / "gpu"
    for name in ("poses", "code", "images"):
        (gpu / name).mkdir(parents=True)
    receipts = []
    for index, (actual, receipt) in enumerate(zip(row["problem"]["image_paths"], row["images"])):
        suffix = ".png" if receipt["mime_type"] == "image/png" else ".jpg"
        path = f"images/synthetic_replacement-v{index+1}{suffix}"
        shutil.copyfile(actual, gpu / path)
        receipts.append({**receipt, "path": path})
    for name in ("vggt_pose.py", "vggt_pose_worker.py", "vggt_pose30_replacement_worker.py"):
        shutil.copyfile(parent.base.PACKAGE / name, gpu / "code" / name)
    old = read(parent.DESTINATION / "gpu/gpu-input.json")
    config = {
        **{k: old[k] for k in ("weight_sha256", "vggt_source_identity", "environment")},
        "worker_identity": {p.name: file_digest(p) for p in (gpu / "code").iterdir()},
        "items": [{"sample_id": row["problem"]["sample_id"],
                   "input_sha256": row["problem"]["input_sha256"], "images": receipts}],
    }
    write(gpu / "gpu-input.json", config, exclusive=True)
    pose = read(parent.DESTINATION / original["pose_path"])
    pose.update(sample_id=row["problem"]["sample_id"], input_sha256=row["problem"]["input_sha256"])
    path = "poses/synthetic_replacement.json"
    write(gpu / path, pose, exclusive=True)
    write(gpu / "gpu-complete.json", {
        "status": "complete", "joint_forwards": 1,
        "config_sha256": file_digest(gpu / "gpu-input.json"),
        "files": {path: file_digest(gpu / path)},
    }, exclusive=True)
    for name in ("launch.json", "gpu-started.json"):
        write(gpu / name, {"synthetic": True}, exclusive=True)


def run():
    source_binding = parent.SOURCE, parent.SOURCE_SHA, parent.DESTINATION
    fixtures.FakeClient.calls = Counter()
    try:
        with tempfile.TemporaryDirectory(prefix="mmsi-pose30-recovery-synthetic-") as temporary:
            root = Path(temporary)
            parent.SOURCE = root / "synthetic-source.json"
            write(parent.SOURCE, {"synthetic": True})
            parent.SOURCE_SHA = file_digest(parent.SOURCE)
            parent.DESTINATION = root / "parent"
            original = failed_parent(parent.DESTINATION)
            parent_bytes = (parent.DESTINATION / "frozen.json").read_bytes()
            directory = root / "continuation"
            replacement(directory, original)
            manifest = recovery.prepare(directory)
            assert len(manifest["reused_predictions"]) == 5
            assert sum(len(i["arm_order"]) for i in manifest["pending_items"]) == 55
            assert Counter(i["api_id"] for i in manifest["items"]) == {
                api: 5 for api in ("pool-01", "pool-02", "pool-03", "pool-07", "pool-05", "pool-06")}
            parent_attempted = {(p["sample_id"], p["arm"]) for p in
                                read(parent.DESTINATION / "frozen.json")["predictions"]
                                if p["status"] != "not_dispatched"}
            new_attempts = {(i["problem"]["sample_id"], arm) for i in manifest["pending_items"]
                            for arm in i["arm_order"]}
            assert not parent_attempted & new_attempts
            keys = {api: f"synthetic-{api}" for api in manifest["api_ids"]}
            fixtures.fixtures.rejects(lambda: recovery.host_engine().score_internal(
                directory, root / "never-read", "absent"))
            with contextlib.redirect_stdout(StringIO()):
                result = recovery.execute(directory, keys, client_factory=fixtures.FakeClient)
                frozen = recovery.freeze(directory)
            assert result["status"] == "complete" and result["dispatched"] == 55
            assert len(frozen["predictions"]) == 60
            assert all(p["status"] == "answered" for p in frozen["predictions"])
            assert sum(fixtures.FakeClient.calls.values()) == 55
            assert (parent.DESTINATION / "frozen.json").read_bytes() == parent_bytes
            refs = root / "synthetic-references.json"
            write(refs, {"references": [{
                "benchmark": "mmsi", "sample_id": i["problem"]["sample_id"], "reference_answer": "A",
            } for i in manifest["items"]]})
            value = recovery.host_engine().score_internal(directory, refs, file_digest(refs))
            assert value["overall"]["direct_correct"] == value["overall"]["pose_correct"] == 30
            with contextlib.redirect_stdout(StringIO()):
                audit = recovery.report(directory)
            assert audit["cumulative_model_requests"] == 61 and len(audit["rows"]) == 30
            fixtures.fixtures.rejects(lambda: recovery.execute(directory, keys, client_factory=fixtures.FakeClient))
    finally:
        parent.SOURCE, parent.SOURCE_SHA, parent.DESTINATION = source_binding
    return {"status": "passed", "synthetic_only": True, "network_disabled": True,
            "real_model_requests": 0, "real_references_read": False,
            "retained_predictions": 5, "new_requests_verified": 55,
            "dispatched_sample_arm_repeated": 0, "final_pairs_verified": 30}


if __name__ == "__main__":
    print(json.dumps(run()), flush=True)
