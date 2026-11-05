"""Offline checks for the experiment boundary; no data, labels or live requests."""
from copy import deepcopy
import importlib.util
from pathlib import Path

from skill3d.evaluation import blink_experiment as exp

spec = importlib.util.spec_from_file_location("blink_paired", Path(__file__).with_name("paired.py"))
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


def fixture():
    problems = [{"sample_id": f"test-{i}", "source_split": "test",
                 "input_sha256": f"input-{i}"} for i in range(93)]
    problems += [{"sample_id": "train-only", "source_split": "train"}]
    inspection = {
        "problems": problems, "seed": {"text": "frozen_skill"},
        "content_sha256": "preflight",
        "cache_identities": {p["sample_id"]: {
            "images": ["original-1", "original-2"], "artifact_sha256": "geometry",
        } for p in problems},
    }
    profile = {"model": {"model": "qwen3.8-flash"},
               "solver": {"seed": 137}, "inputs": {
                   "host_answers": "/never/open/labels", "prepared_root": "/public/cache"}}
    return profile, inspection


def test_full_paired_denominators_and_public_worker_boundary():
    profile, inspection = fixture()
    jobs = paired.jobs_for(exp, Path("/run"), profile, inspection)
    assert len(jobs) == 186
    assert len({j["operation_id"] for j in jobs}) == 186
    assert all(j["identity"]["source_split"] == "test" for j in jobs)
    assert len([j for j in jobs if j["identity"]["arm"] == "direct"]) == 93
    for job in jobs:
        worker = job["worker"]
        assert "host_answers" not in worker and "reference_answer" not in worker
        assert worker["max_requests"] == 465 and worker["request_concurrency"] == 4
        if job["identity"]["arm"] == "direct":
            assert "prepared_root" not in worker and "artifact_sha256" not in worker
            assert worker["skill_text"] == "" and job["identity"]["allowed_tools"] == []
        else:
            assert worker["skill_text"] == "frozen_skill"
    for a, b in zip(jobs[::2], jobs[1::2]):
        assert a["identity"]["arm"] != b["identity"]["arm"]
        for key in ("sample_id", "input_sha256", "images", "model", "solver", "seed"):
            assert a["identity"][key] == b["identity"][key]


def test_denominator_change_rejected():
    import pytest
    profile, inspection = fixture()
    changed = deepcopy(inspection)
    changed["problems"].pop(0)
    with pytest.raises(ValueError, match="denominator"):
        paired.jobs_for(exp, Path("/run"), profile, changed)


def test_saved_receipt_rejects_count_and_record_tampering(tmp_path):
    import pytest
    profile, inspection = fixture()
    job = next(j for j in paired.jobs_for(exp, tmp_path, profile, inspection)
               if j["identity"]["arm"] == "direct")
    directory = tmp_path / "requests" / job["operation_id"]
    exp.write_once(directory / "001.intent.json", {
        "parameters": {"seed": 137, "temperature": 0}, "payload_sha256": "payload"})
    exp.write_once(directory / "001.response.json", {
        "status": "received", "response_identity": {
            "model": "qwen3.8-flash", "model_reported": True}})
    receipt = {
        "identity": job["identity"], "status": "complete",
        "request_records": paired.request_records(exp, tmp_path, job["operation_id"]),
        "result": {
            "sample_id": job["identity"]["sample_id"], "source_split": "test",
            "input_sha256": job["identity"]["input_sha256"],
            "request_count": 1, "attempted_tool_calls": 0, "succeeded_tool_calls": 0,
            "rounds": [{"phase": "direct"}],
        },
    }
    paired.validate_receipt(exp, tmp_path, job, receipt)
    changed = deepcopy(receipt)
    changed["result"]["request_count"] = 2
    with pytest.raises(ValueError, match="reported_request_count"):
        paired.validate_receipt(exp, tmp_path, job, changed)
    changed = deepcopy(receipt)
    changed["result"]["attempted_tool_calls"] = 1
    with pytest.raises(ValueError, match="direct_used_tools"):
        paired.validate_receipt(exp, tmp_path, job, changed)
    (directory / "001.response.json").write_text("{}")
    with pytest.raises(ValueError, match="request_records_changed"):
        paired.validate_receipt(exp, tmp_path, job, receipt)
    changed = deepcopy(inspection)
    changed["problems"][0] = changed["problems"][1]
    with pytest.raises(ValueError, match="denominator"):
        paired.jobs_for(exp, Path("/run"), profile, changed)
