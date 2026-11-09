"""Explicit all-API recovery: retain complete frozen pairs and replace incomplete pairs."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from xcodeapi.config import Profile

from ..traces import credential_free, digest, file_digest, read, write
from .depth_comparison import runner
from .depth_comparison.inputs import request
from .depth_comparison.pool import BASE_URL, MODEL, Client, inference
from .depth_comparison.evaluation import summarize
from .prototype.client import ServiceError
from .prototype.direct import operator_lock
from .prototype.evaluation import reference_answers
from .prototype.protocol import LETTERS, parse_answer

NEW_APIS = ["pool-07", "pool-08", "pool-09", "pool-10", "extra-11"]
ALL_APIS = [f"pool-{i:02d}" for i in range(1, 11)] + ["extra-11"]
DEFAULT_PARENT = Path("/Users/bytedance/wangjh/MMSI深度图对比-20261107/run-recovered")
DEFAULT_RUN = DEFAULT_PARENT.parent / "run-all-api"
TERMINAL = {"answered", "invalid_answer"}


def settings(api):
    return {**inference(api), "timeout_seconds": 180}


class BoundedClient(Client):
    def __init__(self, api, key):
        super().__init__(api, key)
        self.inference = settings(api)
        self.profile = Profile("responses", BASE_URL, MODEL, "XCODE_GPT_API_KEY",
                               timeout_seconds=180)


def code_identity():
    return {**runner.code_identity(),
            "MMSIGPT/depth_all_api_recovery.py": file_digest(__file__)}


def parent_evidence(directory):
    manifest = runner.validate(directory)
    frozen = runner.validate_frozen(directory, manifest)
    execution = read(Path(directory) / "execution.json")
    return manifest, frozen, execution


def plan(manifest, frozen, execution, historical_requests):
    predictions = {(p["sample_id"], p["arm"]): p for p in frozen["predictions"]}
    reused, pending = [], []
    for item in manifest["items"]:
        complete = all(predictions[item["problem"]["sample_id"], arm]["status"] in TERMINAL
                       for arm in ("direct", "depth"))
        (reused if complete else pending).append(copy.deepcopy(item))
    if (manifest["questions"] != 18 or len(reused) != 13 or len(pending) != 5
            or execution["status"] != "interrupted" or execution["dispatched"] != 29
            or historical_requests != 12):
        raise ValueError("unexpected_parent_recovery_state")
    for api, item in zip(NEW_APIS, pending):
        item["original_api_id"] = item["api_id"]
        item["api_id"] = api
    return reused, pending, historical_requests + execution["dispatched"] + 2 * len(pending)


def prepare(directory=DEFAULT_RUN, parent=DEFAULT_PARENT):
    directory, parent = Path(directory), Path(parent).resolve()
    manifest, frozen, execution = parent_evidence(parent)
    from .depth_comparison_report import ancestors
    prior = ancestors(parent, manifest)
    historical = sum(row[3]["dispatched"] for row in prior)
    reused, pending, cumulative = plan(manifest, frozen, execution, historical)
    authorization = read(parent.parent / "api-reallocation.json")
    if (authorization["parent_run_identity"] != manifest["run_identity"]
            or authorization["user_instruction"] != "api 全都用上"
            or authorization["planned_new_paired_requests"] != 10
            or authorization["planned_cumulative_paired_requests"] != cumulative
            or not authorization["old_processes_confirmed_exited"] or cumulative > 60):
        raise ValueError("reallocation_authorization_or_budget_changed")
    if not directory.parent.is_dir() or directory.exists():
        raise ValueError("requires_new_directory_in_existing_parent")
    directory.mkdir()
    for item in pending:
        for receipt in item["depth_images"]:
            target = directory / receipt["path"]
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(parent / receipt["path"], target)
    result = {
        "schema": "mmsi-depth-all-api-recovery/1",
        "parent_directory": str(parent), "parent_run_identity": manifest["run_identity"],
        "parent_freeze_identity": frozen["freeze_identity"],
        "authorization_sha256": file_digest(parent.parent / "api-reallocation.json"),
        "reused_items": reused, "new_items": pending,
        "questions": 18, "reused_pairs": 13, "new_pairs": 5,
        "max_requests": 10, "historical_requests": historical,
        "parent_requests": execution["dispatched"],
        "cumulative_paired_requests_limit": cumulative, "original_paired_budget": 60,
        "api_ids": NEW_APIS, "all_api_ids": ALL_APIS, "max_inflight": 5,
        "inference_by_api": {api: settings(api) for api in NEW_APIS},
        "code_identity": code_identity(), "environment": runner.environment_identity(),
        "reference_answer_visible": False, "automatic_retry": False, "tools_enabled": False,
        "prepared_at": time.time(),
    }
    result["run_identity"] = digest(result)
    write(directory / "manifest.json", result, exclusive=True)
    write(directory / "preflight.json", {
        "run_identity": result["run_identity"], "model_requests": 0,
        "reference_answers_read": False, "parent_frozen_verified": True,
        "reused_pairs": 13, "new_pairs": 5, "new_requests": 10,
    }, exclusive=True)
    validate(directory, verify_requests=True)
    return result


def validate(directory, *, verify_requests=False):
    directory = Path(directory)
    result = read(directory / "manifest.json")
    parent, frozen, execution = parent_evidence(result["parent_directory"])
    reused, pending, cumulative = plan(parent, frozen, execution, result["historical_requests"])
    if (result["schema"] != "mmsi-depth-all-api-recovery/1"
            or digest({k: v for k, v in result.items() if k != "run_identity"}) != result["run_identity"]
            or result["parent_run_identity"] != parent["run_identity"]
            or result["parent_freeze_identity"] != frozen["freeze_identity"]
            or result["authorization_sha256"] != file_digest(
                Path(result["parent_directory"]).parent / "api-reallocation.json")
            or result["reused_items"] != reused or result["new_items"] != pending
            or result["code_identity"] != code_identity()
            or result["environment"] != runner.environment_identity()
            or result["api_ids"] != NEW_APIS or result["all_api_ids"] != ALL_APIS
            or result["inference_by_api"] != {api: settings(api) for api in NEW_APIS}
            or result["cumulative_paired_requests_limit"] != cumulative or cumulative > 60
            or result["max_requests"] != 10 or result["max_inflight"] != 5
            or result["questions"] != 18 or result["reused_pairs"] != 13 or result["new_pairs"] != 5
            or result["parent_requests"] != 29 or result["original_paired_budget"] != 60
            or result["reference_answer_visible"] is not False
            or result["automatic_retry"] is not False or result["tools_enabled"] is not False):
        raise ValueError("recovery_identity_changed")
    preflight = read(directory / "preflight.json")
    if (preflight["run_identity"] != result["run_identity"] or preflight["model_requests"] != 0
            or preflight["reference_answers_read"] is not False):
        raise ValueError("preflight_changed")
    if verify_requests:
        for item in pending:
            for arm in item["arm_order"]:
                if digest(request(directory, item, arm)) != item["payload_sha256"][arm]:
                    raise ValueError("original_payload_changed")
    return result


def execute(directory, keys, *, client_factory=BoundedClient):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory, verify_requests=True)
        if any((directory / name).exists() for name in (
                "execution.started.json", "frozen.json", "traces")):
            raise ValueError("no_restart_or_automatic_retry")
        if list(keys) != ALL_APIS or len(set(keys.values())) != 11:
            raise ValueError("all_distinct_credentials_required")
        clients = {api: client_factory(api, keys[api]) for api in NEW_APIS}
        def public(value):
            for key in keys.values():
                credential_free(value, key)
        public(manifest)
        stop, guard = threading.Event(), threading.Lock()
        state = {"dispatched": 0, "inflight": 0, "peak_inflight": 0}
        write(directory / "execution.started.json", {
            "run_identity": manifest["run_identity"], "max_requests": 10, "max_inflight": 5,
            "reference_answer_visible": False, "automatic_retry": False,
            "credential_storage": "process_memory_only", "at": time.time(),
        }, exclusive=True)
        def slot(item):
            api = item["api_id"]
            for arm in item["arm_order"]:
                if stop.is_set():
                    return
                body = request(directory, item, arm)
                public(body)
                if digest(body) != item["payload_sha256"][arm]:
                    stop.set()
                    raise ValueError("payload_changed_before_dispatch")
                dispatch_path, result_path = runner.paths(directory, item, arm)
                dispatch = {
                    "index": item["index"], "sample_id": item["problem"]["sample_id"],
                    "input_sha256": item["problem"]["input_sha256"], "api_id": api, "arm": arm,
                    "run_identity": manifest["run_identity"], "requested": settings(api),
                    "payload_sha256": item["payload_sha256"][arm],
                    "reference_answer_visible": False, "automatic_retry": False, "at": time.time(),
                }
                with guard:
                    if stop.is_set():
                        return
                    if state["dispatched"] >= 10:
                        raise ValueError("request_budget_exceeded")
                    write(dispatch_path, dispatch, exclusive=True)
                    state["dispatched"] += 1
                    state["inflight"] += 1
                    state["peak_inflight"] = max(state["inflight"], state["peak_inflight"])
                print(json.dumps({"api_id": api, "arm": arm, "status": "dispatched",
                                  "requests": state["dispatched"]}), flush=True)
                try:
                    response = clients[api].call(body)
                    public(response)
                    parsed = parse_answer(response["text"], len(item["problem"]["options"]))
                    record = {**parsed, "status": "answered" if parsed["answer"] else "invalid_answer",
                              "response": response}
                except ServiceError as exc:
                    stop.set()
                    record = {"status": "service_error", "answer": None, "parse_status": "not_parsed",
                              "error_code": exc.code, "http_status": exc.http_status}
                except Exception:
                    stop.set()
                    record = {"status": "service_error", "answer": None, "parse_status": "not_parsed",
                              "error_code": "client_outcome_unknown", "http_status": None}
                record.update({**{k: dispatch[k] for k in (
                    "index", "sample_id", "input_sha256", "api_id", "arm", "run_identity",
                    "reference_answer_visible", "automatic_retry")},
                    "dispatch_sha256": file_digest(dispatch_path), "at": time.time()})
                public(record)
                write(result_path, record, exclusive=True)
                with guard:
                    state["inflight"] -= 1
                print(json.dumps({"api_id": api, "arm": arm, "status": record["status"]}), flush=True)
        status = "interrupted"
        try:
            with ThreadPoolExecutor(max_workers=5) as pool:
                list(pool.map(slot, manifest["new_items"]))
            status = "stopped_service_error" if stop.is_set() else "complete"
        finally:
            stop.set()
            write(directory / "execution.json", {
                **state, "status": status, "run_identity": manifest["run_identity"],
                "reference_answer_visible": False, "automatic_retry": False, "at": time.time(),
            }, exclusive=True)
        return read(directory / "execution.json")


def collect(directory, manifest):
    directory = Path(directory)
    _, parent_freeze, _ = parent_evidence(manifest["parent_directory"])
    parent_predictions = {(p["sample_id"], p["arm"]): p for p in parent_freeze["predictions"]}
    execution, started = [read(directory / name) for name in ("execution.json", "execution.started.json")]
    if (execution["run_identity"] != manifest["run_identity"] or execution["inflight"] != 0
            or execution["peak_inflight"] > 5
            or execution["status"] not in ("complete", "interrupted", "stopped_service_error")
            or started["run_identity"] != manifest["run_identity"]
            or started["max_requests"] != 10 or started["max_inflight"] != 5
            or started["reference_answer_visible"] is not False or started["automatic_retry"] is not False
            or execution["reference_answer_visible"] is not False or execution["automatic_retry"] is not False):
        raise ValueError("execution_changed")
    files = {name: file_digest(directory / name) for name in (
        "manifest.json", "preflight.json", "execution.started.json", "execution.json")}
    records, known, dispatched = [], set(), 0
    for item in manifest["reused_items"]:
        for arm in item["arm_order"]:
            records.append({**parent_predictions[item["problem"]["sample_id"], arm],
                            "source": "parent", "source_run_identity": manifest["parent_run_identity"]})
    for item in manifest["new_items"]:
        for receipt in item["depth_images"]:
            files[receipt["path"]] = file_digest(directory / receipt["path"])
            if files[receipt["path"]] != receipt["sha256"]:
                raise ValueError("depth_bytes_changed")
        sent = []
        for arm in item["arm_order"]:
            dispatch_path, result_path = runner.paths(directory, item, arm)
            known.update((dispatch_path.name, result_path.name))
            record = {"index": item["index"], "sample_id": item["problem"]["sample_id"],
                      "task_type": item["problem"]["task_type"],
                      "input_sha256": item["problem"]["input_sha256"], "api_id": item["api_id"],
                      "arm": arm, "status": "not_dispatched", "answer": None, "source": "new"}
            if result_path.exists() and not dispatch_path.exists():
                raise ValueError("result_without_dispatch")
            if dispatch_path.exists():
                dispatch = read(dispatch_path)
                if (any(dispatch[k] != record[k] for k in (
                        "index", "sample_id", "input_sha256", "api_id", "arm"))
                        or dispatch["run_identity"] != manifest["run_identity"]
                        or dispatch["requested"] != settings(item["api_id"])
                        or dispatch["payload_sha256"] != item["payload_sha256"][arm]
                        or dispatch["reference_answer_visible"] is not False
                        or dispatch["automatic_retry"] is not False):
                    raise ValueError("dispatch_changed")
                dispatched += 1
                sent.append(arm)
                files[str(dispatch_path.relative_to(directory))] = file_digest(dispatch_path)
                record["status"] = "outcome_unknown"
            if result_path.exists():
                result = read(result_path)
                if (any(result[k] != record[k] for k in (
                        "index", "sample_id", "input_sha256", "api_id", "arm"))
                        or result["run_identity"] != manifest["run_identity"]
                        or result["dispatch_sha256"] != file_digest(dispatch_path)
                        or result["reference_answer_visible"] is not False
                        or result["automatic_retry"] is not False):
                    raise ValueError("result_changed")
                if result["status"] in TERMINAL:
                    parsed = parse_answer(result["response"]["text"], len(item["problem"]["options"]))
                    if (parsed["answer"] != result["answer"] or parsed["parse_status"] != result["parse_status"]
                            or result["status"] != ("answered" if parsed["answer"] else "invalid_answer")
                            or result["response"]["tool_call_count"] != 0):
                        raise ValueError("parsed_answer_or_tools_changed")
                elif result["status"] != "service_error" or result["answer"] is not None:
                    raise ValueError("invalid_status")
                files[str(result_path.relative_to(directory))] = file_digest(result_path)
                record.update(status=result["status"], answer=result["answer"])
            records.append(record)
        if sent != item["arm_order"][:len(sent)]:
            raise ValueError("arm_order_changed")
    traces = directory / "traces"
    if traces.exists() and any(path.name not in known for path in traces.iterdir()):
        raise ValueError("unexpected_attempt")
    if dispatched != execution["dispatched"] or dispatched > 10:
        raise ValueError("dispatch_budget_changed")
    if execution["status"] == "complete" and (dispatched != 10 or any(
            p["status"] not in TERMINAL for p in records)):
        raise ValueError("complete_missing_predictions")
    return sorted(records, key=lambda p: (p["index"], p["arm"])), files


def freeze(directory):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory)
        records, files = collect(directory, manifest)
        if (directory / "frozen.json").exists():
            return validate_frozen(directory, manifest)
        result = {"schema": "mmsi-depth-all-api-freeze/1", "run_identity": manifest["run_identity"],
                  "parent_freeze_identity": manifest["parent_freeze_identity"],
                  "predictions": records, "files": files, "reference_answer_visible": False,
                  "automatic_retry": False, "frozen_at": time.time()}
        result["freeze_identity"] = digest(result)
        write(directory / "frozen.json", result, exclusive=True)
        return result


def validate_frozen(directory, manifest):
    result = read(Path(directory) / "frozen.json")
    records, files = collect(directory, manifest)
    if (digest({k: v for k, v in result.items() if k != "freeze_identity"}) != result["freeze_identity"]
            or result["run_identity"] != manifest["run_identity"]
            or result["parent_freeze_identity"] != manifest["parent_freeze_identity"]
            or result["predictions"] != records or result["files"] != files):
        raise ValueError("frozen_evidence_changed")
    return result


def score(directory, references, expected_sha256):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory)
        frozen = validate_frozen(directory, manifest)
        if read(directory / "execution.json")["status"] != "complete":
            raise ValueError("do_not_read_reference_for_incomplete_run")
        raw = Path(references).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("reference_identity_changed")
        items = sorted(manifest["reused_items"] + manifest["new_items"], key=lambda i: i["index"])
        answers = reference_answers(raw, [{"sample_id": i["problem"]["sample_id"]} for i in items])
        predictions = {(p["sample_id"], p["arm"]): p for p in frozen["predictions"]}
        pairs = []
        for item in items:
            sample = item["problem"]["sample_id"]
            answer = answers[sample]
            if answer not in LETTERS[:len(item["problem"]["options"])]:
                raise ValueError("invalid_reference")
            pair = {"sample_id": sample, "task_type": item["problem"]["task_type"],
                    "api_id": item["api_id"], "reference_answer": answer,
                    "input_sha256": item["problem"]["input_sha256"]}
            for arm in ("direct", "depth"):
                p = predictions[sample, arm]
                pair[arm] = {**p, "correct": p["status"] == "answered" and p["answer"] == answer}
            pairs.append(pair)
        result = {
            "schema": "mmsi-depth-all-api-score/1", "run_identity": manifest["run_identity"],
            "freeze_identity": frozen["freeze_identity"], "reference_file_sha256": expected_sha256,
            "reference_access": "host_only_after_all_18_prediction_pairs_freeze",
            "denominator": "all_18_frozen_selected_pairs", "overall": summarize(pairs),
            "by_task": {task: summarize([p for p in pairs if p["task_type"] == task])
                        for task in sorted({p["task_type"] for p in pairs})},
            "by_api": {api: summarize([p for p in pairs if p["api_id"] == api]) for api in ALL_APIS},
            "by_source": {source: summarize([p for p in pairs if p["direct"]["source"] == source])
                          for source in ("parent", "new")},
            "pairs": pairs, "scored_at": time.time(),
        }
        result["score_identity"] = digest(result)
        path = directory / "score.json"
        if path.exists():
            previous = read(path)
            if (digest({k: v for k, v in previous.items() if k != "score_identity"})
                    != previous["score_identity"] or any(
                        previous[k] != result[k] for k in result if k not in ("scored_at", "score_identity"))):
                raise ValueError("existing_score_changed")
            return previous
        write(path, result, exclusive=True)
        return result


def controlled_execute(directory):
    from .depth_service_probe import all_keys
    from .depth_comparison.mac_control import ROOT, clean_environment
    keys = all_keys()
    result = subprocess.run(
        [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.depth_all_api_recovery",
         "execute", "--directory", str(directory)],
        input=json.dumps(keys).encode(), env=clean_environment(), cwd=ROOT)
    if result.returncode:
        raise RuntimeError("recovery_worker_failed")


def controlled_score(directory):
    from .depth_comparison.mac_control import ROOT, REFERENCE, REFERENCE_SHA, clean_environment
    from .depth_comparison.control import SSH, PYTHON
    manifest = validate(directory)
    validate_frozen(directory, manifest)
    if read(Path(directory) / "execution.json")["status"] != "complete":
        raise ValueError("incomplete_no_reference_access")
    code = "from pathlib import Path; import sys; sys.stdout.buffer.write(Path(" + repr(REFERENCE) + ").read_bytes())"
    response = subprocess.run(SSH + [shlex.quote(PYTHON) + " -B -c " + shlex.quote(code)],
                              capture_output=True, timeout=60)
    if response.returncode or hashlib.sha256(response.stdout).hexdigest() != REFERENCE_SHA:
        raise ValueError("host_reference_transfer_failed")
    with tempfile.TemporaryDirectory(prefix="mmsi-all-api-host-score-") as temporary:
        path = Path(temporary) / "references.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(response.stdout)
        result = subprocess.run(
            [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.depth_all_api_recovery",
             "score", "--directory", str(directory), "--references", str(path),
             "--reference-sha256", REFERENCE_SHA], env=clean_environment(), cwd=ROOT)
        if result.returncode:
            raise RuntimeError("host_score_failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "controlled-execute", "execute",
                                         "freeze", "controlled-score", "score"))
    parser.add_argument("--directory", default=str(DEFAULT_RUN))
    parser.add_argument("--references")
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare(args.directory)
        print(json.dumps({"status": "prepared", "new_requests": 10, "reused_pairs": 13,
                          "cumulative_paired_requests": 51, "run_identity": result["run_identity"]}))
    elif args.phase == "controlled-execute":
        controlled_execute(args.directory)
    elif args.phase == "execute":
        result = execute(args.directory, json.load(sys.stdin))
        freeze(args.directory)
        print(json.dumps(result))
    elif args.phase == "freeze":
        result = freeze(args.directory)
        print(json.dumps({"freeze_identity": result["freeze_identity"]}))
    elif args.phase == "controlled-score":
        controlled_score(args.directory)
    else:
        result = score(args.directory, args.references, args.reference_sha256)
        print(json.dumps(result["overall"]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print('{"status":"failed","reason_code":"all_api_phase_failed"}', file=sys.stderr)
        raise SystemExit(2)
