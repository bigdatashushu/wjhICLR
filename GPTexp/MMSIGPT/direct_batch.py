"""Ten independent API lanes, one original-RGB attempt per fixed MMSI question."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

from xcodeapi.config import Profile

from ..traces import digest, file_digest, read, write
from .depth_comparison.pool import BASE_URL, MODEL, Client, inference
from .prototype.client import ServiceError
from .prototype.direct import operator_lock
from .prototype.inputs import load_images, public_problem
from .prototype.protocol import messages, payload, parse_answer, PROMPT_VERSION, PARSER_VERSION

ROOT = Path(__file__).resolve().parents[2]
APIS = tuple(f"pool-{i:02d}" for i in range(1, 11))
SCHEMA = "mmsi-direct-ten-api-batch/1"


def settings(api):
    return {**inference(api), "timeout_seconds": 180}


def source_identity():
    names = (
        "GPTexp/__init__.py", "GPTexp/api_pool.py", "GPTexp/response_policy.py",
        "GPTexp/traces.py", "GPTexp/MMSIGPT/__init__.py",
        "GPTexp/MMSIGPT/direct_batch.py", "GPTexp/MMSIGPT/prototype/__init__.py",
        "GPTexp/MMSIGPT/prototype/client.py", "GPTexp/MMSIGPT/prototype/direct.py",
        "GPTexp/MMSIGPT/prototype/inputs.py", "GPTexp/MMSIGPT/prototype/protocol.py",
        "GPTexp/MMSIGPT/depth_comparison/__init__.py",
        "GPTexp/MMSIGPT/depth_comparison/pool.py",
    )
    paths = [ROOT / name for name in names]
    paths += sorted((ROOT / "GPTexp/vendor/xcodeapi").glob("*.py"))
    return {str(p.relative_to(ROOT)): file_digest(p) for p in paths}


def request(item):
    return payload(messages(item["problem"], load_images(item["problem"], item["images"])),
                   settings(item["api_id"]))


def prepare(directory, public_source, qualification):
    directory, public_source = Path(directory).resolve(), Path(public_source).resolve()
    if directory.exists() or not directory.parent.is_dir():
        raise ValueError("requires_new_run_directory")
    source = read(public_source)
    if (source.get("reference_answer_visible") is not False or source["questions"] != 240
            or digest({k: v for k, v in source.items() if k != "run_identity"})
            != source["run_identity"]):
        raise ValueError("public_source_identity_changed")
    if ({r["api_id"] for r in qualification
         if r.get("http_status") == 200 and r.get("terra_listed") is True} != set(APIS)):
        raise ValueError("requires_ten_current_qualified_apis")
    items = []
    for index, original in enumerate(source["items"]):
        item = {
            "index": index, "problem": public_problem(original["problem"]),
            "images": original["images"], "api_id": APIS[index % len(APIS)],
        }
        item["payload_sha256"] = digest(request(item))
        items.append(item)
    tasks = Counter(i["problem"]["task_type"] for i in items)
    if (len(items) != 240 or len({i["problem"]["sample_id"] for i in items}) != 240
            or len(tasks) != 6 or set(tasks.values()) != {40}):
        raise ValueError("requires_exact_six_by_forty_question_set")
    manifest = {
        "schema": SCHEMA, "items": items, "questions": 240, "max_requests": 240,
        "api_ids": list(APIS), "max_inflight": 10,
        "inference_by_api": {api: settings(api) for api in APIS},
        "task_counts": dict(tasks), "source_identity": source_identity(),
        "source_manifest_sha256": file_digest(public_source),
        "source_run_identity": source["run_identity"],
        "prompt_version": PROMPT_VERSION, "parser_version": PARSER_VERSION,
        "sample_role": source["sample_role"], "source_split": "test",
        "reference_answer_visible": False, "automatic_retry": False, "tools_enabled": False,
        "prepared_geometry_loaded": False, "depth_images_sent": False,
        "failure_policy": "retain_failure_and_continue_only_previously_undispatched_questions",
        "prepared_at": time.time(),
    }
    manifest["run_identity"] = digest(manifest)
    directory.mkdir()
    (directory / "traces").mkdir()
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "api-qualification.json", {
        "results": qualification, "model_requests": 0,
        "qualification_request": "GET_models", "credentials_persisted": False,
    }, exclusive=True)
    validate(directory, verify_requests=True)
    write(directory / "preflight.json", {
        "run_identity": manifest["run_identity"], "questions": 240,
        "original_images_verified": True, "reference_answers_read": False,
        "model_requests": 0, "max_inflight": 10,
    }, exclusive=True)
    return manifest


def validate(directory, *, verify_requests=False):
    manifest = read(Path(directory) / "manifest.json")
    if (manifest["schema"] != SCHEMA
            or digest({k: v for k, v in manifest.items() if k != "run_identity"})
            != manifest["run_identity"] or manifest["source_identity"] != source_identity()
            or manifest["api_ids"] != list(APIS) or manifest["max_inflight"] != 10
            or manifest["questions"] != 240 or manifest["max_requests"] != 240
            or len(manifest["items"]) != 240
            or manifest["inference_by_api"] != {api: settings(api) for api in APIS}
            or manifest["prompt_version"] != PROMPT_VERSION
            or manifest["parser_version"] != PARSER_VERSION
            or any(manifest[k] is not False for k in (
                "reference_answer_visible", "automatic_retry", "tools_enabled",
                "prepared_geometry_loaded", "depth_images_sent"))):
        raise ValueError("direct_manifest_or_code_changed")
    ids, tasks = set(), Counter()
    for index, item in enumerate(manifest["items"]):
        problem = public_problem(item["problem"])
        sample = problem["sample_id"]
        if (set(item) != {"index", "problem", "images", "api_id", "payload_sha256"}
                or item["index"] != index or sample in ids
                or item["api_id"] != APIS[index % 10]):
            raise ValueError("question_or_slot_binding_changed")
        ids.add(sample)
        tasks[problem["task_type"]] += 1
        if verify_requests and digest(request(item)) != item["payload_sha256"]:
            raise ValueError("original_request_changed")
    if len(tasks) != 6 or set(tasks.values()) != {40} or dict(tasks) != manifest["task_counts"]:
        raise ValueError("task_denominator_changed")
    return manifest


def trace_paths(directory, item):
    stem = f"{item['index']:04d}"
    root = Path(directory) / "traces"
    return root / (stem + ".dispatch.json"), root / (stem + ".result.json")


class BatchClient(Client):
    def __init__(self, api, key):
        super().__init__(api, key)
        self.inference = settings(api)
        self.profile = Profile("responses", BASE_URL, MODEL, "XCODE_GPT_API_KEY",
                               timeout_seconds=self.inference["timeout_seconds"])


def execute(directory, keys, *, client_factory=BatchClient):
    directory = Path(directory).resolve()
    with operator_lock(directory):
        manifest = validate(directory, verify_requests=True)
        if (set(keys) != set(APIS) or len(set(keys.values())) != 10
                or (directory / "execution.started.json").exists()
                or any((directory / "traces").iterdir())):
            raise ValueError("keys_or_new_run_requirement_failed")
        preflight = read(directory / "preflight.json")
        if preflight["run_identity"] != manifest["run_identity"] or preflight["model_requests"] != 0:
            raise ValueError("preflight_changed")
        clients = {api: client_factory(api, keys[api]) for api in APIS}
        for client in clients.values():
            client.assert_public(manifest)
        write(directory / "execution.started.json", {
            "run_identity": manifest["run_identity"], "pid": __import__("os").getpid(),
            "at": time.time(), "max_requests": 240, "max_inflight": 10,
            "reference_answer_visible": False, "automatic_retry": False,
        }, exclusive=True)
        lock = threading.Lock()
        counts, completed_by_api = Counter(), Counter()
        active, peak, completed = 0, 0, 0

        def progress():
            write(directory / "progress.json", {
                "run_identity": manifest["run_identity"], "completed": completed,
                "remaining": 240 - completed, "inflight": active,
                "peak_inflight": peak, "status_counts": dict(counts),
                "completed_by_api": dict(completed_by_api), "at": time.time(),
            })

        def lane(api):
            nonlocal active, peak, completed
            client = clients[api]
            for item in [i for i in manifest["items"] if i["api_id"] == api]:
                body = request(item)
                if digest(body) != item["payload_sha256"]:
                    raise ValueError("request_changed_before_dispatch")
                for guard in clients.values():
                    guard.assert_public(body)
                dispatch_path, result_path = trace_paths(directory, item)
                dispatch = {
                    "run_identity": manifest["run_identity"], "index": item["index"],
                    "sample_id": item["problem"]["sample_id"],
                    "input_sha256": item["problem"]["input_sha256"], "api_id": api,
                    "payload_sha256": item["payload_sha256"], "requested": settings(api),
                    "reference_answer_visible": False, "automatic_retry": False,
                    "at": time.time(),
                }
                write(dispatch_path, dispatch, exclusive=True)
                with lock:
                    active += 1
                    peak = max(peak, active)
                    progress()
                try:
                    response = client.call(body)
                    for guard in clients.values():
                        guard.assert_public(response)
                    parsed = parse_answer(response["text"], len(item["problem"]["options"]))
                    result = {
                        "status": "answered" if parsed["answer"] is not None else "invalid_answer",
                        **parsed, "response": response,
                    }
                except ServiceError as error:
                    result = {
                        "status": "service_error", "answer": None, "parse_status": "not_parsed",
                        "error_code": error.code, "http_status": error.http_status,
                    }
                result.update(
                    run_identity=manifest["run_identity"], index=item["index"], api_id=api,
                    sample_id=item["problem"]["sample_id"],
                    input_sha256=item["problem"]["input_sha256"],
                    dispatch_sha256=file_digest(dispatch_path), at=time.time(),
                    reference_answer_visible=False, automatic_retry=False)
                for guard in clients.values():
                    guard.assert_public(result)
                write(result_path, result, exclusive=True)
                with lock:
                    active -= 1
                    completed += 1
                    counts[result["status"]] += 1
                    completed_by_api[api] += 1
                    progress()
                    print(json.dumps({
                        "completed": completed, "remaining": 240 - completed, "api_id": api,
                        "sample_id": result["sample_id"], "status": result["status"],
                        **({"http_status": result["http_status"]}
                           if "http_status" in result else {}),
                    }), flush=True)

        status = "all_attempts_terminal"
        try:
            with ThreadPoolExecutor(max_workers=10) as pool:
                list(pool.map(lane, APIS))
        except Exception:
            status = "interrupted"
        value = {
            "run_identity": manifest["run_identity"], "status": status,
            "completed": completed, "remaining": 240 - completed,
            "peak_inflight": peak, "status_counts": dict(counts), "at": time.time(),
            "reference_answer_visible": False, "automatic_retry": False,
        }
        write(directory / "execution.json", value, exclusive=True)
        if status != "all_attempts_terminal" or completed != 240:
            raise ValueError("online_execution_interrupted_no_automatic_restart")
        return value


def collect(directory, manifest):
    directory = Path(directory)
    started = read(directory / "execution.started.json")
    execution = read(directory / "execution.json")
    if (started["run_identity"] != manifest["run_identity"]
            or execution["run_identity"] != manifest["run_identity"]
            or execution["status"] != "all_attempts_terminal" or execution["completed"] != 240
            or execution["peak_inflight"] > 10):
        raise ValueError("requires_all_240_terminal_attempts")
    files = {name: file_digest(directory / name) for name in (
        "manifest.json", "preflight.json", "api-qualification.json",
        "execution.started.json", "execution.json")}
    predictions = []
    for item in manifest["items"]:
        dp, rp = trace_paths(directory, item)
        dispatch, result = read(dp), read(rp)
        for value in (dispatch, result):
            if (value["run_identity"] != manifest["run_identity"]
                    or value["sample_id"] != item["problem"]["sample_id"]
                    or value["input_sha256"] != item["problem"]["input_sha256"]
                    or value["index"] != item["index"] or value["api_id"] != item["api_id"]
                    or value["reference_answer_visible"] is not False
                    or value["automatic_retry"] is not False):
                raise ValueError("trace_binding_changed")
        if (dispatch["payload_sha256"] != item["payload_sha256"]
                or dispatch["requested"] != settings(item["api_id"])
                or result["dispatch_sha256"] != file_digest(dp)):
            raise ValueError("dispatch_or_result_identity_changed")
        if result["status"] in ("answered", "invalid_answer"):
            parsed = parse_answer(result["response"]["text"], len(item["problem"]["options"]))
            if (parsed["answer"] != result["answer"]
                    or parsed["parse_status"] != result["parse_status"]
                    or result["status"] != ("answered" if parsed["answer"] else "invalid_answer")
                    or result["response"]["tool_call_count"] != 0):
                raise ValueError("prediction_parser_or_tools_changed")
        elif result["status"] != "service_error" or result["answer"] is not None:
            raise ValueError("invalid_terminal_status")
        for path in (dp, rp):
            files[str(path.relative_to(directory))] = file_digest(path)
        predictions.append({
            "index": item["index"], "sample_id": item["problem"]["sample_id"],
            "input_sha256": item["problem"]["input_sha256"],
            "task_type": item["problem"]["task_type"], "api_id": item["api_id"],
            "status": result["status"], "answer": result["answer"],
        })
    expected = {Path(name).name for name in files if name.startswith("traces/")}
    if {p.name for p in (directory / "traces").iterdir()} != expected or len(expected) != 480:
        raise ValueError("unexpected_or_missing_trace_files")
    if dict(Counter(p["status"] for p in predictions)) != execution["status_counts"]:
        raise ValueError("execution_counts_changed")
    return predictions, files


def freeze(directory):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory)
        predictions, files = collect(directory, manifest)
        value = {
            "schema": "mmsi-direct-ten-api-frozen/1",
            "run_identity": manifest["run_identity"], "predictions": predictions,
            "files": files, "reference_answer_visible": False, "frozen_at": time.time(),
        }
        value["freeze_identity"] = digest(value)
        write(directory / "frozen.json", value, exclusive=True)
        return value


def validate_frozen(directory):
    manifest = validate(directory)
    frozen = read(Path(directory) / "frozen.json")
    predictions, files = collect(directory, manifest)
    if (frozen["run_identity"] != manifest["run_identity"]
            or digest({k: v for k, v in frozen.items() if k != "freeze_identity"})
            != frozen["freeze_identity"] or frozen["predictions"] != predictions
            or frozen["files"] != files):
        raise ValueError("prediction_freeze_changed")
    return manifest, frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("execute", "freeze", "validate"))
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    try:
        if args.phase == "execute":
            import sys
            execute(args.directory, json.load(sys.stdin))
        elif args.phase == "freeze":
            freeze(args.directory)
        else:
            validate(args.directory, verify_requests=True)
    except Exception:
        print('{"status":"failed","reason_code":"direct_batch_operation_failed"}', flush=True)
        raise SystemExit(2)
