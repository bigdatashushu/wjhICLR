"""Immutable paired run, bounded concurrent slots, prediction freeze then host score."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
import threading
import time

from ...traces import digest, file_digest, read, write
from ..prototype.direct import operator_lock
from ..prototype.protocol import PARSER_VERSION, PROMPT_VERSION, parse_answer
from .depth import RENDERER_VERSION
from .inputs import TOKENIZER_SHA256, prepare_item, request, tokenizer
from .pool import Client, inference
from ..prototype.client import ServiceError

PACKAGE = Path(__file__).resolve().parent
GPTEXP = PACKAGE.parents[1]


def code_identity():
    files = list(PACKAGE.glob("*.py"))
    files += [PACKAGE / "README.md", PACKAGE.parent / "depth_prompt.py",
              PACKAGE.parent / "__init__.py"]
    files += list((PACKAGE.parent / "prototype").glob("*.py"))
    files += [GPTEXP / name for name in (
        "__init__.py", "traces.py", "response_policy.py", "api_pool.py")]
    files += list((GPTEXP / "vendor/xcodeapi").glob("*.py"))
    return {str(path.relative_to(GPTEXP)): file_digest(path) for path in sorted(files)}


def environment_identity():
    import numpy
    import PIL
    import tiktoken
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "pillow": PIL.__version__, "tiktoken": tiktoken.__version__}


def prepare(directory, selection, tokenizer_path):
    if (digest({k: v for k, v in selection.items() if k != "selection_identity"})
            != selection["selection_identity"] or selection["questions"] != 30
            or selection["max_requests"] != 60
            or selection["reference_answer_visible"] is not False):
        raise ValueError("selection_identity_changed")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("requires_empty_new_run_directory")
    count = tokenizer(tokenizer_path)
    items = [prepare_item(directory, item, count) for item in selection["items"]]
    return save_preparation(directory, selection, items)


def save_preparation(directory, selection, items, *, continuation=None):
    directory = Path(directory)
    manifest = {
        "schema": "mmsi-depth-paired/1", "selection": selection,
        "items": items, "questions": selection["questions"],
        "max_requests": selection["max_requests"],
        "api_ids": selection["api_ids"], "max_inflight": len(selection["api_ids"]),
        "inference_by_api": {api: inference(api) for api in selection["api_ids"]},
        "code_identity": code_identity(), "environment": environment_identity(),
        "prompt_version": PROMPT_VERSION, "parser_version": PARSER_VERSION,
        "renderer_version": RENDERER_VERSION,
        "tokenizer": {"name": "o200k_base", "sha256": TOKENIZER_SHA256},
        "reference_answer_visible": False, "automatic_retry": False, "tools_enabled": False,
        "prepared_at": time.time(),
    }
    if continuation is not None:
        manifest["continuation"] = continuation
    manifest["run_identity"] = digest(manifest)
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "preflight.json", {
        "run_identity": manifest["run_identity"], "model_requests": 0,
        "reference_answers_read": False, "questions": manifest["questions"],
        "original_images_verified": True, "depth_content_hashes_verified": True,
        "added_text_tokens": [item["added_text_tokens"] for item in items],
    }, exclusive=True)
    validate(directory, verify_requests=True)
    return manifest


def validate(directory, *, verify_requests=False):
    manifest = read(Path(directory) / "manifest.json")
    total = manifest["questions"]
    if (manifest["schema"] != "mmsi-depth-paired/1"
            or digest({k: v for k, v in manifest.items() if k != "run_identity"})
            != manifest["run_identity"] or manifest["code_identity"] != code_identity()
            or manifest["environment"] != environment_identity()
            or manifest["reference_answer_visible"] is not False
            or manifest["automatic_retry"] is not False or manifest["tools_enabled"] is not False
            or total not in (18, 24, 30) or len(manifest["items"]) != total
            or manifest["max_requests"] != 2 * total
            or manifest["prompt_version"] != PROMPT_VERSION
            or manifest["parser_version"] != PARSER_VERSION
            or manifest["renderer_version"] != RENDERER_VERSION
            or manifest["max_inflight"] != len(manifest["api_ids"])
            or manifest["inference_by_api"] != {
                api: inference(api) for api in manifest["api_ids"]}):
        raise ValueError("run_configuration_or_code_changed")
    selection = manifest["selection"]
    if (digest({k: v for k, v in selection.items() if k != "selection_identity"})
            != selection["selection_identity"] or selection["questions"] != total
            or selection["max_requests"] != manifest["max_requests"]
            or selection["api_ids"] != manifest["api_ids"]
            or selection["reference_answer_visible"] is not False):
        raise ValueError("selection_identity_changed")
    if total in (18, 24):
        from .continuation import validate_continuation
        validate_continuation(manifest)
    elif "continuation" in manifest:
        raise ValueError("unexpected_continuation")
    ids = [item["problem"]["sample_id"] for item in manifest["items"]]
    tasks = Counter(item["problem"]["task_type"] for item in manifest["items"])
    if (len(set(ids)) != total or len(tasks) != 6 or set(tasks.values()) != {total // 6}
            or [item["index"] for item in manifest["items"]] != list(range(total))
            or Counter(item["arm_order"][0] for item in manifest["items"])
            != {"direct": total // 2, "depth": total // 2}):
        raise ValueError("paired_selection_or_order_changed")
    for item in manifest["items"]:
        if (sorted(item["arm_order"]) != ["depth", "direct"]
                or item["api_id"] not in manifest["api_ids"]
                or item["annotation"]["fixed_tokens"] > 80
                or item["annotation"]["total_tokens"] > 160
                or item["added_text_tokens"] > 160):
            raise ValueError("paired_arm_binding_changed")
        if verify_requests:
            for arm in item["arm_order"]:
                if digest(request(directory, item, arm)) != item["payload_sha256"][arm]:
                    raise ValueError("prepared_request_changed")
    preflight = read(Path(directory) / "preflight.json")
    if preflight["run_identity"] != manifest["run_identity"] or preflight["model_requests"] != 0:
        raise ValueError("preflight_identity_changed")
    return manifest


def paths(directory, item, arm):
    stem = Path(directory) / "traces" / f"{item['index']:04d}-{arm}"
    return Path(str(stem) + ".dispatch.json"), Path(str(stem) + ".result.json")


def execute(directory, keys, *, client_factory=Client):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory, verify_requests=True)
        if (directory / "execution.started.json").exists() or (directory / "frozen.json").exists():
            raise ValueError("already_started_no_resume_or_retry")
        if (directory / "traces").exists():
            raise ValueError("existing_attempts_no_resume")
        if set(keys) != set(manifest["api_ids"]):
            raise ValueError("credential_slots_changed")
        clients = {api: client_factory(api, key) for api, key in keys.items()}
        for client in clients.values():
            client.assert_public(manifest)
        stop = threading.Event()
        guard = threading.Lock()
        state = {"dispatched": 0, "peak_inflight": 0, "inflight": 0}
        write(directory / "execution.started.json", {
            "run_identity": manifest["run_identity"], "max_requests": manifest["max_requests"],
            "max_inflight": manifest["max_inflight"], "at": time.time(),
            "reference_answer_visible": False, "automatic_retry": False,
            "credential_storage": "process_memory_only",
        }, exclusive=True)

        def slot(api):
            client = clients[api]
            for item in manifest["items"]:
                if item["api_id"] != api:
                    continue
                for arm in item["arm_order"]:
                    if stop.is_set():
                        return
                    body = request(directory, item, arm)
                    if digest(body) != item["payload_sha256"][arm]:
                        stop.set()
                        raise ValueError("request_changed_after_preflight")
                    for check_client in clients.values():
                        check_client.assert_public(body)
                    dispatch_path, result_path = paths(directory, item, arm)
                    dispatch = {
                        "index": item["index"], "sample_id": item["problem"]["sample_id"],
                        "input_sha256": item["problem"]["input_sha256"], "arm": arm, "api_id": api,
                        "run_identity": manifest["run_identity"],
                        "payload_sha256": item["payload_sha256"][arm],
                        "requested": inference(api), "reference_answer_visible": False,
                        "automatic_retry": False, "at": time.time(),
                    }
                    with guard:
                        if stop.is_set():
                            return
                        if state["dispatched"] >= manifest["max_requests"]:
                            raise ValueError("request_budget_exceeded")
                        write(dispatch_path, dispatch, exclusive=True)
                        state["dispatched"] += 1
                        state["inflight"] += 1
                        state["peak_inflight"] = max(state["peak_inflight"], state["inflight"])
                    try:
                        response = client.call(body)
                        for check_client in clients.values():
                            check_client.assert_public(response)
                        parsed = parse_answer(response["text"], len(item["problem"]["options"]))
                        result = {
                            "status": "answered" if parsed["answer"] else "invalid_answer",
                            **parsed, "response": response,
                        }
                    except ServiceError as exc:
                        stop.set()
                        result = {"status": "service_error", "answer": None,
                                  "parse_status": "not_parsed", "error_code": exc.code,
                                  "http_status": exc.http_status}
                    except Exception:
                        stop.set()
                        # Includes possible credential echoes; no exception text is persisted.
                        result = {"status": "service_error", "answer": None,
                                  "parse_status": "not_parsed",
                                  "error_code": "client_outcome_unknown", "http_status": None}
                    result.update({
                        **{key: dispatch[key] for key in (
                            "index", "sample_id", "input_sha256", "arm", "api_id", "run_identity",
                            "reference_answer_visible", "automatic_retry")},
                        "dispatch_sha256": file_digest(dispatch_path), "at": time.time(),
                    })
                    for check_client in clients.values():
                        check_client.assert_public(result)
                    write(result_path, result, exclusive=True)
                    with guard:
                        state["inflight"] -= 1
                        progress = state["dispatched"]
                    print(f"{progress}/{manifest['max_requests']} "
                          f"{result['sample_id']} {arm}: {result['status']}", flush=True)
                    if stop.is_set():
                        return

        execution_status = "interrupted"
        try:
            with ThreadPoolExecutor(max_workers=len(clients)) as pool:
                list(pool.map(slot, manifest["api_ids"]))
            execution_status = "stopped_service_error" if stop.is_set() else "complete"
        finally:
            stop.set()
            write(directory / "execution.json", {
                **state, "status": execution_status, "run_identity": manifest["run_identity"],
                "reference_answer_visible": False, "automatic_retry": False, "at": time.time(),
            }, exclusive=True)
        return read(directory / "execution.json")


def collect(directory, manifest):
    directory = Path(directory)
    started = read(directory / "execution.started.json")
    execution = read(directory / "execution.json")
    if (started["run_identity"] != manifest["run_identity"]
            or execution["run_identity"] != manifest["run_identity"]
            or started["reference_answer_visible"] is not False
            or started["automatic_retry"] is not False or execution["inflight"] != 0
            or started["max_requests"] != manifest["max_requests"]
            or started["max_inflight"] != manifest["max_inflight"]
            or execution["peak_inflight"] > manifest["max_inflight"]
            or execution["status"] not in ("complete", "stopped_service_error", "interrupted")):
        raise ValueError("execution_identity_or_terminal_state_changed")
    known, records, files = set(), [], {}
    for name in ("manifest.json", "preflight.json", "execution.started.json", "execution.json"):
        files[name] = file_digest(directory / name)
    for item in manifest["items"]:
        for receipt in item["depth_images"]:
            files[receipt["path"]] = file_digest(directory / receipt["path"])
            if files[receipt["path"]] != receipt["sha256"]:
                raise ValueError("depth_file_changed_before_freeze")
        arm_dispatched = []
        for arm in item["arm_order"]:
            dispatch_path, result_path = paths(directory, item, arm)
            known.update((dispatch_path.name, result_path.name))
            record = {
                "index": item["index"], "sample_id": item["problem"]["sample_id"],
                "task_type": item["problem"]["task_type"],
                "input_sha256": item["problem"]["input_sha256"], "api_id": item["api_id"],
                "arm": arm, "status": "not_dispatched", "answer": None,
            }
            if result_path.exists() and not dispatch_path.exists():
                raise ValueError("result_without_dispatch")
            if dispatch_path.exists():
                dispatch = read(dispatch_path)
                if (any(dispatch[key] != record[key] for key in (
                        "index", "sample_id", "input_sha256", "api_id", "arm"))
                        or dispatch["run_identity"] != manifest["run_identity"]
                        or dispatch["payload_sha256"] != item["payload_sha256"][arm]
                        or dispatch["requested"] != inference(item["api_id"])
                        or dispatch["reference_answer_visible"] is not False
                        or dispatch["automatic_retry"] is not False):
                    raise ValueError("dispatch_identity_changed")
                arm_dispatched.append(arm)
                files[str(dispatch_path.relative_to(directory))] = file_digest(dispatch_path)
                record["status"] = "outcome_unknown"
            if result_path.exists():
                result = read(result_path)
                if (any(result[key] != record[key] for key in (
                        "index", "sample_id", "input_sha256", "api_id", "arm"))
                        or result["run_identity"] != manifest["run_identity"]
                        or result["dispatch_sha256"] != file_digest(dispatch_path)
                        or result["reference_answer_visible"] is not False
                        or result["automatic_retry"] is not False):
                    raise ValueError("result_identity_changed")
                if result["status"] in ("answered", "invalid_answer"):
                    parsed = parse_answer(result["response"]["text"], len(item["problem"]["options"]))
                    if (parsed["answer"] != result["answer"]
                            or parsed["parse_status"] != result["parse_status"]
                            or result["status"] != ("answered" if parsed["answer"] else "invalid_answer")
                            or result["response"]["tool_call_count"] != 0):
                        raise ValueError("answer_parse_or_tool_protocol_changed")
                elif result["status"] != "service_error" or result["answer"] is not None:
                    raise ValueError("invalid_result_status")
                files[str(result_path.relative_to(directory))] = file_digest(result_path)
                record.update(status=result["status"], answer=result["answer"])
            records.append(record)
        if arm_dispatched != item["arm_order"][:len(arm_dispatched)]:
            raise ValueError("paired_arm_order_changed")
    traces = directory / "traces"
    if traces.exists() and any(path.name not in known for path in traces.iterdir()):
        raise ValueError("unexpected_attempt_file")
    dispatched = sum(record["status"] != "not_dispatched" for record in records)
    if dispatched != execution["dispatched"] or dispatched > manifest["max_requests"]:
        raise ValueError("request_count_changed")
    if execution["status"] == "complete" and (
            dispatched != manifest["max_requests"] or any(record["status"] not in (
                "answered", "invalid_answer") for record in records)):
        raise ValueError("complete_run_missing_outcomes")
    return records, files


def freeze(directory):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory)
        records, files = collect(directory, manifest)
        if (directory / "frozen.json").exists():
            return validate_frozen(directory, manifest)
        result = {
            "schema": "mmsi-depth-prediction-freeze/1", "run_identity": manifest["run_identity"],
            "predictions": records, "files": files, "reference_answer_visible": False,
            "automatic_retry": False, "frozen_at": time.time(),
        }
        result["freeze_identity"] = digest(result)
        write(directory / "frozen.json", result, exclusive=True)
        return result


def validate_frozen(directory, manifest):
    frozen = read(Path(directory) / "frozen.json")
    records, files = collect(directory, manifest)
    if (digest({k: v for k, v in frozen.items() if k != "freeze_identity"})
            != frozen["freeze_identity"] or frozen["run_identity"] != manifest["run_identity"]
            or frozen["predictions"] != records or frozen["files"] != files):
        raise ValueError("records_changed_after_prediction_freeze")
    return frozen
