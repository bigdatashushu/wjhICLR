"""Prepare and execute a serial Direct run with one durable dispatch per question."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import fcntl
import platform
from pathlib import Path
import time

from ...traces import digest, file_digest, read, write
from .client import ResponsesClient, ServiceError, configuration
from .inputs import load_images, public_problem, read_public_source, select
from .protocol import PARSER_VERSION, PROMPT_VERSION, messages, parse_answer, payload, prompt

SCHEMA = "mmsi-direct-prototype/1"
PACKAGE = Path(__file__).resolve().parent
GPTEXP = PACKAGE.parents[1]


def code_identity():
    """Pin only this implementation and the small read-only utilities it imports."""
    files = list(PACKAGE.glob("*.py"))
    files += [PACKAGE / "README.md", PACKAGE.parent / "AGENTS.md", PACKAGE.parent / "__init__.py"]
    files += [GPTEXP / name for name in (
        "__init__.py", "traces.py", "api_pool.py", "response_policy.py")]
    files += list((GPTEXP / "vendor/xcodeapi").glob("*.py"))
    return {str(p.relative_to(GPTEXP)): file_digest(p) for p in sorted(files)}


def environment_identity():
    import PIL

    return {"python": platform.python_version(), "pillow": PIL.__version__}


@contextmanager
def operator_lock(directory):
    with (Path(directory) / "operator.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("run_has_active_operator") from None
        yield


def item_request(item, inference):
    problem = public_problem(item["problem"])
    images = load_images(problem, item["images"])
    body = payload(messages(problem, images), inference)
    if (digest(prompt(problem)) != item["prompt_sha256"]
            or digest(body) != item["payload_sha256"]):
        raise ValueError("prepared_request_identity_changed")
    return body


def prepare(directory, source, config, api_id, *, per_type=1, sample_ids=None,
            excluded=(), all_samples=False, max_output_tokens=8192):
    api, profile, inference, config_sha = configuration(
        config, api_id, max_output_tokens=max_output_tokens)
    rows, source_sha = read_public_source(source)
    selected = select(rows, sample_ids=sample_ids, excluded=excluded,
                      per_type=per_type, all_samples=all_samples, seed=inference["seed"])
    if not selected:
        raise ValueError("empty_question_selection")
    items = []
    for index, row in enumerate(selected):
        problem = row["problem"]
        images = load_images(problem, row["images"])
        body = payload(messages(problem, images), inference)
        items.append({
            "index": index, "problem": problem, "images": [image[0] for image in images],
            "prompt_sha256": digest(prompt(problem)), "payload_sha256": digest(body),
        })
    manifest = {
        "schema": SCHEMA, "prompt_version": PROMPT_VERSION, "parser_version": PARSER_VERSION,
        "inference": inference, "config_sha256": config_sha, "source_manifest_sha256": source_sha,
        "source_files": code_identity(), "environment": environment_identity(),
        "questions": len(items), "max_requests": len(items), "max_inflight": 1,
        "task_counts": dict(Counter(item["problem"]["task_type"] for item in items)),
        "selection": {
            "mode": "explicit_ids" if sample_ids is not None else "all" if all_samples else "stratified",
            "per_type": per_type if sample_ids is None and not all_samples else None,
            "seed": inference["seed"], "excluded_sample_ids": sorted(set(excluded)),
            "sample_ids": [item["problem"]["sample_id"] for item in items],
        },
        "items": items, "automatic_retry": False, "reference_answer_visible": False,
        "tools_enabled": False, "prepared_geometry_loaded": False, "prepared_at": time.time(),
    }
    manifest["run_identity"] = digest(manifest)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("prepare_requires_empty_new_run_directory")
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "preparation.json", {
        "status": "ready", "run_identity": manifest["run_identity"], "model_requests": 0,
        "questions": len(items), "reference_answers_read": False, "original_images_verified": True,
    }, exclusive=True)
    return {"status": "ready", "questions": len(items), "max_requests": len(items),
            "run_identity": manifest["run_identity"], "model_requests": 0}


def validate_manifest(directory, *, verify_images=False):
    manifest = read(Path(directory) / "manifest.json")
    if (manifest.get("schema") != SCHEMA
            or digest({k: v for k, v in manifest.items() if k != "run_identity"}) != manifest["run_identity"]
            or manifest["source_files"] != code_identity()
            or manifest["environment"] != environment_identity()
            or manifest["prompt_version"] != PROMPT_VERSION or manifest["parser_version"] != PARSER_VERSION
            or manifest["reference_answer_visible"] is not False or manifest["automatic_retry"] is not False
            or manifest["tools_enabled"] is not False or manifest["prepared_geometry_loaded"] is not False
            or manifest["max_inflight"] != 1 or manifest["questions"] != len(manifest["items"])
            or manifest["max_requests"] != manifest["questions"]):
        raise ValueError("run_manifest_or_source_identity_changed")
    ids = []
    for index, item in enumerate(manifest["items"]):
        problem = public_problem(item["problem"])
        ids.append(problem["sample_id"])
        if item["index"] != index or digest(prompt(problem)) != item["prompt_sha256"]:
            raise ValueError("public_question_order_or_prompt_changed")
        if verify_images:
            item_request(item, manifest["inference"])
    if len(set(ids)) != len(ids) or ids != manifest["selection"]["sample_ids"]:
        raise ValueError("question_selection_changed")
    if dict(Counter(i["problem"]["task_type"] for i in manifest["items"])) != manifest["task_counts"]:
        raise ValueError("task_denominator_changed")
    preparation = read(Path(directory) / "preparation.json")
    if preparation["run_identity"] != manifest["run_identity"] or preparation["model_requests"] != 0:
        raise ValueError("preparation_receipt_changed")
    return manifest


def trace_paths(directory, item):
    root = Path(directory) / "traces"
    stem = f"{item['index']:04d}"
    return root / (stem + ".dispatch.json"), root / (stem + ".result.json")


def execute(directory, config, *, client_factory=ResponsesClient):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate_manifest(directory, verify_images=True)
        if (directory / "execution.started.json").exists() or (directory / "frozen.json").exists():
            raise ValueError("run_already_started_or_frozen_no_resume")
        if (directory / "traces").exists() and any((directory / "traces").iterdir()):
            raise ValueError("existing_dispatch_records_no_resume")
        inference = manifest["inference"]
        api, profile, current, config_sha = configuration(
            config, inference["api_id"], max_output_tokens=inference["max_output_tokens"])
        if config_sha != manifest["config_sha256"] or current != inference:
            raise ValueError("external_endpoint_configuration_changed")
        client = client_factory(api, profile, inference)
        client.assert_public(manifest)
        write(directory / "execution.started.json", {
            "run_identity": manifest["run_identity"], "at": time.time(),
            "automatic_retry": False, "max_requests": manifest["max_requests"],
            "reference_answer_visible": False,
        }, exclusive=True)
        requests, status = 0, "complete"
        try:
            for item in manifest["items"]:
                body = item_request(item, inference)
                client.assert_public(body)
                dispatch_path, result_path = trace_paths(directory, item)
                dispatch = {
                    "index": item["index"], "sample_id": item["problem"]["sample_id"],
                    "run_identity": manifest["run_identity"], "payload_sha256": digest(body),
                    "input_sha256": item["problem"]["input_sha256"],
                    "prompt_sha256": item["prompt_sha256"], "images": item["images"],
                    "requested": inference, "at": time.time(), "automatic_retry": False,
                    "reference_answer_visible": False,
                }
                # A crash after this fsynced record consumes the attempt conservatively.
                write(dispatch_path, dispatch, exclusive=True)
                requests += 1
                try:
                    response = client.call(body)
                    client.assert_public(response)
                    parsed = parse_answer(response["text"], len(item["problem"]["options"]))
                    result = {
                        "status": "answered" if parsed["answer"] is not None else "invalid_answer",
                        **parsed, "response": response,
                    }
                except ServiceError as exc:
                    # Error messages, HTTP bodies, headers and credentials never enter records.
                    result = {"status": "service_error", "answer": None, "parse_status": "not_parsed",
                              "error_code": exc.code, "http_status": exc.http_status}
                    status = "stopped_service_error"
                result.update(
                    index=item["index"], sample_id=item["problem"]["sample_id"],
                    run_identity=manifest["run_identity"], input_sha256=item["problem"]["input_sha256"],
                    dispatch_sha256=file_digest(dispatch_path), reference_answer_visible=False,
                    automatic_retry=False, at=time.time())
                client.assert_public(result)
                write(result_path, result, exclusive=True)
                print(f"{requests}/{manifest['questions']} {result['sample_id']}: {result['status']}",
                      flush=True)
                if result["status"] == "service_error":
                    break
        except BaseException:
            write(directory / "execution.json", {
                "run_identity": manifest["run_identity"], "status": "interrupted",
                "dispatched": requests, "at": time.time(), "automatic_retry": False,
                "reference_answer_visible": False,
            }, exclusive=True)
            raise
        summary = {
            "run_identity": manifest["run_identity"], "status": status, "dispatched": requests,
            "remaining": manifest["questions"] - requests, "at": time.time(),
            "automatic_retry": False, "reference_answer_visible": False,
        }
        write(directory / "execution.json", summary, exclusive=True)
        return summary
