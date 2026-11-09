#!/usr/bin/env python3
"""CV3D evaluation entry point. Default: CPU preflight; --execute makes paid calls."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import json
from pathlib import Path
import sys
import threading

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "CV3D"

from skill3d.small_image.datasets import load_evaluation_answers
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig

from .client import NativeToolClient
from .common import (
    InferenceConfig, PROTOCOL, digest, file_sha256, read_json, write_json,
)
from .data import index_prepared, load_inputs
from .inference import prepare_request, solve

_STOP_STATUSES = {"service_error", "host_error", "budget_exhausted", "cancelled"}


def summarize(inputs, answers, results):
    """Every selected question stays in the full denominator, including missing/error."""
    def section(selected):
        rows = [results[p.problem.sample_id] for p in selected
                if p.problem.sample_id in results]
        correct = sum(
            r.get("status") == "answered"
            and r.get("answer") == answers[r["sample_id"]] for r in rows)
        statuses = Counter(r["status"] for r in rows)
        statuses["not_run"] = len(selected) - len(rows)
        tools = sum(r.get("attempted_tool_calls", 0) for r in rows)
        successful = sum(r.get("succeeded_tool_calls", 0) for r in rows)
        return {
            "total_questions": len(selected), "recorded_questions": len(rows),
            "correct": correct,
            "accuracy_full_denominator": correct / len(selected) if selected else None,
            "accuracy_recorded_denominator": correct / len(rows) if rows else None,
            "legal_answers": sum(r.get("status") == "answered" for r in rows),
            "statuses": dict(statuses),
            "model_requests": sum(r.get("request_count", 0) for r in rows),
            "attempted_tool_calls": tools, "succeeded_tool_calls": successful,
            "tool_success_rate": successful / tools if tools else None,
            "tool_seconds": round(sum(r.get("tool_seconds", 0) for r in rows), 6),
        }

    return {
        "protocol": PROTOCOL,
        "complete": len(results) == len(inputs) and not any(
            r["status"] in _STOP_STATUSES for r in results.values()),
        "overall": section(inputs),
        "by_task": {
            task: section([p for p in inputs if p.problem.task_type == task])
            for task in ("Depth", "Distance")
        },
    }


def source_identity():
    local = Path(__file__).resolve().parent
    root = local.parents[1]
    files = list(local.glob("*.py"))
    # Freeze the reused implementations too, rather than relying on Git HEAD.
    for relative in (
        "small_image/datasets.py", "small_image/storage.py",
        "small_image/precompute.py", "small_image/tools.py",
        "schemas/tool.py", "schemas/tool_response.py", "schemas/authorization.py",
        "tools/registry.py", "tools/contract.py",
        "synthesis/bailian_client.py", "synthesis/request_audit.py",
        "synthesis/request_governor.py",
    ):
        files.append(root / "src" / "skill3d" / relative)
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(files) if not path.name.startswith("test_")
    }


def _resume_results(inputs, identities, output, *, for_resume=True):
    results = {}
    for public_input in inputs:
        sample_id = public_input.problem.sample_id
        directory = output / "episodes" / sample_id
        path = directory / "result.json"
        if path.exists():
            row = read_json(path)
            receipt = row.get("receipt_sha256")
            payload = {k: v for k, v in row.items() if k != "receipt_sha256"}
            if (row.get("sample_id") != sample_id
                    or row.get("input_identity") != identities[sample_id]
                    or digest(payload) != receipt):
                raise ValueError("saved result identity or receipt changed")
            if for_resume and row["status"] in _STOP_STATUSES:
                raise ValueError("previous stop requires diagnosis; automatic retry disabled")
            results[sample_id] = row
        elif for_resume and directory.exists() and any(directory.iterdir()):
            raise ValueError("unresolved episode attempt; automatic retry disabled")
    return results


def run_evaluation(inputs, prepared_index, identities, *, output, config, max_requests,
                   concurrency, base_url=None, api_key_env="DASHSCOPE_API_KEY"):
    with (output / "evaluation.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("an evaluator already owns this run directory") from None
        return _run_evaluation(
            inputs, prepared_index, identities, output=output, config=config,
            max_requests=max_requests, concurrency=concurrency,
            base_url=base_url, api_key_env=api_key_env)


def _run_evaluation(inputs, prepared_index, identities, *, output, config, max_requests,
                    concurrency, base_url, api_key_env):
    results = _resume_results(inputs, identities, output)
    # GT stays in the host evaluator. It is never passed to solve/client/dispatcher.
    dataset_root = read_json(output / "manifest.json")["dataset_root"]
    answers = load_evaluation_answers(
        dataset_root, "cv3d", inputs[0].problem.source_split)
    governor = RequestGovernor(
        output / "budget", RequestGovernorConfig(
            max_requests=max_requests, max_inflight=concurrency))
    check_client = NativeToolClient(config, base_url=base_url, api_key_env=api_key_env)
    if not check_client.configured:
        raise ValueError("model credentials are not configured")
    stop = threading.Event()
    pending = iter(p for p in inputs if p.problem.sample_id not in results)

    def worker(public_input):
        return solve(
            public_input, prepared_index[public_input.problem.input_sha256],
            client=NativeToolClient(config, base_url=base_url, api_key_env=api_key_env),
            governor=governor, output_directory=(
                output / "episodes" / public_input.problem.sample_id),
            stop_event=stop, expected_identity=identities[public_input.problem.sample_id])

    def publish():
        scores = summarize(inputs, answers, results)
        scores["budget"] = governor.summary()
        write_json(output / "scores.json", scores)
        print(json.dumps({
            "complete": scores["complete"], **scores["overall"],
            "reserved_requests": scores["budget"]["reserved_requests"],
        }, ensure_ascii=False), flush=True)
        return scores

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        active = {}

        def fill():
            while len(active) < concurrency and not stop.is_set():
                public_input = next(pending, None)
                if public_input is None:
                    break
                active[executor.submit(worker, public_input)] = public_input

        fill()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                public_input = active.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    stop.set()
                    row = {
                        "protocol": PROTOCOL,
                        "sample_id": public_input.problem.sample_id,
                        "task_type": public_input.problem.task_type,
                        "source_split": public_input.problem.source_split,
                        "input_identity": identities[public_input.problem.sample_id],
                        "status": "host_error", "answer": None,
                        "error_code": type(exc).__name__, "request_count": 0,
                    }
                    row["receipt_sha256"] = digest(row)
                    path = output / "episodes" / row["sample_id"] / "result.json"
                    if not path.exists():
                        write_json(path, row, exclusive=True)
                results[row["sample_id"]] = row
                if row["status"] in _STOP_STATUSES:
                    stop.set()
            publish()
            fill()
    return publish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--official-annotations", required=True, type=Path,
                        help="unmodified Hugging Face test_3d.jsonl")
    parser.add_argument("--prepared-root", type=Path,
                        help="existing smallimage/prepared directory; no GPU preparation")
    parser.add_argument("--output-dir", type=Path, help="independent experiment directory")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--check-prompts", action="store_true",
                        help="verify official prompts and split only; no cache or credentials")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--max-requests", type=int, help="total model-request budget")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--base-url", help="explicit OpenAI-compatible HTTPS /v1 endpoint")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    args = parser.parse_args(argv)
    if args.concurrency not in range(1, 5):
        parser.error("concurrency must be 1..4")
    if args.check_prompts and (args.execute or args.score_only):
        parser.error("--check-prompts cannot execute or score")
    if args.execute and args.score_only:
        parser.error("--score-only cannot execute")
    if not args.check_prompts and args.output_dir is None:
        parser.error("--output-dir is required")
    if not (args.check_prompts or args.score_only) and (
            args.prepared_root is None or args.max_requests is None or args.max_requests < 1):
        parser.error("--prepared-root and positive --max-requests are required")
    try:
        inputs = load_inputs(
            args.dataset_root, args.official_annotations, split=args.split,
            limit=args.limit, sample_ids=args.sample_id)
        if args.check_prompts:
            print(json.dumps({
                "status": "official_prompts_verified", "model_requests": 0,
                "questions": len(inputs), "split": args.split,
                "tasks": dict(Counter(p.problem.task_type for p in inputs)),
                "public_inputs_sha256": digest([p.identity() for p in inputs]),
            }))
            return 0
        output = args.output_dir.expanduser().resolve()
        if args.score_only:
            manifest = read_json(output / "manifest.json")
            if [p.identity() for p in inputs] != manifest["inputs"]:
                raise ValueError("score input selection differs from manifest")
            results = _resume_results(
                inputs, manifest["request_identities"], output, for_resume=False)
            answers = load_evaluation_answers(args.dataset_root, "cv3d", args.split)
            scores = summarize(inputs, answers, results)
            if (output / "budget/state.json").is_file():
                budget = manifest["budget"]
                scores["budget"] = RequestGovernor(
                    output / "budget", RequestGovernorConfig(
                        budget["max_requests"], budget["concurrency"])).summary()
            write_json(output / "scores.json", scores)
            print(json.dumps(scores, ensure_ascii=False))
            return 0
        if output.exists() and not args.resume:
            raise ValueError("output exists; use --resume to preserve history")
        if args.resume and not (output / "manifest.json").is_file():
            raise ValueError("resume requires the original manifest")
        prepared_index = index_prepared(args.prepared_root)
        config = InferenceConfig(timeout_s=args.timeout_s)
        client = NativeToolClient(config, base_url=args.base_url, api_key_env=args.api_key_env)
        identities = {}
        for public_input in inputs:
            key = public_input.problem.input_sha256
            if key not in prepared_index:
                raise ValueError("missing prepared input")
            _, _, _, _, identity = prepare_request(public_input, prepared_index[key])
            identities[public_input.problem.sample_id] = identity
        manifest = {
            "protocol": PROTOCOL, "dataset_root": str(args.dataset_root.expanduser().resolve()),
            "official_annotations_sha256": file_sha256(args.official_annotations),
            "inputs": [p.identity() for p in inputs], "request_identities": identities,
            "model": config.public_dict(), "endpoint_sha256": client.endpoint_hash,
            "budget": {"max_requests": args.max_requests, "concurrency": args.concurrency},
            "source_sha256": source_identity(),
            "tool_call_limit": None, "answer_sessions_per_question": 1,
            "generated_programs": False, "skill_loaded": False,
        }
        if args.resume:
            if read_json(output / "manifest.json") != manifest:
                raise ValueError("run identity changed during resume")
        else:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "manifest.json", manifest, exclusive=True)
            # Only one inspectable prompt is needed; all request hashes are in the manifest.
            first = inputs[0]
            _, messages, tools, png, identity = prepare_request(
                first, prepared_index[first.problem.input_sha256])
            preview = {
                "messages": [{
                    "role": "user", "content": [
                        part if part["type"] == "text" else {
                            "type": "image_url", "image_url": "see ordered image hashes in identity",
                        } for part in messages[0]["content"]
                    ],
                }],
                "tools": tools, "identity": identity,
            }
            write_json(output / "request_preview.json", preview, exclusive=True)
            (output / "depth_preview.png").write_bytes(png)
            write_json(output / "preflight.json", {
                "status": "preflight_passed", "model_requests": 0, "gpu_requests": 0,
                "questions": len(inputs), "manifest_sha256": digest(manifest),
            }, exclusive=True)
        if not args.execute:
            print(json.dumps({
                "status": "preflight_passed", "model_requests": 0, "questions": len(inputs),
                "manifest_sha256": digest(manifest),
            }))
            return 0
        scores = run_evaluation(
            inputs, prepared_index, identities, output=output, config=config,
            max_requests=args.max_requests, concurrency=args.concurrency,
            base_url=args.base_url, api_key_env=args.api_key_env)
        return 0 if scores["complete"] else 2
    except Exception as exc:
        print(json.dumps({
            "status": "stopped", "error_type": type(exc).__name__,
            "automatic_retry": False,
        }), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
