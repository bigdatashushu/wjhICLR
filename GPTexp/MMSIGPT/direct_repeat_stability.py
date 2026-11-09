"""Bounded repetitions of the identical original RGB request on its original API."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

from xcodeapi.config import Profile

from ..traces import credential_free, digest, file_digest, read, write
from .depth_comparison import runner
from .depth_comparison.pool import BASE_URL, MODEL, Client, inference
from .prototype.client import ServiceError
from .prototype.direct import operator_lock
from .prototype.inputs import load_images, public_problem
from .prototype.protocol import messages, parse_answer, payload

SOURCE = Path("/Users/bytedance/wangjh/MMSI深度图对比-20261107/run-recovered")
DESTINATION = SOURCE.parent / "direct-repeat-stability"
SAMPLES = ("MMSI_874", "MMSI_926")
REPEATS = 4


def settings(api):
    return {**inference(api), "timeout_seconds": 180}


class RepeatClient(Client):
    def __init__(self, api, key):
        super().__init__(api, key)
        self.inference = settings(api)
        self.profile = Profile("responses", BASE_URL, MODEL, "XCODE_GPT_API_KEY",
                               timeout_seconds=180)


def body(item):
    return payload(messages(item["problem"], load_images(item["problem"], item["images"])),
                   settings(item["api_id"]))


def code_identity():
    return {**runner.code_identity(),
            "MMSIGPT/direct_repeat_stability.py": file_digest(__file__)}


def selected(source, samples):
    manifest = read(Path(source) / "manifest.json")
    if (digest({k: v for k, v in manifest.items() if k != "run_identity"})
            != manifest["run_identity"] or manifest["reference_answer_visible"] is not False):
        raise ValueError("original_public_manifest_changed")
    originals = {i["problem"]["sample_id"]: i for i in manifest["items"]}
    items = []
    for index, sample in enumerate(samples):
        original = originals[sample]
        item = {
            "index": index, "source_index": original["index"],
            "problem": public_problem(original["problem"]), "images": original["images"],
            "api_id": original["api_id"], "payload_sha256": original["payload_sha256"]["direct"],
        }
        if digest(body(item)) != item["payload_sha256"]:
            raise ValueError("original_direct_payload_changed")
        items.append(item)
    return manifest, items


def prepare(directory=DESTINATION, source=SOURCE, *, samples=SAMPLES):
    directory, source = Path(directory), Path(source).resolve()
    original, items = selected(source, samples)
    if len(items) != 2 or len({i["api_id"] for i in items}) != 2:
        raise ValueError("requires_two_original_api_slots")
    if not directory.parent.is_dir() or directory.exists():
        raise ValueError("requires_new_directory_with_existing_parent")
    directory.mkdir()
    manifest = {
        "schema": "mmsi-original-direct-repeat/1", "source_directory": str(source),
        "source_run_identity": original["run_identity"],
        "source_manifest_sha256": file_digest(source / "manifest.json"),
        "items": items, "samples": list(samples), "repeats_per_question": REPEATS,
        "max_requests": 8, "max_inflight": 2,
        "inference_by_api": {i["api_id"]: settings(i["api_id"]) for i in items},
        "code_identity": code_identity(), "environment": runner.environment_identity(),
        "reference_answer_visible": False, "automatic_retry": False, "tools_enabled": False,
        "user_instruction": "对 MMSI_874 和 MMSI_926 用原图 API 重复运行几次，检查答案稳定性",
        "prepared_at": time.time(),
    }
    manifest["run_identity"] = digest(manifest)
    write(directory / "manifest.json", manifest, exclusive=True)
    write(directory / "preflight.json", {
        "run_identity": manifest["run_identity"], "model_requests": 0,
        "reference_answers_read": False, "original_rgb_bytes_and_payloads_verified": True,
        "depth_files_read": False, "repeats_per_question": 4, "max_requests": 8,
    }, exclusive=True)
    validate(directory)
    return manifest


def validate(directory):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    original, items = selected(manifest["source_directory"], manifest["samples"])
    preflight = read(directory / "preflight.json")
    if (manifest["schema"] != "mmsi-original-direct-repeat/1"
            or digest({k: v for k, v in manifest.items() if k != "run_identity"}) != manifest["run_identity"]
            or manifest["source_manifest_sha256"] != file_digest(
                Path(manifest["source_directory"]) / "manifest.json")
            or manifest["source_run_identity"] != original["run_identity"]
            or manifest["items"] != items or len(items) != 2
            or manifest["max_requests"] != 8 or manifest["max_inflight"] != 2
            or manifest["repeats_per_question"] != 4
            or manifest["inference_by_api"] != {i["api_id"]: settings(i["api_id"]) for i in items}
            or manifest["code_identity"] != code_identity()
            or manifest["environment"] != runner.environment_identity()
            or manifest["reference_answer_visible"] is not False
            or manifest["automatic_retry"] is not False or manifest["tools_enabled"] is not False
            or preflight["run_identity"] != manifest["run_identity"]
            or preflight["model_requests"] != 0 or preflight["reference_answers_read"] is not False):
        raise ValueError("repeat_configuration_changed")
    return manifest


def paths(directory, item, repeat):
    stem = Path(directory) / "traces" / f"{item['index']:04d}-r{repeat + 1:02d}"
    return Path(str(stem) + ".dispatch.json"), Path(str(stem) + ".result.json")


def execute(directory, keys, *, client_factory=RepeatClient):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory)
        if any((directory / name).exists() for name in (
                "execution.started.json", "traces", "frozen.json")):
            raise ValueError("no_automatic_restart_or_retry")
        clients = {i["api_id"]: client_factory(i["api_id"], keys[i["api_id"]])
                   for i in manifest["items"]}
        def public(value):
            for key in keys.values():
                credential_free(value, key)
        public(manifest)
        stop, guard = threading.Event(), threading.Lock()
        state = {"dispatched": 0, "inflight": 0, "peak_inflight": 0}
        write(directory / "execution.started.json", {
            "run_identity": manifest["run_identity"], "max_requests": 8, "max_inflight": 2,
            "reference_answer_visible": False, "automatic_retry": False,
            "credential_storage": "process_memory_only", "at": time.time(),
        }, exclusive=True)
        def slot(item):
            for repeat in range(REPEATS):
                if stop.is_set():
                    return
                request = body(item)
                public(request)
                if digest(request) != item["payload_sha256"]:
                    stop.set()
                    raise ValueError("request_changed_before_dispatch")
                dispatch_path, result_path = paths(directory, item, repeat)
                dispatch = {
                    "sample_id": item["problem"]["sample_id"], "index": item["index"],
                    "repeat": repeat + 1, "api_id": item["api_id"],
                    "input_sha256": item["problem"]["input_sha256"],
                    "payload_sha256": item["payload_sha256"], "requested": settings(item["api_id"]),
                    "run_identity": manifest["run_identity"], "reference_answer_visible": False,
                    "automatic_retry": False, "at": time.time(),
                }
                with guard:
                    if stop.is_set():
                        return
                    if state["dispatched"] >= 8:
                        raise ValueError("repeat_budget_exceeded")
                    write(dispatch_path, dispatch, exclusive=True)
                    state["dispatched"] += 1
                    state["inflight"] += 1
                    state["peak_inflight"] = max(state["peak_inflight"], state["inflight"])
                try:
                    response = clients[item["api_id"]].call(request)
                    public(response)
                    parsed = parse_answer(response["text"], len(item["problem"]["options"]))
                    result = {**parsed, "status": "answered" if parsed["answer"] else "invalid_answer",
                              "response": response}
                except ServiceError as exc:
                    stop.set()
                    result = {"status": "service_error", "answer": None, "parse_status": "not_parsed",
                              "error_code": exc.code, "http_status": exc.http_status}
                except Exception:
                    stop.set()
                    result = {"status": "service_error", "answer": None, "parse_status": "not_parsed",
                              "error_code": "client_outcome_unknown", "http_status": None}
                result.update({
                    **{k: dispatch[k] for k in ("sample_id", "index", "repeat", "api_id", "input_sha256",
                                              "run_identity", "reference_answer_visible", "automatic_retry")},
                    "dispatch_sha256": file_digest(dispatch_path), "at": time.time()})
                public(result)
                write(result_path, result, exclusive=True)
                with guard:
                    state["inflight"] -= 1
                print(json.dumps({k: result[k] for k in ("sample_id", "repeat", "status", "answer")}), flush=True)
        status = "interrupted"
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(slot, manifest["items"]))
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
    execution, started = [read(directory / name) for name in (
        "execution.json", "execution.started.json")]
    if (execution["run_identity"] != manifest["run_identity"] or execution["inflight"] != 0
            or execution["peak_inflight"] > 2
            or execution["status"] not in ("complete", "interrupted", "stopped_service_error")
            or started["run_identity"] != manifest["run_identity"] or started["max_requests"] != 8
            or started["max_inflight"] != 2):
        raise ValueError("repeat_execution_changed")
    files = {name: file_digest(directory / name) for name in (
        "manifest.json", "preflight.json", "execution.json", "execution.started.json")}
    records, known, dispatched = [], set(), 0
    for item in manifest["items"]:
        seen_gap = False
        for repeat in range(REPEATS):
            dispatch_path, result_path = paths(directory, item, repeat)
            known.update((dispatch_path.name, result_path.name))
            record = {"sample_id": item["problem"]["sample_id"], "repeat": repeat + 1,
                      "index": item["index"], "api_id": item["api_id"],
                      "input_sha256": item["problem"]["input_sha256"],
                      "status": "not_dispatched", "answer": None}
            if result_path.exists() and not dispatch_path.exists():
                raise ValueError("repeat_result_without_dispatch")
            if dispatch_path.exists():
                if seen_gap:
                    raise ValueError("repeat_sequence_gap")
                dispatch = read(dispatch_path)
                if (any(dispatch[k] != record[k] for k in (
                        "sample_id", "repeat", "index", "api_id", "input_sha256"))
                        or dispatch["payload_sha256"] != item["payload_sha256"]
                        or dispatch["requested"] != settings(item["api_id"])
                        or dispatch["run_identity"] != manifest["run_identity"]
                        or dispatch["reference_answer_visible"] is not False
                        or dispatch["automatic_retry"] is not False):
                    raise ValueError("repeat_dispatch_changed")
                dispatched += 1
                files[str(dispatch_path.relative_to(directory))] = file_digest(dispatch_path)
                record["status"] = "outcome_unknown"
            else:
                seen_gap = True
            if result_path.exists():
                result = read(result_path)
                if (any(result[k] != record[k] for k in (
                        "sample_id", "repeat", "index", "api_id", "input_sha256"))
                        or result["run_identity"] != manifest["run_identity"]
                        or result["dispatch_sha256"] != file_digest(dispatch_path)
                        or result["reference_answer_visible"] is not False
                        or result["automatic_retry"] is not False):
                    raise ValueError("repeat_result_changed")
                if result["status"] in ("answered", "invalid_answer"):
                    parsed = parse_answer(result["response"]["text"], len(item["problem"]["options"]))
                    if (parsed["answer"] != result["answer"] or parsed["parse_status"] != result["parse_status"]
                            or result["response"]["tool_call_count"] != 0
                            or result["status"] != ("answered" if parsed["answer"] else "invalid_answer")):
                        raise ValueError("repeat_parse_changed")
                elif result["status"] != "service_error" or result["answer"] is not None:
                    raise ValueError("invalid_repeat_status")
                files[str(result_path.relative_to(directory))] = file_digest(result_path)
                record.update(status=result["status"], answer=result["answer"])
            records.append(record)
    traces = directory / "traces"
    if traces.exists() and any(p.name not in known for p in traces.iterdir()):
        raise ValueError("unexpected_repeat_trace")
    if dispatched != execution["dispatched"] or dispatched > 8:
        raise ValueError("repeat_count_changed")
    if execution["status"] == "complete" and (dispatched != 8 or any(
            r["status"] not in ("answered", "invalid_answer") for r in records)):
        raise ValueError("complete_repeat_missing_results")
    return records, files


def freeze(directory):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate(directory)
        records, files = collect(directory, manifest)
        result = {"schema": "mmsi-original-direct-repeat-freeze/1",
                  "run_identity": manifest["run_identity"], "predictions": records,
                  "files": files, "reference_answer_visible": False, "frozen_at": time.time()}
        result["freeze_identity"] = digest(result)
        if (directory / "frozen.json").exists():
            return validate_frozen(directory, manifest)
        write(directory / "frozen.json", result, exclusive=True)
        return result


def validate_frozen(directory, manifest):
    frozen = read(Path(directory) / "frozen.json")
    records, files = collect(directory, manifest)
    if (digest({k: v for k, v in frozen.items() if k != "freeze_identity"}) != frozen["freeze_identity"]
            or frozen["run_identity"] != manifest["run_identity"]
            or frozen["predictions"] != records or frozen["files"] != files):
        raise ValueError("repeat_freeze_changed")
    return frozen


def export(directory):
    directory = Path(directory)
    manifest = validate(directory)
    frozen = validate_frozen(directory, manifest)
    cases = []
    for item in manifest["items"]:
        records = [r for r in frozen["predictions"] if r["index"] == item["index"]]
        answers = [r["answer"] for r in records]
        counts = Counter(r["answer"] for r in records if r["status"] == "answered")
        original_path = Path(manifest["source_directory"]) / "traces" / (
            f"{item['source_index']:04d}-direct.result.json")
        source_freeze = read(Path(manifest["source_directory"]) / "frozen.json")
        if (digest({k: v for k, v in source_freeze.items() if k != "freeze_identity"})
                != source_freeze["freeze_identity"] or source_freeze["files"].get(
                    str(original_path.relative_to(manifest["source_directory"]))) != file_digest(original_path)):
            raise ValueError("original_direct_result_changed")
        original = read(original_path)
        responses = [read(paths(directory, item, repeat)[1])
                     for repeat in range(REPEATS) if paths(directory, item, repeat)[1].exists()]
        cases.append({
            "sample_id": item["problem"]["sample_id"], "api_id": item["api_id"],
            "original_direct_answer": original["answer"], "new_answers": answers,
            "answer_counts": dict(counts), "status_counts": dict(Counter(r["status"] for r in records)),
            "all_four_same_valid_answer": len(counts) == 1 and sum(counts.values()) == 4,
            "matches_original": counts.get(original["answer"], 0),
            "options": item["problem"]["options"], "responses": responses,
        })
    execution = read(directory / "execution.json")
    report = {
        "schema": "mmsi-original-direct-repeat-summary/1",
        "run_identity": manifest["run_identity"], "freeze_identity": frozen["freeze_identity"],
        "execution_status": execution["status"], "new_requests": execution["dispatched"],
        "reference_answers_read": False, "original_predictions_read_only_after_freeze": True,
        "cases": cases, "summarized_at": time.time(),
    }
    report["summary_identity"] = digest(report)
    write(directory / "summary.json", report)
    lines = [
        "# MMSI 原图重复调用稳定性", "",
        "两题各新增 4 次调用，使用各自原 API。原图字节、图序、题目、选项、提示和请求"
        "payload 哈希与原 Direct 调用相同。每个 API 串行重复，两题并发；单次超时 180 秒，"
        "无自动重试。全部预测冻结后才读取原 Direct 答案作比较，未读取参考答案。", "",
        "| 题目 | API | 原 Direct | 新 4 次答案 | 新答案分布 | 与原答案一致 |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for case in cases:
        lines.append(f"| {case['sample_id']} | {case['api_id']} | {case['original_direct_answer']} | "
                     f"{', '.join(str(a) for a in case['new_answers'])} | "
                     f"{json.dumps(case['answer_counts'], ensure_ascii=False)} | {case['matches_original']}/4 |")
    lines += [
        "", "请求配置：gpt-5.6-terra，reasoning effort=high，temperature=0，seed=137，"
        "max_output_tokens=8192。返回的模型、采样元数据、reasoning tokens 及耗时逐次记录。", "",
        f"本次派发 {execution['dispatched']}/8 次，执行状态 {execution['status']}。",
        "同题重复出现不同答案说明原图回答存在波动；小样本全部相同仅表明本次调用窗口内一致。", "",
        "详细逐次结果在 traces/ 和 summary.json；原来的 18 题比较及其冻结记录保持不变。", "",
        f"Run identity：`{manifest['run_identity']}`", "",
        f"Freeze identity：`{frozen['freeze_identity']}`", "",
    ]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def controlled_execute(directory):
    from .depth_service_probe import all_keys
    from .depth_comparison.mac_control import ROOT, clean_environment
    keys = all_keys()
    result = subprocess.run(
        [sys.executable, "-B", "-m", "GPTexp.MMSIGPT.direct_repeat_stability",
         "execute", "--directory", str(directory)], input=json.dumps(keys).encode(),
        env=clean_environment(), cwd=ROOT, capture_output=True)
    for key in keys.values():
        if key.encode() in result.stdout or key.encode() in result.stderr:
            raise ValueError("credential_in_repeat_worker_output")
    print(result.stdout.decode(), end="", flush=True)
    if result.returncode:
        raise RuntimeError("repeat_worker_failed")
    export(directory)
    paths_to_scan = [p for p in Path(directory).rglob("*") if p.is_file()] + [Path(__file__)]
    if any(key.encode() in p.read_bytes() for p in paths_to_scan for key in keys.values()):
        raise ValueError("credential_in_repeat_artifact")
    write(Path(directory) / "verification.json", {
        "status": "passed", "credential_scan": "passed_all_11_keys",
        "files_scanned": len(paths_to_scan), "request_payloads_match_original": True,
        "reference_answers_read": False, "model_requests": read(Path(directory) / "execution.json")["dispatched"],
    }, exclusive=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "controlled-execute", "execute", "report"))
    parser.add_argument("--directory", default=str(DESTINATION))
    args = parser.parse_args()
    if args.phase == "prepare":
        manifest = prepare(args.directory)
        print(json.dumps({"status": "prepared", "max_requests": 8, "run_identity": manifest["run_identity"]}))
    elif args.phase == "controlled-execute":
        controlled_execute(args.directory)
    elif args.phase == "execute":
        execute(args.directory, json.load(sys.stdin))
        freeze(args.directory)
    else:
        report = export(args.directory)
        print(json.dumps([{k: c[k] for k in (
            "sample_id", "new_answers", "answer_counts", "matches_original")} for c in report["cases"]]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print('{"status":"failed","reason_code":"direct_repeat_phase_failed"}', file=sys.stderr)
        raise SystemExit(2)
