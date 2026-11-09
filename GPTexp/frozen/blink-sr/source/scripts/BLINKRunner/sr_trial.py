#!/usr/bin/env python3
"""Frozen 30/70 val split and 20-question paired SR trial with full JSON traces."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import multiprocessing
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr
from sr_adapter import FullRecordedClient, FullResponseClient
from sr_protocol import TOOLS, relation_for
from skill3d.small_image.blink_protocol import sha_value, strict_json
from skill3d.small_image.storage import save_prepared

SIZE, SEED, CAP, CONCURRENCY = 20, 137, 80, 8


def now():
    return datetime.now(timezone.utc).isoformat()


def split_manifest(data_root):
    data, items = sr.load_inputs(data_root, ("val",))
    ids = [p.sample_id for p, _ in items]
    rng = random.Random(SEED)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    train = sorted(shuffled[:43], key=ids.index)
    test = sorted(shuffled[43:], key=ids.index)
    diagnostic = sorted(rng.sample(test, SIZE), key=ids.index)
    return {
        "schema": "blink-sr-val-split/1", "dataset_revision": data["resolved_revision"],
        "input_index_sha256": data["input_index_sha256"], "seed": SEED,
        "official_source_split": "val", "train_fraction": .3, "test_fraction": .7,
        "rounding": "train=round(143*0.30)=43, internal_test=100",
        "train": train, "internal_test": test, "diagnostic20": diagnostic,
        "train_questions": 43, "test_questions": 100, "diagnostic_questions": SIZE,
        "selection": "uniform public-only shuffle then sample; no label stratification",
        "training_evolution": False, "reference_answers_read": False,
        "original_inputs": [
            {"sample_id": p.sample_id, "question_sha256": sha_value(p.question),
             "image_sha256": sr.image_identities(p)[0]["sha256"]}
            for p, _ in items],
    }


def selection(root):
    frozen = sr.read(Path(root) / "split.json")
    actual = split_manifest(Path(root) / "dataset")
    if {k: v for k, v in frozen.items() if k != "content_sha256"} != actual:
        raise ValueError("sr_frozen_selection_changed")
    return frozen


def freeze(root, *, parent=None, direct_seal=None):
    root = Path(root).resolve()
    split = selection(root)
    ready = sr.check(root / "dataset", ("val",), prepared_root=root / "prepared",
                     sample_ids=split["diagnostic20"])
    delivery = check_actual_initial_prompts(root, ready)
    jobs = []
    skill = sr.SKILL.read_text()
    for i, item in enumerate(ready["inputs"]):
        for arm in ("direct", "skill_tools"):
            spec = {"arm": arm, "problem": item["problem"],
                    "solver": ready["configuration"]["solver"],
                    "official_prompt": item["official_prompt"]}
            if arm == "skill_tools":
                spec.update(prepared_root=str(root / "prepared"), skill_text=skill,
                            artifact_sha256=item["prepared_artifact_sha256"])
            prefix = "sr-corrected" if parent is not None and arm == "skill_tools" else "sr"
            jobs.append({"operation_id": f"{prefix}-{i:03d}-{arm}", "spec": spec})
    continuation = None
    if parent is not None:
        from sr_continuation import seed
        old = sr.read(Path(parent) / "manifest.json")
        previous = old["inspection"]
        if (old["split_sha256"] != split["content_sha256"]
                or previous["configuration"] != ready["configuration"]
                or previous["skill_sha256"] != ready["skill_sha256"]
                or {k: v for k, v in previous["sources"].items()
                    if not k.startswith("scripts/BLINKRunner/sr")} !=
                {k: v for k, v in ready["sources"].items()
                    if not k.startswith("scripts/BLINKRunner/sr")}
                or any({k: v for k, v in x.items() if k not in {
                    "skill_initial_payload_sha256", "available_tools"}} !=
                       {k: v for k, v in y.items() if k not in {
                    "skill_initial_payload_sha256", "available_tools"}}
                       for x, y in zip(previous["inputs"], ready["inputs"]))):
            raise ValueError("sr_continuation_unrelated_experiment_change")
        continuation = seed(root, parent, direct_seal, jobs)
    manifest = sr.write_once(root / "manifest.json", {
        "schema": "blink-sr-paired20/1", "inspection": ready, "jobs": jobs,
        "split_sha256": split["content_sha256"], "request_cap": CAP,
        "concurrency": CONCURRENCY, "created_at": now(),
        "evaluation_role": "internal_test_diagnostic", "initial_skill_frozen": True,
        "training_evolution": False, "reference_answer_visible": False,
        "automatic_retry": False,
        **({"continuation_sha256": continuation["content_sha256"]} if continuation else {}),
    })
    return sr.write_once(root / "preparation.json", {
        "manifest_sha256": manifest["content_sha256"], "model_requests": 0,
        "reference_answers_read": False, "questions_per_arm": SIZE,
        "new_episodes": SIZE if continuation else SIZE * 2,
        "reused_direct_episodes": SIZE if continuation else 0,
        "zero_request_delivery_sha256": delivery["content_sha256"],
    })


def check_actual_initial_prompts(root, ready):
    """Intercept the actual solver transport for every item without any API call."""
    class CaptureOnly(BaseException):
        pass

    class Capture:
        def __init__(self):
            self.base = FullResponseClient(**ready["configuration"]["model"])
            self.payload = None
        def __getattr__(self, name):
            return getattr(self.base, name)
        def chat(self, messages, **kwargs):
            self.payload = self.base.build_payload(messages, **kwargs)
            raise CaptureOnly()

    checked = []
    for item in ready["inputs"]:
        public = item["problem"]
        p = sr.SmallImageProblem(**{
            k: tuple(v) if k in {"options", "image_paths"} else v
            for k, v in public.items() if k != "input_sha256"})
        prepared = sr.prepared_for(p, Path(root) / "prepared")
        client = Capture()
        solver = sr.SmallImageSolver(
            client, config=sr.solver_config(ready["configuration"]["solver"]),
            registry_builder=sr.scoped_registry, skill_text=sr.SKILL.read_text(),
            official_prompt=item["official_prompt"])
        try:
            solver.solve(p, prepared=prepared, categories=prepared.categories)
        except CaptureOnly:
            pass
        if (client.payload is None
                or sha_value(client.payload) != item["skill_initial_payload_sha256"]
                or set(item["available_tools"]) != TOOLS):
            raise ValueError("sr_actual_initial_prompt_preflight_mismatch")
        checked.append({"sample_id": p.sample_id,
                        "actual_payload_sha256": sha_value(client.payload),
                        "tools": item["available_tools"]})
    return sr.write_once(Path(root) / "zero_request_delivery.json", {
        "checked": checked, "model_requests": 0, "reference_answers_read": False,
    })


def verify(root, manifest):
    if selection(root)["content_sha256"] != manifest["split_sha256"]:
        raise ValueError("sr_split_changed")
    ready = manifest["inspection"]
    if sr.source_identity() != ready["sources"] or sr.file_sha(sr.SKILL) != ready["skill_sha256"]:
        raise ValueError("sr_frozen_source_changed")
    if sr.check(root / "dataset", ("val",), prepared_root=root / "prepared",
                sample_ids=[row["problem"]["sample_id"] for row in ready["inputs"]]) != ready:
        raise ValueError("sr_preflight_changed")


def worker(root, job, model):
    root = Path(root)
    governor = sr.RequestGovernor(root / "budget", sr.RequestGovernorConfig(CAP, CONCURRENCY))
    client = FullRecordedClient(FullResponseClient(**model), governor, root,
                                job["operation_id"], "solver")
    # Public job only: no host reference path, GT, other samples, or other arm output.
    result = sr.solve_one(job["spec"], client)
    result.update(request_trace=client.request_trace,
                  execution_trace=result.get("execution_trace", []),
                  evaluation_role="internal_test_diagnostic")
    sr.write_once(root / "episodes" / (job["operation_id"] + ".json"), {
        "job_sha256": sha_value(job), "result": result,
    })
    return {"operation_id": job["operation_id"], "status": result["status"]}


def run(root):
    root = Path(root).resolve()
    manifest = sr.read(root / "manifest.json")
    verify(root, manifest)
    with (root / "active.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        governor = sr.RequestGovernor(root / "budget", sr.RequestGovernorConfig(CAP, CONCURRENCY))
        selected, continuation = initial_jobs(root, manifest, governor)
        if (root / "paid_run.json").exists():
            raise ValueError("sr_paid_run_replay_forbidden")
        model = manifest["inspection"]["configuration"]["model"]
        if not os.environ.get(model["api_key_env"], ""):
            raise ValueError("sr_missing_transient_auth")
        sr.write_once(root / "paid_run.json", {
            "at": now(), "manifest_sha256": manifest["content_sha256"],
            "new_episodes": len(selected), "automatic_retry": False,
        })
        jobs = iter(selected)
        results, pending = [], {}
        failure = False
        with ProcessPoolExecutor(max_workers=CONCURRENCY,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            for _ in range(CONCURRENCY):
                job = next(jobs, None)
                if job is not None:
                    pending[pool.submit(worker, str(root), job, model)] = job
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    job = pending.pop(future)
                    try:
                        value = future.result()
                    except BaseException as exc:
                        value = {"operation_id": job["operation_id"], "status": "worker_error",
                                 "error_type": type(exc).__name__}
                    results.append(value)
                    failure |= value["status"] in sr.OPERATIONAL | {"worker_error"}
                    print(__import__("json").dumps({"at": now(), "finished": len(results),
                                                    **value}), flush=True)
                if not failure:
                    for _ in done:
                        job = next(jobs, None)
                        if job is not None:
                            pending[pool.submit(worker, str(root), job, model)] = job
        budget = governor.summary()
        if failure or any(budget[k] for k in ("error_requests", "unknown_requests", "active_requests")):
            sr.write_once(root / "stopped.json", {
                "at": now(), "results": results, "budget": budget, "automatic_retry": False,
            })
            raise ValueError("sr_operational_failure_stopped_without_retry")
        if len(results) != len(selected):
            raise ValueError("sr_incomplete_trial")
        return sr.write_once(root / "completed.json", {
            "at": now(), "manifest_sha256": manifest["content_sha256"],
            "episodes": SIZE * 2, "new_episodes": len(results), "budget": budget,
            "reused_direct_episodes": SIZE if continuation else 0,
            "reference_answers_read": False, "automatic_retry": False,
        })


def initial_jobs(root, manifest, governor):
    if "continuation_sha256" in manifest:
        from sr_continuation import verify as verify_continuation
        continuation = verify_continuation(root, manifest, initial=True)
        selected = [j for j in manifest["jobs"] if j["spec"]["arm"] == "skill_tools"]
        if {j["operation_id"] for j in selected} != set(continuation["new_operations"]):
            raise ValueError("sr_continuation_jobs_changed")
        return selected, continuation
    if governor.summary()["reserved_requests"] or (Path(root) / "requests").exists():
        raise ValueError("sr_paid_run_replay_forbidden")
    return manifest["jobs"], None


def precompute_shard(root, config_path, shard):
    root = Path(root).resolve()
    split = selection(root)
    import yaml
    from sr_native import build_precomputer
    config = yaml.safe_load(Path(config_path).read_text())["precompute"]
    _, items = sr.load_inputs(root / "dataset", ("val",))
    by_id = {p.sample_id: p for p, _ in items}
    precomputer = build_precomputer(config, seed=SEED)
    receipts = []
    for sample_id in split["diagnostic20"][shard::2]:
        p = by_id[sample_id]
        relation = relation_for(p.question)
        target = root / "prepared" / p.input_sha256
        if target.exists():
            prepared = sr.prepared_for(p, root / "prepared")
        else:
            prepared = precomputer.process(p, categories=tuple(dict.fromkeys(
                (relation["subject"], relation["target"]))))
            if prepared.errors or not prepared.has_relative_depth:
                sr.write_once(root / f"precompute-failed-{shard}.json", {
                    "sample_id": sample_id, "error_codes": prepared.errors,
                    "model_requests": 0, "automatic_retry": False,
                })
                raise ValueError("sr_precompute_failed")
            save_prepared(prepared, target)
            sr.prepared_for(p, root / "prepared")
        receipts.append({"sample_id": sample_id, "input_sha256": p.input_sha256,
                         "artifact_sha256": prepared.artifact_sha256,
                         "categories": list(prepared.categories),
                         "candidates": len(prepared.candidates),
                         "has_relative_depth": prepared.has_relative_depth})
        print(__import__("json").dumps({"finished": len(receipts), "sample_id": sample_id}),
              flush=True)
    return sr.write_once(root / f"precompute-{shard}.json", {
        "at": now(), "shard": shard, "physical_gpu": 3 + shard,
        "config_sha256": sr.file_sha(config_path), "inputs": receipts,
        "reference_answers_read": False, "model_requests": 0,
    })


def original_content(content):
    """Validate the shared original_messages frame marker separately from prompt."""
    if (not isinstance(content, list) or len(content) != 3
            or [x.get("type") for x in content] != ["text", "text", "image_url"]
            or set(content[0]) != {"type", "text"}
            or content[1] != {"type": "text", "text": "Image 1:"}
            or set(content[2]) != {"type", "image_url"}
            or set(content[2]["image_url"]) != {"url"}):
        raise ValueError("sr_original_message_framing_changed")
    return content[0]["text"], [content[2]]


def audit_and_report(root, host_data_root, output):
    """After solving, independently verify every payload/image/tool receipt and score."""
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest, completed = sr.read(root / "manifest.json"), sr.read(root / "completed.json")
    if completed["manifest_sha256"] != manifest["content_sha256"] or completed["episodes"] != 40:
        raise ValueError("sr_trial_completion_changed")
    split = sr.read(root / "split.json")
    if split["content_sha256"] != manifest["split_sha256"]:
        raise ValueError("sr_trial_split_changed")
    records = {}
    continuation = None
    if "continuation_sha256" in manifest:
        from sr_continuation import verify as verify_continuation
        continuation = verify_continuation(root, manifest)
    for job in manifest["jobs"]:
        receipt = sr.read(root / "episodes" / (job["operation_id"] + ".json"))
        result, public = receipt["result"], job["spec"]["problem"]
        if (receipt["job_sha256"] != sha_value(job)
                or result["input_sha256"] != public["input_sha256"]
                or result["sample_id"] != public["sample_id"]
                or result["source_split"] != "val"
                or result["reference_answer_visible"] is not False):
            raise ValueError("sr_episode_identity_audit_failed")
        arm = job["spec"]["arm"]
        trace = result["request_trace"]
        if len(trace) != result["request_count"] or not 1 <= len(trace) <= (1 if arm == "direct" else 3):
            raise ValueError("sr_request_denominator_failed")
        item = next(x for x in manifest["inspection"]["inputs"] if x["problem"]["sample_id"] == public["sample_id"])
        for index, request in enumerate(trace, 1):
            stem = root / "requests" / job["operation_id"] / f"{index:03d}"
            intent = sr.read(stem.with_suffix(".intent.json"))
            full = sr.read(stem.with_suffix(".full.json"))
            payload_receipt = sr.read(stem.with_suffix(".payload.json"))
            response_receipt = sr.read(stem.with_suffix(".response.json"))
            payload = request["payload"]
            if (sha_value(payload) != intent["payload_sha256"]
                    or {k: v for k, v in full.items() if k != "content_sha256"} != request
                    or payload_receipt["payload"] != payload
                    or response_receipt["status"] != "received"
                    or response_receipt["model_response"] != request["response_json"][
                        "choices"][0]["message"]["content"]
                    or payload["seed"] != SEED or payload["temperature"] != 0
                    or payload["max_tokens"] != 8192 or payload["enable_thinking"] is not True
                    or payload["model"] != "qwen3.8-flash"
                    or request["response_json"]["model"] != "qwen3.8-flash"):
                raise ValueError("sr_request_policy_audit_failed")
            content = payload["messages"][0]["content"]
            text, images = original_content(content)
            import hashlib
            image_sha = hashlib.sha256(base64.b64decode(
                images[0]["image_url"]["url"].split(",", 1)[1])).hexdigest()
            if len(images) != 1 or image_sha != item["images"][0]["sha256"]:
                raise ValueError("sr_original_image_payload_audit_failed")
            if job["spec"]["official_prompt"] not in text:
                raise ValueError("sr_official_question_delivery_failed")
            if arm == "direct":
                if text != job["spec"]["official_prompt"] + sr.DIRECT_SUFFIX:
                    raise ValueError("sr_direct_extra_evidence")
            elif job["spec"]["skill_text"] not in text:
                raise ValueError("sr_skill_delivery_audit_failed")
            if index == 1 and sha_value(payload) != item[
                    "direct_payload_sha256" if arm == "direct" else "skill_initial_payload_sha256"]:
                raise ValueError("sr_initial_payload_changed")
        calls = [x for cell in result["execution_trace"] for x in cell["tool_calls"]]
        if arm == "direct" and (calls or result["attempted_tool_calls"]):
            raise ValueError("sr_direct_tool_call")
        if len(calls) != result["attempted_tool_calls"]:
            raise ValueError("sr_full_tool_trace_denominator_failed")
        for cell in result["execution_trace"]:
            if cell["observation_reads"]:
                raise ValueError("sr_unexpected_observation_reader")
            for call in cell["tool_calls"]:
                if call.get("tool", call.get("name")) not in TOOLS:
                    raise ValueError("sr_unallowed_tool")
            for tool_result in cell["tool_results"]:
                auth = tool_result["authorization"]
                if not auth or auth not in cell["authorization_receipts"]:
                    raise ValueError("sr_tool_authorization_missing")
        records[(arm, public["sample_id"])] = result
    if len(records) != 40 or {sid for _, sid in records} != set(split["diagnostic20"]):
        raise ValueError("sr_paired_denominator_failed")
    budget = completed["budget"]
    checked_payloads = sum(r["request_count"] for r in records.values())
    excluded = continuation["excluded_parent_tool_requests"] if continuation else 0
    state = strict_json((root / "budget/state.json").read_text())
    current_ops = {j["operation_id"] for j in manifest["jobs"]}
    if (budget["reserved_requests"] != checked_payloads + excluded
            or budget["reserved_requests"] != len(state["requests"])
            or any(row["status"] != "completed" for row in state["requests"])
            or any(sum(row["context"].get("variant_id") == job["operation_id"]
                       for row in state["requests"]) !=
                   records[(job["spec"]["arm"], job["spec"]["problem"]["sample_id"])]["request_count"]
                   for job in manifest["jobs"])
            or sum(row["context"].get("variant_id") not in current_ops
                   for row in state["requests"]) != excluded
            or budget["reserved_requests"] > CAP
            or any(budget[k] for k in ("error_requests", "unknown_requests", "active_requests"))):
        raise ValueError("sr_global_budget_audit_failed")
    # This is the first reference access in this process, after all trace checks.
    data, items = sr.load_inputs(host_data_root, ("val",))
    references = sr.load_references(host_data_root, data)
    by_id = {p.sample_id: p for p, _ in items}
    pairs, groups = [], {}
    for sid in split["diagnostic20"]:
        pair = {"sample_id": sid, "source_split": "val",
                "evaluation_role": "internal_test_diagnostic", "reference": references[sid],
                "question": by_id[sid].question, "relation": relation_for(by_id[sid].question)}
        for arm in ("direct", "skill_tools"):
            result = records[(arm, sid)]
            pair[arm] = {"correct": result["status"] == "answered" and result["answer"] == references[sid],
                         "trajectory": result}
        pairs.append(pair)
    for arm in ("direct", "skill_tools"):
        results = [records[(arm, sid)] for sid in split["diagnostic20"]]
        groups[arm] = {
            "questions": SIZE, "correct": sum(p[arm]["correct"] for p in pairs),
            "accuracy": sum(p[arm]["correct"] for p in pairs) / SIZE,
            "legal_answers": sum(r["status"] == "answered" for r in results),
            "statuses": dict(Counter(r["status"] for r in results)),
            "requests": sum(r["request_count"] for r in results),
            "attempted_tool_calls": sum(r["attempted_tool_calls"] for r in results),
            "succeeded_tool_calls": sum(r["succeeded_tool_calls"] for r in results),
            "family_scores": {
                f: {"questions": len(ps), "correct": sum(p[arm]["correct"] for p in ps)}
                for f in sorted({p["relation"]["family"] for p in pairs})
                for ps in [[p for p in pairs if p["relation"]["family"] == f]]},
        }
    agreement = Counter(
        "both_correct" if p["direct"]["correct"] and p["skill_tools"]["correct"] else
        "direct_only" if p["direct"]["correct"] else
        "skill_only" if p["skill_tools"]["correct"] else "both_wrong" for p in pairs)
    report = sr.write_once(output / "results_and_full_trajectories.json", {
        "schema": "blink-sr-paired20-report/1", "at": now(), "split": split,
        "manifest_sha256": manifest["content_sha256"], "model": "qwen3.8-flash",
        "groups": groups, "paired_outcomes": dict(agreement), "budget": budget,
        "continuation": continuation,
        "audit": {"status": "passed", "checked_episodes": 40,
                  "checked_payloads": checked_payloads,
                  "excluded_parent_tool_requests": excluded,
                  "audit_processor_sha256": sr.file_sha(Path(__file__)),
                  "original_image_bytes": True, "same_question_pairs": True,
                  "reference_answer_visible_during_solving": False,
                  "full_program_tool_and_authorization_trace": True, "automatic_retry": False},
        "pairs": pairs,
    })
    direct, skill = groups["direct"], groups["skill_tools"]
    text = (
        "# BLINK Spatial Relation：20 道内部测试对照\n\n"
        "从官方 val 143 道按 seed=137 随机划分：训练 43 道、内部测试 100 道；"
        "内部测试均匀抽 20 道，两路使用完全相同的原始问题与图片。训练集本轮未做 Skill 演化。\n\n"
        "| 方法 | 正确 | 准确率 | 合法答案 | 模型请求 | 成功工具调用 |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| Qwen3.8-Flash 直跑 | {direct['correct']}/20 | {direct['accuracy']:.0%} | "
        f"{direct['legal_answers']}/20 | {direct['requests']} | 0 |\n"
        f"| 当前 SR Skill + 工具架构 | {skill['correct']}/20 | {skill['accuracy']:.0%} | "
        f"{skill['legal_answers']}/20 | {skill['requests']} | {skill['succeeded_tool_calls']} |\n\n"
        f"架构相对直跑：{(skill['accuracy']-direct['accuracy'])*100:+.0f} 个百分点。"
        f"配对结果：{dict(agreement)}。\n\n"
        "两路统一 thinking=true、temperature=0、seed=137、8192 输出 tokens；"
        "直跑每题 1 请求，架构最多 2 轮程序和 1 轮最终回答。"
        "模型并发 8；预计算使用 3090 物理卡 3/4、DINO、SAM2、Intern 类别审核、MoGe2 相对深度。\n\n"
        "全部 40 条 episode 和请求/原图/Skill/工具授权/预算独立核验通过。"
        "完整 JSON 包含实际请求、供应商原始响应与思考内容、程序、执行错误、工具输入输出、"
        "授权回执及主机评分。原始逐文件回执保存在 raw/ 中。\n\n"
        + (f"工具目录修正后复用已核验的 20 条直跑结果，只补跑工具组。"
           f"修正前工具组的 {excluded} 次请求保留在上一版原始归档，不计入当前成绩；"
           f"累计实际请求 {budget['reserved_requests']}/{CAP}，本次比较使用 {checked_payloads} 次。"
           "修正发生在读取答案和评分之前。\n\n" if continuation else "")
        +
        "这是固定初始 Skill 在 20 道题上的诊断结果；训练演化后的效果和完整 100 道测试效果仍需单独实验。\n"
    )
    with (output / "report.md").open("x") as stream:
        stream.write(text)
    return {k: report[k] for k in ("groups", "paired_outcomes", "audit")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["split", "freeze", "run", "precompute", "report"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--shard", type=int, choices=[0, 1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--direct-seal", type=Path)
    args = parser.parse_args()
    if args.action == "split":
        value = sr.write_once(args.output, split_manifest(args.data_root))
    elif args.action == "freeze":
        value = freeze(args.root, parent=args.continue_from, direct_seal=args.direct_seal)
    elif args.action == "run":
        value = run(args.root)
    elif args.action == "precompute":
        value = precompute_shard(args.root, args.config, args.shard)
    else:
        value = audit_and_report(args.root, args.data_root, args.output)
    print(__import__("json").dumps({k: v for k, v in value.items()
                                     if k not in {"original_inputs", "pairs", "inputs",
                                                  "train", "internal_test", "diagnostic20"}},
                                    ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(__import__("json").dumps({"status": "stopped", "error_type": type(exc).__name__}))
        raise SystemExit(1)
