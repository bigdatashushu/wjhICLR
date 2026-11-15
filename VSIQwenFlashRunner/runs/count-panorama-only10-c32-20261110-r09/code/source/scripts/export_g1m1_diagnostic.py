#!/usr/bin/env python3
"""Audit completed diagnostic outputs, score offline and export readable JSON."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys


NAMES = {
    "obj_appearance_order": "出现顺序", "object_abs_distance": "绝对距离",
    "object_counting": "物体计数", "object_rel_direction": "相对方向",
    "object_rel_distance": "相对距离", "object_size_estimation": "物体尺寸",
    "room_size_estimation": "房间面积", "route_planning": "路线规划",
}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def write_collection(path, metadata, question_paths):
    """Keep full trajectories while loading only one question at a time."""
    with path.open("w") as stream:
        stream.write("{\n")
        for key, value in sorted(metadata.items()):
            stream.write("  " + json.dumps(key) + ": ")
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write(",\n")
        stream.write('  "questions": [\n')
        for index, question_path in enumerate(question_paths):
            if index:
                stream.write(",\n")
            json.dump(read(question_path), stream, ensure_ascii=False,
                      sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n  ]\n}\n")


def review_value(value, source_file, pointer=""):
    """Point large tool data to its exact location in the full JSON."""
    if (isinstance(value, str) and len(value) > 65536
            and pointer.rsplit("/", 1)[-1] not in {"response_text", "program_source"}):
        return {"large_text_reference": {
            "source_file": source_file, "json_pointer": pointer,
            "character_count": len(value), "sha256": hashlib.sha256(value.encode()).hexdigest(),
            "preview": value[:400],
        }}
    if isinstance(value, list):
        if len(value) > 128:
            return {"large_array_reference": {
                "source_file": source_file, "json_pointer": pointer,
                "item_count": len(value),
            }}
        return [review_value(item, source_file, pointer + "/" + str(index))
                for index, item in enumerate(value)]
    if isinstance(value, dict):
        return {key: review_value(
            item, source_file,
            pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
            for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    sys.path.insert(0, str(root / "code/source/src"))
    from skill3d.evaluation.fitset_v11 import (
        _variant_key, content_sha256, load_label_metadata, score_answer,
    )
    experiment = read(root / "diagnostic.json")
    system = root / "system"
    manifest, summary = read(system / "manifest.json"), read(system / "summary.json")
    assert content_sha256({k: v for k, v in manifest.items()
                          if k != "identity_sha256"}) == manifest["identity_sha256"]
    assert summary["status"] == "complete" and summary["completed_variants"] == 40
    assert summary["pending_variants"] == 0
    assert manifest["reference_answer_visible_to_online_model"] is False
    assert manifest["online_executor_input_is_label_free"] is True
    rows = [read(path) for path in sorted((system / "results").glob("*.json"))]
    expected = {row["variant_id"] for row in experiment["variants"]}
    assert {row["variant_id"] for row in rows} == expected and len(rows) == 40
    scheduler = read(system / "episode_pool_receipt.json")
    assert scheduler["status"] == "drained"
    assert set(scheduler["completed_variant_ids"]) == expected
    assert scheduler["run_identity"] == manifest["identity_sha256"]
    assert Counter(row["question_type"] for row in rows) == experiment["task_counts"]
    labels = load_label_metadata(root / "inputs/host_evaluation/metadata.jsonl")
    questions = {row["qa_id"]: row for row in read(root / "inputs/sample_questions.json")}
    requests = read(root / "requests/state.json")
    baseline_campaigns = {experiment["identity_sha256"]}
    if (root / "pair_completion.json").exists():
        pair = read(root / "pair_completion.json")
        continuation = read(root / "direct_continuation/diagnostic.json")
        assert pair["status"] == "complete"
        assert pair["diagnostic_identity"] == experiment["identity_sha256"]
        assert pair["continuation_identity"] == continuation["identity_sha256"]
        assert content_sha256({k: v for k, v in continuation.items()
                              if k != "identity_sha256"}) == continuation["identity_sha256"]
        assert continuation["direct_continuation"]["origin_identity"] == experiment["identity_sha256"]
        baseline_campaigns.add(continuation["identity_sha256"])
    assert not any(row["status"] != "completed" for row in requests["requests"])
    assert len(requests["requests"]) <= experiment["runtime"]["max_requests"]
    assert requests["peak_inflight"] <= requests["policy"]["max_inflight"]
    assert len({row["request_id"] for row in requests["requests"]}) == len(requests["requests"])
    request_by_variant = defaultdict(list)
    system_requests = []
    direct_requests = []
    for row in requests["requests"]:
        assert row["category"] == "solver"
        if experiment.get("comparison") and row["context"]["variant_id"].startswith("baseline:"):
            assert row["context"]["campaign_id"] in baseline_campaigns
            assert row["exception_type"] is None
            direct_requests.append(row)
            continue
        assert row["context"]["variant_id"] in expected
        assert row["context"]["run_identity"] == manifest["identity_sha256"]
        assert row["exception_type"] is None
        assert row["completed_at"] >= row["started_at"]
        request_by_variant[row["context"]["variant_id"]].append(row)
        system_requests.append(row)
    if experiment.get("comparison"):
        assert len(direct_requests) == 40
        assert len(system_requests) <= experiment["comparison"]["system_request_cap"]
    trajectories, reviews, scored, audits = [], [], [], []
    for result in sorted(rows, key=lambda row: (row["question_type"], row["qa_id"])):
        qa, variant = result["qa_id"], result["variant_id"]
        assert 1 <= len(request_by_variant[variant]) <= 7
        outcome_root = system / "variants" / _variant_key(variant)
        outcome = read(outcome_root / "outcome.json")
        preflight_path = system / "preflight" / (_variant_key(variant) + ".json")
        preflight = read(preflight_path) if preflight_path.exists() else None
        if preflight is not None:
            assert preflight["runtime_evidence_state"] == "g1m1"
        assert outcome["initial_skill_selection"]["evidence_state"] == "g1m1"
        assert outcome["scoring_status"] in {"reference_unavailable", "incomplete"}
        assert outcome["score"] is None and outcome["correct"] is None
        assert outcome["mra_value"] is None
        label = labels[qa]
        assert label["scene_id"] == result["scene_id"]
        assert label["question_type"] == result["question_type"]
        evaluation = score_answer(
            answer=result["answer"], task=result["question_type"], options=label["options"],
            reference=label["reference_answer"],
            available=result["status"] == "completed" and result["final_state"] == "answer"
            and not result["service_errors"],
        )
        topics = {}
        for path in sorted((outcome_root / "trace").glob("*.jsonl")):
            topics[path.stem] = [json.loads(line) for line in path.read_text().splitlines() if line]
        assert topics and outcome["rounds"]
        assert all("response_text" in row and "program_source" in row
                   for row in outcome["rounds"])
        receipt_files = {}
        for path in sorted((system / "request_seeds").glob(_variant_key(variant) + "*.json")):
            receipt_files[path.name] = read(path)
        request_receipts = {
            row["request_index"]: row for row in receipt_files.values()
            if "request_identity" in row
        }
        response_receipts = {
            row["request_index"]: row for row in receipt_files.values()
            if "request_identity_sha256" in row
        }
        indices = set(range(1, len(request_by_variant[variant]) + 1))
        assert set(request_receipts) == set(response_receipts) == indices
        assert result["model_request_count"] == len(indices)
        for index in indices:
            request, response = request_receipts[index], response_receipts[index]
            assert request["request_seed"] == response["request_seed"] == 137
            assert request["retry_of"] is None
            assert response["status"] == "completed" and response["exception_type"] is None
            assert response["request_identity_sha256"] == content_sha256(request["request_identity"])
            if experiment["runtime"].get("enable_thinking"):
                assert request["request_identity"]["parameters"]["enable_thinking"] is True
                assert response["actual_request_identity"]["parameters"]["enable_thinking"] is True
        usage = Counter()
        for request in request_by_variant[variant]:
            for key, value in request["usage"].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] += value
        item = {
            "schema": "vsi-g1m1-question-trajectory/1",
            "qa_id": qa, "variant_id": variant, "question_type": result["question_type"],
            "question": questions[qa], "scene_id": result["scene_id"],
            "source_split": result["split"], "experiment_role": "test",
            "seed": 137, "evidence_state": "g1m1",
            "diagnostic_identity": experiment["identity_sha256"],
            "source_sha256": experiment["source_sha256"],
            "reference_answer_visible_during_solving": False,
            "preflight": preflight,
            "question_preflight_skipped": preflight is None,
            "input_preparation": read(outcome_root / "plan.json"),
            "online_result": result,
            "rounds": outcome["rounds"], "outcome": outcome,
            "trace_topics": topics, "model_request_receipts": receipt_files,
            "requests": request_by_variant[variant], "token_usage": dict(usage),
        }
        trajectory_path = root / "trajectories" / (qa + ".json")
        write(trajectory_path, item)
        trajectories.append((result["question_type"], trajectory_path))
        full_file = str(trajectory_path.relative_to(root))
        reviews.append({
            "schema": "vsi-g1m1-question-trajectory-review/1",
            "qa_id": qa, "question_type": result["question_type"],
            "question": questions[qa], "online_result": result,
            "offline_evaluation": evaluation, "full_trajectory_file": full_file,
            "initial_skill_selection": outcome["initial_skill_selection"],
            "rounds": review_value(outcome["rounds"], full_file, "/rounds"),
            "visual_fallback": review_value(
                outcome["eval_visual_fallback"], full_file, "/outcome/eval_visual_fallback"),
            "authorization_receipts": outcome["authorization_receipts"],
            "image_ledger": review_value(outcome["image_ledger"], full_file, "/outcome/image_ledger"),
            "requests": request_by_variant[variant], "token_usage": dict(usage),
        })
        scored.append({
            **result, "question": questions[qa]["question"], "options": label["options"],
            "reference_answer": label["reference_answer"], "offline_evaluation": evaluation,
            "actual_model_requests": len(request_by_variant[variant]),
            "token_usage": dict(usage), "trajectory_file": f"trajectories/{qa}.json",
        })
        audits.append({
            "qa_id": qa, "g1m1_runtime_state": True,
            "question_preflight_skipped": preflight is None,
            "online_reference_unavailable": True,
            "online_scoring_status": outcome["scoring_status"],
            "trace_topics": sorted(topics), "actual_model_requests": len(request_by_variant[variant]),
            "untrusted_answer": outcome["answer_untrusted"],
            "static_error_rounds": sum(
                r.get("error_code") in {"ast_violation", "program_syntax_error"}
                for r in outcome["rounds"]),
            "execution_error_rounds": sum(bool(
                (r.get("execution_diagnostics") or {}).get("error_message")) for r in outcome["rounds"]),
            "authorization_denials": sum(r.get("allowed") is False
                                        for r in outcome["authorization_receipts"]),
        })
        del item, topics, outcome
    groups = {}
    for task in sorted(NAMES):
        selected = [row for row in scored if row["question_type"] == task]
        assert len(selected) == 5
        groups[task] = {
            "name": NAMES[task], "n": 5, "metric": selected[0]["offline_evaluation"]["metric"],
            "score": statistics.mean(row["offline_evaluation"]["score"] for row in selected),
            "legal_answers": sum(row["offline_evaluation"]["legal_answer"] for row in selected),
            "answered": sum(row["offline_evaluation"]["available"] for row in selected),
            "tool_calls": sum(row["tool_call_count"] for row in selected),
            "model_requests": sum(row["actual_model_requests"] for row in selected),
            "mean_elapsed_s": statistics.mean(row["elapsed_s"] for row in selected),
        }
    start, finish = read(root / "solve_start.json"), read(root / "solve_receipt.json")
    assert start["diagnostic_identity"] == experiment["identity_sha256"]
    assert start["automatic_model_retry"] is False
    assert start["question_preflight_skipped"] is True
    assert start["reference_answer_visible"] is finish["reference_answer_visible"] is False
    assert finish["status"] == "complete"
    total_usage = Counter()
    for row in system_requests:
        total_usage.update({k: v for k, v in row["usage"].items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})
    aggregate = {
        "schema": "vsi-g1m1-diagnostic40-results/1", "status": "complete",
        "n_questions": 40, "n_scenes": experiment["n_scenes"], "seed": 137, "role": "test",
        "mean_task_score": statistics.mean(group["score"] for group in groups.values()),
        "metric_note": "Mean of four Accuracy tasks and four MRA tasks; not overall exact-match accuracy.",
        "by_task": groups, "runtime": experiment["runtime"],
        "wall_time_s": finish["finished_at"] - start["started_at"],
        "actual_episode_executors": scheduler["executor_count"],
        "actual_peak_dispatched_episodes": scheduler["peak_dispatched"],
        "actual_peak_request_concurrency": requests["peak_inflight"],
        "actual_model_requests": len(system_requests), "token_usage": dict(total_usage),
        "answer_source_counts": dict(Counter(row["answer_source"] for row in scored)),
        "online_scoring_status_counts": dict(Counter(row["scoring_status"] for row in scored)),
        "legal_answers": sum(row["offline_evaluation"]["legal_answer"] for row in scored),
        "answered": sum(row["offline_evaluation"]["available"] for row in scored),
        "service_errors": sum(bool(row["service_errors"]) for row in scored),
        "unconfigured_detector_questions": [
            row["qa_id"] for row in scored
            if row["service_errors"] and all(
                error.get("tool") == "detect_objects"
                and "endpoint is not configured" in str(error.get("error", ""))
                for error in row["service_errors"])
        ],
        "run_errors": sum(row["final_state"] == "run_error" for row in scored),
        "untrusted_answers": sum(row["untrusted_answer"] for row in audits),
        "static_error_rounds": sum(row["static_error_rounds"] for row in audits),
        "execution_error_rounds": sum(row["execution_error_rounds"] for row in audits),
        "authorization_denials": sum(row["authorization_denials"] for row in audits),
        "tool_calls": sum(row["tool_call_count"] for row in scored),
        "source_sha256": experiment["source_sha256"],
        "diagnostic_identity": experiment["identity_sha256"],
        "online_reference_answer_visible": False, "scoring": "offline only after all online calls",
    }
    write(root / "results.json", {"summary": aggregate, "questions": scored})
    write(root / "summary.json", aggregate)
    write(root / "trajectory_review.json", {
        "schema": "vsi-g1m1-diagnostic40-trajectory-review/1",
        "diagnostic_identity": experiment["identity_sha256"],
        "large_data": "Arrays over 128 items and tool text over 64 KiB are referenced by source file and JSON pointer; full values remain in trajectories/. Model outputs and programs remain inline.",
        "questions": reviews})
    write_collection(root / "trajectories.json", {
        "schema": "vsi-g1m1-diagnostic40-trajectories/1",
        "diagnostic_identity": experiment["identity_sha256"]},
        [path for _, path in trajectories])
    (root / "trajectories_by_type").mkdir(exist_ok=True)
    for task in sorted(NAMES):
        write_collection(root / "trajectories_by_type" / (task + ".json"),
                         {"question_type": task},
                         [path for question_type, path in trajectories if question_type == task])
    write(root / "audit.json", {
        "status": "passed", "question_count": 40, "type_counts": experiment["task_counts"],
        "reference_answer_hidden": True, "questions": audits,
        "requests_completed": len(system_requests), "automatic_model_retry": False,
        "question_preflight_skipped": all(row["question_preflight_skipped"] for row in audits),
        "request_budget_respected": True, "request_identity_verified": True,
        "model_output_and_program_preserved": True,
        "seed_request_response_receipts_verified": True})
    lines = [
        "# g1m1 40 题并发诊断", "",
        f"已完成 40/40 道测试题，8 类各 5 道，覆盖 {experiment['n_scenes']} 个场景，seed=137。",
        f"使用冻结初始 g1m1 Skill 和 Qwen3.8-Flash；求解时参考答案不可见，全部请求结束后离线评分。", "",
        "| 题型 | 题数 | 指标 | 得分 | 有效答案 | 模型请求 | 工具调用 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for group in groups.values():
        lines.append(f"| {group['name']} | 5 | {group['metric']} | {group['score']:.1%} | "
                     f"{group['answered']}/5 | {group['model_requests']} | {group['tool_calls']} |")
    lines.extend([
        "", f"八类平均得分 **{aggregate['mean_task_score']:.1%}**。选择题按 Accuracy，数值题按 MRA；"
        "该平均值是两类指标的综合均值。",
        f"模型请求 {aggregate['actual_model_requests']} 次，实际峰值并发 "
        f"{aggregate['actual_peak_request_concurrency']}，求解阶段耗时 {aggregate['wall_time_s']/60:.1f} 分钟。",
        f"有效答案 {aggregate['answered']}/40，服务错误 {aggregate['service_errors']}，"
        f"最终运行错误 {aggregate['run_errors']}，工具调用 {aggregate['tool_calls']} 次。", "",
        f"按用户要求跳过重复题目预检。题目执行并发设为 "
        f"{aggregate['runtime']['episode_concurrency']}，模型请求并发上限为 "
        f"{requests['policy']['max_inflight']}；上面的峰值来自本轮实际请求收据。",
        f"实际启动 {scheduler['executor_count']} 个题目执行进程，"
        f"同时派发的题目峰值为 {scheduler['peak_dispatched']}。",
        f"轨迹诊断：执行报错轮次 {aggregate['execution_error_rounds']}，"
        f"授权拒绝 {aggregate['authorization_denials']}，"
        f"不可信答案 {aggregate['untrusted_answers']}。", "",
        "detect_objects 补检接口未配置的题号："
        + (", ".join(aggregate["unconfigured_detector_questions"]) or "无")
        + "。有工具服务错误的题保留在完整 40 题分母中，按无效结果计 0 分。", "",
        "- [逐题答案与离线得分](results.json)",
        "- [便于查看的逐轮轨迹 JSON](trajectory_review.json)",
        "- [全部 40 题轨迹 JSON](trajectories.json)",
        "- [逐题轨迹目录](trajectories/)",
        "- [按题型分组的轨迹目录](trajectories_by_type/)",
        "- [原始轨迹压缩包](native_artifacts.tar.gz)",
        "- [汇总 JSON](summary.json)",
        "- [输入、并发与参考答案隔离核验](audit.json)", "",
        "每题轨迹包含问题、Skill 身份、逐轮模型输出、生成程序、工具结果、授权收据、"
        "图像账本、终态、请求身份及 token 用量。标准答案只在离线 results.json 与本报告中呈现。",
        "trajectory_review.json 保留逐轮模型输出与程序；超过 128 项的数组及超过 64 KiB 的工具"
        "文本通过完整轨迹路径与 JSON Pointer 定位。长文本另有预览、字符数及 SHA-256，"
        "全部原始值保存在完整轨迹中。",
    ])
    lines.extend(["", "## 逐题结果", "",
                  "| 题型 | 题号 | 预测 | 标准答案 | 得分 | 请求 | 工具 | 轨迹 |",
                  "|---|---|---|---|---:|---:|---:|---|"])
    for row in scored:
        answer = str(row["answer"]).replace("|", "\\|").replace("\n", " ")
        reference = str(row["reference_answer"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {NAMES[row['question_type']]} | {row['qa_id']} | {answer} | {reference} | "
            f"{row['offline_evaluation']['score']:.0%} | {row['actual_model_requests']} | "
            f"{row['tool_call_count']} | [JSON]({row['trajectory_file']}) |")
    lines.extend(["", "## 按题型查看轨迹", ""])
    for task, name in NAMES.items():
        lines.append(f"- [{name}（5 题）](trajectories_by_type/{task}.json)")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    artifact_paths = [
        root / "report.md", root / "results.json", root / "summary.json",
        root / "trajectories.json", root / "trajectory_review.json", root / "audit.json",
        *[path for _, path in trajectories],
        *sorted((root / "trajectories_by_type").glob("*.json")),
    ]
    artifacts = []
    for path in artifact_paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        artifacts.append({"path": str(path.relative_to(root)),
                          "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    write(root / "artifacts.json", {
        "schema": "vsi-g1m1-diagnostic40-artifacts/1",
        "diagnostic_identity": experiment["identity_sha256"], "files": artifacts})
    print(json.dumps(aggregate, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
