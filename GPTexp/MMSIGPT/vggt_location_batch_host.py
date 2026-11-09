"""Independent reference access only after all 240 unique attempts are frozen."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

from ..traces import digest, read, write
from . import vggt_location_batch as batch
from .prototype.direct import operator_lock
from .prototype.evaluation import reference_answers
from .prototype.protocol import LETTERS
from .vggt_pose30_host import TASK_NAMES


def summarize(rows):
    n = len(rows)
    effective = [p for p in rows if p["status"] in ("answered", "invalid_answer")]
    correct = sum(p["correct"] for p in rows)
    return {
        "questions": n, "correct": correct, "accuracy": correct / n if n else None,
        "effective_questions": len(effective),
        "effective_accuracy": correct / len(effective) if effective else None,
        "legal_answers": sum(p["status"] == "answered" for p in rows),
        "invalid_answers": sum(p["status"] == "invalid_answer" for p in rows),
        "service_errors": sum(p["status"] == "service_error" for p in rows),
    }


def score_internal(root, references, expected_sha):
    root = Path(root).resolve()
    with operator_lock(root):
        manifest, frozen = batch.validate_campaign(root)
        raw = Path(references).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError("reference_identity_changed")
        answers = reference_answers(raw, frozen["predictions"])
        predictions = {p["sample_id"]: p for p in frozen["predictions"]}
        rows, actual, usage = [], [], Counter()
        tool_calls = 0
        for item in manifest["items"]:
            sample = item["problem"]["sample_id"]
            prediction = predictions[sample]
            answer = answers[sample]
            if answer not in LETTERS[:len(item["problem"]["options"])]:
                raise ValueError("invalid_reference_option")
            results = []
            for phase in frozen["phases"]:
                _, path = batch.old.shared.paths(phase["directory"], item, batch.ARM)
                if path.exists():
                    results.append(read(path))
            if len(results) != 1:
                raise ValueError("duplicate_or_missing_final_result")
            result = results[0]
            response = result.get("response", {})
            reported = response.get("reported")
            if reported is not None:
                actual.append(reported)
            tool_calls += response.get("tool_call_count", 0)
            for k, v in response.get("usage", {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    usage[k] += v
            rows.append({
                **prediction, "reference_answer": answer,
                "correct": prediction["status"] == "answered" and prediction["answer"] == answer,
                "source_split": item["problem"]["source_split"],
                "previously_selected_or_dispatched": item["previously_selected_or_dispatched"],
                "question": item["problem"]["question"], "options": item["problem"]["options"],
                "relation_text": item["relation_text"], "reported": reported,
                "usage": response.get("usage"), "elapsed_seconds": response.get("elapsed_seconds"),
                "error_code": result.get("error_code"),
            })
        if tool_calls:
            raise ValueError("unexpected_model_tools")
        metadata = []
        for reported in actual:
            if reported not in [row["reported"] for row in metadata]:
                metadata.append({"reported": reported, "count": actual.count(reported)})
        value = {
            "schema": "mmsi-location-batch-host-score/1",
            "run_identity": manifest["run_identity"], "freeze_identity": frozen["freeze_identity"],
            "reference_file_sha256": expected_sha,
            "reference_access": "independent_clean_process_after_240_terminal_attempts_frozen",
            "reference_answer_visible_during_solving": False,
            "metric": "extracted_letter_exact_match", "judge_fallback": False,
            "denominator": "all_240_selected_with_effective_denominator_also_reported",
            "sample_role": manifest["sample_role"], "overall": summarize(rows),
            "by_task": {task: summarize([p for p in rows if p["task_type"] == task])
                        for task in sorted(TASK_NAMES)},
            "by_api": {api: summarize([p for p in rows if p["api_id"] == api])
                       for api in manifest["api_ids"]},
            "by_history": {
                "new_to_recorded_history": summarize([
                    p for p in rows if not p["previously_selected_or_dispatched"]]),
                "historical_overlap": summarize([
                    p for p in rows if p["previously_selected_or_dispatched"]]),
            },
            "unique_solver_attempts": 240, "peak_inflight": max(
                read(Path(p["directory"]) / "execution.json")["peak_inflight"]
                for p in frozen["phases"]),
            "gpu_joint_forwards": 240, "model_tool_calls": tool_calls,
            "reported_metadata": metadata, "numeric_usage": dict(usage),
            "rows": rows, "scored_at": time.time(),
        }
        value["score_identity"] = digest(value)
        write(root / "score.json", value, exclusive=True)
        report(root, value)
        print(json.dumps({"status": "scored", "overall": value["overall"],
                          "by_task": value["by_task"]}, ensure_ascii=False), flush=True)
        return value


def report(root, value):
    totals = value["overall"]
    effective_accuracy = (f"{totals['effective_accuracy']:.1%}"
                          if totals["effective_accuracy"] is not None else "—")
    lines = [
        "# MMSI 六类各 40 题：原始 RGB＋拍摄位置关系", "",
        f"共答对 **{totals['correct']}/240（{totals['accuracy']:.1%}）**。"
        f"有效回答 {totals['effective_questions']} 题，"
        f"有效口径准确率 {effective_accuracy}；"
        f"服务错误 {totals['service_errors']} 题，非法答案 {totals['invalid_answers']} 题。", "",
        "| 题型 | 答对 / 抽样 | 全量准确率 | 有效题数 | 有效准确率 | 服务错误 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, row in value["by_task"].items():
        acc = f"{row['effective_accuracy']:.1%}" if row["effective_accuracy"] is not None else "—"
        lines.append(
            f"| {TASK_NAMES[task]} | {row['correct']}/40 | {row['accuracy']:.1%} | "
            f"{row['effective_questions']} | {acc} | {row['service_errors']} |")
    lines.extend([
        "", "抽样固定为六类各40题、seed=137、共240个唯一题目，全部排除上一轮30题。"
        "优先选取历史记录中未出现的题目，共230题；物体—区域补入8道历史题，"
        "区域—区域补入2道历史题。历史重叠在selection.json和逐题结果中明确标记。"
        "保留原source_split=test，本轮用途为分层开发诊断。", "",
        "沿用上一轮原始RGB＋拍摄位置关系流程：两张官方原图按原顺序发送，"
        "追加双向拍摄点位置关系文字，然后发送原题和原选项。"
        "本轮未调用目标选择或目标检测工具。", "",
        f"6个API槽位并发，每槽顺序完成40题；实测峰值并发{value['peak_inflight']}。"
        "共240次唯一题目请求；服务错误保留，未对失败题自动重试。"
        "若服务错误触发停发，后续独立阶段只派发之前从未派发的题目。"
        "全部阶段结束并冻结后，独立评分进程才读取官方参考答案。", "",
        "请求参数沿用上一轮：gpt-5.6-terra，reasoning_effort=high，temperature=0，"
        "seed=137，max_output_tokens=8192，timeout=180秒。实际服务回报：", "",
        "```json", json.dumps(value["reported_metadata"], ensure_ascii=False, indent=2), "```", "",
        "服务回报的temperature/seed可能与请求值不同，报告保留实际值。", "",
        "240组相机预计算使用RTX4090物理GPU3；VGGT源码和权重与上一轮核对一致。"
        "联合双图、官方crop预处理、聚合器BF16、相机头FP32。"
        "没有执行深度/点云/跟踪头，没有向GPT发送深度图、位姿数值或相机朝向关系。", "",
        "逐题问题、关系文字、预测、GT、真实服务参数与token用量见score.json；"
        "原始响应和请求身份保留在各阶段traces目录。", "",
        f"run_identity：`{value['run_identity']}`", "",
        f"freeze_identity：`{value['freeze_identity']}`", "",
        f"score_identity：`{value['score_identity']}`", "",
    ])
    (Path(root) / "实验结果.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--reference-sha256", required=True)
    args = parser.parse_args()
    try:
        score_internal(args.directory, args.references, args.reference_sha256)
    except Exception:
        print('{"status":"failed","reason_code":"location_batch_host_score_failed"}', flush=True)
        raise SystemExit(2)
