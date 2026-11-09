#!/usr/bin/env python3
"""Compare completed legal answers with the verified historical VSI baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from skill3d.evaluation.accuracy import mca_correct
from skill3d.evaluation.mra import mra_from_text
from skill3d.infra.m5_identity import file_identity


def summarize(rows):
    n = len(rows)
    original = sum(row["original_score"] for row in rows)
    current = sum(row["qwen8b_score"] for row in rows)
    return {
        "n": n, "original_score_sum": original, "qwen8b_score_sum": current,
        "original_mean_score": original / n if n else None,
        "qwen8b_mean_score": current / n if n else None,
        "delta_percentage_points": 100 * (current - original) / n if n else None,
        "improved": sum(row["qwen8b_score"] > row["original_score"] for row in rows),
        "declined": sum(row["qwen8b_score"] < row["original_score"] for row in rows),
        "tied": sum(row["qwen8b_score"] == row["original_score"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    root = args.run_dir
    sampling = json.loads((root / "sampling_manifest.json").read_text())
    selected = {str(row["qa_id"]): row for row in sampling["questions"]}
    original_path = root / "original_pilot_results.json"
    original = json.loads(original_path.read_text())
    if (set(original["questions"]) != set(selected)
            or original["reference_answers_read"] is not False
            or original["model_requests"] != 0):
        raise ValueError("historical pilot coverage or collection contract differs")
    files = {path.stem: path for path in (root / "outcomes").glob("*.json")}
    if not set(files) <= set(selected) or (args.require_all and set(files) != set(selected)):
        raise ValueError("completed pilot scope differs")
    references = {}
    with args.metadata.open() as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["qa_id"]) in files:
                references[str(row["qa_id"])] = row
    rows, excluded = [], []
    for qa, path in sorted(files.items(), key=lambda item: int(item[0])):
        result = json.loads(path.read_text())
        outcome = result["outcome"]
        before, question, reference = original["questions"][qa], selected[qa], references[qa]
        if (result["reference_answer_visible"] is not False
                or before["question"] != question["question"]
                or reference["question"] != question["question"]
                or before["options"] != (question.get("options") or [])
                or (reference.get("options") or []) != (question.get("options") or [])
                or before["source_split"] != result["source_split"]
                or reference["source_split"] != result["source_split"]
                or before["question_type"] != result["question_type"]):
            raise ValueError("paired public question/source identity differs")
        if not before["legal_answer"] or not outcome["legal_answer"]:
            excluded.append({"qa_id": qa, "reason": "one_or_both_answers_illegal"})
            continue
        if before["episode_status"] != "answered":
            raise ValueError("legal original result has no terminal answer")
        reference_answer = reference["ground_truth"]
        if outcome["is_mca"]:
            def score(answer):
                return float(mca_correct(answer, reference_answer))
        else:
            def score(answer):
                return mra_from_text(answer, reference_answer)
        current_answer = outcome.get("predicted") or outcome.get("answer")
        rows.append({
            "qa_id": qa, "task": result["question_type"],
            "original_answer": before["answer"], "qwen8b_answer": current_answer,
            "original_score": score(before["answer"]),
            "qwen8b_score": score(current_answer),
            "original_legal": True, "qwen8b_legal": True,
            "original_result_file": before["result_file"],
            "original_result_sha256": before["result_sha256"],
            "qwen8b_result_file": str(path.relative_to(root)),
            "qwen8b_result_sha256": file_identity(path)["sha256"],
        })
    summary = {
        "schema": "vsi8b-completed-mutually-legal-paired-scores/1", "at": time.time(),
        "local_scoring_only": True, "reference_answers_visible_during_solve": False,
        "model_requests": 0, "pilot_completed": len(files),
        "baseline_stage": original["stage"], "baseline_solver_model": original["model"],
        "baseline_solver_configuration": original["solver_configuration"],
        "original_pilot_results_sha256": file_identity(original_path)["sha256"],
        "sampling_sha256": file_identity(root / "sampling_manifest.json")["sha256"],
        "metadata_sha256": file_identity(args.metadata)["sha256"],
        "inclusion_rule": "Both completed terminal answers have legal_answer=true; an answered legal original result is included even when evidence status is incomplete.",
        "metric": "MCA exact match and numeric 10-threshold MRA, same mutually legal denominator.",
        "overall": summarize(rows), "excluded": excluded, "rows": rows,
        "by_task": {
            task: summarize([row for row in rows if row["task"] == task])
            for task in sampling["type_counts"]
        },
    }
    (root / "paired_scores.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    overall = summary["overall"]
    lines = [
        "# 原流程与 Qwen8B 描述流程的共同合法题目得分",
        "",
        f"本轮完成 {len(files)}/40；共同合法题目 {overall['n']} 题。",
        "",
        "| 题型 | 配对题数 | 原流程 | 8B 描述流程 | 差值/百分点 |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, group in [*summary["by_task"].items(), ("全部", overall)]:
        if group["n"]:
            lines.append(
                f"| {task} | {group['n']} | {group['original_mean_score']:.2%} | "
                f"{group['qwen8b_mean_score']:.2%} | {group['delta_percentage_points']:+.2f} |")
        else:
            lines.append(f"| {task} | 0 | — | — | — |")
    lines += [
        "",
        f"逐题比较：提高 {overall['improved']} 题、降低 {overall['declined']} 题、"
        f"持平 {overall['tied']} 题。",
        "",
        "原侧为历史 VSI1000 initial_validation 的最终解答，本侧为当前 pilot。"
        "题目、选项、题型与来源分区逐题一致；参考答案只在 Mac 本地评分时读取。"
        "数值题保留官方 MRA 的部分得分。逐题答案、得分和文件身份见 paired_scores.json。",
    ]
    (root / "paired_score_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"pilot_completed": len(files), **overall}))


if __name__ == "__main__":
    main()
