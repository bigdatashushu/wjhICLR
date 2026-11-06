#!/usr/bin/env python3
"""Local posthoc scoring; private references never enter the online runner."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

from skill3d.evaluation.accuracy import mca_correct
from skill3d.evaluation.mra import mra_from_text
from skill3d.infra.m5_identity import content_sha256, file_identity


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n")


def write_full_aggregate(root, summary, selected, trajectory_index):
    """Copy verified native JSON in chunks rather than holding all episodes."""
    destination = root / "results_and_full_trajectories.json"
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        def value(data):
            json.dump(data, handle, ensure_ascii=False, allow_nan=False)

        def native(path):
            with path.open(encoding="utf-8") as source:
                shutil.copyfileobj(source, handle, length=1024 * 1024)

        def mapping(directory, *, arrays=False):
            handle.write("{")
            for index, qa in enumerate(selected):
                if index:
                    handle.write(",\n")
                value(qa)
                handle.write(":")
                if arrays:
                    handle.write("[")
                    for position, path in enumerate(sorted((root / directory / qa).glob("*.json"))):
                        if position:
                            handle.write(",\n")
                        native(path)
                    handle.write("]")
                else:
                    native(root / directory / f"{qa}.json")
            handle.write("}")

        handle.write('{"summary":')
        value(summary)
        handle.write(',\n"outcomes":')
        mapping("outcomes")
        handle.write(',\n"per_question_trajectories":')
        mapping("trajectories")
        handle.write(',\n"trajectory_index":')
        value(trajectory_index)
        handle.write(',\n"program_executions":')
        mapping("program_executions", arrays=True)
        handle.write(',\n"preparation":')
        mapping("bindings")
        handle.write("}\n")
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    sampling = json.loads((root / "sampling_manifest.json").read_text())
    selected = {str(row["qa_id"]): row for row in sampling["questions"]}
    # Read references only after all 40 immutable outcomes have arrived.
    files = {path.stem: path for path in (root / "outcomes").glob("*.json")}
    if set(files) != set(selected):
        raise ValueError("posthoc scoring requires all forty outcomes")
    trajectory_index = json.loads((root / "trajectory_index.json").read_text())
    if (set(trajectory_index["questions"]) != set(selected) or any(
            not row["completeness"]["outcome_written"]
            or not row["completeness"]["all_model_requests_returned"]
            for row in trajectory_index["questions"].values())):
        raise ValueError("posthoc scoring requires all forty complete trajectory exports")
    metadata = {str(row["qa_id"]): row for row in (
        json.loads(line) for line in args.metadata.read_text().splitlines())}
    candidates, rows = [], []
    for qa_id, question in selected.items():
        result = json.loads(files[qa_id].read_text())
        outcome = result["outcome"]
        reference = metadata[qa_id]
        if (result["reference_answer_visible"] is not False
                or reference["question"] != question["question"]
                or (reference["options"] or []) != (question.get("options") or [])
                or reference["source_split"] != result["source_split"]):
            raise ValueError("posthoc question/source identity mismatch")
        score = (float(mca_correct(outcome.get("predicted") or outcome.get("answer") or "",
                                   reference["ground_truth"]))
                 if outcome["is_mca"] else
                 mra_from_text(outcome.get("predicted") or outcome.get("answer"),
                               reference["ground_truth"]))
        row = {
            "qa_id": qa_id, "task": result["question_type"],
            "answer": outcome.get("answer"), "score": score,
            "legal_answer": outcome["legal_answer"],
            "answer_source": outcome["answer_source"],
            "agent_rounds": outcome["agent_rounds"],
            "tool_calls": sum(round_.get("tool_calls", 0) for round_ in outcome.get("rounds", [])),
            "outcome_sha256": file_identity(files[qa_id])["sha256"],
        }
        rows.append(row)
        binding = json.loads((root / "bindings" / f"{qa_id}.json").read_text())
        for receipt in binding["stats"].get("candidate_receipts", []):
            visual = receipt.get("semantic_visual")
            if not visual:
                continue
            description = visual.get("target_description") or {}
            comparison = receipt["image_regeneration_comparison"]
            binary = visual.get("binary_review") or {}
            candidates.append({
                "qa_id": qa_id, "task": result["question_type"],
                "candidate_id": receipt["candidate_id"],
                "category": receipt["category"], "frame_slot": receipt["frame_slot"],
                "description": description.get("description"),
                "old_p_yes": comparison["original_p_yes"],
                "new_p_yes": binary.get("p_yes"),
                "old_pass": comparison["original_pass"],
                "new_pass": binary.get("semantic_pass"),
                "three_image_hashes_equal": comparison["all_three_image_hashes_equal"],
                "description_status": description.get("status"),
                "review_status": visual["status"],
                "description_request_sha256": description.get("request_sha256"),
                "description_response_model": description.get("raw_response", {}).get("model"),
                "review_request_file": visual.get("request_file"),
            })
    by_task = {}
    for task in sampling["type_counts"]:
        group = [row for row in rows if row["task"] == task]
        reviews = [row for row in candidates if row["task"] == task]
        by_task[task] = {
            "n": len(group), "mean_score": sum(row["score"] for row in group) / len(group),
            "legal_answers": sum(row["legal_answer"] for row in group),
            "reviewed_candidates": len(reviews),
            "old_pass": sum(row["old_pass"] is True for row in reviews),
            "new_pass": sum(row["new_pass"] is True for row in reviews),
        }
    summary = {
        "schema": "vsi8b-pilot-posthoc/1", "n_questions": 40,
        "mean_score": sum(row["score"] for row in rows) / len(rows),
        "legal_answers": sum(row["legal_answer"] for row in rows),
        "scoring": "Original MCA exact match and official-equivalent numeric MRA, mean over 40",
        "metadata_sha256": file_identity(args.metadata)["sha256"],
        "sampling_sha256": file_identity(root / "sampling_manifest.json")["sha256"],
        "trajectory_index_sha256": file_identity(root / "trajectory_index.json")["sha256"],
        "reference_answer_visible_during_solve": False, "by_task": by_task,
        "n_reviewed_candidates": len(candidates),
        "three_images_unchanged": sum(row["three_image_hashes_equal"] for row in candidates),
        "old_pass": sum(row["old_pass"] is True for row in candidates),
        "new_pass": sum(row["new_pass"] is True for row in candidates),
        "reject_to_pass": sum(row["old_pass"] is not True and row["new_pass"] is True
                              for row in candidates),
        "pass_to_reject": sum(row["old_pass"] is True and row["new_pass"] is not True
                              for row in candidates),
        "rows": rows,
    }
    write(root / "posthoc_scores.json", summary)
    write(root / "candidate_review_comparison.json", candidates)
    write_full_aggregate(root, summary, selected, trajectory_index)
    lines = [
        "# VSI8B 40 题试跑结果",
        "",
        f"完成 40/40 题，平均官方分数 {summary['mean_score']:.2%}，"
        f"合法答案 {summary['legal_answers']}/40。主解题模型 Qwen3.8-Flash，thinking 开启。",
        "",
        "SAM2、Qwen3-VL-8B-Instruct-FP8 和原 Intern-Decision-4B 在 4090 运行；"
        "重建、冻结 RGB 帧、原解析和 NMS 后检测框沿用已核验缓存。",
        "",
        "| 题型 | 题数 | 分数 | 合法答案 | 审核候选 | 原通过 | 新通过 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, result in by_task.items():
        lines.append(f"| {task} | {result['n']} | {result['mean_score']:.2%} | "
                     f"{result['legal_answers']} | {result['reviewed_candidates']} | "
                     f"{result['old_pass']} | {result['new_pass']} |")
    lines += [
        "",
        f"共重审 {len(candidates)} 个候选；三图哈希完全一致的有 "
        f"{summary['three_images_unchanged']} 个。原未通过→新通过 "
        f"{summary['reject_to_pass']} 个，原通过→新未通过 {summary['pass_to_reject']} 个。",
        "",
        "本轮加入了 8B 描述并补生成 SAM2 掩码；三图相异的候选同时发生视觉输入变化，"
        "审核概率与通过数的变化不能全部归因于描述。候选未配独立语义真值，"
        "不能仅凭通过数判断审核准确率。逐候选描述、原/新概率和图像哈希见 "
        "`candidate_review_comparison.json`。",
        "",
        "完整轨迹见 `results_and_full_trajectories.json`，"
        "包含原生模型响应与 reasoning_content、程序、各轮工具输入输出、授权回执和准备记录。"
        "逐请求三图保存在准备目录的 JSON 中，原始补生成掩码以 NPZ 保存。",
        "",
        "每题独立完整 JSON 位于 `trajectories/<qa_id>.json`，索引为 "
        "`trajectory_index.json`；包含历史失败尝试、原生审核三图输入和错误记录。"
        "当前解题请求等待上限为 900 秒（15 分钟），详见 `TRAJECTORIES.md`。",
        "",
        "评分在本地读取参考答案后执行；在线问题、描述、审核和解题记录保持参考答案不可见。",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "n_questions", "mean_score", "legal_answers", "n_reviewed_candidates",
        "three_images_unchanged", "reject_to_pass", "pass_to_reject")}))


if __name__ == "__main__":
    main()
