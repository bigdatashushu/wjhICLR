#!/usr/bin/env python3
"""Collect verified repaired results and score against the fixed two baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from launch_choice_paired100 import LOCAL, REMOTE, OLD, remote
from collect_proportional100 import value


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def collect():
    files = ["choice_policy_verified.json", "cached_preparation_manifest.json"]
    files += ["harness/" + name for name in (
        "native_trajectory_verified.json", "compact_score_inputs.json",
        "preflight_verified.json", "runtime_identity.json", "live_status.json",
        "trajectory_index.json", "parallel_execution.json")]
    files += ["execute.pid.json", "verify.pid.json", "preflight.pid.json"]
    code = """import json
from pathlib import Path
r=Path(%r)
print(json.dumps({name:json.loads((r/name).read_text()) for name in %r}))
""" % (REMOTE, files)
    for name, data in json.loads(remote(code, timeout=120)).items():
        write(LOCAL / name, data)


def paired_summary(rows, baseline):
    differences = np.array([r["new_score"] - r[baseline + "_score"] for r in rows])
    rng = np.random.default_rng(137)
    means = differences[rng.integers(0, len(rows), size=(10000, len(rows)))].mean(axis=1)
    return {
        "n": len(rows), "delta_percentage_points": 100 * float(differences.mean()),
        "improved": int((differences > 0).sum()), "declined": int((differences < 0).sum()),
        "tied": int((differences == 0).sum()),
        "paired_bootstrap_95pct_ci_pp": (100 * np.quantile(means, [.025, .975])).tolist(),
    }


def means(rows):
    return {arm + "_mean": sum(r[arm + "_score"] for r in rows) / len(rows)
            for arm in ("direct", "old", "new")}


def score(metadata):
    spec = read(LOCAL / "experiment_spec.json")
    sampling = read(LOCAL / "sampling_manifest.json")
    native = read(LOCAL / "harness/native_trajectory_verified.json")
    policy = read(LOCAL / "choice_policy_verified.json")
    baseline = read(OLD / "direct_vs_8b_scores.json")
    if (native["questions_verified"] != 100 or policy["questions_verified"] != 100
            or hashlib.sha256((OLD / "direct_vs_8b_scores.json").read_bytes()).hexdigest()
            != spec["baseline_scores_sha256"]
            or hashlib.sha256(metadata.read_bytes()).hexdigest()
            != baseline["metadata_sha256"]):
        raise ValueError("complete verified candidate or fixed baseline identity differs")
    reference = {str(q["qa_id"]): q for q in map(json.loads, metadata.read_text().splitlines())}
    prior = {q["qa_id"]: q for q in baseline["rows"]}
    current = read(LOCAL / "harness/compact_score_inputs.json")
    rows = []
    if set(current) != {str(q["qa_id"]) for q in sampling["questions"]}:
        raise ValueError("paired panel denominator changed")
    for question in sampling["questions"]:
        qa = str(question["qa_id"])
        ref = reference[qa]
        record = current[qa]
        previous = prior[qa]
        if (question["question"] != ref["question"]
                or (question.get("options") or []) != (ref.get("options") or [])
                or question["source_split"] != ref["source_split"]
                or record["source_split"] != question["source_split"]
                or record["reference_answer_visible"] is not False):
            raise ValueError("posthoc reference or source-split identity differs")
        outcome = record["outcome"]
        diagnostic = policy["questions"][qa]
        row = {
            "qa_id": qa, "task": previous["task"], "source_split": question["source_split"],
            "is_mca": diagnostic["is_mca"],
            "direct_answer": previous["direct_answer"], "direct_score": previous["direct_score"],
            "old_answer": previous["harness_answer"], "old_score": previous["harness_score"],
            "new_answer": outcome.get("predicted") if outcome.get("predicted") is not None
                          else outcome.get("answer"),
            "new_score": value(outcome, ref), "new_legal": outcome["legal_answer"],
            "new_outcome_sha256": record["outcome_sha256"],
            **diagnostic,
        }
        fallback = diagnostic["fallback"]
        original = fallback.get("original_result") or {}
        if fallback.get("isolated_choice") and original.get("answer") is not None:
            row["same_run_pre_fallback_score"] = value(
                {"answer": original["answer"], "predicted": original["answer"],
                 "legal_answer": original.get("legal_answer", True), "is_mca": True}, ref)
        rows.append(row)
    choices = [r for r in rows if r["is_mca"]]
    numeric = [r for r in rows if not r["is_mca"]]
    old_visual_choices = [r for r in choices if r["old_answer_basis"] == "visual_estimate"]
    by_task = {}
    for task, expected in sampling["type_counts"].items():
        group = [r for r in rows if r["task"] == task]
        if len(group) != expected:
            raise ValueError("frozen task quota changed")
        by_task[task] = {"n": len(group), **means(group),
                         "new_vs_old": paired_summary(group, "old")}
    b = sum(r["new_score"] == 1 and r["old_score"] == 0 for r in choices)
    c = sum(r["new_score"] == 0 and r["old_score"] == 1 for r in choices)
    discordant = b + c
    p = min(1., 2 * sum(math.comb(discordant, i) for i in range(min(b, c) + 1))
            / 2 ** discordant) if discordant else 1.
    isolated = [r for r in rows if r["fallback"].get("isolated_choice")]
    before_after = [r for r in isolated if "same_run_pre_fallback_score" in r]
    summary = {
        "schema": "vsi-choice-repair-real-paired100/1", "n_questions": 100,
        "completed": 100, "new_legal_answers": sum(r["new_legal"] for r in rows),
        **means(rows), "new_vs_old": paired_summary(rows, "old"),
        "new_vs_direct": paired_summary(rows, "direct"),
        "choices": {"n": len(choices), **means(choices),
                    "new_vs_old": paired_summary(choices, "old"),
                    "new_vs_direct": paired_summary(choices, "direct"),
                    "wrong_to_correct": b, "correct_to_wrong": c, "mcnemar_exact_p": p},
        "numeric": {"n": len(numeric), **means(numeric),
                    "new_vs_old": paired_summary(numeric, "old")},
        "old_visual_choices": (
            {"n": len(old_visual_choices), **means(old_visual_choices),
             "new_vs_old": paired_summary(old_visual_choices, "old")}
            if old_visual_choices else {"n": 0}),
        "isolated_choice": {
            "n": len(isolated), "requests_verified": policy["isolated_requests_verified"],
            "accepted": sum(bool(r["fallback"].get("accepted")) for r in isolated),
            "mean_score": sum(r["new_score"] for r in isolated) / len(isolated)
                          if isolated else None,
            "same_run_pre_answer_n": len(before_after),
            "same_run_pre_answer_score": sum(r["same_run_pre_fallback_score"] for r in before_after)
                                        / len(before_after) if before_after else None,
            "same_run_post_answer_score": sum(r["new_score"] for r in before_after)
                                         / len(before_after) if before_after else None,
        },
        "by_task": by_task, "native_model_requests": native["model_requests"],
        "token_usage": policy["token_usage"],
        "elapsed_s": read(LOCAL / "harness/live_status.json")["elapsed_s"],
        "sampling_sha256": spec["sampling_sha256"],
        "source_manifest_sha256": hashlib.sha256(
            (LOCAL / "source_manifest.json").read_bytes()).hexdigest(),
        "metadata_sha256": baseline["metadata_sha256"],
        "reference_answer_visible_during_solve": False,
        "baseline_reused_from_prior_run": True, "diagnostic_fixed_panel": True,
        "scientific_interpretation": "same-panel diagnostic; baseline was recorded earlier; "
                                     "model output variation and prompt changes are not separated",
        "rows": rows, "scored_at": time.time(),
    }
    write(LOCAL / "paired_repair_scores.json", summary)
    labels = {
        "obj_appearance_order": "首次出现顺序", "object_abs_distance": "绝对距离",
        "object_counting": "物体计数", "object_rel_direction": "相对方向",
        "object_rel_distance": "相对距离", "object_size_estimation": "物体尺寸",
        "room_size_estimation": "房间面积", "route_planning": "路径规划",
    }
    lines = [
        "# 选择题视觉回退修复：真实配对 100 题",
        "",
        f"新版本完成 100/100 题，合法答案 {summary['new_legal_answers']}/100。"
        f"总体平均分 {summary['new_mean']:.2%}，原 Harness {summary['old_mean']:.2%}，"
        f"已有 Qwen 直答 {summary['direct_mean']:.2%}。",
        f"相对原 Harness {summary['new_vs_old']['delta_percentage_points']:+.2f} 个百分点；"
        f"{summary['new_vs_old']['improved']} 题提高、"
        f"{summary['new_vs_old']['declined']} 题下降、"
        f"{summary['new_vs_old']['tied']} 题持平。",
        "",
        "| 组别 | 题数 | Qwen 直答 | 原 Harness | 修复后 |",
        "|---|---:|---:|---:|---:|",
        f"| 全部题目（MCA/MRA 平均分） | 100 | {summary['direct_mean']:.2%} | "
        f"{summary['old_mean']:.2%} | {summary['new_mean']:.2%} |",
        f"| 选择题（准确率） | {len(choices)} | {summary['choices']['direct_mean']:.2%} | "
        f"{summary['choices']['old_mean']:.2%} | {summary['choices']['new_mean']:.2%} |",
        f"| 数值题（MRA 平均分） | {len(numeric)} | {summary['numeric']['direct_mean']:.2%} | "
        f"{summary['numeric']['old_mean']:.2%} | {summary['numeric']['new_mean']:.2%} |",
        "",
        "## 选择题与实际补答",
        "",
        f"选择题相对旧版错→对 {b} 题、对→错 {c} 题。"
        f"配对 McNemar 精确双侧 p={p:.4f}。",
        f"独立原图补答实际触发 {len(isolated)} 题，逐请求核验通过 "
        f"{policy['isolated_requests_verified']} 题，接受 "
        f"{summary['isolated_choice']['accepted']} 题。"
        "核验包括单条 user 消息、准确的提示文本、32 张原始 RGB 的 JPEG 字节及账本哈希，"
        "不含工具历史、裁剪或场景总览。",
    ]
    if before_after:
        lines.append(
            f"本次求解中，已有视觉答案后再补答的 {len(before_after)} 题："
            f"补答前准确率 {summary['isolated_choice']['same_run_pre_answer_score']:.2%}，"
            f"补答后 {summary['isolated_choice']['same_run_post_answer_score']:.2%}。")
    if old_visual_choices:
        group = summary["old_visual_choices"]
        lines.append(
            f"原版以 visual_estimate 作答的选择题固定子集共 {group['n']} 题："
            f"旧版 {group['old_mean']:.2%}，修复后 {group['new_mean']:.2%}；"
            f"{group['new_vs_old']['improved']} 题提高、"
            f"{group['new_vs_old']['declined']} 题下降、"
            f"{group['new_vs_old']['tied']} 题持平。")
    lines += ["", "## 分题型结果", "",
              "| 题型 | 题数 | Qwen 直答 | 原 Harness | 修复后 |",
              "|---|---:|---:|---:|---:|"]
    for task, group in by_task.items():
        lines.append(f"| {labels[task]} | {group['n']} | {group['direct_mean']:.2%} | "
                     f"{group['old_mean']:.2%} | {group['new_mean']:.2%} |")
    lower, upper = summary["new_vs_old"]["paired_bootstrap_95pct_ci_pp"]
    lines += [
        "", "## 协议与证据", "",
        "固定原 100 题及各题 source_split；其中 33 题来自 final_test，"
        "本轮作为已见题面板的诊断盲评，不能视为新的独立测试集。"
        "8B 描述、审核、掩码、几何证据和 Skill 快照复用原始字节。"
        "修复后完整求解重新在线运行，不用旧答案生成新结果。",
        "主模型 qwen3.8-flash，thinking=true，seed=137，temperature=0，"
        "max_tokens=8192，timeout=900 秒，6 个求解轮次，最多一次视觉补答，"
        "每题请求硬上限 12；未自动重试，未发起新的准备模型或 GPU 请求。"
        "参考答案与在线求解保持隔离；阶段统计仅对已完成且请求核验通过的题目在 Mac "
        "离线评分，最终 100 题统计在完整原生轨迹核验后完成。",
        f"新版本共 {native['model_requests']} 次主模型请求；"
        f"usage={json.dumps(policy['token_usage'], ensure_ascii=False)}。",
        f"相对原版总体分差的配对 bootstrap 95% 区间：[{lower:+.2f}, {upper:+.2f}] 个百分点"
        "（10000 次重采样，统计 seed=137）。",
        "旧 Harness 和直答结果来自此前运行。本次同时含上下文隔离、方向/完整候选提示和"
        "部分排序恢复说明；配对分差反映这组修复在当前面板的表现，"
        "不能单独归因于上下文隔离，也没有剥离模型输出波动。",
        f"完整原生轨迹保存在 Titan：`{REMOTE}/harness/trajectories/`。"
        "本地保存逐题分数、输入身份及完整核验回执。",
        "逐题答案和分数见 `paired_repair_scores.json`；"
        "请求策略审计见 `choice_policy_verified.json`。",
    ]
    (LOCAL / "comparison_report.md").write_text("\n".join(lines) + "\n")
    report = LOCAL.parents[1] / "vsi-choice-repair-paired100-report.md"
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "completed", "new_legal_answers", "direct_mean", "old_mean", "new_mean",
        "new_vs_old", "choices", "isolated_choice")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    for key in ("DASHSCOPE_API_KEY", "QWEN3VL8B_API_KEY", "VSI_SAM2_TOKEN",
                "OPENROUTER_API_KEY", "MAKERS_MODELS_KEY"):
        os.environ.pop(key, None)
    collect()
    score(args.metadata)


if __name__ == "__main__":
    main()
