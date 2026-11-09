#!/usr/bin/env python3
"""Verify thinking and RGB identity, then score the forty paired answers offline."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

from export_g1m1_diagnostic import NAMES, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    sys.path.insert(0, str(root / "code/source/src"))
    from skill3d.evaluation.fitset_v11 import load_label_metadata, score_answer, _variant_key
    from skill3d.infra.experiment_campaign import digest
    experiment = read(root / "diagnostic.json")
    assert read(root / "exit.json")["status"] == "complete"
    assert read(root / "collection_receipt.json")["status"] == "passed"
    assert read(root / "audit.json")["status"] == "passed"
    direct_identity = read(root / "direct/manifest.json")
    assert direct_identity["enable_thinking"] is direct_identity["rgb_only"] is True
    assert direct_identity["campaign_identity"] == experiment["identity_sha256"]
    system = {row["qa_id"]: row for row in read(root / "results.json")["questions"]}
    direct_rows = [read(path) for path in (root / "direct/responses").glob("*.json")]
    direct = {row["qa_id"]: row for row in direct_rows}
    assert len(direct_rows) == len(direct) == 40
    assert set(direct) == set(system) == {v["qa_id"] for v in experiment["variants"]}
    labels = load_label_metadata(root / "inputs/host_evaluation/metadata.jsonl")
    paired, usage = [], Counter()
    for qa_id, system_row in sorted(system.items()):
        row, label = direct[qa_id], labels[qa_id]
        assert row["campaign_identity"] == experiment["identity_sha256"]
        assert row["scene_id"] == system_row["scene_id"] == label["scene_id"]
        assert row["question_type"] == label["question_type"]
        assert row["reference_answer_visible"] is False and row["tool_calls"] == 0
        actual = row["actual_request_identity"]
        assert actual["parameters"]["enable_thinking"] is True
        assert actual["parameters"]["seed"] == 137
        assert actual["parameters"]["temperature"] == 0
        assert actual["parameters"]["max_tokens"] == direct_identity["max_tokens"]
        key = digest(qa_id)[:24]
        receipt = read(root / "direct/preflight" / (key + ".json"))["input_receipt"]
        intent = read(root / "direct/intents" / (key + ".json"))
        assert intent["request_sha256"] == row["request_sha256"]
        plan = read(root / "system/variants" / _variant_key(system_row["variant_id"]) / "plan.json")
        transport = plan["published_transport"]
        assert all(transport[field] == receipt[field] for field in (
            "qa_id", "n_images", "frame_set_hash", "source_frame_indices",
            "pixel_sha256", "source_catalog_sha256"))
        assert transport["frozen_scene"]["sha256"] == receipt["frozen_scene_sha256"]
        assert receipt["reference_answer_visible"] is False and receipt["n_images"] == 32
        score = score_answer(
            answer=row["answer"], task=label["question_type"], options=label["options"],
            reference=label["reference_answer"],
            available=row["status"] == "completed" and row["final_state"] == "answer")
        system_score = system_row["offline_evaluation"]["score"]
        paired.append({
            "qa_id": qa_id, "question_type": label["question_type"],
            "system_answer": system_row["answer"], "direct_answer": row["answer"],
            "reference_answer": label["reference_answer"],
            "system_score": system_score, "direct_score": score["score"],
            "delta": system_score - score["score"], "direct_evaluation": score,
            "same_frozen_rgb_identity": True,
        })
        usage.update({k: v for k, v in row["usage"].items()
                      if isinstance(v, (int, float)) and not isinstance(v, bool)})
    groups = {}
    for task, name in NAMES.items():
        selected = [row for row in paired if row["question_type"] == task]
        assert len(selected) == 5
        groups[task] = {
            "name": name, "n": 5,
            "system_score": statistics.mean(row["system_score"] for row in selected),
            "direct_score": statistics.mean(row["direct_score"] for row in selected),
            "delta": statistics.mean(row["delta"] for row in selected)}
    summary = {
        "status": "complete", "n_pairs": 40, "both_enable_thinking": True,
        "system_score": statistics.mean(row["system_score"] for row in paired),
        "direct_score": statistics.mean(row["direct_score"] for row in paired),
        "delta": statistics.mean(row["delta"] for row in paired),
        "improved": sum(row["delta"] > 0 for row in paired),
        "decreased": sum(row["delta"] < 0 for row in paired),
        "tied": sum(row["delta"] == 0 for row in paired),
        "system_legal_answers": read(root / "summary.json")["legal_answers"],
        "direct_legal_answers": sum(row["direct_evaluation"]["legal_answer"] for row in paired),
        "system_requests": read(root / "summary.json")["actual_model_requests"],
        "direct_requests": 40, "direct_token_usage": dict(usage), "by_task": groups,
        "diagnostic_identity": experiment["identity_sha256"],
        "protocol": "prepared-32", "reference_answer_visible_during_solving": False,
        "metric": "Mean of four Accuracy tasks and four MRA tasks"}
    write(root / "comparison.json", {"summary": summary, "pairs": paired})
    lines = [
        "# 同一 40 题 thinking 对比", "",
        f"新版 Harness：**{summary['system_score']:.2%}**；Qwen3.8-Flash 直答："
        f"**{summary['direct_score']:.2%}**；差值：**{summary['delta']*100:+.2f} 个百分点**。", "",
        "两组均开启 thinking，复用原 40 题的 32 帧、seed=137、temperature=0 和 token 上限。"
        "直答组每题一次请求，只使用原始 RGB 和题目。参考答案在全部调用结束后用于离线评分。", "",
        "| 题型 | Harness | Qwen 直答 | 差值 |",
        "|---|---:|---:|---:|"]
    for group in groups.values():
        lines.append(f"| {group['name']} | {group['system_score']:.1%} | "
                     f"{group['direct_score']:.1%} | {group['delta']*100:+.1f} pp |")
    lines.extend([
        "", f"逐题比较：提高 {summary['improved']} 题，下降 {summary['decreased']} 题，"
        f"持平 {summary['tied']} 题。得分采用 VSI-Bench 的 Accuracy/MRA 综合均值。", "",
        "[逐题对比分数](comparison.json) · [Harness 轨迹](trajectory_review.json) · "
        "[直答原始响应](direct/responses/) · [输入与请求清单](diagnostic.json)"])
    (root / "comparison_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
