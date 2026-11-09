#!/usr/bin/env python3
"""Verify thinking and RGB identity, then score the forty paired answers offline."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

from export_g1m1_diagnostic import NAMES, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--direct-continuation", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    sys.path.insert(0, str(root / "code/source/src"))
    from skill3d.evaluation.fitset_v11 import load_label_metadata, score_answer, _variant_key, content_sha256
    from skill3d.infra.experiment_campaign import digest
    experiment = read(root / "diagnostic.json")
    pair_path = root / "pair_completion.json"
    assert read(pair_path if pair_path.exists() else root / "exit.json")["status"] == "complete"
    assert read(root / "collection_receipt.json")["status"] == "passed"
    assert read(root / "audit.json")["status"] == "passed"
    arm_runs = [(root, experiment)]
    if pair_path.exists():
        pair = read(pair_path)
        extra = args.direct_continuation or root / "direct_continuation"
        extra = extra.resolve()
        continuation = read(extra / "diagnostic.json")
        assert content_sha256({k: v for k, v in continuation.items()
                               if k != "identity_sha256"}) == continuation["identity_sha256"]
        assert pair["diagnostic_identity"] == experiment["identity_sha256"]
        assert pair["continuation_identity"] == continuation["identity_sha256"]
        assert continuation["input_files"] == experiment["input_files"]
        assert continuation["runtime"] == experiment["runtime"]
        pinned = continuation["direct_continuation"]
        assert pinned["origin_identity"] == experiment["identity_sha256"]
        assert pinned["origin_source_sha256"] == experiment["source_sha256"]
        for relative, sha in pinned["files"].items():
            assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == sha
        for relative, sha in pair["continuation_files"].items():
            assert hashlib.sha256((extra / relative).read_bytes()).hexdigest() == sha
        arm_runs.append((extra, continuation))
    system = {row["qa_id"]: row for row in read(root / "results.json")["questions"]}
    direct_rows, provenance, direct_identities = [], {}, {}
    for arm_root, arm_manifest in arm_runs:
        identity = read(arm_root / "direct/manifest.json")
        assert identity["enable_thinking"] is identity["rgb_only"] is True
        assert identity["campaign_identity"] == arm_manifest["identity_sha256"]
        direct_identities[arm_manifest["identity_sha256"]] = identity
        rows = [read(path) for path in (arm_root / "direct/responses").glob("*.json")]
        assert {row["qa_id"] for row in rows} == set(identity["qa_ids"])
        for row in rows:
            assert row["qa_id"] not in provenance
            provenance[row["qa_id"]] = (arm_root, arm_manifest)
        direct_rows.extend(rows)
    direct = {row["qa_id"]: row for row in direct_rows}
    assert len(direct_rows) == len(direct) == 40
    assert set(direct) == set(system) == {v["qa_id"] for v in experiment["variants"]}
    labels = load_label_metadata(root / "inputs/host_evaluation/metadata.jsonl")
    paired, usage = [], Counter()
    request_model_identity = None
    for qa_id, system_row in sorted(system.items()):
        row, label = direct[qa_id], labels[qa_id]
        arm_root, arm_manifest = provenance[qa_id]
        assert row["campaign_identity"] == arm_manifest["identity_sha256"]
        direct_identity = direct_identities[arm_manifest["identity_sha256"]]
        assert row["scene_id"] == system_row["scene_id"] == label["scene_id"]
        assert row["question_type"] == label["question_type"]
        assert row["reference_answer_visible"] is False and row["tool_calls"] == 0
        actual = row["actual_request_identity"]
        assert actual["parameters"]["enable_thinking"] is True
        assert actual["parameters"]["seed"] == 137
        assert actual["parameters"]["temperature"] == 0
        assert actual["parameters"]["max_tokens"] == direct_identity["max_tokens"]
        key = digest(qa_id)[:24]
        preflight = read(arm_root / "direct/preflight" / (key + ".json"))
        receipt = preflight["input_receipt"]
        assert digest(preflight["request_identity"]) == row["request_sha256"]
        model_identity = preflight["request_identity"]["model_identity"]
        if request_model_identity is None:
            request_model_identity = model_identity
        assert model_identity == request_model_identity
        intent = read(arm_root / "direct/intents" / (key + ".json"))
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
            "direct_diagnostic_identity": arm_manifest["identity_sha256"],
            "direct_source_sha256": arm_manifest["source_sha256"],
            "direct_response_file": str((arm_root / "direct/responses" / (key + ".json")).relative_to(root)),
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
        "direct_source_provenance": [
            {"diagnostic_identity": arm_manifest["identity_sha256"],
             "source_sha256": arm_manifest["source_sha256"],
             "n_questions": len(direct_identities[arm_manifest["identity_sha256"]]["qa_ids"])}
            for _, arm_manifest in arm_runs],
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
    if len(arm_runs) > 1:
        lines.extend([
            "", "直答前 2 题保留首批请求与响应；其余 38 题在修正方向题子类型校验后，"
            "由独立冻结源码继续。题目、RGB、提示、模型参数和总预算保持一致，"
            "没有重复模型调用。逐题 JSON 记录两批源码身份。"
            "首批停止及恢复过程保存在 startup_history/ 和 pair_completion.json。",
            "[续跑直答响应](direct_continuation/direct/responses/)"])
    (root / "comparison_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
