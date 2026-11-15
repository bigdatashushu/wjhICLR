#!/usr/bin/env python3
"""Verify the paired request evidence and compute official VSI scores offline."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys


NAMES = {"object_rel_distance": "相对距离", "object_counting": "物体计数"}


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    sys.path.insert(0, str(root / "code/source/src"))
    from skill3d.evaluation.fitset_v11 import (
        content_sha256, load_label_metadata, score_answer, _variant_key)
    from skill3d.infra.experiment_campaign import digest
    from skill3d.synthesis.prompt_compaction import COMPACTION_VERSION
    assert COMPACTION_VERSION == "vsi-prompt-compaction/1.0"
    experiment = read(root / "diagnostic.json")
    assert content_sha256({k: v for k, v in experiment.items()
                           if k != "identity_sha256"}) == experiment["identity_sha256"]
    assert read(root / "exit.json")["status"] == "complete"
    collection = read(root / "collection_receipt.json")
    assert collection["diagnostic_identity"] == experiment["identity_sha256"]
    for relative, identity in collection["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == identity["sha256"]
    for relative, sha in experiment["input_files"].items():
        assert hashlib.sha256((root / "inputs" / relative).read_bytes()).hexdigest() == sha
    for row in read(root / "code/code_identity.json")["files"]:
        assert hashlib.sha256((root / "code/source" / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    system_identity = read(root / "system/manifest.json")
    assert system_identity["reference_answer_visible_to_online_model"] is False
    assert system_identity["online_executor_input_is_label_free"] is True
    system_summary = read(root / "system/summary.json")
    assert system_summary["status"] == "complete"
    assert system_summary["completed_variants"] == experiment["n_questions"]
    scheduler = read(root / "system/episode_pool_receipt.json")
    assert scheduler["status"] == "drained"
    direct_identity = read(root / "direct/manifest.json")
    assert direct_identity["enable_thinking"] is direct_identity["rgb_only"] is True
    assert direct_identity["max_requests_per_question"] == 1
    system = {read(p)["qa_id"]: read(p) for p in (root / "system/results").glob("*.json")}
    direct = {read(p)["qa_id"]: read(p) for p in (root / "direct/responses").glob("*.json")}
    expected = {v["qa_id"] for v in experiment["variants"]}
    assert set(system) == set(direct) == expected
    assert Counter(r["question_type"] for r in system.values()) == experiment["task_counts"]
    state = read(root / "requests/state.json")
    assert state["policy"]["max_inflight"] == experiment["runtime"]["request_concurrency"]
    assert state["peak_inflight"] <= state["policy"]["max_inflight"]
    requests = state["requests"]
    assert all(r["status"] == "completed" and r["exception_type"] is None for r in requests)
    system_requests = [r for r in requests if not r["context"]["variant_id"].startswith("baseline:")]
    direct_requests = [r for r in requests if r["context"]["variant_id"].startswith("baseline:")]
    assert len(direct_requests) == experiment["n_questions"]
    assert len(system_requests) <= experiment["comparison"]["system_request_cap"]
    labels = load_label_metadata(root / "inputs/host_evaluation/metadata.jsonl")
    questions = {row["qa_id"]: row for row in read(root / "inputs/sample_questions.json")}
    pairs = []
    for qa in sorted(expected):
        s, d, label = system[qa], direct[qa], labels[qa]
        assert s["scoring_status"] in {"reference_unavailable", "incomplete"}
        assert s["score"] is None and not s["service_errors"]
        assert d["reference_answer_visible"] is False and d["tool_calls"] == 0
        actual = d["actual_request_identity"]["parameters"]
        assert actual["enable_thinking"] is True
        assert actual["seed"] == 137 and actual["temperature"] == 0
        assert actual["max_tokens"] == 8192
        key = _variant_key(s["variant_id"])
        plan = read(root / "system/variants" / key / "plan.json")
        outcome = read(root / "system/variants" / key / "outcome.json")
        assert outcome["initial_skill_selection"]["evidence_state"] == "g1m1"
        assert outcome["scoring_status"] in {"reference_unavailable", "incomplete"}
        assert outcome["score"] is None and outcome["correct"] is None
        assert outcome["first_synthesis"]["prompt_compaction_version"] == COMPACTION_VERSION
        assert all(r["prompt_compaction_version"] == COMPACTION_VERSION for r in outcome["rounds"])
        transport = plan["published_transport"]
        receipt = read(root / "direct/preflight" / (digest(qa)[:24] + ".json"))["input_receipt"]
        for field in ("qa_id", "n_images", "frame_set_hash", "source_frame_indices",
                      "pixel_sha256", "source_catalog_sha256"):
            assert transport[field] == receipt[field]
        assert transport["frozen_scene"]["sha256"] == receipt["frozen_scene_sha256"]
        assert receipt["n_images"] == 32 and receipt["reference_answer_visible"] is False
        seed_rows = [read(p) for p in (root / "system/request_seeds").glob(key + "*.json")]
        input_rows = [r for r in seed_rows if "request_identity" in r]
        output_rows = [r for r in seed_rows if "request_identity_sha256" in r]
        assert len(input_rows) == len(output_rows) == s["model_request_count"]
        assert 1 <= len(input_rows) <= 7
        for row in input_rows:
            params = row["request_identity"]["parameters"]
            assert row["request_seed"] == 137 and row["retry_of"] is None
            assert params["enable_thinking"] is True and params["temperature"] == 0
        for row in output_rows:
            assert row["status"] == "completed" and row["exception_type"] is None
            assert row["actual_request_identity"]["parameters"]["enable_thinking"] is True
        scores = {}
        for arm, row in (("system", s), ("direct", d)):
            scores[arm] = score_answer(
                answer=row["answer"], task=label["question_type"], options=label["options"],
                reference=label["reference_answer"],
                available=row["status"] == "completed" and row["final_state"] == "answer")
        pairs.append({
            "qa_id": qa, "question_type": label["question_type"], "question": questions[qa]["question"],
            "system_answer": s["answer"], "direct_answer": d["answer"],
            "reference_answer": label["reference_answer"],
            "system_score": scores["system"]["score"], "direct_score": scores["direct"]["score"],
            "delta": scores["system"]["score"] - scores["direct"]["score"],
            "system_evaluation": scores["system"], "direct_evaluation": scores["direct"],
            "same_frozen_rgb_identity": True, "system_model_requests": s["model_request_count"],
            "system_tool_calls": s["tool_call_count"], "system_rounds": s["agent_rounds"],
            "system_elapsed_s": s["elapsed_s"], "system_answer_source": s["answer_source"],
            "initial_prompt_characters": len(outcome.get("m8_prompt") or ""),
        })
    groups = {}
    for task, n in experiment["task_counts"].items():
        rows = [r for r in pairs if r["question_type"] == task]
        assert len(rows) == n
        groups[task] = {
            "name": NAMES[task], "n": n, "metric": "MRA" if task == "object_counting" else "Accuracy",
            "system_score": statistics.mean(r["system_score"] for r in rows),
            "direct_score": statistics.mean(r["direct_score"] for r in rows),
            "delta": statistics.mean(r["delta"] for r in rows),
        }
    usage = {}
    for arm, rows in (("system", system_requests), ("direct", direct_requests)):
        counter = Counter()
        for row in rows:
            counter.update({k: v for k, v in row["usage"].items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})
        usage[arm] = dict(counter)
    summary = {
        "status": "complete", "n_pairs": len(pairs),
        "system_score": statistics.mean(r["system_score"] for r in pairs),
        "direct_score": statistics.mean(r["direct_score"] for r in pairs),
        "delta": statistics.mean(r["delta"] for r in pairs), "by_task": groups,
        "improved": sum(r["delta"] > 0 for r in pairs), "decreased": sum(r["delta"] < 0 for r in pairs),
        "tied": sum(r["delta"] == 0 for r in pairs),
        "system_requests": len(system_requests), "direct_requests": len(direct_requests),
        "system_tool_calls": sum(r["system_tool_calls"] for r in pairs), "token_usage": usage,
        "configured_api_concurrency": state["policy"]["max_inflight"],
        "observed_request_peak": state["peak_inflight"],
        "both_enable_thinking": True, "seed": 137, "temperature": 0,
        "system_scene_overview": True, "direct_scene_overview": False,
        "diagnostic_identity": experiment["identity_sha256"],
    }
    result = {"summary": summary, "pairs": pairs}
    (root / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_data = args.report.with_suffix(".json")
    report_data.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 精简提示词工具臂与 Qwen3.8-Flash 直答配对实验", "",
        f"共 {len(pairs)} 道题。工具臂 {summary['system_score']:.2%}，"
        f"直答 {summary['direct_score']:.2%}，差值 {summary['delta'] * 100:+.2f} 个百分点。", "",
        "| 题型 | 数量 | 指标 | 精简工具臂 | Qwen 直答 | 差值 |",
        "|---|---:|---|---:|---:|---:|"]
    for task in ("object_rel_distance", "object_counting"):
        g = groups[task]
        lines.append(f"| {g['name']} | {g['n']} | {g['metric']} | {g['system_score']:.1%} | "
                     f"{g['direct_score']:.1%} | {g['delta'] * 100:+.1f} pp |")
    lines.extend([
        "", "每类从固定测试集按 seed 137 随机抽取 10 题，两臂使用相同题目和原始 32 帧；"
        "均为 qwen3.8-flash、thinking=true、temperature=0、max_tokens=8192。"
        "工具臂使用精简提示词、完整冻结 Skill 和场景总览；直答每题一次原始 RGB 请求。"
        "工具臂沿用最多 6 个求解轮及原配置的一次视觉 fallback。参考答案仅供离线评分。", "",
        "相对距离使用 Accuracy，物体计数使用官方 MRA；总分为两类等权均值。", "",
        f"逐题提高 {summary['improved']}，下降 {summary['decreased']}，持平 {summary['tied']}。"
        f"工具臂 {summary['system_requests']} 次模型请求、{summary['system_tool_calls']} 次工具调用；"
        f"直答 {summary['direct_requests']} 次模型请求。", "",
        f"API 并发上限 {summary['configured_api_concurrency']}，实际调度峰值 "
        f"{summary['observed_request_peak']}。Token 用量：{json.dumps(usage, ensure_ascii=False)}。", "",
        f"逐题答案及分数见 [{report_data.name}]({report_data.name})。"
        f"冻结实验身份：`{experiment['identity_sha256']}`。"])
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
