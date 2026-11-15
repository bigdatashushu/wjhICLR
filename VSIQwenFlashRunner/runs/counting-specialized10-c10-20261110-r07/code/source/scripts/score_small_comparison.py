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
    return json.loads(path.read_text(encoding="utf-8"))


def verify_root(root, content_sha256):
    experiment = read(root / "diagnostic.json")
    assert content_sha256({k: v for k, v in experiment.items()
                           if k != "identity_sha256"}) == experiment["identity_sha256"]
    collection = read(root / "collection_receipt.json")
    assert collection["diagnostic_identity"] == experiment["identity_sha256"]
    for relative, identity in collection["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == identity["sha256"]
    for relative, sha in experiment["input_files"].items():
        assert hashlib.sha256((root / "inputs" / relative).read_bytes()).hexdigest() == sha
    for row in read(root / "code/code_identity.json")["files"]:
        assert hashlib.sha256((root / "code/source" / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    return experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--direct-run-root", type=Path,
                        help="Reuse a fully completed direct arm from a separately collected run.")
    args = parser.parse_args()
    root = args.run_root.resolve()
    sys.path.insert(0, str(root / "code/source/src"))
    from skill3d.evaluation.fitset_v11 import (
        content_sha256, load_label_metadata, score_answer, _variant_key)
    from skill3d.infra.experiment_campaign import digest
    from skill3d.synthesis.prompt_compaction import COMPACTION_VERSION
    assert COMPACTION_VERSION == "vsi-prompt-compaction/1.0"
    experiment = verify_root(root, content_sha256)
    assert read(root / "exit.json")["status"] == "complete"
    direct_root = args.direct_run_root.resolve() if args.direct_run_root else root
    direct_experiment = experiment
    if direct_root != root:
        direct_experiment = verify_root(direct_root, content_sha256)
        assert direct_experiment["variants"] == experiment["variants"]
        assert all(direct_experiment["input_files"][key] == sha
                   for key, sha in experiment["input_files"].items() if key != "panel.json")
        assert read(direct_root / "direct_exit.json")["exit_code"] == 0
        assert read(direct_root / "direct_receipt.json")["status"] == "complete"
        assert read(direct_root / "direct_start.json")["automatic_model_retry"] is False
    system_identity = read(root / "system/manifest.json")
    assert system_identity["reference_answer_visible_to_online_model"] is False
    assert system_identity["online_executor_input_is_label_free"] is True
    system_summary = read(root / "system/summary.json")
    assert system_summary["status"] == "complete"
    assert system_summary["completed_variants"] == experiment["n_questions"]
    scheduler = read(root / "system/episode_pool_receipt.json")
    assert scheduler["status"] == "drained"
    direct_identity = read(direct_root / "direct/manifest.json")
    assert direct_identity["campaign_identity"] == direct_experiment["identity_sha256"]
    assert direct_identity["enable_thinking"] is direct_identity["rgb_only"] is True
    assert direct_identity["max_requests_per_question"] == 1
    assert direct_identity["model"] == "qwen3.8-flash"
    system = {read(p)["qa_id"]: read(p) for p in (root / "system/results").glob("*.json")}
    direct = {read(p)["qa_id"]: read(p) for p in (direct_root / "direct/responses").glob("*.json")}
    expected = {v["qa_id"] for v in experiment["variants"]}
    assert set(system) == set(direct) == expected
    assert Counter(r["question_type"] for r in system.values()) == experiment["task_counts"]
    state = read(root / "requests/state.json")
    assert state["policy"]["max_inflight"] == experiment["runtime"]["request_concurrency"]
    assert state["peak_inflight"] <= state["policy"]["max_inflight"]
    requests = state["requests"]
    assert all(r["status"] == "completed" and r["exception_type"] is None for r in requests)
    system_requests = [r for r in requests if not r["context"]["variant_id"].startswith("baseline:")]
    direct_state = state if direct_root == root else read(direct_root / "requests/state.json")
    assert direct_state["policy"]["max_inflight"] == state["policy"]["max_inflight"]
    assert direct_state["peak_inflight"] <= direct_state["policy"]["max_inflight"]
    direct_requests = [r for r in direct_state["requests"]
                       if r["context"]["variant_id"].startswith("baseline:")]
    assert all(r["status"] == "completed" and r["exception_type"] is None
               for r in direct_requests)
    if direct_root != root:
        assert len(direct_state["requests"]) == len(direct_requests)
        assert len(requests) == len(system_requests)
    assert len(direct_requests) == experiment["n_questions"]
    assert len(system_requests) <= experiment["comparison"]["system_request_cap"]
    health = read(root / "gateway_final_health.json")
    assert health["identity"] == read(root / "solve_health.json")["identity"]
    assert health["identity"] == read(direct_root / "direct_health.json")["identity"]
    upstream = health["upstream"]
    assert upstream["started"] == upstream["succeeded"] == len(system_requests) + len(direct_requests)
    assert upstream["active"] == upstream["failed"] == upstream["unknown"] == 0
    assert upstream["peak"] <= health["max_concurrency"] == state["policy"]["max_inflight"]
    labels = load_label_metadata(root / "inputs/host_evaluation/metadata.jsonl")
    questions = {row["qa_id"]: row for row in read(root / "inputs/sample_questions.json")}
    pairs = []
    for qa in sorted(expected):
        s, d, label = system[qa], direct[qa], labels[qa]
        assert s["scoring_status"] in {"reference_unavailable", "incomplete"}
        assert s["score"] is None
        tool_faults = s["service_errors"]
        if tool_faults:
            assert s["status"] == "incomplete"
            assert all(error["tool"] == "detect_objects"
                       and "endpoint is not configured / prompt is empty / frame is None"
                       in error["error"] for error in tool_faults)
        assert d["reference_answer_visible"] is False and d["tool_calls"] == 0
        assert d["campaign_identity"] == direct_experiment["identity_sha256"]
        actual = d["actual_request_identity"]["parameters"]
        assert actual["enable_thinking"] is True
        assert actual["seed"] == 137 and actual["temperature"] == 0
        assert actual["max_tokens"] == 8192
        assert actual["model"] == "qwen3.8-flash"
        key = _variant_key(s["variant_id"])
        plan = read(root / "system/variants" / key / "plan.json")
        outcome = read(root / "system/variants" / key / "outcome.json")
        assert outcome["initial_skill_selection"]["evidence_state"] == "g1m1"
        assert outcome["scoring_status"] in {"reference_unavailable", "incomplete"}
        assert outcome["score"] is None and outcome["correct"] is None
        assert [error for row in outcome["rounds"]
                for error in row.get("service_errors", [])] == tool_faults
        assert outcome["first_synthesis"]["prompt_compaction_version"] == COMPACTION_VERSION
        assert all(r["prompt_compaction_version"] == COMPACTION_VERSION for r in outcome["rounds"])
        first_images = outcome["first_synthesis"]["image_round"]
        source_jpegs = [sha for image_id, sha in zip(
            first_images["image_ids"], first_images["jpeg_sha256"])
            if image_id.startswith("frame-")]
        assert len(source_jpegs) == 32
        assert source_jpegs == [r["jpeg_sha256"]
                                for r in d["actual_request_identity"]["images_in_order"]]
        transport = plan["published_transport"]
        receipt = read(direct_root / "direct/preflight" / (digest(qa)[:24] + ".json"))["input_receipt"]
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
            assert params["seed"] == 137 and params["max_tokens"] == 8192
            assert params["model"] == "qwen3.8-flash"
        for row in output_rows:
            assert row["status"] == "completed" and row["exception_type"] is None
            params = row["actual_request_identity"]["parameters"]
            assert params["enable_thinking"] is True and params["temperature"] == 0
            assert params["seed"] == 137 and params["max_tokens"] == 8192
            assert params["model"] == "qwen3.8-flash"
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
            "system_status": s["status"], "system_service_errors": tool_faults,
            "system_scored_as_unavailable": s["status"] != "completed",
            "initial_prompt_characters": (len(outcome["m8_prompt"])
                                          if outcome.get("m8_prompt") else None),
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
        "status": ("complete_with_tool_faults"
                   if any(r["system_service_errors"] for r in pairs) else "complete"),
        "n_pairs": len(pairs),
        "system_tool_fault_questions": [
            r["qa_id"] for r in pairs if r["system_service_errors"]],
        "system_score": statistics.mean(r["system_score"] for r in pairs),
        "direct_score": statistics.mean(r["direct_score"] for r in pairs),
        "delta": statistics.mean(r["delta"] for r in pairs), "by_task": groups,
        "improved": sum(r["delta"] > 0 for r in pairs), "decreased": sum(r["delta"] < 0 for r in pairs),
        "tied": sum(r["delta"] == 0 for r in pairs),
        "system_requests": len(system_requests), "direct_requests": len(direct_requests),
        "system_tool_calls": sum(r["system_tool_calls"] for r in pairs), "token_usage": usage,
        "configured_api_concurrency": state["policy"]["max_inflight"],
        "observed_request_peak": max(state["peak_inflight"], direct_state["peak_inflight"]),
        "observed_upstream_peak": upstream["peak"],
        "both_enable_thinking": True, "seed": 137, "temperature": 0,
        "system_scene_overview": True, "direct_scene_overview": False,
        "diagnostic_identity": experiment["identity_sha256"],
        "direct_diagnostic_identity": direct_experiment["identity_sha256"],
        "direct_results_reused": direct_root != root,
    }
    result = {"summary": summary, "pairs": pairs}
    (root / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_data = args.report.with_suffix(".json")
    report_data.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 精简提示词工具臂与 Qwen3.8-Flash 直答配对实验", "",
        f"共 {len(pairs)} 道题。工具臂 {summary['system_score']:.2%}，"
        f"直答 {summary['direct_score']:.2%}，差值 {summary['delta'] * 100:+.2f} 个百分点。", "",
        "| 题型 | 数量 | 指标 | 精简工具臂 | Qwen 直答 | 差值 |",
        "|---|---:|---|---:|---:|---:|"]
    for task in ("object_rel_distance", "object_counting"):
        if task not in groups:
            continue
        g = groups[task]
        lines.append(f"| {g['name']} | {g['n']} | {g['metric']} | {g['system_score']:.1%} | "
                     f"{g['direct_score']:.1%} | {g['delta'] * 100:+.1f} pp |")
    lines.extend([
        "", ("沿用原冻结样本的相同十题。" if experiment.get("prior_sample_sha256") else
             "每类从固定测试集按 seed 137 随机抽取 10 题。")
        + "两臂使用相同题目和原始 32 帧；"
        "均为 qwen3.8-flash、thinking=true、temperature=0、max_tokens=8192。"
        "工具臂使用精简提示词、完整冻结 Skill 和场景总览；直答每题一次原始 RGB 请求。"
        "工具臂沿用最多 6 个求解轮及原配置的一次视觉 fallback。参考答案仅供离线评分。", "",
        ("物体计数使用官方 MRA；总分为十题均值。" if set(groups) == {"object_counting"} else
         "相对距离使用 Accuracy，物体计数使用官方 MRA；总分为各题型等权均值。"), "",
        f"逐题提高 {summary['improved']}，下降 {summary['decreased']}，持平 {summary['tied']}。"
        f"工具臂 {summary['system_requests']} 次模型请求、{summary['system_tool_calls']} 次工具调用；"
        f"直答 {summary['direct_requests']} 次模型请求。", "",
        f"API 并发上限 {summary['configured_api_concurrency']}，实际调度峰值 "
        f"{summary['observed_request_peak']}，上游实际峰值 {summary['observed_upstream_peak']}。"
        f"Token 用量：{json.dumps(usage, ensure_ascii=False)}。", "",
        f"逐题答案及分数见 [{report_data.name}]({report_data.name})。"
        f"冻结实验身份：`{experiment['identity_sha256']}`。"])
    if direct_root != root:
        lines.extend(["", f"直答结果复用自 `{direct_root.name}`（20 题全部完成，未重复调用）。"
                      f"直答身份：`{direct_experiment['identity_sha256']}`。"])
    if summary["system_tool_fault_questions"]:
        lines.extend([
            "", "工具运行故障："
            + "、".join(summary["system_tool_fault_questions"])
            + " 的 detect_objects 返回检测服务地址未配置或输入为空的错误。"
            "这些题保留在配对统计中，沿用运行器的 incomplete 状态按 0 分计；"
            "逐题 JSON 保留原始错误、最终答案和状态。上游模型请求全部完成。"])
    lines.extend([
        "", "| QA | 题型 | 工具答案 | 直答 | 参考答案 | 工具得分 | 直答得分 | 工具状态 |",
        "|---|---|---|---|---|---:|---:|---|"])
    for row in pairs:
        answers = [str(row[key]).replace("|", "\\|").replace("\n", " ")
                   for key in ("system_answer", "direct_answer", "reference_answer")]
        lines.append(f"| {row['qa_id']} | {NAMES[row['question_type']]} | "
                     f"{answers[0]} | {answers[1]} | {answers[2]} | "
                     f"{row['system_score']:.1%} | {row['direct_score']:.1%} | "
                     f"{row['system_status']} |")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
