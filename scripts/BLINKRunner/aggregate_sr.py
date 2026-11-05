"""Merge independently verified disjoint SR20 and remaining80 into internal_test100."""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr
from sr_trial import now, split_manifest


def aggregate(first, remaining, output, host_data_root):
    reports = [sr.read(p / "results_and_full_trajectories.json") for p in (first, remaining)]
    manifests = [sr.read(p / "raw/manifest.json") for p in (first, remaining)]
    split = reports[0]["split"]
    if (reports[1]["split"] != split
            or split_manifest(host_data_root) != {k: v for k, v in split.items()
                                                 if k != "content_sha256"}):
        raise ValueError("aggregate_dataset_or_split_changed")
    permitted = {"scripts/BLINKRunner/sr_trial.py", "scripts/BLINKRunner/sr_start.py"}
    a, b = [m["inspection"] for m in manifests]
    if (a["configuration"] != b["configuration"] or a["skill_sha256"] != b["skill_sha256"]
            or a["protocol"] != b["protocol"]
            or {k: v for k, v in a["sources"].items() if k not in permitted} !=
               {k: v for k, v in b["sources"].items() if k not in permitted}):
        raise ValueError("aggregate_model_skill_or_scientific_source_changed")
    sources, pairs = [], {}
    for directory, report, manifest in zip((first, remaining), reports, manifests):
        verified = sr.read(directory / "local_verified.json")
        geometry = sr.read(directory / "geometry_audit.json")
        archive_sha = geometry.get("prepared_archive_sha256")
        if archive_sha is None:
            archive_sha = sr.read(directory / "diagnostic_details.json")["prepared_archive_sha256"]
        if (report["audit"]["status"] != "passed" or verified["status"] != "passed"
                or sr.file_sha(directory / "prepared_evidence.tar.gz") != archive_sha
                or verified["manifest_sha256"] != manifest["content_sha256"]
                or report["manifest_sha256"] != manifest["content_sha256"]
                or {r["sample_id"]: r["artifact_sha256"] for r in geometry["inputs"]} !=
                   {r["problem"]["sample_id"]: r["prepared_artifact_sha256"]
                    for r in manifest["inspection"]["inputs"]}):
            raise ValueError("aggregate_component_audit_missing")
        expected = (split["diagnostic20"] if directory == first else
                    [sid for sid in split["internal_test"] if sid not in split["diagnostic20"]])
        if [p["sample_id"] for p in report["pairs"]] != expected:
            raise ValueError("aggregate_component_selection_changed")
        for pair in report["pairs"]:
            sid = pair["sample_id"]
            if sid in pairs:
                raise ValueError("aggregate_duplicate_pair")
            pairs[sid] = pair
        sources.append({
            "directory": str(directory), "questions": len(report["pairs"]),
            "report_sha256": report["content_sha256"],
            "manifest_sha256": manifest["content_sha256"],
            "local_verification_sha256": verified["content_sha256"],
            "geometry_audit_sha256": geometry["content_sha256"],
            "paid_requests": report["budget"]["reserved_requests"],
            "comparison_requests": report["audit"]["checked_payloads"],
            "excluded_historical_requests": report["audit"].get("excluded_parent_tool_requests", 0)})
    if set(pairs) != set(split["internal_test"]) or set(pairs) & set(split["train"]):
        raise ValueError("aggregate_internal_test_denominator_changed")
    ordered = [pairs[sid] for sid in split["internal_test"]]
    groups, wrong = {}, []
    for arm in ("direct", "skill_tools"):
        trajectories = [p[arm]["trajectory"] for p in ordered]
        calls = [call for t in trajectories for cell in t["execution_trace"]
                 for call in cell["tool_calls"]]
        errors = [
            {"sample_id": t["sample_id"], "status": row["status"],
             "final_status": t["status"], "recovered": t["status"] == "answered"}
            for t in trajectories for row in t["rounds"]
            if row["status"] in {"program_parse_error", "program_rejected", "run_error",
                                 "program_missing_action"}]
        usage = Counter()
        for t in trajectories:
            for request in t["request_trace"]:
                usage.update({k: request["response_json"]["usage"].get(k, 0)
                              for k in ("prompt_tokens", "completion_tokens", "total_tokens")})
        groups[arm] = {
            "questions": 100, "correct": sum(p[arm]["correct"] for p in ordered),
            "accuracy": sum(p[arm]["correct"] for p in ordered) / 100,
            "legal_answers": sum(t["status"] == "answered" for t in trajectories),
            "statuses": dict(Counter(t["status"] for t in trajectories)),
            "requests": sum(t["request_count"] for t in trajectories),
            "attempted_tool_calls": sum(t["attempted_tool_calls"] for t in trajectories),
            "succeeded_tool_calls": sum(t["succeeded_tool_calls"] for t in trajectories),
            "tool_using_questions": sum(bool(t["attempted_tool_calls"]) for t in trajectories),
            "tool_calls_by_name": dict(Counter(c.get("tool", c.get("name")) for c in calls)),
            "program_error_rounds": errors, "token_usage": dict(usage),
            "family_scores": {
                family: {"questions": len(ps), "correct": sum(p[arm]["correct"] for p in ps)}
                for family in sorted({p["relation"]["family"] for p in ordered})
                for ps in [[p for p in ordered if p["relation"]["family"] == family]]}}
    outcomes = Counter(
        "both_correct" if p["direct"]["correct"] and p["skill_tools"]["correct"] else
        "direct_only" if p["direct"]["correct"] else
        "skill_only" if p["skill_tools"]["correct"] else "both_wrong" for p in ordered)
    for p in ordered:
        if not p["direct"]["correct"] or not p["skill_tools"]["correct"]:
            wrong.append({"sample_id": p["sample_id"], "question": p["question"],
                          "reference": p["reference"],
                          **{arm: p[arm]["trajectory"]["answer"] for arm in groups}})
    output.mkdir(exist_ok=False)
    full = sr.write_once(output / "results_and_full_trajectories.json", {
        "schema": "blink-sr-internal-test100-report/1", "at": now(),
        "split": split, "model": "qwen3.8-flash", "configuration": a["configuration"],
        "skill_sha256": a["skill_sha256"], "training_evolution": False,
        "groups": groups, "paired_outcomes": dict(outcomes), "components": sources,
        "comparison_requests": sum(s["comparison_requests"] for s in sources),
        "total_paid_requests": sum(s["paid_requests"] for s in sources),
        "excluded_historical_requests": sum(s["excluded_historical_requests"] for s in sources),
        "audit": {"status": "passed", "checked_episodes": 200,
                  "same_model_skill_and_scientific_sources": True,
                  "exact_internal_test100_coverage": True, "duplicate_pairs": 0,
                  "training_questions_used": 0, "reference_answer_visible_during_solving": False,
                  "full_program_tool_and_authorization_trace": True,
                  "automatic_retry": False},
        "pairs": ordered})
    sr.write_once(output / "diagnostic_details.json", {
        "report_sha256": full["content_sha256"], "groups": groups, "wrong_pairs": wrong,
        "components": [{k: r[k] for k in ("groups", "paired_outcomes", "audit")} for r in reports]})
    d, t = groups["direct"], groups["skill_tools"]
    text = (
        "# BLINK SR：完整 100 题内部测试\n\n"
        "官方 val 共 143 题，seed=137 随机划分为训练 43 题、内部测试 100 题。"
        "本结果合并已核验的首批 20 题和剩余 80 题；没有重复题，训练集未参与 Skill 演化。\n\n"
        "| 方法 | 正确 | 准确率 | 合法答案 | 模型请求 | 成功工具调用 |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| Qwen3.8-Flash 直跑 | {d['correct']}/100 | {d['accuracy']:.0%} | "
        f"{d['legal_answers']}/100 | {d['requests']} | 0 |\n"
        f"| 当前 SR Skill + 工具 | {t['correct']}/100 | {t['accuracy']:.0%} | "
        f"{t['legal_answers']}/100 | {t['requests']} | {t['succeeded_tool_calls']} |\n\n"
        f"SR 相对直跑 {(t['accuracy'] - d['accuracy']) * 100:+.0f} 个百分点。"
        f"配对：两组均正确 {outcomes['both_correct']} 题、仅直跑正确 {outcomes['direct_only']} 题、"
        f"仅 SR 正确 {outcomes['skill_only']} 题、两组均错误 {outcomes['both_wrong']} 题。\n\n"
        f"SR 在 {t['tool_using_questions']}/100 题中实际调用工具；调用分布 "
        f"{t['tool_calls_by_name']}。程序错误回合 {len(t['program_error_rounds'])} 个，"
        f"最终状态分布 {t['statuses']}。\n\n"
        "两组统一使用 qwen3.8-flash、thinking=true、temperature=0、seed=137、"
        "max_tokens=8192。直跑每题 1 请求，SR 每题最多 3 请求，模型并发 8。"
        "几何预计算保持 DINO、SAM2、Intern 类别确认及 MoGe2 相对深度，"
        "每个对象最多 2 候选、总计最多 4 候选。允许工具为 "
        "view_candidates、view_depth_summary、point_depth。\n\n"
        "200 条 episode 的实际请求、原图字节、原始模型响应与思考、程序、工具输入输出、"
        "授权回执和预算均已核验。完整轨迹位于 results_and_full_trajectories.json；"
        "两批逐文件原始回执与几何归档在各自目录中保留。\n\n"
        f"当前有效比较使用 {full['comparison_requests']} 次模型请求。含首批修正前已保留的 "
        f"{full['excluded_historical_requests']} 次历史工具请求，累计实际请求 "
        f"{full['total_paid_requests']} 次。历史工具结果不计入成绩。\n\n"
        "该结果衡量固定初始 Skill 在这 100 道内部测试题上的表现；"
        "训练集演化后的效果需要另行评测。\n")
    (output / "report.md").write_text(text)
    return {"groups": groups, "paired_outcomes": dict(outcomes),
            "report_sha256": full["content_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--remaining", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=sr.DEFAULT_ROOT)
    args = parser.parse_args()
    import json
    print(json.dumps(aggregate(args.first.resolve(), args.remaining.resolve(),
                               args.output.resolve(), args.data_root), ensure_ascii=False))
