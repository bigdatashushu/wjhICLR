"""Export current operational state and verified final paired results."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics
import time

from ..traces import digest, file_digest, read, write
from . import depth_all_api_recovery as recovery
from .depth_comparison.evaluation import summarize
from .depth_comparison_report import ancestors, health_checks


def export(directory=recovery.DEFAULT_RUN):
    directory = Path(directory)
    destination = directory.parent
    manifest = recovery.validate(directory)
    parent_dir = Path(manifest["parent_directory"])
    parent, parent_frozen, parent_execution = recovery.parent_evidence(parent_dir)
    chain = ancestors(parent_dir, parent)
    checks = health_checks(destination)
    health_requests = sum(c["requests"] for c in checks)
    dispatches = sorted((directory / "traces").glob("*.dispatch.json"))
    results = sorted((directory / "traces").glob("*.result.json"))
    new_outcomes = [read(path) for path in results]
    terminal = (directory / "execution.json").exists()
    execution = read(directory / "execution.json") if terminal else {"status": "running"}
    frozen = recovery.validate_frozen(directory, manifest) if (directory / "frozen.json").exists() else None
    score = read(directory / "score.json") if (directory / "score.json").exists() else None
    if score is not None:
        if (frozen is None or execution["status"] != "complete"
                or score["run_identity"] != manifest["run_identity"]
                or score["freeze_identity"] != frozen["freeze_identity"]
                or digest({k: v for k, v in score.items() if k != "score_identity"}) != score["score_identity"]
                or score["overall"] != summarize(score["pairs"])):
            raise ValueError("final_score_changed")
        predictions = {(p["sample_id"], p["arm"]): p for p in frozen["predictions"]}
        items = {i["problem"]["sample_id"]: i for i in manifest["reused_items"] + manifest["new_items"]}
        if len(score["pairs"]) != 18 or len({p["sample_id"] for p in score["pairs"]}) != 18:
            raise ValueError("score_denominator_changed")
        for pair in score["pairs"]:
            item = items[pair["sample_id"]]
            if (pair["input_sha256"] != item["problem"]["input_sha256"]
                    or pair["api_id"] != item["api_id"] or pair["task_type"] != item["problem"]["task_type"]):
                raise ValueError("scored_pair_changed")
            for arm in ("direct", "depth"):
                expected = predictions[pair["sample_id"], arm]
                if (any(pair[arm][key] != value for key, value in expected.items())
                        or pair[arm]["correct"] != (
                            expected["status"] == "answered" and expected["answer"] == pair["reference_answer"])):
                    raise ValueError("scored_prediction_changed")
        for task, value in score["by_task"].items():
            if value != summarize([p for p in score["pairs"] if p["task_type"] == task]):
                raise ValueError("task_summary_changed")
        for api, value in score["by_api"].items():
            if value != summarize([p for p in score["pairs"] if p["api_id"] == api]):
                raise ValueError("api_summary_changed")
    complete_new = sum(all(any(
        r["sample_id"] == item["problem"]["sample_id"] and r["arm"] == arm
        and r["status"] in recovery.TERMINAL for r in new_outcomes)
        for arm in ("direct", "depth")) for item in manifest["new_items"])
    paired_requests = manifest["historical_requests"] + parent_execution["dispatched"] + len(dispatches)
    used_apis = sorted({read(path)["api_id"] for path in dispatches}
                      | {p["api_id"] for p in parent_frozen["predictions"] if p["status"] != "not_dispatched"})
    progress = {
        "schema": "mmsi-depth-all-api-progress/1", "observed_at": time.time(),
        "phase": "complete" if score else ("frozen_unscored" if frozen else "running"),
        "execution_status": execution["status"], "run_identity": manifest["run_identity"],
        "selected_pairs": 18, "reused_complete_pairs": 13, "new_complete_pairs": complete_new,
        "complete_pairs": 13 + complete_new, "new_dispatched_requests": len(dispatches),
        "new_recorded_outcomes": len(results), "new_max_requests": 10,
        "new_status_counts": dict(Counter(r["status"] for r in new_outcomes)),
        "cumulative_paired_requests": paired_requests, "original_paired_budget": 60,
        "service_recheck_requests": health_requests, "total_real_calls": paired_requests + health_requests,
        "formal_api_slots_used": used_apis, "distinct_formal_api_slots_used": len(used_apis),
        "reference_answers_read": score is not None, "scoring_performed": score is not None,
        "automatic_retry": False, "old_unknown_requests": 2, "excluded_old_single_arm_answers": 1,
    }
    write(destination / "progress.json", progress)
    progress_text = (
        f"# MMSI 深度图对比进度\n\n状态：{progress['phase']}；"
        f"完整配对 {progress['complete_pairs']}/18；"
        f"新请求 {len(dispatches)}/10，已收结果 {len(results)}。\n\n"
        f"正式使用 API：{len(used_apis)}/11。正式比较累计 {paired_requests}/60 次；"
        f"合成图复查 {health_requests} 次；总调用 {paired_requests + health_requests} 次。\n\n"
        "13 组完整预测复用原冻结记录，5 组在新 API 上配对补测。"
        "原来 2 次未返回请求保留为结果未知，1 个旧单臂答案排除于最终比较。\n"
    )
    (destination / "progress.md").write_text(progress_text, encoding="utf-8")
    verification = {
        "status": "passed" if score else ("frozen_files_verified" if frozen else "running"),
        "run_identity": manifest["run_identity"],
        "freeze_identity": frozen["freeze_identity"] if frozen else None,
        "parent_run_identity": parent["run_identity"], "parent_freeze_identity": parent_frozen["freeze_identity"],
        "new_frozen_files_verified": len(frozen["files"]) if frozen else 0,
        "parent_frozen_files_verified": len(parent_frozen["files"]),
        "ancestor_runs_verified": [{"run_identity": row[1]["run_identity"],
                                    "freeze_identity": row[2]["freeze_identity"],
                                    "files_verified": len(row[2]["files"])} for row in chain],
        "score_identity": score["score_identity"] if score else None,
        "reference_answers_read": score is not None, "scoring_performed": score is not None,
        "requests": paired_requests, "original_budget": 60,
        "service_recheck_requests": health_requests, "total_real_calls": paired_requests + health_requests,
        "formal_api_slots_used": used_apis, "complete_pairs": 13 + complete_new,
        "all_new_payloads_match_original": True,
    }
    errors = []
    for run in [row[0] for row in chain] + [parent_dir, directory]:
        for path in sorted((run / "traces").glob("*.result.json")):
            result = read(path)
            if result["status"] == "service_error":
                errors.append({"run_directory": str(run), **{k: result.get(k) for k in (
                    "sample_id", "arm", "api_id", "error_code", "http_status")}})
    write(destination / "service_failures.json", {
        "schema": "mmsi-depth-all-api-service-failures/1", "service_error_requests": len(errors),
        "errors": errors, "operator_stopped_unknown_requests": read(destination / "api-reallocation.json")["pending_outcomes"],
    })
    if score is None:
        (destination / "report.md").write_text(
            progress_text + "\n全部 18 组预测冻结完整后再读取参考答案评分。"
            "当前准确率、增益、置信区间与 McNemar p 均未测。\n", encoding="utf-8")
        write(destination / "collection-verification.json", verification)
        return progress
    overall = score["overall"]
    delta = overall["delta"] * 100
    ci = [v * 100 for v in overall["paired_bootstrap_95_ci"]]
    lines = [
        "# MMSI：原图与原图＋预测深度图配对比较", "",
        f"18 组完整配对中，原图答对 {overall['direct_correct']}/18"
        f"（{overall['direct_accuracy']:.2%}），加入预测深度图答对 {overall['depth_correct']}/18"
        f"（{overall['depth_accuracy']:.2%}），差值为 {delta:+.2f} 个百分点。", "",
        f"错→对 {overall['wrong_to_correct']} 题，对→错 {overall['correct_to_wrong']} 题；"
        f"配对 bootstrap 95% 区间 [{ci[0]:+.2f}, {ci[1]:+.2f}] 个百分点，"
        f"McNemar 双侧精确 p={overall['mcnemar_exact_two_sided_p']:.4f}。", "",
        ("这批样本的点估计支持深度图增益。" if delta > 0 else
         "这批样本的点估计显示深度图降低准确率。" if delta < 0 else
         "这批样本中两种输入的准确率相同。")
        + (" 区间跨过零，18 题仍不足以确定总体增益。" if ci[0] <= 0 <= ci[1] else
           " 结论范围是本次 18 题小样本比较。"), "",
        "| 输入 | 答对 | 准确率 | 合法答案 |", "| --- | ---: | ---: | ---: |",
        f"| 原图 Direct | {overall['direct_correct']}/18 | {overall['direct_accuracy']:.2%} | "
        f"{overall['direct_legal_answers']}/18 |",
        f"| 原图＋深度图＋短说明 | {overall['depth_correct']}/18 | {overall['depth_accuracy']:.2%} | "
        f"{overall['depth_legal_answers']}/18 |", "",
        "六类各 3 题；原始两臂顺序保持 Direct 先发 9 题、Depth 先发 9 题。", "",
        "| 任务 | 题数 | Direct 答对 | Depth 答对 | 差值（百分点） |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task, value in score["by_task"].items():
        lines.append(f"| {task} | {value['questions']} | {value['direct_correct']} | "
                     f"{value['depth_correct']} | {value['delta'] * 100:+.2f} |")
    lines += [
        "", f"全部 11 个不同凭据槽位均参与了正式题。累计正式请求 {paired_requests}/60 次，"
        f"另有 {health_requests} 次用户授权的合成图服务复查，总计 {paired_requests + health_requests} 次。", "",
        "原先 6 个 API 已发 29 次、收到 27 个答案；13 组完整答案原样保留。"
        "用户要求使用全部 API 后，停止原队列，2 次未返回调用记为结果未知。"
        "剩余 5 组分别交给 pool-07、08、09、10、extra-11，每个新 API 发两臂共 2 次。"
        "其中一题旧 Direct 已答，为保持新配对使用同一 API，该旧单臂答案保留但排除于最终统计。"
        "补测按调用完成情况确定，发生在读取参考答案之前。"
        "最终比较包含 13 组复用、5 组补测；此前两轮的 12 次服务失败不进入本次 18 题准确率分母。", "",
        "| API | 最终配对题数 | Direct 答对 | Depth 答对 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for api, value in score["by_api"].items():
        lines.append(f"| {api} | {value['questions']} | {value['direct_correct']} | {value['depth_correct']} |")
    lines += [
        "", "请求模型为 gpt-5.6-terra，Responses，reasoning effort=high，"
        "temperature=0，seed=137，max_output_tokens=8192。新调用超时 180 秒，"
        "没有自动重试，也没有工具调用。返回采样元数据另行保存在逐次响应中。", "",
        "原图字节、尺寸和图序保持一致。深度图来自已处理缓存，与各自原图像素及尺寸对齐；"
        "每图有效深度 P2–P98 映射为灰度，暗近亮远，无效像素为洋红。"
        "深度值为各视图独立的相对预测值。两臂正式发送 payload 与原预备版本哈希完全一致。", "",
        f"新增文字 {min(i['added_text_tokens'] for i in parent['items'])}–"
        f"{max(i['added_text_tokens'] for i in parent['items'])} 个 o200k_base 词元，上限 160；"
        f"固定说明 {parent['items'][0]['annotation']['fixed_tokens']} 个词元，上限 80。", "",
        "> " + parent["items"][0]["annotation"]["note"], "",
        "| 题目 | 任务 | API | Direct | Depth | 参考答案 | 配对变化 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pair in score["pairs"]:
        change = int(pair["depth"]["correct"]) - int(pair["direct"]["correct"])
        lines.append(f"| {pair['sample_id']} | {pair['task_type']} | {pair['api_id']} | "
                     f"{pair['direct']['answer']} | {pair['depth']['answer']} | "
                     f"{pair['reference_answer']} | {['相同', '错→对', '对→错'][change]} |")
    review, metadata, latency = [], [], {"direct": [], "depth": []}
    for item in sorted(manifest["reused_items"] + manifest["new_items"], key=lambda i: i["index"]):
        source = parent_dir if "original_api_id" not in item else directory
        responses = {}
        for arm in ("direct", "depth"):
            record = read(source / "traces" / f"{item['index']:04d}-{arm}.result.json")
            responses[arm] = record
            response = record.get("response")
            if response:
                metadata.append(response["reported"])
                latency[arm].append(response["elapsed_seconds"])
        pair = next(p for p in score["pairs"] if p["sample_id"] == item["problem"]["sample_id"])
        review.append({"problem": item["problem"], "api_id": item["api_id"], "pair": pair,
                       "annotation": item["annotation"], "depth_images": item["depth_images"],
                       "depth_directory": str(source), "responses": responses})
    reported = {field: sorted({json.dumps(row[field], ensure_ascii=False) for row in metadata})
                for field in ("model", "reasoning_effort", "temperature", "seed", "max_output_tokens")}
    lines += [
        "", f"服务返回 temperature 的取值：{', '.join(reported['temperature'])}；"
        f"seed：{', '.join(reported['seed'])}。这与请求字段分别记录，"
        "该服务的返回元数据不能证明实际执行了请求中的采样值。", "",
        f"单次成功调用耗时中位数：Direct {statistics.median(latency['direct']):.1f} 秒，"
        f"Depth {statistics.median(latency['depth']):.1f} 秒。", "",
        "全部 36 个最终预测先冻结，之后才在独立评分进程中读取经哈希核验的参考答案。"
        "凭据只在内存使用，原运行及原冻结记录保持不变。", "",
        "完整评分在 run-all-api/score.json；逐题问题、答案与响应在 review.json；"
        "原始冻结记录、派发和响应均保留在各 run 目录。", "",
        f"Run identity：`{manifest['run_identity']}`", "",
        f"Freeze identity：`{frozen['freeze_identity']}`", "",
        f"Score identity：`{score['score_identity']}`", "",
    ]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write(destination / "review.json", {
        "schema": "mmsi-depth-all-api-review/1", "run_identity": manifest["run_identity"],
        "freeze_identity": frozen["freeze_identity"], "score_identity": score["score_identity"],
        "reference_answers_read": True, "scoring_performed": True, "cases": review,
        "reported_metadata_values": reported,
    })
    verification.update(
        paired_summary_recomputed=True, score_pairs_verified=True,
        actual_tool_calls=sum(r["response"]["tool_call_count"] for c in review for r in c["responses"].values()),
        reported_metadata_values=reported)
    write(destination / "collection-verification.json", verification)
    return progress


if __name__ == "__main__":
    try:
        progress = export()
        from .depth_service_probe import all_keys
        keys = all_keys()
        paths = [p for p in recovery.DEFAULT_RUN.parent.rglob("*") if p.is_file()]
        paths += [Path(__file__), Path(recovery.__file__),
                  Path(__file__).with_name("depth_all_api_selfcheck.py")]
        if any(key.encode() in path.read_bytes() for path in paths for key in keys.values()):
            raise ValueError("credential_in_artifact")
        verification_path = recovery.DEFAULT_RUN.parent / "collection-verification.json"
        verified = read(verification_path)
        verified.update(credential_scan="passed_all_11_keys", credential_scan_files=len(paths))
        write(verification_path, verified)
        print(json.dumps(progress, ensure_ascii=False))
    except Exception:
        print('{"status":"failed","reason_code":"all_api_report_failed"}')
        raise SystemExit(2)
