"""Export a concise Chinese report from frozen, independently scored paired evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def verified_unscored_run(directory):
    directory = Path(directory)
    load = lambda name: json.loads((directory / name).read_bytes())
    manifest, frozen, execution = [load(name) for name in (
        "manifest.json", "frozen.json", "execution.json")]
    for value, identity in ((manifest, "run_identity"), (frozen, "freeze_identity")):
        if digest({k: v for k, v in value.items() if k != identity}) != value[identity]:
            raise ValueError("unscored_evidence_identity_changed")
    if (frozen["run_identity"] != manifest["run_identity"]
            or execution["run_identity"] != manifest["run_identity"]
            or execution["inflight"] != 0 or (directory / "score.json").exists()
            or manifest["reference_answer_visible"] is not False):
        raise ValueError("unscored_run_binding_or_isolation_changed")
    for name, expected in frozen["files"].items():
        path = directory / name
        if (not path.resolve().is_relative_to(directory.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError("unscored_frozen_file_changed")
    dispatched = sum(row["status"] != "not_dispatched" for row in frozen["predictions"])
    if dispatched != execution["dispatched"]:
        raise ValueError("unscored_dispatch_count_changed")
    return manifest, frozen, execution


def export_unscored(destination, directory=None):
    """Service failures produce operational evidence, with no accuracy or GT access."""
    destination = Path(destination)
    child_directory = Path(directory) if directory else destination / "run"
    child, child_freeze, child_execution = verified_unscored_run(child_directory)
    chain = ancestors(child_directory, child)
    chain.append((child_directory, child, child_freeze, child_execution))
    runs = [(f"第 {i + 1} 轮", *run) for i, run in enumerate(chain)]
    checks = health_checks(destination)
    health_requests = sum(check["requests"] for check in checks)
    errors, rows = [], []
    for name, directory, manifest, frozen, execution in runs:
        statuses = Counter(p["status"] for p in frozen["predictions"])
        if (execution["status"] != "stopped_service_error"
                or set(statuses) - {"service_error", "not_dispatched", "answered", "invalid_answer"}):
            raise ValueError("unexpected_unscored_run_outcome")
        for path in sorted((directory / "traces").glob("*.result.json")):
            result = json.loads(path.read_bytes())
            if result["status"] == "service_error":
                errors.append({"host": name, **{k: result.get(k) for k in (
                    "sample_id", "arm", "api_id", "status", "error_code", "http_status")}})
        predictions = {(p["sample_id"], p["arm"]): p for p in frozen["predictions"]}
        complete_pairs = sum(all(predictions[item["problem"]["sample_id"], arm]["status"]
                                 in ("answered", "invalid_answer") for arm in ("direct", "depth"))
                             for item in manifest["items"])
        rows.append(f"| {name} | {manifest['questions']} | {execution['dispatched']} | "
                    f"{statuses['service_error']} | {statuses['answered']} | {complete_pairs} |")
    total_requests = sum(execution["dispatched"] for _, _, _, _, execution in runs)
    arm_counts = Counter(p["arm"] for _, _, _, frozen, _ in runs
                         for p in frozen["predictions"] if p["status"] != "not_dispatched")
    child_predictions = {(p["sample_id"], p["arm"]): p for p in child_freeze["predictions"]}
    complete_pairs = sum(all(child_predictions[item["problem"]["sample_id"], arm]["status"]
                             in ("answered", "invalid_answer") for arm in ("direct", "depth"))
                         for item in child["items"])
    note = child["items"][0]["annotation"]["note"]
    lines = [
        "# MMSI：直接添加深度图的测试状态", "",
        f"当前无法判断深度图增益。前后 {len(runs)} 轮比较累计 {total_requests} 次真实请求，"
        f"包含 {len(errors)} 次服务错误；最新运行完整配对 {complete_pairs}/{child['questions']}，"
        "已停止派发并冻结。", "",
        "| 运行 | 预备题数 | 真实请求 | 服务错误 | 有效预测 | 完整配对 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |", *rows, "",
        f"原图请求 {arm_counts['direct']} 次，原图＋深度请求 {arm_counts['depth']} 次，"
        "准确率、增益、置信区间和 McNemar p 均为未测，"
        "服务错误不能解释为模型答错或深度图无增益。", "",
        "原计划从 365 题公共清单中抽取 30 对，六类各 5 题。每轮失败后冻结原运行，"
        f"排除所有已派发题目；最新运行仅含此前尚未发送的 {child['questions']} 对，"
        f"六类各 {child['questions']//6} 题。各轮尝试题目完全不重叠。"
        f"比较累计请求 {total_requests}/60，失败请求未重复发送。", "",
        f"用户另行要求的服务复查累计 {health_requests} 次；"
        f"全部真实推理调用累计 {total_requests + health_requests} 次。"
        "合成图服务复查不属于准确率比较样本，也不计入原定 60 次配对预算。", "",
        "请求配置为 gpt-5.6-terra、Responses、reasoning effort=high、temperature=0、"
        "seed=137、max_output_tokens=8192。每题两臂绑定同一个 API，未调用工具。"
        "逐请求服务错误码及 HTTP 状态保存在轨迹中。", "",
        "原图保持原始字节与图序。深度图来自已处理缓存，与原图像素及尺寸对齐；"
        "每图有效值 P2–P98 映射为灰度，暗近亮远，无效像素洋红。"
        "深度是每图独立的相对预测值。原始字节及两臂请求 payload "
        "均校验过完整哈希，并与首次准备版本保持一致。", "",
        "固定深度说明为 59 个 o200k_base 词元，上限 80；"
        f"本轮实际新增文字为 {min(i['added_text_tokens'] for i in child['items'])}–"
        f"{max(i['added_text_tokens'] for i in child['items'])} 词元，上限 160。", "",
        "> " + note, "",
        "没有读取或评分参考答案。凭据仅在进程内存使用。"
        "",
        "| 位置 | 题目 | 臂 | API slot | HTTP |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in errors:
        lines.append(f"| {row['host']} | {row['sample_id']} | {row['arm']} | "
                     f"{row['api_id']} | {row['http_status']} |")
    lines += ["", f"`{child_directory.relative_to(destination)}/` 保存最新输入清单、"
              "深度图、逐次派发与响应记录、预测冻结及文件哈希。",
              f"`review.json` 保存 {child['questions']} 道题目、两臂状态及已有响应记录；"
              "`service_failures.json` 汇总历次比较服务错误；没有生成 score.json。",
              "", f"Mac run identity：`{child['run_identity']}`。",
              "", f"Mac freeze identity：`{child_freeze['freeze_identity']}`。", ""]
    predictions = {(p["sample_id"], p["arm"]): p for p in child_freeze["predictions"]}
    review = []
    for item in child["items"]:
        sample = item["problem"]["sample_id"]
        responses = {}
        for arm in ("direct", "depth"):
            path = child_directory / "traces" / f"{item['index']:04d}-{arm}.result.json"
            responses[arm] = json.loads(path.read_bytes()) if path.exists() else None
        review.append({"sample_id": sample, "problem": item["problem"],
                       "annotation": item["annotation"], "depth_images": item["depth_images"],
                       "outcomes": {arm: predictions[sample, arm] for arm in ("direct", "depth")},
                       "responses": responses})
    save = lambda name, value: (destination / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    save("review.json", {"schema": "mmsi-depth-unscored-review/1",
                        "run_identity": child["run_identity"], "cases": review,
                        "reference_answers_read": False, "scoring_performed": False})
    save("service_failures.json", {"schema": "mmsi-depth-service-failures/1",
                                  "requests": len(errors), "errors": errors,
                                  "effective_predictions": sum(Counter(p["status"] for p in frozen["predictions"])["answered"]
                                                               for _, _, _, frozen, _ in runs),
                                  "complete_pairs": complete_pairs})
    save("collection-verification.json", {
        "status": "frozen_files_verified", "scientific_comparison": "unavailable_service_errors",
        "requests": total_requests, "original_budget": 60, "reference_answers_read": False,
        "service_recheck_requests": health_requests,
        "total_real_inference_calls": total_requests + health_requests, "service_checks": checks,
        "runs": [{"host": name, "run_identity": manifest["run_identity"],
                  "freeze_identity": frozen["freeze_identity"],
                  "files_verified": len(frozen["files"])}
                 for name, _, manifest, frozen, _ in runs]})
    print(json.dumps({"report": str(destination / "report.md"), "requests": total_requests,
                      "complete_pairs": complete_pairs, "accuracy": None, "gain": None}, ensure_ascii=False))


def ancestors(directory, manifest):
    """Verify each frozen failed ancestor and the cumulative continuation accounting."""
    result = []
    seen = {Path(directory).resolve()}
    current = manifest
    while "continuation" in current:
        info = current["continuation"]
        parent_directory = Path(info["parent_directory"])
        resolved = parent_directory.resolve()
        if resolved in seen:
            raise ValueError("continuation_cycle")
        seen.add(resolved)
        parent, frozen, execution = verified_unscored_run(parent_directory)
        if (parent["run_identity"] != info["parent_run_identity"]
                or frozen["freeze_identity"] != info["parent_freeze_identity"]
                or execution["status"] != "stopped_service_error"
                or execution["dispatched"] != 6
                or Counter(p["status"] for p in frozen["predictions"])
                != {"service_error": 6, "not_dispatched": 2 * parent["questions"] - 6}
                or info["parent_requests"]
                != parent.get("continuation", {}).get("parent_requests", 0) + 6
                or info["child_max_requests"] != current["max_requests"]
                or info["combined_max_requests"] != info["parent_requests"] + current["max_requests"]
                or info["combined_max_requests"] > info["original_budget"]):
            raise ValueError("continuation_ancestry_or_budget_changed")
        result.append((parent_directory, parent, frozen, execution))
        current = parent
    result.reverse()
    attempted = {p["sample_id"] for _, _, frozen, _ in result
                 for p in frozen["predictions"] if p["status"] != "not_dispatched"}
    if manifest.get("continuation"):
        if (sum(execution["dispatched"] for _, _, _, execution in result)
                != manifest["continuation"]["parent_requests"]
                or attempted != set(manifest["continuation"]["excluded_attempted_sample_ids"])
                or attempted & {i["problem"]["sample_id"] for i in manifest["items"]}):
            raise ValueError("attempted_pair_reintroduced_or_accounting_changed")
    return result


def health_checks(destination):
    checks = []
    for path in sorted(Path(destination).glob("service-recheck*.json"), key=lambda p: p.stat().st_mtime_ns):
        data = json.loads(path.read_bytes())
        if (data["reference_answers_read"] is not False
                or data["automatic_retry"] is not False
                or data["model_requests"] < 1):
            raise ValueError("health_check_isolation_or_count_changed")
        checks.append({"file": path.name, "requests": data["model_requests"],
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "successful_slots": data.get("successful_slots", [])})
    return checks


def export(destination, directory=None):
    destination = Path(destination)
    directory = Path(directory) if directory else destination / "run"
    if not (directory / "score.json").exists():
        return export_unscored(destination, directory)
    read = lambda name: json.loads((directory / name).read_bytes())
    manifest, frozen, score, execution = [read(name) for name in (
        "manifest.json", "frozen.json", "score.json", "execution.json")]
    for value, identity in ((manifest, "run_identity"), (frozen, "freeze_identity"),
                            (score, "score_identity")):
        if digest({k: v for k, v in value.items() if k != identity}) != value[identity]:
            raise ValueError("evidence_identity_changed")
    if (score["run_identity"] != manifest["run_identity"]
            or frozen["run_identity"] != manifest["run_identity"]
            or score["freeze_identity"] != frozen["freeze_identity"]
            or execution["run_identity"] != manifest["run_identity"]):
        raise ValueError("score_and_prediction_freeze_disagree")
    for name, expected in frozen["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError("collected_evidence_file_changed")
    prior_runs = ancestors(directory, manifest)
    checks = health_checks(destination)
    paired_requests = execution["dispatched"] + sum(
        previous["dispatched"] for _, _, _, previous in prior_runs)
    health_requests = sum(check["requests"] for check in checks)
    run_path = directory.relative_to(destination)
    pairs = score["pairs"]
    overall = score["overall"]
    total = manifest["questions"]
    predictions = {(row["sample_id"], row["arm"]): row for row in frozen["predictions"]}
    if (len(pairs) != total or overall["questions"] != total
            or [p["sample_id"] for p in pairs]
            != [i["problem"]["sample_id"] for i in manifest["items"]]):
        raise ValueError("score_denominator_or_sample_order_changed")
    for item, pair in zip(manifest["items"], pairs):
        if (pair["input_sha256"] != item["problem"]["input_sha256"]
                or pair["api_id"] != item["api_id"]
                or pair["task_type"] != item["problem"]["task_type"]):
            raise ValueError("pair_input_binding_changed")
        for arm in ("direct", "depth"):
            original = predictions[pair["sample_id"], arm]
            if (any(pair[arm][key] != value for key, value in original.items())
                    or pair[arm]["correct"] != (
                        original["status"] == "answered"
                        and original["answer"] == pair["reference_answer"])):
                raise ValueError("scored_answer_differs_from_frozen_prediction")
    for arm in ("direct", "depth"):
        if (sum(p[arm]["correct"] for p in pairs) != overall[arm + "_correct"]
                or dict(Counter(p[arm]["status"] for p in pairs))
                != overall[arm + "_status_counts"]
                or overall[arm + "_accuracy"] != overall[arm + "_correct"] / total
                or overall[arm + "_legal_answers"]
                != sum(p[arm]["status"] == "answered" for p in pairs)):
            raise ValueError("reported_counts_disagree_with_pair_outcomes")
    gains = sum(not p["direct"]["correct"] and p["depth"]["correct"] for p in pairs)
    losses = sum(p["direct"]["correct"] and not p["depth"]["correct"] for p in pairs)
    discordant = gains + losses
    pvalue = (min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(gains, losses) + 1))
                  / 2**discordant) if discordant else 1.0)
    if (overall["wrong_to_correct"] != gains or overall["correct_to_wrong"] != losses
            or overall["delta"] != (gains - losses) / total
            or overall["mcnemar_exact_two_sided_p"] != pvalue
            or overall["complete_pairs"] != sum(all(p[arm]["status"] in (
                "answered", "invalid_answer") for arm in ("direct", "depth")) for p in pairs)):
        raise ValueError("paired_statistics_disagree_with_frozen_outcomes")
    percent = lambda x: f"{100 * x:.1f}%"
    paired_delta = f"{100 * overall['delta']:+.1f}"
    complete = overall["complete_pairs"] == overall["questions"]
    conclusion = (
        f"在这 {total} 对题中，原图答对 {overall['direct_correct']}/{total}，"
        f"原图加深度图答对 {overall['depth_correct']}/{total}，差值 {paired_delta} 个百分点。"
        if complete else
        f"本轮完成 {overall['complete_pairs']}/{total} 对，执行状态为 {execution['status']}；"
        "服务中断或未测题使完整比较尚未完成，不能据此判断深度图增益。"
    )
    lines = [
        "# MMSI：直接添加深度图的配对测试",
        "",
        conclusion,
        "",
        f"比较范围为已有 365 题公共清单中的 {total} 道未审阅题，六类各 {total//6} 题。"
        "原图臂使用官方字母回答提示；深度臂添加对应预测深度图及必要的短说明。"
        "同题两臂固定同一个 API、模型和参数，未调用工具。",
        "",
        "| 指标 | 原图 | 原图＋深度图 |",
        "| --- | ---: | ---: |",
        f"| 正确数（固定分母） | {overall['direct_correct']}/{total} | {overall['depth_correct']}/{total} |",
        f"| 准确率 | {percent(overall['direct_accuracy'])} | {percent(overall['depth_accuracy'])} |",
        f"| 合法回答数 | {overall['direct_legal_answers']}/{total} | {overall['depth_legal_answers']}/{total} |",
        f"| 合法回答率 | {percent(overall['direct_legal_answer_rate'])} | "
        f"{percent(overall['depth_legal_answer_rate'])} |",
        "",
        f"加入深度后，错→对 {overall['wrong_to_correct']} 题，对→错 "
        f"{overall['correct_to_wrong']} 题。完整配对 {overall['complete_pairs']}/{total}。",
        "",
    ]
    if complete:
        low, high = overall["paired_bootstrap_95_ci"]
        lines += [
            f"准确率差值的配对 bootstrap 95% 区间为 [{100*low:+.1f}, {100*high:+.1f}] "
            f"个百分点；精确 McNemar 双侧 p={overall['mcnemar_exact_two_sided_p']:.4f}。",
            "",
            f"这项结果对应“深度图＋说明”的整体效果。{total} 对是小样本诊断；"
            "区间与配对转移共同用于判断观察到的差值是否稳定，不能直接概括整个 MMSI。",
            "",
        ]
    lines += [
        "| 题型 | 原图正确 | 深度正确 | 差值（题） | 完整配对 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task, value in score["by_task"].items():
        lines.append(f"| {task} | {value['direct_correct']}/{value['questions']} | "
                     f"{value['depth_correct']}/{value['questions']} | "
                     f"{value['depth_correct']-value['direct_correct']:+d} | "
                     f"{value['complete_pairs']}/{value['questions']} |")
    lines += [
        "",
        "## 执行与输入",
        "",
        "- 模型：gpt-5.6-terra；Responses；reasoning effort=high；请求 temperature=0、"
        "seed=137、max_output_tokens=8192。",
        f"- API slot 数：{len(manifest['api_ids'])}；实际峰值并发："
        f"{execution['peak_inflight']}；真实请求数：{execution['dispatched']}/{manifest['max_requests']}。",
        f"- 原图先与深度先各 {total//2} 对；六个 API 各负责 {total//6} 对，并轮换题型。",
        "- 全部原图保持原始字节与顺序。深度图与对应原图尺寸和像素对齐，"
        "每图有效深度 P2–P98 映射为灰度，暗近亮远，无效为洋红。",
        "- 深度仅是每图独立的相对预测值，不能视为米或跨图统一深度尺度。",
        f"- 固定说明 59 个 o200k_base 词元；本轮实际新增文字范围 "
        f"{min(i['added_text_tokens'] for i in manifest['items'])}–"
        f"{max(i['added_text_tokens'] for i in manifest['items'])} 词元，上限 160。",
        "- 参考答案只在预测全部冻结并验证后由独立评分进程读取。"
        "服务失败无自动重试，也未换 key 补答。",
        "",
    ]
    if "continuation" in manifest:
        info = manifest["continuation"]
        lines += [
            f"前 {len(prior_runs)} 轮的 {info['parent_requests']} 次服务失败请求保留且未重发。"
            f"本轮仅含此前从未派发的 {total} 对；比较累计请求 "
            f"{paired_requests}/{info['original_budget']}。", ""]
    if checks:
        lines += [
            f"用户另行要求的服务复查累计 {health_requests} 次，使用 32×32 红色合成图。"
            f"最近一次全部 API 复查成功 {len(checks[-1]['successful_slots'])}/11。"
            f"连同配对实验，累计真实推理调用 {paired_requests + health_requests} 次。"
            "服务复查不属于准确率比较样本，也不计入原定 60 次配对预算。", ""]
    metadata_counts = {"returned_model": Counter(), "returned_effort": Counter(),
                       "returned_temperature": Counter(), "returned_seed": Counter()}
    elapsed = {"direct": [], "depth": []}
    token_usage = {"direct": Counter(), "depth": Counter()}
    review = []
    for item, pair in zip(manifest["items"], pairs):
        entry = {"sample_id": pair["sample_id"], "task_type": pair["task_type"],
                 "question": item["problem"]["question"], "options": item["problem"]["options"],
                 "original_image_paths": item["problem"]["image_paths"],
                 "depth_images": item["depth_images"], "annotation": item["annotation"],
                 "pair": pair, "responses": {}}
        for arm in ("direct", "depth"):
            path = directory / "traces" / f"{item['index']:04d}-{arm}.result.json"
            result = json.loads(path.read_bytes()) if path.exists() else None
            entry["responses"][arm] = result
            response = result.get("response") if result else None
            if response:
                reported = response.get("reported", {})
                for target, source in (
                    ("returned_model", "model"), ("returned_effort", "reasoning_effort"),
                    ("returned_temperature", "temperature"), ("returned_seed", "seed")):
                    metadata_counts[target][str(reported.get(source))] += 1
                elapsed[arm].append(response["elapsed_seconds"])
                for key, value in response.get("usage", {}).items():
                    if type(value) in (int, float):
                        token_usage[arm][key] += value
        review.append(entry)
    lines += [
        "服务返回的参数单独记录如下；请求参数与服务报告值的差异保留在逐请求 JSON 中。",
        "",
        f"- 实际报告模型：{dict(metadata_counts['returned_model'])}。",
        f"- 实际报告 reasoning effort：{dict(metadata_counts['returned_effort'])}。",
        f"- 实际报告 temperature：{dict(metadata_counts['returned_temperature'])}；"
        f"seed：{dict(metadata_counts['returned_seed'])}。",
        "",
    ]
    if (set(metadata_counts["returned_temperature"]) - {"0", "0.0"}
            or set(metadata_counts["returned_seed"]) - {"137"}):
        lines += [
            "服务返回的采样元数据未确认请求中的 temperature=0 和 seed=137 "
            "均已应用。两臂请求设置相同；本报告保留服务返回值，并将结果解释为本次"
            "配对测试观察到的差异。", ""]
    if all(elapsed.values()):
        lines += [
            f"单请求耗时中位数：原图 {statistics.median(elapsed['direct']):.1f} 秒，"
            f"深度 {statistics.median(elapsed['depth']):.1f} 秒。",
            "",
        ]
    lines += [
        "## 逐题结果",
        "",
        "| 题目 | 原图答案 | 深度答案 | 标准答案 | 原图正确 | 深度正确 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for pair in pairs:
        lines.append(f"| {pair['sample_id']} | {pair['direct']['answer'] or '—'} | "
                     f"{pair['depth']['answer'] or '—'} | {pair['reference_answer']} | "
                     f"{'✓' if pair['direct']['correct'] else '✗'} | "
                     f"{'✓' if pair['depth']['correct'] else '✗'} |")
    lines += [
        "",
        "## 保存的证据",
        "",
        f"- `{run_path}/manifest.json`：源码、模型、参数、图像字节与深度输入身份。",
        f"- `{run_path}/traces/`：逐次 dispatch、实际响应、合法答案提取和服务状态。",
        f"- `{run_path}/frozen.json`：预测冻结及所有证据文件哈希。",
        f"- `{run_path}/score.json`：冻结后独立评分、配对统计和每题结果。",
        "- `review.json`：题目、选项、两臂响应和评分汇总，供轨迹复查。",
        "",
        f"run identity：`{manifest['run_identity']}`。",
        "",
        f"freeze identity：`{frozen['freeze_identity']}`。",
        "",
        f"score identity：`{score['score_identity']}`。",
        "",
    ]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (destination / "review.json").write_text(json.dumps({
        "schema": "mmsi-depth-review/1", "run_identity": manifest["run_identity"],
        "freeze_identity": frozen["freeze_identity"], "cases": review,
        "run_directory": str(directory), "paired_requests": paired_requests,
        "service_recheck_requests": health_requests,
        "total_real_inference_calls": paired_requests + health_requests,
        "token_usage": {arm: dict(value) for arm, value in token_usage.items()},
        "reference_access": "host_only_after_prediction_freeze",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "collection-verification.json").write_text(json.dumps({
        "status": "passed", "frozen_files_verified": len(frozen["files"]),
        "run_identity": manifest["run_identity"], "freeze_identity": frozen["freeze_identity"],
        "score_identity": score["score_identity"], "overall_counts_recomputed": True,
        "scored_answers_match_frozen_predictions": True,
        "gain_loss_and_mcnemar_recomputed": True,
        "run_directory": str(directory), "paired_requests": paired_requests,
        "original_paired_budget": 60, "service_recheck_requests": health_requests,
        "total_real_inference_calls": paired_requests + health_requests,
        "service_checks": checks,
        "prior_runs": [{"directory": str(path), "run_identity": previous["run_identity"],
                        "freeze_identity": previous_freeze["freeze_identity"],
                        "files_verified": len(previous_freeze["files"]),
                        "dispatched": previous_execution["dispatched"]}
                       for path, previous, previous_freeze, previous_execution in prior_runs],
        "verified_on": "Mac after collection",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(destination / "report.md"), "overall": overall},
                     ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    parser.add_argument("--run-directory")
    args = parser.parse_args()
    export(args.destination, args.run_directory)
