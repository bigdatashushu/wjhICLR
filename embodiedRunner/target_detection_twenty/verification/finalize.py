"""Add measured context costs and concrete failure evidence to the final report."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from target_detection_twenty.control import OUT, VERIFY, load, save, verify_identity
from target_detection_twenty.report import first_observation
from target_detection_r05.depth import decode_raw


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def main():
    verify_identity()
    result, context = load(OUT / "results.json"), load(OUT / "context-costs.json")
    assert result["status"] == context["status"] == "verified"
    assert result["completed"] == context["completed"] == 20
    assert result["model_requests"] == context["aggregate"]["current"]["requests"]
    assert result["total_usage"] == context["aggregate"]["current"]["total_usage"]
    assert context["aggregate"]["rgbd"]["requests"] == 75
    statuses = Counter(pair["current_status"] for pair in result["pairs"])
    failures = Counter()
    phase_requests = Counter()
    ungrounded = []
    protocol_details = []
    for pair in result["pairs"]:
        detail = context["pairs"][pair["identity"]]["current"]
        for phase, row in detail["by_phase"].items():
            phase_requests[phase] += row["requests"]
        if detail["error"]:
            failures[detail["error"]["code"]] += 1
            protocol_details.append({
                "identity": pair["identity"], "code": detail["error"]["code"],
                "last_response_root_keys": detail["last_response_root_keys"],
                "last_tool_name": detail["last_tool_name"],
                "contains_action_plan": detail["last_response_contains_action_plan"],
            })
        if detail["error"] and detail["error"]["code"] == "target_not_grounded_in_public_instruction":
            ungrounded.append({
                "identity": pair["identity"], "public_instruction": detail["public_instruction"],
                "unmatched_sources": [row for row in detail["first_target_source_bindings"]
                                      if not row["matches_public_instruction"]],
            })
    actual_pairs = [pair for pair in result["pairs"] if pair["legacy_full"] is not None]
    input_summary = {
        field: sum(pair["initial_identity"].get(field) is True for pair in result["pairs"])
        for field in ("initial_instruction_equal", "initial_actions_equal",
                      "initial_rgb_pixels_equal", "initial_depth_png_equal")}
    input_differences = []
    for pair in result["pairs"]:
        if pair["task"]["environment"] != "navigation":
            continue
        current = first_observation(Path(pair["current_directory"]) / "trajectory.jsonl")["public"]
        prior = first_observation(Path(pair["rgbd_result"]).with_name("trajectory.jsonl"))["public"]
        a, b = decode_raw(current["depth"]), decode_raw(prior["depth"])
        valid = np.isfinite(a) & np.isfinite(b)
        differences = np.abs(a[valid].astype(float) - b[valid].astype(float))
        input_differences.append({
            "identity": pair["identity"],
            "rgb_mean_absolute_difference": pair["initial_identity"]["initial_rgb_pixel_mean_absolute_difference"],
            "depth_png_equal": pair["initial_identity"]["initial_depth_png_equal"],
            "depth_finite_masks_equal": bool(np.array_equal(np.isfinite(a), np.isfinite(b))),
            "depth_mean_absolute_difference_m": float(differences.mean()),
            "depth_max_absolute_difference_m": float(differences.max()),
            "depth_changed_pixels": int(np.count_nonzero(differences)),
        })
    supplement = {
        "status_counts": dict(statuses), "error_code_counts": dict(failures),
        "model_requests_by_phase": dict(phase_requests), "source_binding_failures": ungrounded,
        "protocol_error_details": protocol_details,
        "historical_full_matched_selected": len(actual_pairs),
        "historical_full_matched_successes": sum(pair["legacy_full"]["success"] == 1 for pair in actual_pairs),
        "current_successes_on_full_matched_tasks": sum(pair["current_success"] for pair in actual_pairs),
        "costs": context["aggregate"],
        "initial_input_match_counts": input_summary,
        "navigation_initial_input_differences": input_differences,
        "regressed_tasks_with_planner_error": [
            pair["identity"] for pair in result["pairs"]
            if pair["rgbd_success"] == 1 and pair["current_success"] == 0
            and pair["current_status"] == "planner_error"],
    }
    result["diagnostics"] = supplement
    if (VERIFY / "service-stop.json").exists():
        result["diagnostics"]["operational_stop"] = load(VERIFY / "service-stop.json")
    save(OUT / "results.json", result)
    lines = [
        "", "## 结果解释", "",
        f'相对历史 RGB-D，由失败变成功 {len(result["changed_to_success"])} 题，'
        f'由成功变失败 {len(result["changed_to_failure"])} 题。'
        f'退步题中 {len(supplement["regressed_tasks_with_planner_error"])} 题结束于模型/工具协议错误。'
        "这些错误包括原文绑定、跳过检测阶段和 JSON 字段位置；检测图与裁剪的任务收益"
        "同时受到交互协议和固定请求预算影响。",
        "",
        f'初始题目与动作列表全部一致，RGB 像素完全一致 '
        f'{input_summary["initial_rgb_pixels_equal"]}/20，深度图完全一致 '
        f'{input_summary["initial_depth_png_equal"]}/20。导航五题存在 RGB 差异，'
        "下表用 0–255 像素值的平均绝对差和米制深度差量化；"
        "这些题不属于初始图像逐像素相同的对照。",
        "",
        "| 导航题 | RGB 平均绝对差 | 深度 PNG 相同 | 深度平均绝对差/m | 深度最大绝对差/m |",
        "|---|---:|---|---:|---:|",
        *[f'| {row["identity"]} | {row["rgb_mean_absolute_difference"]:.6f} | '
          f'{"是" if row["depth_png_equal"] else "否"} | '
          f'{row["depth_mean_absolute_difference_m"]:.6g} | '
          f'{row["depth_max_absolute_difference_m"]:.6g} |' for row in input_differences],
        "",
        "## 上下文与实际成本", "",
        "| 项目 | 历史 RGB-D | 本轮 R05 |",
        "|---|---:|---:|",
    ]
    a, b = context["aggregate"]["rgbd"], context["aggregate"]["current"]
    for label, field in (("模型请求", "requests"), ("有 token 记录的响应", "responses_with_usage"),
                         ("单请求最大输入 token", "maximum_prompt_tokens")):
        lines.append(f'| {label} | {a[field]:,} | {b[field]:,} |')
    for label, field in (("累计输入 token", "prompt_tokens"), ("累计输出 token", "completion_tokens"),
                         ("累计总 token", "total_tokens")):
        lines.append(f'| {label} | {a["total_usage"][field]:,} | {b["total_usage"][field]:,} |')
    lines.extend([
        "",
        f'请求数为历史的 {b["requests"]/a["requests"]:.2f} 倍，'
        f'累计总 token 为 {b["total_usage"]["total_tokens"]/a["total_usage"]["total_tokens"]:.2f} 倍。'
        "这些是服务返回的真实用量，包含多轮重发的文本与图片上下文。",
        "",
        "| 本轮请求阶段 | 请求数 |",
        "|---|---:|",
    ])
    names = {"extract_and_detect": "首次提取与检测", "detect": "新观测检测",
             "inspect": "裁剪或翻页", "review": "视觉复核",
             "action_plan": "官方动作生成", "official_plan": "官方规划"}
    for phase, count in phase_requests.items():
        lines.append(f'| {names[phase]} | {count} |')
    lines.extend([
        "",
        "同一观测的后续请求包含官方提示、当前 RGB-D、感知协议和已完成的工具对话。"
        "Manipulation 另带未标注原始 RGB，后续轮次再带候选图或裁剪。"
        "最终动作仍由 Qwen 根据这些上下文生成。",
        "",
        "## 实际终止原因", "",
        "| 结束状态 | 题数 |", "|---|---:|",
    ])
    for name, count in statuses.items():
        lines.append(f'| {name} | {count} |')
    if failures:
        lines.extend(["", "| 错误代码 | 题数 |", "|---|---:|"])
        for code, count in failures.items():
            lines.append(f'| {code} | {count} |')
        lines.extend(["", "| 题目 | 报错 | 最后响应根字段 | 包含动作计划 |",
                      "|---|---|---|---|"])
        for row in protocol_details:
            keys = ", ".join(row["last_response_root_keys"])
            lines.append(f'| {row["identity"]} | {row["code"]} | {cell(keys)} | '
                         f'{"是" if row["contains_action_plan"] else "否"} |')
        lines.extend([
            "",
            "`tool_and_action_branches_must_be_separate` 也会在工具响应含未允许的根字段时触发；"
            "错误代码本身不证明模型同时输出了动作。上表按真实响应字段区分。",
        ])
    if ungrounded:
        lines.extend([
            "", "原文绑定失败发生在检测和动作执行之前。以下保留触发失败的真实字段：", "",
            "| 题目 | 公开原题 | 未匹配的 source_phrase |",
            "|---|---|---|",
        ])
        for row in ungrounded:
            sources = "; ".join(item["source_phrase"] for item in row["unmatched_sources"])
            lines.append(f'| {row["identity"]} | {cell(row["public_instruction"])} | {cell(sources)} |')
    lines.extend([
        "",
        f'旧 Full 可匹配的 {len(actual_pairs)} 题：旧 Full 成功 '
        f'{supplement["historical_full_matched_successes"]}/{len(actual_pairs)}，'
        f'本轮成功 {supplement["current_successes_on_full_matched_tasks"]}/{len(actual_pairs)}。'
        "该补充对照的动作与子目标协议不同。",
        "",
        "- [逐请求阶段、输入规模与真实用量](context-costs.json)",
        "- [启动目录故障与零请求证明](verification/setup-failure.json)",
        "",
    ])
    if result["operational_failures"]:
        lines.extend([
            "Manipulation base 0 在第 14 个动作发送后连接关闭：13 个动作已收到 ACK，"
            "最后一个动作结果未知。承载 worker 的 SSH 进程返回码为 255，"
            "未保存 stderr，因此退出根因未定。检测服务与自有进程状态随后重新核验，"
            "只继续了未启动题；该题未重跑，动作未重发。",
            "",
            "- [运行故障、回执边界与续跑记录](verification/service-stop.json)",
            "",
        ])
    path = OUT / "report.md"
    text = path.read_text(encoding="utf-8")
    marker = "\n## 结果解释"
    if marker in text:
        text = text.split(marker, 1)[0]
    path.write_text(text.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")
    assert "\ufffd" not in path.read_text(encoding="utf-8")
    print(json.dumps(supplement, ensure_ascii=False))


if __name__ == "__main__":
    main()
