"""Render an audited BLINK paired run as a concise Chinese Markdown report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in record.items() if k != "content_sha256"}
    actual = hashlib.sha256(json.dumps(
        body, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    if record["content_sha256"] != actual:
        raise ValueError("report_input_content_changed")
    return record


def report(root):
    completion = read(root / "completion.json")
    audit = read(root / "final_audit.json")
    manifest = read(root / "manifest.json")
    comparison = read(root / "comparison.json")
    if (completion["status"] != "complete" or audit["status"] != "passed"
            or audit["completion_sha256"] != completion["content_sha256"]
            or audit["manifest_sha256"] != manifest["content_sha256"]):
        raise ValueError("report_requires_bound_final_audit")
    scores = completion["scores"]
    costs = completion["costs"]
    pairs = completion["paired"]
    names = {"fixed_skill": "固定 trajectory-1 Skill＋工具", "direct": "纯 Qwen"}
    lines = [
        "# BLINK 93 题完整对照实验",
        "",
        f"完成时间：{completion['at']}。两组各完成 93 题，共 186 次题目求解；最终独立验收通过。",
        "",
        "| 方法 | 正确题数 | 准确率 | 合法答案 | 模型请求 | 成功工具调用 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("fixed_skill", "direct"):
        score = scores[arm]
        cost = costs[arm]
        lines.append(f"| {names[arm]} | {score['correct']}/93 | "
                     f"{score['accuracy']:.2%} | {score['legal']}/93 | "
                     f"{cost['requests']} | {cost['succeeded_tool_calls']} |")
    difference = completion["accuracy_difference_pp"]
    net = scores["fixed_skill"]["correct"] - scores["direct"]["correct"]
    lines.extend([
        "",
        f"Skill 组相对纯 Qwen 的准确率差为 **{difference:+.2f} 个百分点**，正确题数差为 **{net:+d} 题**。",
        "",
        "## 逐题配对结果",
        "",
        "| 结果 | 题数 |",
        "| --- | ---: |",
        f"| 两组都正确 | {pairs.get('both_correct', 0)} |",
        f"| 仅 Skill 组正确 | {pairs.get('skill_only', 0)} |",
        f"| 仅纯 Qwen 正确 | {pairs.get('direct_only', 0)} |",
        f"| 两组都错误 | {pairs.get('both_wrong', 0)} |",
        "",
        "## 范围与配置",
        "",
        "本次使用原内部划分的 BLINK Multi-view Reasoning **93 道 test 题**，每题两张原图。"
        "40 道 train 题保持原分组。该范围对应本项目选用的 BLINK 子任务及内部划分。",
        "",
        "两组沿用 24 题对照的模型及参数：`qwen3.8-flash`、temperature 0、seed 137、"
        "thinking 开启、最大输出 8192 token、单请求超时 600 秒、最多 4 并发。"
        "同题的双组顺序使用 seed 137 固定随机化。",
        "",
        "Skill 组使用同一份冻结的 trajectory-1 Skill，每题最多 4 次请求；"
        "纯 Qwen 每题 1 次请求，输入为题目和原始两张图片。"
        "本轮采用独立预算，上限 465 次请求，全程没有 Skill 训练或演化。",
        "",
        f"Skill artifact SHA-256：`{manifest['preflight']['seed']['content_sha256']}`。",
        "",
        "## 请求用量与执行质量",
        "",
        "| 方法 | 输入 token | 输出 token | 总 token | 程序错误题数 | 运行错误题数 | 拒绝授权次数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for arm in ("fixed_skill", "direct"):
        usage = costs[arm]["usage"]
        score = scores[arm]
        lines.append(
            f"| {names[arm]} | {usage.get('prompt_tokens', 0):,} | "
            f"{usage.get('completion_tokens', 0):,} | {usage.get('total_tokens', 0):,} | "
            f"{score['program_errors']} | {score['runtime_errors']} | "
            f"{score['denied_authorizations']} |")
    usage = completion["request_usage"]
    lines.extend([
        "",
        f"实际请求 {usage['reserved_requests']}/465，服务错误 {usage['error_requests']}，"
        f"未决请求 {usage['unknown_requests']}，结束时活跃请求 {usage['active_requests']}。"
        "token 为服务返回的计数。",
        "",
        "## 最终验收",
        "",
        "独立检查确认：两组题目和原图身份一致；实际请求使用原图字节和相同图序；"
        "实际模型与参数符合冻结配置；Skill 交付哈希一致；请求账本和逐题记录对应；"
        "最终评分在 186 个预测完成冻结且求解进程退出后于宿主侧读取标签。"
        "准确率和逐题配对统计已独立重算，线上求解不接收参考答案，没有自动请求重试。",
        "",
        f"服务返回的 backend fingerprint 可观察："
        f"{'是' if audit['backend_fingerprint_observed'] else '否'}。",
        "",
        "## 结果文件",
        "",
        "- [逐题对照 comparison.json](comparison.json)",
        "- [完成回执 completion.json](completion.json)",
        "- [独立验收 final_audit.json](final_audit.json)",
        "- [冻结配置 manifest.json](manifest.json)",
        "- [预测冻结记录 prediction_freeze.json](prediction_freeze.json)",
        "- `results/`：两组全部逐题结果。",
        "- `requests/`：实际请求参数、输入哈希、响应和 token 用量。",
        "",
        f"服务器运行目录：`{manifest['jobs'][0]['worker']['run_root']}`。",
        "",
    ])
    if (root / "live_scores").is_dir():
        lines.extend([
            "用户请求的实时分数由独立宿主进程统计：先冻结当时已完成预测的快照，"
            "再读取标签计分；这些分数不传给求解器，也不改变冻结方法或后续调度。"
            "对应快照保存在 `live_scores/`。",
            "",
        ])
    target = root / "BLINK93_对照实验报告.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(target), "scores": scores,
                      "accuracy_difference_pp": difference}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report(args.root.resolve())
