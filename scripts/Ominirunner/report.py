"""Readable report of audited real paired results; contains no ground truth."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    __package__ = "Ominirunner"

from .common import ADDED_IDS, PRIOR_IDS, read


def percent(value):
    return f"{value*100:.1f}%"


def report(root):
    score, audit, frozen = read(root/"scores.json"), read(root/"audit.json"), read(root/"frozen.json")
    if audit["status"] != "passed" or audit["run_identity"] != score["run_identity"]:
        raise ValueError("report_requires_audited_score")
    if frozen.get("measurement_reviewer") and frozen.get("question_replay"):
        from .host_report import report as host_report
        return host_report(root, score, audit, frozen)
    config = frozen["config"]
    n = score["pairs"]
    groups = ("float", "int", "yes_no", "other_text")
    labels = ("Float MRA", "Int", "Yes/No", "其他文本")
    metric_headers = [
        f'{label}（{score["table"]["direct"]["metrics"].get(group, {}).get("n", 0)}）'
        for group, label in zip(groups, labels)]
    caps = "／".join("不设硬上限" if n is None else str(n) for n in config["text_caps"])
    selection = frozen.get("selection")
    selection_description = (
        f"排除旧 {len(selection['excluded_sample_ids'])} 题及其图片，在 "
        f"{selection['eligible_questions']} 条公开候选中，按 "
        f"`SHA256('{selection['seed']}:'+sample_id)` 升序抽取 {n} 题。"
        f"题型分布为 {selection['task_types']}；抽样未读取答案或得分，两组均重新运行。"
        if selection else
        "保留此前 12 题，另按公开类型和题号加入 8 题；新增题用 "
        "`SHA256('omni20:137:'+sample_id)` 排序，在未使用的题目中选取 4 float、2 int、2 str。"
        "本轮复用此前同一批 20 题，检查原图标点、局部表面复核与结构化复核记录下的作答表现。")
    lines = [
        f"# Omni3D-Bench：原图表面复核程序组与千问直答的 {n} 题对照",
        "",
        f"本次完成 {n} 题、{2*n} 个 episode，按固定 VADAR 官方规则评分。"
        + selection_description,
        "",
        "| 方法 | "+" | ".join(metric_headers)+" | 逐题平均分 | 合法答案 | 请求 | 实际 token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, label in (("direct", "Qwen 原始 RGB 直接作答"), ("skill", "RGB＋深度＋工具＋静态 Skill")):
        row = score["table"][arm]
        metrics = row["metrics"]
        values = [percent(metrics[g]["value"]) if g in metrics else "—" for g in groups]
        lines.append("| "+label+" | "+" | ".join(values)+(
            f' | {percent(row["sample_weighted_score"])} | {row["legal"]}/{n}'
            f' | {row["requests"]} | {row["usage"]["total_tokens"]:,} |'))
    paired = {}
    for row in score["scored"]:
        paired.setdefault(row["sample_id"], {})[row["arm"]] = row
    wins = sum(r["skill"]["score"] > r["direct"]["score"] for r in paired.values())
    losses = sum(r["skill"]["score"] < r["direct"]["score"] for r in paired.values())
    ties = len(paired)-wins-losses
    lines += [
        "",
        f"逐题得分比较：静态方案胜 {wins} 题、负 {losses} 题、平 {ties} 题。"
        "Float 每题分数为十个相对误差阈值判定的平均；其他题为精确匹配。",
        "",
        "两组均为 `qwen3.8-flash`、`enable_thinking=true`、temperature 0、seed 137、"
        "每请求 `max_tokens=8192`。原图 JPEG 与完整官方题文逐题相同。"
        "直接组仅收到原图与原生答案 JSON 格式要求；静态组增加现有 MoGe2 深度、"
        f'逐题范围候选与局部数值工具、一份冻结短 Skill，最多 {config["max_program_rounds"]} '
        "个程序轮及一个 JSON 终结轮。工具调用与数值测量不设次数上限。",
        "",
        f"每次请求重新组装上下文。各轮文本限制为 {caps}，"
        f'超过 {config.get("text_record_threshold", 10000)} 字符的请求记录长度与超出量。'
        "程序组首轮两图，有测量后增加 RGB 标点副本与待复核点的原像素局部图；"
        "程序轮保留全部测量事实，终结轮最多八条测量；局部图最多两页。"
        "新测量在下一次请求中交付，由同一个千问模型记录 ID 和使用决定。"
        "完整程序与 thinking 留在宿主轨迹，不累积回模型上下文。",
        "",
        "## 运行与证据",
        "",
        f'- 审计时间：{datetime.fromtimestamp(audit["at"], timezone(timedelta(hours=8))).isoformat()}。',
        f'- 同一冻结身份：`{audit["run_identity"]}`，在线依赖源码 {audit["source_pins"]} 个文件。',
        f'- 本轮新增收费 {audit["charged_requests"]} 次；复用 {audit.get("reused_direct", 0)} '
        '份原始直答。表中请求与 token 包含复用结果的历史用量，新增费用单独核验。',
        f'- episode 调度并发 {frozen["max_episode_workers"]}，'
        f'本轮实际请求峰值 {audit.get("peak_request_concurrency", 2)}。',
        "- 参考答案只由完成后的宿主评分读取，报告与在线请求没有 GT。",
    ]
    for arm in ("direct", "skill"):
        entries = [r for r in audit["episodes"] if r["arm"] == arm]
        row = score["table"][arm]
        lines.append(
            f'- {arm}：状态 {row["statuses"]}；工具调用 {row["tool_calls"]}，'
            f'已准入数值测量 {sum(e["measurements"] for e in entries)}、'
            f'有效测量 {sum(e["valid_measurements"] for e in entries)}；'
            f'最大实际文本 {max(e["max_text_chars"] for e in entries)} 字符；'
            f'超一万字符请求 {sum(len(e.get("over_threshold_rounds", [])) for e in entries)} 次；'
            f'含 thinking 正文的响应 {sum(e["thinking_content_requests"] for e in entries)} 次；'
            f'平均 episode {row["mean_episode_seconds"]:.1f} 秒。')
        lines.append(
            f'  实际输入 token {row["usage"]["prompt_tokens"]:,}，'
            f'实际输出（包含服务计入的 thinking）{row["usage"]["completion_tokens"]:,}。')
        if arm == "skill":
            round_statuses = Counter()
            for entry in entries:
                round_statuses.update(entry["round_statuses"])
            measured_answers = sum(e["status"] == "answered" and e["valid_measurements"] > 0
                                   for e in entries)
            lines.append(f'- 静态组各轮状态：{dict(round_statuses)}；'
                         f'成功答案中具有有效数值测量的题目 {measured_answers}。')
            decisions = Counter()
            for entry in entries:
                decisions.update(entry["surface_review_decisions"])
            lines.append(f'- 同模型表面复核决定：{dict(decisions)}；'
                         f'结束时待复核测量 {sum(e["pending_surface_reviews"] for e in entries)}。'
                         "这些决定是模型判断，宿主验证记录与图像映射，未替模型认证对象绑定。")
            lines.append(f'- 声明的答案依据：{dict(Counter(e["answer_basis"] for e in entries))}；'
                         f'通过宿主算术核验的答案 {sum(e["verified_arithmetic"] for e in entries)}。'
                         "具有成功测量不等于答案使用了测量；完整范围和尺度锚点仍是模型声明。")
            changes = Counter()
            for entry in entries:
                changes.update(entry.get("evidence_activity") or {})
            if changes:
                lines.append(
                    f'- 可观察进展：新增观察 {changes["new_observations"]}，'
                    f'不同内容的观察 {changes["novel_observations"]}，'
                    f'复核状态变化 {changes["changed_reviews"]}，'
                    f'无进展轮次 {changes["no_progress_rounds"]}；'
                    f'成功返回的替换测量 {changes["replacement_measurements"]}，'
                    f'其中内容新增且终态 usable 的替换 {changes["usable_novel_replacements"]}。'
                    "usable 仍是同模型的 RGB 判断，不等于独立视觉认证。")
    completion = []
    for item in read(root/"inputs.json"):
        for arm in ("direct", "skill"):
            result = read(root/"episodes"/(item["sample_id"]+"-"+arm)/"result.json")
            completion += [r["usage"]["completion_tokens"] for r in result["rounds"] if "usage" in r]
    exceeded = sum(n > 8192 for n in completion)
    if exceeded:
        lines.append(f"- {exceeded} 次响应的服务 completion_tokens 超过 8192；费用采用实际 usage，"
                     "不能把 max_tokens 当作 thinking 的硬上限。")
    budget = audit["budget_after"]
    lines.append(f'- 合并历史独立账本后的消费为 {budget["used"]:,}/{budget["global_cap"]:,}；'
                 f'本次追加 {audit["charged_requests"]} 次，'
                 f'按上下文、category、usage、status 的多重集合核对后去重 '
                 f'{budget["mirrored_request_copies_excluded"]:,} 份局部副本。原始账本均保留。')
    subsets = [(label, [paired[i] for i in ids if i in paired])
               for label, ids in (("原 12 题", PRIOR_IDS), ("新增 8 题", ADDED_IDS))]
    subsets = [(label, rows) for label, rows in subsets if rows]
    if subsets:
        lines += ["", "## 原 12 题与新增 8 题", "",
                  "| 子集 | 题数 | 直接逐题平均分 | 静态逐题平均分 | 胜/负/平 |",
                  "|---|---:|---:|---:|---|"]
    for label, selected in subsets:
        d = sum(r["direct"]["score"] for r in selected)/len(selected)
        s = sum(r["skill"]["score"] for r in selected)/len(selected)
        w = sum(r["skill"]["score"] > r["direct"]["score"] for r in selected)
        l = sum(r["skill"]["score"] < r["direct"]["score"] for r in selected)
        lines.append(f"| {label} | {len(selected)} | {percent(d)} | {percent(s)} | "
                     f"{w}/{l}/{len(selected)-w-l} |")
    failures = audit.get("program_failures", [])
    if failures:
        lines += ["", "## 程序接口与失败", ""]
        lines.append("逐轮记录格式修正和执行错误；实际最终失败按零分保留。")
        for row in failures:
            reason = ("；".join(row["guard_violations"]) if row["guard_violations"] else
                      row.get("feedback_code") or
                      f'{row["error_code"]} / {row["error_type"]}')
            lines.append(f'- 题 {row["sample_id"]}，第 {row["round"]+1} 次请求：'
                         f'{row["status"]}，{reason}。')
    lines += [
        "",
        "## 逐题结果",
        "",
        "| 题号 | 分组 | 直接预测 | 直接分数 | 静态预测 | 静态分数 | 静态请求/工具 |",
        "|---|---|---|---:|---|---:|---|",
    ]
    order = [item["sample_id"] for item in read(root/"inputs.json")]
    for sample_id in order:
        d, s = paired[sample_id]["direct"], paired[sample_id]["skill"]
        def clean(text):
            return str(text).replace("|", "\\|").replace("\n", " ")
        lines.append(f'| {sample_id} | {d["group"]} | {clean(d["prediction"])} | {d["score"]:.2f} | '
                     f'{clean(s["prediction"])} | {s["score"]:.2f} | {s["requests"]}/{s["tool_calls"]} |')
    lines += [
        "",
        (f"本次是重新抽取的 {n} 题配对实验。" if selection else
         "本次是已检查过的 20 题诊断复测。") + "程序组同时增加几何输入、复核提示、"
        "图像和多轮计算，组间差值衡量这一整套配置，不能单独归因于工具或表面复核。"
        "浮点题使用 MRA，因此逐题平均分不等于二元准确率。",
        "",
        "评分来源：VADAR commit `56018ebc2ecf7a430fc94e64e80e2768b0931739`。"
        "int 对预测字符串执行 int；str 只执行 lower，不 trim；float 使用严格 `<` 的 "
        ".50、.45、.40、.35、.30、.25、.20、.15、.10、.05 十个阈值。全部失败保留分母。",
        "",
        "机器证据：[scores.json](scores.json)、[audit.json](audit.json)、"
        "[frozen.json](frozen.json)、[preflight.json](preflight.json)。"
        "完整输出、thinking、程序、工具、复核和提示文本汇总见 "
        "[trajectories.json](trajectories.json)，原始请求及图像见 episodes 目录。",
        "",
    ]
    (root/"report.md").write_text("\n".join(lines), encoding="utf-8")
    print("report.md created", flush=True)


if __name__ == "__main__":
    report(Path(sys.argv[1]))
