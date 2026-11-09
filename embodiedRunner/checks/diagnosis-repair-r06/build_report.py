"""Build the final complete report without overwriting any earlier run or report."""
from collections import Counter
import difflib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "src")]
from protocol import sha256
from action_identity import source_identity


def load(path):
    return json.loads(path.read_bytes())


def link(path, text=None):
    return f"[{text or path.name}](file://{path})"


def main():
    historical = load(ROOT / "runs/diagnosis-repair-r05-evidence.json")
    pilot = load(HERE / "pilot/verification.json")
    full = load(HERE / "full/verification.json")
    analysis = load(HERE / "full/episode-analysis.json")
    assert analysis["status"] == "complete"
    regression_suites = [s.attrib for s in ET.parse(HERE / "regression-confirmation.xml").getroot().iter("testsuite")]
    assert regression_suites and all(int(s.get("failures", 0)) == int(s.get("errors", 0)) == 0
                                     for s in regression_suites)
    by_ep = {r["episode_id"]: r for r in full["episodes"]}
    assert len(analysis["episodes"]) == 15 and set(analysis["episodes"]) == set(by_ep)
    baseline = load(ROOT / "checks/diagnosis-repair-r05/embodied_diagnosis_baseline_20261106.json")
    preservation = {
        "protected_files": len(baseline["protected"]),
        "other_project_files": len(baseline["other_projects"]),
        "protected_changed_or_missing": [p for p, h in baseline["protected"].items()
                                         if not Path(p).is_file() or sha256(Path(p).read_bytes()) != h],
        "other_project_changed_or_missing": [
            p for p, h in baseline["other_projects"].items()
            if not (ROOT.parent / p).is_file() or sha256((ROOT.parent / p).read_bytes()) != h]}
    assert not preservation["protected_changed_or_missing"] and not preservation["other_project_changed_or_missing"]
    followup = load(HERE / "followup-preservation-baseline.json")["files"]
    preservation["completed_followup_files"] = len(followup)
    preservation["completed_followup_changed_or_missing"] = [
        p for p, h in followup.items() if not (ROOT / p).is_file() or sha256((ROOT / p).read_bytes()) != h]
    assert not preservation["completed_followup_changed_or_missing"]
    r05_deployed = load(ROOT / "checks/diagnosis-repair-r05/deployment.json")["source"]
    r06_deployed = load(HERE / "full/deployment.json")["source"]
    assert r06_deployed == source_identity() == load(HERE / "pilot/deployment.json")["source"]
    changes = {}
    for p in (HERE / "source-before").glob("*.py"):
        target = ROOT / p.name
        changes[p.name] = {"before_sha256": sha256(p.read_bytes()), "after_sha256": sha256(target.read_bytes()),
                           "diff": "".join(difflib.unified_diff(
                               p.read_text().splitlines(True), target.read_text().splitlines(True),
                               fromfile=str(p), tofile=str(target)))}
    matrix = []
    for row in historical["matrix"]:
        ep = row["episode_id"]
        newer = by_ep[ep]
        matrix.append({"episode_id": ep, **{v: row[v]["success"] for v in ("Direct", "Full", "r02", "r04")},
                       "r05_pilot": newer["initial_comparisons"].get("r05_pilot", {}).get("success"),
                       "r06_full": newer["success"], "r06_steps": newer["steps"], "r06_qwen": newer["qwen"],
                       "r06_initial_vs_r04": newer["initial_comparisons"]["r04"],
                       "r06_analysis": analysis["episodes"][ep]})
    evidence = {
        "schema": "embodied-complete-diagnosis-and-repair-r06/1", "status": "validated",
        "scope": "15 development tasks, historical Direct/Full/r02/r04; independent r05/r06 pilots and r06 full run.",
        "matrix": matrix,
        "historical": {"evidence_path": str(ROOT / "runs/diagnosis-repair-r05-evidence.json"),
                       "evidence_sha256": sha256((ROOT / "runs/diagnosis-repair-r05-evidence.json").read_bytes()),
                       "inventory_path": str(ROOT / "checks/diagnosis-repair-r05/embodied_diagnosis_inventory_20261106.json")},
        "r05_full_followup": {"path": str(HERE / "r05-followup-read.json"),
                             "sha256": sha256((HERE / "r05-followup-read.json").read_bytes())},
        "r06_pilot": pilot, "r06_full": full, "r06_full_analysis": analysis,
        "r06_pilot_analysis": load(HERE / "pilot/analysis.json"),
        "followup_consumption": load(HERE / "consumption.json"),
        "r06_source_changes": changes, "r05_source_identity": r05_deployed,
        "r06_source_identity": r06_deployed, "r06_offline": load(HERE / "offline-final.json"),
        "r06_regression": regression_suites,
        "r06_prior_regression_timeout": load(HERE / "regression-timeout-diagnosis.json"),
        "request_count_reconciliation": load(HERE / "request-count-reconciliation.json"),
        "preservation": preservation, "git_commit_or_push": False,
        "scientific_variables": {"r05": "Joint targeted host/control/prompt repair.",
                                 "r06_vs_r05": "Lifecycle guidance and generic output example only."},
        "limits": ["Development task selection was used for diagnosis and follow-up.",
                   "Navigation initial pixel differences are measured; hidden reset state and exact provider weights are not established.",
                   "Component effects are not individually identified by the joint r05 change.",
                   "Predicted depth structure checks do not establish metric or object-surface accuracy."]}
    out = ROOT / "runs/diagnosis-repair-r06-final-evidence.json"
    with out.open("x") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
        f.write("\n")
    if (HERE / "preservation-final.json").exists():
        assert load(HERE / "preservation-final.json") == preservation
    else:
        with (HERE / "preservation-final.json").open("x") as f:
            json.dump(preservation, f, indent=2)
            f.write("\n")
    totals = {v: sum(row[v] for row in matrix) for v in ("Direct", "Full", "r02", "r04", "r06_full")}
    lines = [
        "# embodiedRunner：完整下降诊断与修复验证",
        "",
        f"独立 r06 完整开发集复跑为 **{full['successes']}/15**，"
        f"Navigation/ALFRED/Habitat 分别为 "
        f"{full['environment_successes']['eb_navigation']}/5、"
        f"{full['environment_successes']['eb_alfred']}/5、"
        f"{full['environment_successes']['eb_habitat']}/5。"
        "历史 Direct=12/15、旧 Full=8/15、r02=1/15、r04=1/15。"
        "结果属于用于诊断的开发题；导航逐题列出实际初始像素差异。",
        "",
        "主要下降机制是新增子目标转换协议阻断合法待执行动作、重复感知与格式修复占用有限调用，"
        "再叠加模型的导航方位和家务顺序错误。上下文完整交付后，示例提取和子目标转换语义仍需修正。"
        "r05 修复匹配 Find 完成判据、示例提取、格式反馈、工具可用性和感知后的程序出口；"
        "r06 根据 r05 新失败，改用单动作 action_ack 示例并说明未完成时 keep。"
        "保持持物、放置目标、循环、几何、身份及预算检查。",
        "",
        "深度不是这批下降的可证原因：旧 Full 运行了 218 次 MoGe2，但没有程序读取深度；"
        "r02/r04 没有运行 MoGe2。存在边界混合和预测内参偏差的数值风险，"
        "尚无同帧官方深度和物体表面真值来证明物理精度。官方 RGB 与 MoGe2 预测来源已明确区分。",
        "",
        "全部官方规则、历史 15 题逐版本事件链和工具/深度证据完整保留在本文后半及 "
        + link(out, "证据 JSON") + "。",
        "",
        "## 全部 15 题与复跑证据", "",
        "| 任务 | Direct | 旧 Full | r02 | r04 | r05 七题 | r06 全部 | r06 动作/请求 | 初始像素与 r04 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in matrix:
        pairs = row["r06_initial_vs_r04"]
        lines.append("| " + row["episode_id"] + " | "
                     + " | ".join(str(int(row[v])) for v in ("Direct", "Full", "r02", "r04"))
                     + " | " + ("—" if row["r05_pilot"] is None else str(int(row["r05_pilot"])))
                     + f" | {int(row['r06_full'])} | {row['r06_steps']}/{row['r06_qwen']} | "
                     + ("一致" if pairs["initial_pixel_equal"] else
                        f"不同，均差{pairs['mean_channel_abs_difference']:.6f}/最大{pairs['max_channel_abs_difference']}")
                     + " |")
    lines += [
        "", f"r05 七题 pilot：4/7，同七题 r04：1/7。r06 两题生命周期 pilot：{pilot['successes']}/2；"
        "r05 同两题为 0/2。pilot 与全 15 题分别计数。",
        "", f"r06 全部使用 {full['qwen_requests']} 次 Qwen 请求和 {full['actual_actions']} 个实际动作。"
        "运行摘要161次为所有模型/感知请求总和：Qwen155、DINO2、SAM2、Intern2、MoGe0。"
        "15 条轨迹都正常结束；运行与轨迹审计通过，参考答案在线隐藏，操作性服务失败不自动重试。"
        "模型参数、Skill 正文、调用预算、渲染/感知配置及官方任务来源保持原身份。",
        "", f"完整请求文本最大 {full['request_text_max']} 字符，均值 {full['request_text_mean']:.2f}，"
        f"{full['request_text_over_16000']} 次超过 16000 字符软目标。关键字段始终完整保留，图片 token 另计。",
        "", "## r06 逐题完整轨迹复核", ""]
    for ep in sorted(by_ep):
        row, notes = by_ep[ep], analysis["episodes"][ep]
        lines += ["### " + ep, "",
                  f"{'成功' if row['success'] else '失败'}；{row['steps']} 动作、{row['qwen']} 请求；"
                  f"官方进度 {row['progress']}；停止原因 {row['terminal_reason']}。",
                  link(Path(row["trajectory_path"]), "原始完整轨迹") + "。",
                  link(HERE / "full" / (ep.replace(":", "-") + "-review.json"), "完整公开决策链")
                  + "；" + link(HERE / "full" / (ep.replace(":", "-") + "-rgb.png"), "全部公开 RGB") + "。",
                  "", notes["finding"], ""]
        for ref in notes["evidence"]:
            p = Path(row["trajectory_path"]).parent / "events.jsonl"
            lines.append(f"- [e{ref['event']}](file://{p}#L{ref['event']+1})：{ref['detail']}")
        lines += ["", "未解决/解释范围：" + notes["limit"], ""]
    lines += ["## 修复、离线验证与选择依据", "",
              "r05 的全部方案比较、真实提案重放和代码 diff 见下方历史报告。"
              "r06 仅改变生命周期提示及通用输出示例；启动白名单和两题配置用于隔离运行。",
              "六个 r05 真实失败提案仍拒绝错误的 advance，保留原目标改 keep 后，"
              "合法的 Find Sink/Toilet 通过；单动作 ACK 反例也通过。"
              "206 个旧请求快照重建保留全部 Skill、动作、历史及公开反馈；"
              "其中一个关键内容完整的请求为16033字符，超过软目标33字符。"
              "最终确认54项相关回归通过。此前一次回归为53通过、1失败，"
              "原始隔离执行回执显示10秒墙钟超时（10094ms/10229ms），没有修改源码或预算，"
              "保留失败XML与超时诊断，在线请求没有重试。"
              "r05先前272项回归、额外格式检查及Linux隔离检查通过。"
              "这些离线检查证明协议与交付，真实任务效果由上述独立复跑衡量。",
              "", link(HERE / "offline-final.json") + "；" + link(HERE / "pilot/verification.json")
              + "；" + link(HERE / "full/verification.json") + "。",
              "", "完整上下文仍有内容上的缺口：ALFRED 当前两条官方示例都为灯光任务，"
              "均展示 Find→pickup→Find lamp→turn on，没有清洗或嵌套放置前提。"
              "Ladle 的开水链与该结构相似，但目前不能认定示例复制是唯一原因。"
              "冻结 Skill 还允许 no productive action remains 时停止，任务卡则写 empty only when done；"
              "这项措辞差异保留为独立验证线索。本轮始终保持参测 Skill 正文不变。",
              "", "按实际调用而非配置名称核验后续工具消费："
              + link(HERE / "consumption.json", "后续工具与 RGB 审计") + "。"
              "每次求解输入均核验为对应当前500×500 RGB；后端请求、返回、准备输入和程序读取分别计数。",
              "r05七题、r06两题及r06全15题的实际程序工具、reader、ctx.frames读取均为0。"
              "r06全15题只有Habitat000/003各一次DINO/SAM/Intern准备，均无深度；"
              "准备结果进入模型卡片，但后续没有完成源物体定位。"
              "因此不能把本轮恢复归功于几何工具，也不能把这些剩余失败归因于预测深度。",
              "", "## 剩余问题与下一实验", "", analysis["next_experiment"],
              "", f"保存核验：原证据 {preservation['protected_files']} 文件和其他项目 "
              f"{preservation['other_project_files']} 文件均未改变。"
              "r05/r06 使用独立服务器目录、源码身份和输出；原始失败轨迹完整保留。未提交或推送 Git。",
              "", "## 历史完整 15 题诊断、官方协议及工具/深度审查", "",
              "以下是已完成 r05 的原报告全文；其中 r05 pilot 的时间和分母保持原记载，"
              "最新完整运行结果以上文 r06 为准。", "",
              (ROOT / "runs/diagnosis-repair-r05.md").read_text()]
    report = ROOT / "runs/diagnosis-repair-r06-final.md"
    with report.open("x") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"status": "validated", "totals": totals, "report": str(report),
                      "evidence": str(out), "preservation": preservation}))


if __name__ == "__main__":
    main()
