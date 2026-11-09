"""Build the trace-linked diagnosis without changing any historical artifact."""
from collections import Counter
from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value):
    if isinstance(value, list):
        return [compact(x) for x in value]
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()
                if k not in {"trace_refs", "code_refs", "semantic_index", "contact_sheets"}}
    return value


def link(path, text=None):
    return f"[{text or path.name}](file://{path})"


def anchored(text, record):
    # Every zero-based event eN has physical line N+1 in the retained JSONL.
    import re
    return re.sub(r"(?<![A-Za-z0-9_])e(\d+)(?![A-Za-z0-9_])", lambda m: (
        f"[e{m[1]}](file://{record['events_path']}#L{int(m[1])+1})"), text)


def suite(name):
    return [s.attrib for s in ET.parse(HERE / name).getroot().iter("testsuite")]


def main():
    inventory = load("embodied_diagnosis_inventory_20261106.json")
    navigation = load("embodied_navigation_agent2_20261106.json")["report"]
    alfred = load("embodied_alfred_agent3_20261106.json")
    habitat = load("embodied_habitat_agent4_20261106.json")
    protocol = load("embodied_protocol_agent1_20261106.json")
    tools = load("embodied_tools_agent5_20261106.json")
    replay = load("raw-replay.json")
    plan = load("embodied_main_repair_plan_20261106.json")
    records = {(x["episode_id"], x["version"]): x for x in inventory["records"]}
    episodes = sorted({r["episode_id"] for r in inventory["records"]})
    versions = ("Direct", "Full", "r02", "r04")
    changes = {}
    for p in (HERE / "source-before").glob("*.py"):
        target = ROOT / p.name.replace("tests__", "tests/")
        if target.exists() and sha(p) != sha(target):
            changes[str(target.relative_to(ROOT))] = {
                "before_sha256": sha(p), "after_sha256": sha(target),
                "preserved_path": str(p),
                "diff": "".join(difflib.unified_diff(
                    p.read_text().splitlines(True), target.read_text().splitlines(True),
                    fromfile=str(p), tofile=str(target)))}
    baseline = load("embodied_diagnosis_baseline_20261106.json")
    preservation = {
        "protected_files": len(baseline["protected"]),
        "other_project_files": len(baseline["other_projects"]),
        "protected_changed_or_missing": [
            name for name, digest in baseline["protected"].items()
            if not Path(name).is_file() or sha(Path(name)) != digest],
        "other_project_changed_or_missing": [
            name for name, digest in baseline["other_projects"].items()
            if not (ROOT.parent / name).is_file() or sha(ROOT.parent / name) != digest]}
    assert not preservation["protected_changed_or_missing"]
    assert not preservation["other_project_changed_or_missing"]
    pilot = None
    if (HERE / "pilot-verification.json").exists():
        pilot = load("pilot-verification.json")
    fullrun = load("fullrun-verification.json") if (HERE / "fullrun-verification.json").exists() else None
    output = {
        "schema": "embodied-performance-diagnosis-and-repair/1",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "offline_passed_pilot_running" if pilot is None else "validated",
        "scope": "15 development tasks, four historical versions; r05 independent pilot.",
        "count_boundary": "Historical task outcomes, proposal replay, and new live outcomes have separate denominators.",
        "inventory": inventory,
        "matrix": [{**{k: records[(ep, k)] for k in versions}, "episode_id": ep} for ep in episodes],
        "navigation": compact(navigation),
        "alfred": compact(alfred),
        "habitat": compact(habitat),
        "official_protocol": protocol,
        "tools_and_depth": tools,
        "selected_plan": plan,
        "raw_counterfactual_replay": replay,
        "source_changes": changes,
        "offline_regression": suite("regression-final.xml"),
        "format_feedback_regression": suite("format-feedback.xml"),
        "context_replay": load("context-replay.json"),
        "linux_regression": suite("linux-r05.xml") if (HERE / "linux-r05.xml").exists() else None,
        "pilot": pilot, "fullrun": fullrun, "preservation": preservation,
        "git_commit_or_push": False, "credentials_persisted": False}
    (ROOT / "runs/diagnosis-repair-r05-evidence.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# embodiedRunner：15 题下降归因、针对性修复与验证",
        "",
        "旧 Full 的 8/15 降到 r04 的 1/15，主要涉及新增控制协议阻断合法动作、"
        "无信息的请求与修复消耗预算，以及导航和家务策略偏离。"
        "完整上下文确已交付，但 Habitat 示例中的可执行计划实际漏提取，"
        "模型也继续把“下一步打算做什么”误写成“上一步已经完成”。"
        "当前轨迹不支持把下降归因于深度：r02/r04 没有运行 MoGe，旧 Full 的数值深度也没有被程序读取。",
        "",
        "五职责共同覆盖全部 60 条轨迹、7,542 个事件、772 次请求响应。"
        "先完成官方协议和完整轨迹阅读，再由主 agent 独立复核关键 RGB、原始响应和真实拒绝；"
        "修复与新实验均集中在 embodiedRunner。原始证据入口与事件索引见 "
        + link(ROOT / "runs/diagnosis-repair-r05-evidence.json", "可追溯证据 JSON") + "。",
        "",
        "## 结果与可比范围", "",
        "| 任务 | Direct | 旧 Full | r02 | r04 | 初始像素四版一致 |",
        "|---|---:|---:|---:|---:|---|"]
    for ep in episodes:
        rs = [records[(ep, v)] for v in versions]
        lines.append("| " + ep + " | " + " | ".join(str(int(r["success"])) for r in rs)
                     + " | " + ("是" if len({r["initial_pixel_sha256"] for r in rs}) == 1 else "否") + " |")
    lines += [
        "",
        "| 版本 | 成功 | 实际动作 | Qwen 请求 |",
        "|---|---:|---:|---:|",
        "| Direct | 12/15 | 157 | 193 |",
        "| 旧 Full | 8/15 | 216 | 228 |",
        "| r02 | 1/15 | 143 | 206 |",
        "| r04 | 1/15 | 91 | 145 |",
        "",
        "10 道家务任务的初始 RGB 四版像素一致，旧 Full 6/10 → r04 1/10；"
        "导航渲染差异不能解释这部分下降。Navigation004 的 r02/r04 初始图同一锅处于正前方，"
        "平均通道差 0.0051867、最大 6，不能仅由哈希变化断言不同起始相机。",
        "",
        "四版模型参数、任务来源、预算与参测 Skill 身份一致；"
        "772 次请求使用同一 MODEL_CONFIG，579 次 Full 请求包含对应 Skill。"
        "同名模型及相同参数不能证明服务端权重固定或输出确定。"
        "Direct 是本项目使用官方 prompt/parser/examples 的适配基线："
        "宿主只执行首动作，输出上限 4096，使用本项目预算；"
        "论文原始评测允许多动作执行并使用 2048 输出上限，故 12/15 不是原论文逐项复刻结果。",
        "",
        "r04 145 次请求含完整 Skill、动作表、精选示例、实际动作及完整公开反馈。"
        "130 次后续请求携带 progress；执行子目标同步不一致为 0；请求前最多 17 步历史，"
        "没有因 40 步上限丢失记录。文本均低于 16,000 字符目标，图片 token 另计。"
        "这证明交付覆盖，不证明示例选取、决策策略和门禁语义正确。",
        "",
        "## 官方协议与宿主新增控制", "",
        "| 环境 | 官方成功、进度及停止 | 动作/反馈语义 |",
        "|---|---|---|",
        "| Navigation | XZ 目标距离≤1 为成功，无需 Stop；done 也可为20步上限；公开progress恒0 | 平移/旋转ACK只证明动作执行；目标可见优先前进/横移，丢失目标才逐步旋转 |",
        "| ALFRED | 当前全部目标条件满足才成功；progress为条件完成比例 | Find设置命名控制目标；多数交互执行前清空它，包括失败pickup；put down有目标为Put，无目标为Drop |",
        "| Habitat | success检查当前完整PDDL goal；progress累计曾满足的subgoals；30步上限；must_call_stop=false | 命名导航/拾取/放置为宏动作，前置条件由环境检验；不要求检测框或认证mask；允许无效动作 |",
        "",
        "子目标 JSON、advance 证据检查、持物/目的地门禁、循环拦截、"
        "每 decision 三次生成和一次感知均为宿主新增协议。"
        "拒绝子目标会消耗调用但不执行环境动作；不能把这些拒绝直接称为官方动作失败。"
        "详细规则来源：", ""]
    for key, value in protocol["sources"].items():
        lines.append(f"- **{key}**：{value}")
    lines += ["", "## 全部任务的跨版本因果链", "",
              "下面的 eN 是原 events.jsonl 的零基事件索引，物理行号为 N+1。"
              "每条版本同时链接原轨迹，具体决策 ID、输入响应、工具与动作反馈保存在证据 JSON；"
              "历史源码行号按相应历史版本解释，修改前源码另保存在 source-before。"]
    groups = [
        ("Navigation", navigation["episodes"]),
        ("ALFRED", alfred["episodes"]),
        ("Habitat", habitat["tasks"])]
    for name, tasks in groups:
        for task in tasks:
            ep = task["episode_id"]
            lines += ["", "### " + ep, "", str(task.get("conclusion", task.get("comparison",
                           task.get("visual_path", task.get("reference_path_limit", "")))))]
            task_versions = task["versions"]
            if isinstance(task_versions, dict):
                task_versions = [{**v, "version": k} for k, v in task_versions.items()]
            for version in task_versions:
                label = version["version"]
                v = "Full" if label == "OldFull" else label
                rec = records[(ep, v)]
                lines += ["", "**" + label + "**："
                          + ("成功" if rec["success"] else "失败")
                          + f"；{rec['environment_steps']} 动作、{rec['event_kinds']['model_request']} 请求。"
                          + link(Path(rec["trajectory_path"]), "原轨迹") + "。"]
                first = version.get("first_divergence", version.get("first_deviation",
                        version.get("first_failed_continuation", version.get("first_effective_path_deviation"))))
                if first is not None:
                    lines.append("首次偏离/阻断：" + json.dumps(first, ensure_ascii=False) + "。")
                causal = version.get("causal_chain", "")
                if isinstance(causal, list):
                    for c in causal:
                        lines.append(anchored(c.get("fact", str(c)), rec)
                                     + "（事件：" + ",".join("e" + str(i) for i in c.get("events", [])) + "）。")
                else:
                    lines.append(anchored(str(causal), rec))
                for key in ("input_excerpt", "response_excerpt", "tool_io", "feedback",
                            "uncertainty", "not_causal", "counterevidence",
                            "perception_counterevidence", "why_context_not_enough"):
                    if key in version:
                        value = version[key]
                        lines.append(f"{key}：" + anchored(value if isinstance(value, str) else
                                     json.dumps(value, ensure_ascii=False), rec))
                if "anchors" in version:
                    lines.append("关键输入→响应→动作→反馈：" +
                                 json.dumps(version["anchors"], ensure_ascii=False))
            if "rgb_review" in task:
                lines.append("RGB复核：" + task["rgb_review"])
    lines += [
        "", "## 预算消耗：终止现象与实际原因分开", "",
        "对原 r04 请求卡及原响应逐项回放，145 次调用分为："
        "格式拒收10、子目标完成不支持17、实际感知选择14、已感知后重复选择7、接受程序97。"
        "97个程序对应91个真实动作、4个动作门禁拒绝（3循环、1已持物再次拾取）和2次空计划停止。"
        "17次子目标拒绝包含1次感知提案。旧检查的13次unsupported和4次perception错误是"
        "后续修复请求携带错误码的次数，不能替代实际拒绝计数17和7。",
        "",
        "r04 的9题单轮额度、3题总Qwen额度、2题空计划、1题成功，只是终止分类。"
        "以上逐题链解释其额度被重复感知、错误完成声明、相同格式错误、错误搜索或动作重规划如何消耗。"
        "Navigation004 的13个实际动作全部成功，18次请求，零循环拒绝；"
        "把该题归于“循环门禁太严”与原轨迹不符。",
        "", "## 工具配置与深度可靠性", "",
        "| 版本 | Qwen | DINO | SAM | Intern审核 | MoGe | 程序工具 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| Direct | 193 | 0 | 0 | 0 | 0 | 0 |",
        "| 旧 Full | 228 | 200 | 130 | 130 | 218 | 11 |",
        "| r02 | 206 | 14 | 13 | 13 | 0 | 12 |",
        "| r04 | 145 | 14 | 8 | 8 | 0 | 4 |",
        "",
        "27次程序工具全部为 view_candidates；几何工具、ReadObservation 和程序 ctx.frames 读取均为0。"
        "旧 Full 的218次 MoGe 推理没有数值被程序消费，不能将成功归功于深度。"
        "r02/r04 的16次候选工具发生于未准备检测的 decision，原来返回ok/count=0；"
        "14个程序据此选择搜索或停止分支。此为真实状态缺陷，但 r02 Navigation004 读空仍前进成功，"
        "不能把所有失败由此解释。",
        "",
        "求解模型只收到当前500×500 RGB。Navigation模拟器具备米制深度；"
        "ALFRED2.1以毫米输出，V2关闭官方深度渲染；Habitat使用head_rgb。"
        "旧 Full 实际准备的是独立MoGe2预测，只授予每视图相对深度/点图能力，"
        "没有世界变换或米制授权。",
        "",
        "218个预测网格与RGB均为500×500；有效区域没有非正/非有限深度，"
        "points.Z与depth最大差约1.91e-6，预测K重投影最大误差约0.000136px。"
        "这是内部一致性；缺少同帧官方深度GT，不能视为物理精度验证。"
        "Navigation000旧Full预测FOV约87.906°，官方默认100°，"
        "表明内参估计存在偏离；该题成功且没有消费几何，无法将该偏差归因到动作失败。",
        "",
        "151个候选均保持mask_quality_assessed=false、geometry_usable=false。"
        "Toaster错误框、toy-airplane实际剪刀/机器人框、plate实际sink框的审核拒绝有RGB依据。"
        "镜中浴缸类别可成立，但不能认证可交互物体表面。"
        "5×5局部深度中位数也不等于mask内物体表面距离。",
        "",
        "保留当前RGB、完整动作与反馈、按需检测、类别审核、候选schema、几何权限边界及结果身份回执。"
        "避免每步无条件深度计算、未准备候选的成功空结果、同decision反复感知，"
        "以及已有tool.data后不必要地Yield再调用模型。"
        "候选替代策略与同帧官方深度精度审计适合作为分别冻结的新变量。",
        "", "## 方案选择与实际修改", "",
        "| 方案 | 证据与选择 |",
        "|---|---|",
        "| 针对完成条件、字段反馈、示例和感知后出口修复 | 已实施；直接对应真实拒绝和预算浪费，并保留反例 |",
        "| 全部取消门禁 | 不作为本次方案；会丢失双拾取、错误放置及循环保护 |",
        "| 增加调用预算 | 不采用；相同错误/重复感知无新增信息且改变比较预算 |",
        "| 再次压缩上下文 | 不采用；完整交付已验证，具体语义/示例缺失才是待修问题 |",
        "| 替代候选/官方深度 | 分开实验；会改变感知变量，目前没有深度消费导致失败的证据 |",
        "",
        "实际修复如下：", "",
        "- task_card：仅从明确find/locate描述保守解析空object，仍要求最新成功命名Find完全匹配；"
        "增加keep执行待完成动作、revise纠正目标及Put/Drop区别的反馈。",
        "- action_solver/action_host：格式错误仍拒收；补有限字段原因与修复提示，跨invalid和新decision保留。",
        "- task_context：恢复Output及单双引号action_id完整官方计划，保留来源hash；"
        "恢复Navigation官方策略；强调命名宏动作不以检测mask为前置条件。",
        "- request_options：已有感知或最后调用明确输出program_source，tool.data可在同程序直接消费；预算保持原值。",
        "- evidence/sandbox_wire/action_tools：候选可用能力仅由真实检测调用授予；"
        "检测完成但为空仍合法，未检测不得伪装成功空列表；不修改共享src。",
        "- 测试与部署：空检测测试改为实际空检测器；fake-run记录真实感知配置身份；"
        "归档使用独立文件且不覆盖旧包，pilot配置仅缩小选择范围。",
        "",
        "修改前原件与逐文件hash/diff见证据JSON source_changes；"
        "完整Skill正文、模型参数、预算、持物/目的地/循环/几何/身份门禁及停止语义保留。"
        "空计划额外宿主反馈、历史Find复用及扩大候选均未混入本次变量。",
        "", "## 离线验证", "",
        "完整项目回归272通过、1项Linux专用检查在Mac跳过，4个subtests通过；"
        "补充跨decision格式提示测试1通过。Linux原生隔离、实际空检测Yield/Read和格式反馈3项通过。",
        "",
        "145次原始响应重放：8次拒绝改变为可接受定位完成，"
        "分别为ALFRED000 e15/e19/e23、003 e15/e19/e23、001 e44/e48。"
        "001仍已持Plate，两个pickup105提案继续被already_holding_before_pickup拒绝。"
        "ALFRED002 e37及e74/e78/e82提前完成/不匹配Cabinet保持拒绝。"
        "10个原始格式错误全部仍被拒收，原因9个advance.source=none、1个depth层级错误。"
        "16个未准备候选的真实原程序在隔离执行中均产生confidence_gate拒绝回执且不提交动作。"
        "这验证协议行为，不代表反事实任务将成功。",
        "",
        "在145个原始公开请求快照上重建新提示，完整任务/动作表/历史/Skill及request_options检查均通过。"
        "文本最大15,947字符、平均13,797.75、超16,000目标为0；"
        "3次因可选段落空间不足按整例移除示例，关键字段与实际反馈完整保留。",
        "", "## 新真实运行", ""]
    if pilot is None:
        lines += ["r05七题pilot已启动，结果尚未完成，不报告成功率改善。"]
    else:
        lines += [pilot["narrative"], "", "| 任务 | r04 | r05 | r05动作 | r05请求 | 初始像素与r04 |",
                  "|---|---:|---:|---:|---:|---|"]
        for row in pilot["episodes"]:
            lines.append(f"| {row['episode_id']} | {int(row['r04_success'])} | "
                         f"{int(row['success'])} | {row['steps']} | {row['qwen']} | "
                         + ("一致" if row["initial_pixel_equal"] else "不同，另报数值差") + " |")
        lines += ["", "运行身份/预算/Skill/参数/完整轨迹与源文件审计：" +
                  link(HERE / "pilot-verification.json"), "",
                  "未解决案例：" + pilot.get("unresolved", "见逐题新轨迹分析。")]
    if fullrun is not None:
        lines += ["", fullrun["narrative"]]
    lines += [
        "",
        "联合控制/提示修复作为一个变量，不能分解每项组件的独立成功率贡献。"
        "参考答案在线隐藏，操作性服务失败不自动重试。pilot与任何全15题运行分母分别报告。"
        "所有成功以官方结果为准，公开动作ACK和历史progress不代替任务成功。",
        "",
        "## 保存与后续实验", "",
        f"再次核验{preservation['protected_files']}个原始存档文件、"
        f"{preservation['other_project_files']}个其他项目已有修改文件，变化/缺失均为0。"
        "r03中断记录、r04已结束的冻结目录及全部原始JSON/图像/失败轨迹保留。"
        "未commit/push，凭据未写入源码、报告或日志。",
        "",
        "下一步优先在相同任务、参数与预算下对仍失败的策略做受控比较："
        "保持当前修复，单独比较简化子目标生命周期的语义动作控制；"
        "对Navigation003的四向碰撞困局保留完整动作/视图作独立诊断。"
        "之后再分别验证替代候选和预测深度物理精度，不把感知配置变化混进控制协议的效果。",
    ]
    (ROOT / "runs/diagnosis-repair-r05.md").write_text("\n".join(lines) + "\n")
    (HERE / "preservation-final.json").write_text(json.dumps(preservation, indent=2) + "\n")
    print(json.dumps({"report": str(ROOT / "runs/diagnosis-repair-r05.md"),
                      "evidence": str(ROOT / "runs/diagnosis-repair-r05-evidence.json"),
                      "status": output["status"], "source_files_changed": len(changes),
                      "preservation": preservation}))


if __name__ == "__main__":
    main()
