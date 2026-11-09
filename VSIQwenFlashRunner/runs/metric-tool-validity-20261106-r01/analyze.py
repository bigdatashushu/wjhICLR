"""Independent host scoring and audit report over immutable public measurements."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "g1m1-thinking40-20261106-r01"
SOURCE = FROZEN / "code/source/src/skill3d"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location("frozen_mra", SOURCE / "evaluation/mra.py")
mra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mra)

traces = load(HERE / "public_tool_traces.json")
replay = load(HERE / "cpu_replay.json")
visuals = load(HERE / "visual_evidence.json")
assert replay["primitive_sha256"] == sha(SOURCE / "tools/distance_primitives.py")
assert traces["model_requests"] == replay["model_requests"] == visuals["model_requests"] == 0
assert traces["reference_answers_read"] is replay["reference_answers_read"] is False
scores = {r["qa_id"]: r for r in load(FROZEN / "resume_results.json")["pairs"]}
# Evaluation labels are read only in this independent host analysis, after replay.
metadata_path = FROZEN / "inputs/host_evaluation/metadata.jsonl"
metadata = {str(r["qa_id"]): r for r in map(json.loads, metadata_path.read_text().splitlines())}
replays = {r["qa_id"]: r for r in replay["records"]}
TASKS = {
    "object_abs_distance": ("绝对距离", "object_surface_distance"),
    "object_size_estimation": ("物体尺寸", "object_3d_extent"),
    "room_size_estimation": ("房间面积", "plane_fit_room_size"),
}
rows = []
successes = []
for record in sorted(traces["records"], key=lambda r: (r["variant"]["question_type"], int(r["qa_id"]))):
    qa = record["qa_id"]
    truth = float(metadata[qa]["ground_truth"])
    score = scores[qa]
    rr = replays[qa]
    calls = rr["calls"]
    raw_results = {r["result_id"]: r for rd in record["rounds"] for r in rd["results"]}
    used_ids = record["answer_payload"]["used_result_ids"]
    numeric_answer = float(record["answer"])
    derived = record["answer_basis"] == "tool_derived"
    assert abs(mra.mra_one(numeric_answer, truth) - score["system_score"]) < 1e-12
    assert abs(mra.mra_one(float(score["direct_answer"]), truth) - score["direct_score"]) < 1e-12
    selected = None
    if derived:
        assert len(used_ids) == 1
        selected = raw_results[used_ids[0]]
        assert selected["status"] == "ok"
        payload = selected["payload"]
        task = record["variant"]["question_type"]
        if task == "object_abs_distance":
            value = float(payload["distance"])
            assert abs(value - payload["distance_normalized"] * rr["metric_scale"]) < 1e-12
        elif task == "object_size_estimation":
            value = float(payload["extent_longest_metric"]) * 100
            assert max(abs(a * rr["metric_scale"] - b) for a, b in
                       zip(payload["extent_normalized"], payload["extent_metric"])) < 1e-12
            assert abs(max(payload["extent_metric"]) - payload["extent_longest_metric"]) < 1e-12
        else:
            value = float(payload["room_area_m2"])
            rc = next(c for c in calls if c["result_id"] == used_ids[0])
            d1, d2 = rc["replay"]["wall_extents_normalized"]
            assert abs(d1 * d2 * rr["metric_scale"] ** 2 - value) < 1e-12
        assert abs(numeric_answer - value) < 1e-10
        assert record["attribution"]["ledger"]["verified_used_result_ids"] == used_ids
    else:
        assert not used_ids
        value = None
    for call in calls:
        if call["original_status"] == "ok":
            assert all(diff == 0.0 for diff in call["differences"].values())
            successes.append(call)
    rows.append({
        "qa_id": qa, "task": record["variant"]["question_type"],
        "question": metadata[qa]["question"], "reference_answer": truth,
        "unit": record["answer_payload"]["unit"], "final_answer": numeric_answer,
        "answer_basis": record["answer_basis"],
        "selected_result_ids": used_ids,
        "selected_metric_answer": value,
        "selected_object_args": None if selected is None else selected["args"],
        "relative_error_selected": None if value is None else mra.relative_error(value, truth),
        "relative_error_final": mra.relative_error(numeric_answer, truth),
        "system_mra": score["system_score"],
        "direct_answer": float(score["direct_answer"]), "direct_mra": score["direct_score"],
        "metric_call_count": len(calls),
        "successful_metric_call_count": sum(c["original_status"] == "ok" for c in calls),
        "failed_metric_call_count": sum(c["original_status"] != "ok" for c in calls),
        "metric_scale": rr["metric_scale"], "scale_self_consistency": rr["scale_self_consistency"],
        "outcome_source": record["outcome_source"], "outcome_sha256": record["outcome_sha256"],
        "plan_sha256": record["plan_sha256"],
        "failure_codes": [(c.get("failure_cause") or {}).get("code") for c in calls
                          if c["original_status"] != "ok"],
    })

groups = {}
for task in TASKS:
    group = [r for r in rows if r["task"] == task]
    measured = [r for r in group if r["selected_metric_answer"] is not None]
    groups[task] = {
        "questions": len(group),
        "questions_with_metric_attempt": sum(r["metric_call_count"] > 0 for r in group),
        "questions_with_successful_metric_output": sum(r["successful_metric_call_count"] > 0 for r in group),
        "questions_using_metric_answer": len(measured),
        "metric_calls": sum(r["metric_call_count"] for r in group),
        "successful_metric_calls": sum(r["successful_metric_call_count"] for r in group),
        "selected_mean_relative_error": statistics.mean(r["relative_error_selected"] for r in measured),
        "selected_within_10_percent": sum(r["relative_error_selected"] <= .1 for r in measured),
        "selected_within_20_percent": sum(r["relative_error_selected"] <= .2 for r in measured),
        "selected_mra": statistics.mean(r["system_mra"] for r in measured),
        "same_selected_questions_direct_mra": statistics.mean(r["direct_mra"] for r in measured),
        "all_questions_system_mra": statistics.mean(r["system_mra"] for r in group),
        "all_questions_direct_mra": statistics.mean(r["direct_mra"] for r in group),
    }
measured = [r for r in rows if r["selected_metric_answer"] is not None]
summary = {
    "questions": len(rows),
    "questions_using_metric_answer": len(measured),
    "successful_calls_replayed": len(successes),
    "maximum_replay_numeric_difference": 0.0,
    "metric_calls": sum(r["metric_call_count"] for r in rows),
    "failed_metric_calls": sum(r["failed_metric_call_count"] for r in rows),
    "selected_within_10_percent": sum(r["relative_error_selected"] <= .1 for r in measured),
    "selected_within_20_percent": sum(r["relative_error_selected"] <= .2 for r in measured),
    "all_questions_system_mra": statistics.mean(r["system_mra"] for r in rows),
    "all_questions_direct_mra": statistics.mean(r["direct_mra"] for r in rows),
    "selected_mra": statistics.mean(r["system_mra"] for r in measured),
    "same_selected_questions_direct_mra": statistics.mean(r["direct_mra"] for r in measured),
}
assert summary["questions"] == 15 and summary["questions_using_metric_answer"] == 11
assert summary["successful_calls_replayed"] == 20 and summary["failed_metric_calls"] == 2
assert summary["selected_within_20_percent"] == 4
audit = {
    "schema": "metric-tool-validity-report/1", "date": "2026-11-06",
    "frozen_run": str(FROZEN),
    "scope": "Three metric tasks in the existing frozen 40-question diagnostic; five questions each",
    "model_requests": 0,
    "label_access": "Reference answers read only by independent host scoring after label-free CPU replay",
    "source_identities": {
        "trace_collection_sha256": sha(HERE / "public_tool_traces.json"),
        "cpu_replay_sha256": sha(HERE / "cpu_replay.json"),
        "score_archive_sha256": sha(FROZEN / "resume_results.json"),
        "evaluation_metadata_sha256": sha(metadata_path),
        "distance_primitives_sha256": replay["primitive_sha256"],
        "mra_source_sha256": sha(SOURCE / "evaluation/mra.py"),
    },
    "checks": {
        "all_successful_original_calls_cpu_replayed": True,
        "all_original_used_result_ids_verified": True,
        "all_selected_unit_conversions_verified": True,
        "scoring_matches_original_frozen_scores": True,
        "binding_manifests_payloads_and_replayed_geometry_hash_verified": True,
        "four_visual_cases_rgb_and_mask_hash_verified": True,
    },
    "summary": summary, "groups": groups, "questions": rows,
    "interpretation_limits": [
        "20% is an explicit diagnostic error band, not a calibrated gate or official accuracy metric",
        "Extra candidates are offline diagnostics, never substituted for original selected answers",
        "Internal scale consistency is not independent absolute-scale accuracy",
        "System/direct differences do not isolate the causal effect of m1; a matched g1m0 arm was not rerun",
        "Small diagnostic panel is not the full test-set performance",
    ],
}
(HERE / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))

report = [
    "# 三类米制工具实际有效性核验",
    "",
    "结论：确定性计算和单位换算通过核验，当前测量流程的任务有效性不足。"
    "成功返回不能作为完整对象或完整房间测量准确的依据；本轮只能支持局部有效，尚不能支持稳定增益。",
    "",
    "复用冻结的 `g1m1-thinking40-20261106-r01`：三类各 5 题，共 15 题。"
    "没有新增模型/GPU调用，也没有修改工具、Skill、冻结输入或结果。"
    "服务器重放不读标签，参考答案只在本机独立评测时读取。",
    "",
    "20 次成功调用已用原始点云、原参数在 CPU 上全部复现，所有核对字段数值差为 0。"
    "对象点云、绑定清单及房间点图与冻结身份的哈希一致。"
    "11 个实际采用的米制结果，尺度相乘、米到厘米乘 100、面积按尺度平方、答案归因均通过。",
    "",
    "| 题型／工具 | 产生并采用测量的题数 | 所用测量平均相对误差 | 相对误差≤20% | 所用测量 MRA | 同题直答 MRA |",
    "|---|---:|---:|---:|---:|---:|",
]
for task, (name, tool) in TASKS.items():
    g = groups[task]
    report.append(
        f"| {name}／`{tool}` | {g['questions_using_metric_answer']}/5 | "
        f"{g['selected_mean_relative_error']:.1%} | {g['selected_within_20_percent']}/{g['questions_using_metric_answer']} | "
        f"{g['selected_mra']:.1%} | {g['same_selected_questions_direct_mra']:.1%} |"
    )
report += [
    "",
    "MRA 是官方数值题相对误差多阈值平均分，不是正确率。20% 是本次直观诊断误差带，"
    "不作为新的工具授权阈值。其余 4 题是视觉估计：2 题测量失败、2 题目标候选缺失而未调用。"
    "15 题最终平均 MRA 为 50.0%，同题直答为 84.7%。这里只是现有系统与直答的配对观察，"
    "没有单独重跑 g1m0，不能把全部差异归因于 m1。",
    "",
    "## 绝对距离",
    "",
    "- QA798：壁炉—电视 1.2201 m，参考 1.3 m，相对误差 6.1%，是有效测量实例。",
    "- QA4636：门—衣柜 0.04104 m，参考 0.5 m，误差 91.8%。原算法确实得到该值，"
    "1% 双向最近邻分位数、尺度换算及返回字段都正确。改成同一预处理点云的严格最小值为 "
    "0.00974 m，与参考值偏差更大，不能靠改取最小值修复。",
    "- 图像显示所选门 `obj_1` 的掩码主要覆盖打开门的薄边，衣柜 `obj_3` 也来自单帧。"
    "这对可见表面的距离不能证明完整对象、门状态及参考标注范围一致。"
    "现有证据不足以把该例唯一归因于实例选择、动态状态、重建或绝对尺度。",
    "- QA1599：鼠标／插排原始点云经预处理仅剩 31／70 点，均不足 100 点下限，"
    "工具显式失败并退回视觉估计。QA4681 缺 window，QA4757 缺 closet，均没有米制测量。",
    "",
    "## 物体尺寸",
    "",
    "- 4 个实际采用结果全部低估，误差 17.3%–46.4%。最长世界轴 extent 是可见点集的包围范围，"
    "未验证完整表面覆盖，且并非物体自身坐标系的长宽高。",
    "- QA3400：参考沙发 229 cm，实际只调用第一个候选 `obj_0`，得到 122.815 cm。"
    "该候选只在第 2 帧可见，沙发在画面右侧被截断；程序把类别／target 角色与 `validated` 当成了完整目标。",
    "- 冻结资产中 `obj_3` 第 4 帧覆盖更完整的沙发，离线复算得到 218.767 cm，误差 4.5%。"
    "这一结果仅用于解释候选选择与覆盖问题，没有替换原答案，也不计入上表。"
    "它说明相同尺度下工具可以给出有效测量，当前选择流程未稳定选到合适表面。",
    "- QA2385：砧板原采用 39.348 cm，参考 55 cm。另一个通过类别准入的候选 `obj_2` "
    "在第 4 帧的掩码覆盖餐桌，说明 `category-only`／`quality_assessed=false` 不能作为目标实例质量证明。",
    "- QA373：stove 两候选因 `instance_not_suspect` 未通过绑定，调用被拒绝；最终 60 cm "
    "来自视觉估计，不能计为米制工具成功。",
    "",
    "## 房间面积",
    "",
    "- 5 题均成功测量，但 4 题高估，最大误差 145.7%。",
    "- QA3822：返回 38.326 m²，参考 15.6 m²；`fit_quality=0.899`，降级标记为空。"
    "重放的平面内矩形边长为 8.456×4.533 m，乘积确实是返回面积。",
    "- 源码用地面法向确定平面，再对整场有效点云在平面内两个 PCA 方向取 1%–99% 范围，"
    "相乘得到矩形面积。它没有求实际地板边界多边形或多房间面积并集，"
    "因此不能保证范围与题目要求的合并空间一致。该例仍可能叠加尺度／重建误差。",
    "- `fit_quality=sqrt(floor_fit_ratio*ground_planarity)` 衡量候选地面点的平面拟合，"
    "不衡量房间边界完整性或面积准确性；它不能替代面积核验。",
    "",
    "## 逐题原始采用结果",
    "",
    "| QA | 题型 | 参考值 | 原采用工具值 | 最终答案 | 工具相对误差 | 系统／直答 MRA | 原路径 |",
    "|---|---|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    name = TASKS[row["task"]][0]
    val = "—" if row["selected_metric_answer"] is None else f"{row['selected_metric_answer']:.4f}"
    err = "—" if row["relative_error_selected"] is None else f"{row['relative_error_selected']:.1%}"
    path = "采用米制工具" if row["selected_metric_answer"] is not None else (
        "工具失败后视觉估计" if row["failed_metric_call_count"] else "缺目标，视觉估计"
    )
    report.append(f"| {row['qa_id']} | {name} | {row['reference_answer']:g} {row['unit']} | {val} | "
                  f"{row['final_answer']:.4f} | {err} | {row['system_mra']:.0%}／{row['direct_mra']:.0%} | {path} |")
report += [
    "",
    "## 证据与复核",
    "",
    "- [结构化审计结果及来源哈希](audit.json)",
    "- [全部 15 题公开轨迹、原采用 result_id、调用参数](public_tool_traces.json)",
    "- [原点云 CPU 重放与逐字段差值](cpu_replay.json)",
    "- [沙发候选及覆盖图](qa3400_rgb_masks_extended.png)",
    "- [门／衣柜原帧与掩码](qa4636_rgb_masks.png)",
    "- [砧板候选原帧与掩码](qa2385_rgb_masks.png)",
    "- [房间原始帧](qa3822_rgb_masks.png)",
    "",
    f"冻结实现：`{SOURCE / 'tools/geometry_tools.py'}`，距离工具行 899–944、尺寸行 401–442、"
    "房间行 466–552；`distance_primitives.py` 距离行 735–828、尺寸行 847–919、"
    "房间范围行 1179–1196、面积换算行 1289–1298。",
    "",
    "本轮验证支持保留这些工具作为有条件的测量来源；"
    "把类别通过、空降级标记或高地面拟合分直接等同于可信绝对测量，当前证据不支持。"
    "后续应先处理完整目标表面选择与房间边界定义，再用独立样本核验绝对尺度和整体增益。",
]
(HERE / "report.md").write_text("\n".join(report) + "\n")
print(json.dumps({"summary": summary, "groups": groups, "report": str(HERE / "report.md")}, ensure_ascii=False))
