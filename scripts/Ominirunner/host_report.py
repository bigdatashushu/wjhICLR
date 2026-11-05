"""Report a completed host-review replay and its original paired comparison."""
from collections import Counter
from pathlib import Path

from .common import read, write


def report(root, scores, audit, frozen):
    prior_root = Path(frozen["question_replay"]["source_root"])
    prior = read(prior_root / "scores.json")
    if prior["run_identity"] != frozen["question_replay"]["source_run_identity"]:
        raise ValueError("prior_score_identity_changed")
    ids = [item["sample_id"] for item in read(root / "inputs.json")]
    current = {(r["sample_id"], r["arm"]): r for r in scores["scored"]}
    old = {(r["sample_id"], r["arm"]): r for r in prior["scored"]}
    if set(current) != set(old) or len(current) != 2 * len(ids):
        raise ValueError("comparison_question_denominator_changed")
    if any(current[(i, "direct")]["score"] != old[(i, "direct")]["score"] for i in ids):
        raise ValueError("reused_direct_score_changed")
    rows = []
    for sample_id in ids:
        rows.append({"sample_id": sample_id, "group": current[(sample_id, "skill")]["group"],
                     "direct_score": old[(sample_id, "direct")]["score"],
                     "prior_skill_score": old[(sample_id, "skill")]["score"],
                     "new_skill_score": current[(sample_id, "skill")]["score"],
                     "new_status": current[(sample_id, "skill")]["status"]})
    def paired(key):
        wins = sum(r["new_skill_score"] > r[key] for r in rows)
        losses = sum(r["new_skill_score"] < r[key] for r in rows)
        return {"wins": wins, "ties": len(rows) - wins - losses, "losses": losses}
    comparison = {
        "schema": "omni-host-replay-comparison/1", "run_identity": scores["run_identity"],
        "prior_run_identity": prior["run_identity"], "n": len(ids), "rows": rows,
        "versus_prior_skill": paired("prior_skill_score"), "versus_direct": paired("direct_score"),
        "new_solver_requests": audit["solver_requests"], "new_intern_requests": audit["jev_requests"],
        "reused_direct_requests": audit["historical_requests"], "reference_answers_in_report": False,
    }
    write(root / "comparison.json", comparison, exclusive=True)
    lines = [f"# Omni3D 同一 {len(ids)} 题重跑", "",
             f"原题序、图文、缓存和 Qwen 参数保持冻结；复用 {len(ids)} 个历史 direct，"
             "本次执行宿主 Intern 审核流程。", "",
             "| 方法 | 逐题平均分 | Float MRA | Int | Yes/No | 其他文本 | 合法答案 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    methods = [("历史 direct", prior["table"]["direct"]),
               ("此前 Skill", prior["table"]["skill"]), ("新流程", scores["table"]["skill"])]
    for name, value in methods:
        metrics = value["metrics"]
        cells = [f'{metrics[g]["value"]:.1%}（{metrics[g]["n"]}）' if g in metrics else "—"
                 for g in ("float", "int", "yes_no", "other_text")]
        lines.append(f'| {name} | {value["sample_weighted_score"]:.1%} | '
                     + " | ".join(cells) + f' | {value["legal"]}/{len(ids)} |')
    for name, key in (("此前 Skill", "versus_prior_skill"), ("direct", "versus_direct")):
        p = comparison[key]
        lines += ["", f'相对 {name}：胜 {p["wins"]}、平 {p["ties"]}、负 {p["losses"]}。']
    episode_rows = [r for r in audit["episodes"] if r["arm"] == "skill"]
    decisions = Counter()
    statuses = Counter()
    for episode in episode_rows:
        decisions.update(episode["surface_review_decisions"])
        statuses.update(episode["round_statuses"])
    lines += ["",
              f'新增请求：Qwen {audit["solver_requests"]} 次，Intern {audit["jev_requests"]} 次；'
              f'历史 direct {audit["historical_requests"]} 次，本轮新增 direct 0 次。',
              f'Qwen token：{audit["usage"]}；Intern token：{audit["jev_usage"]}。',
              f'审核决定：{dict(decisions)}；程序轮状态：{dict(statuses)}。',
              "",
              "主模型提出对象、量别和原图点位；Intern 对实际 RGB 作二元审核，"
              "仅 p(yes)>0.5 放行。下一请求交付获批精确事实或重选通知，"
              "不交付概率、审核图或被拒测量值。",
              f'每题审核上限 {frozen["measurement_reviewer"]["max_requests"]} 次，'
              "主模型最多 5 个程序轮加 1 个最终 JSON 轮。",
              "上下文、审核回执、替换关系和本地／共享收费已离线重放。",
              "",
              "统计覆盖全部 20 题。这是固定开发题集的重跑观察；"
              "小样本和生成波动限制了对全量表现及单项修改效果的推断。",
              "这些题没有测量审核真值，审核决定数量不能用于计算错误放行率。",
              "",
              "| 题号 | 历史 direct | 此前 Skill | 新流程 | 新状态 |",
              "|---|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f'| {r["sample_id"]} | {r["direct_score"]:.2f} | '
                     f'{r["prior_skill_score"]:.2f} | {r["new_skill_score"]:.2f} | {r["new_status"]} |')
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison
