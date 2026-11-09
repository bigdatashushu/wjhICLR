"""Describe observed frozen camera-pose results and audit public artifacts."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path

from ..traces import digest, file_digest, read, write
from . import vggt_pose_trial as trial

NAMES = {
    "Positional Relationship (Cam.–Cam.)": "相机–相机",
    "Positional Relationship (Cam.–Obj.)": "相机–物体",
    "Positional Relationship (Cam.–Reg.)": "相机–区域",
    "Positional Relationship (Obj.–Obj.)": "物体–物体",
    "Positional Relationship (Obj.–Reg.)": "物体–区域",
    "Positional Relationship (Reg.–Reg.)": "区域–区域",
}


def report(directory=trial.DESTINATION):
    directory = Path(directory)
    manifest = trial.validate(directory, requests=True)
    frozen = trial.validate_frozen(directory, manifest)
    score = read(directory / "score.json")
    if (digest({k: v for k, v in score.items() if k != "score_identity"}) != score["score_identity"]
            or score["freeze_identity"] != frozen["freeze_identity"]
            or score["run_identity"] != manifest["run_identity"]):
        raise ValueError("score_identity_changed")
    predictions = {(p["sample_id"], p["arm"]): p for p in frozen["predictions"]}
    for pair in score["pairs"]:
        for arm in ("direct", "pose"):
            p = predictions[pair["sample_id"], arm]
            if (any(pair[arm][k] != v for k, v in p.items())
                    or pair[arm]["correct"] != (
                        p["status"] == "answered" and p["answer"] == pair["reference_answer"])):
                raise ValueError("scored_prediction_changed")
    if trial.summarize(score["pairs"]) != score["overall"]:
        raise ValueError("score_aggregate_changed")
    responses = {"direct": [], "pose": []}
    for item in manifest["items"]:
        for arm in ("direct", "pose"):
            result = read(trial.paths(directory, item, arm)[1])
            if "response" in result:
                responses[arm].append(result["response"])
    reported = Counter(json.dumps(r["reported"], sort_keys=True, ensure_ascii=False)
                       for arm in responses.values() for r in arm)
    usage = {}
    for arm, records in responses.items():
        numeric = Counter()
        for r in records:
            for key, value in r["usage"].items():
                if type(value) in (int, float):
                    numeric[key] += value
        usage[arm] = dict(numeric)
    overall = score["overall"]
    changed_from_old = [p["sample_id"] for p in score["pairs"]
                        if p["old_direct_answer"] != p["direct"]["answer"]]
    summary = (
        f"本轮六题：原图答对 {overall['direct_correct']}/6"
        f"（{overall['direct_accuracy']:.1%}），原图＋VGGT 相机位姿答对 "
        f"{overall['pose_correct']}/6（{overall['pose_accuracy']:.1%}），"
        f"变化 {overall['delta_percentage_points']:+.1f} 个百分点。"
        f"位姿组修复 {overall['gains']} 题，退步 {overall['regressions']} 题，"
        f"两组答案不同 {overall['answers_changed']} 题。"
    )
    conclusion = (
        "这批结果支持继续检验相机位姿的帮助。"
        if overall["pose_correct"] > overall["direct_correct"] else
        "这批结果未显示相机位姿带来准确率提升。"
    )
    lines = [
        "# MMSI：原图与联合 VGGT 相机位姿对比",
        "", summary, "", conclusion,
        "六题取自此前已运行的题单，六类各一题，包含历史退步题 MMSI_926 和 MMSI_874。"
        "这是开发诊断，结果不能当作独立测试集或整套 MMSI 的提升。",
        "", "| 题目 | 类型 | 参考答案 | 原图 | 原图＋位姿 | 配对变化 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for pair in score["pairs"]:
        direct, pose = pair["direct"], pair["pose"]
        change = ("修复" if not direct["correct"] and pose["correct"] else
                  "退步" if direct["correct"] and not pose["correct"] else
                  "均正确" if direct["correct"] else "均错误")
        cell = lambda p: (p["answer"] or "无合法答案") + (" ✓" if p["correct"] else " ✗")
        lines.append(
            f"| {pair['sample_id']} | {NAMES[pair['task_type']]} | {pair['reference_answer']} | "
            f"{cell(direct)} | {cell(pose)} | {change} |")
    lines += [
        "", "每题两张原始 RGB 在 VGGT 中联合估计，使用 1B 权重、官方 crop 预处理"
        "（宽 518、patch 14）、BF16 主干和 FP32 相机头。严格核对全部权重后，"
        "仅运行聚合主干和相机头；深度、点图和跟踪头均未执行。",
        "", "模型两组都收到相同字节、相同顺序的原始 RGB，以及相同官方题目、选项"
        "和答案指令。位姿组增加两台相机的双向相对位置、旋转矩阵及前向向量文字。"
        "相机坐标为 x 向右、y 向下、z 向前；位置由 C=−Rᵀt 换算，"
        "两台相机间距归一化为 1。相机位姿没有经过真值精度核验，"
        "不提供米制距离、物体坐标、房间前方或地理朝向。",
        "", "请求模型为 gpt-5.6-terra，reasoning=high，temperature=0，seed=137，"
        "max_output_tokens=8192，单次超时 180 秒。六题使用六个 API 槽，"
        "每题两组共享同一槽；原图先发与位姿先发各三题。总共派发 12 次，"
        "最多同时 6 次，每组每题一次，无工具调用，无自动重试。",
        "", "服务实际回报的配置如下（保留服务回报，不把请求参数视为实际采样保证）：",
    ]
    lines += ["", "| 服务回报 | 响应数 |", "| --- | --- |"]
    for value, count in sorted(reported.items()):
        lines.append(f"| {value} | {count} |")
    lines += [
        "", f"本轮原图答案与历史原图不同的题目：{', '.join(changed_from_old) or '无'}。"
        "跨轮 API 槽和响应采样不构成固定随机性保证；历史结果用于诊断背景，"
        "本轮效果按同一轮六个完整配对计算。",
        "", "预测全部冻结后，独立评分进程从已验证的历史评分文件取参考答案；"
        "GPU 和在线请求进程不读取参考答案。",
        "", f"- 运行身份：`{manifest['run_identity']}`",
        f"- 预测冻结身份：`{frozen['freeze_identity']}`",
        f"- 评分身份：`{score['score_identity']}`",
        f"- VGGT 权重 SHA256：`{manifest['weight_sha256']}`",
        "- 原始相机外参、内参、位姿编码及预处理哈希：`gpu/poses/*.json`",
        "- 输入与请求哈希：`manifest.json`；逐次响应及用量：`traces/*.result.json`",
        "", "运行入口：`python -B -m GPTexp.MMSIGPT.vggt_pose_trial`。"
        "该入口要求新的实验目录，已执行目录拒绝自动重启。",
        "",
    ]
    report_path = directory / "报告.md"
    if report_path.exists():
        raise ValueError("report_already_exists")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    audit = {
        "status": "passed", "run_identity": manifest["run_identity"],
        "freeze_identity": frozen["freeze_identity"], "score_identity": score["score_identity"],
        "paired_questions": 6, "model_requests": 12, "max_inflight": 6,
        "legal_answers": {arm: sum(p[arm]["status"] == "answered" for p in score["pairs"])
                          for arm in ("direct", "pose")},
        "tool_calls": sum(r["tool_call_count"] for rows in responses.values() for r in rows),
        "reference_answer_visible_in_model_requests": False, "depth_images_sent": False,
        "reported_metadata": [{"value": json.loads(value), "count": count}
                              for value, count in sorted(reported.items())],
        "numeric_usage_by_arm": usage, "original_direct_changes_from_history": changed_from_old,
        "report_sha256": file_digest(report_path), "report_source_sha256": file_digest(__file__),
        "verified_at": datetime.now().isoformat(),
    }
    # Never write secret values, hashes of keys, or raw exception text.
    from .depth_service_probe import all_keys
    keys = all_keys()
    scan = list(p for p in directory.rglob("*") if p.is_file())
    scan += sorted(trial.PACKAGE.glob("vggt_pose*.py"))
    for path in scan:
        raw = path.read_bytes()
        if any(key.encode() in raw for key in keys.values()):
            raise ValueError("nonpublic_artifact_detected")
    audit["credential_scan"] = {
        "status": "passed", "distinct_configured_slots": len(keys), "files_checked": len(scan),
        "secret_values_persisted": False,
    }
    write(directory / "audit.json", audit, exclusive=True)
    print(json.dumps({"status": "report_and_audit_complete", **overall,
                      "legal_answers": audit["legal_answers"], "tool_calls": audit["tool_calls"],
                      "credential_scan": audit["credential_scan"]}, ensure_ascii=False))


if __name__ == "__main__":
    report()
