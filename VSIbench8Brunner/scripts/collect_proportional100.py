#!/usr/bin/env python3
"""Monitor both arms, verify server-native traces, then score locally."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from launch_proportional100 import LOCAL, REMOTE, PYTHON_TITAN
from skill3d.evaluation.accuracy import mca_correct
from skill3d.evaluation.mra import mra_from_text
from skill3d.routing.task_classifier import canonical_task


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def remote(control, code, stdin=None, timeout=60):
    result = subprocess.run(
        ["ssh", "-S", control, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         "wangy@10.10.0.34", PYTHON_TITAN + " -c " + shlex.quote(code)],
        input=stdin, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("paired collector remote command failed")
    return result.stdout


def query(control):
    code = """import json,time
from pathlib import Path
r=Path(%r); result={}
for arm in ('direct','harness'):
    pidfile=r/(arm+'.pid.json')
    if not pidfile.exists():
        result[arm]={'launched':False,'alive':False,'state':{'status':'not_launched','completed':0}}
        continue
    launch=json.loads(pidfile.read_text()); p=Path('/proc')/str(launch['pid'])
    alive=p.exists()
    if alive:
        fields=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
        assert fields[19]==launch['process_identity']['start_ticks']
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==launch['process_identity']['boot_id']
        alive=fields[0]!='Z'
    target=r/arm
    state=json.loads((target/'live_status.json').read_text()) if (target/'live_status.json').exists() else {'status':'preflight','completed':0}
    index=json.loads((target/'trajectory_index.json').read_text()) if (target/'trajectory_index.json').exists() else {'questions':{}}
    result[arm]={'launched':True,'alive':alive,'state':state,'process_identity':launch['process_identity'],
        'native_trajectories':sum(bool(v['completeness']['outcome_written'] and v['completeness']['all_model_requests_returned']) for v in index['questions'].values()),
        'outcome_files':len(list((target/'outcomes').glob('*.json')))}
print(json.dumps({'checked_at':time.time(),'arms':result}))
""" % REMOTE
    return json.loads(remote(control, code))


def value(outcome, reference):
    if not outcome["legal_answer"]:
        return 0.
    answer = outcome.get("predicted")
    if answer is None:
        answer = outcome.get("answer")
    return (float(mca_correct(answer or "", reference["ground_truth"]))
            if outcome["is_mca"] else float(mra_from_text(answer, reference["ground_truth"])))


def score(metadata_path):
    sampling = read(LOCAL / "sampling_manifest.json")
    selected = {str(r["qa_id"]): r for r in sampling["questions"]}
    audit = read(LOCAL / "paired_native_verified.json")
    if (audit["sampling_sha256"] != hashlib.sha256(
            (LOCAL / "sampling_manifest.json").read_bytes()).hexdigest()
            or any(a["questions_verified"] != 100 for a in audit["arms"].values())):
        raise ValueError("local scoring requires complete paired native verification")
    inputs = {arm: read(LOCAL / arm / "compact_score_inputs.json")
              for arm in ("direct", "harness")}
    if any(set(values) != set(selected) for values in inputs.values()):
        raise ValueError("paired scoring input scope changed")
    metadata = {str(r["qa_id"]): r for r in
                map(json.loads, metadata_path.read_text().splitlines())}
    rows = []
    for qa, question in selected.items():
        reference = metadata[qa]
        if (reference["question"] != question["question"]
                or (reference.get("options") or []) != (question.get("options") or [])
                or reference["source_split"] != question["source_split"]):
            raise ValueError("posthoc reference/public question identity mismatch")
        result = {"qa_id": qa, "task": canonical_task(question["question_type"]),
                  "raw_task": question["question_type"], "source_split": question["source_split"]}
        for arm in ("direct", "harness"):
            record = inputs[arm][qa]
            if (record["qa_id"] != qa or record["source_split"] != question["source_split"]
                    or record["reference_answer_visible"] is not False):
                raise ValueError("paired outcome question/source identity mismatch")
            outcome = record["outcome"]
            result.update({
                arm + "_answer": outcome.get("predicted") if outcome.get("predicted") is not None
                else outcome.get("answer"),
                arm + "_score": value(outcome, reference),
                arm + "_legal": bool(outcome["legal_answer"]),
                arm + "_outcome_sha256": record["outcome_sha256"],
            })
        result["mutually_legal"] = result["direct_legal"] and result["harness_legal"]
        rows.append(result)
    pairs = [r for r in rows if r["mutually_legal"]]
    by_task = {}
    for task, expected in sampling["type_counts"].items():
        group = [r for r in rows if r["task"] == task]
        legal = [r for r in group if r["mutually_legal"]]
        if len(group) != expected:
            raise ValueError("posthoc type denominator differs from frozen quota")
        by_task[task] = {
            "n": len(group), "mutually_legal": len(legal),
            "direct_mean_score": sum(r["direct_score"] for r in group) / len(group),
            "harness_mean_score": sum(r["harness_score"] for r in group) / len(group),
            "direct_legal": sum(r["direct_legal"] for r in group),
            "harness_legal": sum(r["harness_legal"] for r in group),
        }
    summary = {
        "schema": "vsi-proportional100-direct-versus-8b/1", "n_questions": 100,
        "completed_each_arm": 100, "paired_mutually_legal": len(pairs),
        "direct_all100_mean_score": sum(r["direct_score"] for r in rows) / 100,
        "harness_all100_mean_score": sum(r["harness_score"] for r in rows) / 100,
        "direct_common_legal_mean_score": sum(r["direct_score"] for r in pairs) / len(pairs) if pairs else None,
        "harness_common_legal_mean_score": sum(r["harness_score"] for r in pairs) / len(pairs) if pairs else None,
        "direct_legal_answers": sum(r["direct_legal"] for r in rows),
        "harness_legal_answers": sum(r["harness_legal"] for r in rows),
        "improved_questions": sum(r["harness_score"] > r["direct_score"] for r in rows),
        "declined_questions": sum(r["harness_score"] < r["direct_score"] for r in rows),
        "tied_questions": sum(r["harness_score"] == r["direct_score"] for r in rows),
        "model": "qwen3.8-flash", "enable_thinking": True, "seed": 137,
        "temperature": 0., "max_tokens": 8192, "timeout_s": 900,
        "scoring": "MCA exact match and official-equivalent 10-threshold numeric MRA; illegal answers score zero",
        "sampling_sha256": audit["sampling_sha256"],
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "reference_answer_visible_during_solve": False,
        "trajectory_location": "Titan", "by_task": by_task, "rows": rows,
    }
    summary["delta_percentage_points"] = 100 * (
        summary["harness_all100_mean_score"] - summary["direct_all100_mean_score"])
    write(LOCAL / "direct_vs_8b_scores.json", summary)
    labels = {
        "obj_appearance_order": "首次出现顺序", "object_abs_distance": "绝对距离",
        "object_counting": "物体计数", "object_rel_direction": "相对方向",
        "object_rel_distance": "相对距离", "object_size_estimation": "物体尺寸",
        "room_size_estimation": "房间面积", "route_planning": "路径规划",
    }
    lines = [
        "# 新选 100 题：Qwen 直答与 8B 描述 Harness",
        "",
        f"两边均完成 100/100 题。Qwen 直答平均分 {summary['direct_all100_mean_score']:.2%}，"
        f"8B 描述 Harness 平均分 {summary['harness_all100_mean_score']:.2%}，"
        f"差值 {summary['delta_percentage_points']:+.2f} 个百分点。",
        f"合法答案分别为 {summary['direct_legal_answers']}/100 和 {summary['harness_legal_answers']}/100；"
        f"{summary['improved_questions']} 题提高、{summary['declined_questions']} 题下降、"
        f"{summary['tied_questions']} 题持平。",
        "",
        "| 题型 | 题数 | Qwen 直答 | 8B 描述 Harness |",
        "|---|---:|---:|---:|",
    ]
    for task, group in by_task.items():
        lines.append(f"| {labels[task]} | {group['n']} | {group['direct_mean_score']:.2%} | "
                     f"{group['harness_mean_score']:.2%} |")
    lines += [
        "",
        f"双方答案均合法的共同题目为 {len(pairs)} 题。"
        + (f"共同题目平均分分别为 {summary['direct_common_legal_mean_score']:.2%} 和 "
           f"{summary['harness_common_legal_mean_score']:.2%}。" if pairs else ""),
        "",
        "从原 1000 题中排除有历史结果或已发起解题请求的 778 题，"
        "按剩余 222 题的题型比例采用最大余数法选 100 题，方向题再按难度分层，seed=137。"
        "所有题目的原 source_split 保留。本轮使用固定方法，不进行 Skill 学习或版本选择。",
        "直答使用经哈希与发布回执核验的 Qwen 官方工具 32 帧；"
        "Harness 使用冻结 RGB、已发布几何与候选缓存，加上 8B 描述和原二元审核服务。"
        "两边主模型采样参数一致，各自的视觉输入按运行回执解释。",
        "",
        "MCA 使用答案精确匹配；数值题使用官方等价的十阈值 MRA。"
        "平均分覆盖全部 100 题，非法答案记零。参考答案只在运行与原生轨迹核验结束后的 Mac 评分中读取。",
        "",
        "逐题答案、得分与原生 outcome 哈希见 `direct_vs_8b_scores.json`。",
        "全部原生模型请求与响应、reasoning、程序、工具 IO、描述审核记录及掩码保存在 Titan：",
        f"`{REMOTE}/direct/` 和 `{REMOTE}/harness/`。",
        "逐题完整快照位于各自的 `trajectories/<qa_id>.json`。"
        "轨迹、原生源文件与掩码已在 Titan 逐字节哈希核验；"
        "Mac 保存小型评分输入与核验回执，不代表 Mac 有完整轨迹副本。",
    ]
    continuation = (audit.get("final_test_admission") or {}).get(
        "native_continuation_verification")
    if continuation:
        summary["native_continuation"] = continuation
        write(LOCAL / "direct_vs_8b_scores.json", summary)
        lines += [
            "",
            "运行恢复记录：原准备额度 9710 次耗尽后，按固定剩余候选补充 "
            f"{continuation['additional_preparation_requests']} 次，最终上限为 "
            f"{continuation['continued_preparation_cap']} 次。"
            "4 题复用原生 SAM2 掩码和已返回的 8B 描述、二元审核响应；"
            "只发起此前尚未发出的请求。63 题已完成的答案及完整轨迹保持逐字节一致。",
            "原来的 28 条零请求 final_test 准入结果已存入失败历史，另 5 题此前未发起解题请求；"
            "这 33 题均以显式盲评准入完成。全部失败历史保留，未重复已发出的模型请求，"
            "未修改冻结源码、样本或方法配置。",
        ]
    (LOCAL / "comparison_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "completed_each_arm", "paired_mutually_legal", "direct_all100_mean_score",
        "harness_all100_mean_score", "delta_percentage_points")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-path", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    with (LOCAL / ".paired-collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous, deadline = None, time.monotonic() + 6 * 3600
        while True:
            status = query(args.control_path)
            write(LOCAL / "paired_live_status.json", status)
            counts = tuple((a, r["state"].get("completed", 0), r["state"]["status"],
                            r["native_trajectories"] if r["launched"] else 0)
                           for a, r in status["arms"].items())
            if counts != previous:
                print(json.dumps(status), flush=True)
                previous = counts
            if all(r["state"]["status"] == "complete" and r["state"]["completed"] == 100
                   and r["native_trajectories"] == 100 for r in status["arms"].values()):
                break
            if any(r["launched"] and not r["alive"]
                   and r["state"]["status"] != "complete" for r in status["arms"].values()):
                write(LOCAL / "collector_stopped.json", status)
                raise RuntimeError("owned arm stopped; native attempts retained without retry")
            if time.monotonic() > deadline:
                raise RuntimeError("paired collection exceeded its six-hour bound")
            time.sleep(15)
        audit_file = Path(__file__).with_name("verify_proportional_native.py")
        audit_remote = REMOTE + "/verify_proportional_native_" + hashlib.sha256(
            audit_file.read_bytes()).hexdigest()[:16] + ".py"
        upload = """import sys
from pathlib import Path
p=Path(%r)
b=sys.stdin.buffer.read()
if p.exists():
    assert p.read_bytes()==b
else:
    p.write_bytes(b)
""" % audit_remote
        remote(args.control_path, upload, stdin=audit_file.read_bytes())
        command = [
            "ssh", "-S", args.control_path, "-o", "BatchMode=yes", "wangy@10.10.0.34",
            PYTHON_TITAN + " " + shlex.quote(audit_remote)
            + " --root " + shlex.quote(REMOTE),
        ]
        subprocess.run(command, check=True, timeout=3600)
        files = ["paired_native_verified.json"]
        for arm in ("direct", "harness"):
            files += [arm + "/" + name for name in (
                "compact_score_inputs.json", "native_trajectory_verified.json",
                "runtime_identity.json", "trajectory_index.json", "live_status.json",
                "preflight_verified.json")]
            files.append(arm + ".pid.json")
        pull = """import json
from pathlib import Path
r=Path(%r)
print(json.dumps({f:json.loads((r/f).read_text()) for f in %r}))
""" % (REMOTE, files)
        for name, data in json.loads(remote(args.control_path, pull, timeout=120)).items():
            write(LOCAL / name, data)
        for name in ("DASHSCOPE_API_KEY", "QWEN3VL8B_API_KEY", "VSI_SAM2_TOKEN",
                     "OPENROUTER_API_KEY", "MAKERS_MODELS_KEY", "VSI8B_SSH_PASSWORD"):
            os.environ.pop(name, None)
        score(args.metadata)


if __name__ == "__main__":
    main()
